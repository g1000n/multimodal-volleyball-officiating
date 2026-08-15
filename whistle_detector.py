"""
whistle_detector.py

A clean, importable wrapper around the audio team's trained whistle model
(models/whistle_svm_model.pkl, a triple-ensemble SVM+RandomForest+
HistGradientBoosting voting classifier), extracted from their
scripts/08_realtime_ui.py.

WHY THIS FILE EXISTS: 08_realtime_ui.py is a full Tkinter GUI app -- its
detection state machine (threshold, confirm-frames, cooldown) lives inside
the app class, tightly coupled to its own audio thread and GUI polling loop.
There's no clean function to just import and call. This file extracts the
SAME logic (feature math copied verbatim -- do not modify without checking
scripts/05_extract_features.py and scripts/08_realtime_ui.py too, since
training and live inference must agree exactly on feature computation) into
a plain class with a callback interface, so it can run as a background
thread alongside your existing camera/gesture loop in ONE process, instead
of needing two separate windows/processes talking over some IPC mechanism.

--------------------------------------------------------------------
NEW -- raw_audio_callback (shares this stream instead of a second one):

live_deployment.py separately needed to record the whole session's raw
audio to a WAV file, for muxing into the recorded videos. The naive way to
do that is to open a SECOND, independent sd.InputStream on the same device
-- but that's exactly what caused a real, confirmed bug: two simultaneous
streams competing for one virtual device (Camo) meant the second stream
opened without error but silently received zero real audio data. Confirmed
directly: disabling the whistle detector entirely made the second stream
start working, proving it was a conflict, not a broken device.

FIX: this class now accepts an optional `raw_audio_callback` -- if
provided, every raw audio chunk this stream already receives (before any
whistle-detection processing) is ALSO forwarded to that callback. This lets
an external recorder (SessionAudioRecorder in live_deployment.py) receive
the same audio data passively, without ever opening its own competing
stream. Purely additive -- does not change any detection logic, timing, or
feature computation. If raw_audio_callback is None (the default), behavior
is 100% identical to before this change.

USAGE:
    from whistle_detector import WhistleDetector

    def on_whistle(timestamp, confidence):
        engine.on_whistle_detected(timestamp)
        print(f"Whistle detected! confidence={confidence:.2f}")

    def on_raw_audio(chunk):
        audio_recorder.feed(chunk)  # optional -- omit if you don't need session audio

    detector = WhistleDetector(on_whistle_callback=on_whistle, raw_audio_callback=on_raw_audio)
    detector.start()   # runs in a background thread, returns immediately
    ...
    detector.stop()    # call on shutdown, alongside cap.release() etc.

All the tunable constants below match scripts/08_realtime_ui.py's defaults
exactly -- these are the SAME values their real-time GUI app uses,
confirmed working in their testing. Adjust only if you have a specific
reason, and know you're diverging from their validated defaults.
"""

import time
import queue
import threading
from pathlib import Path

import numpy as np
import joblib
from scipy.fftpack import dct
from scipy.signal import butter, sosfilt

MODEL_PATH = Path(__file__).resolve().parent / "models" / "whistle_svm_model.pkl"

SR = 22050
WINDOW_SEC = 1.5
STEP_SEC = 0.5
N_MFCC = 13
N_MELS = 40
N_FFT = 512
HOP_LENGTH = 256

ENERGY_THRESHOLD = 0.002
DEFAULT_THRESHOLD = 0.70
COOLDOWN_SEC = 2.0
CONFIRM_FRAMES = 2
SILENCE_FRAMES_TO_CLEAR = 3
GATE_MIN_WHISTLE_RATIO = 0.35
GATE_MAX_PITCH_INSTABILITY = 20.0

SOS_BANDPASS = butter(4, [1800, 4800], btype='bandpass', fs=SR, output='sos')


def apply_bandpass(y):
    return sosfilt(SOS_BANDPASS, y)


def normalize(y):
    max_abs = np.max(np.abs(y))
    if max_abs < 1e-4:
        return y
    return y / (max_abs + 1e-9)


def compute_hnr(signal, sr=SR):
    if np.max(np.abs(signal)) < 1e-5:
        return 0.0
    min_lag, max_lag = int(sr / 5000), int(sr / 1000)
    if max_lag <= min_lag or max_lag >= len(signal):
        return 0.0
    r0 = float(np.dot(signal, signal))
    if r0 <= 0:
        return 0.0
    r_max = 0.0
    for lag in range(min_lag, max_lag):
        r = float(np.dot(signal[:-lag], signal[lag:]))
        if r > r_max:
            r_max = r
    if r_max <= 0 or r_max >= r0:
        return 0.0
    hnr = 10 * np.log10(r_max / (r0 - r_max + 1e-9))
    return float(np.clip(hnr, -20.0, 40.0))


def compute_advanced_features(raw_y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    y = apply_bandpass(raw_y)
    window = np.hanning(n_fft)
    frames = [y[i:i + n_fft] * window for i in range(0, len(y) - n_fft + 1, hop_length)]
    if not frames:
        return np.zeros(92)
    frames = np.array(frames)
    stft_matrix = np.fft.rfft(frames, n=n_fft, axis=-1)
    magnitude_spectrum = np.abs(stft_matrix)
    power_spectrum = magnitude_spectrum ** 2
    low_mel, high_mel = 0, 2595 * np.log10(1 + (sr / 2) / 700)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fbank = np.zeros((n_mels, int(n_fft / 2 + 1)))
    for m in range(1, n_mels + 1):
        f_m_minus, f_m, f_m_plus = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            if (f_m - f_m_minus) > 0:
                fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if (f_m_plus - f_m) > 0:
                fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    filter_banks = np.dot(power_spectrum, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    mfcc = dct(filter_banks, type=2, axis=-1, norm='ortho')[:, :n_mfcc]
    mfcc_mean, mfcc_std = np.mean(mfcc, axis=0), np.std(mfcc, axis=0)
    if len(mfcc) > 1:
        deltas = np.diff(mfcc, axis=0)
        delta_mean, delta_std = np.mean(deltas, axis=0), np.std(deltas, axis=0)
    else:
        deltas = np.zeros((1, n_mfcc))
        delta_mean, delta_std = np.zeros(n_mfcc), np.zeros(n_mfcc)
    if len(deltas) > 1:
        delta_deltas = np.diff(deltas, axis=0)
        dd_mean, dd_std = np.mean(delta_deltas, axis=0), np.std(delta_deltas, axis=0)
    else:
        dd_mean, dd_std = np.zeros(n_mfcc), np.zeros(n_mfcc)
    zcr_frames = [np.sum(np.abs(np.diff(np.sign(f)))) / 2 / len(f) for f in frames]
    zcr_mean, zcr_std = np.mean(zcr_frames), np.std(zcr_frames)
    freqs = np.fft.rfftfreq(n_fft, d=1 / sr)
    centroid_frames = []
    for mag in magnitude_spectrum:
        mag_sum = np.sum(mag)
        centroid_frames.append(np.sum(freqs * mag) / mag_sum if mag_sum > 0 else 0.0)
    centroid_mean, centroid_std = np.mean(centroid_frames), np.std(centroid_frames)
    flatness_frames = []
    for power_frame in power_spectrum:
        power_eps = power_frame + 1e-9
        log_mean = np.mean(np.log(power_eps))
        arith_mean = np.mean(power_eps)
        flatness_frames.append(np.exp(log_mean) / arith_mean if arith_mean > 0 else 0.0)
    flatness_mean, flatness_std = np.mean(flatness_frames), np.std(flatness_frames)
    bandwidth_frames = []
    for idx, power_frame in enumerate(power_spectrum):
        p_sum = np.sum(power_frame)
        if p_sum > 0:
            centroid = centroid_frames[idx]
            variance = np.sum(power_frame * (freqs - centroid) ** 2) / p_sum
            bandwidth_frames.append(np.sqrt(variance))
        else:
            bandwidth_frames.append(0.0)
    bandwidth_mean, bandwidth_std = np.mean(bandwidth_frames), np.std(bandwidth_frames)
    whistle_mask = (freqs >= 2000) & (freqs <= 4500)
    whistle_ratio_frames, peak_concentration_frames, dom_pitch_bins = [], [], []
    for mag in magnitude_spectrum:
        total_e = np.sum(mag) + 1e-9
        band_mags = mag[whistle_mask]
        band_e = np.sum(band_mags)
        whistle_ratio_frames.append(band_e / total_e)
        if band_e > 0:
            peak_concentration_frames.append(np.max(band_mags) / band_e)
        else:
            peak_concentration_frames.append(0.0)
        dom_pitch_bins.append(np.argmax(mag))
    whistle_ratio_mean, whistle_ratio_std = np.mean(whistle_ratio_frames), np.std(whistle_ratio_frames)
    peak_conc_mean, peak_conc_std = np.mean(peak_concentration_frames), np.std(peak_concentration_frames)
    pitch_stability = np.std(dom_pitch_bins)
    hnr_value = compute_hnr(y, sr=sr)
    return np.concatenate([
        mfcc_mean, mfcc_std,
        delta_mean, delta_std,
        dd_mean, dd_std,
        [zcr_mean, zcr_std], [centroid_mean, centroid_std],
        [flatness_mean, flatness_std], [bandwidth_mean, bandwidth_std],
        [whistle_ratio_mean, whistle_ratio_std],
        [peak_conc_mean, peak_conc_std],
        [pitch_stability, hnr_value]
    ])


def compute_gate_metrics(raw_y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH):
    y = apply_bandpass(raw_y)
    window = np.hanning(n_fft)
    frames = [y[i:i + n_fft] * window for i in range(0, len(y) - n_fft + 1, hop_length)]
    if not frames:
        return 0.0, 0.0, 999.0
    frames = np.array(frames)
    magnitude_spectrum = np.abs(np.fft.rfft(frames, n=n_fft, axis=-1))
    freqs = np.fft.rfftfreq(n_fft, d=1 / sr)
    whistle_mask = (freqs >= 2000) & (freqs <= 4500)
    ratios, peaks, dom_bins = [], [], []
    for mag in magnitude_spectrum:
        total_e = np.sum(mag) + 1e-9
        band_mags = mag[whistle_mask]
        band_e = np.sum(band_mags)
        ratios.append(band_e / total_e)
        peaks.append(np.max(band_mags) / band_e if band_e > 0 else 0.0)
        dom_bins.append(np.argmax(mag))
    return float(np.mean(ratios)), float(np.mean(peaks)), float(np.std(dom_bins))


class WhistleDetector:
    def __init__(self, on_whistle_callback, model_path=MODEL_PATH, threshold=DEFAULT_THRESHOLD,
                 device=None, log_dir="data/whistle_logs", raw_audio_callback=None):
        """
        on_whistle_callback: called as on_whistle_callback(timestamp, confidence)
        every time a whistle event fires.

        device: sounddevice input device index (int) or None to use whatever
        the OS currently considers "default" -- NOT recommended for game day.

        log_dir: every audio window gets logged to a CSV here.

        raw_audio_callback: NEW -- optional, called as raw_audio_callback(chunk)
        with every raw audio chunk this stream receives, BEFORE any whistle-
        detection processing. Lets an external recorder (e.g.
        SessionAudioRecorder in live_deployment.py) capture the same session
        audio WITHOUT opening a second, competing stream on the same device --
        see module docstring for the real bug this fixes. None (default) =
        identical behavior to before this parameter existed.
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Whistle model not found at {model_path}. "
                f"Make sure models/whistle_svm_model.pkl exists (from the audio branch's "
                f"scripts/06_train_model.py output) and MODEL_PATH points at it correctly."
            )
        self.model = joblib.load(model_path)
        self.on_whistle_callback = on_whistle_callback
        self.raw_audio_callback = raw_audio_callback
        self.threshold = threshold
        self.device = device
        self._running = False
        self._thread = None
        self._audio_queue = queue.Queue()

        self.last_prob = 0.0
        self.last_rms = 0.0

        import csv
        import os
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"whistle_session_{int(time.time())}.csv")
        self._log_file = open(self.log_path, "w", newline="")
        self._log_writer = csv.writer(self._log_file)
        self._log_writer.writerow([
            "timestamp", "rms", "energy_gate_passed", "prob", "whistle_ratio",
            "pitch_instability", "physics_gate_passed", "considered_whistle_this_window",
            "confirmed_event_fired",
        ])
        print(f"Whistle detection logging to: {self.log_path}")

    def start(self):
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("sounddevice not installed -- run: pip install sounddevice")
        self._running = True
        self._thread = threading.Thread(target=self._audio_loop, args=(sd,), daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._log_file.close()
        print(f"\nWhistle log saved to: {self.log_path}")

    def _audio_loop(self, sd):
        step_samples = int(STEP_SEC * SR)
        audio_buffer = np.zeros(int(WINDOW_SEC * SR))

        def callback(indata, frames, time_info, status):
            chunk = indata[:, 0].copy()
            self._audio_queue.put(chunk)
            # NEW: forward the same raw chunk to an external recorder, if
            # one was provided -- no second stream opened, see module
            # docstring.
            if self.raw_audio_callback is not None:
                self.raw_audio_callback(chunk)

        in_event = False
        above_count = 0
        silence_counter = 0
        last_event_end = 0.0

        with sd.InputStream(device=self.device, channels=1, samplerate=SR, blocksize=step_samples, callback=callback):
            while self._running:
                try:
                    raw = self._audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                n = len(raw)
                audio_buffer = np.roll(audio_buffer, -n)
                audio_buffer[-n:] = raw

                filtered = apply_bandpass(audio_buffer)
                rms = np.sqrt(np.mean(filtered ** 2))

                prob = 0.0
                gate_ok = False
                whistle_ratio = 0.0
                pitch_instability = 999.0

                energy_gate_passed = rms >= ENERGY_THRESHOLD
                if energy_gate_passed:
                    signal = normalize(audio_buffer)
                    feat = compute_advanced_features(signal, SR).reshape(1, -1)
                    prob = float(self.model.predict_proba(feat)[0, 1])
                    whistle_ratio, peak_conc, pitch_instability = compute_gate_metrics(signal, SR)
                    gate_ok = (whistle_ratio >= GATE_MIN_WHISTLE_RATIO
                               and pitch_instability <= GATE_MAX_PITCH_INSTABILITY)

                self.last_prob = prob
                self.last_rms = rms

                now = time.monotonic()
                is_whistle = (prob >= self.threshold) and gate_ok
                event_fired_this_window = False

                if is_whistle:
                    above_count += 1
                    silence_counter = 0
                    if not in_event and above_count >= CONFIRM_FRAMES:
                        if (now - last_event_end) >= COOLDOWN_SEC:
                            in_event = True
                            event_fired_this_window = True
                            self.on_whistle_callback(time.time(), prob)
                else:
                    above_count = 0
                    if in_event:
                        silence_counter += 1
                        if silence_counter >= SILENCE_FRAMES_TO_CLEAR:
                            in_event = False
                            last_event_end = now

                self._log_writer.writerow([
                    f"{time.time():.3f}", f"{rms:.5f}", int(energy_gate_passed),
                    f"{prob:.4f}", f"{whistle_ratio:.4f}", f"{pitch_instability:.2f}",
                    int(gate_ok), int(is_whistle), int(event_fired_this_window),
                ])
                self._log_file.flush()


if __name__ == "__main__":
    import sys

    device_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None

    def _test_callback(timestamp, confidence):
        print(f"\nWHISTLE DETECTED at {timestamp:.3f} (confidence {confidence:.2f})")

    print(f"Starting standalone whistle detector test (device={device_arg if device_arg is not None else 'OS default'}).")
    print("Blow a whistle near the mic... Press Ctrl+C to stop.\n")

    detector = WhistleDetector(on_whistle_callback=_test_callback, device=device_arg)
    detector.start()
    try:
        while True:
            time.sleep(0.5)
            print(f"\rlast_prob={detector.last_prob:.2f}  last_rms={detector.last_rms:.4f}", end="")
    except KeyboardInterrupt:
        print("\nStopping...")
        detector.stop()