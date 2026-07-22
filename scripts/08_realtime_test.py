"""
Step 8: Real-Time Whistle Detector (86 Dimensions)

Binary whistle detection only -- per the thesis scope ("...classification of
audio segments into 'Whistle' and 'No Whistle' categories... the system is
delimited to processing whistle sounds solely as the audio trigger for
confirming referee calls"). The whistle's role is to trigger/validate a
referee call; WHAT the call means comes from the gesture recognition module,
not from whistle duration. Short/long blast classification has been removed.

WINDOW_SEC matches the training clip length (TARGET_LEN in
02_extract_whistle_clips.py = 1.5s). WHISTLE_COOLDOWN_SEC prevents a single
whistle from re-triggering multiple times while its tail/echo is still audible.
"""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import joblib
from scipy.fftpack import dct
import queue
import time

ROOT = Path(__file__).parent.parent
SR = 22050
WINDOW_SEC = 1.5   # must match TARGET_LEN used in 02_extract_whistle_clips.py
STEP_SEC = 0.5
N_MFCC = 13

OPTIMIZED_THRESHOLD = 0.30
WHISTLE_COOLDOWN_SEC = 2.0  # minimum time after a whistle ends before a new one can trigger

def normalize(y):
    max_abs = np.max(np.abs(y))
    if max_abs < 0.005: return y
    return y / (max_abs + 1e-9)

def compute_pure_mfcc(y, sr=SR, n_mfcc=N_MFCC, n_fft=512, hop_length=256, n_mels=40):
    window = np.hanning(n_fft)
    frames = [y[i:i+n_fft] * window for i in range(0, len(y) - n_fft + 1, hop_length)]
    if not frames: return np.zeros(86)
    frames = np.array(frames)

    stft_matrix = np.fft.rfft(frames, n=n_fft, axis=-1)
    magnitude_spectrum = np.abs(stft_matrix)
    power_spectrum = magnitude_spectrum ** 2

    low_mel = 0
    high_mel = 2595 * np.log10(1 + (sr / 2) / 700)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, int(n_fft / 2 + 1)))
    for m in range(1, n_mels + 1):
        f_m_minus, f_m, f_m_plus = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            if (f_m - f_m_minus) > 0: fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if (f_m_plus - f_m) > 0: fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

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

    freqs = np.fft.rfftfreq(n_fft, d=1/sr)
    centroid_frames = [np.sum(freqs * mag) / np.sum(mag) if np.sum(mag) > 0 else 0.0 for mag in magnitude_spectrum]
    centroid_mean, centroid_std = np.mean(centroid_frames), np.std(centroid_frames)

    flatness_frames = []
    for p_frame in power_spectrum:
        p_eps = p_frame + 1e-9
        flatness_frames.append(np.exp(np.mean(np.log(p_eps))) / np.mean(p_eps) if np.mean(p_eps) > 0 else 0.0)
    flatness_mean, flatness_std = np.mean(flatness_frames), np.std(flatness_frames)

    bandwidth_frames = []
    for idx, p_frame in enumerate(power_spectrum):
        p_sum = np.sum(p_frame)
        if p_sum > 0:
            variance = np.sum(p_frame * (freqs - centroid_frames[idx])**2) / p_sum
            bandwidth_frames.append(np.sqrt(variance))
        else:
            bandwidth_frames.append(0.0)
    bandwidth_mean, bandwidth_std = np.mean(bandwidth_frames), np.std(bandwidth_frames)

    return np.concatenate([
        mfcc_mean, mfcc_std, delta_mean, delta_std, dd_mean, dd_std,
        [zcr_mean, zcr_std], [centroid_mean, centroid_std],
        [flatness_mean, flatness_std], [bandwidth_mean, bandwidth_std]
    ])

def live_test(model):
    try:
        import sounddevice as sd
    except ImportError:
        print("Error: 'sounddevice' library is missing. Run: python -m pip install sounddevice")
        return

    ENERGY_THRESHOLD = 0.002
    audio_buffer = np.zeros(int(WINDOW_SEC * SR))

    audio_queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        audio_queue.put(indata[:, 0].copy())

    print(f"Listening for whistle (Threshold {OPTIMIZED_THRESHOLD}, window={WINDOW_SEC}s)...")
    print("Press Ctrl+C to terminate execution stream safely.")

    step_samples = int(STEP_SEC * SR)
    stream = sd.InputStream(channels=1, samplerate=SR, blocksize=step_samples, callback=callback)

    with stream:
        try:
            in_whistle_event = False
            silence_counter = 0
            last_event_end_time = 0.0  # tracks when the last whistle ended, for cooldown

            while True:
                raw_input = audio_queue.get()

                frames_received = len(raw_input)
                audio_buffer = np.roll(audio_buffer, -frames_received)
                audio_buffer[-frames_received:] = raw_input

                rms_energy = np.sqrt(np.mean(audio_buffer**2))
                is_whistle_now = False

                if rms_energy >= ENERGY_THRESHOLD:
                    signal = normalize(audio_buffer)
                    feat = compute_pure_mfcc(signal, SR).reshape(1, -1)
                    prob = model.predict_proba(feat)[0, 1]

                    if prob >= OPTIMIZED_THRESHOLD:
                        is_whistle_now = True

                now = time.monotonic()

                if is_whistle_now:
                    silence_counter = 0

                    # Only allow a NEW event to start if the cooldown has elapsed
                    # since the last one ended -- prevents a single whistle's tail
                    # from re-triggering as a second detection.
                    if not in_whistle_event:
                        if (now - last_event_end_time) < WHISTLE_COOLDOWN_SEC:
                            continue  # still in cooldown, ignore this trigger
                        in_whistle_event = True
                        print("\n========================================================")
                        print("★ WHISTLE DETECTED (Waking up gesture recognition...) ★")
                        print("========================================================")
                else:
                    if in_whistle_event:
                        silence_counter += 1
                        if silence_counter >= 3:
                            in_whistle_event = False
                            last_event_end_time = now  # start the cooldown clock

        except KeyboardInterrupt:
            print("\nShutting down live audio session context safely.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/08_realtime_test.py live")
        return

    model_path = ROOT / "models" / "whistle_svm_model.pkl"
    if not model_path.exists():
        return

    model = joblib.load(model_path)
    if sys.argv[1] == "live":
        live_test(model)

if __name__ == "__main__":
    main()