"""
Step 8 (UI version): Real-Time Whistle Detector with a visual dashboard.

Shows, live:
  - Current whistle probability as a bar (red when above threshold)
  - Current RMS energy level
  - A big status indicator that flashes on whistle detection
  - A scrolling log of detected whistle events with timestamps
  - An adjustable threshold slider -- tune it live without editing code,
    useful for reducing false positives from court noise (ball hits,
    squeaks, cheering) without a full retrain.

Requires: sounddevice (pip install sounddevice) -- tkinter ships with Python.

Run: python scripts/08b_realtime_ui.py
"""
import sys
import queue
import time
import tkinter as tk
from pathlib import Path
from datetime import datetime

import numpy as np
import joblib
from scipy.fftpack import dct

ROOT = Path(__file__).parent.parent
SR = 22050
WINDOW_SEC = 1.5   # must match TARGET_LEN in 02_extract_whistle_clips.py
STEP_SEC = 0.2     # smaller step = more responsive UI updates
N_MFCC = 13

DEFAULT_THRESHOLD = 0.45   # raised from 0.30 -- reduces court-noise false positives
ENERGY_THRESHOLD = 0.002
CONFIRM_FRAMES = 2         # require N consecutive above-threshold frames to trigger
                            # (prevents one stray loud transient from false-triggering)
END_SILENCE_FRAMES = 3
WHISTLE_COOLDOWN_SEC = 2.0


def normalize(y):
    max_abs = np.max(np.abs(y))
    if max_abs < 0.005:
        return y
    return y / (max_abs + 1e-9)


def compute_pure_mfcc(y, sr=SR, n_mfcc=N_MFCC, n_fft=512, hop_length=256, n_mels=40):
    window = np.hanning(n_fft)
    frames = [y[i:i + n_fft] * window for i in range(0, len(y) - n_fft + 1, hop_length)]
    if not frames:
        return np.zeros(86)
    frames = np.array(frames)

    stft_matrix = np.fft.rfft(frames, n=n_fft, axis=-1)
    magnitude_spectrum = np.abs(stft_matrix)
    power_spectrum = magnitude_spectrum ** 2

    low_mel = 0
    high_mel = 2595 * np.log10(1 + (sr / 2) / 700)
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
            variance = np.sum(p_frame * (freqs - centroid_frames[idx]) ** 2) / p_sum
            bandwidth_frames.append(np.sqrt(variance))
        else:
            bandwidth_frames.append(0.0)
    bandwidth_mean, bandwidth_std = np.mean(bandwidth_frames), np.std(bandwidth_frames)

    return np.concatenate([
        mfcc_mean, mfcc_std, delta_mean, delta_std, dd_mean, dd_std,
        [zcr_mean, zcr_std], [centroid_mean, centroid_std],
        [flatness_mean, flatness_std], [bandwidth_mean, bandwidth_std]
    ])


class WhistleDetectorUI:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("Whistle Detection Monitor")
        self.root.geometry("480x460")
        self.root.configure(bg="#1e1e1e")

        self.threshold = tk.DoubleVar(value=DEFAULT_THRESHOLD)
        self.audio_buffer = np.zeros(int(WINDOW_SEC * SR))
        self.audio_queue = queue.Queue()
        self.above_count = 0
        self.in_whistle_event = False
        self.silence_count = 0
        self.last_event_end_time = 0.0
        self.event_count = 0

        self._build_ui()
        self._start_audio_stream()
        self.root.after(50, self._poll)

    def _build_ui(self):
        title = tk.Label(self.root, text="WHISTLE DETECTION", font=("Segoe UI", 16, "bold"),
                          fg="white", bg="#1e1e1e")
        title.pack(pady=(12, 4))

        self.status_label = tk.Label(self.root, text="LISTENING...", font=("Segoe UI", 20, "bold"),
                                      fg="#4caf50", bg="#1e1e1e")
        self.status_label.pack(pady=(4, 16))

        # Probability meter
        tk.Label(self.root, text="Whistle Probability", fg="#aaaaaa", bg="#1e1e1e",
                 font=("Segoe UI", 9)).pack()
        self.prob_canvas = tk.Canvas(self.root, width=420, height=30, bg="#333333", highlightthickness=0)
        self.prob_canvas.pack(pady=(2, 2))
        self.prob_bar = self.prob_canvas.create_rectangle(0, 0, 0, 30, fill="#2196f3", width=0)
        self.threshold_line = self.prob_canvas.create_line(0, 0, 0, 30, fill="yellow", width=2)
        self.prob_text = tk.Label(self.root, text="0.00", fg="white", bg="#1e1e1e", font=("Segoe UI", 9))
        self.prob_text.pack()

        # Energy meter
        tk.Label(self.root, text="Mic Energy Level", fg="#aaaaaa", bg="#1e1e1e",
                 font=("Segoe UI", 9)).pack(pady=(10, 0))
        self.energy_canvas = tk.Canvas(self.root, width=420, height=14, bg="#333333", highlightthickness=0)
        self.energy_canvas.pack(pady=(2, 10))
        self.energy_bar = self.energy_canvas.create_rectangle(0, 0, 0, 14, fill="#888888", width=0)

        # Threshold slider -- adjustable live, no code edits needed
        tk.Label(self.root, text="Detection Threshold (raise to reduce false positives from court noise)",
                 fg="#aaaaaa", bg="#1e1e1e", font=("Segoe UI", 8)).pack(pady=(4, 0))
        slider = tk.Scale(self.root, from_=0.1, to=0.9, resolution=0.01, orient="horizontal",
                           variable=self.threshold, length=420, bg="#1e1e1e", fg="white",
                           troughcolor="#333333", highlightthickness=0)
        slider.pack()

        # Event log
        tk.Label(self.root, text="Detected Whistle Events", fg="#aaaaaa", bg="#1e1e1e",
                 font=("Segoe UI", 9)).pack(pady=(10, 0))
        self.log_box = tk.Listbox(self.root, height=8, width=55, bg="#252525", fg="#4caf50",
                                   font=("Consolas", 9), highlightthickness=0, borderwidth=0)
        self.log_box.pack(pady=(2, 10))

    def _start_audio_stream(self):
        try:
            import sounddevice as sd
        except ImportError:
            self.status_label.config(text="sounddevice not installed", fg="red")
            print("Run: python -m pip install sounddevice")
            return

        def callback(indata, frames, time_info, status):
            if status:
                print(status, file=sys.stderr)
            self.audio_queue.put(indata[:, 0].copy())

        step_samples = int(STEP_SEC * SR)
        self.stream = sd.InputStream(channels=1, samplerate=SR, blocksize=step_samples, callback=callback)
        self.stream.start()

    def _poll(self):
        try:
            while True:
                raw_input = self.audio_queue.get_nowait()
                frames_received = len(raw_input)
                self.audio_buffer = np.roll(self.audio_buffer, -frames_received)
                self.audio_buffer[-frames_received:] = raw_input
        except queue.Empty:
            pass

        rms_energy = np.sqrt(np.mean(self.audio_buffer ** 2))
        prob = 0.0
        thresh = self.threshold.get()

        if rms_energy >= ENERGY_THRESHOLD:
            signal = normalize(self.audio_buffer)
            feat = compute_pure_mfcc(signal, SR).reshape(1, -1)
            prob = self.model.predict_proba(feat)[0, 1]

        is_above = prob >= thresh
        now = time.monotonic()

        if is_above:
            self.above_count += 1
            self.silence_count = 0
        else:
            self.above_count = 0
            if self.in_whistle_event:
                self.silence_count += 1
                if self.silence_count >= END_SILENCE_FRAMES:
                    self.in_whistle_event = False
                    self.last_event_end_time = now

        if (self.above_count >= CONFIRM_FRAMES and not self.in_whistle_event
                and (now - self.last_event_end_time) >= WHISTLE_COOLDOWN_SEC):
            self.in_whistle_event = True
            self.event_count += 1
            ts = datetime.now().strftime("%H:%M:%S")
            self.log_box.insert(0, f"[{self.event_count:03d}] {ts}  prob={prob:.2f}")
            self._flash_detected()

        self._update_meters(prob, rms_energy, thresh)
        self.root.after(50, self._poll)

    def _update_meters(self, prob, energy, thresh):
        w = 420
        self.prob_canvas.coords(self.prob_bar, 0, 0, min(w, w * prob), 30)
        color = "#f44336" if prob >= thresh else "#2196f3"
        self.prob_canvas.itemconfig(self.prob_bar, fill=color)
        self.prob_canvas.coords(self.threshold_line, w * thresh, 0, w * thresh, 30)
        self.prob_text.config(text=f"{prob:.2f}  (threshold {thresh:.2f})")

        energy_frac = min(1.0, energy / 0.05)
        self.energy_canvas.coords(self.energy_bar, 0, 0, w * energy_frac, 14)

        if not self.in_whistle_event:
            self.status_label.config(text="LISTENING...", fg="#4caf50")

    def _flash_detected(self):
        self.status_label.config(text="★ WHISTLE DETECTED ★", fg="#ffeb3b", bg="#c62828")
        self.root.configure(bg="#c62828")
        self.root.after(600, self._unflash)

    def _unflash(self):
        self.status_label.config(bg="#1e1e1e")
        self.root.configure(bg="#1e1e1e")


def main():
    model_path = ROOT / "models" / "whistle_svm_model.pkl"
    if not model_path.exists():
        print(f"Model not found at {model_path}. Run 06_train_model.py first.")
        return

    model = joblib.load(model_path)
    root = tk.Tk()
    app = WhistleDetectorUI(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()