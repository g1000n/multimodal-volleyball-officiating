import sys
import time
import queue
import threading
from pathlib import Path
from datetime import datetime
import numpy as np
import joblib
from scipy.fftpack import dct
from scipy.signal import butter, sosfilt
import tkinter as tk

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "whistle_svm_model.pkl"

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

GATE_MIN_WHISTLE_RATIO = 0.35
GATE_MAX_PITCH_INSTABILITY = 15.0

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
    
    low_mel, high_mel = 0, 2595 * np.log10(1 + (sr/2)/700)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    
    fbank = np.zeros((n_mels, int(n_fft/2 + 1)))
    for m in range(1, n_mels + 1):
        f_m_minus, f_m, f_m_plus = bin_points[m-1], bin_points[m], bin_points[m + 1]
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
    
    freqs = np.fft.rfftfreq(n_fft, d=1/sr)
    
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
    freqs = np.fft.rfftfreq(n_fft, d=1/sr)
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

class WhistleDetectorApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("Whistle Detector")
        self.root.geometry("450x680")
        self.root.configure(bg="#1e1e1e")
        self.root.resizable(False, False)

        self.threshold = tk.DoubleVar(value=DEFAULT_THRESHOLD)
        self.running = False

        self.audio_queue = queue.Queue()
        self.gui_queue = queue.Queue()
        self.worker_thread = None

        self._build_ui()
        self.root.after(50, self._poll_gui_queue)

    def _build_ui(self):
        tk.Label(self.root, text="Whistle Detector", font=("Segoe UI", 18, "bold"),
                 bg="#1e1e1e", fg="white").pack(pady=(16, 4))

        self.status_canvas = tk.Canvas(self.root, width=200, height=200, bg="#1e1e1e",
                                       highlightthickness=0)
        self.status_canvas.pack(pady=10)
        self.status_circle = self.status_canvas.create_oval(10, 10, 190, 190, fill="#3a3a3a",
                                                            outline="")
        self.status_text = self.status_canvas.create_text(100, 100, text="IDLE",
                                                          font=("Segoe UI", 16, "bold"), fill="white")

        tk.Label(self.root, text="Confidence", bg="#1e1e1e", fg="#aaaaaa", font=("Segoe UI", 10)).pack()

        self.conf_bar_bg = tk.Canvas(self.root, width=360, height=22, bg="#2a2a2a",
                                     highlightthickness=0)
        self.conf_bar_bg.pack(pady=(2, 6))
        self.conf_bar_fill = self.conf_bar_bg.create_rectangle(0, 0, 0, 22, fill="#4caf50", width=0)
        
        self.threshold_line = self.conf_bar_bg.create_line(0, 0, 0, 22, fill="yellow", width=2)

        self.conf_label = tk.Label(self.root, text="0%", bg="#1e1e1e", fg="white", font=("Segoe UI", 10))
        self.conf_label.pack()

        tk.Label(self.root, text="Mic Energy Level (post-bandpass)", fg="#aaaaaa", bg="#1e1e1e",
                 font=("Segoe UI", 9)).pack(pady=(10, 0))
        
        self.energy_canvas = tk.Canvas(self.root, width=360, height=14, bg="#2a2a2a",
                                       highlightthickness=0)
        self.energy_canvas.pack(pady=(2, 10))
        self.energy_bar = self.energy_canvas.create_rectangle(0, 0, 0, 14, fill="#888888", width=0)

        tk.Label(self.root, text="Sensitivity threshold (raise this if shouting / ball hits trigger it)",
                 bg="#1e1e1e", fg="#aaaaaa", font=("Segoe UI", 9), wraplength=380).pack(pady=(14, 0))

        tk.Scale(self.root, from_=0.10, to=0.95, resolution=0.01, orient="horizontal",
                 variable=self.threshold, length=360, bg="#1e1e1e", fg="white",
                 troughcolor="#2a2a2a", highlightthickness=0, showvalue=True).pack()

        self.toggle_btn = tk.Button(self.root, text="Start Listening", command=self._toggle,
                                    bg="#4caf50", fg="white", font=("Segoe UI", 12, "bold"),
                                    relief="flat", padx=20, pady=8)
        self.toggle_btn.pack(pady=16)

        tk.Label(self.root, text="Detection log", bg="#1e1e1e", fg="#aaaaaa", font=("Segoe UI", 9)).pack()

        log_frame = tk.Frame(self.root, bg="#1e1e1e")
        log_frame.pack(pady=(2, 10))
        
        self.log_list = tk.Listbox(log_frame, width=55, height=10, bg="#2a2a2a", fg="white",
                                   borderwidth=0, highlightthickness=0, font=("Consolas", 9))
        self.log_list.pack()

    def _toggle(self):
        self._stop() if self.running else self._start()

    def _start(self):
        try:
            import sounddevice as sd
        except ImportError:
            self._log_message("ERROR: run pip install sounddevice")
            return

        if not MODEL_PATH.exists():
            self._log_message(f"ERROR: model not found at {MODEL_PATH}")
            return

        self.running = True
        self.toggle_btn.config(text="Stop Listening", bg="#e53935")
        self._set_status("LISTENING")

        self.worker_thread = threading.Thread(target=self._audio_loop, args=(sd,), daemon=True)
        self.worker_thread.start()

    def _stop(self):
        self.running = False
        self.toggle_btn.config(text="Start Listening", bg="#4caf50")
        self._set_status("IDLE")

    def _audio_loop(self, sd):
        step_samples = int(STEP_SEC * SR)
        audio_buffer = np.zeros(int(WINDOW_SEC * SR))

        def callback(indata, frames, time_info, status):
            self.audio_queue.put(indata[:, 0].copy())

        in_event = False
        above_count = 0
        silence_counter = 0
        last_event_end = 0.0

        with sd.InputStream(channels=1, samplerate=SR, blocksize=step_samples, callback=callback):
            while self.running:
                try:
                    raw = self.audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                n = len(raw)
                audio_buffer = np.roll(audio_buffer, -n)
                audio_buffer[-n:] = raw

                filtered = apply_bandpass(audio_buffer)
                rms = np.sqrt(np.mean(filtered ** 2))
                prob = 0.0
                gate_ok = False

                if rms >= ENERGY_THRESHOLD:
                    signal = normalize(audio_buffer)
                    feat = compute_advanced_features(signal, SR).reshape(1, -1)
                    prob = float(self.model.predict_proba(feat)[0, 1])

                    whistle_ratio, peak_conc, pitch_instability = compute_gate_metrics(signal, SR)
                    gate_ok = (whistle_ratio >= GATE_MIN_WHISTLE_RATIO
                               and pitch_instability <= GATE_MAX_PITCH_INSTABILITY)

                self.gui_queue.put(("metrics", (prob, rms)))

                now = time.monotonic()
                threshold = self.threshold.get()
                is_whistle = (prob >= threshold) and gate_ok

                if is_whistle:
                    above_count += 1
                    silence_counter = 0

                    if not in_event and above_count >= CONFIRM_FRAMES:
                        if (now - last_event_end) < COOLDOWN_SEC:
                            continue
                        in_event = True
                        self.gui_queue.put(("detected", prob))
                else:
                    above_count = 0
                    if in_event:
                        silence_counter += 1
                        if silence_counter >= 3:
                            in_event = False
                            last_event_end = now
                            self.gui_queue.put(("cleared", None))

    def _poll_gui_queue(self):
        try:
            while True:
                kind, value = self.gui_queue.get_nowait()
                
                if kind == "metrics":
                    self._update_metrics(*value)
                elif kind == "detected":
                    self._set_status("WHISTLE!")
                    self._log_message(f"Whistle detected (confidence {value * 100:.0f}%)")
                elif kind == "cleared":
                    if self.running:
                        self._set_status("LISTENING")
        except queue.Empty:
            pass

        self.root.after(50, self._poll_gui_queue)

    def _update_metrics(self, prob, rms):
        w = 360
        thresh = self.threshold.get()
        
        width = int(w * prob)
        color = "#e53935" if prob >= thresh else "#4caf50"
        self.conf_bar_bg.coords(self.conf_bar_fill, 0, 0, width, 22)
        self.conf_bar_bg.itemconfig(self.conf_bar_fill, fill=color)
        
        self.conf_bar_bg.coords(self.threshold_line, int(w * thresh), 0, int(w * thresh), 22)
        self.conf_label.config(text=f"{prob * 100:.0f}% (threshold {thresh:.2f})")
        
        energy_frac = min(1.0, rms / 0.05)
        self.energy_canvas.coords(self.energy_bar, 0, 0, int(w * energy_frac), 14)

    def _set_status(self, text):
        self.status_canvas.itemconfig(self.status_text, text=text)
        color = "#e53935" if text == "WHISTLE!" else ("#4caf50" if text == "LISTENING" else "#3a3a3a")
        self.status_canvas.itemconfig(self.status_circle, fill=color)

    def _log_message(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_list.insert(0, f"[{ts}] {msg}")

def main():
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}. Run 06_train_model.py first.")
        return
        
    print("Loading model...")
    model = joblib.load(MODEL_PATH)
    
    root = tk.Tk()
    WhistleDetectorApp(root, model)
    root.mainloop()

if __name__ == "__main__":
    main()