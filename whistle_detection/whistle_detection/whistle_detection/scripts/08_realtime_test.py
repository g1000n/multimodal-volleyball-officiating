"""
Step 8: Real-time (or offline) whistle detection using the trained model.

Two modes:
  - offline: replay a recorded file through the same pipeline (test this
    first, before live mic testing)
  - live: use the system microphone with a sliding window

Run with: python 08_realtime_test.py offline path/to/file.wav
      or: python 08_realtime_test.py live
"""
import sys
from pathlib import Path

import numpy as np
import librosa
import joblib

ROOT = Path(__file__).parent.parent
SR = 22050
WINDOW_SEC = 1.0
N_MFCC = 13


def normalize(y):
    return y / (np.max(np.abs(y)) + 1e-9)


def extract_mfcc(y, sr=SR, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])


def offline_test(model, file_path):
    y, sr = librosa.load(file_path, sr=SR)
    win_len = int(WINDOW_SEC * sr)

    detections = []
    for start in range(0, len(y) - win_len, win_len):
        chunk = normalize(y[start:start + win_len])
        feat = extract_mfcc(chunk, sr).reshape(1, -1)
        pred = model.predict(feat)[0]
        if pred == 1:
            t = start / sr
            detections.append(t)
            print(f"  Whistle detected at ~{t:.2f}s")

    print(f"\nTotal detections: {len(detections)}")


def live_test(model):
    import sounddevice as sd

    def callback(indata, frames, time_info, status):
        signal = normalize(indata[:, 0])
        feat = extract_mfcc(signal, SR).reshape(1, -1)
        pred = model.predict(feat)[0]
        if pred == 1:
            print("Whistle detected!")

    print("Listening... press Enter to stop")
    with sd.InputStream(channels=1, samplerate=SR,
                         blocksize=int(WINDOW_SEC * SR), callback=callback):
        input()


def main():
    if len(sys.argv) < 2:
        print("Usage: python 08_realtime_test.py [offline <path>|live]")
        return

    model = joblib.load(ROOT / "models" / "whistle_svm_model.pkl")
    mode = sys.argv[1]

    if mode == "offline":
        if len(sys.argv) < 3:
            print("Usage: python 08_realtime_test.py offline path/to/file.wav")
            return
        offline_test(model, sys.argv[2])
    elif mode == "live":
        live_test(model)
    else:
        print("Unknown mode. Use 'offline' or 'live'.")


if __name__ == "__main__":
    main()
