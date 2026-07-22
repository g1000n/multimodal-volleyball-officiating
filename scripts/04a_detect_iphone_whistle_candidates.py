"""
Step 4a: Detect candidate whistle timestamps in iPhone recordings.

Your iPhone recordings don't come with a whistles.json like Volleylitics does --
this script proposes candidate timestamps by finding sharp energy peaks (whistles
are short, loud, high-energy bursts against quieter background), then you confirm
or reject each candidate by ear before anything gets extracted as training data.

Run this BEFORE 04b_extract_iphone_whistles.py.

Output: processed/iphone_whistle_candidates.csv -- one row per candidate spike,
with a VLC jump time and a blank "confirmed" column for you to fill in (y/n).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import find_peaks

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "raw_data" / "iphone_recordings"
OUT_CSV = ROOT / "processed" / "iphone_whistle_candidates.csv"

SR = 22050
FRAME_SEC = 0.1          # RMS window size
MIN_GAP_SEC = 1.0        # candidates closer than this are merged (same whistle)
PERCENTILE_THRESHOLD = 92  # only the loudest ~8% of frames become candidates

# Only scan folders that contain whistle sounds. "negative_audio" is deliberately
# excluded -- it has no whistles, and running candidate detection on it would just
# waste time (or worse, false-flag a loud non-whistle noise as a "whistle").
WHISTLE_SUBDIRS = ["positive_audio", "positive_negative_audio"]


def to_mmss(t):
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def rms_envelope(y, sr, frame_sec=FRAME_SEC):
    frame_len = int(frame_sec * sr)
    n_frames = len(y) // frame_len
    envelope = np.array([
        np.sqrt(np.mean(y[i * frame_len:(i + 1) * frame_len] ** 2))
        for i in range(n_frames)
    ])
    times = np.arange(n_frames) * frame_sec
    return times, envelope


def main():
    rows = []
    source_files = []
    for sub in WHISTLE_SUBDIRS:
        d = RAW_DIR / sub
        if not d.exists():
            print(f"  Note: expected folder not found: {d}")
            continue
        source_files.extend(d.glob("*.wav"))
        source_files.extend(d.glob("*.WAV"))
        source_files.extend(d.glob("*_converted.wav"))
    source_files = sorted(set(source_files))

    if not source_files:
        print(f"No .wav files found in {WHISTLE_SUBDIRS} under {RAW_DIR}.")
        print("If you only have .m4a files, convert them to .wav first (see 04's convert_to_wav), then rerun this.")
        return

    for wav_path in source_files:
        print(f"Scanning {wav_path.relative_to(RAW_DIR)}...")
        y, sr = sf.read(wav_path)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        if sr != SR:
            # simple guard -- flag rather than silently mismatch downstream SR assumptions
            print(f"  Note: {wav_path.name} is at {sr} Hz, not {SR} Hz. Extraction step "
                  f"will resample; candidate timestamps below are still valid in seconds.")

        times, envelope = rms_envelope(y, sr)
        if len(envelope) == 0:
            continue

        threshold = np.percentile(envelope, PERCENTILE_THRESHOLD)
        peak_indices, _ = find_peaks(
            envelope, height=threshold, distance=int(MIN_GAP_SEC / FRAME_SEC)
        )

        for idx in peak_indices:
            t = times[idx]
            rows.append({
                "source_file": str(wav_path.relative_to(RAW_DIR)),
                "t_candidate_sec": round(float(t), 2),
                "vlc_jump_time": to_mmss(max(0, t - 1)),
                "confirmed": "",  # fill in: y = real whistle, n = false positive
                "notes": "",
            })

    if not rows:
        print("No candidates found. Try lowering PERCENTILE_THRESHOLD and rerun.")
        return

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n{len(df)} candidate whistle timestamps found -> {OUT_CSV}")
    print("Open this CSV, listen to each candidate in VLC (Ctrl+T, jump to vlc_jump_time),")
    print("and mark 'confirmed' as y or n. Then run 04b_extract_iphone_whistles.py.")


if __name__ == "__main__":
    main()