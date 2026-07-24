"""
Step 3: Extract negative (non-whistle) clips from the SAME match audio,
sampled away from any whistle timestamp.

These are "hard negatives" -- crowd noise, ball hits, shouting -- from the
exact same recording domain as your positives.
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf
import pandas as pd
from scipy import signal

ROOT = Path(__file__).parent.parent
JSON_PATH = ROOT / "raw_data" / "volleylitics" / "whistles_all_reanchored.json"
AUDIO_DIR = ROOT / "raw_data" / "volleylitics"
OUT_DIR = ROOT / "processed" / "clips" / "non_whistle"

SR = 22050
SEG_LEN = 1.5         # Updated to 1.5s to match positive whistle clips
MIN_GAP = 2.0         # seconds away from any whistle timestamp
PER_MATCH = 60        # negatives to pull per match


def extract_negatives(audio, sr, whistle_times, n_segments, seg_len=SEG_LEN, min_gap=MIN_GAP):
    duration = len(audio) / sr
    seg_samples = int(seg_len * sr)
    segments = []
    attempts = 0
    max_attempts = n_segments * 30

    while len(segments) < n_segments and attempts < max_attempts:
        t = np.random.uniform(0, max(0.1, duration - seg_len))
        if all(abs(t - wt) > min_gap for wt in whistle_times):
            start = int(t * sr)
            segments.append((t, audio[start:start + seg_samples]))
        attempts += 1

    return segments


def main():
    with open(JSON_PATH) as f:
        whistles = json.load(f)

    by_match = defaultdict(list)
    for w in whistles:
        by_match[w["match_id"]].append(w["t_anchor"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_rows = []

    for match_id, whistle_times in by_match.items():
        wav_path = AUDIO_DIR / f"{match_id}.wav"
        if not wav_path.exists():
            print(f"  Skipping {match_id}: audio file not found")
            continue

        print(f"Extracting negatives from {match_id}...")
        
        # Safe audio loading without librosa/soxr dependency
        audio, native_sr = sf.read(wav_path)
        
        # Convert stereo to mono if necessary
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample to 22050 Hz if necessary using scipy
        if native_sr != SR:
            num_samples = int(len(audio) * SR / native_sr)
            audio = signal.resample(audio, num_samples)
            sr = SR
        else:
            sr = native_sr

        segments = extract_negatives(audio, sr, whistle_times, PER_MATCH)

        for i, (t, clip) in enumerate(segments):
            fname = f"{match_id}_neg{i}.wav"
            out_path = OUT_DIR / fname
            sf.write(out_path, clip, sr)

            index_rows.append({
                "match_id": match_id,
                "filename": fname,
                "start_time": t,
                "label": 0,
                "source": "volleylitics_negative",
            })

    df = pd.DataFrame(index_rows)
    out_csv = ROOT / "processed" / "match_negative_index.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nExtracted {len(df)} match negative clips -> {out_csv}")


if __name__ == "__main__":
    main()