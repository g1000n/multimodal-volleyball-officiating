"""
Step 4b: Extract whistle-positive clips from iPhone recordings, using the
confirmed candidates from 04a_detect_iphone_whistle_candidates.py.

Run this AFTER filling in the "confirmed" column (y/n) in
processed/iphone_whistle_candidates.csv.

Uses the SAME PRE/POST/TARGET_LEN as 02_extract_whistle_clips.py so iPhone-sourced
whistle clips are directly comparable/poolable with Volleylitics-sourced ones.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "raw_data" / "iphone_recordings"
CANDIDATES_CSV = ROOT / "processed" / "iphone_whistle_candidates.csv"
OUT_DIR = ROOT / "processed" / "clips" / "whistle"

SR = 22050
PRE = 0.3            # matches 02_extract_whistle_clips.py
POST = 1.2           # matches 02_extract_whistle_clips.py
TARGET_LEN = 1.5     # matches 02_extract_whistle_clips.py


def extract_clip(audio, sr, t_anchor, pre=PRE, post=POST, target_len=TARGET_LEN):
    center = int(t_anchor * sr)
    start = max(0, center - int(pre * sr))
    end = center + int(post * sr)
    clip = audio[start:end]

    target_samples = int(target_len * sr)
    if len(clip) < target_samples:
        clip = np.pad(clip, (0, target_samples - len(clip)))
    else:
        clip = clip[:target_samples]
    return clip


def main():
    if not CANDIDATES_CSV.exists():
        print(f"Missing {CANDIDATES_CSV}. Run 04a_detect_iphone_whistle_candidates.py first.")
        return

    df = pd.read_csv(CANDIDATES_CSV)
    confirmed = df[df["confirmed"].astype(str).str.strip().str.lower() == "y"]
    if confirmed.empty:
        print("No rows marked confirmed='y' yet. Fill in the CSV and rerun.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_rows = []

    for source_file, group in confirmed.groupby("source_file"):
        wav_path = RAW_DIR / source_file
        if not wav_path.exists():
            print(f"  Skipping {source_file}: not found at {wav_path}")
            continue

        print(f"Extracting {len(group)} confirmed whistle(s) from {source_file}...")
        audio, sr = sf.read(wav_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        stem = Path(source_file).stem
        for i, row in enumerate(group.itertuples()):
            clip = extract_clip(audio, sr, row.t_candidate_sec)
            fname = f"iphone_{stem}_w{i}.wav"
            out_path = OUT_DIR / fname
            sf.write(out_path, clip, sr)

            index_rows.append({
                "match_id": f"iphone_{stem}",  # own group, kept separate from Volleylitics matches
                "filename": fname,
                "start_time": row.t_candidate_sec,
                "label": 1,
                "source": "iphone_whistle",
            })

    out_csv = ROOT / "processed" / "iphone_whistle_index.csv"
    out_df = pd.DataFrame(index_rows)
    out_df.to_csv(out_csv, index=False)
    print(f"\nExtracted {len(out_df)} iPhone whistle clips -> {out_csv}")


if __name__ == "__main__":
    main()