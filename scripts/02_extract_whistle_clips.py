"""
Step 2: Extract whistle clips from the 14 match recordings using t_anchor
timestamps from the reanchored JSON.

Adjusted PRE/POST/TARGET_LEN based on QC pass, and added deduplication (merge) logic.
This version bypasses librosa and uses soundfile to prevent Windows DLL import crashes.
"""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import soundfile as sf

ROOT = Path(__file__).parent.parent
JSON_PATH = ROOT / "raw_data" / "volleylitics" / "whistles_all_reanchored.json"
AUDIO_DIR = ROOT / "raw_data" / "volleylitics"
OUT_DIR = ROOT / "processed" / "clips" / "whistle"

SR = 22050
PRE = 0.3
POST = 1.2
TARGET_LEN = 1.5
MAX_PER_MATCH = 60
MERGE_GAP = 0.5

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
    with open(JSON_PATH) as f:
        whistles = json.load(f)

    by_match = defaultdict(list)
    for w in whistles:
        by_match[w["match_id"]].append(w)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_rows = []
    total_merged = 0

    for match_id, entries in by_match.items():
        wav_path = AUDIO_DIR / f"{match_id}.wav"
        if not wav_path.exists():
            print(f"  Skipping {match_id}: audio file not found at {wav_path}")
            continue

        entries = sorted(entries, key=lambda w: w["t_anchor"])
        merged = []
        for w in entries:
            if merged and (w["t_anchor"] - merged[-1]["t_anchor"]) < MERGE_GAP:
                total_merged += 1
                continue
            merged.append(w)
        entries = merged

        print(f"Processing {match_id} ({len(entries)} whistles)...")
        audio, sr = sf.read(wav_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        entries = entries[:MAX_PER_MATCH]

        for w in entries:
            clip = extract_clip(audio, sr, w["t_anchor"])
            fname = f"{match_id}_w{w['whistle_id']}.wav"
            out_path = OUT_DIR / fname
            sf.write(out_path, clip, sr)

            index_rows.append({
                "match_id": match_id,
                "filename": fname,
                "start_time": w["t_anchor"],
                "label": 1,
                "source": "volleylitics_whistle",
                "whistle_type": w.get("type", "unknown"),
            })

    import pandas as pd
    df = pd.DataFrame(index_rows)
    out_csv = ROOT / "processed" / "whistle_index.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nMerged {total_merged} near-duplicate anchor(s) (< {MERGE_GAP}s apart)")
    print(f"Extracted {len(df)} whistle clips -> {out_csv}")

if __name__ == "__main__":
    main()