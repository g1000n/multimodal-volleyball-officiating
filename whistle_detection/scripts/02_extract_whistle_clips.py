"""
Step 2: Extract whistle clips from the 14 match recordings using t_anchor
timestamps from the reanchored JSON.

Adjust PRE / POST / TARGET_LEN after your listening QC pass from step 1.

NOTE ON 'type' FIELD: labeling is currently binary (whistle=1). The JSON's
"type" field (serve / rally_end / other) is preserved in the index CSV as
an extra column ("whistle_type") even though it isn't used for anything yet.
That means if you later want to classify by type, you won't need to
re-extract clips -- just re-run step 5/6 using that column instead of the
binary label.
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import librosa
import soundfile as sf

ROOT = Path(__file__).parent.parent
JSON_PATH = ROOT / "raw_data" / "volleylitics" / "whistles_all_reanchored.json"
AUDIO_DIR = ROOT / "raw_data" / "volleylitics"
OUT_DIR = ROOT / "processed" / "clips" / "whistle"

SR = 22050          # matches Volleylitics native sample rate
PRE = 0.3            # seconds before anchor
POST = 0.7           # seconds after anchor
TARGET_LEN = 1.0     # final fixed clip length in seconds
MAX_PER_MATCH = 60   # cap so no single match dominates the dataset
                      # (every one of your 10 annotated matches has 265+ whistles,
                      # so this cap is doing real balancing work -- ~600 total clips)


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

    for match_id, entries in by_match.items():
        wav_path = AUDIO_DIR / f"{match_id}.wav"
        if not wav_path.exists():
            print(f"  Skipping {match_id}: audio file not found at {wav_path}")
            continue

        print(f"Processing {match_id} ({len(entries)} whistles)...")
        audio, sr = librosa.load(wav_path, sr=SR)

        # cap per match to avoid one match dominating
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
                "whistle_type": w.get("type", "unknown"),  # kept for future use
            })

    import pandas as pd
    df = pd.DataFrame(index_rows)
    out_csv = ROOT / "processed" / "whistle_index.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nExtracted {len(df)} whistle clips -> {out_csv}")


if __name__ == "__main__":
    main()
