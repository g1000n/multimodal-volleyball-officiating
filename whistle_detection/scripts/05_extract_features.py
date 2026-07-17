"""
Step 5: Merge the three index files (whistle, match negatives, iphone
negatives) and extract MFCC features for every clip.

Each clip -> one fixed-length feature vector (mean + std of 13 MFCCs = 26 dims).
"""
from pathlib import Path

import numpy as np
import librosa
import pandas as pd

ROOT = Path(__file__).parent.parent
CLIPS_DIR = ROOT / "processed" / "clips"
SR = 22050
N_MFCC = 13


def extract_mfcc(y, sr=SR, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)
    return np.concatenate([mean, std])


def normalize(y):
    return y / (np.max(np.abs(y)) + 1e-9)


def main():
    index_files = [
        ROOT / "processed" / "whistle_index.csv",
        ROOT / "processed" / "match_negative_index.csv",
        ROOT / "processed" / "iphone_negative_index.csv",
    ]

    dfs = []
    for f in index_files:
        if f.exists():
            dfs.append(pd.read_csv(f))
        else:
            print(f"  Warning: {f} not found, skipping (run earlier steps first)")

    if not dfs:
        print("No index files found. Run steps 2-4 first.")
        return

    full_index = pd.concat(dfs, ignore_index=True)
    print(f"Total clips indexed: {len(full_index)}")
    print(full_index["label"].value_counts())

    feature_rows = []
    for _, row in full_index.iterrows():
        subfolder = "whistle" if row["label"] == 1 else "non_whistle"
        clip_path = CLIPS_DIR / subfolder / row["filename"]

        if not clip_path.exists():
            continue

        y, sr = librosa.load(clip_path, sr=SR)
        y = normalize(y)
        feat = extract_mfcc(y, sr)

        feature_rows.append({
            "match_id": row["match_id"],
            "filename": row["filename"],
            "label": row["label"],
            "source": row["source"],
            **{f"mfcc_{i}": v for i, v in enumerate(feat)},
        })

    feat_df = pd.DataFrame(feature_rows)
    out_csv = ROOT / "processed" / "features.csv"
    feat_df.to_csv(out_csv, index=False)
    print(f"\nFeature matrix saved -> {out_csv} ({len(feat_df)} rows, "
          f"{feat_df.filter(like='mfcc_').shape[1]} feature dims)")


if __name__ == "__main__":
    main()
