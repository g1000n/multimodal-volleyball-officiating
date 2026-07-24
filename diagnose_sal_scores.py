"""
diagnose_sal_scores.py

Quick check: for every service_authorization_left TEST clip, what raw
sigmoid score does that class's own output neuron actually give it?
This tells us whether the 0.00/0.00 collapse is:
  - a THRESHOLD problem (scores cluster just under 0.5, e.g. 0.35-0.48)
    -> safe, quick fix: lower DECISION_THRESHOLD, or give SAL its own
       lower per-class threshold
  - a LEARNING problem (scores near 0.0-0.15)
    -> NOT safe to patch with a threshold change; would need more/
       better SAL data, not something to attempt the night before a
       real test

Run:
    python diagnose_sal_scores.py
"""

import csv
import json
import numpy as np
import torch

from model import GestureCNNLSTM
from train import (
    normalize_sequence,
    resample_sequence,
    SEQUENCE_LENGTH,
    TOTAL_FEATURES,
    ablate,
    ABLATE_HAND_COORDS,
    ABLATED_FEATURE_COUNT,
)

MANIFEST_PATH = "data/dataset_manifest.csv"
MODEL_PATH = "models/final_model.pt"
LABEL_MAP_PATH = "models/label_map.json"
TARGET_CLASS = "service_authorization_left"


def main():
    with open(LABEL_MAP_PATH) as f:
        label_map_data = json.load(f)
    real_label_to_idx = label_map_data["real_label_to_idx"]
    target_idx = real_label_to_idx[TARGET_CLASS]

    input_size = ABLATED_FEATURE_COUNT if ABLATE_HAND_COORDS else TOTAL_FEATURES
    model = GestureCNNLSTM(input_size=input_size, num_classes=len(real_label_to_idx))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with open(MANIFEST_PATH, "r") as f:
        rows = list(csv.DictReader(f))

    test_rows = [r for r in rows if r["gesture_label"] == TARGET_CLASS and r["split"] == "test"]
    print(f"Found {len(test_rows)} test clips for {TARGET_CLASS}.\n")

    scores = []
    with torch.no_grad():
        for row in test_rows:
            raw = np.load(row["keypoint_path"])
            normalized = normalize_sequence(raw)
            normalized = ablate(normalized)
            resampled = resample_sequence(normalized, SEQUENCE_LENGTH)
            x = torch.tensor(resampled, dtype=torch.float32).unsqueeze(0)

            logits = model(x)
            probs = torch.sigmoid(logits).numpy()[0]
            score = probs[target_idx]
            scores.append(score)
            print(f"  {row['clip_path']:<60} {TARGET_CLASS} score: {score:.3f}  (person: {row['person_id']})")

    scores = np.array(scores)
    print(f"\nMean score: {scores.mean():.3f} | Max: {scores.max():.3f} | Min: {scores.min():.3f}")
    print(f"How many clips scored above 0.35: {(scores > 0.35).sum()}/{len(scores)}")
    print(f"How many clips scored above 0.20: {(scores > 0.20).sum()}/{len(scores)}")

    if scores.max() > 0.35:
        print("\n-> Scores get reasonably close to 0.5 -- likely a THRESHOLD problem, "
              "safe to try lowering DECISION_THRESHOLD or giving this class its own lower threshold.")
    else:
        print("\n-> Scores are consistently very low -- likely a genuine LEARNING problem, "
              "not safe to patch with a threshold change alone.")


if __name__ == "__main__":
    main()