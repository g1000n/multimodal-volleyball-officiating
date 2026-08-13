"""
analyze_hand_detection.py

Reads the ALREADY-EXTRACTED .npy keypoint files (from extract_keypoints.py)
and reports left/right hand detection rates per gesture class.

This does NOT re-run MediaPipe — it just reads the detection flags that
were already saved in each .npy file (columns 108 and 109), so it runs
in seconds instead of re-processing every video.

Run this any time you want to check hand-detection quality without
re-doing the slow extraction step.
""" 

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import numpy as np
from collections import defaultdict

MANIFEST_PATH = "data/dataset_manifest.csv"

LEFT_DETECTED_COL = 108
RIGHT_DETECTED_COL = 109


def analyze():
    with open(MANIFEST_PATH, "r") as f:
        rows = list(csv.DictReader(f))

    rows = [r for r in rows if r.get("keypoint_path") and r["keypoint_path"] != ""]

    if not rows:
        print("No rows with extracted keypoints found in the manifest.")
        return

    rates_by_class = defaultdict(list)

    for row in rows:
        keypoints = np.load(row["keypoint_path"])

        if keypoints.shape[1] < 110:
            print(f"WARNING: {row['keypoint_path']} has only {keypoints.shape[1]} "
                  f"features — this file predates the hand-flag update. "
                  f"Re-run extract_keypoints.py to regenerate it.")
            continue

        left_rate = keypoints[:, LEFT_DETECTED_COL].mean()
        right_rate = keypoints[:, RIGHT_DETECTED_COL].mean()
        rates_by_class[row["gesture_label"]].append((left_rate, right_rate))

    print("=" * 60)
    print("HAND DETECTION RATE PER CLASS (from existing keypoint files)")
    print("=" * 60)
    print("A naturally unused hand (resting arm in a one-handed gesture) SHOULD")
    print("show low detection — that's expected, not a problem. Watch for the")
    print("hand that SHOULD be active also showing low detection — that's the")
    print("real issue to investigate.\n")
    print(f"  {'class':<30} {'left hand':>12} {'right hand':>12}  {'clips':>6}")

    for gesture_label, rate_pairs in sorted(rates_by_class.items()):
        left_rates = [p[0] for p in rate_pairs]
        right_rates = [p[1] for p in rate_pairs]
        print(f"  {gesture_label:<30} {np.mean(left_rates):>11.1%} "
              f"{np.mean(right_rates):>12.1%}  {len(rate_pairs):>6}")


if __name__ == "__main__":
    analyze()