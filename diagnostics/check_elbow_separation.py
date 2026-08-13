"""
check_elbow_separation.py

Diagnostic: for two given classes, prints the average elbow-angle value
(index 120=left, 121=right, per extract_keypoints.py's raw 122-feature
layout) across all training clips of each class, so you can see whether
this feature actually separates them or not. Run from your project root
where data/dataset_manifest.csv and the .npy keypoint files live.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import csv
import numpy as np
import sys

MANIFEST_PATH = "data/dataset_manifest.csv"
LEFT_ELBOW_IDX = 120
RIGHT_ELBOW_IDX = 121

def avg_elbow(rows, elbow_idx):
    vals = []
    for r in rows:
        if not r.get("keypoint_path"):
            continue
        arr = np.load(r["keypoint_path"])
        vals.append(arr[:, elbow_idx].mean())
    return np.array(vals)

def main():
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))

    for label, elbow_idx, side in [
        ("team_to_serve_left", LEFT_ELBOW_IDX, "left"),
        ("service_authorization_left", LEFT_ELBOW_IDX, "left"),
    ]:
        class_rows = [r for r in rows if r["gesture_label"] == label and r["person_id"] != "pmax"]
        vals = avg_elbow(class_rows, elbow_idx)
        print(f"{label:<32} n={len(vals):<4} mean={vals.mean():.3f} std={vals.std():.3f} "
              f"min={vals.min():.3f} max={vals.max():.3f}")

if __name__ == "__main__":
    main()