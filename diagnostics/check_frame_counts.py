"""
check_frame_counts.py

Compares raw frame_count between pmax's clips and your own real clips,
for the same shared classes. If pmax's clips are systematically much
shorter/longer (or just always exactly 30 vs. your varying real
lengths), that's a real candidate explanation for the training
instability -- resample_sequence()'s interpolation behaves differently
depending on how much "real" motion information it's stretching or
compressing to reach SEQUENCE_LENGTH=50.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import csv
import numpy as np

MANIFEST_PATH = "data/dataset_manifest.csv"
SHARED_CLASSES = ["ball_out", "double_contact", "team_to_serve_left", "team_to_serve_right"]

def main():
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))

    for label in SHARED_CLASSES:
        pmax_counts = [int(r["frame_count"]) for r in rows
                       if r["gesture_label"] == label and r["person_id"] == "pmax" and r.get("frame_count")]
        real_counts = [int(r["frame_count"]) for r in rows
                       if r["gesture_label"] == label and r["person_id"] != "pmax" and r.get("frame_count")]

        pmax_arr = np.array(pmax_counts)
        real_arr = np.array(real_counts)

        print(f"\n{label}:")
        if len(pmax_arr):
            print(f"  pmax  n={len(pmax_arr):<4} mean={pmax_arr.mean():.1f} std={pmax_arr.std():.1f} "
                  f"min={pmax_arr.min()} max={pmax_arr.max()}")
        if len(real_arr):
            print(f"  real  n={len(real_arr):<4} mean={real_arr.mean():.1f} std={real_arr.std():.1f} "
                  f"min={real_arr.min()} max={real_arr.max()}")

if __name__ == "__main__":
    main()