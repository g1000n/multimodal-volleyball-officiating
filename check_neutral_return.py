"""
check_neutral_return.py

Automated check across EVERY clip in the manifest: does this clip
return to a neutral/resting pose at the end, or does it stay at the
gesture's peak position? Computes the distance between the pose at
the START of the clip and the pose at the END -- a clip that properly
follows the neutral -> gesture -> neutral convention should have a
LOW start-vs-end distance (it ends roughly where it began). A clip
that's cut off at the peak (arm still raised) will show a HIGH
distance.

WHY THIS MATTERS: this project has previously root-caused a real
accuracy regression to clips that didn't follow this convention
(mixed filming styles created inconsistent temporal shapes within one
class after resample_sequence() stretches everything to a fixed
length). This script checks whether MaxLSB's (pmax) converted clips
have the same issue, compared against your own real contributors,
across ALL classes at once -- instead of manually watching every clip.

Run:
    python check_neutral_return.py

Prints a per-class, per-person-group (pmax vs yours) summary table of
average start-vs-end pose distance, so you can see at a glance whether
pmax's clips are systematically different.
"""

import csv
from collections import defaultdict
import numpy as np

MANIFEST_PATH = "data/dataset_manifest.csv"
EDGE_FRAMES = 3  # average over this many frames at the start/end, to reduce single-frame noise


def start_end_distance(seq):
    """seq: raw (frames, 122) array. Compares the average pose (first
    24 values = 8 landmarks x,y,visibility) over the first EDGE_FRAMES
    frames against the same over the last EDGE_FRAMES frames."""
    if len(seq) < EDGE_FRAMES * 2:
        edge = max(1, len(seq) // 3)
    else:
        edge = EDGE_FRAMES

    pose = seq[:, :24].reshape(len(seq), 8, 3)
    pose_xy = pose[:, :, :2]  # drop visibility for the distance calc

    start_pose = pose_xy[:edge].mean(axis=0)
    end_pose = pose_xy[-edge:].mean(axis=0)

    return float(np.linalg.norm(end_pose - start_pose))


def main():
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))

    rows = [r for r in rows if r.get("keypoint_path")]

    # group[gesture_label]["pmax" or "yours"] -> list of distances
    grouped = defaultdict(lambda: defaultdict(list))

    for r in rows:
        seq = np.load(r["keypoint_path"])
        dist = start_end_distance(seq)
        group_key = "pmax" if r["person_id"] == "pmax" else "yours"
        grouped[r["gesture_label"]][group_key].append((dist, r["keypoint_path"]))

    print(f"{'class':<30} {'group':<8} {'n':>4} {'mean dist':>10} {'max dist':>10}")
    print("-" * 66)
    flagged = []
    for label in sorted(grouped):
        for group_key in ("pmax", "yours"):
            entries = grouped[label][group_key]
            if not entries:
                continue
            dists = [d for d, _ in entries]
            mean_d = np.mean(dists)
            max_d = np.max(dists)
            print(f"{label:<30} {group_key:<8} {len(dists):>4} {mean_d:>10.4f} {max_d:>10.4f}")

        # Flag classes where pmax's mean is notably higher than yours --
        # suggests pmax's clips are systematically not returning to
        # neutral the way your own clips do.
        pmax_entries = grouped[label]["pmax"]
        real_entries = grouped[label]["yours"]
        if pmax_entries and real_entries:
            pmax_mean = np.mean([d for d, _ in pmax_entries])
            real_mean = np.mean([d for d, _ in real_entries])
            if pmax_mean > real_mean * 1.5 and pmax_mean > 0.05:
                flagged.append((label, pmax_mean, real_mean))

    print("\n" + "=" * 66)
    if flagged:
        print("FLAGGED -- pmax's clips show notably higher start-vs-end distance")
        print("(i.e. likely NOT returning to neutral, unlike your own clips):")
        for label, pmax_mean, real_mean in flagged:
            print(f"  {label:<30} pmax={pmax_mean:.4f}  yours={real_mean:.4f}  "
                  f"({pmax_mean/real_mean:.1f}x higher)")
    else:
        print("No classes flagged -- pmax's start/end pose distance looks "
              "comparable to your own clips across all checked classes.")


if __name__ == "__main__":
    main()