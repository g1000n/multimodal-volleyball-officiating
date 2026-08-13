"""
check_elevation_separation.py

Diagnostic: does WRIST ELEVATION (how high the wrist sits relative to
the shoulder) actually separate ball_in from team_to_serve_left/right
in the real data? This is a DIFFERENT signal than elbow angle (which
measures how BENT the arm is) -- ball_in and team_to_serve are both
fairly straight-armed gestures, so elbow angle can't tell them apart.
The hypothesis (from live testing + visual review) is that ball_in
points lower/more downward than team_to_serve's higher point.

Elevation metric: (shoulder_y - wrist_y) / shoulder_width_in_frame
  - POSITIVE and LARGE  = wrist well above the shoulder (a high point)
  - near ZERO or NEGATIVE = wrist at or below shoulder height (a lower point)
(image y increases downward, so shoulder_y - wrist_y > 0 means wrist is
physically higher up in the frame than the shoulder)

team_to_serve_left/right are side-specific -- uses that side's own
shoulder/wrist. ball_in has no fixed side (filmed on either arm under
one label), so it uses whichever arm shows the LARGER elevation in
each clip (the "active" pointing arm).

Run from your project root:
    python check_elevation_separation.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import csv
import numpy as np

MANIFEST_PATH = "data/dataset_manifest.csv"

# Raw pose layout: 8 landmarks x (x, y, visibility), in this order:
# LEFT_SHOULDER(0), RIGHT_SHOULDER(1), LEFT_ELBOW(2), RIGHT_ELBOW(3),
# LEFT_WRIST(4), RIGHT_WRIST(5), LEFT_HIP(6), RIGHT_HIP(7)
LEFT_SHOULDER, RIGHT_SHOULDER = 0, 1
LEFT_WRIST, RIGHT_WRIST = 4, 5


def landmark_xy(pose_frame, idx):
    return pose_frame[idx * 3], pose_frame[idx * 3 + 1]


def elevation_for_side(seq, side):
    shoulder_idx = LEFT_SHOULDER if side == "left" else RIGHT_SHOULDER
    wrist_idx = LEFT_WRIST if side == "left" else RIGHT_WRIST
    left_sh_x, left_sh_y = landmark_xy(seq[:, :24], LEFT_SHOULDER) if False else (None, None)

    vals = []
    for frame in seq:
        sh_x, sh_y = landmark_xy(frame, shoulder_idx)
        wr_x, wr_y = landmark_xy(frame, wrist_idx)
        l_sh_x, l_sh_y = landmark_xy(frame, LEFT_SHOULDER)
        r_sh_x, r_sh_y = landmark_xy(frame, RIGHT_SHOULDER)
        shoulder_width = abs(l_sh_x - r_sh_x)
        if shoulder_width < 1e-6:
            continue
        elevation = (sh_y - wr_y) / shoulder_width
        vals.append(elevation)
    return np.array(vals) if vals else np.array([0.0])


def avg_elevation_ball_in(seq):
    left_vals = elevation_for_side(seq, "left")
    right_vals = elevation_for_side(seq, "right")
    # whichever arm is more active (higher average elevation) is the
    # pointing arm for this clip
    return max(left_vals.mean(), right_vals.mean())


def main():
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))

    results = {}
    for label, side in [
        ("team_to_serve_left", "left"),
        ("team_to_serve_right", "right"),
    ]:
        class_rows = [r for r in rows if r["gesture_label"] == label and r.get("keypoint_path") and r["person_id"] != "pmax"]
        vals = []
        for r in class_rows:
            seq = np.load(r["keypoint_path"])
            vals.append(elevation_for_side(seq, side).mean())
        vals = np.array(vals)
        results[label] = vals
        print(f"{label:<24} n={len(vals):<4} mean={vals.mean():.3f} std={vals.std():.3f} "
              f"min={vals.min():.3f} max={vals.max():.3f}")

    ball_in_rows = [r for r in rows if r["gesture_label"] == "ball_in" and r.get("keypoint_path")]
    ball_in_vals = np.array([avg_elevation_ball_in(np.load(r["keypoint_path"])) for r in ball_in_rows])
    results["ball_in"] = ball_in_vals
    print(f"{'ball_in':<24} n={len(ball_in_vals):<4} mean={ball_in_vals.mean():.3f} std={ball_in_vals.std():.3f} "
          f"min={ball_in_vals.min():.3f} max={ball_in_vals.max():.3f}")

    print("\nOverlap check:")
    for label in ["team_to_serve_left", "team_to_serve_right"]:
        tts_vals = results[label]
        overlap = np.sum((ball_in_vals.max() >= tts_vals.min()) & (ball_in_vals.min() <= tts_vals.max()))
        print(f"  ball_in range [{ball_in_vals.min():.3f}, {ball_in_vals.max():.3f}] vs "
              f"{label} range [{tts_vals.min():.3f}, {tts_vals.max():.3f}]")
        if ball_in_vals.max() < tts_vals.min() or ball_in_vals.min() > tts_vals.max():
            print(f"    -> CLEAN SEPARATION (no overlap)")
        else:
            print(f"    -> RANGES OVERLAP")


if __name__ == "__main__":
    main()