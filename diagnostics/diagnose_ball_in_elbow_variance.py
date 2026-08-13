"""
diagnose_ball_in_arm_direction.py

CORRECTED diagnostic. diagnose_ball_in_elbow_variance.py measured
elbow_angle -- how STRAIGHT the elbow joint is (0=bent, 1=straight).
That's a real, useful signal (confirms the arm is always fully
extended), but it says nothing about which DIRECTION the extended arm
points relative to the body -- which is the actual thing being debated
(does "point 45 degrees toward the floor" mean a fixed direction, or
does it vary based on where the ball landed).

This script measures that directly: for frames where the elbow is
confirmed near-fully-straight (elbow_angle > STRAIGHT_ARM_THRESHOLD --
i.e. only the "held point" portion of the gesture, not the swing-up
motion), it computes the angle between the extended arm (shoulder ->
wrist vector) and the torso's own vertical axis (shoulder -> hip
vector, same side). 0 degrees = arm pointing straight down along the
body; larger angles = arm swung further out to the side/forward. This
is the actual geometric quantity the FIVB "45 degrees away from the
body" phrasing describes.

Uses RAW (un-normalized) coordinates directly from the already-
extracted .npy keypoint files -- pose landmark order (from
extract_keypoints.py's UPPER_BODY_LANDMARKS) is:
  0 LEFT_SHOULDER, 1 RIGHT_SHOULDER, 2 LEFT_ELBOW, 3 RIGHT_ELBOW,
  4 LEFT_WRIST, 5 RIGHT_WRIST, 6 LEFT_HIP, 7 RIGHT_HIP
each as (x, y, visibility) -- 3 values per landmark, 24 total.

USAGE:
    python diagnose_ball_in_arm_direction.py
    python diagnose_ball_in_arm_direction.py ball_out
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sys
import csv
import numpy as np

MANIFEST_PATH = "data/dataset_manifest.csv"

# Landmark index (not feature-column index) -- multiply by 3 to get the
# starting feature column for (x, y, visibility).
LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER = 0, 1
LM_LEFT_ELBOW, LM_RIGHT_ELBOW = 2, 3
LM_LEFT_WRIST, LM_RIGHT_WRIST = 4, 5
LM_LEFT_HIP, LM_RIGHT_HIP = 6, 7

POSE_FEATURES = 24
HAND_COORD_FEATURES = 84
HAND_FLAG_FEATURES = 2
FINGER_FEATURES = 10
LEFT_ELBOW_ANGLE_IDX = POSE_FEATURES + HAND_COORD_FEATURES + HAND_FLAG_FEATURES + FINGER_FEATURES  # 120
RIGHT_ELBOW_ANGLE_IDX = LEFT_ELBOW_ANGLE_IDX + 1  # 121

STRAIGHT_ARM_THRESHOLD = 0.9  # only look at frames where the elbow is confirmed
                                # near-fully-extended -- the "held point" portion.
MIN_VISIBILITY = 0.3           # skip landmarks MediaPipe wasn't confident about


def xy(raw_frame, landmark_idx):
    base = landmark_idx * 3
    return raw_frame[base], raw_frame[base + 1], raw_frame[base + 2]  # x, y, visibility


def arm_to_torso_angle_degrees(raw_frame, side):
    if side == "left":
        shoulder_lm, wrist_lm, hip_lm = LM_LEFT_SHOULDER, LM_LEFT_WRIST, LM_LEFT_HIP
    else:
        shoulder_lm, wrist_lm, hip_lm = LM_RIGHT_SHOULDER, LM_RIGHT_WRIST, LM_RIGHT_HIP

    sx, sy, s_vis = xy(raw_frame, shoulder_lm)
    wx, wy, w_vis = xy(raw_frame, wrist_lm)
    hx, hy, h_vis = xy(raw_frame, hip_lm)

    if s_vis < MIN_VISIBILITY or w_vis < MIN_VISIBILITY or h_vis < MIN_VISIBILITY:
        return None

    torso_vec = np.array([hx - sx, hy - sy])   # shoulder -> hip = the body's "down" axis
    arm_vec = np.array([wx - sx, wy - sy])     # shoulder -> wrist = the extended arm

    torso_norm = np.linalg.norm(torso_vec)
    arm_norm = np.linalg.norm(arm_vec)
    if torso_norm < 1e-6 or arm_norm < 1e-6:
        return None

    cos_angle = np.clip(np.dot(torso_vec, arm_vec) / (torso_norm * arm_norm), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def load_manifest_rows(target_label):
    with open(MANIFEST_PATH, "r") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("gesture_label") == target_label and r.get("keypoint_path")]


def analyze_clip(keypoint_path):
    raw = np.load(keypoint_path)
    left_elbow_angles = raw[:, LEFT_ELBOW_ANGLE_IDX]
    right_elbow_angles = raw[:, RIGHT_ELBOW_ANGLE_IDX]

    # Determine the active side the same way the elbow-variance script did --
    # whichever arm reaches the higher peak straightness.
    left_peak = left_elbow_angles.max()
    right_peak = right_elbow_angles.max()
    active_side = "left" if left_peak >= right_peak else "right"
    active_elbow_angles = left_elbow_angles if active_side == "left" else right_elbow_angles

    straight_frame_indices = np.where(active_elbow_angles > STRAIGHT_ARM_THRESHOLD)[0]
    if len(straight_frame_indices) == 0:
        return None

    directions = []
    for i in straight_frame_indices:
        angle = arm_to_torso_angle_degrees(raw[i], active_side)
        if angle is not None:
            directions.append(angle)

    if not directions:
        return None

    directions = np.array(directions)
    return {
        "active_side": active_side,
        "mean_direction_deg": directions.mean(),
        "std_direction_deg": directions.std(),
        "num_straight_frames": len(directions),
    }


def main():
    target_label = sys.argv[1] if len(sys.argv) > 1 else "ball_in"
    rows = load_manifest_rows(target_label)

    if not rows:
        print(f"No clips found with gesture_label == '{target_label}'.")
        return

    print(f"Analyzing arm DIRECTION (not just extension) across {len(rows)} '{target_label}' clips...\n")

    results, skipped = [], []
    for row in rows:
        result = analyze_clip(row["keypoint_path"])
        if result is None:
            skipped.append(row["keypoint_path"])
            continue
        result["clip"] = row["keypoint_path"]
        result["person_id"] = row.get("person_id", "?")
        results.append(result)

    if not results:
        print("No usable clips -- none had a confirmed straight-arm portion with visible landmarks.")
        return

    means = np.array([r["mean_direction_deg"] for r in results])

    print(f"{'clip':<50} {'person':<8} {'side':<6} {'mean_deg':>9} {'std_deg':>8} {'n_frames':>9}")
    for r in sorted(results, key=lambda x: x["mean_direction_deg"]):
        print(f"{r['clip']:<50} {r['person_id']:<8} {r['active_side']:<6} "
              f"{r['mean_direction_deg']:>9.1f} {r['std_direction_deg']:>8.1f} {r['num_straight_frames']:>9}")

    print("\n" + "=" * 70)
    print(f"SUMMARY across {len(results)} clips")
    print("Angle = degrees between the extended arm and the body's own vertical axis")
    print("(0 deg = pointing straight down along the body, 90 deg = pointing straight out to the side)")
    print("=" * 70)
    print(f"  overall mean angle:      {means.mean():.1f} deg")
    print(f"  overall std deviation:   {means.std():.1f} deg   <-- THIS is the number that answers the debate")
    print(f"  min / max (per-clip mean): {means.min():.1f} deg / {means.max():.1f} deg")
    print(f"  median:                  {np.median(means):.1f} deg")

    if means.std() < 8:
        print("\n  -> Tight clustering. Your referee's ball_in direction IS consistently fixed,")
        print("     matching the FIVB-standardized reading -- the earlier elbow-only check")
        print("     wasn't wrong about consistency, it was just measuring the wrong thing.")
    elif means.std() < 15:
        print("\n  -> Moderate spread -- some real directional variance across clips.")
    else:
        print("\n  -> Wide spread -- the arm direction genuinely varies clip to clip,")
        print("     supporting the original high-variance concern despite the elbow always being straight.")

    if skipped:
        print(f"\n{len(skipped)} clip(s) skipped (no confirmed straight-arm frames with visible landmarks):")
        for path in skipped:
            print(f"  - {path}")


if __name__ == "__main__":
    main()