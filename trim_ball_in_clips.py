"""
trim_ball_in_clips.py

Automatically re-cuts ball_in's raw video clips to tighter bounds around
the actual gesture, instead of manually scrubbing 199 clips by hand.

HOW: reuses the elbow-angle time series already sitting in the
extracted .npy keypoint files (same columns train.py's tie-breaker
uses). Finds the first and last frame where the active arm is
confirmed near-fully-extended (elbow_angle > ACTIVE_THRESHOLD), adds a
small buffer on each side for natural lead-in/lead-out motion, and
re-writes the ORIGINAL video (not the keypoints) trimmed to that range.

WHY THIS MATTERS FOR ball_in SPECIFICALLY: unlike the other classes,
whose neutral start/end pose (arms down) looks nothing like their
active gesture, ball_in's active pose sits fairly close to a relaxed/
resting arm (per the earlier direction diagnostic). Generous neutral
padding on this class means the model gets trained on a lot of frames
that LOOK idle but are labeled "ball_in" for the whole clip -- directly
teaching it to associate idle-looking poses with this class. Trimming
tighter removes that mislabeled padding without needing to refilm
anything.

DOES NOT overwrite your original clips. Writes tightened copies to a
separate output folder so you can compare, and only swap them into
data/raw_clips/ball_in/ once you're satisfied.

USAGE:
    python trim_ball_in_clips.py
    python trim_ball_in_clips.py ball_out   # reuse on any other class
"""

import os
import csv
import numpy as np
import cv2

MANIFEST_PATH = "data/dataset_manifest.csv"
OUTPUT_DIR_TEMPLATE = "data/raw_clips_trimmed/{label}"

POSE_FEATURES = 24
HAND_COORD_FEATURES = 84
HAND_FLAG_FEATURES = 2
FINGER_FEATURES = 10
LEFT_ELBOW_ANGLE_IDX = POSE_FEATURES + HAND_COORD_FEATURES + HAND_FLAG_FEATURES + FINGER_FEATURES  # 120
RIGHT_ELBOW_ANGLE_IDX = LEFT_ELBOW_ANGLE_IDX + 1  # 121

ACTIVE_THRESHOLD = 0.85   # elbow_angle above this = counted as "arm extended, gesture active"
BUFFER_FRAMES = 4         # extra frames kept before/after the detected active range,
                            # for natural lead-in/lead-out motion -- tune if trims feel too tight


def load_manifest_rows(target_label):
    with open(MANIFEST_PATH, "r") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("gesture_label") == target_label and r.get("keypoint_path") and r.get("clip_path")]


def find_active_range(keypoint_path):
    raw = np.load(keypoint_path)
    left_angles = raw[:, LEFT_ELBOW_ANGLE_IDX]
    right_angles = raw[:, RIGHT_ELBOW_ANGLE_IDX]

    active_side = "left" if left_angles.max() >= right_angles.max() else "right"
    active_angles = left_angles if active_side == "left" else right_angles

    active_frame_indices = np.where(active_angles > ACTIVE_THRESHOLD)[0]
    if len(active_frame_indices) == 0:
        return None

    start = max(0, active_frame_indices[0] - BUFFER_FRAMES)
    end = min(len(raw) - 1, active_frame_indices[-1] + BUFFER_FRAMES)
    return start, end, len(raw)


def trim_video(clip_path, start_frame, end_frame, output_path):
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        if start_frame <= frame_idx <= end_frame:
            writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return True


def main():
    import sys
    target_label = sys.argv[1] if len(sys.argv) > 1 else "ball_in"
    rows = load_manifest_rows(target_label)

    if not rows:
        print(f"No clips found with gesture_label == '{target_label}' that have both a clip_path and keypoint_path.")
        return

    output_dir = OUTPUT_DIR_TEMPLATE.format(label=target_label)
    print(f"Trimming {len(rows)} '{target_label}' clips -> {output_dir}/\n")
    print(f"{'clip':<45} {'orig_frames':>11} {'kept_frames':>11} {'kept_%':>7}")

    total_orig, total_kept = 0, 0
    skipped = []

    for row in rows:
        active_range = find_active_range(row["keypoint_path"])
        if active_range is None:
            skipped.append(row["clip_path"])
            continue

        start, end, orig_len = active_range
        kept_len = end - start + 1
        filename = os.path.basename(row["clip_path"])
        output_path = os.path.join(output_dir, filename)

        ok = trim_video(row["clip_path"], start, end, output_path)
        if not ok:
            skipped.append(row["clip_path"])
            continue

        total_orig += orig_len
        total_kept += kept_len
        print(f"{filename:<45} {orig_len:>11} {kept_len:>11} {kept_len / orig_len * 100:>6.1f}%")

    print("\n" + "=" * 50)
    if total_orig > 0:
        print(f"Overall: kept {total_kept}/{total_orig} frames ({total_kept / total_orig * 100:.1f}%) across all clips")
    print(f"Trimmed clips written to: {output_dir}/")
    print("Originals in data/raw_clips/ were NOT modified.")
    print("Review a few trimmed clips before swapping them in -- if BUFFER_FRAMES feels")
    print("too tight or too loose, adjust the constant at the top of this script and rerun.")

    if skipped:
        print(f"\n{len(skipped)} clip(s) skipped (no confirmed active frames or unreadable video):")
        for path in skipped:
            print(f"  - {path}")


if __name__ == "__main__":
    main()