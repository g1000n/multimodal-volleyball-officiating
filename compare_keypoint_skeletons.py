"""
compare_keypoint_skeletons.py

Plays back saved keypoint sequences (.npy files, already extracted by
extract_keypoints.py OR converted from MaxLSB's data via
convert_maxlsb_nothing_data.py -- both are saved in this project's
same 122-feature format) as a SKELETON-ONLY animation, no video needed.

WHY: MaxLSB's raw clips aren't available to you, only his already-
converted keypoints -- but since those keypoints are stored in the same
format as your own extracted clips, they can be animated identically.
This lets you literally WATCH how he performed each gesture (arm
range, speed, whether he returns to neutral at the end) and compare it
side-by-side against one of your own real contributors performing the
SAME gesture class.

HOW TO USE:
    python compare_keypoint_skeletons.py

You'll get a NUMBERED MENU of gesture classes to pick from (easier
than typing the exact name). It'll then show, side by side:
  LEFT panel  -- one of pmax's (MaxLSB's) clips for that class
  RIGHT panel -- one of your own real contributors' clips for that class

Controls:
  N - next pair of clips (cycles through available ones)
  Q - quit
"""

import os
import csv
import cv2
import numpy as np

MANIFEST_PATH = "data/dataset_manifest.csv"
CANVAS_SIZE = 500
PLAYBACK_DELAY_MS = 60

# Matches extract_keypoints.py's UPPER_BODY_LANDMARKS order exactly:
# [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
#  LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP]
POSE_CONNECTIONS = [
    (0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7), (6, 7),
]


def draw_skeleton_frame(features_122, label_text, canvas_size=CANVAS_SIZE):
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    canvas[:] = (25, 25, 25)

    pose = features_122[:24].reshape(8, 3)
    points_px = []
    for x, y, vis in pose:
        px, py = int(x * canvas_size), int(y * canvas_size)
        points_px.append((px, py, vis))

    for a, b in POSE_CONNECTIONS:
        xa, ya, va = points_px[a]
        xb, yb, vb = points_px[b]
        if va > 0.3 and vb > 0.3:
            cv2.line(canvas, (xa, ya), (xb, yb), (0, 200, 0), 2)
    for px, py, vis in points_px:
        if vis > 0.3:
            cv2.circle(canvas, (px, py), 5, (0, 255, 0), -1)

    left_hand = features_122[24:66].reshape(21, 2)
    right_hand = features_122[66:108].reshape(21, 2)
    left_det = features_122[108]
    right_det = features_122[109]

    if left_det > 0.5:
        for x, y in left_hand:
            cv2.circle(canvas, (int(x * canvas_size), int(y * canvas_size)), 3, (255, 255, 0), -1)
    if right_det > 0.5:
        for x, y in right_hand:
            cv2.circle(canvas, (int(x * canvas_size), int(y * canvas_size)), 3, (255, 0, 255), -1)

    cv2.putText(canvas, label_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def load_all_rows():
    with open(MANIFEST_PATH) as f:
        return list(csv.DictReader(f))


def choose_gesture_class(rows):
    labels_with_pmax = sorted(set(
        r["gesture_label"] for r in rows
        if r.get("keypoint_path") and any(rr["gesture_label"] == r["gesture_label"] and rr["person_id"] == "pmax" for rr in rows)
    ))
    if not labels_with_pmax:
        print("No classes with pmax data found.")
        return None

    print("Gesture classes with pmax (MaxLSB) data available:")
    for i, label in enumerate(labels_with_pmax, start=1):
        pmax_count = sum(1 for r in rows if r["gesture_label"] == label and r["person_id"] == "pmax")
        real_count = sum(1 for r in rows if r["gesture_label"] == label and r["person_id"] != "pmax" and r.get("keypoint_path"))
        note = "" if real_count > 0 else "  (no comparison clips -- pmax-only view)"
        print(f"  {i}. {label}  (pmax: {pmax_count} clips, yours: {real_count} clips){note}")

    choice = input(f"\nPick a class (1-{len(labels_with_pmax)}): ").strip()
    try:
        return labels_with_pmax[int(choice) - 1]
    except (ValueError, IndexError):
        print("Invalid choice.")
        return None


def main():
    rows = load_all_rows()
    gesture_label = choose_gesture_class(rows)
    if gesture_label is None:
        return

    class_rows = [r for r in rows if r["gesture_label"] == gesture_label and r.get("keypoint_path")]
    pmax_rows = [r for r in class_rows if r["person_id"] == "pmax"]
    real_rows = [r for r in class_rows if r["person_id"] != "pmax"]

    if not pmax_rows:
        print("No pmax clips for this class -- nothing to show.")
        return

    # CHANGED: previously required BOTH pmax and real clips to exist,
    # crashing/refusing otherwise. Now falls back to a pmax-only single
    # panel if you have zero of your own clips for this class right now
    # (e.g. "nothing" currently has 0 real clips after removing p09's).
    comparison_mode = len(real_rows) > 0
    if not comparison_mode:
        print(f"\nNo real clips currently available for '{gesture_label}' -- showing pmax ONLY (single panel).")
    else:
        print(f"\nFound {len(pmax_rows)} pmax clips and {len(real_rows)} of your own clips for '{gesture_label}'.")
    print("Controls: N = next pair, Q = quit\n")

    idx = 0
    while True:
        pmax_row = pmax_rows[idx % len(pmax_rows)]
        pmax_seq = np.load(pmax_row["keypoint_path"])

        if comparison_mode:
            real_row = real_rows[idx % len(real_rows)]
            real_seq = np.load(real_row["keypoint_path"])
            print(f"[{idx+1}] pmax clip: {os.path.basename(pmax_row['keypoint_path'])} ({len(pmax_seq)} frames)  "
                  f"|  your clip ({real_row['person_id']}): {os.path.basename(real_row['keypoint_path'])} ({len(real_seq)} frames)")
            max_len = max(len(pmax_seq), len(real_seq))
        else:
            print(f"[{idx+1}] pmax clip: {os.path.basename(pmax_row['keypoint_path'])} ({len(pmax_seq)} frames)")
            max_len = len(pmax_seq)

        advance = False
        while not advance:
            for frame_i in range(max_len):
                pmax_frame = pmax_seq[min(frame_i, len(pmax_seq) - 1)]
                left_panel = draw_skeleton_frame(pmax_frame, f"pmax (MaxLSB) [{frame_i+1}/{len(pmax_seq)}]")

                if comparison_mode:
                    real_frame = real_seq[min(frame_i, len(real_seq) - 1)]
                    right_panel = draw_skeleton_frame(real_frame, f"{real_row['person_id']} (yours) [{frame_i+1}/{len(real_seq)}]")
                    combined = np.hstack([left_panel, right_panel])
                else:
                    combined = left_panel

                cv2.imshow("Skeleton Comparison (no video needed)", combined)
                key = cv2.waitKey(PLAYBACK_DELAY_MS) & 0xFF
                if key == ord('n'):
                    advance = True
                    break
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    return

        idx += 1


if __name__ == "__main__":
    main()