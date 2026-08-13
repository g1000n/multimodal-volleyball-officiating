"""
review_and_flag_pmax_clips.py

Skeleton-only review tool (no video needed) for pmax's (MaxLSB's)
converted keypoint clips -- lets you watch each one as an animated
stick figure and flag suspicious ones for removal, one class at a
time. Built specifically to check the hypothesis: do some of pmax's
team_to_serve_left/right clips have an arm angle low enough to look
ambiguous with ball_in (which is a lower, more downward-pointing
motion than team_to_serve)?

Flagged clips get written to flagged_pmax_clips.txt (one keypoint_path
per line) -- nothing is deleted automatically. Run
remove_flagged_pmax_clips.py afterward (or the equivalent manual
filter) once you've reviewed and are ready to actually exclude them.

HOW TO USE:
    python review_and_flag_pmax_clips.py

You'll get a numbered menu of pmax's classes to review. Then, one clip
at a time:
    K - keep (looks correct for this class)
    D - flag for removal (looks ambiguous / wrong, e.g. team_to_serve
        clip with an arm angle that looks more like ball_in)
    R - replay this clip
    B - go back to the previous clip
    Q - quit and save flagged list so far
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import os
import csv
import cv2
import numpy as np

MANIFEST_PATH = "data/dataset_manifest.csv"
FLAGGED_OUTPUT_PATH = "flagged_pmax_clips.txt"
CANVAS_SIZE = 550
PLAYBACK_DELAY_MS = 60

POSE_CONNECTIONS = [
    (0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7), (6, 7),
]


def draw_skeleton_frame(features_122, header_lines, canvas_size=CANVAS_SIZE):
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

    # Elbow angle readout -- directly relevant to the ball_in vs
    # team_to_serve angle hypothesis. Indices 120/121 in the raw
    # (unablated) saved keypoints.
    left_elbow_angle = features_122[120]
    right_elbow_angle = features_122[121]
    cv2.putText(canvas, f"L elbow angle: {left_elbow_angle:.2f}  R elbow angle: {right_elbow_angle:.2f}",
                (10, canvas_size - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    for i, line in enumerate(header_lines):
        cv2.putText(canvas, line, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas


def choose_gesture_class(rows):
    labels = sorted(set(r["gesture_label"] for r in rows if r["person_id"] == "pmax" and r.get("keypoint_path")))
    if not labels:
        print("No pmax classes found.")
        return None
    print("pmax (MaxLSB) classes available to review:")
    for i, label in enumerate(labels, start=1):
        count = sum(1 for r in rows if r["gesture_label"] == label and r["person_id"] == "pmax")
        print(f"  {i}. {label}  ({count} clips)")
    choice = input(f"\nPick a class (1-{len(labels)}): ").strip()
    try:
        return labels[int(choice) - 1]
    except (ValueError, IndexError):
        print("Invalid choice.")
        return None


def main():
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))

    gesture_label = choose_gesture_class(rows)
    if gesture_label is None:
        return

    clip_rows = [r for r in rows if r["gesture_label"] == gesture_label and r["person_id"] == "pmax" and r.get("keypoint_path")]
    print(f"\nReviewing {len(clip_rows)} pmax clips for '{gesture_label}'.")
    print("Controls: K=keep  D=flag for removal  R=replay  B=back  Q=quit+save\n")

    flagged = []
    idx = 0
    while 0 <= idx < len(clip_rows):
        row = clip_rows[idx]
        seq = np.load(row["keypoint_path"])
        clip_name = os.path.basename(row["keypoint_path"])

        decision = None
        while decision is None:
            for frame_i in range(len(seq)):
                header = [
                    f"[{idx+1}/{len(clip_rows)}] {clip_name}",
                    f"frame {frame_i+1}/{len(seq)}",
                ]
                frame_img = draw_skeleton_frame(seq[frame_i], header)
                cv2.imshow("Review pmax clips (K=keep D=flag R=replay B=back Q=quit)", frame_img)
                key = cv2.waitKey(PLAYBACK_DELAY_MS) & 0xFF
                if key == ord('k'):
                    decision = "keep"
                    break
                elif key == ord('d'):
                    decision = "flag"
                    break
                elif key == ord('r'):
                    decision = "replay"
                    break
                elif key == ord('b'):
                    decision = "back"
                    break
                elif key == ord('q'):
                    decision = "quit"
                    break
            if decision == "replay":
                decision = None  # loop again, play same clip

        cv2.destroyAllWindows()

        if decision == "flag":
            flagged.append(row["keypoint_path"])
            print(f"  FLAGGED: {clip_name}")
            idx += 1
        elif decision == "keep":
            idx += 1
        elif decision == "back":
            idx = max(0, idx - 1)
            if flagged and flagged[-1] == row["keypoint_path"]:
                flagged.pop()  # un-flag if we're going back over a flagged one
        elif decision == "quit":
            break

    if flagged:
        with open(FLAGGED_OUTPUT_PATH, "a") as f:
            for path in flagged:
                f.write(path + "\n")
        print(f"\n{len(flagged)} clips flagged this session, appended to {FLAGGED_OUTPUT_PATH}")
    else:
        print("\nNo clips flagged this session.")

    print(f"Total flagged so far (all sessions): see {FLAGGED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()