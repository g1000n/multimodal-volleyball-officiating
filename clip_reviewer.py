"""
clip_reviewer.py

Interactive review tool for your raw training clips. Plays each clip
on loop with the original footage on the left and the live MediaPipe
skeleton overlay on the right, so you can see exactly what the
extraction pipeline "sees" for that clip while deciding whether it's
good, mislabeled, or sloppy.

Progress is saved automatically after every decision — quit anytime
(Q) and it resumes exactly where you left off next time you run it.

Flagging a clip here does NOT delete or move anything. It just marks
it. Run apply_flagged_removals.py afterward to actually move flagged
clips out of the dataset (into a quarantine folder, not permanently
deleted).

CONTROLS (while a clip is playing):
  SPACE - pause / resume playback
  D     - flag this clip for removal, move to next clip
  K     - keep this clip (mark reviewed, not flagged), move to next
  N     - skip without deciding (move to next, leave unreviewed)
  B     - go back to the previous clip
  R     - restart current clip from the beginning
  Q     - quit and save progress

Run from your project root:
    python clip_reviewer.py
"""

import os
import csv
import json
import cv2
import numpy as np
import mediapipe as mp

from extract_keypoints import extract_pose_features, extract_hand_features, debug_get_crop_info

MANIFEST_PATH = "data/dataset_manifest.csv"
PROGRESS_PATH = "data/clip_review_progress.json"
DISPLAY_HEIGHT = 480  # each side panel resized to this height for consistent display

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def load_clip_list():
    with open(MANIFEST_PATH, "r") as f:
        rows = list(csv.DictReader(f))
    return rows


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r") as f:
            return json.load(f)
    return {"reviewed": {}, "last_index": 0}


def save_progress(progress):
    os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


def draw_skeleton_frame(frame_bgr, pose_model, hands_model):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pose_results = pose_model.process(frame_rgb)
    pose_features, pose_landmarks = extract_pose_features(pose_results)
    hand_coords, left_det, right_det, _, _ = extract_hand_features(frame_rgb, pose_landmarks, hands_model)

    skeleton_frame = frame_bgr.copy()
    frame_height, frame_width = skeleton_frame.shape[:2]

    if pose_results.pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            skeleton_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2),
        )

    left_points = hand_coords[:42].reshape(21, 2)
    right_points = hand_coords[42:].reshape(21, 2)

    if left_det > 0.5:
        for x, y in left_points:
            px, py = int(x * frame_width), int(y * frame_height)
            cv2.circle(skeleton_frame, (px, py), 3, (255, 255, 0), -1)

    if right_det > 0.5:
        for x, y in right_points:
            px, py = int(x * frame_width), int(y * frame_height)
            cv2.circle(skeleton_frame, (px, py), 3, (255, 0, 255), -1)

    # DEBUG: draw the actual crop box(es) used and which strategy was chosen
    mode, boxes = debug_get_crop_info(pose_landmarks, frame_width, frame_height)
    box_color = (0, 165, 255) if mode == "combined" else (255, 100, 0)
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(skeleton_frame, (x1, y1), (x2, y2), box_color, 2)
    cv2.putText(skeleton_frame, f"crop mode: {mode}", (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    return skeleton_frame


def resize_to_height(frame, target_height):
    h, w = frame.shape[:2]
    scale = target_height / h
    return cv2.resize(frame, (int(w * scale), target_height))


def choose_gesture_filter(rows):
    labels = sorted(set(r["gesture_label"] for r in rows))
    counts = {label: sum(1 for r in rows if r["gesture_label"] == label) for label in labels}

    menu_options = labels + ["all"]

    print("Available gesture classes:")
    for i, label in enumerate(menu_options, start=1):
        if label == "all":
            print(f"  {i}. all  (review everything)")
        else:
            print(f"  {i}. {label}  ({counts[label]} clips)")

    choice = input(f"\nEnter a number (1-{len(menu_options)}): ").strip()

    try:
        choice_num = int(choice)
        selected = menu_options[choice_num - 1]
    except (ValueError, IndexError):
        print(f"'{choice}' not a valid option, defaulting to 'all'.")
        selected = "all"

    if selected == "all":
        return rows, "ALL"

    filtered = [r for r in rows if r["gesture_label"] == selected]
    return filtered, selected


def main():
    rows = load_clip_list()
    if not rows:
        print("Manifest is empty or missing. Run build_manifest.py first.")
        return

    rows, filter_key = choose_gesture_filter(rows)

    progress = load_progress()
    if "last_index_by_filter" not in progress:
        progress["last_index_by_filter"] = {}

    index = progress["last_index_by_filter"].get(filter_key, 0)
    index = max(0, min(index, len(rows) - 1))

    pose_model = mp_pose.Pose(
        static_image_mode=False, model_complexity=0,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    hands_model = mp_hands.Hands(
        static_image_mode=True, max_num_hands=1,
        min_detection_confidence=0.28, min_tracking_confidence=0.28,
    )

    print(f"Resuming at clip {index + 1}/{len(rows)}.")
    print("SPACE=pause/resume  D=flag for removal  K=keep  N=skip  B=back  R=restart  Q=quit\n")

    while 0 <= index < len(rows):
        row = rows[index]
        clip_path = row["clip_path"]
        gesture_label = row["gesture_label"]
        person_id = row["person_id"]

        status = progress["reviewed"].get(clip_path, "unreviewed")

        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            print(f"Could not open {clip_path}, skipping.")
            index += 1
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        delay_ms = max(1, int(1000 / fps))

        paused = False
        advance_action = None  # set to "next", "prev", "flag", "keep", "skip" to break the playback loop

        while advance_action is None:
            if not paused:
                success, frame = cap.read()
                if not success:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop back to start
                    continue

                skeleton_frame = draw_skeleton_frame(frame, pose_model, hands_model)

                left_panel = resize_to_height(frame, DISPLAY_HEIGHT)
                right_panel = resize_to_height(skeleton_frame, DISPLAY_HEIGHT)
                combined = np.hstack([left_panel, right_panel])

                cv2.putText(combined, "ORIGINAL", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(combined, "SKELETON", (left_panel.shape[1] + 10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                info_text = f"[{index+1}/{len(rows)}] {gesture_label} | {person_id} | status: {status}"
                cv2.putText(combined, info_text, (10, DISPLAY_HEIGHT - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                filename_text = os.path.basename(clip_path)
                cv2.putText(combined, filename_text, (10, DISPLAY_HEIGHT - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                pause_text = "PAUSED" if paused else "PLAYING (loops automatically)"
                pause_color = (0, 0, 255) if paused else (0, 255, 0)
                cv2.putText(combined, pause_text, (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, pause_color, 2)

                controls_lines = [
                    "SPACE = pause/resume    R = restart clip",
                    "K = keep clip           D = flag for removal",
                    "N = skip (undecided)    B = back to previous clip",
                    "Q = quit and save progress",
                ]
                for i, line in enumerate(controls_lines):
                    cv2.putText(combined, line, (10, 85 + i * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("Clip Reviewer", combined)

            key = cv2.waitKey(delay_ms if not paused else 30) & 0xFF

            if key == ord(' '):
                paused = not paused
            elif key == ord('d'):
                progress["reviewed"][clip_path] = "flagged"
                advance_action = "next"
            elif key == ord('k'):
                progress["reviewed"][clip_path] = "kept"
                advance_action = "next"
            elif key == ord('n'):
                advance_action = "next"
            elif key == ord('b'):
                advance_action = "prev"
            elif key == ord('r'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            elif key == ord('q'):
                advance_action = "quit"

        cap.release()

        if advance_action == "quit":
            progress["last_index_by_filter"][filter_key] = index
            save_progress(progress)
            print(f"Progress saved at clip {index + 1}/{len(rows)} (filter: {filter_key}). Run again to resume.")
            break
        elif advance_action == "prev":
            index = max(0, index - 1)
            progress["last_index_by_filter"][filter_key] = index
            save_progress(progress)
        else:  # "next"
            index += 1
            progress["last_index_by_filter"][filter_key] = index
            save_progress(progress)

    cv2.destroyAllWindows()
    pose_model.close()
    hands_model.close()

    if index >= len(rows):
        flagged_count = sum(1 for v in progress["reviewed"].values() if v == "flagged")
        kept_count = sum(1 for v in progress["reviewed"].values() if v == "kept")
        print(f"\nAll clips reviewed. {kept_count} kept, {flagged_count} flagged for removal.")
        print("Run apply_flagged_removals.py to move flagged clips out of the dataset.")


if __name__ == "__main__":
    main()