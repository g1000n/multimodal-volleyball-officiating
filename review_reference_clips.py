"""
review_reference_clips.py

Same side-by-side original + skeleton view as clip_reviewer.py, but for
your reference_clips/ (YouTube/broadcast footage) instead of your own
raw training clips. No flagging — these aren't part of your training
set, this is purely to visually see what the extraction pipeline
"sees" on external footage, to understand why detection rates are
lower there.

CONTROLS:
  SPACE - pause / resume
  R     - restart current clip
  N     - next clip
  B     - previous clip
  Q     - quit

Run from your project root:
    python review_reference_clips.py
"""

import os
import cv2
import numpy as np
import mediapipe as mp

from extract_keypoints import extract_pose_features, extract_hand_features

REFERENCE_DIR = "data/reference_clips"
DISPLAY_HEIGHT = 480

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def collect_reference_clips():
    clips = []
    for label in sorted(os.listdir(REFERENCE_DIR)):
        folder_path = os.path.join(REFERENCE_DIR, label)
        if not os.path.isdir(folder_path):
            continue
        for filename in sorted(os.listdir(folder_path)):
            clips.append({"path": os.path.join(folder_path, filename), "label": label, "filename": filename})
    return clips


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

    return skeleton_frame


def resize_to_height(frame, target_height):
    h, w = frame.shape[:2]
    scale = target_height / h
    return cv2.resize(frame, (int(w * scale), target_height))


def main():
    clips = collect_reference_clips()
    if not clips:
        print(f"No clips found in {REFERENCE_DIR}.")
        return

    pose_model = mp_pose.Pose(
        static_image_mode=False, model_complexity=0,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    hands_model = mp_hands.Hands(
        static_image_mode=True, max_num_hands=1,
        min_detection_confidence=0.28, min_tracking_confidence=0.28,
    )

    index = 0
    print(f"{len(clips)} reference clips loaded.")
    print("SPACE=pause/resume  R=restart  N=next  B=back  Q=quit\n")

    while 0 <= index < len(clips):
        clip = clips[index]
        cap = cv2.VideoCapture(clip["path"])
        if not cap.isOpened():
            print(f"Could not open {clip['path']}, skipping.")
            index += 1
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        delay_ms = max(1, int(1000 / fps))
        paused = False
        advance_action = None

        while advance_action is None:
            if not paused:
                success, frame = cap.read()
                if not success:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                skeleton_frame = draw_skeleton_frame(frame, pose_model, hands_model)
                left_panel = resize_to_height(frame, DISPLAY_HEIGHT)
                right_panel = resize_to_height(skeleton_frame, DISPLAY_HEIGHT)
                combined = np.hstack([left_panel, right_panel])

                cv2.putText(combined, "ORIGINAL", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(combined, "SKELETON", (left_panel.shape[1] + 10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                info_text = f"[{index+1}/{len(clips)}] label: {clip['label']}"
                cv2.putText(combined, info_text, (10, DISPLAY_HEIGHT - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(combined, clip["filename"], (10, DISPLAY_HEIGHT - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                pause_text = "PAUSED" if paused else "PLAYING (loops)"
                pause_color = (0, 0, 255) if paused else (0, 255, 0)
                cv2.putText(combined, pause_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pause_color, 2)

                controls = "SPACE=pause  R=restart  N=next  B=back  Q=quit"
                cv2.putText(combined, controls, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("Reference Clip Viewer", combined)

            key = cv2.waitKey(delay_ms if not paused else 30) & 0xFF

            if key == ord(' '):
                paused = not paused
            elif key == ord('r'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            elif key == ord('n'):
                advance_action = "next"
            elif key == ord('b'):
                advance_action = "prev"
            elif key == ord('q'):
                advance_action = "quit"

        cap.release()

        if advance_action == "quit":
            break
        elif advance_action == "prev":
            index = max(0, index - 1)
        else:
            index += 1

    cv2.destroyAllWindows()
    pose_model.close()
    hands_model.close()


if __name__ == "__main__":
    main()