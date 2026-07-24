"""
view_reference_clip_skeleton.py

Standalone skeleton viewer for SPECIFIC reference clip files (not
manifest-based like skeleton_check.py or clip_reviewer.py -- those only
know about data/raw_clips/ via dataset_manifest.csv, and reference
clips in data/reference_clips/ were never added there).

Built specifically to check one hypothesis: are the two failing
team_to_serve_right reference clips actually being mislabeled by
MediaPipe itself -- i.e. does the cyan (left) dot cluster land on the
referee's true left hand, or is MediaPipe internally flipping its own
left/right assignment on this footage? If the overlay's colors don't
match the anatomically correct side for the direction the referee is
facing, that confirms it's a MediaPipe-level labeling issue on this
footage, not a problem with your training data or model.

IMPORTANT: this checks BOTH hand-level AND pose-level left/right
assignment. The current model has ABLATE_HAND_COORDS=True, meaning it
never sees raw hand coordinates at all -- what it actually uses to
distinguish team_to_serve_left/right is POSE landmarks (shoulders,
elbows, wrists via mp_pose) and elbow angles, NOT the hand-tracking
model's left/right labeling. A generic green pose skeleton (MediaPipe's
default drawing style) doesn't visually distinguish left from right --
so this version explicitly color-codes LEFT_SHOULDER/ELBOW/WRIST and
RIGHT_SHOULDER/ELBOW/WRIST separately, in the same cyan/magenta
convention as the hand dots, so you can check the side that actually
matters for this model.

Controls:
  SPACE - pause / resume
  N     - next clip in the list
  B     - previous clip in the list
  R     - restart current clip
  Q     - quit

Run from your project root:
    python view_reference_clip_skeleton.py
"""

import os
import cv2
import numpy as np
import mediapipe as mp

from extract_keypoints import extract_pose_features, extract_hand_features, debug_get_crop_info

# Edit this list to point at whichever reference clips you want to check.
# Defaults to the two failing team_to_serve_right clips from the latest run.
CLIP_PATHS = [
    "data/reference_clips/team_to_serve_right/team_to_serve_right_ref_youtube_01.mp4",
    "data/reference_clips/team_to_serve_right/team_to_serve_right_ref_youtube_02.mp4",
]

DISPLAY_HEIGHT = 600
WINDOW_NAME = "Reference Clip Skeleton Check"

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def draw_skeleton_frame(frame_bgr, pose_model, hands_model):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pose_results = pose_model.process(frame_rgb)
    pose_features, pose_landmarks = extract_pose_features(pose_results)
    hand_coords, left_det, right_det, _, _ = extract_hand_features(frame_rgb, pose_landmarks, hands_model)

    skeleton_frame = frame_bgr.copy()
    h, w = skeleton_frame.shape[:2]

    if pose_results.pose_landmarks is not None:
        # Generic skeleton first (context/reference), then explicit
        # left/right color-coded joints drawn on top -- THIS is the part
        # that actually matters for the ablated model, since it uses
        # mp_pose's own LEFT_*/RIGHT_* landmark assignment directly.
        mp_drawing.draw_landmarks(
            skeleton_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0, 150, 0), thickness=1),
        )

        landmarks = pose_results.pose_landmarks.landmark
        left_joint_ids = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST]
        right_joint_ids = [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST]
        joint_labels = ["shoulder", "elbow", "wrist"]

        for joint_id, label in zip(left_joint_ids, joint_labels):
            lm = landmarks[joint_id]
            if lm.visibility > 0.3:
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(skeleton_frame, (px, py), 8, (255, 255, 0), -1)  # cyan = POSE left
                cv2.putText(skeleton_frame, f"L-{label}", (px + 10, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        for joint_id, label in zip(right_joint_ids, joint_labels):
            lm = landmarks[joint_id]
            if lm.visibility > 0.3:
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(skeleton_frame, (px, py), 8, (255, 0, 255), -1)  # magenta = POSE right
                cv2.putText(skeleton_frame, f"R-{label}", (px + 10, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

    left_points = hand_coords[:42].reshape(21, 2)
    right_points = hand_coords[42:].reshape(21, 2)

    # Smaller dots for hand-level left/right -- kept for reference, but
    # NOT what this specific model actually relies on (see docstring).
    if left_det > 0.5:
        for x, y in left_points:
            cv2.circle(skeleton_frame, (int(x * w), int(y * h)), 2, (150, 150, 0), -1)
    if right_det > 0.5:
        for x, y in right_points:
            cv2.circle(skeleton_frame, (int(x * w), int(y * h)), 2, (150, 0, 150), -1)

    cv2.putText(skeleton_frame, "BIG dots = POSE L/R (what the model actually uses)", (10, h - 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(skeleton_frame, "CYAN = left   MAGENTA = right", (10, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(skeleton_frame, "(small dots = hand-tracking L/R, not used by current model)", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return skeleton_frame


def resize_to_height(frame, target_height):
    h, w = frame.shape[:2]
    scale = target_height / h
    return cv2.resize(frame, (int(w * scale), target_height))


def main():
    valid_paths = [p for p in CLIP_PATHS if os.path.exists(p)]
    missing = [p for p in CLIP_PATHS if p not in valid_paths]
    if missing:
        print("WARNING: these paths don't exist, check CLIP_PATHS at the top of this file:")
        for p in missing:
            print(f"  - {p}")
    if not valid_paths:
        print("No valid clip paths found. Nothing to show.")
        return

    pose_model = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands_model = mp_hands.Hands(static_image_mode=True, max_num_hands=2,
                                  min_detection_confidence=0.1, min_tracking_confidence=0.28)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    index = 0
    quit_all = False
    while 0 <= index < len(valid_paths) and not quit_all:
        clip_path = valid_paths[index]
        print(f"\n[{index+1}/{len(valid_paths)}] {clip_path}")

        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            print(f"  Could not open {clip_path} -- skipping.")
            index += 1
            continue

        paused = False
        advance_action = None
        read_failures = 0

        while advance_action is None:
            if not paused:
                success, frame = cap.read()
                if not success:
                    read_failures += 1
                    if read_failures >= 30:
                        print("  Won't decode -- skipping.")
                        advance_action = "next"
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                read_failures = 0

                skeleton_frame = draw_skeleton_frame(frame, pose_model, hands_model)
                display_frame = resize_to_height(skeleton_frame, DISPLAY_HEIGHT)

                cv2.putText(display_frame, os.path.basename(clip_path), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display_frame, f"[{index+1}/{len(valid_paths)}]  SPACE=pause N=next B=back R=restart Q=quit",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(33) & 0xFF
            if key == ord(' '):
                paused = not paused
            elif key == ord('n'):
                advance_action = "next"
            elif key == ord('b'):
                advance_action = "prev"
            elif key == ord('r'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                read_failures = 0
            elif key == ord('q'):
                advance_action = "quit"
                quit_all = True

        cap.release()

        if advance_action == "prev":
            index = max(0, index - 1)
        else:
            index += 1

    cv2.destroyAllWindows()
    pose_model.close()
    hands_model.close()


if __name__ == "__main__":
    main()