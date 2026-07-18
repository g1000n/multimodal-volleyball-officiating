"""
live_auto_inference.py

Continuous, hands-free gesture detection. Unlike live_interface.py
(manual countdown + fixed recording window), this watches motion
continuously and automatically detects when a gesture starts and ends
— no spacebar needed. This is the "idle gate + confirmation lock"
design from the original project plan: only start paying attention
when the referee's arms move meaningfully, and only finalize a
prediction once they've returned to rest for a sustained period.

HOW IT WORKS:
  - Tracks frame-to-frame motion in the pose+hand keypoints.
  - IDLE state: motion is low, nothing happening. A short rolling
    buffer of recent frames is kept so the very start of a gesture
    isn't missed once motion begins.
  - ACTIVE state: motion crossed the threshold — a gesture is likely
    happening. Frames are being recorded.
  - Once motion drops back down and STAYS low for IDLE_FRAMES_TO_CONFIRM
    frames in a row, the gesture is considered finished. The recorded
    sequence is run through the model, the prediction is printed, and
    it goes back to watching for the next gesture — fully automatic,
    runs continuously.

CALIBRATION — IMPORTANT, DO THIS FIRST:
  Run this once and just stand still (idle) for a few seconds, then do
  one clean gesture, then stand still again. Watch the "motion" number
  printed on-screen the whole time. Note:
    - the typical value while standing still (idle)
    - the peak value while actively gesturing
  Set MOTION_THRESHOLD to roughly halfway between those two numbers.
  The defaults below are a starting guess — they will likely need
  tuning for your specific camera/distance/zoom setup.

Controls:
  Q - quit
  S - toggle skeleton overlay on/off (pose + hand keypoints drawn live,
      useful for seeing exactly what MediaPipe is tracking at distance —
      if the skeleton looks jittery, missing, or offset from the real
      body, that's a direct sign detection is struggling before you
      even get to a prediction)
"""

import time
from collections import deque

import cv2
import json
import numpy as np
import torch
import mediapipe as mp

from extract_keypoints import (
    extract_pose_features,
    extract_hand_features,
    compute_elbow_angles,
)
from train import normalize_sequence, resample_sequence, SEQUENCE_LENGTH
from model import GestureCNNLSTM

MODEL_PATH = "models/final_model.pt"
LABEL_MAP_PATH = "models/label_map.json"
CAMERA_INDEX = 1  # set to whatever index your iPhone/Camo feed showed in list_cameras.py

# --- Tunable motion-detection parameters (calibrate these first!) ---
MOTION_THRESHOLD = 0.035        # motion score above this = "something is happening"
IDLE_FRAMES_TO_CONFIRM = 15     # consecutive low-motion frames needed to consider a gesture finished (~0.5s at 30fps)
MIN_ACTIVE_FRAMES = 12          # ignore tiny flickers shorter than this many frames
PRE_BUFFER_SIZE = 10            # keep this many recent idle frames so gesture start isn't clipped
MAX_ACTIVE_FRAMES = 300         # safety cap (~10s at 30fps) so a stuck "active" state can't buffer forever

# Which part of the 122-feature vector to use for motion scoring
# (pose + hand coords only — skip flags/fingers/elbow, which are more
# categorical and jump around even during small tracking noise)
MOTION_FEATURE_SLICE = slice(0, 108)

SHOW_SKELETON_DEFAULT = True  # starting state — press S anytime to toggle

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def load_model():
    with open(LABEL_MAP_PATH) as f:
        label_to_idx = json.load(f)
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    model = GestureCNNLSTM(input_size=122, num_classes=len(label_to_idx))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, idx_to_label


def predict_clip(frame_sequence, model, idx_to_label):
    raw = np.array(frame_sequence)
    normalized = normalize_sequence(raw)
    resampled = resample_sequence(normalized, SEQUENCE_LENGTH)
    x = torch.tensor(resampled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).numpy()[0]

    top_idx = int(np.argmax(probs))
    top_label = idx_to_label[top_idx]
    top_conf = probs[top_idx]

    print("\n" + "=" * 50)
    print(f"DETECTED GESTURE: {top_label}  ({top_conf:.1%} confidence)")
    for idx, label in sorted(idx_to_label.items()):
        print(f"  {label:<30} {probs[idx]:.1%}")
    print("=" * 50)
    return top_label, top_conf


def extract_frame_features(frame_rgb, pose_model, hands_model):
    pose_results = pose_model.process(frame_rgb)
    pose_features, pose_landmarks = extract_pose_features(pose_results)
    hand_coords, left_det, right_det, left_fingers, right_fingers = extract_hand_features(
        frame_rgb, pose_landmarks, hands_model
    )
    elbow_angles = compute_elbow_angles(pose_landmarks)

    features = np.concatenate([
        pose_features, hand_coords,
        np.array([left_det, right_det]),
        left_fingers, right_fingers, elbow_angles,
    ])

    # Extra info returned purely for skeleton drawing — not used in the
    # feature vector itself.
    draw_info = {
        "pose_results": pose_results,
        "hand_coords": hand_coords,   # 84 values: 21 left (x,y) + 21 right (x,y), full-frame normalized
        "left_detected": left_det,
        "right_detected": right_det,
    }

    return features, draw_info


def draw_skeleton_overlay(frame, draw_info):
    """
    Draws the pose skeleton (via MediaPipe's built-in connections) and
    hand keypoints (as colored dots — left hand cyan, right hand
    magenta) directly onto the frame. Purely visual, doesn't affect
    anything fed to the model. Useful for seeing live whether MediaPipe
    is actually tracking the referee correctly at distance, or silently
    failing/jittering.
    """
    frame_height, frame_width = frame.shape[:2]

    pose_results = draw_info["pose_results"]
    if pose_results.pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2),
        )

    hand_coords = draw_info["hand_coords"]
    left_points = hand_coords[:42].reshape(21, 2)
    right_points = hand_coords[42:].reshape(21, 2)

    if draw_info["left_detected"] > 0.5:
        for x, y in left_points:
            px, py = int(x * frame_width), int(y * frame_height)
            cv2.circle(frame, (px, py), 3, (255, 255, 0), -1)  # cyan = left hand

    if draw_info["right_detected"] > 0.5:
        for x, y in right_points:
            px, py = int(x * frame_width), int(y * frame_height)
            cv2.circle(frame, (px, py), 3, (255, 0, 255), -1)  # magenta = right hand


def main():
    model, idx_to_label = load_model()

    pose_model = mp_pose.Pose(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    hands_model = mp_hands.Hands(
        static_image_mode=True, max_num_hands=1,
        min_detection_confidence=0.28, min_tracking_confidence=0.28,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: could not open camera at index {CAMERA_INDEX}.")
        return

    STATE_IDLE = "idle"
    STATE_ACTIVE = "active"
    state = STATE_IDLE

    pre_buffer = deque(maxlen=PRE_BUFFER_SIZE)
    active_sequence = []
    idle_streak = 0
    prev_features = None
    last_result_text = ""
    last_result_time = 0
    show_skeleton = SHOW_SKELETON_DEFAULT

    print("Watching for motion... stand still first to see your idle baseline, then try a gesture.")
    print("Press Q to quit, S to toggle skeleton overlay.\n")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        features, draw_info = extract_frame_features(frame_rgb, pose_model, hands_model)

        if show_skeleton:
            draw_skeleton_overlay(frame, draw_info)

        if prev_features is None:
            motion_score = 0.0
        else:
            diff = features[MOTION_FEATURE_SLICE] - prev_features[MOTION_FEATURE_SLICE]
            motion_score = float(np.linalg.norm(diff))
        prev_features = features

        if state == STATE_IDLE:
            pre_buffer.append(features)
            if motion_score > MOTION_THRESHOLD:
                state = STATE_ACTIVE
                active_sequence = list(pre_buffer)
                idle_streak = 0

        elif state == STATE_ACTIVE:
            active_sequence.append(features)

            if motion_score <= MOTION_THRESHOLD:
                idle_streak += 1
            else:
                idle_streak = 0

            gesture_finished = idle_streak >= IDLE_FRAMES_TO_CONFIRM
            buffer_maxed_out = len(active_sequence) >= MAX_ACTIVE_FRAMES

            if gesture_finished or buffer_maxed_out:
                if len(active_sequence) >= MIN_ACTIVE_FRAMES:
                    label, conf = predict_clip(active_sequence, model, idx_to_label)
                    last_result_text = f"{label} ({conf:.0%})"
                    last_result_time = time.time()
                else:
                    print(f"(motion blip ignored — only {len(active_sequence)} frames)")

                state = STATE_IDLE
                active_sequence = []
                pre_buffer.clear()
                idle_streak = 0

        # --- on-screen overlay for calibration + live feedback ---
        state_color = (0, 0, 255) if state == STATE_ACTIVE else (0, 255, 0)
        cv2.putText(frame, f"STATE: {state.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
        cv2.putText(frame, f"motion: {motion_score:.4f}  (threshold: {MOTION_THRESHOLD})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if last_result_text and (time.time() - last_result_time) < 4:
            cv2.putText(frame, f"LAST: {last_result_text}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        skeleton_status = "ON" if show_skeleton else "OFF"
        cv2.putText(frame, f"skeleton: {skeleton_status} (press S)", (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Live Auto Gesture Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            show_skeleton = not show_skeleton

    cap.release()
    cv2.destroyAllWindows()
    pose_model.close()
    hands_model.close()


if __name__ == "__main__":
    main()