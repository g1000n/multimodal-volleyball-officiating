"""
live_inference.py

Real-time webcam test for the trained gesture model. Reuses the exact
same feature-extraction functions as extract_keypoints.py and the exact
same normalization/resampling logic as train.py, so what you test here
matches what the model actually learned from.

Controls:
  SPACE - triggers a 3-second "get ready" countdown, then auto-records
          for RECORD_DURATION_SECONDS and predicts automatically
  Q     - quit

Stand at neutral before pressing SPACE. During the countdown, get into
position. Recording starts right after the countdown — begin your
gesture from neutral, perform it, and return to neutral before the
timer runs out.

Run from your project root (same folder as extract_keypoints.py, train.py, model.py):
    python live_inference.py

NOTE on camera source: cv2.VideoCapture(0) uses your default webcam. If
you want to test using an iPhone camera instead (e.g. to approximate your
real deployment distance/zoom), you'll need either:
  - iPhone connected via a capture card / HDMI-to-USB adapter (shows up
    as its own camera index — try 1, 2, etc. if 0 doesn't work), or
  - A continuity-camera-style app that makes the iPhone appear as a
    webcam device to Windows (search "iPhone as webcam Windows" for
    current app options — this changes over time, so no single fixed
    recommendation here).
Once connected, just change CAMERA_INDEX below to match.
"""

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
CAMERA_INDEX = 1  # change if using an external/iPhone camera source .. changed to index 1 (iphone 14's index)

COUNTDOWN_SECONDS = 3       # time to get into neutral position before recording starts
RECORD_DURATION_SECONDS = 7  # how long it records after the countdown

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


def load_model():
    with open(LABEL_MAP_PATH) as f:
        label_to_idx = json.load(f)
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    num_classes = len(label_to_idx)
    input_size = 122  # must match TOTAL_FEATURES in train.py

    model = GestureCNNLSTM(input_size=input_size, num_classes=num_classes)
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
    print(f"PREDICTED: {top_label}  ({top_conf:.1%} confidence)")
    print("Full breakdown:")
    for idx, label in sorted(idx_to_label.items()):
        print(f"  {label:<30} {probs[idx]:.1%}")
    print("=" * 50)


def main():
    model, idx_to_label = load_model()

    pose_model = mp_pose.Pose(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    hands_model = mp_hands.Hands(
        static_image_mode=True, max_num_hands=1,
        min_detection_confidence=0.4, min_tracking_confidence=0.4,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: could not open camera at index {CAMERA_INDEX}.")
        return

    import time

    STATE_IDLE = "idle"
    STATE_COUNTDOWN = "countdown"
    STATE_RECORDING = "recording"

    state = STATE_IDLE
    frame_sequence = []
    state_start_time = None

    print(f"Press SPACE: {COUNTDOWN_SECONDS}s to get ready, then {RECORD_DURATION_SECONDS}s auto-recording. Q to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose_model.process(frame_rgb)
        pose_features, pose_landmarks = extract_pose_features(pose_results)
        hand_coords, left_det, right_det, left_fingers, right_fingers = extract_hand_features(
            frame_rgb, pose_landmarks, hands_model
        )
        elbow_angles = compute_elbow_angles(pose_landmarks)

        frame_features = np.concatenate([
            pose_features, hand_coords,
            np.array([left_det, right_det]),
            left_fingers, right_fingers, elbow_angles,
        ])

        now = time.time()

        if state == STATE_COUNTDOWN:
            elapsed = now - state_start_time
            remaining = COUNTDOWN_SECONDS - elapsed
            if remaining <= 0:
                state = STATE_RECORDING
                state_start_time = now
                frame_sequence = []
                print("\nRecording...")
            else:
                cv2.putText(frame, f"GET READY: {remaining:.1f}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        elif state == STATE_RECORDING:
            elapsed = now - state_start_time
            remaining = RECORD_DURATION_SECONDS - elapsed
            frame_sequence.append(frame_features)
            if remaining <= 0:
                state = STATE_IDLE
                print(f"Recording stopped. {len(frame_sequence)} frames captured. Predicting...")
                if len(frame_sequence) > 0:
                    predict_clip(frame_sequence, model, idx_to_label)
                else:
                    print("No frames captured, skipping prediction.")
            else:
                cv2.putText(frame, f"RECORDING: {remaining:.1f}s left ({len(frame_sequence)} frames)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        else:  # STATE_IDLE
            cv2.putText(frame, "Press SPACE to start", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Live Gesture Test", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and state == STATE_IDLE:
            state = STATE_COUNTDOWN
            state_start_time = time.time()

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose_model.close()
    hands_model.close()


if __name__ == "__main__":
    main()