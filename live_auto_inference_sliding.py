"""
live_auto_inference_sliding.py

EXPERIMENTAL: sliding-window + majority-vote live detection, adapted
from the mechanism in MaxLSB/volley-judge's main.py. Runs on your
EXISTING trained checkpoint -- no retraining, no data changes. This is
a different SEGMENTATION strategy layered on top of the same model.

WHY THIS EXISTS:
Your original live_auto_inference.py waits for motion to drop and STAY
low (idle gate) before finalizing one prediction -- this assumes a
real rest/pause between gestures. If your referees flow directly from
one gesture into the next with no pause, that idle-gate design merges
both gestures into one garbled capture.

This version never waits for idle at all. It keeps a rolling buffer of
the most recent RAW frames, and on every frame (well, every
INFERENCE_EVERY_N_FRAMES frames, for performance) it resamples that
buffer to SEQUENCE_LENGTH and runs the model -- continuously, all the
time, transitions included. A single window's prediction is noisy
(especially mid-transition), so instead of trusting one prediction, it
keeps a rolling vote over the last VOTE_WINDOW_SIZE predictions and
only "commits" a detected gesture once a class wins by
VOTES_NEEDED_TO_COMMIT votes. Once committed, it's handed to
decision_engine.py immediately, and a dedup check (like volley-judge's
`actions[predicted_label] != sentence[-1]`) prevents re-firing the
same gesture repeatedly while it's still being held.

IMPORTANT CAVEAT (read before trusting this blindly):
Your model was trained ONLY on clean, isolated, single-gesture clips.
It has never seen a genuinely idle/transitional window and has no
"Nothing" class to fall back on (unlike volley-judge, which explicitly
trained one). That means during transitions or idle moments, the model
WILL still confidently output some real class -- it has no way to say
"nothing is happening right now." The CONFIDENCE_THRESHOLD below is
your only defense against that: it discards low-confidence per-window
predictions before they even enter the vote, so hopefully only
genuinely-held gestures (which the model is actually good at) win
votes, while noisy transitional windows get filtered out for being
low-confidence. This is a real experiment, not a guaranteed fix --
if it produces too many false detections during transitions/idle
moments, that's your signal the "Nothing" class + retrain route
(see accompanying instructions) is the one to actually do.

Controls:
  Q - quit
  S - toggle skeleton overlay
  W - TEMPORARY: manually simulate a whistle detection
"""

import time
from collections import deque, Counter

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
from train import (
    normalize_sequence,
    resample_sequence,
    SEQUENCE_LENGTH,
    TOTAL_FEATURES,
    ablate,
    ABLATE_HAND_COORDS,
    ABLATED_FEATURE_COUNT,
)
from model import GestureCNNLSTM
from decision_engine import DecisionEngine
from scoreboard_gui import ScoreboardGUI

MODEL_PATH = "models/final_model.pt"
LABEL_MAP_PATH = "models/label_map.json"
CAMERA_INDEX = 1

# --- Sliding window + majority vote parameters (tune these) ---
ROLLING_WINDOW_FRAMES = 60      # ~2s at 30fps -- covers one full gesture; adjust to your typical gesture duration
INFERENCE_EVERY_N_FRAMES = 3    # run inference every Nth frame, not every single frame (perf -- CNN-LSTM forward pass isn't free)
CONFIDENCE_THRESHOLD = 0.80     # RAISED from 0.60 -- idle standing falsely committed as a real gesture at 0.60, since the model has no "Nothing" class and always outputs some real class. Higher threshold = fewer false idle commits, but also slower/stricter to commit genuine gestures.
VOTE_WINDOW_SIZE = 10
VOTES_NEEDED_TO_COMMIT = 9       # RAISED from 7 -- requires a much stronger, more sustained majority before committing anything
COMMIT_COOLDOWN_SECONDS = 1.0   # minimum time between committing the SAME label twice in a row (prevents rapid re-firing while a gesture is held)

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def load_model():
    with open(LABEL_MAP_PATH) as f:
        label_to_idx = json.load(f)
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    input_size = ABLATED_FEATURE_COUNT if ABLATE_HAND_COORDS else TOTAL_FEATURES
    model = GestureCNNLSTM(input_size=input_size, num_classes=len(label_to_idx))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, idx_to_label


def classify_window(raw_window_frames, model, idx_to_label):
    """Resamples the current rolling window to SEQUENCE_LENGTH and runs
    one forward pass. Returns (label, confidence) for THIS window only
    -- caller decides whether to trust it via the vote mechanism."""
    raw = np.array(raw_window_frames)
    normalized = normalize_sequence(raw)
    normalized = ablate(normalized)
    resampled = resample_sequence(normalized, SEQUENCE_LENGTH)
    x = torch.tensor(resampled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).numpy()[0]

    top_idx = int(np.argmax(probs))
    return idx_to_label[top_idx], float(probs[top_idx])


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

    draw_info = {
        "pose_results": pose_results,
        "hand_coords": hand_coords,
        "left_detected": left_det,
        "right_detected": right_det,
    }
    return features, draw_info


def draw_skeleton_overlay(frame, draw_info):
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
            cv2.circle(frame, (int(x * frame_width), int(y * frame_height)), 3, (255, 255, 0), -1)
    if draw_info["right_detected"] > 0.5:
        for x, y in right_points:
            cv2.circle(frame, (int(x * frame_width), int(y * frame_height)), 3, (255, 0, 255), -1)


def main():
    model, idx_to_label = load_model()
    engine = DecisionEngine()
    gui = ScoreboardGUI()

    pose_model = mp_pose.Pose(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    hands_model = mp_hands.Hands(
        static_image_mode=True, max_num_hands=2,
        min_detection_confidence=0.1, min_tracking_confidence=0.28,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: could not open camera at index {CAMERA_INDEX}.")
        return

    rolling_window = deque(maxlen=ROLLING_WINDOW_FRAMES)
    recent_confident_predictions = deque(maxlen=VOTE_WINDOW_SIZE)
    last_committed_label = None
    last_commit_time = 0
    frame_counter = 0
    show_skeleton = True
    last_result_text = ""
    last_result_time = 0
    whistle_flash_until = 0

    print("Sliding-window mode: continuously classifying, no idle-gate.")
    print(f"Committing a label needs {VOTES_NEEDED_TO_COMMIT}/{VOTE_WINDOW_SIZE} recent confident "
          f"(>{CONFIDENCE_THRESHOLD:.0%}) predictions to agree.")
    print("Press Q to quit, S to toggle skeleton, W to simulate a whistle.\n")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        features, draw_info = extract_frame_features(frame_rgb, pose_model, hands_model)
        if show_skeleton:
            draw_skeleton_overlay(frame, draw_info)

        rolling_window.append(features)
        frame_counter += 1

        current_window_label = None
        current_window_conf = 0.0

        if len(rolling_window) == ROLLING_WINDOW_FRAMES and frame_counter % INFERENCE_EVERY_N_FRAMES == 0:
            current_window_label, current_window_conf = classify_window(
                list(rolling_window), model, idx_to_label
            )

            if current_window_conf >= CONFIDENCE_THRESHOLD:
                recent_confident_predictions.append(current_window_label)

                # Majority vote over recent CONFIDENT predictions only --
                # low-confidence windows (likely transitions/idle) never
                # even entered this deque, so they can't win a vote.
                if len(recent_confident_predictions) == VOTE_WINDOW_SIZE:
                    vote_counts = Counter(recent_confident_predictions)
                    top_label, top_count = vote_counts.most_common(1)[0]

                    if top_count >= VOTES_NEEDED_TO_COMMIT:
                        now = time.time()
                        is_new_gesture = (top_label != last_committed_label)
                        cooldown_passed = (now - last_commit_time) >= COMMIT_COOLDOWN_SECONDS

                        if is_new_gesture or cooldown_passed:
                            print(f"\nCOMMITTED: {top_label}  (vote: {top_count}/{VOTE_WINDOW_SIZE})")
                            result = engine.on_gesture_detected(top_label, now)
                            print(f"  -> decision_engine: {result}")

                            if result["event"] == "point_awarded":
                                gui.update_display(
                                    left_score=engine.score["left"],
                                    right_score=engine.score["right"],
                                    gesture_text=top_label,
                                    whistle_status="waiting for next whistle...",
                                )
                            elif result["event"] == "reason_attached":
                                gui.update_display(gesture_text=f"{top_label} ({result['side']})")
                            elif result["event"] == "ignored":
                                gui.update_display(gesture_text=f"{top_label} (ignored: {result['reason']})")

                            last_committed_label = top_label
                            last_commit_time = now
                            last_result_text = f"{top_label} ({top_count}/{VOTE_WINDOW_SIZE})"
                            last_result_time = now
                            # Clear the vote buffer after committing so the
                            # NEXT gesture has to build its own fresh
                            # majority, rather than the old one lingering
                            # and delaying detection of a new gesture.
                            recent_confident_predictions.clear()

        gui.tick()

        # --- on-screen live feedback ---
        cv2.putText(frame, "SLIDING WINDOW MODE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        if current_window_label is not None:
            cv2.putText(frame, f"window: {current_window_label} ({current_window_conf:.0%})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        vote_str = ", ".join(f"{k}:{v}" for k, v in Counter(recent_confident_predictions).items())
        cv2.putText(frame, f"votes: [{vote_str}]", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if last_result_text and (time.time() - last_result_time) < 4:
            cv2.putText(frame, f"LAST: {last_result_text}", (10, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if time.time() < whistle_flash_until:
            cv2.putText(frame, "WHISTLE (manual)", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.imshow("Live Sliding-Window Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            show_skeleton = not show_skeleton
        elif key == ord('w'):
            engine.on_whistle_detected(time.time())
            gui.update_display(whistle_status="detected (manual)")
            whistle_flash_until = time.time() + 1.0
            print("  -> manual whistle simulated")

    cap.release()
    cv2.destroyAllWindows()
    gui.close()
    pose_model.close()
    hands_model.close()


if __name__ == "__main__":
    main()