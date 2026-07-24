"""
live_auto_inference_sliding.py (MULTI-LABEL, GUIDED SELF-TEST VERSION)

Sliding-window + majority-vote live detection, multi-label sigmoid
architecture, now with a GUIDED ON-SCREEN TEST SEQUENCE so you can run
this alone -- no need to watch the console while performing gestures.
Everything you need is drawn on the camera window itself:
  - A big instruction banner telling you exactly what to do right now
    and a countdown of how long you have.
  - A live probability bar for every real class (style borrowed from
    MaxLSB's prob_viz()), so you can see confidence at a glance.
  - A running "gesture history" bar at the bottom (style borrowed from
    his backend.py's ' -> '.join(sentence) display), showing the last
    few COMMITTED gestures in order -- this is what tells you whether
    back-to-back sequences actually got split into two clean events.

TEST SEQUENCE (edit TEST_PLAN below to change it):
  Phase 1 (idle):        stand still -- checks for false positives
  Phase 2 (isolated):     do ONE gesture, then rest -- checks basic
                          detection with a pause, cycles through a
                          few different gestures
  Phase 3 (transition):  do TWO gestures back-to-back, NO PAUSE --
                          the actual back-to-back test

Each active phase gets a 3-2-1 "get ready" countdown first, mirroring
MaxLSB's data_collection()'s "STARTING COLLECTION" pause, so you have
a moment to get into position before it starts watching.

Controls:
  Q - quit
  S - toggle skeleton overlay
  W - TEMPORARY: manually simulate a whistle detection
  N - skip to the next phase early (if a phase finishes before its
      timer, or you want to move on)
"""

import time
import csv
import os
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
    DECISION_THRESHOLD,
    NOTHING_LABEL,
    decide_label,
    apply_tie_breaker,
)
from model import GestureCNNLSTM
from decision_engine import DecisionEngine
from scoreboard_gui import ScoreboardGUI

MODEL_PATH = "models/final_model.pt"
LABEL_MAP_PATH = "models/label_map.json"
CAMERA_INDEX = 1
LOG_DIR = "data/live_test_logs"

# --- Sliding window parameters (unchanged) ---
ROLLING_WINDOW_FRAMES = 60
INFERENCE_EVERY_N_FRAMES = 3
VOTE_WINDOW_SIZE = 10
VOTES_NEEDED_TO_COMMIT = 7
COMMIT_COOLDOWN_SECONDS = 1.0

# --- Guided test plan -- EDIT THIS to change the sequence ---
GET_READY_SECONDS = 4

TEST_PLAN = [
    {"type": "idle", "label": "STAND STILL / IDLE", "duration": 15},
    {"type": "isolated", "label": "team_to_serve_right", "duration": 12},
    {"type": "idle", "label": "REST / PAUSE", "duration": 5},
    {"type": "isolated", "label": "ball_out", "duration": 12},
    {"type": "idle", "label": "REST / PAUSE", "duration": 5},
    {"type": "isolated", "label": "team_to_serve_left", "duration": 12},
    {"type": "idle", "label": "REST / PAUSE", "duration": 5},
    {"type": "isolated", "label": "double_contact", "duration": 12},
    {"type": "idle", "label": "REST / PAUSE", "duration": 5},
    {"type": "isolated", "label": "service_authorization_left", "duration": 12},
    {"type": "idle", "label": "REST / PAUSE", "duration": 5},
    {"type": "isolated", "label": "service_authorization_right", "duration": 12},
    {"type": "idle", "label": "REST / PAUSE", "duration": 5},
    {"type": "isolated", "label": "end_of_set", "duration": 12},
    {"type": "idle", "label": "REST / PAUSE", "duration": 5},
    {"type": "transition", "label": "team_to_serve_right  ->  ball_out  (NO PAUSE)", "duration": 15},
    {"type": "idle", "label": "REST / PAUSE", "duration": 5},
    {"type": "transition", "label": "team_to_serve_left  ->  double_contact  (NO PAUSE)", "duration": 15},
    {"type": "idle", "label": "REST / PAUSE", "duration": 5},
    {"type": "transition", "label": "team_to_serve_right  ->  double_contact  (NO PAUSE)", "duration": 15},
    {"type": "idle", "label": "REST / PAUSE", "duration": 5},
    {"type": "transition", "label": "team_to_serve_left  ->  ball_out  (NO PAUSE)", "duration": 15},
    {"type": "idle", "label": "TEST COMPLETE -- press Q to quit", "duration": 999},
]

PHASE_COLORS = {
    "idle": (150, 150, 150),
    "isolated": (0, 255, 255),
    "transition": (0, 100, 255),
}

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def load_model():
    with open(LABEL_MAP_PATH) as f:
        label_map_data = json.load(f)
    real_label_to_idx = label_map_data["real_label_to_idx"]
    idx_to_real_label = {int(v): k for k, v in real_label_to_idx.items()}

    input_size = ABLATED_FEATURE_COUNT if ABLATE_HAND_COORDS else TOTAL_FEATURES
    model = GestureCNNLSTM(input_size=input_size, num_classes=len(real_label_to_idx))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, idx_to_real_label, list(real_label_to_idx.keys())


def classify_window(raw_window_frames, model, idx_to_real_label):
    """Returns (predicted_label, top1_prob, full_probs_array) --
    full_probs_array is used for the on-screen probability bars."""
    raw = np.array(raw_window_frames)
    normalized = normalize_sequence(raw)
    normalized = ablate(normalized)
    resampled = resample_sequence(normalized, SEQUENCE_LENGTH)
    x = torch.tensor(resampled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).numpy()[0]

    predicted_label, top1_label, top2_label, top1_prob, top2_prob = decide_label(probs, idx_to_real_label)

    if predicted_label != NOTHING_LABEL:
        raw_resampled_for_tiebreak = resample_sequence(raw, SEQUENCE_LENGTH)
        predicted_label = apply_tie_breaker(raw_resampled_for_tiebreak, top1_label, top2_label, top1_prob, top2_prob)

    return predicted_label, top1_prob, probs


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


def draw_probability_bars(frame, probs, real_labels, origin_y=190):
    for i, label in enumerate(real_labels):
        score = probs[i]
        bar_color = (0, 255, 0) if score > DECISION_THRESHOLD else (255, 255, 255)
        text = f"{label:<28} {score:.2f}"
        cv2.putText(frame, text, (10, origin_y + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, bar_color, 2, cv2.LINE_AA)
        bar_x1 = 320
        bar_width = int(150 * score)
        cv2.rectangle(frame, (bar_x1, origin_y + i * 28 - 15),
                       (bar_x1 + bar_width, origin_y + i * 28 + 5), bar_color, -1)


def draw_gesture_history_bar(frame, history, frame_width, frame_height):
    text = "  ->  ".join(history) if history else "(no gestures committed yet)"
    rect_y1, rect_y2 = frame_height - 60, frame_height - 15
    cv2.rectangle(frame, (0, rect_y1), (frame_width, rect_y2), (40, 40, 40), -1)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = max(10, (frame_width - text_size[0]) // 2)
    cv2.putText(frame, text, (text_x, frame_height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def draw_phase_banner(frame, phase, remaining_seconds, get_ready_remaining, frame_width):
    color = PHASE_COLORS.get(phase["type"], (255, 255, 255))

    if get_ready_remaining > 0:
        countdown_text = f"GET READY: {int(get_ready_remaining) + 1}"
        text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 4)[0]
        text_x = (frame_width - text_size[0]) // 2
        cv2.putText(frame, countdown_text, (text_x, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 165, 255), 4, cv2.LINE_AA)
        cv2.putText(frame, f"Next: {phase['label']}", (max(10, text_x - 40), 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return

    label_text = phase["label"]
    time_text = f"{int(remaining_seconds) + 1}s"

    cv2.rectangle(frame, (0, 0), (frame_width, 60), (30, 30, 30), -1)
    cv2.putText(frame, label_text, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    cv2.putText(frame, time_text, (frame_width - 90, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

    phase_type_text = {
        "idle": "(just watching)",
        "isolated": "(perform this ONE gesture, then hold/rest)",
        "transition": "(perform BOTH, back-to-back, NO PAUSE between them)",
    }.get(phase["type"], "")
    cv2.putText(frame, phase_type_text, (15, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def main():
    model, idx_to_real_label, real_labels = load_model()
    engine = DecisionEngine()
    gui = ScoreboardGUI()

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"session_{int(time.time())}.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["timestamp", "phase_label", "phase_type", "pose_detected", "left_hand_detected",
                         "right_hand_detected", "window_predicted_label"]
                         + [f"score_{label}" for label in real_labels]
                         + ["vote_committed_label", "vote_count", "engine_event", "engine_reason"])
    print(f"Logging every window's scores to: {log_path}\n")

    pose_model = mp_pose.Pose(
        static_image_mode=False, model_complexity=0,  # CHANGED from 1 -- matches extract_keypoints.py's
        # training-time setting exactly. Live throughput was measured at
        # only ~10-11fps via log timestamps, meaning ROLLING_WINDOW_FRAMES=60
        # was actually spanning ~5.6s of real time, not the assumed ~2s --
        # a major contributor to both the missed-fast-gesture and
        # tail-spike issues. model_complexity=0 is meaningfully cheaper
        # per frame; re-measure actual fps after this change.
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
    recent_predictions = deque(maxlen=VOTE_WINDOW_SIZE)
    last_committed_label = None
    last_commit_time = 0
    frame_counter = 0
    show_skeleton = True
    whistle_flash_until = 0
    gesture_history = deque(maxlen=6)
    last_probs = None
    last_decision_text = ""
    last_decision_color = (255, 255, 255)
    last_decision_time = 0
    fps_frame_times = deque(maxlen=30)  # rolling window for a live measured-FPS readout

    phase_index = 0
    phase_start_time = time.time()
    get_ready_start_time = time.time()
    paused = False
    pause_started_at = 0.0
    total_paused_duration = 0.0  # accumulated pause time, subtracted from elapsed so pausing doesn't eat into your phase time

    print("Guided self-test running -- follow the on-screen banner. Press N to skip a phase early, Q to quit.\n")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_height, frame_width = frame.shape[:2]
        fps_frame_times.append(time.time())
        measured_fps = (len(fps_frame_times) - 1) / (fps_frame_times[-1] - fps_frame_times[0]) if len(fps_frame_times) > 1 else 0.0
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        features, draw_info = extract_frame_features(frame_rgb, pose_model, hands_model)
        if show_skeleton:
            draw_skeleton_overlay(frame, draw_info)

        phase = TEST_PLAN[phase_index]
        now = time.time()
        current_pause_elapsed = (now - pause_started_at) if paused else 0.0
        effective_now = now - total_paused_duration - current_pause_elapsed  # frozen while paused

        # CHANGED: idle/rest phases don't need a "get ready" countdown --
        # only active gesture phases (isolated/transition) do. Previously
        # every phase, including REST, got a pointless 4s countdown first.
        needs_get_ready = phase["type"] in ("isolated", "transition")
        get_ready_remaining = (GET_READY_SECONDS - (effective_now - get_ready_start_time)) if needs_get_ready else 0
        phase_elapsed = effective_now - phase_start_time
        phase_remaining = phase["duration"] - phase_elapsed

        in_get_ready = get_ready_remaining > 0

        if not in_get_ready and not paused and phase_remaining <= 0 and phase_index < len(TEST_PLAN) - 1:
            phase_index += 1
            phase_start_time = effective_now
            get_ready_start_time = effective_now
            recent_predictions.clear()
            print(f"\n=== Phase: {TEST_PLAN[phase_index]['label']} ===")

        rolling_window.append(features)
        frame_counter += 1

        if not in_get_ready and not paused and len(rolling_window) == ROLLING_WINDOW_FRAMES and frame_counter % INFERENCE_EVERY_N_FRAMES == 0:
            current_label, current_conf, full_probs = classify_window(list(rolling_window), model, idx_to_real_label)
            last_probs = full_probs
            recent_predictions.append(current_label)

            committed_label_this_frame = ""
            vote_count_this_frame = ""
            engine_event_this_frame = ""
            engine_reason_this_frame = ""

            if len(recent_predictions) == VOTE_WINDOW_SIZE:
                vote_counts = Counter(recent_predictions)
                top_label, top_count = vote_counts.most_common(1)[0]

                if top_count >= VOTES_NEEDED_TO_COMMIT:
                    commit_now = time.time()

                    if top_label == NOTHING_LABEL:
                        last_committed_label = None
                    else:
                        is_new_gesture = (top_label != last_committed_label)
                        cooldown_passed = (commit_now - last_commit_time) >= COMMIT_COOLDOWN_SECONDS

                        if is_new_gesture or cooldown_passed:
                            result = engine.on_gesture_detected(top_label, commit_now)
                            committed_label_this_frame = top_label
                            vote_count_this_frame = top_count
                            engine_event_this_frame = result["event"]
                            engine_reason_this_frame = result.get("reason", "")

                            last_decision_text = f"{top_label}: {result['event']}"
                            if result["event"] == "ignored":
                                last_decision_text += f" ({result['reason']})"
                                last_decision_color = (0, 0, 255)  # red -- rejected
                            else:
                                gesture_history.append(top_label)  # CHANGED: only record ACCEPTED events
                                last_decision_color = (0, 255, 0)  # green -- accepted
                            last_decision_time = commit_now

                            if result["event"] == "point_awarded":
                                gui.update_display(
                                    left_score=engine.score["left"],
                                    right_score=engine.score["right"],
                                    gesture_text=top_label,
                                )
                            elif result["event"] == "reason_attached":
                                gui.update_display(gesture_text=f"{top_label} ({result['side']})")

                            last_committed_label = top_label
                            last_commit_time = commit_now
                            recent_predictions.clear()

            log_writer.writerow([f"{time.time():.3f}", phase["label"], phase["type"],
                                 int(draw_info["pose_results"].pose_landmarks is not None),
                                 int(draw_info["left_detected"] > 0.5), int(draw_info["right_detected"] > 0.5),
                                 current_label]
                                 + [f"{score:.4f}" for score in full_probs]
                                 + [committed_label_this_frame, vote_count_this_frame,
                                    engine_event_this_frame, engine_reason_this_frame])
            log_file.flush()  # ensures rows survive even an abrupt Ctrl+C quit, not just clean Q-key exits

        gui.tick()

        draw_phase_banner(frame, phase, phase_remaining, get_ready_remaining, frame_width)
        if last_probs is not None:
            draw_probability_bars(frame, last_probs, real_labels)
        draw_gesture_history_bar(frame, list(gesture_history), frame_width, frame_height)

        if last_decision_text and (time.time() - last_decision_time) < 4:
            cv2.putText(frame, f"decision_engine: {last_decision_text}", (10, frame_height - 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, last_decision_color, 2, cv2.LINE_AA)

        if time.time() < whistle_flash_until:
            cv2.putText(frame, "WHISTLE (manual)", (frame_width - 220, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        cv2.putText(frame, f"measured fps: {measured_fps:.1f}", (frame_width - 220, frame_height - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if paused:
            cv2.putText(frame, "PAUSED (press P to resume)", (frame_width // 2 - 180, frame_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow("Guided Self-Test (multi-label)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            show_skeleton = not show_skeleton
        elif key == ord('p'):
            if not paused:
                paused = True
                pause_started_at = time.time()
            else:
                paused = False
                total_paused_duration += (time.time() - pause_started_at)
        elif key == ord('n'):
            phase_index = min(phase_index + 1, len(TEST_PLAN) - 1)
            phase_start_time = effective_now
            get_ready_start_time = effective_now
            recent_predictions.clear()
            print(f"\n=== Skipped to phase: {TEST_PLAN[phase_index]['label']} ===")
        elif key == ord('w'):
            engine.on_whistle_detected(time.time())
            whistle_flash_until = time.time() + 1.0

    cap.release()
    cv2.destroyAllWindows()
    gui.close()
    pose_model.close()
    hands_model.close()
    log_file.close()

    print(f"\nLog saved to: {log_path}")
    print("Full gesture history this session:")
    print("  ->  ".join(gesture_history) if gesture_history else "  (nothing committed)")


if __name__ == "__main__":
    main()