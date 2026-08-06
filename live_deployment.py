"""
live_deployment.py

REAL DEPLOYMENT SCRIPT -- built directly on simulate_game_test.py's proven
foundation (streak-based commit, pending-confirmation for team_to_serve vs
service_authorization, pose-detection gate, settle window via
decision_engine.py).

RECORDS TWO VIDEOS every session: a pure RAW feed (no overlay), and a FULL
WINDOW recording (everything the BACKSTAGE window shows -- skeleton,
probability bars, score, banners, gesture history -- pixel-exact match to
what you saw live).

--------------------------------------------------------------------
SYNC PASS (this version) -- brings this in line with the new
decision_engine.py (two-phase service_authorization design + the
left/right scoring fix):

1. CANCELLATION NOW COMMITS THE AUTHORIZATION: previously, cancelling
   a pending team_to_serve because a same-side service_authorization
   streak took over just cleared local state -- it never told
   decision_engine.py about the authorization at all. The new engine
   needs that on_gesture_detected() call to register Phase 1
   (service_authorization) and consume its whistle. Both cancellation
   sites (the top-priority race-condition-fixed check, and the
   mirrored check inside the streak-commit branch) now call
   do_commit() for the authorization gesture instead of only clearing
   state.

2. do_commit()'s expected_step LOGIC now handles the new
   "authorization_acknowledged" event -- previously it fell through to
   the generic else-branch (still logged/colored correctly, but the
   on-screen banner never explicitly transitioned to "waiting for
   whistle #2"). Now sets expected_step = "whistle" so the banner is
   accurate.

3. DISPLAY_NAME flipped (and ball_in/ball_touched added, matching
   replay_recorded_footage.py). The gesture class names are tied to
   the REFEREE's own left/right -- how the training data was filmed
   and labeled -- but decision_engine.py's GESTURE_TO_SCORE_SIDE now
   deliberately scores the OPPOSITE (audience/court-facing) side. This
   dict's KEYS still match the real model class names (lookups still
   work); only the printed VALUES changed, so what's shown on the
   gesture history bar matches which side of the scoreboard just moved
   instead of contradicting it.

REQUIRE_WHISTLE_FOR_SCORING is still False by default here (informational
whistle, matching the documented reason: detection isn't yet validated
during actual gesture motion, only standing still). Flip to True only
after that validation -- see replay_recorded_footage.py's version of
this flag for how to test the new two-whistle enforcement without
touching this default.

Controls:
  W - manual whistle (only matters if real whistle detection isn't active)
  Q / ESC - quit
  S - toggle skeleton overlay
  P - pause/resume (nothing gets processed while paused)
  [ / ] - manually adjust LEFT score down/up (mistake-correction safety net)
  - / + - manually adjust RIGHT score down/up
  R - manually clear the last-attached reason for the current point
"""

import time
import csv
import os
from collections import deque

import cv2
import json
import numpy as np
import torch
import mediapipe as mp

import extract_keypoints
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
    NOTHING_LABEL,
    decide_label,
    apply_tie_breaker,
)
from model import GestureCNNLSTM
from decision_engine import DecisionEngine

MODEL_PATH = "models/final_model.pt"
LABEL_MAP_PATH = "models/label_map.json"
CAMERA_INDEX = 1
LOG_DIR = "data/live_test_logs"

# ============================================================
# TUNABLE SETTINGS
# ============================================================

ROLLING_WINDOW_FRAMES = 24
INFERENCE_EVERY_N_FRAMES = 3
STREAK_NEEDED_TO_COMMIT = 5
FAULT_STREAK_NEEDED_TO_COMMIT = 3
COMMIT_COOLDOWN_SECONDS = 2.0
STALE_VOTE_SECONDS = 3.0
TEAM_TO_SERVE_CONFIRM_DELAY_SECONDS = 2.0
MIN_POSE_DETECTED_FRACTION = 0.5
HISTORY_CLEAR_AFTER_IDLE_SECONDS = 6.0
CONSOLE_REFRESH_AFTER_IDLE_SECONDS = 8.0

GAME_WIN_SCORE = 25
GAME_WIN_BY_MARGIN = 2

FAST_HAND_CROP_MODE = False

# RECORD_RAW_FOOTAGE: saves the pure camera feed (no skeleton, no UI text,
# no overlays -- exactly what the camera captured) to a video file, so you
# can later replay that EXACT real session through the pipeline via
# replay_recorded_footage.py to test code changes against real, previously-
# observed footage.
RECORD_RAW_FOOTAGE = True  # TUNABLE
RECORDINGS_DIR = "data/raw_recordings"
RAW_RECORD_FPS = 10  # TUNABLE. Matches your measured real live throughput.

# RECORD_FULL_WINDOW_FOOTAGE: saves a SECOND video -- exactly what the
# BACKSTAGE window shows live: skeleton, probability bars, score bar,
# instruction banner, gesture history, everything. Captured from the same
# fully-overlaid frame right before it's displayed, so it's a pixel-exact
# recording of what you saw during the session.
RECORD_FULL_WINDOW_FOOTAGE = True  # TUNABLE
FULL_WINDOW_RECORD_FPS = RAW_RECORD_FPS  # kept identical so both videos stay frame-synced

WHISTLE_DEVICE_INDEX = None  # TUNABLE -- set to your confirmed-working index, e.g. 3

REQUIRE_WHISTLE_FOR_SCORING = False  # TUNABLE

# NOTE: display text is flipped relative to the model's literal class
# names. The gesture classes are named after the REFEREE's own
# left/right (how the data was filmed/labeled), but what's shown on
# screen -- and what actually gets scored, see decision_engine.py's
# GESTURE_TO_SCORE_SIDE -- should reflect the audience/court-facing
# side instead, so what you see on the gesture history bar matches
# which side of the scoreboard just moved.
DISPLAY_NAME = {
    "team_to_serve_left": "Team to Serve Right",
    "team_to_serve_right": "Team to Serve Left",
    "ball_out": "Ball Out",
    "double_contact": "Double Contact",
    "end_of_set": "End of Set",
    "service_authorization_left": "Service Auth. Right",
    "service_authorization_right": "Service Auth. Left",
    "ball_in": "Ball In",
    "ball_touched": "Ball Touched",
}

# NOTE: this map is INTERNAL bookkeeping only -- it pairs a pending
# team_to_serve gesture with its same-side service_authorization
# cancellation partner (both tied to the model's own literal class-name
# suffix, i.e. the REFEREE's side, not the scored side). It must NOT be
# flipped -- the actual score-side flip lives entirely in
# decision_engine.py's GESTURE_TO_SCORE_SIDE, applied only once
# do_commit() calls into the engine.
SCORING_SIDE_MAP = {"team_to_serve_left": "left", "team_to_serve_right": "right"}

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

extract_keypoints.LIVE_FAST_MODE = FAST_HAND_CROP_MODE


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


def try_load_whistle_detector(on_whistle):
    try:
        from whistle_detector import WhistleDetector
    except ImportError:
        print("whistle_detector.py not found -- using MANUAL WHISTLE MODE (press W).")
        return None

    try:
        detector = WhistleDetector(on_whistle_callback=on_whistle, device=WHISTLE_DEVICE_INDEX)
    except FileNotFoundError as e:
        print(f"Real whistle model not available yet ({e})")
        print("Falling back to MANUAL WHISTLE MODE (press W).")
        return None

    detector.start()
    print("REAL whistle detection ACTIVE (mic-based). W key still works as backup.")
    return detector


def classify_window(raw_window_frames, model, idx_to_real_label):
    raw = np.array(raw_window_frames)

    pose_part = raw[:, :24]
    frames_with_pose = np.any(pose_part != 0, axis=1)
    pose_detected_fraction = frames_with_pose.mean()
    if pose_detected_fraction < MIN_POSE_DETECTED_FRACTION:
        return NOTHING_LABEL, 0.0, np.zeros(len(idx_to_real_label))

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

    draw_info = {"pose_results": pose_results, "hand_coords": hand_coords,
                 "left_detected": left_det, "right_detected": right_det}
    return features, draw_info


def draw_skeleton_overlay(frame, draw_info):
    pose_results = draw_info["pose_results"]
    if pose_results.pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2),
        )


def draw_instruction_banner(frame, expected_step, frame_width, pending_label=None, pending_remaining=None, whistle_mode="manual"):
    if whistle_mode == "auto" and not REQUIRE_WHISTLE_FOR_SCORING:
        mode_text = "WHISTLE (info only)"
        mode_color = (0, 200, 255)
    elif whistle_mode == "auto":
        mode_text = "AUTO WHISTLE"
        mode_color = (0, 255, 0)
    else:
        mode_text = "MANUAL WHISTLE (W)"
        mode_color = (0, 165, 255)

    if pending_label is not None:
        side = SCORING_SIDE_MAP[pending_label]
        text = f"team_to_serve_{side} PENDING -- confirming in {pending_remaining:.1f}s"
        cv2.rectangle(frame, (0, 0), (frame_width, 60), (30, 30, 30), -1)
        cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, mode_text, (frame_width - 220, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 2)
        return

    messages = {
        "whistle": ("Waiting for WHISTLE", (0, 165, 255)),
        "scoring_gesture": ("Waiting for team_to_serve_LEFT/RIGHT", (0, 255, 255)),
        "reason_gesture": ("Waiting for fault/reason gesture", (0, 100, 255)),
    }
    text, color = messages[expected_step]
    cv2.rectangle(frame, (0, 0), (frame_width, 60), (30, 30, 30), -1)
    cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    cv2.putText(frame, mode_text, (frame_width - 220, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 2)


def draw_probability_bars(frame, probs, real_labels, origin_y=190):
    for i, label in enumerate(real_labels):
        score = probs[i]
        bar_color = (0, 255, 0) if score > 0.5 else (255, 255, 255)
        text = f"{label:<28} {score:.2f}"
        cv2.putText(frame, text, (10, origin_y + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, bar_color, 2, cv2.LINE_AA)
        bar_x1 = 320
        bar_width = int(150 * score)
        cv2.rectangle(frame, (bar_x1, origin_y + i * 28 - 15),
                       (bar_x1 + bar_width, origin_y + i * 28 + 5), bar_color, -1)


def draw_vote_progress(frame, streak_label, streak_count, frame_width, origin_y=155):
    if streak_label is None:
        return
    pretty = DISPLAY_NAME.get(streak_label, streak_label)
    text = f"building streak: {pretty} ({streak_count}/{STREAK_NEEDED_TO_COMMIT})"
    color = (0, 255, 0) if streak_count >= STREAK_NEEDED_TO_COMMIT else (0, 200, 255)
    cv2.putText(frame, text, (15, origin_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_score_bar(frame, engine, frame_width, frame_height):
    score_text = f"LEFT {engine.score['left']}  -  {engine.score['right']} RIGHT   (first to {engine.win_score}, win by {engine.win_by_margin})"
    cv2.rectangle(frame, (0, 65), (frame_width, 100), (20, 20, 20), -1)
    text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = max(10, (frame_width - text_size[0]) // 2)
    cv2.putText(frame, score_text, (text_x, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


SCOREBOARD_WIDTH = 900
SCOREBOARD_HEIGHT = 400


def build_scoreboard_canvas(engine, paused):
    canvas = np.zeros((SCOREBOARD_HEIGHT, SCOREBOARD_WIDTH, 3), dtype=np.uint8)
    canvas[:] = (25, 25, 25)

    if paused:
        text = "PAUSED"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 6)[0]
        text_x = (SCOREBOARD_WIDTH - text_size[0]) // 2
        cv2.putText(canvas, text, (text_x, SCOREBOARD_HEIGHT // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 165, 255), 6, cv2.LINE_AA)
        return canvas

    if engine.set_over:
        winner = "LEFT" if engine.score["left"] > engine.score["right"] else "RIGHT"
        text = f"{winner} WINS"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 6)[0]
        text_x = (SCOREBOARD_WIDTH - text_size[0]) // 2
        cv2.putText(canvas, text, (text_x, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 6, cv2.LINE_AA)

    cv2.putText(canvas, "TEAM 1", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(canvas, "TEAM 2", (SCOREBOARD_WIDTH - 300, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 200, 200), 2, cv2.LINE_AA)

    left_text = str(engine.score["left"])
    right_text = str(engine.score["right"])
    cv2.putText(canvas, left_text, (140, 320), cv2.FONT_HERSHEY_SIMPLEX, 5.5, (255, 255, 255), 12, cv2.LINE_AA)
    cv2.putText(canvas, right_text, (SCOREBOARD_WIDTH - 320, 320), cv2.FONT_HERSHEY_SIMPLEX, 5.5, (255, 255, 255), 12, cv2.LINE_AA)

    dash_size = cv2.getTextSize("-", cv2.FONT_HERSHEY_SIMPLEX, 3.0, 8)[0]
    cv2.putText(canvas, "-", ((SCOREBOARD_WIDTH - dash_size[0]) // 2, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, (150, 150, 150), 8, cv2.LINE_AA)

    return canvas


def draw_strict_score_small(frame, strict_engine, frame_width):
    text = f"[if whistle required] L {strict_engine.score['left']} - {strict_engine.score['right']} R"
    cv2.putText(frame, text, (frame_width - 260, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)


def draw_gesture_history_bar(frame, history, frame_width, frame_height):
    pretty_history = [DISPLAY_NAME.get(label, label) for label in history]
    text = "  ->  ".join(pretty_history) if pretty_history else "(no gestures committed yet)"
    rect_y1, rect_y2 = frame_height - 60, frame_height - 15
    cv2.rectangle(frame, (0, rect_y1), (frame_width, rect_y2), (40, 40, 40), -1)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x = max(10, (frame_width - text_size[0]) // 2)
    cv2.putText(frame, text, (text_x, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    model, idx_to_real_label, real_labels = load_model()
    engine = DecisionEngine(win_score=GAME_WIN_SCORE, win_by_margin=GAME_WIN_BY_MARGIN)
    strict_engine = DecisionEngine(win_score=GAME_WIN_SCORE, win_by_margin=GAME_WIN_BY_MARGIN)

    os.makedirs(LOG_DIR, exist_ok=True)
    session_start_time = time.time()
    log_path = os.path.join(LOG_DIR, f"deployment_{int(session_start_time)}.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["timestamp", "expected_step", "window_predicted_label",
                          "vote_committed_label", "engine_event", "engine_reason",
                          "score_left", "score_right",
                          "strict_score_left", "strict_score_right"])

    strict_log_path = os.path.join(LOG_DIR, f"deployment_strict_{int(session_start_time)}.csv")
    strict_log_file = open(strict_log_path, "w", newline="")
    strict_log_writer = csv.writer(strict_log_file)
    strict_log_writer.writerow(["timestamp", "event_type", "label", "engine_event", "engine_reason",
                                 "strict_score_left", "strict_score_right"])

    print(f"Logging to: {log_path}")
    print(f"Strict (whistle-required) log: {strict_log_path}")
    print(f"SESSION START (epoch seconds): {session_start_time:.3f}")
    print(f"REAL GAME: first to {GAME_WIN_SCORE}, win by {GAME_WIN_BY_MARGIN}.\n")

    whistle_mode = {"value": "manual"}
    whistle_flash_until_holder = {"value": 0.0}

    def on_whistle(timestamp, confidence=None):
        engine.on_whistle_detected(timestamp)
        strict_engine.on_whistle_detected(timestamp)
        strict_log_writer.writerow([f"{timestamp:.3f}", "whistle", "", "", "",
                                     strict_engine.score["left"], strict_engine.score["right"]])
        strict_log_file.flush()
        whistle_flash_until_holder["value"] = time.time() + 1.5
        conf_str = f" (confidence={confidence:.2f})" if confidence is not None else ""
        print(f"  -> WHISTLE{conf_str} at {timestamp:.3f}")

    detector = try_load_whistle_detector(on_whistle)
    whistle_mode["value"] = "auto" if detector is not None else "manual"
    if not REQUIRE_WHISTLE_FOR_SCORING:
        print("REQUIRE_WHISTLE_FOR_SCORING is False -- whistle is informational only, "
              "scoring will NOT be blocked by a missed detection.")

    pose_model = mp_pose.Pose(static_image_mode=False, model_complexity=0,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands_model = mp_hands.Hands(static_image_mode=True, max_num_hands=2,
                                  min_detection_confidence=0.1, min_tracking_confidence=0.28)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: could not open camera at index {CAMERA_INDEX}.")
        return

    raw_writer = None
    raw_recording_path = None
    if RECORD_RAW_FOOTAGE:
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        raw_recording_path = os.path.join(RECORDINGS_DIR, f"raw_{int(session_start_time)}.mp4")
        raw_writer = cv2.VideoWriter(raw_recording_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                      RAW_RECORD_FPS, (frame_w, frame_h))
        print(f"Recording RAW (no overlay) footage to: {raw_recording_path}")

    full_window_writer = None
    full_window_recording_path = None
    if RECORD_FULL_WINDOW_FOOTAGE:
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        full_window_recording_path = os.path.join(RECORDINGS_DIR, f"fullwindow_{int(session_start_time)}.mp4")
        full_window_writer = cv2.VideoWriter(full_window_recording_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                              FULL_WINDOW_RECORD_FPS, (frame_w, frame_h))
        print(f"Recording FULL WINDOW (everything shown live) footage to: {full_window_recording_path}")

    rolling_window = deque(maxlen=ROLLING_WINDOW_FRAMES)
    streak_label = None
    streak_count = 0
    last_confident_append_time = 0.0
    last_committed_label = None
    last_commit_time = 0
    frame_counter = 0
    show_skeleton = True
    gesture_history = deque(maxlen=8)
    last_decision_text = ""
    last_probs = None
    last_decision_color = (255, 255, 255)
    last_decision_time = 0

    pending_scoring_label = None
    pending_scoring_since = 0.0

    expected_step = "whistle"
    paused = True
    last_scoreboard_state = [None]
    last_scoreboard_canvas = [build_scoreboard_canvas(engine, paused)]
    fps_frame_times = deque(maxlen=30)

    print("Live deployment running. Press Q/ESC to quit.\n")

    def do_commit(label, now):
        nonlocal last_decision_text, last_decision_color, last_decision_time
        nonlocal last_committed_label, last_commit_time, expected_step
        result = engine.on_gesture_detected(label, now)
        strict_result = strict_engine.on_gesture_detected(label, now)
        strict_log_writer.writerow([f"{now:.3f}", "gesture", label, strict_result["event"],
                                     strict_result.get("reason", ""),
                                     strict_engine.score["left"], strict_engine.score["right"]])
        strict_log_file.flush()
        last_decision_text = f"{label}: {result['event']}"
        if result["event"] == "ignored":
            last_decision_text += f" ({result['reason']})"
            last_decision_color = (0, 0, 255)
        else:
            gesture_history.append(label)
            last_decision_color = (0, 255, 0)
            if result["event"] == "point_awarded":
                expected_step = "reason_gesture"
            elif result["event"] == "reason_attached":
                expected_step = "whistle"
            elif result["event"] == "authorization_acknowledged":
                # CHANGED: previously fell through unhandled -- now
                # correctly shows "waiting for whistle #2" next, since
                # the engine just consumed whistle #1 for this
                # authorization and needs a genuine second whistle
                # before team_to_serve can be accepted.
                expected_step = "whistle"
        last_decision_time = now
        last_committed_label = label
        last_commit_time = now
        return result

    while True:
        success, frame = cap.read()
        if not success:
            break

        if raw_writer is not None:
            raw_writer.write(frame)

        frame_height, frame_width = frame.shape[:2]
        fps_frame_times.append(time.time())
        measured_fps = (len(fps_frame_times) - 1) / (fps_frame_times[-1] - fps_frame_times[0]) if len(fps_frame_times) > 1 else 0.0

        if not paused:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            features, draw_info = extract_frame_features(frame_rgb, pose_model, hands_model)
            if show_skeleton:
                draw_skeleton_overlay(frame, draw_info)

            rolling_window.append(features)
            frame_counter += 1

            if not REQUIRE_WHISTLE_FOR_SCORING:
                engine.on_whistle_detected(time.time())

        if not paused and len(rolling_window) == ROLLING_WINDOW_FRAMES and frame_counter % INFERENCE_EVERY_N_FRAMES == 0:
            current_label, current_conf, full_probs = classify_window(list(rolling_window), model, idx_to_real_label)
            last_probs = full_probs

            if current_label != NOTHING_LABEL:
                if current_label == streak_label:
                    streak_count += 1
                else:
                    streak_label = current_label
                    streak_count = 1
                last_confident_append_time = time.time()
            elif streak_label is not None and (time.time() - last_confident_append_time) > STALE_VOTE_SECONDS:
                streak_label = None
                streak_count = 0

            if gesture_history and (time.time() - last_commit_time) > HISTORY_CLEAR_AFTER_IDLE_SECONDS:
                gesture_history.clear()

            now = time.time()
            committed_label_this_frame = ""
            engine_event_this_frame = ""
            engine_reason_this_frame = ""

            cancel_check_triggered = False
            if (pending_scoring_label is not None and streak_label is not None
                    and streak_count >= FAULT_STREAK_NEEDED_TO_COMMIT):
                pending_side = SCORING_SIDE_MAP[pending_scoring_label]
                pending_same_side_auth = f"service_authorization_{pending_side}"
                if streak_label == pending_same_side_auth:
                    # CHANGED: actually commit the authorization to the
                    # engine now, instead of only clearing local state.
                    # The new decision_engine.py needs this call to
                    # register Phase 1 (service_authorization) and
                    # consume its whistle.
                    result = do_commit(streak_label, now)
                    committed_label_this_frame = streak_label
                    engine_event_this_frame = result["event"]
                    engine_reason_this_frame = result.get("reason", "")
                    pending_scoring_label = None
                    streak_label = None
                    streak_count = 0
                    cancel_check_triggered = True

            if cancel_check_triggered:
                pass

            elif pending_scoring_label is not None and (now - pending_scoring_since) >= TEAM_TO_SERVE_CONFIRM_DELAY_SECONDS:
                result = do_commit(pending_scoring_label, now)
                committed_label_this_frame = pending_scoring_label
                engine_event_this_frame = result["event"]
                engine_reason_this_frame = result.get("reason", "")
                pending_scoring_label = None
                streak_label = None
                streak_count = 0
                rolling_window.clear()
                engine.last_settle_start_time = None
                strict_engine.last_settle_start_time = None

            elif streak_label is not None and streak_count >= (STREAK_NEEDED_TO_COMMIT if streak_label in SCORING_SIDE_MAP else FAULT_STREAK_NEEDED_TO_COMMIT):
                top_label = streak_label

                if top_label in SCORING_SIDE_MAP:
                    if pending_scoring_label != top_label:
                        pending_scoring_label = top_label
                        pending_scoring_since = now
                    streak_count = STREAK_NEEDED_TO_COMMIT

                else:
                    if pending_scoring_label is not None:
                        pending_side = SCORING_SIDE_MAP[pending_scoring_label]
                        pending_same_side_auth = f"service_authorization_{pending_side}"

                        if top_label == pending_same_side_auth:
                            # CHANGED: same fix as above -- commit the
                            # authorization instead of only clearing.
                            result = do_commit(top_label, now)
                            committed_label_this_frame = top_label
                            engine_event_this_frame = result["event"]
                            engine_reason_this_frame = result.get("reason", "")
                            pending_scoring_label = None
                            streak_label = None
                            streak_count = 0
                        else:
                            result = do_commit(pending_scoring_label, now)
                            committed_label_this_frame = pending_scoring_label
                            engine_event_this_frame = result["event"]
                            engine_reason_this_frame = result.get("reason", "")
                            pending_scoring_label = None
                            engine.last_settle_start_time = None
                            strict_engine.last_settle_start_time = None
                            fault_result = do_commit(top_label, now)
                            committed_label_this_frame = top_label
                            engine_event_this_frame = fault_result["event"]
                            engine_reason_this_frame = fault_result.get("reason", "")
                            streak_label = None
                            streak_count = 0
                            rolling_window.clear()
                    else:
                        is_new_gesture = (top_label != last_committed_label)
                        cooldown_passed = (now - last_commit_time) >= COMMIT_COOLDOWN_SECONDS

                        if is_new_gesture or cooldown_passed:
                            result = do_commit(top_label, now)
                            committed_label_this_frame = top_label
                            engine_event_this_frame = result["event"]
                            engine_reason_this_frame = result.get("reason", "")
                            streak_label = None
                            streak_count = 0
                            rolling_window.clear()

            log_writer.writerow([f"{time.time():.3f}", expected_step, current_label,
                                  committed_label_this_frame, engine_event_this_frame, engine_reason_this_frame,
                                  engine.score["left"], engine.score["right"],
                                  strict_engine.score["left"], strict_engine.score["right"]])
            log_file.flush()

        if engine.set_over:
            winner = "LEFT" if engine.score["left"] > engine.score["right"] else "RIGHT"
            cv2.rectangle(frame, (0, 0), (frame_width, 100), (0, 100, 0), -1)
            cv2.putText(frame, f"MATCH OVER -- {winner} WINS {engine.score['left']}-{engine.score['right']}",
                        (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
        else:
            pending_remaining = None
            if pending_scoring_label is not None:
                pending_remaining = max(0.0, TEAM_TO_SERVE_CONFIRM_DELAY_SECONDS - (time.time() - pending_scoring_since))
            draw_instruction_banner(frame, expected_step, frame_width, pending_scoring_label, pending_remaining, whistle_mode["value"])
            draw_score_bar(frame, engine, frame_width, frame_height)
            draw_strict_score_small(frame, strict_engine, frame_width)
            draw_vote_progress(frame, streak_label, streak_count, frame_width)

        if last_probs is not None:
            draw_probability_bars(frame, last_probs, real_labels)

        draw_gesture_history_bar(frame, list(gesture_history), frame_width, frame_height)

        if last_decision_text and (time.time() - last_decision_time) < 4:
            cv2.putText(frame, f"decision_engine: {last_decision_text}", (10, frame_height - 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, last_decision_color, 2, cv2.LINE_AA)

        if time.time() < whistle_flash_until_holder["value"]:
            whistle_text = "WHISTLE!"
            text_size = cv2.getTextSize(whistle_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 4)[0]
            text_x = (frame_width - text_size[0]) // 2
            cv2.putText(frame, whistle_text, (text_x, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 4, cv2.LINE_AA)

        if paused:
            cv2.putText(frame, "PAUSED -- nothing being processed (press P to resume)",
                        (frame_width // 2 - 260, frame_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"measured fps: {measured_fps:.1f}", (frame_width - 220, frame_height - 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.putText(frame, "(click this window: Q/ESC=quit  P=pause/resume  W=manual whistle  [/]=left score  -/+=right score  R=clear reason)",
                    (10, frame_height - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        if full_window_writer is not None:
            full_window_writer.write(frame)

        cv2.imshow("BACKSTAGE (control)", frame)
        current_scoreboard_state = (engine.score["left"], engine.score["right"], paused, engine.set_over)
        if current_scoreboard_state != last_scoreboard_state[0]:
            last_scoreboard_canvas[0] = build_scoreboard_canvas(engine, paused)
            last_scoreboard_state[0] = current_scoreboard_state
        cv2.imshow("SCOREBOARD", last_scoreboard_canvas[0])

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('p'):
            paused = not paused
            if not paused:
                rolling_window.clear()
                streak_label = None
                streak_count = 0
                pending_scoring_label = None
                frame_counter = 0
                print("  -> RESUMED (cleared stale buffer)")
            else:
                print("  -> PAUSED")
        elif key == ord('s'):
            show_skeleton = not show_skeleton
        elif key == ord('w'):
            on_whistle(time.time())
            if expected_step == "whistle":
                expected_step = "scoring_gesture"
        elif key == ord('['):
            engine.manual_override_score("left", -1)
            print(f"  -> MANUAL: left score -1 -> {engine.score}")
        elif key == ord(']'):
            engine.manual_override_score("left", +1)
            print(f"  -> MANUAL: left score +1 -> {engine.score}")
        elif key == ord('-'):
            engine.manual_override_score("right", -1)
            print(f"  -> MANUAL: right score -1 -> {engine.score}")
        elif key == ord('+') or key == ord('='):
            engine.manual_override_score("right", +1)
            print(f"  -> MANUAL: right score +1 -> {engine.score}")
        elif key == ord('r'):
            engine.manual_clear_reason()
            print("  -> MANUAL: cleared last reason for the current point")

    cap.release()
    if raw_writer is not None:
        raw_writer.release()
    if full_window_writer is not None:
        full_window_writer.release()
    cv2.destroyAllWindows()
    pose_model.close()
    hands_model.close()
    if detector is not None:
        detector.stop()
    log_file.close()
    strict_log_file.close()

    print(f"\nLog saved to: {log_path}")
    print(f"Strict log saved to: {strict_log_path}")
    if raw_recording_path is not None:
        print(f"Raw footage saved to: {raw_recording_path}")
    if full_window_recording_path is not None:
        print(f"Full-window footage saved to: {full_window_recording_path}")
    print(f"Final score (informational whistle): LEFT {engine.score['left']} - {engine.score['right']} RIGHT")
    print(f"Final score (strict, whistle-required): LEFT {strict_engine.score['left']} - {strict_engine.score['right']} RIGHT")


if __name__ == "__main__":
    main()