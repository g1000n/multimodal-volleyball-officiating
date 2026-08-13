"""
simulate_game_test.py

Full end-to-end simulated match, using the REAL trained model and the
REAL decision_engine -- not a scripted replay.

HOW IT WORKS:
  This script does NOT decide who wins each point. YOU decide, by
  which gesture you actually perform (team_to_serve_LEFT vs RIGHT).
  The script's only job is to guide you through the correct sequence at
  each step and display the real, live score as tracked by
  decision_engine.

WHISTLE STATUS: real whistle detection isn't available yet -- W stays
the manual placeholder until then.

--------------------------------------------------------------------
NEW: DELAYED SCORING CONFIRMATION (your idea) -- team_to_serve_left/
right no longer commits the instant its streak hits threshold. Since
team_to_serve and SAME-SIDE service_authorization share the same early
"raise the arm" motion, a normal-paced service_authorization could
build a team_to_serve streak before the distinguishing elbow-bend ever
happens.

Instead: once team_to_serve_X's streak hits threshold, it goes
"pending" for TEAM_TO_SERVE_CONFIRM_DELAY_SECONDS instead of
committing immediately:
  - If service_authorization_X (SAME side) takes over and builds its
    own streak during that wait -> the pending team_to_serve is
    DISCARDED entirely (no point awarded) -- it was really just the
    start of a service_authorization.
  - If ANYTHING ELSE happens instead (a fault gesture builds its own
    streak, or the delay simply elapses with nothing else taking
    over) -> the pending team_to_serve is CONFIRMED and committed for
    real, exactly as you described: "if they do other gestures of
    faults, as long as it's not service authorization it's okay."

ALL THE SETTINGS YOU CAN TUNE YOURSELF ARE MARKED "TUNABLE" BELOW.
--------------------------------------------------------------------

Controls:
  W - simulate whistle (temporary, until real detection exists)
  Q / ESC - quit
  S - toggle skeleton overlay
  [ / ] - manually adjust LEFT score down/up (mistake-correction safety net)
  - / + - manually adjust RIGHT score down/up
  R - manually clear the last-attached reason for the current point
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
    NOTHING_LABEL,
    decide_label,
    apply_tie_breaker,
)
from model import GestureCNNLSTM
from decision_engine import DecisionEngine, SCORING_GESTURES, FAULT_REASON_GESTURES

MODEL_PATH = "models/final_model.pt"
LABEL_MAP_PATH = "models/label_map.json"
CAMERA_INDEX = 1
LOG_DIR = "data/live_test_logs"

# ============================================================
# TUNABLE SETTINGS -- everything you might want to adjust yourself
# ============================================================

# --- Timing / responsiveness ---
ROLLING_WINDOW_FRAMES = 24          # TUNABLE. How many raw frames of context
# each classification looks at. Bigger = more context per guess but slower
# to react to a fast gesture. Smaller = faster reaction but each guess is
# based on less motion, so more error-prone. At your measured ~10fps, 24
# frames ~= 2.4s of real time.

INFERENCE_EVERY_N_FRAMES = 3        # TUNABLE. Classify every Nth camera frame,
# not every single one. 1 = classify every frame (fastest reaction, costs
# more compute). 3 = classify every 3rd frame (what you're currently using).

STREAK_NEEDED_TO_COMMIT = 5         # TUNABLE. How many CONSECUTIVE confident
# hits of the SAME label are needed before it's trusted enough to act on.
# Higher = slower but more resistant to brief false spikes. Lower = faster
# but more prone to committing on noise.

COMMIT_COOLDOWN_SECONDS = 2.0       # TUNABLE. Minimum time between committing
# the SAME label twice in a row (prevents a continuously-held pose from
# re-firing repeatedly). Must stay ABOVE SETTLE_WINDOW_SECONDS in
# decision_engine.py (currently 1.5s) -- see that file if you touch this.

STALE_VOTE_SECONDS = 3.0            # TUNABLE. If it's been this long since
# the last confident hit, a new one is treated as starting fresh instead of
# combining with old, possibly-unrelated ones.

# --- NEW: delayed scoring confirmation (your idea) ---
TEAM_TO_SERVE_CONFIRM_DELAY_SECONDS = 2.0   # TUNABLE. How long team_to_serve
# waits after reaching its own streak before actually committing as a real
# point -- giving a same-side service_authorization time to "take over" if
# that's what's really happening. Too short = doesn't give service_authorization
# enough time to reveal itself. Too long = real points feel sluggish to
# register. Start here and adjust based on how it feels live.

# --- Safety gates ---
MIN_POSE_DETECTED_FRACTION = 0.5    # TUNABLE. If fewer than this fraction of
# the window's frames have a detected pose (e.g. camera not connected,
# you're out of frame), the whole window is forced to "nothing" regardless
# of what the model says.

# --- Practice game win condition ---
TEST_WIN_SCORE = 7                  # TUNABLE. Real games use 25 -- this is
# shortened for practical end-to-end testing time.
TEST_WIN_BY_MARGIN = 2

# --- Testing convenience ---
DISABLE_WHISTLE_REQUIREMENT = True  # TUNABLE. True = score without needing W
# first each time (real whistle detection isn't delivered yet). Set False to
# test the whistle-gated flow once you want that.

# ============================================================

SCORING_SIDE_MAP = {"team_to_serve_left": "left", "team_to_serve_right": "right"}

DISPLAY_NAME = {
    "team_to_serve_left": "Team to Serve Left",
    "team_to_serve_right": "Team to Serve Right",
    "ball_out": "Ball Out",
    "double_contact": "Double Contact",
    "end_of_set": "End of Set",
    "service_authorization_left": "Service Auth. Left",
    "service_authorization_right": "Service Auth. Right",
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


def draw_instruction_banner(frame, expected_step, frame_width, pending_label=None, pending_remaining=None):
    if pending_label is not None:
        side = SCORING_SIDE_MAP[pending_label]
        text = f"team_to_serve_{side} PENDING -- confirming in {pending_remaining:.1f}s (switch to service_authorization_{side} to cancel)"
        color = (0, 200, 255)
        cv2.rectangle(frame, (0, 0), (frame_width, 60), (30, 30, 30), -1)
        cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        return

    messages = {
        "whistle": ("Press W for WHISTLE", (0, 165, 255)),
        "scoring_gesture": ("Perform team_to_serve_LEFT or team_to_serve_RIGHT (your choice)", (0, 255, 255)),
        "reason_gesture": ("Perform a fault/reason gesture: ball_out / double_contact / "
                           "service_authorization_left / service_authorization_right (your choice)", (0, 100, 255)),
    }
    text, color = messages[expected_step]
    cv2.rectangle(frame, (0, 0), (frame_width, 60), (30, 30, 30), -1)
    cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


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


def draw_gesture_history_bar(frame, history, frame_width, frame_height):
    pretty_history = [DISPLAY_NAME.get(label, label) for label in history]
    text = "  ->  ".join(pretty_history) if pretty_history else "(no gestures committed yet)"
    rect_y1, rect_y2 = frame_height - 60, frame_height - 15
    cv2.rectangle(frame, (0, rect_y1), (frame_width, rect_y2), (40, 40, 40), -1)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x = max(10, (frame_width - text_size[0]) // 2)
    cv2.putText(frame, text, (text_x, frame_height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    model, idx_to_real_label, real_labels = load_model()
    engine = DecisionEngine(win_score=TEST_WIN_SCORE, win_by_margin=TEST_WIN_BY_MARGIN)

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"simgame_{int(time.time())}.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["timestamp", "expected_step", "window_predicted_label"]
                         + [f"score_{label}" for label in real_labels]
                         + ["vote_committed_label", "vote_count", "engine_event", "engine_reason",
                            "pending_label", "score_left", "score_right"])
    print(f"Logging to: {log_path}")
    print(f"Simulated set: first to {TEST_WIN_SCORE}, win by {TEST_WIN_BY_MARGIN}.\n")

    pose_model = mp_pose.Pose(
        static_image_mode=False, model_complexity=0,
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
    streak_label = None
    streak_count = 0
    last_confident_append_time = 0.0
    last_committed_label = None
    last_commit_time = 0
    frame_counter = 0
    show_skeleton = True
    gesture_history = deque(maxlen=8)
    last_probs = None
    last_decision_text = ""
    last_decision_color = (255, 255, 255)
    last_decision_time = 0
    whistle_flash_until = 0

    # NEW: pending-confirmation state for team_to_serve
    pending_scoring_label = None
    pending_scoring_since = 0.0

    expected_step = "scoring_gesture" if DISABLE_WHISTLE_REQUIREMENT else "whistle"

    print("Guided simulated match running. Follow the on-screen instruction. Press Q/ESC to quit.\n")

    def do_commit(label, streak_count_for_log, now):
        """Actually calls decision_engine and updates all the display/log state for a real commit."""
        nonlocal last_decision_text, last_decision_color, last_decision_time
        nonlocal last_committed_label, last_commit_time, expected_step
        result = engine.on_gesture_detected(label, now)

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
                expected_step = "scoring_gesture" if DISABLE_WHISTLE_REQUIREMENT else "whistle"

        last_decision_time = now
        last_committed_label = label
        last_commit_time = now
        return result

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_height, frame_width = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        features, draw_info = extract_frame_features(frame_rgb, pose_model, hands_model)
        if show_skeleton:
            draw_skeleton_overlay(frame, draw_info)

        rolling_window.append(features)
        frame_counter += 1

        if DISABLE_WHISTLE_REQUIREMENT:
            engine.on_whistle_detected(time.time())

        committed_label_this_frame = ""
        vote_count_this_frame = ""
        engine_event_this_frame = ""
        engine_reason_this_frame = ""
        current_label = None

        if len(rolling_window) == ROLLING_WINDOW_FRAMES and frame_counter % INFERENCE_EVERY_N_FRAMES == 0:
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

            now = time.time()

            # --- Check if a PENDING team_to_serve should be finalized
            # because the confirmation delay simply elapsed with nothing
            # else taking over ---
            if pending_scoring_label is not None and (now - pending_scoring_since) >= TEAM_TO_SERVE_CONFIRM_DELAY_SECONDS:
                result = do_commit(pending_scoring_label, STREAK_NEEDED_TO_COMMIT, now)
                committed_label_this_frame = pending_scoring_label
                vote_count_this_frame = STREAK_NEEDED_TO_COMMIT
                engine_event_this_frame = result["event"]
                engine_reason_this_frame = result.get("reason", "")
                pending_scoring_label = None
                streak_label = None
                streak_count = 0
                rolling_window.clear()

            elif streak_label is not None and streak_count >= STREAK_NEEDED_TO_COMMIT:
                top_label = streak_label

                if top_label in SCORING_SIDE_MAP:
                    # team_to_serve reached its streak -- don't commit yet,
                    # go pending instead (unless already pending for this
                    # exact label, in which case just keep waiting).
                    if pending_scoring_label != top_label:
                        pending_scoring_label = top_label
                        pending_scoring_since = now
                        print(f"\n(pending: {top_label} reached streak, waiting {TEAM_TO_SERVE_CONFIRM_DELAY_SECONDS}s for confirmation...)")
                    # keep streak_count pinned at threshold so this branch
                    # doesn't re-trigger every single frame while waiting
                    streak_count = STREAK_NEEDED_TO_COMMIT

                else:
                    # A DIFFERENT (non-scoring) label reached ITS streak.
                    if pending_scoring_label is not None:
                        pending_side = SCORING_SIDE_MAP[pending_scoring_label]
                        pending_same_side_auth = f"service_authorization_{pending_side}"

                        if top_label == pending_same_side_auth:
                            # CANCEL: this was really just the continuation
                            # of the pending team_to_serve turning into a
                            # service_authorization -- discard the pending
                            # scoring gesture entirely, no point awarded.
                            print(f"\n(cancelled: {pending_scoring_label} was actually {top_label} -- no point awarded)")
                            last_decision_text = f"{pending_scoring_label} cancelled -> was {top_label}"
                            last_decision_color = (0, 165, 255)
                            last_decision_time = now
                            pending_scoring_label = None
                            streak_label = None
                            streak_count = 0
                        else:
                            # Something else entirely (a real fault gesture,
                            # or a different scoring side) -- per your own
                            # rule, this means the pending team_to_serve
                            # WAS real. Confirm it first, then this new
                            # label gets its own chance next cycle.
                            result = do_commit(pending_scoring_label, STREAK_NEEDED_TO_COMMIT, now)
                            committed_label_this_frame = pending_scoring_label
                            vote_count_this_frame = STREAK_NEEDED_TO_COMMIT
                            engine_event_this_frame = result["event"]
                            engine_reason_this_frame = result.get("reason", "")
                            pending_scoring_label = None
                            # Don't clear streak_label/streak_count here --
                            # let top_label's own streak carry over so it
                            # gets committed on ITS OWN in the next check
                            # (falls through naturally next loop iteration
                            # since streak_count is still >= threshold).

                    else:
                        # No pending team_to_serve -- normal commit, same as before.
                        is_new_gesture = (top_label != last_committed_label)
                        cooldown_passed = (now - last_commit_time) >= COMMIT_COOLDOWN_SECONDS

                        if is_new_gesture or cooldown_passed:
                            result = do_commit(top_label, streak_count, now)
                            committed_label_this_frame = top_label
                            vote_count_this_frame = streak_count
                            engine_event_this_frame = result["event"]
                            engine_reason_this_frame = result.get("reason", "")
                            streak_label = None
                            streak_count = 0
                            rolling_window.clear()

            log_writer.writerow([f"{time.time():.3f}", expected_step, current_label]
                                 + [f"{score:.4f}" for score in full_probs]
                                 + [committed_label_this_frame, vote_count_this_frame,
                                    engine_event_this_frame, engine_reason_this_frame,
                                    pending_scoring_label or "",
                                    engine.score["left"], engine.score["right"]])
            log_file.flush()

        if engine.set_over:
            winner = "LEFT" if engine.score["left"] > engine.score["right"] else "RIGHT"
            cv2.rectangle(frame, (0, 0), (frame_width, 100), (0, 100, 0), -1)
            cv2.putText(frame, f"SET OVER -- {winner} WINS {engine.score['left']}-{engine.score['right']}",
                        (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
        else:
            pending_remaining = None
            if pending_scoring_label is not None:
                pending_remaining = max(0.0, TEAM_TO_SERVE_CONFIRM_DELAY_SECONDS - (time.time() - pending_scoring_since))
            draw_instruction_banner(frame, expected_step, frame_width, pending_scoring_label, pending_remaining)
            draw_score_bar(frame, engine, frame_width, frame_height)
            draw_vote_progress(frame, streak_label, streak_count, frame_width)

        if last_probs is not None:
            draw_probability_bars(frame, last_probs, real_labels)
        draw_gesture_history_bar(frame, list(gesture_history), frame_width, frame_height)

        if last_decision_text and (time.time() - last_decision_time) < 4:
            cv2.putText(frame, f"decision_engine: {last_decision_text}", (10, frame_height - 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, last_decision_color, 2, cv2.LINE_AA)

        if time.time() < whistle_flash_until:
            whistle_text = "WHISTLE!"
            text_size = cv2.getTextSize(whistle_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 4)[0]
            text_x = (frame_width - text_size[0]) // 2
            cv2.putText(frame, whistle_text, (text_x, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 4, cv2.LINE_AA)

        cv2.putText(frame, "(click this window, then Q/ESC=quit  W=whistle  [/]=left score  -/+=right score  R=clear reason)",
                    (10, frame_height - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

        cv2.imshow("Simulated Match", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            show_skeleton = not show_skeleton
        elif key == ord('w'):
            engine.on_whistle_detected(time.time())
            whistle_flash_until = time.time() + 1.5
            print(f"  -> WHISTLE registered at {time.time():.3f}")
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
    cv2.destroyAllWindows()
    pose_model.close()
    hands_model.close()
    log_file.close()

    print(f"\nLog saved to: {log_path}")
    print(f"Final score: LEFT {engine.score['left']} - {engine.score['right']} RIGHT")
    print("Full sequence:")
    print("  ->  ".join(gesture_history) if gesture_history else "  (nothing committed)")


if __name__ == "__main__":
    main()