"""
simulate_game_test.py

Full end-to-end simulated match, using the REAL trained model and the
REAL decision_engine -- not a scripted replay. This is the test you
haven't done yet: everything up to now has been isolated gestures or
short transitions. This runs a whole simulated SET, point by point,
following the actual FIVB sequence, so you can verify the whole system
-- detection, sequencing, scoring, win-condition -- works together
across a realistic number of points.

HOW IT WORKS:
  This script does NOT decide who wins each point. YOU decide, by
  which gesture you actually perform (team_to_serve_LEFT vs RIGHT).
  The script's only job is to guide you through the correct sequence at
  each step and display the real, live score as tracked by
  decision_engine -- exactly like being coached through a real match:

    1. Screen prompts: "Press W for whistle"
    2. You press W (whistle detection isn't wired up yet -- this is a
       manual stand-in until the audio team's detector exists)
    3. Screen prompts: "Perform team_to_serve_LEFT or team_to_serve_RIGHT
       (your choice -- whichever side you want to award the point to)"
    4. You perform it. If accepted, score updates for real via
       decision_engine.
    5. Screen prompts: "Perform a fault/reason gesture (your choice:
       ball_out, double_contact, service_authorization_left/right)"
    6. You perform it. If accepted, it's attached as the reason.
    7. Repeat from step 1.

  The set ends when a side reaches WIN_SCORE with a lead of
  WIN_BY_MARGIN (default: first to 7, win by 2 -- shortened from the
  real 25 specifically for practical end-to-end testing time).

WHISTLE STATUS: real whistle detection isn't available yet (separate
audio team, not yet delivered) -- W stays the manual placeholder until
then. Swap the W-key handler for the real detector's callback later;
everything else in decision_engine.py stays the same either way.

Controls:
  W - simulate whistle (temporary, until real detection exists)
  Q / ESC - quit
  S - toggle skeleton overlay
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
# NOTE: ScoreboardGUI (separate Tkinter window) REMOVED -- having two
# separate windows meant keyboard focus could be on either one, and
# cv2.waitKey() only sees keys typed while ITS window has focus. This
# was very likely why W (and earlier, Q) intermittently failed to
# register. Score is now drawn directly on the same camera window via
# draw_score_bar() below -- one window, no more focus-switching.

MODEL_PATH = "models/final_model.pt"
LABEL_MAP_PATH = "models/label_map.json"
CAMERA_INDEX = 1
LOG_DIR = "data/live_test_logs"

ROLLING_WINDOW_FRAMES = 24  # CHANGED from 60 -- was sized assuming ~30fps (~2s), but measured
# real live throughput is only ~9.7-10.7fps, meaning 60 frames was actually
# spanning ~5.6-6s of real time, far longer than an actual gesture takes.
# 24 frames at ~10fps ~= 2.4s, proportionate to real gesture duration --
# also means a post-clear "cold refill" (see rolling_window.clear() above)
# only takes ~2.4s instead of ~6s before classification can resume.
INFERENCE_EVERY_N_FRAMES = 3
# REPLACED (see main() for the new streak-based commit logic): the old
# VOTE_WINDOW_SIZE/VOTES_NEEDED_TO_COMMIT shared-deque majority vote had
# a real structural flaw -- it tracked ONE mixed history across ALL
# labels, so two DIFFERENT real gestures performed close together
# competed for the same slots. A short-but-genuine team_to_serve_right
# (5 confident hits) got silently erased by a longer ball_out (7 hits)
# that followed it -- team_to_serve_right never committed at all,
# confirmed directly from a live session log. Streak-based commit
# fixes this structurally: each label gets its own count that resets
# to zero the instant a DIFFERENT label appears, so nothing a later
# gesture does can erase an earlier one's progress.
STREAK_NEEDED_TO_COMMIT = 5  # consecutive confident hits of the SAME label needed to commit
COMMIT_COOLDOWN_SECONDS = 2.0  # CHANGED from 1.0 -- must stay ABOVE SETTLE_WINDOW_SECONDS (1.5)
# in decision_engine.py. If cooldown were shorter than the settle window,
# a continuously-held scoring gesture could re-trigger a second point the
# instant the settle window closes, even though nothing new was performed --
# a real edge case, not yet confirmed to have happened, but worth closing.

# Shortened win condition for practical end-to-end testing --
# real games use 25/win-by-2; this is deliberately smaller so a full
# simulated set finishes in a reasonable test session.
TEST_WIN_SCORE = 7
TEST_WIN_BY_MARGIN = 2

# NEW: set True to test gesture detection + scoring/sequencing WITHOUT
# needing to press W first each time -- the engine is fed a continuous
# "fresh" whistle every frame, so the "no recent whistle" gate never
# blocks a scoring gesture. Whistle-gating itself still fully exists in
# decision_engine.py and is unaffected -- this is purely a testing
# convenience in THIS script, not a change to the engine's real logic.
# Set back to False once you want to test the whistle-gated flow again.
DISABLE_WHISTLE_REQUIREMENT = True

# How long a confident vote entry can sit unrefreshed before a brand-new
# confident entry is treated as "starting fresh" rather than combining
# with it -- prevents very old entries (from a completely different
# moment) from silently voting alongside a new one, now that "nothing"
# no longer naturally flushes the buffer over time.
STALE_VOTE_SECONDS = 3.0

# Pretty display names for the sequence bar, mirroring MaxLSB's
# action_fullname() -- e.g. "Point Right -> Out of Bounds -> Substitution"
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


MIN_POSE_DETECTED_FRACTION = 0.5  # if fewer than half the window's frames have a
# detected pose, treat the whole window as "nothing" regardless of what the
# model says -- an all-zero (no-pose) input is a real point in feature space
# the model was never taught to associate with "no signal," and can be
# confidently misclassified as a real gesture (observed live: a blank/
# not-yet-connected camera feed got read as double_contact).


def classify_window(raw_window_frames, model, idx_to_real_label):
    raw = np.array(raw_window_frames)

    # SAFETY GATE: pose features are the first POSE_FEATURES columns of the
    # RAW (pre-ablation) array, and extract_pose_features() returns all
    # zeros when no pose was detected that frame -- checking for that
    # directly, cheaply, without needing a separate tracked flag.
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


def draw_instruction_banner(frame, expected_step, frame_width):
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
    """
    Shows the CURRENT per-label streak, so it's visible that a green
    probability bar is only a single window's raw guess -- not a
    commit. A commit only happens once the SAME label hits
    STREAK_NEEDED_TO_COMMIT confident windows IN A ROW -- a different
    label appearing resets this to zero, so nothing a later gesture
    does can silently erase an earlier one's progress.
    """
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
                            "score_left", "score_right"])
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
    streak_label = None    # the label currently building a consecutive streak
    streak_count = 0        # how many CONSECUTIVE confident hits of streak_label so far
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

    expected_step = "scoring_gesture" if DISABLE_WHISTLE_REQUIREMENT else "whistle"

    print("Guided simulated match running. Follow the on-screen instruction. Press Q/ESC to quit.\n")

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

        # NEW: keep the engine's whistle "fresh" every frame when the
        # requirement is disabled, so the "no recent whistle" gate in
        # decision_engine.py never blocks a scoring gesture during
        # this test mode.
        if DISABLE_WHISTLE_REQUIREMENT:
            engine.on_whistle_detected(time.time())

        if len(rolling_window) == ROLLING_WINDOW_FRAMES and frame_counter % INFERENCE_EVERY_N_FRAMES == 0:
            current_label, current_conf, full_probs = classify_window(list(rolling_window), model, idx_to_real_label)
            last_probs = full_probs

            # NEW: per-label STREAK instead of a shared mixed-label vote
            # deque. "nothing" doesn't touch the streak at all (same
            # spirit as the earlier fix -- an ambiguous moment can
            # neither help nor hurt). Any DIFFERENT real label resets
            # the streak to 1 for itself -- this is the actual fix for
            # the diagnosed bug: a short-but-real team_to_serve_right
            # streak can no longer be silently absorbed/overwritten by
            # a longer ball_out streak that follows it, because they no
            # longer share the same counter at all.
            if current_label != NOTHING_LABEL:
                if current_label == streak_label:
                    streak_count += 1
                else:
                    streak_label = current_label
                    streak_count = 1
                last_confident_append_time = time.time()
            # STALENESS GUARD: if it's been a while since the last
            # confident hit, don't let a stale streak silently resume --
            # start fresh.
            elif streak_label is not None and (time.time() - last_confident_append_time) > STALE_VOTE_SECONDS:
                streak_label = None
                streak_count = 0

            committed_label_this_frame = ""
            vote_count_this_frame = ""
            engine_event_this_frame = ""
            engine_reason_this_frame = ""

            if streak_label is not None and streak_count >= STREAK_NEEDED_TO_COMMIT:
                top_label = streak_label
                commit_now = time.time()
                is_new_gesture = (top_label != last_committed_label)
                cooldown_passed = (commit_now - last_commit_time) >= COMMIT_COOLDOWN_SECONDS

                if is_new_gesture or cooldown_passed:
                    result = engine.on_gesture_detected(top_label, commit_now)
                    committed_label_this_frame = top_label
                    vote_count_this_frame = streak_count
                    engine_event_this_frame = result["event"]
                    engine_reason_this_frame = result.get("reason", "")

                    last_decision_text = f"{top_label}: {result['event']}"
                    if result["event"] == "ignored":
                        last_decision_text += f" ({result['reason']})"
                        last_decision_color = (0, 0, 255)
                    else:
                        gesture_history.append(top_label)
                        last_decision_color = (0, 255, 0)

                        if result["event"] == "point_awarded":
                            expected_step = "reason_gesture"
                        elif result["event"] == "reason_attached":
                            expected_step = "scoring_gesture" if DISABLE_WHISTLE_REQUIREMENT else "whistle"

                    last_decision_time = commit_now
                    last_committed_label = top_label
                    last_commit_time = commit_now
                    streak_label = None
                    streak_count = 0
                    # rolling_window is now cleared on every commit -- forces
                    # a clean slate, no leftover frames of the JUST-FINISHED
                    # gesture lingering in the buffer.
                    rolling_window.clear()

            log_writer.writerow([f"{time.time():.3f}", expected_step, current_label]
                                 + [f"{score:.4f}" for score in full_probs]
                                 + [committed_label_this_frame, vote_count_this_frame,
                                    engine_event_this_frame, engine_reason_this_frame,
                                    engine.score["left"], engine.score["right"]])
            log_file.flush()

        if engine.set_over:
            winner = "LEFT" if engine.score["left"] > engine.score["right"] else "RIGHT"
            cv2.rectangle(frame, (0, 0), (frame_width, 100), (0, 100, 0), -1)
            cv2.putText(frame, f"SET OVER -- {winner} WINS {engine.score['left']}-{engine.score['right']}",
                        (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
        else:
            draw_instruction_banner(frame, expected_step, frame_width)
            draw_score_bar(frame, engine, frame_width, frame_height)
            draw_vote_progress(frame, streak_label, streak_count, frame_width)

        if last_probs is not None:
            draw_probability_bars(frame, last_probs, real_labels)
        draw_gesture_history_bar(frame, list(gesture_history), frame_width, frame_height)

        if last_decision_text and (time.time() - last_decision_time) < 4:
            cv2.putText(frame, f"decision_engine: {last_decision_text}", (10, frame_height - 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, last_decision_color, 2, cv2.LINE_AA)

        # MUCH more obvious whistle acknowledgment than before -- big,
        # centered, longer duration, since "the W doesn't work" turned
        # out to partly be that the old feedback was too subtle to
        # notice even when the press DID register.
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
            print(f"  -> WHISTLE registered at {time.time():.3f}")  # console confirmation too, in case the on-screen flash is missed
            if expected_step == "whistle":
                expected_step = "scoring_gesture"
        # --- MANUAL SCORE OVERRIDE -- human-operator safety net for
        # correcting a mistake live, without needing to fully restart ---
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