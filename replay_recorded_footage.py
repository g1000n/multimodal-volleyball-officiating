"""
replay_recorded_footage.py

Runs the EXACT SAME classification + streak/pending-confirmation +
decision_engine pipeline as live_deployment.py, but reading from a raw
recorded video file (from RECORD_RAW_FOOTAGE in live_deployment.py) instead
of a live camera. This lets you test a code change against REAL,
previously-observed footage -- with real continuous motion, real
transitions, real timing -- which is a stronger validation method than
isolated reference clips, since it reproduces an actual session rather
than curated individual examples.

WHY PACED, NOT INSTANT: all of live_deployment.py's timing logic (settle
window, confirmation delay, streak staleness) runs on real time.time(),
not simulated ticks. To make that logic behave identically to how it did
live, this script paces itself to the recording's own fps rather than
ripping through frames as fast as possible.

LIMITATION: this replays VIDEO only, not audio -- whistle events aren't
in the raw recording. With REQUIRE_WHISTLE_FOR_SCORING now True (see
below), press W manually at the moments a whistle actually occurred in
the original session to reproduce whistle-gated scoring -- same manual
burden as running live_deployment.py itself in manual whistle mode,
which is how every real session has actually been run so far.

--------------------------------------------------------------------
SYNC PASS #2 (this version) -- this script had drifted out of sync with
live_deployment.py AGAIN, same failure mode as before. Ported over:

1. STREAK_NEEDED_TO_COMMIT: 5 -> 3, matching live_deployment.py's
   current value. A fast, genuine team_to_serve gesture could finish
   before 5 consecutive correct inference cycles accumulated.

2. REQUIRE_WHISTLE_FOR_SCORING: False -> True. live_deployment.py's
   REAL current default is True (matches the thesis paper's whistle-
   gating objective) -- the previous "matches live_deployment.py's
   real default" comment justifying False here was itself stale, the
   exact drift this sync pass exists to catch. Since replayed footage
   has no audio, this means manually pressing W at the right moments
   is now required to reproduce scoring during replay -- this is a
   real usage change, not just a constant flip, and is called out
   explicitly in the console output at startup.

3. hands_model now uses static_image_mode=False (was True), matching
   live_deployment.py's performance/tracking-mode fix. This isn't just
   a speed difference -- static_image_mode=True re-runs full palm
   detection from scratch every frame instead of cheap frame-to-frame
   tracking, which can produce different hand-feature values than what
   the live session that this recording came from actually saw.

4. FAST REASON-STREAK TRACKING, ported in for the first time -- this
   script never had it, meaning replay was reproducing the EXACT
   starved-reason-attachment bug live_deployment.py fixed earlier:
   after a point commits, a few frames of leftover team_to_serve/
   service_authorization motion get recognized first and build a
   fresh tolerant streak on THAT label, and the tolerant DECREMENT
   logic doesn't switch to a genuinely new reason gesture (e.g.
   ball_out) for several frames, starving reason attachment past
   REASON_ATTACH_WINDOW in decision_engine.py. Fixed the same way:
   a dedicated fast, non-tolerant reason_streak_label/
   reason_streak_count, active only while expected_step ==
   "reason_gesture", pre-empting the slow/tolerant generic path.
   REASON_STREAK_NEEDED = 2, matching live_deployment.py's current
   (lowered from 3) value.

5. expected_step TRACKING, added for the first time -- prerequisite
   for fix #4 above. Mirrors live_deployment.py exactly: starts as
   "whistle", set to "reason_gesture" on point_awarded, set back to
   "whistle" on reason_attached/authorization_acknowledged. Unlike
   live_deployment.py, this script has no scoring_gesture-vs-
   authorization phase display, so expected_step here exists purely
   to gate the reason-streak fix, not for any UI purpose.

6. HONEST awaiting_whistle_confirmation HANDLING in do_commit() --
   previously this script's do_commit() only checked
   `if result["event"] != "ignored"` before appending to
   gesture_history, which wrongly treated a PENDING (unconfirmed)
   gesture the same as a real commit. Fixed to match
   live_deployment.py: awaiting_whistle_confirmation gets its own
   branch, does NOT append to gesture_history, and does NOT advance
   expected_step. This is a display/history-accuracy fix -- the
   actual score computed by decision_engine.py was already correct
   either way, since on_gesture_detected() itself never updates
   engine.score for a pending result regardless of what do_commit()
   does with the history list.

   NOTE: unlike live_deployment.py, this script has no on_whistle()
   callback surfacing a LATE whistle confirmation into gesture_history
   -- decision_engine.py's own pending_gesture/on_whistle_detected()
   mechanism still works correctly under the hood (the score updates
   correctly if a manual W arrives within the grace window), but a
   late-confirmed gesture won't retroactively appear in this script's
   on-screen history bar the way it does in live_deployment.py. This
   is a known, smaller display-only gap, not ported in this pass --
   flagged here rather than silently left unmentioned.

PREVIOUS SYNC PASS (kept, still correct):
1. FAST/TOLERANT STREAK SPLIT: live_deployment.py's parallel session
   found that a tolerant streak (added to help ball_in survive brief
   interruptions) broke cancellation -- a pending team_to_serve kept
   committing instead of being cancelled by a same-side
   service_authorization, because the tolerant decrement logic was too
   slow to switch streak_label away from the pinned scoring gesture.
   Fix: cancellation now uses its own dedicated, FAST, non-tolerant
   cancellation_streak_count, completely independent of the tolerant
   scoring streak_label/streak_count. Ported in unchanged.

2. ROLLING_WINDOW CLEARED ON CANCELLATION: previously not cleared on
   that specific commit path, causing stale probability-bar readings
   to linger. Now cleared, matching every other commit path.

3. DEBOUNCE VIA seen_different_since_last_commit, replacing the old
   pure COMMIT_COOLDOWN_SECONDS timer -- lets a genuine repeat of the
   same gesture commit again as soon as something different is seen in
   between, rather than waiting out a fixed cooldown.

5. DISPLAY_NAME / constants double-checked against the current
   live_deployment.py to catch any other drift (TEAM_TO_SERVE_CONFIRM_
   DELAY_SECONDS, etc.) -- confirmed still matching this pass.

NOTE: decision_engine.py's own changes (no-auto-stop, REASON_ATTACH_
WINDOW, etc.) apply automatically here since this script imports
DecisionEngine directly -- nothing to port for those.
--------------------------------------------------------------------

USAGE:
    python replay_recorded_footage.py data/raw_recordings/raw_1785010000.mp4

Controls (during replay):
  Q / ESC   - quit
  W         - manual whistle. With REQUIRE_WHISTLE_FOR_SCORING now
              True, this is REQUIRED to reproduce scoring -- press it
              at the moments a whistle actually occurred in the
              original session (cross-reference against that
              session's own deployment_strict_<ts>.csv Whistle
              Detections if you have it, for accurate timing).
  SPACE     - pause/resume. Resuming clears rolling_window/streak/pending
              state, same as live_deployment.py's P key, so stale
              wall-clock timestamps don't cause phantom timeouts.
  A / D     - seek back / forward 5 seconds
  J / L     - seek back / forward 30 seconds
              Seeking also clears stream state, same reason as pause.
              NOTE: seek precision on mp4 depends on nearby keyframes.
"""

import sys
import time
import csv
import os
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
from decision_engine import DecisionEngine, FAULT_REASON_GESTURES

MODEL_PATH = "models/final_model.pt"
LABEL_MAP_PATH = "models/label_map.json"
LOG_DIR = "data/live_test_logs"

# --- Constants matched to the CURRENT live_deployment.py ---
ROLLING_WINDOW_FRAMES = 24
INFERENCE_EVERY_N_FRAMES = 3
# CHANGED: 5 -> 3, matching live_deployment.py's current value. See
# module docstring's "SYNC PASS #2" item 1.
STREAK_NEEDED_TO_COMMIT = 3
FAULT_STREAK_NEEDED_TO_COMMIT = 3
CANCELLATION_STREAK_NEEDED = 3  # matches live_deployment.py's dedicated fast counter
# NEW: ported from live_deployment.py. See module docstring's "SYNC
# PASS #2" item 4 for the bug this fixes.
REASON_STREAK_NEEDED = 2
COMMIT_COOLDOWN_SECONDS = 2.0   # no longer the primary duplicate guard -- see seen_different_since_last_commit
STALE_VOTE_SECONDS = 3.0
TEAM_TO_SERVE_CONFIRM_DELAY_SECONDS = 1.5  # matches live_deployment.py's current value
MIN_POSE_DETECTED_FRACTION = 0.5
GAME_WIN_SCORE = 25
GAME_WIN_BY_MARGIN = 2

SEEK_SMALL_SECONDS = 5
SEEK_LARGE_SECONDS = 30

# CHANGED: False -> True, matching live_deployment.py's REAL current
# default (the thesis paper's whistle-gating objective). The previous
# comment here claiming False "matches live_deployment.py's real
# informational-whistle default" was itself stale -- exactly the kind
# of drift this sync pass exists to catch. See module docstring's
# "SYNC PASS #2" item 2 for what this actually changes for replay
# usage (manual W presses are now required to reproduce scoring).
REQUIRE_WHISTLE_FOR_SCORING = True

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
SCORING_SIDE_MAP = {"team_to_serve_left": "left", "team_to_serve_right": "right"}

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
    if frames_with_pose.mean() < MIN_POSE_DETECTED_FRACTION:
        return NOTHING_LABEL, 0.0, np.zeros(len(idx_to_real_label))

    normalized = ablate(normalize_sequence(raw))
    resampled = resample_sequence(normalized, SEQUENCE_LENGTH)
    x = torch.tensor(resampled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).numpy()[0]

    predicted_label, top1_label, top2_label, top1_prob, top2_prob = decide_label(probs, idx_to_real_label)
    if predicted_label != NOTHING_LABEL:
        raw_resampled = resample_sequence(raw, SEQUENCE_LENGTH)
        predicted_label = apply_tie_breaker(raw_resampled, top1_label, top2_label, top1_prob, top2_prob)
    return predicted_label, top1_prob, probs


def extract_frame_features(frame_rgb, pose_model, hands_model):
    pose_results = pose_model.process(frame_rgb)
    pose_features, pose_landmarks = extract_pose_features(pose_results)
    hand_coords, left_det, right_det, left_fingers, right_fingers = extract_hand_features(
        frame_rgb, pose_landmarks, hands_model
    )
    elbow_angles = compute_elbow_angles(pose_landmarks)
    features = np.concatenate([pose_features, hand_coords, np.array([left_det, right_det]),
                                left_fingers, right_fingers, elbow_angles])
    draw_info = {"pose_results": pose_results}
    return features, draw_info


def draw_skeleton_overlay(frame, draw_info):
    pose_results = draw_info["pose_results"]
    if pose_results.pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2),
        )


def draw_probability_bars(frame, probs, real_labels, origin_y=190):
    for i, label in enumerate(real_labels):
        score = probs[i]
        color = (0, 255, 0) if score > 0.5 else (255, 255, 255)
        pretty = DISPLAY_NAME.get(label, label)
        cv2.putText(frame, f"{pretty:<28} {score:.2f}", (10, origin_y + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (320, origin_y + i * 28 - 15),
                       (320 + int(150 * score), origin_y + i * 28 + 5), color, -1)


def draw_score_bar(frame, engine, frame_width):
    text = f"LEFT {engine.score['left']}  -  {engine.score['right']} RIGHT   [REPLAY]"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(frame, (0, 65), (frame_width, 100), (20, 20, 20), -1)
    cv2.putText(frame, text, (max(10, (frame_width - text_size[0]) // 2), 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def draw_gesture_history_bar(frame, history, frame_width, frame_height):
    pretty = [DISPLAY_NAME.get(l, l) for l in history]
    text = "  ->  ".join(pretty) if pretty else "(no gestures committed yet)"
    cv2.rectangle(frame, (0, frame_height - 60), (frame_width, frame_height - 15), (40, 40, 40), -1)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.putText(frame, text, (max(10, (frame_width - text_size[0]) // 2), frame_height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def draw_cancellation_progress(frame, cancellation_streak_count, pending_scoring_label, frame_width):
    if pending_scoring_label is None or cancellation_streak_count <= 0:
        return
    text = f"cancellation building: {cancellation_streak_count}/{CANCELLATION_STREAK_NEEDED}"
    cv2.putText(frame, text, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)


def parse_session_start_from_filename(video_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    for prefix in ("raw_", "fullwindow_"):
        if base.startswith(prefix):
            digits = base[len(prefix):]
            if digits.isdigit():
                return float(digits)
    return None


def draw_realtime_clock(frame, frame_width, original_start, elapsed_video_seconds):
    if original_start is not None:
        absolute_ts = original_start + elapsed_video_seconds
        text = f"orig T={absolute_ts:.2f}"
    else:
        text = f"video T+{elapsed_video_seconds:07.2f}s"
    cv2.putText(frame, text, (frame_width - 190, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)


def seek_relative(cap, seconds, video_fps):
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    target_frame = max(0.0, current_frame + seconds * video_fps)
    if total_frames > 0:
        target_frame = min(target_frame, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    return target_frame / video_fps if video_fps else 0.0


def main():
    if len(sys.argv) < 2:
        print("Usage: python replay_recorded_footage.py <path_to_raw_recording.mp4>")
        return

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return

    model, idx_to_real_label, real_labels = load_model()
    engine = DecisionEngine(win_score=GAME_WIN_SCORE, win_by_margin=GAME_WIN_BY_MARGIN)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open video file: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frame_interval = 1.0 / video_fps
    print(f"Replaying {video_path} at {video_fps:.1f} fps (paced to match real timing).")

    original_start = parse_session_start_from_filename(video_path)
    if original_start is not None:
        print(f"Recognized original session start ({original_start:.0f}) from filename -- "
              f"clock overlay will show the ORIGINAL absolute timestamp.")
    else:
        print("Could not parse an original session timestamp from the filename -- "
              "clock overlay will show video-relative elapsed time instead.")

    if REQUIRE_WHISTLE_FOR_SCORING:
        print("REQUIRE_WHISTLE_FOR_SCORING is True, matching live_deployment.py's real default.")
        print("This recording has NO audio -- press W manually at the moments a whistle actually")
        print("occurred in the original session to reproduce whistle-gated scoring. If you have")
        print("that session's deployment_strict_<ts>.csv, its Whistle Detections rows give you")
        print("the real timestamps to press W against.")
    else:
        print("REQUIRE_WHISTLE_FOR_SCORING is False -- whistle is informational only. "
              "Scoring proceeds freely.")
    print("Controls: SPACE=pause/resume  A/D=seek 5s  J/L=seek 30s  W=whistle  Q/ESC=quit")

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"replay_{int(time.time())}.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["timestamp", "window_predicted_label", "vote_committed_label",
                          "engine_event", "engine_reason", "score_left", "score_right"])
    print(f"Logging replay results to: {log_path}\n")

    pose_model = mp_pose.Pose(static_image_mode=False, model_complexity=0,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # CHANGED: static_image_mode=True -> False, matching
    # live_deployment.py's performance/tracking-mode fix. See module
    # docstring's "SYNC PASS #2" item 3.
    hands_model = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                  min_detection_confidence=0.1, min_tracking_confidence=0.28)

    rolling_window = deque(maxlen=ROLLING_WINDOW_FRAMES)
    streak_label, streak_count = None, 0
    last_confident_append_time = 0.0
    last_committed_label = None
    last_commit_time = 0
    seen_different_since_last_commit = True
    frame_counter = 0
    gesture_history = deque(maxlen=8)
    last_probs = None
    pending_scoring_label = None
    pending_scoring_since = 0.0
    cancellation_streak_count = 0
    # NEW: ported from live_deployment.py. See module docstring's
    # "SYNC PASS #2" items 4 and 5.
    reason_streak_label = None
    reason_streak_count = 0
    expected_step = "whistle"
    paused = False

    def do_commit(label, now):
        nonlocal last_committed_label, last_commit_time, seen_different_since_last_commit, expected_step
        result = engine.on_gesture_detected(label, now)
        # CHANGED: honest pending-state handling, matching
        # live_deployment.py. See module docstring's "SYNC PASS #2"
        # item 6 -- previously this only checked `!= "ignored"`,
        # which wrongly treated an unconfirmed, still-pending gesture
        # the same as a real commit.
        if result["event"] == "ignored":
            pass
        elif result["event"] == "awaiting_whistle_confirmation":
            pass
        else:
            gesture_history.append(label)
            if result["event"] == "point_awarded":
                expected_step = "reason_gesture"
            elif result["event"] == "reason_attached":
                expected_step = "whistle"
            elif result["event"] == "authorization_acknowledged":
                expected_step = "whistle"
        last_committed_label = label
        last_commit_time = now
        seen_different_since_last_commit = False
        return result

    while True:
        loop_start = time.time()
        if not paused:
            success, frame = cap.read()
            if not success:
                print("\nEnd of recording reached.")
                break
        frame_height, frame_width = frame.shape[:2]

        if not paused:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            features, draw_info = extract_frame_features(frame_rgb, pose_model, hands_model)
            draw_skeleton_overlay(frame, draw_info)
            rolling_window.append(features)
            frame_counter += 1

            if not REQUIRE_WHISTLE_FOR_SCORING:
                engine.on_whistle_detected(time.time())

            if len(rolling_window) == ROLLING_WINDOW_FRAMES and frame_counter % INFERENCE_EVERY_N_FRAMES == 0:
                current_label, current_conf, full_probs = classify_window(list(rolling_window), model, idx_to_real_label)
                last_probs = full_probs

                if current_label != last_committed_label:
                    seen_different_since_last_commit = True

                if pending_scoring_label is not None:
                    pending_side = SCORING_SIDE_MAP[pending_scoring_label]
                    pending_same_side_auth = f"service_authorization_{pending_side}"
                    if current_label == pending_same_side_auth:
                        cancellation_streak_count += 1
                    else:
                        cancellation_streak_count = 0
                else:
                    cancellation_streak_count = 0

                # NEW: fast reason-streak tracking, ported from
                # live_deployment.py. See module docstring's "SYNC
                # PASS #2" item 4.
                if expected_step == "reason_gesture":
                    if current_label in FAULT_REASON_GESTURES or current_label == "end_of_set":
                        if current_label == reason_streak_label:
                            reason_streak_count += 1
                        else:
                            reason_streak_label = current_label
                            reason_streak_count = 1
                    else:
                        reason_streak_label = None
                        reason_streak_count = 0
                else:
                    reason_streak_label = None
                    reason_streak_count = 0

                if current_label != NOTHING_LABEL:
                    if current_label == streak_label:
                        streak_count += 1
                    else:
                        if streak_label is not None and streak_count > 1:
                            streak_count -= 1
                        else:
                            streak_label, streak_count = current_label, 1
                    last_confident_append_time = time.time()
                elif streak_label is not None and (time.time() - last_confident_append_time) > STALE_VOTE_SECONDS:
                    streak_label, streak_count = None, 0

                now = time.time()
                committed_label_this_frame = ""
                engine_event_this_frame = ""
                engine_reason_this_frame = ""

                cancel_check_triggered = False
                if pending_scoring_label is not None and cancellation_streak_count >= CANCELLATION_STREAK_NEEDED:
                    pending_side = SCORING_SIDE_MAP[pending_scoring_label]
                    pending_same_side_auth = f"service_authorization_{pending_side}"

                    result = do_commit(pending_same_side_auth, now)
                    committed_label_this_frame = pending_same_side_auth
                    engine_event_this_frame = result["event"]
                    engine_reason_this_frame = result.get("reason", "")
                    pending_scoring_label = None
                    streak_label, streak_count = None, 0
                    cancellation_streak_count = 0
                    rolling_window.clear()
                    cancel_check_triggered = True

                # NEW: fast reason commit, ported from
                # live_deployment.py. See module docstring's "SYNC
                # PASS #2" item 4.
                reason_check_triggered = False
                if (not cancel_check_triggered and expected_step == "reason_gesture"
                        and reason_streak_label is not None and reason_streak_count >= REASON_STREAK_NEEDED):
                    result = do_commit(reason_streak_label, now)
                    committed_label_this_frame = reason_streak_label
                    engine_event_this_frame = result["event"]
                    engine_reason_this_frame = result.get("reason", "")
                    reason_streak_label = None
                    reason_streak_count = 0
                    streak_label, streak_count = None, 0
                    rolling_window.clear()
                    reason_check_triggered = True

                if cancel_check_triggered or reason_check_triggered:
                    pass

                elif pending_scoring_label is not None and (now - pending_scoring_since) >= TEAM_TO_SERVE_CONFIRM_DELAY_SECONDS:
                    result = do_commit(pending_scoring_label, now)
                    committed_label_this_frame = pending_scoring_label
                    engine_event_this_frame = result["event"]
                    engine_reason_this_frame = result.get("reason", "")
                    pending_scoring_label = None
                    streak_label, streak_count = None, 0
                    cancellation_streak_count = 0
                    rolling_window.clear()
                    engine.last_settle_start_time = None

                elif streak_label is not None and streak_count >= (STREAK_NEEDED_TO_COMMIT if streak_label in SCORING_SIDE_MAP else FAULT_STREAK_NEEDED_TO_COMMIT):
                    top_label = streak_label

                    if top_label in SCORING_SIDE_MAP:
                        if pending_scoring_label != top_label:
                            pending_scoring_label = top_label
                            pending_scoring_since = now
                            cancellation_streak_count = 0
                        streak_count = STREAK_NEEDED_TO_COMMIT
                    else:
                        if pending_scoring_label is not None:
                            pending_side = SCORING_SIDE_MAP[pending_scoring_label]
                            pending_same_side_auth = f"service_authorization_{pending_side}"
                            if top_label == pending_same_side_auth:
                                result = do_commit(top_label, now)
                                committed_label_this_frame = top_label
                                engine_event_this_frame = result["event"]
                                engine_reason_this_frame = result.get("reason", "")
                                pending_scoring_label = None
                                streak_label, streak_count = None, 0
                                cancellation_streak_count = 0
                                rolling_window.clear()
                            else:
                                result = do_commit(pending_scoring_label, now)
                                pending_scoring_label = None
                                cancellation_streak_count = 0
                                engine.last_settle_start_time = None
                                fault_result = do_commit(top_label, now)
                                committed_label_this_frame = top_label
                                engine_event_this_frame = fault_result["event"]
                                engine_reason_this_frame = fault_result.get("reason", "")
                                streak_label, streak_count = None, 0
                                rolling_window.clear()
                        else:
                            is_new = (top_label != last_committed_label)
                            repeat_allowed = seen_different_since_last_commit
                            if is_new or repeat_allowed:
                                result = do_commit(top_label, now)
                                committed_label_this_frame = top_label
                                engine_event_this_frame = result["event"]
                                engine_reason_this_frame = result.get("reason", "")
                                streak_label, streak_count = None, 0
                                rolling_window.clear()

                log_writer.writerow([f"{time.time():.3f}", current_label, committed_label_this_frame,
                                      engine_event_this_frame, engine_reason_this_frame,
                                      engine.score["left"], engine.score["right"]])
                log_file.flush()

        draw_score_bar(frame, engine, frame_width)
        if last_probs is not None:
            draw_probability_bars(frame, last_probs, real_labels)
        draw_gesture_history_bar(frame, list(gesture_history), frame_width, frame_height)
        draw_cancellation_progress(frame, cancellation_streak_count, pending_scoring_label, frame_width)
        draw_realtime_clock(frame, frame_width, original_start, frame_counter / video_fps)

        cv2.imshow("REPLAY", frame)

        elapsed = time.time() - loop_start
        wait_ms = max(1, int((frame_interval - elapsed) * 1000))
        key = cv2.waitKey(wait_ms) & 0xFF

        def clear_stream_state(reason):
            nonlocal rolling_window, streak_label, streak_count, pending_scoring_label, frame_counter, cancellation_streak_count
            nonlocal reason_streak_label, reason_streak_count
            rolling_window.clear()
            streak_label, streak_count = None, 0
            pending_scoring_label = None
            cancellation_streak_count = 0
            # NEW: also reset the reason streak on pause/seek, matching
            # live_deployment.py's P-key handler.
            reason_streak_label = None
            reason_streak_count = 0
            frame_counter = 0
            print(f"  -> cleared streak/rolling-window state ({reason})")

        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            paused = not paused
            if not paused:
                clear_stream_state("resumed from pause")
        elif key == ord('w'):
            engine.on_whistle_detected(time.time())
            print(f"  -> manual whistle at {time.time():.3f}")
        elif key in (ord('a'), ord('d'), ord('j'), ord('l')):
            seconds = {
                ord('a'): -SEEK_SMALL_SECONDS,
                ord('d'): SEEK_SMALL_SECONDS,
                ord('j'): -SEEK_LARGE_SECONDS,
                ord('l'): SEEK_LARGE_SECONDS,
            }[key]
            new_pos_seconds = seek_relative(cap, seconds, video_fps)
            clear_stream_state(f"seeked to ~{new_pos_seconds:.1f}s")

    cap.release()
    cv2.destroyAllWindows()
    pose_model.close()
    hands_model.close()
    log_file.close()

    print(f"\nReplay log saved to: {log_path}")
    print(f"Final replay score: LEFT {engine.score['left']} - {engine.score['right']} RIGHT")


if __name__ == "__main__":
    main()