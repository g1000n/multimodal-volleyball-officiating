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
KNOWN SNAPSHOT (this file): this is the state right after Anouchska's UI
redesign was merged in, BEFORE the reason-streak fix and the
do_commit()/on_whistle() confirmation-handling fixes that came after it.

--------------------------------------------------------------------
FIXES RESTORED (this pass): the two known bugs documented in the previous
snapshot are fixed again, since the user now wants the bottom gesture-
history bar to accurately reflect real commits, and this directly needs
both:
  1. FAST REASON-STREAK TRACKING: restored -- a dedicated fast,
     non-tolerant reason_streak_label/reason_streak_count, active only
     while expected_step == "reason_gesture", so a real fault/reason
     gesture after a point doesn't get starved by the tolerant streak
     resisting the switch away from whatever was recognized right
     before the point committed.
  2. do_commit()/on_whistle() HONEST "awaiting_whistle_confirmation"
     HANDLING: restored -- do_commit() no longer treats a pending
     (unconfirmed) gesture as a success; it gets its own amber/pending
     state and does NOT append to gesture_history. on_whistle() now
     properly surfaces a genuine late-whistle confirmation into
     gesture_history/the decision chip/the main CSV log, instead of
     silently discarding decision_engine's confirmation_result.
Together these mean the bottom bar (gesture_history) now only ever
shows things that ACTUALLY committed -- pending/unconfirmed gestures
never appear there, matching what was asked for.

ALSO THIS PASS: generate_html_report() expanded to include a full,
unfiltered recognition log (every window_predicted_label the model
ever saw, not just the rows where something committed), alongside the
existing commit timeline -- see that function's own docstring.

--------------------------------------------------------------------
SYNC PASS (earlier version) -- brings this in line with the new
decision_engine.py (two-phase service_authorization design + the
left/right scoring fix):

1. CANCELLATION NOW COMMITS THE AUTHORIZATION.
2. do_commit()'s expected_step LOGIC now handles the new
   "authorization_acknowledged" event.
3. DISPLAY_NAME flipped (and ball_in/ball_touched added).

FIXES (earlier session): probability bars use DISPLAY_NAME, duplicate
service_authorization commit fix (idle-since-last-commit guard),
tolerant streak counter, dedicated fast non-tolerant cancellation
counter, rolling_window cleared on every commit path (including
cancellation, which used to leak stale frames and cause readings to
visibly "linger" after a gesture had already ended).

--------------------------------------------------------------------
VISUAL REDESIGN (this version): BACKSTAGE and SCOREBOARD windows
rebuilt for readability and a cleaner, more organized look.

WHAT CHANGED, and why:
- Every text element now sits on a translucent dark panel
  (draw_panel(), alpha-blended via cv2.addWeighted) instead of being
  drawn directly on top of the raw camera feed. This was the actual
  fix for "hard to see depending on background" -- text contrast
  used to depend on whatever happened to be behind the camera subject
  at that moment; now it's consistent regardless of background.
- The gesture-history "sequence" bar at the bottom (the one flagged as
  too small to read) is now rendered as separated pill/chip shapes
  with larger text (was scale 0.6/thickness 2, now scale 0.72/
  thickness 2 on a taller, dedicated panel), and alternates two shades
  so consecutive gestures are visually distinguishable even when the
  displayed names are similar.
- Layout reorganized into clear zones instead of scattered absolute
  y-coordinates: TOP status strip -> SCORE strip -> LEFT confidence
  panel -> BOTTOM decision/history stack -> corner readouts. Each zone
  is a single cohesive panel instead of several independent floating
  text calls.
- Consistent color palette (see COLORS dict below) used everywhere,
  instead of one-off color tuples scattered through each draw
  function -- makes the whole UI read as one designed system rather
  than accumulated patches.
- Top prediction in the confidence panel is now visually highlighted
  (brighter row + leading accent bar), so the current best guess is
  immediately obvious at a glance instead of requiring you to read
  every row.

RECONCILED (this pass): the AUDIO/TIMESTAMP HOOK left in the corner-
readout panel and key-handler section has been resolved. The TIMESTAMP
half is filled -- live Philippine time is now drawn in the corner
readout (see get_ph_time_str()) and logged as a `ph_time` column in
both CSVs. The AUDIO half ("A = reconnect audio") is deliberately NOT
reintroduced: audio recording/muxing was removed from this pipeline
entirely in an earlier cleanup pass (it was causing real, hard-to-debug
video truncation and sync issues) and this deployment script does not
record audio at all -- raw_<ts>.mp4 and fullwindow_<ts>.mp4 are both
silent, same as before this redesign.

ALSO THIS PASS: REQUIRE_WHISTLE_FOR_SCORING flipped True -- the thesis
paper's Objective #3 / Conceptual Framework / Research Procedure all
commit to whistle detection as a hard validation gate, not an
informational-only signal, so the code now matches that. See the
constant's own comment below for what this actually changes at
runtime, and the WHISTLE_DEVICE_INDEX comment for the blocker this
creates until a real device index is set.

ALSO THIS PASS: added an informational-vs-strict HTML report,
generated at session end from the two CSVs this script already
writes -- see generate_html_report()'s own docstring. With
REQUIRE_WHISTLE_FOR_SCORING now True, `engine` and `strict_engine`
are functionally identical (both require real whistles the same way),
so this report will show near-zero divergence between them going
forward -- it's kept mainly as a session log / sanity-check rather
than a live A/B comparison now that the whistle-gating decision itself
is settled.

--------------------------------------------------------------------
PERFORMANCE FIXES (carried forward from a later pass, kept in this
snapshot since they're unrelated to the reason-streak/confirmation
bugs described above):
- hands_model uses static_image_mode=False (was True) -- real live
  testing showed lag specifically when hands became visible, root-
  caused to static_image_mode=True forcing full palm-detection from
  scratch every frame instead of cheap frame-to-frame tracking.
- draw_panel() slices to its own bounding box before copying/blending
  instead of doing a full-frame copy + full-frame cv2.addWeighted()
  blend on every call -- benchmarked 5.2x speedup on panel-drawing
  alone at 1080p.

Controls:
  W - manual whistle (only matters if real whistle detection isn't active)
  Q / ESC - quit
  S - toggle skeleton overlay
  F - toggle SCOREBOARD window fullscreen (for projector/audience display)
  P - pause/resume (nothing gets processed while paused)
  [ / ] - manually adjust LEFT score down/up (mistake-correction safety net)
  - / + - manually adjust RIGHT score down/up
  R - manually clear the last-attached reason for the current point
"""

import time
import datetime
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
from decision_engine import DecisionEngine, FAULT_REASON_GESTURES

MODEL_PATH = "models/final_model.pt"
LABEL_MAP_PATH = "models/label_map.json"
# CHANGED: kept at 1 (Gion's confirmed dual-camera/Camo index), not the 0
# this redesign shipped with -- that 0 was Anouchska's own machine's
# default webcam while building the UI, not a deliberate change to the
# capture setup. Reset this back to 0 (or whatever's correct) if running
# on a different machine.
CAMERA_INDEX = 1
LOG_DIR = "data/live_test_logs"

# ============================================================
# TUNABLE SETTINGS
# ============================================================

ROLLING_WINDOW_FRAMES = 24
INFERENCE_EVERY_N_FRAMES = 3
STREAK_NEEDED_TO_COMMIT = 5
FAULT_STREAK_NEEDED_TO_COMMIT = 3
CANCELLATION_STREAK_NEEDED = 3
# RESTORED: fast, non-tolerant streak requirement for fault/reason
# gestures specifically while waiting to attach a reason to an
# already-awarded point (expected_step == "reason_gesture"). See main
# loop's own comment for the real bug this fixes -- the generic
# tolerant streak_label/streak_count resists switching to a genuinely
# new gesture for several frames after a real gesture was just
# committed, which was silently starving reason attachment past
# REASON_ATTACH_WINDOW (decision_engine.py) in real testing.
REASON_STREAK_NEEDED = 3
COMMIT_COOLDOWN_SECONDS = 2.0
STALE_VOTE_SECONDS = 3.0
TEAM_TO_SERVE_CONFIRM_DELAY_SECONDS = 1.5
MIN_POSE_DETECTED_FRACTION = 0.5
HISTORY_CLEAR_AFTER_IDLE_SECONDS = 6.0
CONSOLE_REFRESH_AFTER_IDLE_SECONDS = 8.0

GAME_WIN_SCORE = 25
GAME_WIN_BY_MARGIN = 2

FAST_HAND_CROP_MODE = True

RECORD_RAW_FOOTAGE = True  # TUNABLE
RECORDINGS_DIR = "data/raw_recordings"
RAW_RECORD_FPS = 10  # TUNABLE. Matches your measured real live throughput.

RECORD_FULL_WINDOW_FOOTAGE = True  # TUNABLE
FULL_WINDOW_RECORD_FPS = RAW_RECORD_FPS

WHISTLE_DEVICE_INDEX = None  # TUNABLE -- still unset. With
# REQUIRE_WHISTLE_FOR_SCORING now True below, this is a hard blocker, not
# just a nice-to-have: with no real detector, every point needs a manual
# W press to register at all. Set to your confirmed-working index, e.g. 3.

# CHANGED: flipped to True to match the thesis paper's Objective #3 /
# Conceptual Framework / Research Procedure, all three of which commit to
# whistle detection as a HARD VALIDATION GATE ("a referee call will be
# validated only when both a recognized gesture and a detected whistle
# occur within a defined temporal window") -- not an informational-only
# signal. With this True, the informational `engine` now behaves exactly
# like `strict_engine` always has: it stops auto-refilling last_whistle_time
# every frame (see the `if not REQUIRE_WHISTLE_FOR_SCORING:` block in the
# main loop below) and instead only accepts REAL whistle detections,
# gated through decision_engine.py's own TEMPORAL_WINDOW /
# WHISTLE_CONFIRMATION_GRACE_SECONDS tolerance for ordering.
# Do not run a real session with this True until WHISTLE_DEVICE_INDEX is
# set and whistle detection has been validated during actual gesture
# motion, not just standing still.
REQUIRE_WHISTLE_FOR_SCORING = True  # TUNABLE

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

extract_keypoints.LIVE_FAST_MODE = FAST_HAND_CROP_MODE

# NEW: Philippine time, for the corner readout (see AUDIO/TIMESTAMP HOOK
# below -- this fills the TIMESTAMP half of that hook; the AUDIO half is
# deliberately not reintroduced here, since audio recording/muxing was
# removed entirely in an earlier cleanup pass and isn't part of this
# pipeline) and for the CSV logs, so a moment in a session can be
# cross-referenced against real-world notes/events afterward instead of
# only having a raw epoch timestamp. Fixed UTC+8 -- the Philippines does
# not observe daylight saving, so no zoneinfo/tzdata dependency is needed.
PH_TIMEZONE = datetime.timezone(datetime.timedelta(hours=8))


def get_ph_time_str(fmt="%Y-%m-%d %H:%M:%S"):
    """Current Philippine time as a formatted string, e.g. '2026-08-16 14:32:07'."""
    return datetime.datetime.now(PH_TIMEZONE).strftime(fmt)


# ============================================================
# VISUAL DESIGN SYSTEM -- one shared palette + drawing helpers,
# used by every draw_* function below instead of one-off color
# tuples scattered per function. All colors are BGR (OpenCV order).
# ============================================================

COLORS = {
    "panel_bg":       (28, 24, 22),     # dark warm slate, used for all translucent panels
    "panel_bg_alt":    (38, 33, 30),    # slightly lighter, for alternating rows/chips
    "panel_border":   (70, 62, 56),
    "text_primary":   (245, 245, 245),
    "text_muted":     (175, 172, 168),
    "accent_green":   (110, 220, 120),  # confident / success / positive
    "accent_amber":   (60, 175, 250),   # pending / building / caution
    "accent_red":     (75, 80, 235),    # ignored / rejected
    "accent_blue":    (235, 178, 90),   # whistle / informational
    "left_team":      (235, 178, 90),   # cool blue -- LEFT side accent
    "right_team":     (90, 140, 250),   # warm orange -- RIGHT side accent
    "highlight_row":  (55, 48, 44),
}

PANEL_ALPHA = 0.72  # translucency for all backing panels -- camera stays
# faintly visible behind text, but contrast is consistent regardless of
# what's actually behind the subject at any given moment.


def draw_panel(frame, x1, y1, x2, y2, color=None, alpha=PANEL_ALPHA, border=False):
    """Draws a translucent filled rectangle behind text, so readability
    never depends on the live camera background. This is the core fix
    for the original 'hard to see on some backgrounds' complaint --
    used behind every text element in the redesign, not just some.

    PERFORMANCE FIX: previously did frame.copy() (a full-frame
    allocation) plus a full-frame cv2.addWeighted() blend for EVERY
    call, even for a small panel like a corner readout. With ~7 panels
    drawn every frame, that's 7 full-frame copies + 7 full-frame blends
    per frame on top of the per-frame MediaPipe inference and window
    letterboxing -- confirmed via real live testing to be a real cause
    of the live view feeling like slow motion after this redesign
    (measured_fps genuinely dropped; nothing about the camera or model
    changed). Slicing to just this panel's own bounding box before
    copying/blending cuts the per-call cost from O(full frame) to
    O(panel area) -- typically a large win, since most panels only
    cover a small fraction of the frame. Visually identical to before:
    the old approach's addWeighted blended the WHOLE frame against an
    overlay that only differed from the original inside the rectangle,
    so outside the rectangle it was blending frame-against-itself (a
    no-op) anyway -- this just skips doing that no-op work explicitly.
    """
    color = color or COLORS["panel_bg"]
    h, w = frame.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    if x2c <= x1c or y2c <= y1c:
        return
    region = frame[y1c:y2c, x1c:x2c]
    overlay = np.empty_like(region)
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, region, 1 - alpha, 0, region)
    if border:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["panel_border"], 1, cv2.LINE_AA)


def draw_text(frame, text, pos, scale=0.55, color=None, thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
    color = color or COLORS["text_primary"]
    cv2.putText(frame, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def text_width(text, scale=0.55, thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
    return cv2.getTextSize(text, font, scale, thickness)[0][0]


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


# ------------------------------------------------------------------
# ZONE 1 -- TOP STATUS STRIP (instruction banner + whistle mode)
# ------------------------------------------------------------------

TOP_STRIP_HEIGHT = 56


def draw_top_status_strip(frame, expected_step, frame_width, pending_label=None,
                           pending_remaining=None, whistle_mode="manual", cancellation_progress=None):
    draw_panel(frame, 0, 0, frame_width, TOP_STRIP_HEIGHT)

    if whistle_mode == "auto" and not REQUIRE_WHISTLE_FOR_SCORING:
        mode_text, mode_color = "WHISTLE (info only)", COLORS["accent_amber"]
    elif whistle_mode == "auto":
        mode_text, mode_color = "AUTO WHISTLE", COLORS["accent_green"]
    else:
        mode_text, mode_color = "MANUAL WHISTLE (W)", COLORS["accent_amber"]

    if pending_label is not None:
        side = SCORING_SIDE_MAP[pending_label]
        status_text = f"TEAM_TO_SERVE_{side.upper()} PENDING -- confirming in {pending_remaining:.1f}s"
        status_color = COLORS["accent_amber"]
    else:
        messages = {
            "whistle": ("Waiting for WHISTLE", COLORS["accent_blue"]),
            "scoring_gesture": ("Waiting for TEAM_TO_SERVE (Left/Right)", COLORS["accent_green"]),
            "reason_gesture": ("Waiting for fault/reason gesture", COLORS["text_muted"]),
        }
        status_text, status_color = messages[expected_step]

    cv2.circle(frame, (22, TOP_STRIP_HEIGHT // 2), 6, status_color, -1, cv2.LINE_AA)
    draw_text(frame, status_text, (38, TOP_STRIP_HEIGHT // 2 + 6), scale=0.62, color=status_color, thickness=2)

    mode_w = text_width(mode_text, scale=0.5, thickness=2)
    draw_text(frame, mode_text, (frame_width - mode_w - 18, TOP_STRIP_HEIGHT // 2 + 5),
              scale=0.5, color=mode_color, thickness=2)

    if pending_label is not None and cancellation_progress is not None and cancellation_progress[0] > 0:
        count, needed = cancellation_progress
        cancel_text = f"cancellation building: {count}/{needed}"
        cancel_w = text_width(cancel_text, scale=0.42, thickness=1)
        draw_text(frame, cancel_text, (frame_width - mode_w - cancel_w - 40, TOP_STRIP_HEIGHT // 2 + 5),
                  scale=0.42, color=COLORS["accent_amber"], thickness=1)


# ------------------------------------------------------------------
# ZONE 2 -- SCORE STRIP (main score + secondary/strict readout)
# ------------------------------------------------------------------

SCORE_STRIP_Y1 = TOP_STRIP_HEIGHT
SCORE_STRIP_HEIGHT = 58


def draw_score_strip(frame, engine, strict_engine, frame_width):
    y1 = SCORE_STRIP_Y1
    y2 = y1 + SCORE_STRIP_HEIGHT
    draw_panel(frame, 0, y1, frame_width, y2)

    left_txt = f"LEFT {engine.score['left']}"
    dash_txt = "-"
    right_txt = f"{engine.score['right']} RIGHT"
    sub_txt = f"(first to {engine.win_score}, win by {engine.win_by_margin})"

    left_w = text_width(left_txt, scale=0.85, thickness=2)
    dash_w = text_width(dash_txt, scale=0.85, thickness=2)
    right_w = text_width(right_txt, scale=0.85, thickness=2)
    sub_w = text_width(sub_txt, scale=0.42, thickness=1)
    total_w = left_w + 24 + dash_w + 24 + right_w + 18 + sub_w
    start_x = max(10, (frame_width - total_w) // 2)

    cy = y1 + SCORE_STRIP_HEIGHT // 2 + 8
    x = start_x
    draw_text(frame, left_txt, (x, cy), scale=0.85, color=COLORS["left_team"], thickness=2)
    x += left_w + 24
    draw_text(frame, dash_txt, (x, cy), scale=0.85, color=COLORS["text_muted"], thickness=2)
    x += dash_w + 24
    draw_text(frame, right_txt, (x, cy), scale=0.85, color=COLORS["right_team"], thickness=2)
    x += right_w + 18
    draw_text(frame, sub_txt, (x, cy), scale=0.42, color=COLORS["text_muted"], thickness=1)

    strict_txt = f"[if whistle required]  L {strict_engine.score['left']} - {strict_engine.score['right']} R"
    strict_w = text_width(strict_txt, scale=0.4, thickness=1)
    draw_text(frame, strict_txt, (frame_width - strict_w - 14, y1 + 18),
              scale=0.4, color=COLORS["text_muted"], thickness=1)


# ------------------------------------------------------------------
# ZONE 3 -- LEFT CONFIDENCE PANEL (probability bars + streak progress)
# ------------------------------------------------------------------

CONF_PANEL_X1 = 10
CONF_PANEL_Y1 = SCORE_STRIP_Y1 + SCORE_STRIP_HEIGHT + 10
CONF_PANEL_WIDTH = 400
CONF_ROW_HEIGHT = 30


CURRENT_PRED_ROW_HEIGHT = 34  # extra space reserved at the top of the panel
# for the large "current prediction" readout, on top of the header label
# and the per-class bars below it.


def draw_confidence_panel(frame, probs, real_labels, streak_label=None, streak_count=0, current_label=None):
    n = len(real_labels)
    panel_h = 34 + CURRENT_PRED_ROW_HEIGHT + n * CONF_ROW_HEIGHT + (34 if streak_label else 10)
    x1, y1 = CONF_PANEL_X1, CONF_PANEL_Y1
    x2, y2 = x1 + CONF_PANEL_WIDTH, y1 + panel_h
    draw_panel(frame, x1, y1, x2, y2, border=True)

    draw_text(frame, "GESTURE CONFIDENCE", (x1 + 14, y1 + 24), scale=0.48,
              color=COLORS["text_muted"], thickness=1)

    top_idx = int(np.argmax(probs)) if len(probs) else -1
    top_score = probs[top_idx] if len(probs) else 0.0

    # NEW: prominent "what is it seeing RIGHT NOW" readout -- separate
    # from (and bigger than) the per-class bar list below, so the
    # current guess is immediately obvious without having to scan for
    # the highlighted row among 8 others. Uses current_label (the raw
    # per-window prediction, including "nothing") rather than just the
    # top1 index, so it correctly shows NOTHING when no class actually
    # cleared the decision threshold -- top_idx/top_score alone can't
    # tell that apart from a genuine low-confidence real-class guess.
    pred_y = y1 + 24 + CURRENT_PRED_ROW_HEIGHT
    if current_label and current_label != NOTHING_LABEL:
        pretty_current = DISPLAY_NAME.get(current_label, current_label)
        pred_text = f"NOW SEEING:  {pretty_current}  ({top_score:.0%})"
        pred_color = COLORS["accent_green"]
    else:
        pred_text = "NOW SEEING:  (nothing / idle)"
        pred_color = COLORS["text_muted"]
    draw_text(frame, pred_text, (x1 + 14, pred_y), scale=0.62, color=pred_color, thickness=2)
    cv2.line(frame, (x1 + 10, pred_y + 10), (x2 - 10, pred_y + 10), COLORS["panel_border"], 1, cv2.LINE_AA)

    row_y = pred_y + 24
    for i, label in enumerate(real_labels):
        score = probs[i]
        is_top = (i == top_idx and score > 0.5)
        row_y1 = row_y + i * CONF_ROW_HEIGHT
        row_y2 = row_y1 + CONF_ROW_HEIGHT - 4

        if is_top:
            cv2.rectangle(frame, (x1 + 4, row_y1 - 2), (x2 - 4, row_y2), COLORS["highlight_row"], -1)
            bar_color = COLORS["accent_green"]
            text_color = COLORS["text_primary"]
        else:
            bar_color = COLORS["accent_amber"] if score > 0.5 else COLORS["text_muted"]
            text_color = COLORS["text_muted"]

        pretty_label = DISPLAY_NAME.get(label, label)
        draw_text(frame, pretty_label, (x1 + 14, row_y1 + 18), scale=0.48, color=text_color, thickness=1)

        bar_x1 = x1 + 210
        bar_x2 = x2 - 60
        bar_max_w = bar_x2 - bar_x1
        cv2.rectangle(frame, (bar_x1, row_y1 + 4), (bar_x2, row_y2 - 4), COLORS["panel_bg_alt"], -1)
        cv2.rectangle(frame, (bar_x1, row_y1 + 4), (bar_x1 + int(bar_max_w * min(score, 1.0)), row_y2 - 4),
                      bar_color, -1)
        pct_text = f"{score:.2f}"
        draw_text(frame, pct_text, (x2 - 52, row_y1 + 18), scale=0.45, color=text_color, thickness=1)

    if streak_label:
        pretty = DISPLAY_NAME.get(streak_label, streak_label)
        needed = STREAK_NEEDED_TO_COMMIT if streak_label in SCORING_SIDE_MAP else FAULT_STREAK_NEEDED_TO_COMMIT
        streak_y = row_y + n * CONF_ROW_HEIGHT + 16
        streak_color = COLORS["accent_green"] if streak_count >= needed else COLORS["accent_amber"]
        draw_text(frame, f"building: {pretty} ({streak_count}/{needed})", (x1 + 14, streak_y),
                  scale=0.46, color=streak_color, thickness=1)


# ------------------------------------------------------------------
# ZONE 4 -- BOTTOM STACK (last decision chip + gesture history pills)
# ------------------------------------------------------------------

def draw_last_decision_chip(frame, text, color, frame_width, frame_height, y_bottom_offset):
    if not text:
        return
    chip_w = text_width(text, scale=0.5, thickness=2) + 24
    x1 = 10
    y2 = frame_height - y_bottom_offset
    y1 = y2 - 28
    draw_panel(frame, x1, y1, x1 + chip_w, y2, color=COLORS["panel_bg_alt"], border=True)
    draw_text(frame, text, (x1 + 12, y2 - 8), scale=0.5, color=color, thickness=2)


def draw_gesture_history_bar(frame, history, frame_width, frame_height):
    """Redesigned: was the main readability complaint (small text, flat
    strip). Now a taller dedicated panel with larger text, rendered as
    separated pill chips (alternating shade) with arrow separators, so
    consecutive entries are visually distinguishable at a glance.

    NOTE: `history` only ever contains labels that ACTUALLY committed --
    do_commit() only appends here on a genuine success (point_awarded,
    reason_attached, authorization_acknowledged), never on a pending/
    unconfirmed "awaiting_whistle_confirmation" state. See do_commit()'s
    and on_whistle()'s own comments for the fix that guarantees this."""
    bar_h = 52
    y1 = frame_height - bar_h
    y2 = frame_height
    draw_panel(frame, 0, y1, frame_width, y2)

    pretty_history = [DISPLAY_NAME.get(label, label) for label in history]

    if not pretty_history:
        text = "(no gestures committed yet)"
        w = text_width(text, scale=0.55, thickness=1)
        draw_text(frame, text, ((frame_width - w) // 2, y1 + 33), scale=0.55,
                  color=COLORS["text_muted"], thickness=1)
        return

    scale, thickness = 0.72, 2
    arrow = "  ->  "
    arrow_w = text_width(arrow, scale=scale, thickness=thickness)
    chip_pad = 16

    chip_widths = [text_width(t, scale=scale, thickness=thickness) + chip_pad * 2 for t in pretty_history]
    total_w = sum(chip_widths) + arrow_w * (len(pretty_history) - 1)
    start_x = max(10, (frame_width - total_w) // 2)

    x = start_x
    chip_y1, chip_y2 = y1 + 8, y2 - 8
    for i, (t, w) in enumerate(zip(pretty_history, chip_widths)):
        bg = COLORS["panel_bg_alt"] if i % 2 == 0 else COLORS["highlight_row"]
        is_last = (i == len(pretty_history) - 1)
        text_color = COLORS["accent_green"] if is_last else COLORS["text_primary"]
        cv2.rectangle(frame, (x, chip_y1), (x + w, chip_y2), bg, -1)
        if is_last:
            cv2.rectangle(frame, (x, chip_y1), (x + w, chip_y2), COLORS["accent_green"], 1, cv2.LINE_AA)
        draw_text(frame, t, (x + chip_pad, y2 - 16), scale=scale, color=text_color, thickness=thickness)
        x += w
        if i < len(pretty_history) - 1:
            draw_text(frame, arrow, (x, y2 - 16), scale=scale, color=COLORS["text_muted"], thickness=thickness)
            x += arrow_w


# ------------------------------------------------------------------
# ZONE 5 -- CORNER READOUTS (fps, pause state, whistle flash, controls)
# ------------------------------------------------------------------

def draw_corner_readout(frame, measured_fps, frame_width, frame_height, extra_line=None, ph_time=None):
    """AUDIO/TIMESTAMP HOOK -- TIMESTAMP half filled in: `ph_time` (the
    live Philippine-time string) is drawn as its own line here, so it's
    burned into the recorded fullwindow video for easier cross-
    referencing against real-world notes later, same reasoning as the
    ph_time column added to the CSV logs. The AUDIO half of this hook
    (an 'A = reconnect audio' status) is deliberately NOT reintroduced --
    audio recording/muxing was removed entirely in an earlier cleanup
    pass and isn't part of this pipeline anymore. `extra_line` is kept
    as a general-purpose second slot for whatever else needs it later.
    """
    lines = [f"fps: {measured_fps:.1f}"]
    if ph_time:
        lines.append(ph_time)
    if extra_line:
        lines.append(extra_line)

    line_w = max(text_width(l, scale=0.42, thickness=1) for l in lines)
    panel_h = 20 * len(lines) + 12
    x2 = frame_width - 10
    x1 = x2 - line_w - 20
    y1 = frame_height - 66 - panel_h
    y2 = y1 + panel_h
    draw_panel(frame, x1, y1, x2, y2, border=True)
    for i, l in enumerate(lines):
        draw_text(frame, l, (x1 + 10, y1 + 20 + i * 20), scale=0.42, color=COLORS["text_muted"], thickness=1)


def draw_controls_footer(frame, frame_width, frame_height):
    text = "Q/ESC = quit    P = pause/resume    W = manual whistle    F = fullscreen scoreboard    [ / ] = left score    - / + = right score    R = clear reason"
    # AUDIO/TIMESTAMP HOOK: append "    A = reconnect audio" to this
    # string once that key handler is restored from Gion's copy.
    w = text_width(text, scale=0.42, thickness=1)
    x = max(10, (frame_width - w) // 2)
    y1 = frame_height - 66 - 22
    draw_panel(frame, 0, y1, frame_width, frame_height - 66)
    draw_text(frame, text, (x, y1 + 16), scale=0.42, color=COLORS["text_muted"], thickness=1)


def draw_whistle_flash(frame, frame_width, frame_height):
    text = "WHISTLE!"
    scale, thickness = 1.3, 3
    w = text_width(text, scale=scale, thickness=thickness)
    x = (frame_width - w) // 2
    y = frame_height // 2 - 40
    draw_panel(frame, x - 24, y - 42, x + w + 24, y + 14, color=COLORS["accent_amber"], alpha=0.85, border=True)
    draw_text(frame, text, (x, y), scale=scale, color=(20, 20, 20), thickness=thickness)


def draw_paused_overlay(frame, frame_width, frame_height):
    text = "PAUSED -- nothing being processed (press P to resume)"
    scale, thickness = 0.7, 2
    w = text_width(text, scale=scale, thickness=thickness)
    x = (frame_width - w) // 2
    y = frame_height // 2
    draw_panel(frame, x - 20, y - 30, x + w + 20, y + 12, color=COLORS["accent_amber"], alpha=0.85, border=True)
    draw_text(frame, text, (x, y), scale=scale, color=(20, 20, 20), thickness=thickness)


def draw_match_over_banner(frame, engine, frame_width):
    winner = "LEFT" if engine.score["left"] > engine.score["right"] else "RIGHT"
    winner_color = COLORS["left_team"] if winner == "LEFT" else COLORS["right_team"]
    text = f"MATCH OVER -- {winner} WINS {engine.score['left']}-{engine.score['right']}"
    draw_panel(frame, 0, 0, frame_width, 70, color=COLORS["panel_bg"], alpha=0.9, border=True)
    w = text_width(text, scale=0.8, thickness=2)
    draw_text(frame, text, (max(10, (frame_width - w) // 2), 44), scale=0.8, color=winner_color, thickness=2)


# ------------------------------------------------------------------
# SCOREBOARD (audience-facing popup window) -- also refreshed to match
# ------------------------------------------------------------------

SCOREBOARD_WIDTH = 900
SCOREBOARD_HEIGHT = 400


def build_scoreboard_canvas(engine, paused):
    canvas = np.zeros((SCOREBOARD_HEIGHT, SCOREBOARD_WIDTH, 3), dtype=np.uint8)
    canvas[:] = (16, 14, 13)

    cv2.rectangle(canvas, (0, 0), (SCOREBOARD_WIDTH // 2, 6), COLORS["left_team"], -1)
    cv2.rectangle(canvas, (SCOREBOARD_WIDTH // 2, 0), (SCOREBOARD_WIDTH, 6), COLORS["right_team"], -1)

    if paused:
        text = "PAUSED"
        w = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 6)[0][0]
        cv2.putText(canvas, text, ((SCOREBOARD_WIDTH - w) // 2, SCOREBOARD_HEIGHT // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, COLORS["accent_amber"], 6, cv2.LINE_AA)
        return canvas

    if engine.set_over:
        winner = "LEFT" if engine.score["left"] > engine.score["right"] else "RIGHT"
        winner_color = COLORS["left_team"] if winner == "LEFT" else COLORS["right_team"]
        text = f"{winner} WINS"
        w = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 6)[0][0]
        cv2.putText(canvas, text, ((SCOREBOARD_WIDTH - w) // 2, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, winner_color, 6, cv2.LINE_AA)

    cv2.putText(canvas, "TEAM 1", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.1, COLORS["left_team"], 2, cv2.LINE_AA)
    cv2.putText(canvas, "TEAM 2", (SCOREBOARD_WIDTH - 300, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.1, COLORS["right_team"], 2, cv2.LINE_AA)

    left_text = str(engine.score["left"])
    right_text = str(engine.score["right"])
    cv2.putText(canvas, left_text, (140, 320), cv2.FONT_HERSHEY_SIMPLEX, 5.5, COLORS["text_primary"], 12, cv2.LINE_AA)
    cv2.putText(canvas, right_text, (SCOREBOARD_WIDTH - 320, 320), cv2.FONT_HERSHEY_SIMPLEX, 5.5, COLORS["text_primary"], 12, cv2.LINE_AA)

    dash_size = cv2.getTextSize("-", cv2.FONT_HERSHEY_SIMPLEX, 3.0, 8)[0]
    cv2.putText(canvas, "-", ((SCOREBOARD_WIDTH - dash_size[0]) // 2, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, COLORS["text_muted"], 8, cv2.LINE_AA)

    sub_text = f"first to {engine.win_score}, win by {engine.win_by_margin}"
    sub_w = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
    cv2.putText(canvas, sub_text, ((SCOREBOARD_WIDTH - sub_w) // 2, SCOREBOARD_HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["text_muted"], 1, cv2.LINE_AA)

    return canvas


# ------------------------------------------------------------------
# WINDOW RESIZE / FULLSCREEN HANDLING
# ------------------------------------------------------------------

def show_frame_letterboxed(window_name, image):
    """
    Displays `image` sized to fit window_name's CURRENT actual pixel
    dimensions -- whatever the user has resized, maximized, or
    fullscreened it to -- while preserving the image's original
    aspect ratio via letterboxing (black bars on whichever side has
    leftover space).

    WHY THIS IS NEEDED: cv2.imshow() on its own just non-uniformly
    stretches the source image to fill the window's exact pixel
    shape. Since every panel/circle/chip in this UI is drawn at fixed
    pixel coordinates relative to the CAMERA's native resolution
    (not the display window), that naive stretch is exactly what
    causes visible warping/distortion the moment someone drags the
    window to a different aspect ratio or fullscreens it on a screen
    with different proportions than the camera feed. This function
    replaces the raw cv2.imshow() call for both windows so the
    content always keeps correct proportions, regardless of window
    shape -- extra space just becomes black bars instead of stretched
    pixels.
    """
    img_h, img_w = image.shape[:2]
    try:
        _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
    except cv2.error:
        win_w, win_h = img_w, img_h  # window not yet realized -- fall back to native size

    if win_w <= 0 or win_h <= 0:
        win_w, win_h = img_w, img_h

    scale = min(win_w / img_w, win_h / img_h)
    new_w, new_h = max(1, int(img_w * scale)), max(1, int(img_h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    x_off = (win_w - new_w) // 2
    y_off = (win_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    cv2.imshow(window_name, canvas)


def generate_html_report(log_path, strict_log_path, output_path, session_start_time):
    """
    Builds a human-readable HTML report from the two CSVs this script
    already writes.

    EXPANDED (this pass): now shows TWO things, not just one --
      1. FULL RECOGNITION LOG: every single logged inference cycle from
         log_path, unfiltered -- including cycles where nothing
         committed. Shows exactly what the model saw (window_predicted_
         label) at every point in the session, alongside whatever DID
         commit that cycle (if anything). This is the "everything it
         recognized or saw" the user asked for -- previously the report
         only showed rows where something actually committed, silently
         dropping every "just recognized, nothing happened yet" cycle.
      2. COMMIT TIMELINE: the original filtered view -- only rows where
         something from EITHER engine (informational or strict)
         actually committed (point awarded, reason attached,
         authorization acknowledged, or explicitly ignored/pending),
         merged chronologically and tagged by which engine. Kept as its
         own section since it's much faster to scan for "what actually
         happened" than the full per-frame log.

    NOTE: with REQUIRE_WHISTLE_FOR_SCORING now True, `engine` and
    `strict_engine` behave identically (both require real whistles the
    same way), so the commit timeline's divergence numbers will
    normally sit at or near zero -- it's kept as a readable session log
    / sanity check rather than the live A/B evidence it was originally
    built for, since that whistle-gating decision is now settled by the
    paper's objectives.
    """
    def read_rows(path):
        with open(path, "r", newline="") as f:
            return list(csv.DictReader(f))

    main_rows = read_rows(log_path)
    strict_rows = read_rows(strict_log_path)

    main_events = [r for r in main_rows if r.get("engine_event")]
    strict_gesture_events = [r for r in strict_rows if r.get("event_type") == "gesture" and r.get("engine_event")]
    strict_whistle_events = [r for r in strict_rows if r.get("event_type") == "whistle"]

    timeline = []
    for r in main_events:
        timeline.append({
            "timestamp": float(r["timestamp"]), "ph_time": r.get("ph_time", ""),
            "engine": "Informational", "event": r["engine_event"],
            "label": r.get("vote_committed_label", ""), "reason": r.get("engine_reason", ""),
            "score": f"{r.get('score_left', '?')} - {r.get('score_right', '?')}",
        })
    for r in strict_gesture_events:
        timeline.append({
            "timestamp": float(r["timestamp"]), "ph_time": r.get("ph_time", ""),
            "engine": "Strict", "event": r["engine_event"],
            "label": r.get("label", ""), "reason": r.get("engine_reason", ""),
            "score": f"{r.get('strict_score_left', '?')} - {r.get('strict_score_right', '?')}",
        })
    timeline.sort(key=lambda e: e["timestamp"])

    event_color = {
        "point_awarded": "#2e7d32", "reason_attached": "#1565c0",
        "authorization_acknowledged": "#616161", "awaiting_whistle_confirmation": "#e65100",
        "ignored": "#c62828",
    }

    final_informational_score = main_rows[-1] if main_rows else {}
    final_strict_score = strict_rows[-1] if strict_rows else {}

    informational_points = sum(1 for r in main_events if r["engine_event"] == "point_awarded")
    strict_points = sum(1 for r in strict_gesture_events if r["engine_event"] == "point_awarded")
    point_gap = informational_points - strict_points

    commit_rows_html = ""
    for e in timeline:
        color = event_color.get(e["event"], "#333")
        commit_rows_html += (
            f"<tr><td>{e['ph_time']}</td><td><b>{e['engine']}</b></td>"
            f"<td style='color:{color};font-weight:bold'>{e['event']}</td>"
            f"<td>{e['label']}</td><td>{e['reason']}</td><td>{e['score']}</td></tr>\n"
        )

    # NEW: full unfiltered recognition log -- every row from log_path,
    # regardless of whether anything committed that cycle. window_
    # predicted_label is what the model actually saw; the remaining
    # columns show whether that cycle also happened to commit something.
    recognition_rows_html = ""
    for r in main_rows:
        seen_label = r.get("window_predicted_label", "")
        committed_label = r.get("vote_committed_label", "")
        event = r.get("engine_event", "")
        color = event_color.get(event, "#333") if event else "#999"
        seen_display = DISPLAY_NAME.get(seen_label, seen_label) if seen_label else ""
        committed_display = DISPLAY_NAME.get(committed_label, committed_label) if committed_label else ""
        recognition_rows_html += (
            f"<tr><td>{r.get('ph_time', '')}</td><td>{seen_display}</td>"
            f"<td>{committed_display}</td>"
            f"<td style='color:{color};font-weight:bold'>{event}</td>"
            f"<td>{r.get('engine_reason', '')}</td>"
            f"<td>{r.get('score_left', '?')} - {r.get('score_right', '?')}</td></tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Session Report -- {get_ph_time_str()}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 30px; background: #fafafa; color: #222; }}
h1, h2 {{ color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; font-size: 14px; }}
th {{ background: #eee; }}
.summary-box {{ background: white; border: 1px solid #ddd; border-radius: 6px; padding: 15px 20px; margin-bottom: 20px; }}
.divergence {{ color: #b71c1c; font-weight: bold; }}
</style></head>
<body>
<h1>Live Deployment Session Report</h1>
<div class="summary-box">
  <p><b>Session start (epoch):</b> {session_start_time:.3f}</p>
  <p><b>Final score (Informational, displayed):</b> {final_informational_score.get('score_left', '?')} - {final_informational_score.get('score_right', '?')}</p>
  <p><b>Final score (Strict, whistle-required):</b> {final_strict_score.get('strict_score_left', '?')} - {final_strict_score.get('strict_score_right', '?')}</p>
  <p><b>Points awarded -- Informational:</b> {informational_points} &nbsp;|&nbsp; <b>Strict:</b> {strict_points}
     {"<span class='divergence'>&nbsp;&nbsp;(" + str(point_gap) + " point(s) informational scored that strict did not)</span>" if point_gap != 0 else ""}</p>
  <p><b>Real whistle detections this session:</b> {len(strict_whistle_events)}</p>
</div>
<h2>Commit Timeline (Informational vs. Strict, side by side)</h2>
<p>Only rows where something actually committed or was explicitly ignored/pending -- fastest way to scan "what happened."</p>
<table>
<tr><th>PH Time</th><th>Engine</th><th>Event</th><th>Label</th><th>Reason</th><th>Score After</th></tr>
{commit_rows_html}
</table>
<h2>Full Recognition Log (Informational engine, every inference cycle)</h2>
<p>Every window the model classified this session, whether or not it led to a commit -- shows exactly what was seen alongside what (if anything) actually happened.</p>
<table>
<tr><th>PH Time</th><th>Seen (window prediction)</th><th>Committed</th><th>Event</th><th>Reason</th><th>Score</th></tr>
{recognition_rows_html}
</table>
</body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


def main():
    model, idx_to_real_label, real_labels = load_model()
    engine = DecisionEngine(win_score=GAME_WIN_SCORE, win_by_margin=GAME_WIN_BY_MARGIN)
    strict_engine = DecisionEngine(win_score=GAME_WIN_SCORE, win_by_margin=GAME_WIN_BY_MARGIN)

    os.makedirs(LOG_DIR, exist_ok=True)
    session_start_time = time.time()
    log_path = os.path.join(LOG_DIR, f"deployment_{int(session_start_time)}.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    # NEW: ph_time column added alongside the existing epoch timestamp --
    # same real-world-cross-referencing reason as the corner readout clock.
    log_writer.writerow(["timestamp", "ph_time", "expected_step", "window_predicted_label",
                          "vote_committed_label", "engine_event", "engine_reason",
                          "score_left", "score_right",
                          "strict_score_left", "strict_score_right"])

    strict_log_path = os.path.join(LOG_DIR, f"deployment_strict_{int(session_start_time)}.csv")
    strict_log_file = open(strict_log_path, "w", newline="")
    strict_log_writer = csv.writer(strict_log_file)
    strict_log_writer.writerow(["timestamp", "ph_time", "event_type", "label", "engine_event", "engine_reason",
                                 "strict_score_left", "strict_score_right"])

    print(f"Logging to: {log_path}")
    print(f"Strict (whistle-required) log: {strict_log_path}")
    print(f"SESSION START (epoch seconds): {session_start_time:.3f}")
    print(f"REAL GAME: first to {GAME_WIN_SCORE}, win by {GAME_WIN_BY_MARGIN}.\n")

    whistle_mode = {"value": "manual"}
    whistle_flash_until_holder = {"value": 0.0}

    def on_whistle(timestamp, confidence=None):
        nonlocal last_decision_text, last_decision_color, last_decision_time
        nonlocal last_committed_label, last_commit_time, seen_different_since_last_commit
        nonlocal expected_step

        # RESTORED: capture whatever's pending BEFORE calling
        # on_whistle_detected() (which clears it) -- needed to know
        # what, if anything, this whistle just retroactively confirmed.
        pending_label_before = engine.pending_gesture["label"] if engine.pending_gesture is not None else None
        pending_label_before_strict = strict_engine.pending_gesture["label"] if strict_engine.pending_gesture is not None else None

        confirmation_result = engine.on_whistle_detected(timestamp)
        confirmation_result_strict = strict_engine.on_whistle_detected(timestamp)

        strict_log_writer.writerow([f"{timestamp:.3f}", get_ph_time_str(), "whistle", "", "", "",
                                     strict_engine.score["left"], strict_engine.score["right"]])
        strict_log_file.flush()
        whistle_flash_until_holder["value"] = time.time() + 1.5
        conf_str = f" (confidence={confidence:.2f})" if confidence is not None else ""
        print(f"  -> WHISTLE{conf_str} at {timestamp:.3f}")

        # RESTORED: previously the confirmation_result from
        # on_whistle_detected() was silently discarded -- a gesture
        # that went pending and then got genuinely confirmed by a
        # whistle arriving within the grace window never showed up
        # anywhere: not in gesture_history, not in the decision chip,
        # not in the main CSV log. engine.score updated correctly
        # under the hood, but the UI had no way of reflecting it. This
        # mirrors what do_commit() already does for an immediate
        # (non-pending) commit.
        if confirmation_result is not None and pending_label_before is not None:
            gesture_history.append(pending_label_before)
            last_decision_text = f"{pending_label_before}: {confirmation_result['event']} (confirmed by late whistle)"
            last_decision_color = COLORS["accent_green"]
            last_decision_time = timestamp
            last_committed_label = pending_label_before
            last_commit_time = timestamp
            seen_different_since_last_commit = False
            if confirmation_result["event"] == "point_awarded":
                expected_step = "reason_gesture"
            elif confirmation_result["event"] == "authorization_acknowledged":
                expected_step = "whistle"
            log_writer.writerow([f"{timestamp:.3f}", get_ph_time_str(), expected_step, pending_label_before,
                                  pending_label_before, confirmation_result["event"], "confirmed_by_late_whistle",
                                  engine.score["left"], engine.score["right"],
                                  strict_engine.score["left"], strict_engine.score["right"]])
            log_file.flush()

        if confirmation_result_strict is not None and pending_label_before_strict is not None:
            strict_log_writer.writerow([f"{timestamp:.3f}", get_ph_time_str(), "gesture", pending_label_before_strict,
                                         confirmation_result_strict["event"], "confirmed_by_late_whistle",
                                         strict_engine.score["left"], strict_engine.score["right"]])
            strict_log_file.flush()

    detector = try_load_whistle_detector(on_whistle)
    whistle_mode["value"] = "auto" if detector is not None else "manual"
    if not REQUIRE_WHISTLE_FOR_SCORING:
        print("REQUIRE_WHISTLE_FOR_SCORING is False -- whistle is informational only, "
              "scoring will NOT be blocked by a missed detection.")
    else:
        print("REQUIRE_WHISTLE_FOR_SCORING is True -- matches the thesis paper's whistle-gating "
              "objective. A gesture with no real whistle detected within the temporal window will "
              "be held pending (or dropped) by decision_engine.py, not scored automatically.")
        if WHISTLE_DEVICE_INDEX is None and detector is None:
            print("  WARNING: no real whistle detector is active and WHISTLE_DEVICE_INDEX is None -- "
                  "EVERY point this session will need a manual W press to register at all.")

    pose_model = mp_pose.Pose(static_image_mode=False, model_complexity=0,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # PERFORMANCE FIX (candidate, needs live testing to confirm): was
    # static_image_mode=True -- unlike pose_model above, this made
    # MediaPipe treat every single frame as a brand-new unrelated image,
    # re-running the full palm-detection network from scratch every
    # frame instead of the much cheaper frame-to-frame tracking
    # static_image_mode=False enables. The EXTRA landmark-regression
    # cost on top of palm detection only kicks in once a hand is
    # actually found -- so this was close to free when hands weren't in
    # frame and got meaningfully more expensive the moment they were,
    # which matches "lags when my hands show" exactly. Also means
    # min_tracking_confidence=0.28 below was previously dead code --
    # tracking mode never activated with static_image_mode=True, so
    # that parameter had no effect until now.
    hands_model = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
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
    seen_different_since_last_commit = True
    frame_counter = 0
    show_skeleton = True
    gesture_history = deque(maxlen=8)
    last_decision_text = ""
    last_probs = None
    last_raw_prediction = None  # NEW: persists the most recent raw
    # per-window prediction (classify_window's current_label) across
    # frames -- inference only runs every INFERENCE_EVERY_N_FRAMES, so
    # without this the "NOW SEEING" readout would flicker back to
    # nothing on every frame in between real inference cycles.
    last_decision_color = COLORS["text_primary"]
    last_decision_time = 0

    pending_scoring_label = None
    pending_scoring_since = 0.0
    cancellation_streak_count = 0
    # RESTORED: fast, non-tolerant streak for fault/reason gestures
    # during the reason_gesture wait -- parallel to
    # cancellation_streak_count above, same reasoning. See
    # REASON_STREAK_NEEDED's own comment.
    reason_streak_label = None
    reason_streak_count = 0

    expected_step = "whistle"
    paused = True
    last_scoreboard_state = [None]
    last_scoreboard_canvas = [build_scoreboard_canvas(engine, paused)]
    fps_frame_times = deque(maxlen=30)

    # WINDOW RESIZE / FULLSCREEN: create both windows explicitly as
    # resizable (cv2.WINDOW_NORMAL), then all display calls route
    # through show_frame_letterboxed() so dragging/maximizing/
    # fullscreening either window never warps its contents -- see
    # that function's docstring for why this is necessary.
    cv2.namedWindow("BACKSTAGE (control)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("SCOREBOARD", cv2.WINDOW_NORMAL)
    scoreboard_fullscreen = False  # toggled with F -- see key handler below

    print("Live deployment running. Press Q/ESC to quit.\n")

    def do_commit(label, now):
        nonlocal last_decision_text, last_decision_color, last_decision_time
        nonlocal last_committed_label, last_commit_time, expected_step
        nonlocal seen_different_since_last_commit
        result = engine.on_gesture_detected(label, now)
        strict_result = strict_engine.on_gesture_detected(label, now)
        strict_log_writer.writerow([f"{now:.3f}", get_ph_time_str(), "gesture", label, strict_result["event"],
                                     strict_result.get("reason", ""),
                                     strict_engine.score["left"], strict_engine.score["right"]])
        strict_log_file.flush()
        last_decision_text = f"{label}: {result['event']}"
        if result["event"] == "ignored":
            last_decision_text += f" ({result['reason']})"
            last_decision_color = COLORS["accent_red"]
        elif result["event"] == "awaiting_whistle_confirmation":
            # RESTORED: this used to fall into the generic "else" branch
            # below, which appends to gesture_history and colors it
            # green -- as if the gesture had actually been committed.
            # It hasn't -- this is a PENDING state waiting on a real
            # whistle to arrive within decision_engine.py's grace
            # window. Now this gets its own honest, amber/pending
            # treatment and does NOT touch gesture_history or
            # expected_step -- if a whistle arrives in time,
            # on_whistle() above appends it correctly THEN; if it
            # expires, nothing was ever shown as committed, which is
            # the truthful outcome. This is the core of "the bottom
            # should only show committed scores."
            last_decision_text += f" (grace {result.get('grace_seconds', '?')}s)"
            last_decision_color = COLORS["accent_amber"]
        else:
            gesture_history.append(label)
            last_decision_color = COLORS["accent_green"]
            if result["event"] == "point_awarded":
                expected_step = "reason_gesture"
            elif result["event"] == "reason_attached":
                expected_step = "whistle"
            elif result["event"] == "authorization_acknowledged":
                expected_step = "whistle"
        last_decision_time = now
        last_committed_label = label
        last_commit_time = now
        seen_different_since_last_commit = False
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
            last_raw_prediction = current_label

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

            # RESTORED -- FAST REASON-STREAK TRACKING: after a point
            # commits, streak_label/streak_count reset to None/0, but
            # the referee doesn't switch to the reason gesture instantly
            # -- a few frames of leftover team_to_serve/service_
            # authorization motion get recognized first and BUILD a
            # fresh tolerant streak on THAT label. Once the real reason
            # gesture (e.g. ball_out) then starts, the tolerant
            # DECREMENT logic below doesn't switch streak_label to it
            # immediately -- it decrements the existing streak_label's
            # count by 1 per differing frame instead, so ball_out can't
            # even become the active streak_label until that stale
            # streak fully decays. This dedicated FAST, non-tolerant
            # counter, scoped to exactly when it matters (only while
            # actively waiting to attach a reason), fixes that.
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
            if pending_scoring_label is not None and cancellation_streak_count >= CANCELLATION_STREAK_NEEDED:
                pending_side = SCORING_SIDE_MAP[pending_scoring_label]
                pending_same_side_auth = f"service_authorization_{pending_side}"

                result = do_commit(pending_same_side_auth, now)
                committed_label_this_frame = pending_same_side_auth
                engine_event_this_frame = result["event"]
                engine_reason_this_frame = result.get("reason", "")
                pending_scoring_label = None
                streak_label = None
                streak_count = 0
                cancellation_streak_count = 0
                rolling_window.clear()
                cancel_check_triggered = True

            # RESTORED: fast reason commit -- fires as soon as
            # reason_streak_count reaches REASON_STREAK_NEEDED,
            # pre-empting the slow/tolerant generic streak_label path
            # below (the one that was starving this). Only relevant
            # when we're actually waiting for a reason (expected_step
            # == "reason_gesture"); decision_engine.py's own
            # REASON_ATTACH_WINDOW is still the authority on whether
            # it's actually still in time -- this only controls how
            # fast the LOCAL UI streak recognizes the gesture as real,
            # not whether the engine accepts it.
            reason_check_triggered = False
            if (not cancel_check_triggered and expected_step == "reason_gesture"
                    and reason_streak_label is not None and reason_streak_count >= REASON_STREAK_NEEDED):
                result = do_commit(reason_streak_label, now)
                committed_label_this_frame = reason_streak_label
                engine_event_this_frame = result["event"]
                engine_reason_this_frame = result.get("reason", "")
                reason_streak_label = None
                reason_streak_count = 0
                streak_label = None
                streak_count = 0
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
                streak_label = None
                streak_count = 0
                cancellation_streak_count = 0
                rolling_window.clear()
                engine.last_settle_start_time = None
                strict_engine.last_settle_start_time = None

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
                            streak_label = None
                            streak_count = 0
                            cancellation_streak_count = 0
                            rolling_window.clear()
                        else:
                            result = do_commit(pending_scoring_label, now)
                            committed_label_this_frame = pending_scoring_label
                            engine_event_this_frame = result["event"]
                            engine_reason_this_frame = result.get("reason", "")
                            pending_scoring_label = None
                            cancellation_streak_count = 0
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
                        repeat_allowed = seen_different_since_last_commit

                        if is_new_gesture or repeat_allowed:
                            result = do_commit(top_label, now)
                            committed_label_this_frame = top_label
                            engine_event_this_frame = result["event"]
                            engine_reason_this_frame = result.get("reason", "")
                            streak_label = None
                            streak_count = 0
                            rolling_window.clear()

            log_writer.writerow([f"{time.time():.3f}", get_ph_time_str(), expected_step, current_label,
                                  committed_label_this_frame, engine_event_this_frame, engine_reason_this_frame,
                                  engine.score["left"], engine.score["right"],
                                  strict_engine.score["left"], strict_engine.score["right"]])
            log_file.flush()

        # --------------------------------------------------------
        # RENDERING -- redesigned zone-based layout. See module
        # docstring "VISUAL REDESIGN" for the reasoning behind each
        # zone. Decision-logic above this point is UNCHANGED.
        # --------------------------------------------------------
        if engine.set_over:
            draw_match_over_banner(frame, engine, frame_width)
        else:
            pending_remaining = None
            cancellation_progress = None
            if pending_scoring_label is not None:
                pending_remaining = max(0.0, TEAM_TO_SERVE_CONFIRM_DELAY_SECONDS - (time.time() - pending_scoring_since))
                cancellation_progress = (cancellation_streak_count, CANCELLATION_STREAK_NEEDED)
            draw_top_status_strip(frame, expected_step, frame_width, pending_scoring_label,
                                   pending_remaining, whistle_mode["value"], cancellation_progress)
            draw_score_strip(frame, engine, strict_engine, frame_width)

        if last_probs is not None:
            draw_confidence_panel(frame, last_probs, real_labels, streak_label, streak_count, last_raw_prediction)

        draw_controls_footer(frame, frame_width, frame_height)
        draw_gesture_history_bar(frame, list(gesture_history), frame_width, frame_height)

        if last_decision_text and (time.time() - last_decision_time) < 4:
            draw_last_decision_chip(frame, f"decision_engine: {last_decision_text}",
                                     last_decision_color, frame_width, frame_height, y_bottom_offset=100)

        if time.time() < whistle_flash_until_holder["value"]:
            draw_whistle_flash(frame, frame_width, frame_height)

        if paused:
            draw_paused_overlay(frame, frame_width, frame_height)

        draw_corner_readout(frame, measured_fps, frame_width, frame_height, ph_time=get_ph_time_str())

        if full_window_writer is not None:
            full_window_writer.write(frame)

        show_frame_letterboxed("BACKSTAGE (control)", frame)
        current_scoreboard_state = (engine.score["left"], engine.score["right"], paused, engine.set_over)
        if current_scoreboard_state != last_scoreboard_state[0]:
            last_scoreboard_canvas[0] = build_scoreboard_canvas(engine, paused)
            last_scoreboard_state[0] = current_scoreboard_state
        show_frame_letterboxed("SCOREBOARD", last_scoreboard_canvas[0])

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
                cancellation_streak_count = 0
                reason_streak_label = None
                reason_streak_count = 0
                frame_counter = 0
                print("  -> RESUMED (cleared stale buffer)")
            else:
                print("  -> PAUSED")
        elif key == ord('s'):
            show_skeleton = not show_skeleton
        elif key == ord('f'):
            # FULLSCREEN TOGGLE: targets the SCOREBOARD window
            # specifically, since that's the audience/projector-facing
            # one -- BACKSTAGE stays windowed for the operator. Works
            # correctly with show_frame_letterboxed() above: going
            # fullscreen just changes what cv2.getWindowImageRect()
            # reports, which the letterbox logic already handles like
            # any other resize, no separate code path needed.
            scoreboard_fullscreen = not scoreboard_fullscreen
            cv2.setWindowProperty(
                "SCOREBOARD", cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if scoreboard_fullscreen else cv2.WINDOW_NORMAL
            )
            print(f"  -> SCOREBOARD fullscreen: {scoreboard_fullscreen}")
        elif key == ord('w'):
            on_whistle(time.time())
            if expected_step == "whistle":
                expected_step = "scoring_gesture"
        # AUDIO/TIMESTAMP HOOK: re-add the 'A = reconnect audio' handler
        # here from Gion's copy -- e.g.:
        #   elif key == ord('a'):
        #       detector = reconnect_audio_device(...)
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

    report_path = os.path.join(LOG_DIR, f"report_{int(session_start_time)}.html")
    generate_html_report(log_path, strict_log_path, report_path, session_start_time)
    print(f"\nInformational-vs-strict session report saved to: {report_path}")
    print("Open it in any browser to review the session's event timeline.")


if __name__ == "__main__":
    main()