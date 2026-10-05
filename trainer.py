"""
trainer.py

Volleyball Officiating TRAINING TOOL -- main application.

Flow:  Welcome + consent (Tk)  ->  Menu (Tk)  ->  training session (OpenCV)  ->  Menu ...

MODES
-----
practice   Free detection. Do any signal; the system names it, shows the model's live
           confidence per signal and a live FIVB checklist. No score.   (panel: "practice
           tool that automatically detects what gesture was done")
drill      Pick ONE signal; perform it as often as you like. Each attempt is graded
           CORRECT / ALMOST / INCORRECT with specific feedback.        (panel: "choose a
           gesture and show whether it was performed properly or incorrectly")
challenge  All signals once, random order, one attempt each. Trainee score = points.
combo      Back-to-back calls as at the end of a rally: Team to Serve, then the reason (Ball Out /
           Ball In / Double Contact), repeated. Runs automatically.
drill reps Choosing several repetitions in Drill runs them back to back (no stopping after each);
           every score is listed at the end and A / D reviews each attempt's checklist.
sim        Match simulation: scenarios where the trainee blows the whistle and then gives the
           right signals in order. The two-team scoreboard only moves when the trainee's call
           is correct.                                                  (panel: "scoring tied to
           the trainee's score")

Attempt scoring lives in gesture_grader.py (FIVB-derived rules, three difficulty levels).
Trainee points: CORRECT = 10, ALMOST = 5, INCORRECT = 0. Whistle in the simulation = 10 / 0.

WHAT IS REUSED FROM THE EXISTING SYSTEM (nothing is retrained)
--------------------------------------------------------------
live_deployment.load_model / classify_window / extract_frame_features  -> recognition
whistle_detector.WhistleDetector                                         -> whistle (W = manual)
The match-officiating program (live_deployment.py) is not modified and still runs as before.

CAMERA / LEFT-RIGHT
-------------------
Features are extracted from the UN-mirrored camera frame, so "left" is the trainee's own left
arm (matches the dataset and MediaPipe). Only the DISPLAY is mirrored (like a mirror), so your
left arm appears on the left of the screen. If the skeleton on screen shows the OPPOSITE arm from
the one you raise, your camera source is already mirrored (e.g. a Camo mirror option): set
INPUT_ALREADY_MIRRORED = True below. (M toggles the display mirror during a session.)

CONTROLS (shown on screen too)
  SPACE/ENTER continue | R retry (drill) | H hint (sim) | W manual whistle | M mirror display
  P pause | T light/dark | Q/ESC end session
"""

import argparse
import csv
import datetime
import html
import json
import os
import random
import re
import sys
import threading
import time
import traceback
from collections import deque

import cv2
import numpy as np

try:                                   # optional: measures CPU and memory for the performance figures
    import psutil
except ImportError:
    psutil = None

import devices
import gesture_grader as gg
import trainer_ui
from devices import open_camera   # noqa: F401  (re-exported: tests and other tools use trainer.open_camera)
from decision_engine import DecisionEngine, SCORING_GESTURES   # unchanged, same class live_deployment.py uses

# ============================================================================
# TUNABLE SETTINGS
# ============================================================================

# Camera / mic / theme settings live in trainer_config.py (edit them there, not here).
try:
    import trainer_config as _cfg
except ImportError:
    _cfg = None
INPUT_ALREADY_MIRRORED = getattr(_cfg, "INPUT_ALREADY_MIRRORED", False)  # camera source already mirrors the picture
MIRROR_DISPLAY = getattr(_cfg, "MIRROR_DISPLAY", True)                   # show yourself like a mirror (display only)
WHISTLE_DEVICE_INDEX = getattr(_cfg, "WHISTLE_DEVICE_INDEX", None)       # sounddevice index of your mic
WHISTLE_THRESHOLD = getattr(_cfg, "WHISTLE_THRESHOLD", 0.70)            # model probability for a whistle window
# (whistle_detector.DEFAULT_THRESHOLD is 0.70; the training tool may use its own operating point -- see trainer_config)

UI_W, UI_H = 1280, 720
TOP_H = 56
CAM_BOX = (0, TOP_H, 900, 506)              # x, y, w, h  (camera view)
BOTTOM_BOX = (0, TOP_H + 506, 900, UI_H - TOP_H - 506)
SIDE_X = 900                                # right-hand information column starts here
SIDE_W = UI_W - SIDE_X

ROLLING_WINDOW_FRAMES = 24       # same as live_deployment.py
INFERENCE_EVERY_N_FRAMES = 3     # same as live_deployment.py
COUNTDOWN_SECONDS = 3
CAPTURE_SECONDS = 5.0            # hold the signal ~1-2 s inside this window, then lower your arms
WHISTLE_WAIT_SECONDS = 12.0
RECORD_FPS = 10                  # matches live_deployment.py RAW_RECORD_FPS
CAPTURE_SECONDS_PAIR = 9.0       # two signals performed back to back (Team to Serve, then the reason)
AFTER_WHISTLE_COUNTDOWN = 1.0    # the signal follows the whistle right away (FIVB 22.2.3)
# automatic runs: (pause showing the next signal, how long each verdict stays up, countdown)
TIMINGS = {"drill": (2.5, 3.0, 3.0), "combo": (2.5, 3.5, 3.0), "challenge": (1.8, 2.2, 2.0),
           "sim": (5.0, 5.5, 2.0)}          # the match simulation timings are only used when "continuous" is on
WHISTLE_EXTRA_SECONDS = 1.5      # a step that starts with the whistle gets this much longer to capture
WHISTLE_ONSET_OFFSET = 1.2       # fallback only: a signal is recognised about half a model window after it started
# (1.2 s at the old 10 fps); the real value is computed from the capture's own frame rate, see _onset_offset()
WHISTLE_DETECT_DELAY = 0.75      # the microphone detector confirms a whistle about this long after it started: a
# 1.5 s window stepped every 0.5 s, two windows in a row to confirm. Measured 2026-10-04 on 37 confirmed whistles
# (median 0.5 s from the first whistle window to confirmation, + about 0.25 s inside that first window).
WHISTLE_GRACE_SECONDS = 1.5     # after the signal ends, how long a whistle-required step still waits for its whistle
NO_READING_CAMERA_CHECK = 3      # this many NO READINGs in a row: stop retrying and ask to check the camera
WHISTLE_TOLERANCE = 0.6          # the whistle may come this late after the estimated start and still count as first
AUTO_INTRO_SECONDS = 2.5
AUTO_RESULT_SECONDS = 3.0
WINDOW_NAME = "Volleyball Officiating Trainer"
SCOREBOARD_WINDOW_NAME = "Match Testing Scoreboard"     # separate window, Match Testing only -- see
SCOREBOARD_W, SCOREBOARD_H = 640, 300                    # _build_scoreboard_canvas / Session.run()
MT_WIN_SCORE = 25          # Match Testing win condition, same rule as live_deployment.py's
MT_WIN_BY_MARGIN = 2       # GAME_WIN_SCORE / GAME_WIN_BY_MARGIN -- deuce-style win-by-2
EARLY_FINISH = True              # end a capture as soon as the signal was seen and the arms are back at the ready
# position, instead of always waiting out CAPTURE_SECONDS -- otherwise the verdict can appear several seconds after the
# trainee finished, which breaks the evaluation forms' 3-second feedback rule
LOW_SIGNALS = {"ball_in"}        # held with the arm pointing down: "arms back at the ready position" is true DURING
# the signal, so only the model (signal no longer seen) can tell when it ended
READY_FRAMES = 6                 # camera frames the arms must be back at the ready position (about 0.2-0.6 s)
FEEDBACK_TARGET_SECONDS = 3.0    # the forms' 3-second rule: logged per attempt as feedback_latency_s
MT_TTS_CONFIRM_DELAY_SECONDS = 1.5   # same value as live_deployment.py's TEAM_TO_SERVE_CONFIRM_DELAY_SECONDS: a
# finished Team to Serve run is held this long before it counts, so a same-side Service Authorization (whose
# beckon starts from a Team-to-Serve-like pose) can take over and cancel it -- see _mt_check_pending_tts
MT_TTS_CANCEL_RECORDS = 2  # same-side Service Authorization windows needed to cancel it; with the stable-label
# lag (2 matching windows) this is 3 consecutive windows, same as live's CANCELLATION_STREAK_NEEDED
CAMERA_RETRY_SLEEP = 0.3         # camera stopped mid-session: wait this long between reconnect attempts
CAMERA_RETRY_EVERY = 3           # reopen the camera every this many failed reads
CAMERA_LOST_SECONDS = 8.0        # give up after this long, save everything and go back to the menu
CSV_COLS = ["ph_time", "kind", "target", "detected", "level", "intent", "note", "verdict", "score", "points", "best_prob", "margin",
            "hold_s", "confused_with", "failed_checks", "check_values", "feedback", "feedback_latency_s",
            # the SAME captured movement, graded at every level, so a report can show all three side by side
            "score_beginner", "verdict_beginner", "score_standard", "verdict_standard", "score_referee", "verdict_referee"]

PH_TZ = datetime.timezone(datetime.timedelta(hours=8))

# Colours are BGR (OpenCV). set_cv_theme() swaps these module-level names at runtime.
CV_THEMES = {
    "dark": dict(C_BG=(18, 15, 14), C_PANEL=(28, 24, 22), C_PANEL2=(44, 38, 34), C_LINE=(78, 70, 64),
                 C_TEXT=(245, 245, 245), C_MUTED=(175, 172, 168), C_GREEN=(110, 220, 120), C_AMBER=(60, 175, 250),
                 C_RED=(75, 80, 235), C_BLUE=(235, 178, 90), C_ORANGE=(90, 140, 250), C_ON_ACCENT=(20, 20, 20)),
    "light": dict(C_BG=(244, 240, 236), C_PANEL=(250, 247, 244), C_PANEL2=(236, 230, 224), C_LINE=(200, 190, 182),
                  C_TEXT=(36, 30, 27), C_MUTED=(117, 102, 91), C_GREEN=(72, 138, 23), C_AMBER=(0, 106, 181),
                  C_RED=(40, 40, 198), C_BLUE=(184, 111, 29), C_ORANGE=(20, 110, 220), C_ON_ACCENT=(255, 255, 255)),
}
# Drawn ON TOP OF THE CAMERA PICTURE, so they keep the same bright colours in both themes.
SK_BODY = (175, 172, 168)
SK_LEFT = (235, 178, 90)
SK_RIGHT = (90, 140, 250)
SK_TARGET = (110, 220, 120)
OV_AMBER = (60, 175, 250)
OV_RED = (75, 80, 235)
OV_ON = (20, 20, 20)

C_BG = C_PANEL = C_PANEL2 = C_LINE = C_TEXT = C_MUTED = C_GREEN = C_AMBER = C_RED = C_BLUE = C_ORANGE = C_ON_ACCENT = None
VERDICT_COLOR = {}
_cv_theme_name = "dark"


def set_cv_theme(name):
    """Switch the training-screen colours between "dark" and "light" (takes effect on the next frame)."""
    global VERDICT_COLOR, _cv_theme_name
    if name not in CV_THEMES:
        name = "dark"
    _cv_theme_name = name
    globals().update(CV_THEMES[name])
    VERDICT_COLOR = {gg.VERDICT_CORRECT: C_GREEN, gg.VERDICT_ALMOST: C_AMBER,
                     gg.VERDICT_INCORRECT: C_RED, gg.VERDICT_NO_READING: C_MUTED}


set_cv_theme(trainer_ui.current_theme())

VERDICT_TEXT = {gg.VERDICT_CORRECT: "CORRECT", gg.VERDICT_ALMOST: "ALMOST",
                gg.VERDICT_INCORRECT: "INCORRECT", gg.VERDICT_NO_READING: "NO READING"}

FONT = cv2.FONT_HERSHEY_SIMPLEX


CLEAN_LINE = "Clean signal. Nice work."


def detected_label(r):
    """What the model actually saw during an attempt, for the result screen and the attempt log: the target when it
    was recognised, otherwise the signal that dominated instead (or nothing). Empty for a NO_READING attempt."""
    if r is None or r.verdict == gg.VERDICT_NO_READING:
        return ""
    if r.recognized:
        return r.target
    return r.confused_with or gg.NOTHING_LABEL


def feedback_view(r):
    """What to show under the checklist. A clean or correct attempt gets no 'How to improve': only a kind word, and an
    optional short 'polish' list when a small check was missed. Camera notes are kept short and quiet.
    Returns (heading or None, [(text, kind)]) with kind in "ok", "tip", "cam"."""
    cam, tips = [], []
    for f in r.feedback:
        if f.startswith("Could not verify"):
            what = f[len("Could not verify: "):].split(". Make sure")[0]
            cam.append(f"Camera could not check: {what}. Keep that part visible.")
        elif f != CLEAN_LINE:
            tips.append(f)
    if r.verdict == gg.VERDICT_CORRECT:
        if not tips:
            return None, [(CLEAN_LINE, "ok")] + [(c, "cam") for c in cam[:1]]
        return "Optional polish", [("Counts as correct.", "ok")] + [(t, "tip") for t in tips[:3]] + [(c, "cam") for c in cam[:1]]
    return "How to improve", [(t, "tip") for t in tips] + [(c, "cam") for c in cam[:1]]


def now_ph_str(fmt="%Y-%m-%d %H:%M:%S"):
    return datetime.datetime.now(PH_TZ).strftime(fmt)


# ============================================================================
# Small drawing kit
# ============================================================================

def panel(img, x1, y1, x2, y2, color=None, alpha=1.0, border=False):
    color = C_PANEL if color is None else color
    h, w = img.shape[:2]
    x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2c <= x1c or y2c <= y1c:
        return
    region = img[y1c:y2c, x1c:x2c]
    if alpha >= 0.999:
        region[:] = color
    else:
        overlay = np.empty_like(region)
        overlay[:] = color
        cv2.addWeighted(overlay, alpha, region, 1 - alpha, 0, region)
    if border:
        cv2.rectangle(img, (x1, y1), (x2, y2), C_LINE, 1, cv2.LINE_AA)


def put(img, text, x, y, scale=0.55, color=None, thick=1):
    color = C_TEXT if color is None else color
    cv2.putText(img, text, (int(x), int(y)), FONT, scale, color, thick, cv2.LINE_AA)


def tw(text, scale=0.55, thick=1):
    return cv2.getTextSize(text, FONT, scale, thick)[0][0]


def wrap(text, max_w, scale=0.5, thick=1):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if tw(trial, scale, thick) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def fit(text, max_w, scale=0.5, thick=1):
    if tw(text, scale, thick) <= max_w:
        return text
    while len(text) > 4 and tw(text + "...", scale, thick) > max_w:
        text = text[:-1]
    return text + "..."


def label_lines(text, max_w, scale=0.45, thick=1, max_lines=2):
    """A check name on the side panel: whole if it fits, else without its (bracketed) part, else wrapped over at most
    max_lines lines. Never cut mid-word with '...' unless even that does not fit."""
    if tw(text, scale, thick) <= max_w:
        return [text]
    short = re.sub(r"\s*\([^)]*\)", "", text).strip()
    if tw(short, scale, thick) <= max_w:
        return [short]
    lines = wrap(short, max_w, scale, thick)
    return lines[:max_lines - 1] + [fit(" ".join(lines[max_lines - 1:]), max_w, scale, thick)]


def draw_icon(img, status, cx, cy, r=8):
    if status == "pass":
        cv2.circle(img, (cx, cy), r, C_GREEN, -1, cv2.LINE_AA)
        cv2.polylines(img, [np.array([[cx - 4, cy], [cx - 1, cy + 3], [cx + 4, cy - 3]], dtype=np.int32)], False, C_ON_ACCENT, 2, cv2.LINE_AA)
    elif status == "fail":
        cv2.circle(img, (cx, cy), r, C_RED, -1, cv2.LINE_AA)
        cv2.line(img, (cx - 3, cy - 3), (cx + 3, cy + 3), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(img, (cx - 3, cy + 3), (cx + 3, cy - 3), (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.circle(img, (cx, cy), r, C_AMBER, -1, cv2.LINE_AA)
        put(img, "?", cx - 4, cy + 5, 0.5, C_ON_ACCENT, 2)


def bar(img, x, y, w, h, frac, color, marker=None):
    cv2.rectangle(img, (x, y), (x + w, y + h), C_PANEL2, -1)
    cv2.rectangle(img, (x, y), (x + int(w * float(np.clip(frac, 0, 1))), y + h), color, -1)
    if marker is not None:
        mx = x + int(w * marker)
        cv2.line(img, (mx, y - 3), (mx, y + h + 3), C_TEXT, 2)


def fit_frame(frame, box):
    """Letterbox a camera frame into box=(x, y, w, h). Returns (canvas_region, placement)."""
    bx, by, bw, bh = box
    h, w = frame.shape[:2]
    s = min(bw / w, bh / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    ox, oy = bx + (bw - nw) // 2, by + (bh - nh) // 2
    return resized, (ox, oy, nw, nh)


def draw_skeleton(ui, feats, placement, mirror, target_side=None):
    """Own stick figure from the raw pose features (cols 0-23), so it follows the mirrored display.
    Left arm = blue, right arm = orange, with L / R tags at the wrists."""
    ox, oy, nw, nh = placement
    pts = {}
    names = ["ls", "rs", "le", "re", "lw", "rw", "lh", "rh"]
    for i, name in enumerate(names):
        x, y, vis = feats[i * 3], feats[i * 3 + 1], feats[i * 3 + 2]
        if vis < 0.3 and not (x or y):
            continue
        xx = (1.0 - x) if mirror else x
        pts[name] = (int(ox + xx * nw), int(oy + y * nh), vis)

    def line(a, b, color, thick=3):
        if a in pts and b in pts and pts[a][2] > 0.3 and pts[b][2] > 0.3:
            cv2.line(ui, pts[a][:2], pts[b][:2], color, thick, cv2.LINE_AA)

    line("ls", "rs", SK_BODY, 2)
    line("ls", "lh", SK_BODY, 2)
    line("rs", "rh", SK_BODY, 2)
    line("lh", "rh", SK_BODY, 2)
    lc = SK_TARGET if target_side == "left" else SK_LEFT
    rc = SK_TARGET if target_side == "right" else SK_RIGHT
    line("ls", "le", lc, 4)
    line("le", "lw", lc, 4)
    line("rs", "re", rc, 4)
    line("re", "rw", rc, 4)
    for name, (x, y, v) in pts.items():
        if v > 0.3:
            cv2.circle(ui, (x, y), 5, (245, 245, 245), -1, cv2.LINE_AA)
    for name, tag, color in (("lw", "L", lc), ("rw", "R", rc)):
        if name in pts and pts[name][2] > 0.3:
            x, y, _ = pts[name]
            cv2.circle(ui, (x, y), 15, color, -1, cv2.LINE_AA)
            put(ui, tag, x - 7, y + 7, 0.65, OV_ON, 2)


def fit_cv_window(name, w, h):
    """Size an OpenCV window to (w, h) or less, so it fits the screen with its title bar and the taskbar (a laptop
    at 1366x768, or any screen Windows scales to 125-150%). The picture inside is letterboxed to whatever size."""
    try:
        import ctypes
        sw, sh = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
    except Exception:                                  # not Windows: keep the requested size
        sw, sh = w + 40, h + 120
    scale = min(1.0, (sw - 40) / w, (sh - 120) / h)
    cv2.resizeWindow(name, max(320, int(w * scale)), max(180, int(h * scale)))


def show_letterboxed(window, image):
    ih, iw = image.shape[:2]
    try:
        _, _, ww, wh = cv2.getWindowImageRect(window)
    except cv2.error:
        ww, wh = iw, ih
    if ww <= 0 or wh <= 0:
        ww, wh = iw, ih
    s = min(ww / iw, wh / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    canvas = np.zeros((wh, ww, 3), dtype=np.uint8)
    canvas[(wh - nh) // 2:(wh - nh) // 2 + nh, (ww - nw) // 2:(ww - nw) // 2 + nw] = cv2.resize(
        image, (nw, nh), interpolation=cv2.INTER_AREA)
    cv2.imshow(window, canvas)


# ============================================================================
# Backend (camera + recognition). Injectable so it can be tested without hardware.
# ============================================================================

class Backend:
    def __init__(self, cap, extract, classify, idx_to_label, close=lambda: None):
        self.cap = cap
        self.extract = extract          # frame_bgr -> (122,) features
        self.classify = classify        # list of frames -> (label, top_prob, probs)
        self.idx_to_label = dict(idx_to_label)
        self.label_to_idx = {v: k for k, v in self.idx_to_label.items()}
        self.real_labels = [self.idx_to_label[i] for i in sorted(self.idx_to_label)]
        self.close = close
        self.cam_index = 0

    def release_camera(self):
        try:
            self.cap.release()
        except Exception:
            pass

    def reopen_camera(self, preferred):
        """Reopen after the setup screen. Falls back to another camera if `preferred` is unavailable."""
        self.cap, used = devices.open_camera(preferred)
        self.cam_index = used
        return used


def pick_camera_index(cli_index, default_index):
    """--camera, then the camera chosen in the setup screen, then trainer_config.CAMERA_INDEX, then the default."""
    if cli_index is not None:
        return cli_index
    saved = trainer_ui.read_settings().get("camera_index")
    if saved is not None:
        return saved
    if getattr(_cfg, "CAMERA_INDEX", None) is not None:
        return _cfg.CAMERA_INDEX
    return default_index


def build_backend(camera_index=None):
    """Loads the EXISTING model + feature extractor from live_deployment.py (unchanged)."""
    import live_deployment as ld
    import mediapipe as mp

    model, idx_to_real_label, _ = ld.load_model()
    cap, cam = devices.open_camera(pick_camera_index(camera_index, ld.CAMERA_INDEX))

    pose_model = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=0,
                                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands_model = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                           min_detection_confidence=0.1, min_tracking_confidence=0.28)

    def extract(frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        feats, _ = ld.extract_frame_features(rgb, pose_model, hands_model)
        return feats

    def classify(window_frames):
        return ld.classify_window(window_frames, model, idx_to_real_label)

    backend = Backend(cap, extract, classify, idx_to_real_label)
    backend.cam_index = cam

    def close():
        backend.release_camera()
        pose_model.close()
        hands_model.close()

    backend.close = close
    return backend


class WhistleHub:
    """Whistle input for the simulation: real detector if available, W key as backup."""

    def __init__(self):
        self._lock = threading.Lock()
        self._times = deque(maxlen=50)
        self._events = []           # every whistle event of the run: {"t", "source": "auto" | "manual", "confidence"}
        self.detector = None
        self.device = None          # microphone index chosen in the setup screen (None = use trainer_config)
        self.error = ""             # why the microphone detector is not running (logged at session start)

    def _record(self, source, confidence=None):
        now = time.time()
        with self._lock:
            self._times.append(now)
            self._events.append({"t": now, "source": source, "confidence": confidence})

    def _on_whistle(self, timestamp, confidence=None):       # called by the detector thread
        self._record("auto", confidence)

    def manual(self):                                          # the W key
        self._record("manual")

    def events_since(self, t):
        with self._lock:
            return [dict(e) for e in self._events if e["t"] >= t]

    def start_time(self, t):
        """When the whistle heard at t actually started: a microphone detection is confirmed about
        WHISTLE_DETECT_DELAY after the whistle began; a W key press is taken as is."""
        with self._lock:
            ev = next((e for e in self._events if e["t"] == t), None)
        return t - WHISTLE_DETECT_DELAY if ev is not None and ev["source"] == "auto" else t

    def start(self):
        if self.detector is not None:
            return
        try:
            from whistle_detector import WhistleDetector
            device = self.device if self.device is not None else WHISTLE_DEVICE_INDEX
            self.detector = WhistleDetector(on_whistle_callback=self._on_whistle, device=device,
                                            threshold=WHISTLE_THRESHOLD)
            self.detector.start(confirm_open=True)      # raises if the mic cannot be opened, instead of failing silently
            print("Whistle detection ACTIVE (mic). Press W as a backup.")
        except Exception as exc:  # missing model, no mic, sounddevice error ...
            self.detector = None
            self.error = str(exc)
            print(f"Whistle detector unavailable ({exc}). Use the W key as the whistle.")

    def stop(self):
        if self.detector is not None:
            try:
                self.detector.stop()
            except Exception:
                pass
            self.detector = None

    @property
    def auto_active(self):
        return self.detector is not None

    def first_since(self, t):
        """Time of the first whistle at or after t, or None."""
        with self._lock:
            c = [x for x in self._times if x >= t]
        return min(c) if c else None

    def heard_since(self, t):
        with self._lock:
            return any(x >= t for x in self._times)


# ============================================================================
# Scenarios (match simulation)
# ============================================================================

def _side_word(side):
    return "LEFT" if side == "left" else "RIGHT"


WHISTLE_LABELS = ("team_to_serve_", "service_authorization_")
WHISTLE_NOTE = "Blow the whistle first. In a match the whistle comes before the signal (FIVB 12.3, 22.2.1)."
REASONS = ("ball_out", "ball_in", "double_contact")


def needs_whistle(label):
    return label.startswith(WHISTLE_LABELS)


def _other_side(side):
    return "right" if side == "left" else "left"


def narrate_rally_end(reason, winner, fault_side):
    """Story for the end of a rally. `winner` is the team that gets the point and serves next."""
    W, F = _side_word(winner), _side_word(fault_side)
    if reason == "ball_out":
        return f"A player on the {F} team hit the ball OUT of bounds. The team on your {W} wins the rally and serves next."
    if reason == "ball_in":
        return (f"The ball hit by the {W} team landed INSIDE the {F} team's court and was not returned. "
                f"The team on your {W} wins the rally and serves next.")
    return (f"A player on the {F} team touched the ball twice (double contact). "
            f"The team on your {W} wins the rally and serves next. Show double contact with your {F} hand "
            f"(the side of the team at fault).")


def make_rally_end(rng, reason, server):
    """Pick who is at fault / who wins, consistently with the rules (6.1.3). Returns (winner, fault_side, text)."""
    receiver = _other_side(server)
    if reason == "ball_in":
        winner = rng.choice([server, receiver])
        fault_side = _other_side(winner)
    else:
        fault_side = rng.choice([server, receiver])
        winner = _other_side(fault_side)
    return winner, fault_side, narrate_rally_end(reason, winner, fault_side)


def make_pair_step(winner, reason, fault_side, text, **extra):
    ctx2 = {"side": fault_side} if reason == "double_contact" else None
    return {"kind": "pair", "labels": [f"team_to_serve_{winner}", reason], "ctxs": [None, ctx2], "text": text,
            "winner": winner, **extra}


def build_combo_queue(choice, real_labels, rng):
    """Back-to-back calls: Team to Serve (side), then the reason, in ONE continuous capture. Repeated `reps` times."""
    have = set(real_labels)
    reasons = [r for r in REASONS if r in have]
    reps = max(1, int(choice.get("reps", 3) or 3))
    key = choice.get("combo", "random")
    queue = []
    for i in range(reps):
        reason = key if key in reasons else (rng.choice(reasons) if reasons else None)
        winner = rng.choice(["left", "right"])
        if reason is None or f"team_to_serve_{winner}" not in have:
            continue
        server = rng.choice(["left", "right"])
        winner, fault_side, story = make_rally_end(rng, reason, server)
        if f"team_to_serve_{winner}" not in have:
            continue
        text = story + " Show the team to serve, then the reason."
        queue.append(make_pair_step(winner, reason, fault_side, text, scenario=i + 1, n_scenarios=reps))
    return queue


def build_sim_queue(rng, real_labels, n_rallies=3):
    """A short narrated set. Like in a match, the whistle and the signal happen together: the trainee blows the
    whistle and immediately gives the signal(s). Everything comes from the same 8 signals and one set (project scope)."""
    have = set(real_labels)
    reasons = [r for r in REASONS if r in have]
    queue = []
    if not reasons:
        return queue
    server = rng.choice(["left", "right"])
    last = None
    total = n_rallies
    for r in range(1, total + 1):
        sa = f"service_authorization_{server}"
        serve_text = (f"Rally {r} of {total}. The team on your {_side_word(server)} is in position and ready to serve. "
                      f"Blow the whistle and authorise the serve.")
        if sa in have:
            queue.append({"kind": "gesture", "label": sa, "text": serve_text, "scenario": r, "n_scenarios": total,
                          "whistle": True, "commit": "serve", "side": server})
        choices = [x for x in reasons if x != last] or reasons
        reason = rng.choice(choices)
        last = reason
        winner, fault_side, story = make_rally_end(rng, reason, server)
        if f"team_to_serve_{winner}" not in have:
            continue
        text = f"Rally {r} of {total} is over. " + story + " Blow the whistle, then show the team to serve and the reason."
        queue.append(make_pair_step(winner, reason, fault_side, text, scenario=r, n_scenarios=total, whistle=True,
                                    commit="rally"))
        server = winner
    if "end_of_set" in have:
        end_text = "The set is over (practice set). Blow the whistle and signal the end of the set."
        queue.append({"kind": "gesture", "label": "end_of_set", "text": end_text, "scenario": total,
                      "n_scenarios": total, "whistle": True, "commit": "set_end"})
    return queue


def add_whistle_steps(queue, choice):
    """Optional: Team to Serve and Service Authorization start with the whistle (FIVB 22.2). The whistle is part of
    the same step: blow it, then give the signal right away, like in a match."""
    if not choice.get("whistle"):
        return queue
    out = []
    for st in queue:
        first = st.get("label") or (st.get("labels") or [""])[0]
        if st["kind"] in ("gesture", "pair") and needs_whistle(first):
            st = dict(st, whistle=True)
        out.append(st)
    return out


def build_queue(mode, choice, real_labels, rng):
    if mode == "sim":
        n = int(choice.get("reps") or 3)
        return build_sim_queue(rng, real_labels, n if n in (3, 5, 10) else 3)
    if mode == "drill":
        reps = int(choice.get("reps", 1) or 1)
        if reps > 1:
            q = [{"kind": "gesture", "label": choice["gesture"], "rep": i + 1, "n_reps": reps} for i in range(reps)]
        else:
            q = [{"kind": "gesture", "label": choice["gesture"], "repeat": True}]
        return add_whistle_steps(q, choice)
    if mode == "combo":
        return add_whistle_steps(build_combo_queue(choice, real_labels, rng), choice)
    if mode == "challenge":
        labels = [l for l in real_labels if l in gg.SIGNALS]
        rng.shuffle(labels)
        return add_whistle_steps([{"kind": "gesture", "label": l} for l in labels], choice)
    return []


# ============================================================================
# Session
# ============================================================================

class Session:
    def __init__(self, backend, trainee, choice, whistle, clock=time.time, rng=None):
        self.be = backend
        self.trainee = trainee
        self.mode = choice["mode"]
        self.level = choice["level"]
        self.choice = choice
        self.whistle = whistle
        self.clock = clock
        self.rng = rng or random.Random()
        self.cfg = gg.LEVEL_CONFIG[self.level]

        self.intent = choice.get("intent", "normal")    # normal | correct | wrong  (labels for evaluation tests)
        self.note = choice.get("note", "") or ""         # free text: what the tester did differently
        stamp = datetime.datetime.now(PH_TZ).strftime("%Y%m%d_%H%M%S")
        suffix = "" if self.intent == "normal" else f"_{self.intent}"
        self.dir = os.path.join(trainer_ui.trainee_dir(trainee["trainee_id"]), f"{stamp}_{self.mode}{suffix}")
        self.rolling = deque(maxlen=ROLLING_WINDOW_FRAMES)
        self.frame_counter = 0
        self.fps_times = deque(maxlen=30)
        self.mirror_display = MIRROR_DISPLAY
        self.paused = False
        self.aspect = 9.0 / 16.0

        self.whistle_required = bool(choice.get("whistle")) and self.mode in ("drill", "combo", "challenge")
        self.queue = build_queue(self.mode, choice, backend.real_labels, self.rng)
        self.uses_whistle = self.mode in ("sim", "match_test") or any(
            q["kind"] == "whistle" or q.get("whistle") for q in self.queue)
        self.continuous = self.mode == "sim" and bool(choice.get("continuous"))
        self.serving = None             # match simulation: the team serving right now ("left" / "right", the trainee's side)
        self.commit = None              # the "committed" message shown at the bottom after a call (match simulation)
        self.commit_log = []
        self.whistle_state = None
        self._whistle_seen = False
        self.restart = False            # summary screen: R = do the same session again
        self.capture_windows = []       # for the whistle log: when each capture ran and how the whistle was judged
        self._t0 = None
        self._log_ready = False
        self.step_i = 0
        self.phase = "practice" if self.mode in ("practice", "match_test") else "intro"
        self.phase_t0 = 0.0
        self.cap_frames, self.cap_records = [], []
        self.result = None
        self.last_rec = None
        self.stable_label = gg.NOTHING_LABEL
        self._prev_label = gg.NOTHING_LABEL
        self.practice_log = []          # Practice mode: one row per detection event (see _save_outputs)
        self._practice_started = None
        self.practice_checks = []
        self.hint = False
        self.attempts = []
        self.attempt_results = []       # parallel to attempts: AttemptResult (gestures) or None (whistle)
        self.review_i = None
        self.reps = int(choice.get("reps", 1) or 1)
        self.auto = (self.mode in ("combo", "challenge") or (self.mode == "drill" and self.reps > 1)
                     or self.continuous)
        self.t_intro, self.t_result, self.t_countdown = TIMINGS.get(self.mode, (2.5, 3.0, COUNTDOWN_SECONDS))
        self.cd_seconds = self.t_countdown
        self.cap_seconds = CAPTURE_SECONDS
        self._ref_cache = {}
        self.sleep = time.sleep
        self._last_ui = None
        self.camera_lost = False
        self.no_reading_streak = 0      # NO READINGs in a row: a virtual camera (Camo) keeps sending a placeholder
        # picture when the phone is unplugged, so a lost camera shows up as "no person", never as a failed read
        self.error = None
        self.error_trace = ""
        self.perf = {"frame_ms": [], "extract_ms": [], "classify_ms": [], "grade_ms": [], "cpu_proc": [], "cpu_sys": [], "ram_mb": [],
                     "feedback_latency_s": []}
        # "signal finished" tracking for the 3-second rule: when the arms were last seen raised, and the recent frames
        self._cap_times = []
        self._arms_up_t = None
        self._recent = deque(maxlen=READY_FRAMES)       # Match Testing: the last few frames, for the same check
        self._recent_t = deque(maxlen=READY_FRAMES)
        self.last_latency = None
        self.results = []               # AttemptResult per signal of the current step (1, or 2 for a pair)
        self.attempt_no = 0
        self.auto_go = False            # becomes True after the trainee presses SPACE once
        self.points = 0
        self.max_points = 0
        self.team = {"left": 0, "right": 0}
        # Match Testing: continuous, unscripted grading of a real performer (see _update's "practice" branch and
        # _mt_end_run / _mt_flush_run below). Not the main focus of the tool; a lightweight secondary mode. The
        # whistle now GATES scoring here, exactly like live_deployment.py's decision_engine.py -- no longer
        # purely observational (see mt_gate below, and _mt_grade_run / _mt_poll_whistle).
        self.mt_in_run = False
        self.mt_run_frames, self.mt_run_records = [], []
        self.mt_run_label = None
        self.mt_last_result = None          # most recently graded AttemptResult, shown on the side panel
        self.practice_scores = []           # Practice mode: one row per fully graded hold (see _mt_grade_run)
        self.mt_whistle_pointer = 0.0        # continuous whistle polling: see _update
        self.mt_whistle_count = 0
        self.mt_history = deque(maxlen=8)    # sequence of labels that actually COMMITTED (point/authorization/
        # reason), mirroring live_deployment.py's own gesture_history -- never a pending/unconfirmed run
        self.mt_last_decision = None    # {"text", "color", "until"} -- the small chip shown ABOVE the history
        # bar, mirroring live_deployment.py's own draw_last_decision_chip (see _mt_set_decision)
        # Reuses the SAME DecisionEngine class live_deployment.py uses (imported unchanged) so a team point in
        # Match Testing only counts once a real whistle-gated sequence confirms it -- fixes a Team to Serve run
        # scoring instantly with no whistle check at all, which let a Service Authorization gesture misclassified
        # as Team to Serve (a real, documented confusion -- see Discussion) score a point with nothing to catch it.
        # Only active when the menu's "Require the whistle first" checkbox is checked for this session
        # (choice["whistle"] -- same key drill/combo/challenge already use, same default of False the menu
        # itself uses). Unchecked, this falls back to the ORIGINAL pre-fix behavior below (form verdict alone
        # decides scoring, whistle purely observational).
        self.mt_gate = (DecisionEngine() if self.mode == "match_test" and choice.get("whistle", False)
                        else None)
        self.mt_pending_form_ok = None       # FIVB form-quality (CORRECT/ALMOST) of whichever team_to_serve run
        # is currently sitting in mt_gate.pending_gesture, re-checked if a later whistle confirms it
        # Match Simulation with "Require the whistle" ticked: the same DecisionEngine decides whether each call's whistle
        # counts (whistle up to TEMPORAL_WINDOW before the signal, or up to WHISTLE_CONFIRMATION_GRACE_SECONDS after
        # it). A point / authorisation then needs BOTH the engine's commit and a passing FIVB form verdict, exactly
        # like Match Testing. Unticked: form verdict alone decides, the whistle is graded as its own attempt only.
        self.sim_gate = DecisionEngine() if self.mode == "sim" and choice.get("whistle", False) else None
        self.sim_gate_event = None           # what the engine did with the current step's call (shown in the commit)
        self.mt_pending_tts = None           # a graded Team to Serve run not yet committed -- live_deployment.py's
        # pending_scoring_label: {"label", "result", "all_levels", "since"} (see _mt_check_pending_tts)
        self.whistle_flash_until = 0.0
        self.whistle_ok = None
        self.banner = ""
        self.summary = None

    # ---- hooks overridable for tests ----
    def _show(self, ui):
        show_letterboxed(WINDOW_NAME, ui)
        if not getattr(self, "_wheel_hooked", False):     # mouse wheel scrolls the side card (after the window exists)
            try:
                cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
            except cv2.error:
                pass
            self._wheel_hooked = True

    def _key(self):
        return cv2.waitKey(1) & 0xFF

    # ---- helpers ----
    @property
    def step(self):
        return self.queue[self.step_i] if self.step_i < len(self.queue) else None

    def _flush_camera(self):
        for _ in range(4):
            self.be.cap.read()

    def _log_event(self, event, detail=""):
        """One line in session_log.csv: what happened, when. Written at once, so nothing is lost if the app stops."""
        try:
            path = os.path.join(self.dir, "session_log.csv")
            new = not os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["ph_time", "seconds_from_start", "event", "detail"])
                start = self._t0 if self._t0 is not None else self.clock()
                w.writerow([now_ph_str(), f"{self.clock() - start:.2f}", event, detail])
        except OSError:
            pass

    def _set_phase(self, phase):
        self.phase = phase
        self.phase_t0 = self.clock()

    def _intro_seconds(self, step):
        """Continuous match simulation: long enough to read the narration."""
        if self.continuous:
            return min(9.0, max(4.0, len(step.get("text", "")) / 14.0))
        return self.t_intro

    def _sync_serving(self):
        """Sets the opening server from the story. After that the serving indicator only moves when the trainee's own
        Team to Serve call is committed (see _grade_current), so it never shows a call that was not made."""
        st = self.step
        if self.serving is None and st is not None and st.get("commit") == "serve":
            self.serving = st["side"]

    def _begin_step(self):
        """Start the current step: the whistle wait, or the countdown before a signal."""
        step = self.step
        self._log_event("step_start", f"{step['kind']}: " + (", ".join(step["labels"]) if step["kind"] == "pair"
                                                              else step.get("label", "whistle")))
        self.rolling.clear()
        if step["kind"] == "whistle":
            self._set_phase("wait_whistle")
        else:
            self.cd_seconds = self.t_countdown
            self._set_phase("countdown")

    def _advance(self):
        step = self.step
        if step is not None and step.get("repeat"):
            self.step_i = step.get("loop_to", self.step_i)      # back to the whistle step, if there is one
            self._set_phase("intro")
            return
        self.step_i += 1
        if self.step_i >= len(self.queue):
            self._finish()
            return
        self._sync_serving()
        nxt = self.step
        if step is not None and step["kind"] == "whistle" and nxt["kind"] in ("gesture", "pair"):
            # the signal follows the whistle right away (FIVB 22.2.3)
            self.rolling.clear()
            self.cd_seconds = AFTER_WHISTLE_COUNTDOWN
            self._set_phase("countdown")
        else:
            self._set_phase("intro")

    def _finish(self):
        self.summary = self._make_summary()
        self.review_i = next((i for i, r in enumerate(self.attempt_results) if r is not None), None)
        self._set_phase("summary")

    def _review_step(self, delta):
        idxs = [i for i, r in enumerate(self.attempt_results) if r is not None]
        if not idxs:
            return
        cur = idxs.index(self.review_i) if self.review_i in idxs else 0
        self.review_i = idxs[(cur + delta) % len(idxs)]

    # ---- attempt logging ----
    def _log_attempt(self, step, res=None, whistle_ok=None, whistle_state=None, all_levels=None, latency=None):
        kind = step["kind"]
        if kind == "whistle":
            state = whistle_state or ("ok" if whistle_ok else "none")
            verdict = {"ok": gg.VERDICT_CORRECT, "late": gg.VERDICT_ALMOST, "none": gg.VERDICT_INCORRECT}[state]
            pts = {"ok": 10, "late": 5, "none": 0}[state]
            fb = {"ok": "", "late": "The whistle came after you started the signal. Blow it first (FIVB 22.2).",
                  "none": "No whistle detected."}[state]
            row = {"ph_time": now_ph_str(), "kind": "whistle", "target": "whistle", "detected": state, "level": self.level,
                   "verdict": verdict, "score": pts * 10, "points": pts, "best_prob": "",
                   "margin": "", "hold_s": "", "confused_with": "", "failed_checks": "", "check_values": "",
                   "intent": self.intent, "note": self.note, "feedback": fb,
                   "score_beginner": "", "verdict_beginner": "", "score_standard": "", "verdict_standard": "",
                   "score_referee": "", "verdict_referee": ""}
        else:
            failed = [c.id for c in res.checks if c.status == "fail"]
            row = {"ph_time": now_ph_str(), "kind": "gesture", "target": step["label"], "detected": detected_label(res),
                   "level": self.level,
                   "verdict": res.verdict, "score": res.score, "points": res.points,
                   "best_prob": f"{res.best_prob:.3f}", "margin": f"{res.margin:.3f}",
                   "hold_s": f"{res.hold_seconds:.2f}", "confused_with": res.confused_with or "",
                   "failed_checks": ";".join(failed), "intent": self.intent, "note": self.note,
                   "check_values": ";".join(
                       f"{c.id}={'' if c.value is None else round(c.value, 2)}[{c.need}]:{c.status}" for c in res.checks),
                   "feedback": " | ".join(res.feedback),
                   "feedback_latency_s": "" if latency is None else f"{latency:.2f}"}
            for lv in gg.LEVELS:
                lv_res = (all_levels or {}).get(lv)
                row[f"score_{lv}"] = lv_res.score if lv_res is not None else ""
                row[f"verdict_{lv}"] = lv_res.verdict if lv_res is not None else ""
            pts = res.points
        if row["verdict"] != gg.VERDICT_NO_READING:
            self.points += pts
            self.max_points += 10
            self.attempts.append(row)
            self.attempt_results.append(res if kind == "gesture" else None)
            self._append_csv(row)
        return row

    def _append_csv(self, row):
        """Every counted attempt is written to attempts.csv at once, so a crash later cannot lose it."""
        try:
            path = os.path.join(self.dir, "attempts.csv")
            new = not os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=CSV_COLS)
                if new:
                    w.writeheader()
                w.writerow(row)
        except OSError:
            pass

    def _make_summary(self):
        att = self.attempts
        n = len(att)
        correct = sum(1 for a in att if a["verdict"] == gg.VERDICT_CORRECT)
        almost = sum(1 for a in att if a["verdict"] == gg.VERDICT_ALMOST)
        per = {}
        for a in att:
            d = per.setdefault(a["target"], {"n": 0, "correct": 0, "score_sum": 0})
            d["n"] += 1
            d["correct"] += a["verdict"] == gg.VERDICT_CORRECT
            d["score_sum"] += a["score"]
        gest = [a for a in att if a["kind"] == "gesture"]
        avg = (sum(a["score"] for a in gest) / len(gest)) if gest else 0.0
        # signals where fewer than half of the attempts were CORRECT, weakest first (the whistle is not a signal)
        needs = sorted((k for k, d in per.items() if k != "whistle" and d["correct"] < d["n"] / 2.0),
                       key=lambda k: (per[k]["correct"] / per[k]["n"], per[k]["score_sum"] / per[k]["n"]))
        acc = 100.0 * correct / n if n else 0.0
        one = (f"{self.mode} ({self.cfg.name}): {correct}/{n} correct, {self.points}/{self.max_points} points"
               if n else f"{self.mode}: no graded attempts")
        return {"mode": self.mode, "level": self.level, "attempts": n, "correct": correct, "almost": almost,
                "incorrect": n - correct - almost, "points": self.points, "max_points": self.max_points,
                "accuracy_pct": round(acc, 1), "avg_gesture_score": round(avg, 1), "per_signal": per, "needs_practice": needs,
                "team_points": dict(self.team) if self.mode == "sim" else None, "one_line": one}

    # ---- main loop ----
    def _show_banner(self, text):
        img = self._last_ui.copy() if self._last_ui is not None else np.zeros((UI_H, UI_W, 3), np.uint8)
        panel(img, 0, TOP_H + 200, UI_W, TOP_H + 300, C_AMBER, 0.92)
        put(img, text, 40, TOP_H + 262, 0.85, (20, 20, 20), 2)
        self._show(img)

    def _sample_system(self, proc):
        if proc is None:
            return
        try:
            self.perf["cpu_proc"].append(proc.cpu_percent(interval=None))
            self.perf["cpu_sys"].append(psutil.cpu_percent(interval=None))
            self.perf["ram_mb"].append(proc.memory_info().rss / (1024 * 1024))
        except Exception:
            pass

    def run(self):
        os.makedirs(self.dir, exist_ok=True)
        writer = cv2.VideoWriter(os.path.join(self.dir, "session.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 RECORD_FPS, (UI_W, UI_H))
        self._t0 = self.clock()
        self.mt_whistle_pointer = self._t0     # the hub keeps every whistle of the app run: earlier sessions' whistles
        # were replayed into this session's decision engine when this started at 0.0
        if self.mode == "match_test":
            cv2.namedWindow(SCOREBOARD_WINDOW_NAME, cv2.WINDOW_NORMAL)
            fit_cv_window(SCOREBOARD_WINDOW_NAME, SCOREBOARD_W, SCOREBOARD_H)
        self._flush_camera()
        if self.uses_whistle:
            self.whistle.start()
        self._log_event("session_start", f"mode={self.mode}; level={self.level}; repetitions={self.reps}; "
                        f"whistle_option={bool(self.choice.get('whistle'))}; continuous={self.continuous}; "
                        f"whistle_detector={'auto (microphone)' if getattr(self.whistle, 'auto_active', False) else 'not running (W key)'}; "
                        f"whistle_threshold={WHISTLE_THRESHOLD:.2f}; "
                        + (f"whistle_detector_error={getattr(self.whistle, 'error', '')}; "
                           if self.uses_whistle and getattr(self.whistle, "error", "") else "")
                        + f"intent={self.intent}")
        if self.queue and self.mode != "practice":
            self._set_phase("intro")
            self._sync_serving()
        proc = psutil.Process() if psutil else None
        if proc is not None:
            try:
                proc.cpu_percent(interval=None)
                psutil.cpu_percent(interval=None)
            except Exception:
                proc = None
        last_sample = time.perf_counter()
        fails = 0

        try:
            while True:
                ok, frame = self.be.cap.read()
                if not ok:
                    fails += 1                       # camera stopped: try to reconnect, then give up cleanly
                    if fails == 1:
                        self._log_event("camera_disconnected", "no frame from the camera")
                    self._show_banner("Camera disconnected. Trying to reconnect...  (Q to stop)")
                    if fails % CAMERA_RETRY_EVERY == 0:
                        try:
                            self.be.reopen_camera(self.be.cam_index)
                        except Exception:
                            pass
                    if self._key() in (27, ord("q")) or fails * CAMERA_RETRY_SLEEP >= CAMERA_LOST_SECONDS:
                        self.camera_lost = True
                        self._log_event("camera_lost", "could not reconnect; the session was stopped and saved")
                        break
                    self.sleep(CAMERA_RETRY_SLEEP)
                    continue
                if fails:
                    self._log_event("camera_reconnected", f"after {fails} failed reads")
                fails = 0
                t_frame = time.perf_counter()
                if INPUT_ALREADY_MIRRORED:
                    frame = cv2.flip(frame, 1)
                self.aspect = frame.shape[0] / max(1, frame.shape[1])
                t = self.clock()
                self.fps_times.append(t)

                feats = np.zeros(122)
                rec = None
                if not self.paused:
                    t0 = time.perf_counter()
                    feats = self.be.extract(frame)
                    self.perf["extract_ms"].append((time.perf_counter() - t0) * 1000.0)
                    self.rolling.append(feats)
                    self.frame_counter += 1
                    if (self.phase in ("practice", "capture") and len(self.rolling) == ROLLING_WINDOW_FRAMES
                            and self.frame_counter % INFERENCE_EVERY_N_FRAMES == 0):
                        t0 = time.perf_counter()
                        label, _conf, probs = self.be.classify(list(self.rolling))
                        self.perf["classify_ms"].append((time.perf_counter() - t0) * 1000.0)
                        rec = {"label": label, "probs": np.array(probs, dtype=float),
                               "frames": np.array(self.rolling), "t": t}
                        self.last_rec = rec
                    self._update(t, feats, rec)
                else:
                    feats = np.array(self.rolling[-1]) if self.rolling else feats

                ui = self._render(frame, feats, t)
                self._last_ui = ui
                writer.write(ui)
                self._show(ui)
                if self.mode == "match_test":
                    show_letterboxed(SCOREBOARD_WINDOW_NAME, self._build_scoreboard_canvas())
                key = self._key()
                self.perf["frame_ms"].append((time.perf_counter() - t_frame) * 1000.0)
                if time.perf_counter() - last_sample >= 2.0:
                    last_sample = time.perf_counter()
                    self._sample_system(proc)
                if self._handle_key(key):
                    break
                if self.phase == "done":
                    break
        except Exception as exc:                    # never lose the results already recorded
            self.error = exc
            self.error_trace = traceback.format_exc()
            self._log_event("error", repr(exc))
        finally:
            try:
                writer.release()
            except Exception:
                pass
            if self.uses_whistle:
                try:
                    self.whistle.stop()
                except Exception:
                    pass
            if self.mode == "match_test":
                try:
                    cv2.destroyWindow(SCOREBOARD_WINDOW_NAME)
                except cv2.error:
                    pass

        if self.mode in ("match_test", "practice"):
            self._mt_flush_run()
        if self.summary is None:
            self.summary = self._make_summary()
        self.summary["performance"] = self._performance()
        self.summary["restart"] = bool(self.restart)
        self._log_event("session_end", self.summary.get("one_line", ""))
        self.summary["camera_lost"] = self.camera_lost
        self.summary["error"] = repr(self.error) if self.error else ""
        try:
            self._save_outputs()
        except Exception:
            self.error_trace += "\n" + traceback.format_exc()
        return self.summary

    def _performance(self):
        """Measured speed and load of THIS session (for the performance items of the expert form)."""
        def mean(x):
            return sum(x) / len(x) if x else 0.0

        fm = self.perf["frame_ms"]
        lat = self.perf["feedback_latency_s"]
        q = max(1, len(fm) // 4)
        fps = lambda ms: (1000.0 / mean(ms)) if ms else 0.0
        return {"frames": len(fm), "seconds": round(sum(fm) / 1000.0, 1), "fps_mean": round(fps(fm), 1),
                "fps_first_quarter": round(fps(fm[:q]), 1), "fps_last_quarter": round(fps(fm[-q:]), 1),
                "frame_ms_mean": round(mean(fm), 1), "frame_ms_max": round(max(fm), 1) if fm else 0.0,
                "extract_ms_mean": round(mean(self.perf["extract_ms"]), 1),
                "classify_ms_mean": round(mean(self.perf["classify_ms"]), 1),
                "grade_ms_mean": round(mean(self.perf["grade_ms"]), 1),
                "grade_ms_max": round(max(self.perf["grade_ms"]), 1) if self.perf["grade_ms"] else 0.0,
                "psutil": psutil is not None,
                "cpu_process_mean_pct": round(mean(self.perf["cpu_proc"]), 1),
                "cpu_process_max_pct": round(max(self.perf["cpu_proc"]), 1) if self.perf["cpu_proc"] else 0.0,
                "cpu_system_mean_pct": round(mean(self.perf["cpu_sys"]), 1),
                "ram_mean_mb": round(mean(self.perf["ram_mb"]), 1),
                "ram_max_mb": round(max(self.perf["ram_mb"]), 1) if self.perf["ram_mb"] else 0.0,
                # the forms' 3-second rule: seconds from "arms back down" to the verdict, per graded attempt
                "feedback_n": len(lat), "feedback_mean_s": round(mean(lat), 2),
                "feedback_max_s": round(max(lat), 2) if lat else 0.0,
                "feedback_within_3s": sum(1 for x in lat if x <= FEEDBACK_TARGET_SECONDS)}

    # ---- state machine ----
    def _update(self, t, feats, rec):
        step = self.step
        if self.phase == "practice":
            if self.mode in ("match_test", "practice") and self.mt_in_run:
                self.mt_run_frames.append(feats)      # every camera frame while a run is presumed active
            if self.mode == "match_test":
                self._recent.append(feats)
                self._recent_t.append(t)
                self._ready_status(list(self._recent), t)
                self._mt_poll_whistle(t)               # whistle logging is match-test only; Practice does not need it
                self._mt_check_pending_tts()
            if rec is not None:
                lab = rec["label"]
                prev_stable = self.stable_label
                if lab == gg.NOTHING_LABEL:
                    self.stable_label = gg.NOTHING_LABEL
                elif lab == self._prev_label:          # same signal on two consecutive windows = stable
                    self.stable_label = lab
                self._prev_label = lab
                self.practice_checks = (gg.grade_form_only(self.stable_label, rec["frames"], self.level, self.aspect)
                                        if self.stable_label in gg.RULES else [])
                if self._practice_started is None:
                    self._practice_started = self.clock()
                if self.stable_label != prev_stable:   # a new detection event: log it (documentation for the paper)
                    passed = sum(1 for c in self.practice_checks if c.status == "pass")
                    total_c = sum(1 for c in self.practice_checks if c.status != "unverified")
                    best_p = float(np.max(rec["probs"])) if rec.get("probs") is not None else None
                    self.practice_log.append({
                        "ph_time": now_ph_str(), "seconds_from_start": round(t - self._practice_started, 2),
                        "detected_label": self.stable_label,
                        "detected_name": gg.pretty_label(self.stable_label) if self.stable_label in gg.RULES else "(none)",
                        "best_prob": "" if best_p is None else round(best_p, 3),
                        "checks_passed": passed, "checks_total": total_c})
                    if self.mode in ("match_test", "practice"):
                        self._mt_handle_transition(prev_stable, feats)     # ends the old run, may start a new one
                if self.mode in ("match_test", "practice") and self.mt_in_run:
                    # this window belongs to the CURRENT stable label (just started, or continuing): record it
                    rec["end"] = len(self.mt_run_frames)
                    self.mt_run_records.append(rec)
        elif self.phase == "intro":
            if self.auto and self.auto_go and step is not None and t - self.phase_t0 >= self._intro_seconds(step):
                self._begin_step()
        elif self.phase == "result":
            if self.auto and t - self.phase_t0 >= self.t_result:
                if any(r.verdict == gg.VERDICT_NO_READING for r in self.results):
                    if self.no_reading_streak < NO_READING_CAMERA_CHECK:
                        self._set_phase("intro")  # not counted: automatically try the same step again
                    # else: stay on the "check the camera" message until SPACE (see _draw_side_result)
                else:
                    self._advance()
        elif self.phase == "countdown":
            if t - self.phase_t0 >= self.cd_seconds:
                self.cap_frames, self.cap_records = [], []
                self._cap_times, self._arms_up_t, self.last_latency, self._last_ready = [], None, None, None
                self.cap_seconds = CAPTURE_SECONDS_PAIR if step is not None and step["kind"] == "pair" else CAPTURE_SECONDS
                if step is not None and step.get("whistle"):
                    self.cap_seconds += WHISTLE_EXTRA_SECONDS
                self._whistle_seen = False
                self.whistle_state = None
                self._set_phase("capture")
                self._log_event("capture_start", f"{self.cap_seconds:.1f} s window")
        elif self.phase == "capture":
            self.cap_frames.append(feats)
            if rec is not None:
                rec["end"] = len(self.cap_frames)      # frame index in the capture where this window ends
                self.cap_records.append(rec)
            if step is not None and step.get("whistle") and not self._whistle_seen:
                if self.whistle.first_since(self.phase_t0) is not None:
                    self._whistle_seen = True
                    self.whistle_flash_until = t + 1.0
                    evs = getattr(self.whistle, "events_since", lambda _t: [])(self.phase_t0)
                    src = evs[0]["source"] if evs else "?"
                    self._log_event("whistle_heard", f"source={src}; {t - self.phase_t0:.2f} s after GO")
            self._cap_times.append(t)
            ready = self._last_ready = self._ready_status(self.cap_frames, t)
            if t - self.phase_t0 >= self.cap_seconds or (EARLY_FINISH and self._signal_finished(step, ready)):
                self._grade_current()
        elif self.phase == "wait_whistle":
            if self.whistle.heard_since(self.phase_t0):
                self._whistle_done(True)
            elif t - self.phase_t0 >= WHISTLE_WAIT_SECONDS:
                self._whistle_done(False)
        elif self.phase == "whistle_result":
            if t - self.phase_t0 >= 1.6:
                self._advance()

    # ---------------------------------------------------------------------
    # Match Testing: continuous, unscripted grading of a real performer.
    # Not the main focus of the tool; a lightweight secondary mode for watching and grading someone perform signals
    # naturally (no per-signal countdown, no expected order). The whistle now GATES scoring here, exactly like
    # live_deployment.py's decision_engine.py -- no longer purely observational (see mt_gate above).
    # ---------------------------------------------------------------------
    def _mt_poll_whistle(self, t):
        if not self.uses_whistle:
            return
        w = self.whistle.first_since(self.mt_whistle_pointer)
        if w is not None:
            self.mt_whistle_count += 1
            elapsed = w - self._t0 if self._t0 is not None else w
            self._log_event("whistle_heard", f"match test whistle #{self.mt_whistle_count} at {elapsed:.2f}s")
            self.mt_whistle_pointer = w + 1e-6      # advance past it so the next poll finds the NEXT whistle, if any
            self.whistle_flash_until = t + 1.0     # show "WHISTLE!" so the performer can see it was heard
            if self.mt_gate is not None:
                # A team_to_serve run graded a moment ago with no whistle yet is sitting in decision_engine.py's
                # own pending_gesture, waiting on exactly this. Capture which label (if any) BEFORE calling
                # on_whistle_detected(), since that call clears pending_gesture the instant it confirms it --
                # same pattern live_deployment.py's own on_whistle() uses.
                pending_before = (self.mt_gate.pending_gesture["label"]
                                   if self.mt_gate.pending_gesture is not None else None)
                confirm = self.mt_gate.on_whistle_detected(w)
                if confirm is not None:
                    self.mt_history.append(pending_before)   # this confirmation IS a real commit -- record it
                    if pending_before in SCORING_GESTURES and self.mt_pending_form_ok:
                        side = pending_before.rsplit("_", 1)[1]   # trainee's own side -- see the note in _mt_grade_run
                        self.team[side] += 1
                        self.serving = side
                        self._log_event("mt_point_awarded", f"{pending_before} confirmed by a later whistle")
                        self._mt_set_decision(f"{pending_before}: point_awarded (late whistle)", C_GREEN)
                    elif pending_before in SCORING_GESTURES:
                        self._mt_set_decision(f"{pending_before}: whistle arrived, form INCORRECT (no point)", C_AMBER)
                    else:
                        self._mt_set_decision(f"{pending_before}: {confirm['event']} (late whistle)", C_BLUE)
                self.mt_pending_form_ok = None

    def _mt_handle_transition(self, old_label, feats):
        """Called when the stable detected label changes. Grades and closes the run that just ended (if it was a
        real signal), then starts a new run if the new stable label is itself a real signal (so a continuous
        sequence like Team to Serve straight into the reason is captured as two back-to-back runs, not one)."""
        if old_label in gg.RULES and self.mt_in_run:
            self._mt_grade_run(old_label)
        if self.stable_label in gg.RULES:
            self.mt_in_run = True
            self.mt_run_label = self.stable_label
            self.mt_run_frames, self.mt_run_records = [feats], []
        else:
            self.mt_in_run = False
            self.mt_run_label = None
            self.mt_run_frames, self.mt_run_records = [], []

    def _mt_grade_run(self, label):
        """Grades one finished run (a hold of the same stable label, start to end) and logs it exactly like any
        other attempt. Runs shorter than 2 recognised windows are discarded quietly (brief flicker, not a real
        attempt) rather than graded and logged as a likely NO_READING or near-zero result."""
        recs, frames = self.mt_run_records, self.mt_run_frames
        if len(recs) < 2 or not frames:
            return
        step_s = max(0.05, (recs[-1]["t"] - recs[0]["t"]) / max(1, len(recs) - 1))
        all_levels = gg.grade_at_all_levels(label, frames, recs, self.be.label_to_idx, aspect=self.aspect,
                                            step_seconds=step_s)
        result = all_levels[self.level]
        mt_latency = None
        if self.mode == "match_test" and len(self._recent) >= READY_FRAMES and gg.ready_position_check(
                np.asarray(list(self._recent), dtype=float), self.level, self.aspect).status == "pass":
            # arms down: the run ended by lowering them; measured from the last frame they were still up
            mt_latency = self._record_latency(self._arms_up_t if self._arms_up_t is not None
                                              and self._arms_up_t < self._recent_t[-1] else None)
        old_cap_frames, old_cap_records = self.cap_frames, self.cap_records
        self.cap_frames, self.cap_records = frames, recs
        self._save_attempt_npz([label], [None], [result], step_s)
        self.cap_frames, self.cap_records = old_cap_frames, old_cap_records
        if result.verdict == gg.VERDICT_NO_READING:
            if self.mode == "match_test":       # not graded, but say so instead of dropping it silently
                self._log_event("mt_no_reading", f"{label}: {result.feedback[0] if result.feedback else ''}")
                self._mt_set_decision("NO READING: step back so your head, shoulders and arms are in frame",
                                      C_AMBER)
            return
        self.mt_last_result = result
        self.results = [result]
        if self.mode == "practice":
            if self._practice_started is None:
                self._practice_started = self.clock()
            row = {"ph_time": now_ph_str(), "seconds_from_start": round(self.clock() - self._practice_started, 2),
                  "target": label, "verdict": result.verdict, "score": result.score, "hold_s": f"{result.hold_seconds:.2f}"}
            for lv in gg.LEVELS:
                row[f"score_{lv}"] = all_levels[lv].score
                row[f"verdict_{lv}"] = all_levels[lv].verdict
            self.practice_scores.append(row)
        elif label in SCORING_GESTURES:
            # Held pending, not committed yet -- exactly like live_deployment.py's pending_scoring_label. A
            # Service Authorization beckon starts from a Team-to-Serve-like pose, so its first half is often
            # recognised as a Team to Serve run of the same side; committing that run straight away scored a point
            # for what was really an authorization. A new Team to Serve replaces an older pending one (live does
            # the same); the original hold time is kept when it is the same label.
            prev = self.mt_pending_tts
            if prev is not None:
                self._log_event("mt_pending_replaced", f"{prev['label']} replaced by {label} before it committed")
            since = prev["since"] if prev is not None and prev["label"] == label else self.clock()
            self.mt_pending_tts = {"label": label, "result": result, "all_levels": all_levels, "since": since,
                                   "latency": mt_latency}
            self._mt_set_decision(f"{label}: pending (confirming {MT_TTS_CONFIRM_DELAY_SECONDS:.1f}s)", C_AMBER)
        else:
            pending = self.mt_pending_tts
            if pending is not None:
                if label == self._mt_same_side_auth(pending["label"]):
                    self._mt_cancel_pending_tts()      # the "Team to Serve" was the start of this authorization
                else:
                    self._mt_commit_pending_tts()      # live: commit the pending point first, then this gesture
            self._mt_commit(label, result, all_levels, mt_latency)

    @staticmethod
    def _mt_same_side_auth(tts_label):
        return f"service_authorization_{tts_label.rsplit('_', 1)[1]}"

    def _mt_check_pending_tts(self):
        """Per-frame resolution of a pending Team to Serve, mirroring live_deployment.py's main loop: a same-side
        Service Authorization that takes over (MT_TTS_CANCEL_RECORDS windows into its own run) cancels it;
        otherwise it commits once MT_TTS_CONFIRM_DELAY_SECONDS have passed. Unlike live, the delay counts from the
        END of the Team to Serve run (this trainer only grades a run once it ends), not from when it was first
        recognised -- so the trainee always gets the full window to move into the beckon."""
        pending = self.mt_pending_tts
        if pending is None:
            return
        if (self.mt_in_run and self.mt_run_label == self._mt_same_side_auth(pending["label"])
                and len(self.mt_run_records) >= MT_TTS_CANCEL_RECORDS):
            self._mt_cancel_pending_tts()
        elif self.clock() - pending["since"] >= MT_TTS_CONFIRM_DELAY_SECONDS:
            self._mt_commit_pending_tts()

    def _mt_cancel_pending_tts(self):
        pending, self.mt_pending_tts = self.mt_pending_tts, None
        auth = self._mt_same_side_auth(pending["label"])
        self._log_event("mt_tts_cancelled", f"{pending['label']} was the start of {auth} -- no point")
        self._mt_set_decision(f"{pending['label']} -> {auth} (no point)", C_BLUE)

    def _mt_commit_pending_tts(self):
        pending, self.mt_pending_tts = self.mt_pending_tts, None
        self._mt_commit(pending["label"], pending["result"], pending["all_levels"], pending.get("latency"))
        if self.mt_gate is not None:
            # live_deployment.py clears the settle window after a delayed Team to Serve commit, so the reason
            # gesture right after it is not swallowed as tail-end noise
            self.mt_gate.last_settle_start_time = None

    def _mt_commit(self, label, result, all_levels, latency=None):
        """Logs one graded Match Testing run and feeds it to the scoreboard / decision engine."""
        self._log_attempt({"kind": "gesture", "label": label}, res=result, all_levels=all_levels, latency=latency)
        # Every graded run (not just team_to_serve) is fed into mt_gate, exactly like
        # live_deployment.py feeds every committed label into decision_engine.py -- this is what lets
        # a Service Authorization correctly consume its whistle, so a stray earlier whistle can't
        # spuriously validate an unrelated later Team to Serve. The FIVB form-quality check
        # (CORRECT/ALMOST) is kept as an ADDITIONAL requirement on top of the whistle gate, not
        # replaced, so nothing that used to correctly score stops scoring.
        if self.mt_gate is not None:
            gate_result = self.mt_gate.on_gesture_detected(label, self.clock())
            event = gate_result["event"]
            if event in ("point_awarded", "authorization_acknowledged", "reason_attached"):
                self.mt_history.append(label)      # the sequence bar: only real commits appear here
            form_ok = result.verdict in (gg.VERDICT_CORRECT, gg.VERDICT_ALMOST)
            if label in SCORING_GESTURES:
                if event == "point_awarded" and form_ok:
                    # side comes from the label itself, NOT gate_result["side"] -- decision_engine.py
                    # flips left/right to the audience-facing side for a real match scoreboard, but
                    # this trainer always means the TRAINEE's own left/right (see gesture_grader.py's
                    # documented convention). Using gate_result["side"] here would silently reverse it.
                    side = label.rsplit("_", 1)[1]
                    self.team[side] += 1
                    self.serving = side
                    self._log_event("mt_point_awarded", f"{label} confirmed by whistle")
                    self._mt_set_decision(f"{label}: point_awarded", C_GREEN)
                elif event == "point_awarded" and not form_ok:
                    # decision_engine confirmed the whistle sequence, but the FIVB form check failed --
                    # this trainer's own extra requirement, so no point is credited even though the real
                    # deployed system would have scored one here. Worth being able to see this happening.
                    self._log_event("mt_withheld", f"{label} whistle-confirmed but form was INCORRECT")
                    self._mt_set_decision(f"{label}: whistle ok, form INCORRECT (no point)", C_AMBER)
                elif event == "awaiting_whistle_confirmation":
                    self.mt_pending_form_ok = form_ok
                    self._log_event("mt_pending", f"{label} waiting up to "
                                    f"{gate_result.get('grace_seconds', '?')}s for a whistle before it can count")
                    self._mt_set_decision(f"{label}: awaiting whistle "
                                          f"({gate_result.get('grace_seconds', '?')}s)", C_AMBER)
                elif event == "ignored":
                    self._log_event("mt_ignored", f"{label}: {gate_result.get('reason', '')}")
                    self._mt_set_decision(f"{label}: ignored ({gate_result.get('reason', '')})", C_RED)
            elif event == "authorization_acknowledged":
                self._mt_set_decision(f"{label}: authorization_acknowledged", C_BLUE)
            elif event == "reason_attached":
                self._mt_set_decision(f"{label}: reason_attached", C_BLUE)
            elif event == "ignored":
                self._log_event("mt_ignored", f"{label}: {gate_result.get('reason', '')}")
                self._mt_set_decision(f"{label}: ignored ({gate_result.get('reason', '')})", C_RED)
        elif label in SCORING_GESTURES and result.verdict in (gg.VERDICT_CORRECT, gg.VERDICT_ALMOST):
            # "Require whistle" unchecked: mt_gate is None, whistle is fully observational, same as this
            # tool's original pre-fix behavior -- FIVB form verdict alone decides scoring (still after the
            # Team to Serve pending hold above). No decision_engine involved, so the sequence bar stays empty.
            side = label.rsplit("_", 1)[1]
            self.team[side] += 1
            self.serving = side
            self._log_event("mt_point_awarded", f"{label} (whistle not required)")
            self._mt_set_decision(f"{label}: point_awarded", C_GREEN)

    def _mt_set_decision(self, text, color, seconds=4.0):
        """The small chip shown just above the gesture-history bar -- mirrors live_deployment.py's own
        draw_last_decision_chip, same 4-second display window."""
        self.mt_last_decision = {"text": text, "color": color, "until": self.clock() + seconds}

    def _mt_flush_run(self):
        """Grades whatever was in progress when the session ended (Q pressed, camera lost), so a real performer's
        last signal is not silently dropped."""
        if self.mt_in_run and self.mt_run_label in gg.RULES:
            self._mt_grade_run(self.mt_run_label)
        self.mt_in_run = False
        if self.mt_pending_tts is not None:
            self._mt_commit_pending_tts()     # session over: nothing can cancel it any more

    def _ready_status(self, frames, t):
        """Ready-position check on the last READY_FRAMES frames ("pass" / "fail" / "unverified"). Remembers the last
        moment the arms were raised, which is when the trainee "finished the signal" for the 3-second rule."""
        if len(frames) < READY_FRAMES:
            return "unverified"
        status = gg.ready_position_check(np.asarray(frames[-READY_FRAMES:], dtype=float), self.level, self.aspect).status
        if status == "fail":
            self._arms_up_t = t
        return status

    def _onset_offset(self):
        """How long after a signal starts the model first recognises it: about half the rolling window, at the frame
        rate this capture actually ran at (the fixed 1.2 s assumed 10 fps; a laptop at 15-18 fps is nearer 0.7 s)."""
        times = self._cap_times
        if len(times) >= 10:
            dt = (times[-1] - times[0]) / (len(times) - 1)
            if dt > 0:
                return (ROLLING_WINDOW_FRAMES / 2.0) * dt
        return WHISTLE_ONSET_OFFSET

    def _signal_end(self, step, ready, final=False):
        """When the trainee finished the step's signal (the LAST one of a pair), or None if not finished yet.
        1. Arms: the arms were up after the signal was first recognised and are now back at the ready position. The end
           is the last frame whose READY_FRAMES window still showed them up (the check is a median, so this is at most
           half a window late -- a delay measured from it can only come out slightly too long, never too short).
        2. Model, for a signal held low that the arm check cannot see (Ball In): the model saw the signal and then
           NOTHING for two windows in a row (one is enough once the capture is over, final=True). The end is the last window that still saw it, minus half a model window
           (that window was at least half the signal), again so the delay is never under-reported."""
        if step is None:
            return None
        target = step["labels"][-1] if step["kind"] == "pair" else None
        hit = (lambda lab: lab == target) if target else (lambda lab: lab != gg.NOTHING_LABEL)
        recs = self.cap_records
        idx = [i for i, r in enumerate(recs) if hit(r["label"])]
        if not idx:
            return None
        # the signal must have been held: two windows in a row. A single window of e.g. Ball In while the arm comes
        # down from Team to Serve ended a Combo / Simulation capture before the real reason signal was shown.
        if not any(b - a == 1 and recs[a]["label"] == recs[b]["label"] for a, b in zip(idx, idx[1:])):
            return None
        first_t = recs[idx[0]]["t"]
        # the arms must have been up AFTER that signal was first recognised: the pause between the two signals of a
        # pair (arms briefly down) must not count as the end
        low_signal = (target or (recs[idx[-1]]["label"])) in LOW_SIGNALS
        if not low_signal and ready == "pass" and self._arms_up_t is not None and self._arms_up_t >= first_t:
            return self._arms_up_t
        tail = recs[idx[-1] + 1:]
        need = 1 if final else 2
        if len(tail) >= need and all(r["label"] == gg.NOTHING_LABEL for r in tail[-need:]):
            times = self._cap_times
            dt = (times[-1] - times[0]) / (len(times) - 1) if len(times) >= 2 else 0.1
            return recs[idx[-1]]["t"] - (ROLLING_WINDOW_FRAMES / 2.0) * dt
        return None

    def _signal_finished(self, step, ready):
        """Early end of a capture (EARLY_FINISH). A step that needs the whistle, with no whistle heard yet, waits
        WHISTLE_GRACE_SECONDS after the signal ended (the detector confirms a whistle ~0.75 s late) instead of the whole
        window -- otherwise a forgotten whistle left the trainee waiting up to ~6 s for the verdict."""
        end = self._signal_end(step, ready)
        if end is None:
            return False
        if step is not None and step.get("whistle") and not self._whistle_seen:
            return self.clock() - end >= WHISTLE_GRACE_SECONDS
        return True

    def _record_latency(self, end_t):
        """Seconds from the end of the signal to now (the verdict is computed now and drawn on the next frame)."""
        if end_t is None:
            return None
        lat = round(max(0.0, self.clock() - end_t), 2)
        self.perf["feedback_latency_s"].append(lat)
        return lat

    def _grade_current(self):
        t_grade = time.perf_counter()
        step = self.step
        recs = self.cap_records
        step_s = 0.3
        if len(recs) >= 2:
            step_s = max(0.05, (recs[-1]["t"] - recs[0]["t"]) / (len(recs) - 1))
        if step["kind"] == "pair":
            labels = list(step["labels"])
            ctxs = step.get("ctxs") or [None] * len(labels)
            results = gg.grade_sequence(labels, self.cap_frames, recs, self.be.label_to_idx, level=self.level,
                                        aspect=self.aspect, step_seconds=step_s, contexts=ctxs)
            # the SAME capture, re-graded at the other two levels, so a report can show all three side by side
            all_levels_by_i = [{} for _ in labels]
            for lv in gg.LEVELS:
                lv_results = gg.grade_sequence(labels, self.cap_frames, recs, self.be.label_to_idx, level=lv,
                                               aspect=self.aspect, step_seconds=step_s, contexts=ctxs)
                for i, r in enumerate(lv_results):
                    all_levels_by_i[i][lv] = r
        else:
            labels = [step["label"]]
            ctxs = [step.get("ctx")]
            results = [gg.grade_attempt(step["label"], self.cap_frames, recs, self.be.label_to_idx,
                                        level=self.level, aspect=self.aspect, step_seconds=step_s, context=step.get("ctx"))]
            all_levels_by_i = [gg.grade_at_all_levels(step["label"], self.cap_frames, recs, self.be.label_to_idx,
                                                      aspect=self.aspect, step_seconds=step_s, context=step.get("ctx"))]
        self.results, self.result = results, results[0]
        if any(r.verdict == gg.VERDICT_NO_READING for r in results):
            self.no_reading_streak += 1
            if self.no_reading_streak == NO_READING_CAMERA_CHECK:
                self._log_event("camera_check", f"{self.no_reading_streak} NO READINGs in a row: asked to check the camera")
        else:
            self.no_reading_streak = 0
        self.perf["grade_ms"].append((time.perf_counter() - t_grade) * 1000.0)
        self.last_latency = self._record_latency(self._signal_end(step, getattr(self, "_last_ready", None),
                                                                   final=True))
        # the whistle: heard during this capture, and before the first signal started?
        self.whistle_state = None
        onset_est = next((r["t"] - self._onset_offset() for r in recs if r["label"] == labels[0]), None)
        if step.get("whistle"):
            wt = self.whistle.first_since(self.phase_t0)
            onset = onset_est
            # Presence only (team decision 2026-10-04): FIVB 22.2.3 sets the ORDER (whistle, then the signals) but no
            # time limit, and the panel did not ask for whistle timing to be graded. A whistle heard anywhere in this
            # attempt counts. The estimated whistle-to-signal gap is still logged (whistle_events.csv) for reference.
            self.whistle_state = "none" if wt is None else "ok"
        self._save_attempt_npz(labels, ctxs, results, step_s)
        self.capture_windows.append({"start": self.phase_t0, "end": self.clock(), "step": self.attempt_no,
                                     "labels": "+".join(labels), "state": self.whistle_state, "onset": onset_est})
        self._log_event("verdict", "; ".join(f"{l}={r.verdict} {r.score}/100" for l, r in zip(labels, results))
                        + (f"; whistle={self.whistle_state}" if self.whistle_state else "")
                        + f"; capture {self.clock() - self.phase_t0:.1f}s"
                        + ("" if self.last_latency is None else f"; feedback {self.last_latency:.2f}s after arms down"))
        if not any(r.verdict == gg.VERDICT_NO_READING for r in results):
            if self.whistle_state is not None:
                self._log_attempt({"kind": "whistle"}, whistle_state=self.whistle_state)
            self.sim_gate_event = self._sim_gate_step(labels, recs) if self.sim_gate is not None else None
            gate_ok = self.sim_gate_event is None or self.sim_gate_event in ("point_awarded",
                                                                             "authorization_acknowledged")
            for label, res, all_levels in zip(labels, results, all_levels_by_i):
                self._log_attempt({"kind": "gesture", "label": label}, res=res, all_levels=all_levels,
                                  latency=self.last_latency)
                if (self.mode == "sim" and res.verdict in (gg.VERDICT_CORRECT, gg.VERDICT_ALMOST)
                        and label.startswith("team_to_serve_") and gate_ok):        # a close call still counts
                    side = label.rsplit("_", 1)[1]
                    self.team[side] += 1
                    self.serving = side                 # the serving indicator follows the committed call only
            if self.mode == "sim":
                self.commit = self._make_commit(step, labels, results)
                self.commit_log.append(f"{self.commit['title']}: {self.commit['lines'][0]}")
                del self.commit_log[:-3]
        self._set_phase("result")

    def _sim_gate_step(self, labels, recs):
        """Match Simulation with the whistle required: replays this capture's whistle and its first signal through
        decision_engine.py in time order (the signal at the moment the model first recognised it) and returns the
        engine's event for that signal -- "point_awarded", "authorization_acknowledged", or something else when the
        whistle did not count. Each call is its own whistle-then-signal cycle, so whistle state from an earlier step
        is cleared first (a whistle blown for the previous call must not validate this one). end_of_set is not
        whistle-gated by the engine, so it returns None (form alone decides, as before)."""
        first = labels[0]
        if first not in SCORING_GESTURES and not first.startswith("service_authorization_"):
            return None
        g = self.sim_gate
        g.last_whistle_time, g.pending_gesture, g.last_settle_start_time = None, None, None
        events = []
        wt = self.whistle.first_since(self.phase_t0)
        if wt is not None:
            events.append((getattr(self.whistle, "start_time", lambda x: x)(wt), 0, None))
        gt = next((r["t"] for r in recs if r["label"] == first), None)
        if gt is not None:
            events.append((gt, 1, first))
        outcome = "not_recognised" if gt is None else "no_whistle"
        for ts, _order, label in sorted(events):
            res = g.on_gesture_detected(label, ts) if label else g.on_whistle_detected(ts)
            if res is not None and res["event"] in ("point_awarded", "authorization_acknowledged"):
                outcome = res["event"]
        self._log_event("sim_gate", f"{first}: {outcome}"
                        + ("" if wt is None or gt is None else f"; whistle {wt - gt:+.2f}s from the signal"))
        return outcome

    def _make_commit(self, step, labels, results):
        """The message a scoreboard would show after the call (match simulation): what was committed, or why not.
        A close call (ALMOST) is committed too; it does not have to be perfect."""
        a, b = self.team["left"], self.team["right"]
        score = f"Score: LEFT {a} - {b} RIGHT."
        wl = {"ok": "Whistle: heard.", "late": "Whistle: heard, but after the signal started.",
              "none": "Whistle: not heard."}.get(self.whistle_state)
        expected = "Correct call: " + self._step_names(step) + "."
        form_ok = lambda r: r.verdict in (gg.VERDICT_CORRECT, gg.VERDICT_ALMOST)
        gate_ok = self.sim_gate_event in (None, "point_awarded", "authorization_acknowledged")
        passes = lambda r: form_ok(r) and gate_ok
        close = lambda r: " (close, not perfect)" if r.verdict == gg.VERDICT_ALMOST else ""
        no_whistle = ("No whistle counted with your call (Require the whistle is on), so nothing was recorded."
                      if form_ok(results[0]) and not gate_ok else None)
        if step["kind"] == "pair":
            W = _side_word(step["winner"])
            if passes(results[0]):
                title, ok = "COMMITTED", True
                lines = [f"Point to the team on your {W}{close(results[0])}. {score}",
                         f"The team on your {W} serves next. Reason: {gg.short_label(labels[1])}"
                         + ("." if results[1].verdict != gg.VERDICT_INCORRECT else " (not confirmed).")]
            else:
                title, ok = "NOT COMMITTED", False
                lines = [no_whistle or "Your call was not clear or not correct, so no point was recorded.", expected]
        elif step.get("commit") == "serve":
            W = _side_word(step["side"])
            if passes(results[0]):
                title, ok = "SERVE AUTHORISED", True
                lines = [f"The team on your {W} may serve{close(results[0])}.", score]
            else:
                title, ok = "NOT AUTHORISED", False
                lines = [no_whistle or "The authorisation was not clear or not correct.", expected]
        else:
            if passes(results[0]):
                title, ok = "SET ENDED", True
                lines = [f"Final {score}{close(results[0])}", "The set is finished."]
            else:
                title, ok = "NOT COMMITTED", False
                lines = ["The end of set signal was not clear or not correct.", expected]
        if wl:
            lines.append(wl)
        return {"ok": ok, "title": title, "lines": lines}

    def _save_attempt_npz(self, labels, ctxs, results, step_s):
        """Keeps the real movement of every attempt so thresholds can be re-checked later without redoing it
        (tools/regrade_attempts.py). Best effort: never interrupts a session."""
        try:
            d = os.path.join(self.dir, "attempts")
            os.makedirs(d, exist_ok=True)
            self.attempt_no += 1
            recs = self.cap_records
            n_cls = len(self.be.real_labels)
            meta = {"targets": labels, "contexts": ctxs, "level": self.level, "aspect": self.aspect,
                    "step_seconds": step_s, "label_to_idx": self.be.label_to_idx, "mode": self.mode,
                    "intent": self.intent, "note": self.note, "ph_time": now_ph_str(),
                    "verdicts": [r.verdict for r in results], "scores": [r.score for r in results]}
            np.savez_compressed(
                os.path.join(d, f"attempt_{self.attempt_no:03d}_{'+'.join(labels)}.npz"),
                capture=np.asarray(self.cap_frames, dtype=np.float32).reshape(-1, 122),
                probs=(np.asarray([r["probs"] for r in recs], dtype=np.float32) if recs else np.zeros((0, n_cls), np.float32)),
                labels=(np.asarray([r["label"] for r in recs]) if recs else np.asarray([], dtype="<U1")),
                windows=(np.asarray([r["frames"] for r in recs], dtype=np.float32) if recs
                         else np.zeros((0, ROLLING_WINDOW_FRAMES, 122), np.float32)),
                ends=np.asarray([r.get("end", -1) for r in recs], dtype=np.int32),
                meta=np.asarray(json.dumps(meta)))
        except Exception as exc:  # pragma: no cover
            print(f"(could not save the movement file: {exc})")

    def _whistle_done(self, ok):
        self.whistle_ok = ok
        self._log_attempt(self.step, whistle_ok=ok)
        self.whistle_flash_until = self.clock() + 1.2
        self._set_phase("whistle_result")

    def _handle_key(self, key):
        """Returns True to end the session loop."""
        if key in (255, -1):
            return False
        ch = chr(key).lower() if 0 <= key < 256 else ""
        if key == 27 or ch == "q":
            self._log_event("end_requested", "Q pressed")
            if self.mode in ("match_test", "practice"):
                self._mt_flush_run()
            if self.mode == "practice":
                self.summary = self._make_summary()
                return True
            if self.phase == "summary":
                self.phase = "done"
                return False
            self._finish()
            return False
        if ch == "m":
            self.mirror_display = not self.mirror_display
        elif ch == "p" and self.phase in ("practice", "intro", "result", "countdown", "capture"):
            if self.phase in ("countdown", "capture"):
                # pausing mid-attempt: this attempt is dropped (never graded or logged), earlier ones are kept, and the
                # same step starts again after resuming
                self._log_event("attempt_cancelled", f"paused during {self.phase}; not graded")
                self.cap_frames, self.cap_records = [], []
                self._set_phase("intro")
            self.paused = not self.paused
            self._log_event("paused" if self.paused else "resumed")
        elif ch == "t":
            new = "light" if _cv_theme_name == "dark" else "dark"
            trainer_ui.set_theme(new)
            set_cv_theme(new)
        elif ch == "w":
            self.whistle.manual()
            self.whistle_flash_until = self.clock() + 1.2
            self._log_event("whistle_key", "W pressed (manual whistle)")
        elif ch == "h":
            self.hint = not self.hint
            self._log_event("hint_on" if self.hint else "hint_off")
        elif ch == "[" and self.mode == "match_test":     # mirrors live_deployment.py's manual score keys
            self.team["left"] = max(0, self.team["left"] - 1)
            self._log_event("mt_manual_score", f"left -1 -> {self.team}")
        elif ch == "]" and self.mode == "match_test":
            self.team["left"] += 1
            self._log_event("mt_manual_score", f"left +1 -> {self.team}")
        elif ch == "-" and self.mode == "match_test":
            self.team["right"] = max(0, self.team["right"] - 1)
            self._log_event("mt_manual_score", f"right -1 -> {self.team}")
        elif ch in ("+", "=") and self.mode == "match_test":
            self.team["right"] += 1
            self._log_event("mt_manual_score", f"right +1 -> {self.team}")
        elif key in (32, 13):
            if self.phase == "intro":
                self.auto_go = True
                self._begin_step()
            elif self.phase == "result":
                if any(r.verdict == gg.VERDICT_NO_READING for r in self.results):
                    self._set_phase("intro")
                else:
                    self._advance()
            elif self.phase == "summary":
                self.phase = "done"
        elif ch == "r" and self.phase == "result" and self.mode == "drill" and not self.auto:
            self._log_event("retry", "R pressed")
            self._set_phase("intro")
        elif ch == "r" and self.phase == "summary" and self.mode != "practice":
            self._log_event("do_again", "R pressed on the summary: the same session starts again")
            self.restart = True
            self.phase = "done"
        elif ch in ("a", "d") and self.phase == "summary":
            self._review_step(-1 if ch == "a" else 1)
        return False

    # ---- rendering ----
    def _render(self, frame, feats, t):
        ui = np.zeros((UI_H, UI_W, 3), dtype=np.uint8)
        ui[:] = C_BG
        view = cv2.flip(frame, 1) if self.mirror_display else frame
        small, placement = fit_frame(view, CAM_BOX)
        ox, oy, nw, nh = placement
        ui[oy:oy + nh, ox:ox + nw] = small
        step = self.step
        target_side = None
        if step and step["kind"] in ("gesture", "pair"):
            lab = step["label"] if step["kind"] == "gesture" else step["labels"][0]
            target_side = "left" if lab.endswith("_left") else "right" if lab.endswith("_right") else None
        if self.phase not in ("summary",):
            draw_skeleton(ui, feats, placement, self.mirror_display, target_side if self.phase in ("countdown", "capture") else None)
        self._draw_top(ui)
        if self.phase == "summary":
            self._draw_summary(ui)
        else:
            self._draw_bottom(ui, t)
            self._draw_side(ui, t)
            self._draw_overlays(ui, t, placement)
            if self.mode == "sim":
                self._draw_scoreboard_banner(ui)
            elif self.mode == "match_test":
                self._draw_gesture_history_bar(ui)
                self._draw_last_decision_chip(ui, t)
            if self.phase in ("intro", "countdown"):
                self._draw_reference(ui, self.step)
        if self.paused:
            panel(ui, 250, 300, 650, 360, OV_AMBER, 0.9)
            put(ui, "PAUSED (press P)", 300, 340, 0.9, OV_ON, 2)
        return ui

    def _draw_top(self, ui):
        panel(ui, 0, 0, UI_W, TOP_H, C_PANEL)
        title = {"practice": "PRACTICE", "drill": "DRILL", "combo": "COMBO DRILL", "challenge": "CHALLENGE",
                "sim": "MATCH SIMULATION", "match_test": "MATCH TESTING"}[self.mode]
        put(ui, title, 20, 36, 0.9, C_GREEN, 2)
        put(ui, f"{self.cfg.name} level", 20 + tw(title, 0.9, 2) + 20, 36, 0.6, C_MUTED, 1)
        if self.intent != "normal":
            badge = "TEST: correct on purpose" if self.intent == "correct" else "TEST: WRONG on purpose"
            put(ui, badge, 20 + tw(title, 0.9, 2) + 20 + tw(f"{self.cfg.name} level", 0.6, 1) + 24, 36, 0.6, C_AMBER, 2)
        if self.mode == "practice":
            right = "no score in practice"
        else:
            acc = 100.0 * sum(1 for a in self.attempts if a["verdict"] == gg.VERDICT_CORRECT) / len(self.attempts) if self.attempts else 0
            right = f"Score {self.points}/{self.max_points}   Accuracy {acc:.0f}%"
        x = UI_W - tw(right, 0.7, 2) - 20
        put(ui, right, x, 36, 0.7, C_TEXT, 2)
        if self.auto and self.queue and self.phase != "summary" and self.mode != "sim":
            prog = f"Attempt {min(self.step_i + 1, len(self.queue))}/{len(self.queue)}"
            put(ui, prog, x - tw(prog, 0.6, 1) - 40, 36, 0.6, C_AMBER, 1)
        if self.mode == "sim":
            put(ui, "SCOREBOARD BELOW", x - tw("SCOREBOARD BELOW", 0.45, 1) - 40, 36, 0.45, C_MUTED, 1)
        elif self.mode == "match_test":
            a, b = self.team["left"], self.team["right"]
            score_txt = f"L {a} - {b} R"
            put(ui, score_txt, x - tw(score_txt, 0.55, 2) - 40, 36, 0.55, C_GREEN, 2)

    def _draw_scoreboard_banner(self, ui):
        """A big, unmissable scoreboard, drawn over the top of the camera view. Requested to be 'more apparent' than
        the small top-bar text: bold, high-contrast, colour-coded by team, with the serving side highlighted."""
        bx, by, bw = CAM_BOX[0] + 10, CAM_BOX[1] + 10, CAM_BOX[2] - 20
        bh = 64
        panel(ui, bx, by, bx + bw, by + bh, (18, 18, 18), alpha=0.82)
        cv2.rectangle(ui, (bx, by), (bx + bw, by + bh), C_AMBER, 2)
        a, b = self.team["left"], self.team["right"]
        left_col = C_GREEN if self.serving == "left" else C_TEXT
        right_col = C_GREEN if self.serving == "right" else C_TEXT
        mid_x = bx + bw // 2
        put(ui, "LEFT", bx + 24, by + 28, 0.6, left_col, 2)
        put(ui, str(a), bx + 24, by + 56, 1.1, left_col, 3)
        put(ui, "-", mid_x - 10, by + 48, 0.9, C_MUTED, 2)
        rb = str(b)
        put(ui, rb, bx + bw - 24 - tw(rb, 1.1, 3), by + 56, 1.1, right_col, 3)
        put(ui, "RIGHT", bx + bw - 24 - tw("RIGHT", 0.6, 2), by + 28, 0.6, right_col, 2)
        if self.serving:
            tag = f"{self.serving.upper()} TO SERVE"
            put(ui, tag, mid_x - tw(tag, 0.5, 1) // 2, by + 20, 0.5, C_GREEN, 1)

    def _draw_gesture_history_bar(self, ui):
        """Match Testing only: the sequence of gestures that actually COMMITTED (point awarded, authorization
        acknowledged, reason attached) -- mirrors live_deployment.py's own gesture_history pill-chip bar.
        Sits at the BOTTOM of the camera view (not the top), same as live_deployment.py's own layout, with
        the last-decision chip stacked just above it (see _draw_last_decision_chip)."""
        bx, bw = CAM_BOX[0] + 10, CAM_BOX[2] - 20
        bh = 44
        by = CAM_BOX[1] + CAM_BOX[3] - bh - 10
        panel(ui, bx, by, bx + bw, by + bh, (18, 18, 18), alpha=0.82)
        cv2.rectangle(ui, (bx, by), (bx + bw, by + bh), C_BLUE, 1)
        if not self.mt_history:
            txt = "(no gestures committed yet)"
            put(ui, txt, bx + (bw - tw(txt, 0.5, 1)) // 2, by + bh // 2 + 6, 0.5, C_MUTED, 1)
            return
        names = [gg.short_label(l) for l in self.mt_history]
        arrow = "  ->  "
        arrow_w = tw(arrow, 0.55, 2)
        chip_pad = 14
        widths = [tw(n, 0.55, 2) + chip_pad * 2 for n in names]
        total_w = sum(widths) + arrow_w * (len(names) - 1)
        x = bx + max(10, (bw - total_w) // 2)
        y1, y2 = by + 6, by + bh - 6
        for i, (n, w) in enumerate(zip(names, widths)):
            bg = C_PANEL2 if i % 2 == 0 else C_LINE
            is_last = i == len(names) - 1
            color = C_GREEN if is_last else C_TEXT
            cv2.rectangle(ui, (x, y1), (x + w, y2), bg, -1)
            if is_last:
                cv2.rectangle(ui, (x, y1), (x + w, y2), C_GREEN, 1, cv2.LINE_AA)
            put(ui, n, x + chip_pad, y2 - 10, 0.55, color, 2)
            x += w
            if i < len(names) - 1:
                put(ui, arrow, x, y2 - 10, 0.55, C_MUTED, 2)
                x += arrow_w

    def _draw_last_decision_chip(self, ui, t):
        """Small message just ABOVE the gesture-history bar, mirroring live_deployment.py's own
        draw_last_decision_chip -- shows the most recent mt_gate event for a few seconds."""
        d = self.mt_last_decision
        if d is None or t >= d["until"]:
            return
        bx, bw = CAM_BOX[0] + 10, CAM_BOX[2] - 20
        chip_w = min(bw, tw(d["text"], 0.5, 1) + 24)
        history_bh, history_gap = 44, 10
        y2 = CAM_BOX[1] + CAM_BOX[3] - history_bh - history_gap - 6
        y1 = y2 - 26
        panel(ui, bx, y1, bx + chip_w, y2, (30, 26, 24), alpha=0.85, border=True)
        put(ui, d["text"], bx + 12, y2 - 8, 0.5, d["color"], 1)

    def _mt_win_condition(self):
        """Match Testing win-by-two check, same rule as live_deployment.py's GAME_WIN_SCORE /
        GAME_WIN_BY_MARGIN (first to MT_WIN_SCORE, win by MT_WIN_BY_MARGIN). Returns the winning
        side or None. This is DISPLAY-ONLY and never stops scoring -- same deliberate design as
        live_deployment.py's own win-condition handling (see Table 13: "Auto-stop removed; scoring
        continues regardless of detected score" -- an earlier auto-stop froze scoring mid-set on a
        premature misdetection-triggered win, preventing further corrections from being logged).
        Match Testing follows that exact same reasoning rather than inventing different behavior."""
        a, b = self.team["left"], self.team["right"]
        if max(a, b) >= MT_WIN_SCORE and abs(a - b) >= MT_WIN_BY_MARGIN:
            return "left" if a > b else "right"
        return None

    def _build_scoreboard_canvas(self):
        """Match Testing's separate scoreboard window (mirrors live_deployment.py having SCOREBOARD as its own
        window instead of an overlay). Trainee's own left/right throughout -- see the left/right note in
        _mt_grade_run for why this never applies decision_engine.py's audience-facing flip."""
        canvas = np.zeros((SCOREBOARD_H, SCOREBOARD_W, 3), dtype=np.uint8)
        canvas[:] = C_BG
        winner = self._mt_win_condition()
        if winner:
            # a small strip ABOVE the score, never replacing it -- scoring keeps updating live
            # underneath exactly as before, matching live_deployment.py's own fix for this
            # (see _mt_win_condition's docstring for why an early version's freeze was wrong)
            txt = f"WIN CONDITION REACHED -- {winner.upper()} (still logging)"
            tw_ = tw(txt, 0.55, 1)
            panel(canvas, 0, 0, SCOREBOARD_W, 26, C_AMBER, alpha=0.9)
            put(canvas, txt, (SCOREBOARD_W - tw_) // 2, 18, 0.55, (20, 20, 20), 1)
        a, b = self.team["left"], self.team["right"]
        left_col = C_GREEN if self.serving == "left" else C_TEXT
        right_col = C_GREEN if self.serving == "right" else C_TEXT
        put(canvas, "LEFT", 50, 60, 0.9, left_col, 2)
        put(canvas, str(a), 50, 210, 4.2, left_col, 9)
        put(canvas, "-", SCOREBOARD_W // 2 - 12, 190, 2.2, C_MUTED, 5)
        rtxt = str(b)
        rw = tw(rtxt, 4.2, 9)
        put(canvas, rtxt, SCOREBOARD_W - 50 - rw, 210, 4.2, right_col, 9)
        rlab = "RIGHT"
        rlw = tw(rlab, 0.9, 2)
        put(canvas, rlab, SCOREBOARD_W - 50 - rlw, 60, 0.9, right_col, 2)
        if self.serving:
            tag = f"YOUR {self.serving.upper()} TO SERVE"
            tw_ = tw(tag, 0.55, 1)
            put(canvas, tag, (SCOREBOARD_W - tw_) // 2, SCOREBOARD_H - 20, 0.55, C_GREEN, 1)
        else:
            sub = "trainee's own left / right"
            sw = tw(sub, 0.45, 1)
            put(canvas, sub, (SCOREBOARD_W - sw) // 2, SCOREBOARD_H - 20, 0.45, C_MUTED, 1)
        return canvas

    def _draw_disclaimer(self, ui, x, w, y_bottom=UI_H - 10):
        lines = wrap(gg.DISCLAIMER, w, 0.38)
        y = y_bottom - 14 * (len(lines) - 1)
        for i, line in enumerate(lines):
            put(ui, line, x, y + i * 14, 0.38, C_MUTED, 1)

    def _ref_image(self, label, max_w, max_h):
        """Optional FIVB illustration from assets/signals/<label>.png (you add these yourself)."""
        key = (label, max_w, max_h)
        if key not in self._ref_cache:
            img = None
            path = os.path.join(trainer_ui.SIGNAL_IMAGE_DIR, f"{label}.png")
            if os.path.exists(path):
                raw = cv2.imread(path, cv2.IMREAD_COLOR)
                if raw is not None and raw.size:
                    sc = min(max_w / raw.shape[1], max_h / raw.shape[0], 1.0)
                    img = cv2.resize(raw, (max(1, int(raw.shape[1] * sc)), max(1, int(raw.shape[0] * sc))),
                                     interpolation=cv2.INTER_AREA)
            self._ref_cache[key] = img
        return self._ref_cache[key]

    def _draw_reference(self, ui, step):
        """While getting ready, show the FIVB picture(s) of the signal(s) in the top right of the camera view."""
        if step is None or step["kind"] == "whistle" or not self._reveal():
            return
        labels = step["labels"] if step["kind"] == "pair" else [step["label"]]
        max_h = 190 if len(labels) == 1 else 150
        x_right = CAM_BOX[0] + CAM_BOX[2] - 14
        y = TOP_H + 14
        for lab in labels:
            img = self._ref_image(lab, 230, max_h)
            if img is None:
                continue
            h, w = img.shape[:2]
            x = x_right - w
            cv2.rectangle(ui, (x - 5, y - 5), (x + w + 5, y + h + 19), (255, 255, 255), -1)
            ui[y:y + h, x:x + w] = img
            put(ui, "FIVB Diagram 11", x, y + h + 14, 0.36, (90, 90, 90), 1)
            y += h + 34

    def _reveal(self):
        """Signals are named unless it is the match simulation (where naming them gives the answer away)."""
        return self.mode != "sim" or self.level == "beginner" or self.hint

    def _step_names(self, step):
        if step["kind"] == "pair":
            return " then ".join(gg.pretty_label(l) for l in step["labels"])
        return gg.pretty_label(step["label"])

    def _draw_commit(self, ui, by, bw, bh):
        """Game-style message after a call, like a scoreboard: what was committed, or why not."""
        c = self.commit
        color = C_GREEN if c["ok"] else C_RED
        cv2.rectangle(ui, (0, by), (12, by + bh), color, -1)
        put(ui, c["title"], 28, by + 40, 1.0, color, 3)
        yy = by + 70
        shown = 0
        for k, line in enumerate(c["lines"]):
            for wl in wrap(line, bw - 60, 0.58):
                if shown >= 3:
                    return
                put(ui, wl, 28, yy, 0.58, C_TEXT if k < 2 else C_MUTED, 1)
                yy += 24
                shown += 1

    def _draw_bottom(self, ui, t):
        bx, by, bw, bh = BOTTOM_BOX
        panel(ui, bx, by, bx + bw, by + bh, C_PANEL2)
        step = self.step
        y = by + 32
        sim = self.mode == "sim"
        whistle = step is not None and bool(step.get("whistle"))
        two = step is not None and step["kind"] == "pair"
        if self.phase == "practice" and self.mode == "match_test":
            for i, line in enumerate(wrap("Perform naturally. Every signal is detected, graded and scored "
                                          "automatically. No set order.", bw - 40, 0.58)):
                put(ui, line, 20, y + i * 24, 0.58, C_TEXT, 1)
            if self.mt_last_result is not None:
                r = self.mt_last_result
                col = {gg.VERDICT_CORRECT: C_GREEN, gg.VERDICT_ALMOST: C_AMBER}.get(r.verdict, C_RED)
                put(ui, f"Last: {gg.pretty_label(r.target)}  {r.verdict}  {r.score}/100", 20, y + 54, 0.55, col, 1)
        elif self.phase == "practice":
            put(ui, "Do any signal. The system will tell you which one it sees.", 20, y, 0.65, C_TEXT, 1)
            lab = self.stable_label
            if lab in gg.SIGNALS:
                for i, line in enumerate(wrap("FIVB: " + gg.SIGNALS[lab]["fivb"], bw - 40, 0.55)):
                    put(ui, line, 20, y + 34 + i * 24, 0.55, C_AMBER, 1)
        elif self.phase == "intro" and step is not None:
            self._draw_intro_bottom(ui, by, bw, y, step)
        elif self.phase == "countdown":
            put(ui, "Get into the ready position: arms relaxed, facing the camera.", 20, y, 0.65, C_TEXT, 1)
            if whistle:
                msg = "At GO: blow the whistle, then give the signal right away."
            elif two:
                msg = "Show the first signal at GO, hold about 2 s, then go straight into the second."
            else:
                msg = "Perform the signal when you see GO, hold it, then lower your arms."
            put(ui, msg, 20, y + 28, 0.58, C_MUTED, 1)
            if step is not None and step.get("text"):
                for i, line in enumerate(wrap(step["text"], bw - 40, 0.52)[:2]):
                    put(ui, line, 20, y + 58 + i * 21, 0.52, C_TEXT, 1)
        elif self.phase == "capture":
            if whistle and two:
                msg = "Whistle, team to serve (hold 2 s), then the reason."
            elif whistle:
                msg = "Blow the whistle, then give the signal and hold it."
            elif two:
                msg = "Team to serve, hold 2 s, then go straight into the reason."
            else:
                msg = "Perform the signal now. Hold it steady, then lower your arms."
            put(ui, msg, 20, y, 0.62, C_GREEN, 2)
            if step is not None and step.get("text"):
                for i, line in enumerate(wrap(step["text"], bw - 40, 0.52)[:2]):
                    put(ui, line, 20, y + 32 + i * 21, 0.52, C_TEXT, 1)
        elif self.phase == "wait_whistle":
            src = "blow your whistle" if self.whistle.auto_active else "press W (no microphone detector)"
            put(ui, f"Waiting for the whistle: {src}.", 20, y, 0.65, C_AMBER, 2)
            if step is not None and step.get("text"):
                for i, line in enumerate(wrap(step["text"], bw - 40, 0.52)[:3]):
                    put(ui, line, 20, y + 32 + i * 21, 0.52, C_TEXT, 1)
        elif self.phase == "result" and self.results:
            if sim and self.commit:
                self._draw_commit(ui, by, bw, bh)
            else:
                notes = [r.note for r in self.results if r.note]
                msg = notes[0] if notes else "Check the panel on the right for details."
                if step is not None and (two or (sim and step["kind"] == "gesture")):
                    msg += "   Correct call: " + self._step_names(step) + "."
                if self.auto:
                    msg += "  The next step starts automatically; every score is listed at the end."
                for i, line in enumerate(wrap(msg, bw - 40, 0.55)[:3]):
                    put(ui, line, 20, y + i * 24, 0.55, C_TEXT, 1)
        if sim and self.commit_log and self.phase in ("intro", "countdown", "capture"):
            put(ui, "Last: " + fit(self.commit_log[-1], bw - 70, 0.46), 20, by + bh - 34, 0.46, C_MUTED, 1)
        if self.continuous and self.auto_go:
            start_keys = "Continuous: the next step starts by itself"
        else:
            start_keys = "SPACE start" + ("  (then it runs by itself)" if self.continuous else "")
        keys = {"practice": ("Q end match testing   P pause   M mirror   T theme   [ / ] left score   - / + right score"
                             if self.mode == "match_test"
                             else "Q end practice   P pause   M mirror   T theme"),
                "intro": start_keys + "   Q end session   M mirror   T theme" + ("   H hint" if sim else ""),
                "result": ("Continuous: next step in a moment" if self.continuous else "SPACE next")
                          + ("   R retry" if self.mode == "drill" and not self.auto else "") + "   Q end session",
                "countdown": "Q end session", "capture": "Q end session",
                "wait_whistle": "W = manual whistle   Q end session", "whistle_result": ""}.get(self.phase, "")
        if self.phase in ("capture", "countdown") and whistle:
            keys = "W = manual whistle   Q end session"
        put(ui, keys, 20, by + bh - 14, 0.5, C_MUTED, 1)

    def _draw_intro_bottom(self, ui, by, bw, y, step):
        sim = self.mode == "sim"
        sc, lh = (0.68, 26) if sim else (0.56, 23)
        lines = wrap(step["text"], bw - 40, sc)[:3] if step.get("text") else []
        for i, line in enumerate(lines):
            put(ui, line, 20, y + i * lh, sc, C_TEXT, 1)
        cur = y + lh * len(lines) + (4 if lines else 0)
        if step["kind"] == "whistle":
            msg = "Step: blow the whistle." if not lines else "Then: blow the whistle."
        elif step["kind"] == "pair":
            msg = (f"Signals: {self._step_names(step)}" if self._reveal()
                   else "Give the team to serve, then the reason.")
        elif sim:
            msg = (f"Signal needed: {self._step_names(step)}" if self._reveal() else "Give the correct signal.")
        elif self.mode == "challenge":
            msg = f"Signal {self.step_i + 1} of {len(self.queue)}: {self._step_names(step)}. One attempt each."
        elif self.auto:
            msg = f"Repetition {step.get('rep', 1)} of {step.get('n_reps', 1)}"
        else:
            msg = "Read the card on the right, then press SPACE when you are ready."
        if step.get("whistle") and not sim:
            msg += "   (blow the whistle first)"
        if cur + 18 <= by + BOTTOM_BOX[3] - 36:
            put(ui, msg, 20, cur + 18, 0.6, C_AMBER, 1)
        if self.auto and cur + 42 <= by + BOTTOM_BOX[3] - 36:
            prompt = "Get ready..." if self.auto_go else "Press SPACE to start. The steps then run automatically."
            put(ui, prompt, 20, cur + 42, 0.52, C_MUTED, 1)

    def _draw_side(self, ui, t):
        x0 = SIDE_X
        panel(ui, x0, TOP_H, UI_W, UI_H, C_PANEL)
        cv2.line(ui, (x0, TOP_H), (x0, UI_H), C_LINE, 1)
        pad = 18
        w = SIDE_W - 2 * pad
        step = self.step
        y = TOP_H + 34
        if self.phase == "practice":
            self._draw_side_practice(ui, x0 + pad, y, w)
        elif self.phase in ("intro", "countdown", "capture") and step is not None:
            if step["kind"] == "whistle":
                put(ui, "WHISTLE", x0 + pad, y, 0.85, C_AMBER, 2)
                for i, line in enumerate(wrap("Blow the whistle first, or press W. In a match the whistle comes before "
                                              "the signal (FIVB 22.2).", w, 0.55)):
                    put(ui, line, x0 + pad, y + 34 + i * 22, 0.55, C_TEXT, 1)
            else:
                if step.get("whistle"):
                    put(ui, "1. Blow the whistle   2. Signal", x0 + pad, y - 6, 0.55, C_AMBER, 2)
                    y += 34
                if step["kind"] == "pair":
                    self._draw_side_pair_card(ui, x0 + pad, y, w, step)
                elif self.phase == "capture":
                    self._draw_side_card(ui, x0 + pad, y, w, step)
                else:
                    self._draw_side_card_scrolled(ui, x0 + pad, y, w, step)
        elif self.phase in ("wait_whistle", "whistle_result"):
            ok = getattr(self, "whistle_ok", None)
            put(ui, "WHISTLE", x0 + pad, y, 0.85, C_AMBER, 2)
            if self.phase == "wait_whistle":
                left = max(0.0, WHISTLE_WAIT_SECONDS - (t - self.phase_t0))
                bar(ui, x0 + pad, y + 30, w, 12, left / WHISTLE_WAIT_SECONDS, C_AMBER)
                put(ui, f"{left:0.0f}s left", x0 + pad, y + 68, 0.55, C_MUTED, 1)
            else:
                put(ui, "Whistle heard! +10" if ok else "No whistle heard. 0 points", x0 + pad, y + 40, 0.65,
                    C_GREEN if ok else C_RED, 2)
        elif self.phase == "result" and self.results:
            if len(self.results) > 1:
                self._draw_side_pair_result(ui, x0 + pad, y, w)
            else:
                self._draw_side_result(ui, x0 + pad, y, w)
            self._draw_whistle_result(ui, x0 + pad)
            self._draw_disclaimer(ui, x0 + pad, w)

    def _draw_whistle_result(self, ui, x):
        state = self.whistle_state
        if state is None:
            return
        text, color = {"ok": ("Whistle: heard. +10 points", C_GREEN),
                       "late": ("Whistle: heard, but after you started. Blow it first. +5 points", C_AMBER),
                       "none": ("Whistle: not heard. 0 points", C_RED)}[state]
        put(ui, fit(text, SIDE_W - 36, 0.44), x, UI_H - 70, 0.44, color, 1)

    def _draw_side_card(self, ui, x, y, w, step, room=UI_H - 120):
        label = step["label"]
        s = gg.SIGNALS.get(label, {})
        hide = self.mode == "sim" and not self._reveal()
        compact = self.phase == "capture"
        if hide:
            put(ui, "SIGNAL", x, y, 0.8, C_GREEN, 2)
            for i, line in enumerate(wrap("The signal is not shown at this level. Press H for a hint.", w, 0.55)):
                put(ui, line, x, y + 32 + i * 22, 0.55, C_TEXT, 1)
        else:
            for i, line in enumerate(wrap(s.get("title", label), w, 0.8, 2)):
                put(ui, line, x, y + i * 30, 0.8, C_GREEN, 2)
            y += 30 * len(wrap(s.get("title", label), w, 0.8, 2)) + 6
            put(ui, "FIVB signal", x, y, 0.5, C_BLUE, 1)
            y += 8
            for line in wrap(s.get("fivb", ""), w, 0.55):
                y += 22
                put(ui, line, x, y, 0.55, C_TEXT, 1)
            if not compact and not (self.mode in ("challenge", "drill") and self.level == "referee"):
                y += 28
                put(ui, "How", x, y, 0.5, C_BLUE, 1)
                for i, stp in enumerate(s.get("howto", []), 1):
                    for j, line in enumerate(wrap(f"{i}. {stp}", w, 0.46)):
                        y += 18
                        put(ui, line, x, y, 0.46, C_MUTED if j else C_TEXT, 1)
            if not compact and label in gg.RULES and self.phase in ("intro", "countdown"):
                y += 26
                put(ui, "What is checked", x, y, 0.5, C_BLUE, 1)
                for it in gg.graded_summary(label):
                    mark = {"Required": "R", "Important": "I", "Scored": "S"}[it["effect"]]
                    for line in label_lines(f"[{mark}] {it['label']}", w, 0.4):
                        y += 17
                        put(ui, line, x, y, 0.4, C_TEXT if mark != "S" else C_MUTED, 1)
                need = "Only 100 is correct" if self.cfg.correct_cut >= 100 else f"{self.cfg.correct_cut}+ is correct"
                for line in wrap(f"R required, I important, S scored. {need}.", w, 0.38):
                    y += 16
                    put(ui, line, x, y, 0.38, C_MUTED, 1)
            if not compact and s.get("not_graded") and y < room:
                y += 24
                put(ui, "Not graded by the camera", x, y, 0.43, C_AMBER, 1)
                for line in wrap(s["not_graded"], w, 0.42):
                    y += 16
                    put(ui, line, x, y, 0.42, C_MUTED, 1)
        if self.phase == "capture":
            self._draw_capture_meter(ui, x, UI_H - 150, w, label)

    def _draw_side_card_scrolled(self, ui, x, y, w, step):
        """Before an attempt the signal card can be longer than the panel (long how-to plus what is checked): draw it on
        a taller sheet and show the part the mouse wheel has scrolled to, with a scrollbar when there is more."""
        key = (step["label"], self.mode, self.level, self.phase, self.hint, x, y, C_PANEL)
        cache = getattr(self, "_card_cache", None)
        if cache is None or cache[0] != key:                    # the card is static: draw it once per signal/phase
            sheet = np.empty((2 * UI_H, UI_W, 3), np.uint8)
            sheet[:] = C_PANEL
            self._draw_side_card(sheet, x, y, w, step, room=2 * UI_H - 120)
            ink = np.where((sheet[:, SIDE_X + 1:] != np.array(C_PANEL, np.uint8)).any(axis=2).any(axis=1))[0]
            self._card_cache = cache = (key, sheet, max(0, (int(ink.max()) + 14 if len(ink) else 0) - UI_H))
        _key, sheet, max_off = cache
        if getattr(self, "_scroll_step", None) != self.step_i:          # a new signal starts at the top
            self._scroll_step, self.side_scroll = self.step_i, 0
        off = self.side_scroll = min(max(0, getattr(self, "side_scroll", 0)), max_off)
        top = y - 30                                    # the card's first line; anything above it (whistle line) stays
        ui[top:UI_H, SIDE_X + 1:] = sheet[top + off:UI_H + off, SIDE_X + 1:]
        if max_off:
            track = UI_H - TOP_H - 8
            th = max(30, track * (UI_H - TOP_H) // (UI_H - TOP_H + max_off))
            ty = TOP_H + 4 + (track - th) * off // max_off
            cv2.rectangle(ui, (UI_W - 6, ty), (UI_W - 2, ty + th), C_MUTED, -1)
            if off < max_off:
                panel(ui, SIDE_X + 1, UI_H - 22, UI_W - 8, UI_H, C_PANEL)
                put(ui, "More below: scroll with the mouse wheel", x, UI_H - 7, 0.4, C_AMBER, 1)

    def _on_mouse(self, event, _x, _y, flags, _param):
        if event == cv2.EVENT_MOUSEWHEEL:
            self.side_scroll = getattr(self, "side_scroll", 0) + (-48 if cv2.getMouseWheelDelta(flags) > 0 else 48)

    def _draw_side_pair_card(self, ui, x, y, w, step):
        if self.mode == "sim" and not self._reveal():
            put(ui, "SIGNALS", x, y, 0.8, C_GREEN, 2)
            for i, line in enumerate(wrap("Team to serve first, then the reason. The signals are not shown at this "
                                          "level. Press H for a hint.", w, 0.55)):
                put(ui, line, x, y + 32 + i * 22, 0.55, C_TEXT, 1)
        else:
            put(ui, "TWO SIGNALS, ONE AFTER THE OTHER", x, y, 0.55, C_BLUE, 1)
            y += 12
            for k, lab in enumerate(step["labels"], 1):
                s = gg.SIGNALS.get(lab, {})
                for i, line in enumerate(wrap(f"{k}. {s.get('title', lab)}", w, 0.72, 2)):
                    y += 26
                    put(ui, line, x, y, 0.72, C_GREEN, 2)
                for line in wrap(s.get("fivb", ""), w, 0.5):
                    y += 20
                    put(ui, line, x, y, 0.5, C_TEXT, 1)
                if k == 2 and lab == "double_contact" and step.get("ctxs") and step["ctxs"][1]:
                    y += 20
                    side = step["ctxs"][1]["side"]
                    put(ui, f"Use your {side.upper()} hand (team at fault).", x, y, 0.5, C_AMBER, 1)
                y += 14
            if self.phase != "capture":
                y += 10
                put(ui, "FIVB 22.2.3.1: the team to serve first,", x, y, 0.46, C_MUTED, 1)
                put(ui, "then the nature of the fault.", x, y + 18, 0.46, C_MUTED, 1)
        if self.phase == "capture":
            self._draw_pair_meter(ui, x, UI_H - 170, w, step)

    def _draw_whistle_row(self, ui, x, y):
        step = self.step
        if step is None or not step.get("whistle"):
            return
        heard = self._whistle_seen or self.whistle.first_since(self.phase_t0) is not None
        draw_icon(ui, "pass" if heard else "unverified", x + 8, y + 8, 8)
        put(ui, "Whistle heard" if heard else "Waiting for the whistle (or press W)", x + 26, y + 14, 0.5,
            C_GREEN if heard else C_AMBER, 1)

    def _draw_pair_meter(self, ui, x, y, w, step):
        self._draw_whistle_row(ui, x, y - 32)
        labels = [r["label"] for r in self.cap_records]
        for k, lab in enumerate(step["labels"]):
            seen = lab in labels
            draw_icon(ui, "pass" if seen else "unverified", x + 8, y + 8 + k * 30, 8)
            put(ui, fit(f"{k + 1}. " + gg.short_label(lab), w - 34, 0.5), x + 26, y + 14 + k * 30, 0.5,
                C_TEXT if seen else C_MUTED, 1)
        rec = self.last_rec
        now = gg.short_label(rec["label"]) if rec is not None and rec["label"] != gg.NOTHING_LABEL else "nothing yet"
        put(ui, "Seeing: " + fit(now, w - 70, 0.5), x, y + 78, 0.5, C_MUTED, 1)
        left = max(0.0, self.cap_seconds - (self.clock() - self.phase_t0))
        bar(ui, x, y + 96, w, 8, left / self.cap_seconds, C_BLUE)

    def _draw_capture_meter(self, ui, x, y, w, target):
        self._draw_whistle_row(ui, x, y - 34)
        rec = self.last_rec
        p = float(rec["probs"][self.be.label_to_idx[target]]) if rec is not None else 0.0
        seen = gg.short_label(rec["label"]) if rec is not None and rec["label"] != gg.NOTHING_LABEL else "nothing yet"
        put(ui, "Recognition of this signal", x, y, 0.5, C_MUTED, 1)
        bar(ui, x, y + 10, w, 16, p, C_GREEN if p >= self.cfg.rec_threshold else C_AMBER, marker=self.cfg.rec_threshold)
        put(ui, f"{p:.0%}   (needs {self.cfg.rec_threshold:.0%})", x, y + 48, 0.5, C_TEXT, 1)
        put(ui, "Seeing: " + fit(seen, w - 60, 0.5), x, y + 72, 0.5, C_MUTED, 1)
        left = max(0.0, self.cap_seconds - (self.clock() - self.phase_t0))
        bar(ui, x, y + 90, w, 8, left / self.cap_seconds, C_BLUE)

    def _draw_side_pair_result(self, ui, x, y, w):
        step = self.step
        for k, (lab, r) in enumerate(zip(step["labels"], self.results)):
            color = VERDICT_COLOR[r.verdict]
            put(ui, fit(f"{k + 1}. " + gg.short_label(lab), w, 0.55), x, y, 0.55, C_BLUE, 1)
            put(ui, VERDICT_TEXT[r.verdict], x, y + 34, 0.9, color, 2)
            put(ui, f"{r.score}/100  +{r.points} session pts", x + 165, y + 32, 0.46, C_TEXT, 1)
            yy = y + 52
            if r.verdict != gg.VERDICT_NO_READING:
                for c in r.checks[:6]:
                    draw_icon(ui, c.status, x + 8, yy + 6, 6)
                    put(ui, fit(c.label, w - 26, 0.42), x + 22, yy + 10, 0.42, C_TEXT if c.status == "pass" else C_MUTED, 1)
                    yy += 18
            _heading, items = feedback_view(r)
            shown = 0
            for text, kind in items:
                if kind == "cam" or shown >= 2:
                    continue
                shown += 1
                for j, line in enumerate(wrap(text, w, 0.44)[:2]):
                    put(ui, ("- " if (j == 0 and kind == "tip") else "") + line, x, yy + 12, 0.44,
                        C_GREEN if kind == "ok" else C_AMBER, 1)
                    yy += 17
            y = max(yy + 26, y + 150)
            cv2.line(ui, (x, y - 16), (x + w, y - 16), C_LINE, 1)
        for i, line in enumerate(wrap("Correct call: " + self._step_names(step), w, 0.42)[:2]):
            put(ui, line, x, min(y + 6 + i * 16, UI_H - 78), 0.42, C_MUTED, 1)

    def _draw_side_result(self, ui, x, y, w, r=None):
        r = r or self.result
        color = VERDICT_COLOR[r.verdict]
        put(ui, VERDICT_TEXT[r.verdict], x, y + 6, 1.15, color, 3)
        if r.verdict == gg.VERDICT_NO_READING:
            lines = wrap(r.feedback[0], w, 0.55)
            for i, line in enumerate(lines):
                put(ui, line, x, y + 50 + i * 22, 0.55, C_TEXT, 1)
            if self.no_reading_streak >= NO_READING_CAMERA_CHECK and self.phase == "result":
                msg = (f"No one has been seen for {self.no_reading_streak} attempts in a row. If the camera picture is "
                       "frozen, black or shows a placeholder, the camera is disconnected: reconnect it, then press "
                       "SPACE to try again, or Q to stop. Your results so far are saved.")
                for i, line in enumerate(wrap(msg, w, 0.55)):
                    put(ui, line, x, y + 70 + (len(lines) + i) * 22, 0.55, C_AMBER, 1 if i else 2)
            return
        put(ui, f"Signal score {r.score}/100   +{r.points} session pts", x, y + 40, 0.56, C_TEXT, 2)
        cut = gg.LEVEL_CONFIG[r.level].correct_cut
        cut_txt = ("Only 100 counts as CORRECT at this level." if cut >= 100
                   else f"Counts as CORRECT from {cut}. 100 is not needed.")
        put(ui, fit(cut_txt, w, 0.42), x, y + 58, 0.42, C_MUTED, 1)
        seen = detected_label(r)
        seen_txt = "nothing" if seen == gg.NOTHING_LABEL else gg.short_label(seen)
        put(ui, fit(f"Recognised as: {seen_txt}", w, 0.45), x, y + 80, 0.45,
            C_GREEN if seen == r.target else C_AMBER, 1)
        put(ui, "Confidence", x, y + 102, 0.45, C_MUTED, 1)     # the model's best probability for the target
        bar(ui, x + 110, y + 92, w - 170, 10, r.best_prob, C_GREEN if r.best_prob >= 0.5 else C_MUTED)
        put(ui, f"{r.best_prob:.2f}", x + w - 52, y + 102, 0.45, C_TEXT, 1)
        y += 116
        rows = [("Recognition", r.parts["recognition"], gg.W_RECOGNITION), ("Distinctness", r.parts["distinctness"], gg.W_DISTINCT),
                ("Hold", r.parts["hold"], gg.W_HOLD), ("FIVB form", r.parts["form"], gg.W_FORM),
                ("Ready pos.", r.parts["ready_position"], gg.W_READY)]
        for name, val, mx in rows:
            put(ui, f"{name}", x, y + 12, 0.45, C_MUTED, 1)
            bar(ui, x + 110, y + 2, w - 170, 10, val / mx, color)
            put(ui, f"{val:.0f}/{mx}", x + w - 52, y + 12, 0.45, C_TEXT, 1)
            y += 22
        y += 8
        # The panel is a fixed-size image (a bigger window only scales it), so lay out to fit: the tips are what the
        # trainee needs most, so they get their room first; passed checks fold into one line when space is short, and
        # a tip that still does not fit is left for the session report rather than cut mid-sentence.
        bottom = UI_H - 78
        heading, items = feedback_view(r)
        tips = [(kind, wrap(text, w, 0.46 if kind != "cam" else 0.42)) for text, kind in items]
        tips_h = (20 if heading else 0) + 19 * sum(len(lines) for _k, lines in tips)
        rows = [(c, label_lines(c.label, w - 30, 0.45)) for c in r.checks]
        if y + 28 + 21 * sum(len(lines) for _c, lines in rows) + tips_h > bottom:
            passed = [c for c, _l in rows if c.status == "pass"]
            rows = [(c, lines) for c, lines in rows if c.status != "pass"]
            if passed:
                rows.append((None, [f"{len(passed)} other check{'s' if len(passed) > 1 else ''} passed"]))
        put(ui, "Checklist", x, y + 8, 0.5, C_BLUE, 1)
        y += 18
        for c, lines in rows:
            draw_icon(ui, "pass" if c is None else c.status, x + 8, y + 8, 7)
            for line in lines:
                put(ui, line, x + 24, y + 12, 0.45, C_TEXT if c is None or c.status == "pass" else C_MUTED, 1)
                y += 21
        y += 10
        if heading:
            put(ui, heading, x, y + 8, 0.5, C_AMBER, 1)
            y += 20
        for kind, lines in tips:
            col = {"ok": C_GREEN, "tip": C_TEXT, "cam": C_MUTED}[kind]
            if y + 19 * len(lines) > bottom:
                put(ui, "More tips in the session report.", x, y + 12, 0.42, C_MUTED, 1)
                return
            for j, line in enumerate(lines):
                put(ui, ("- " if (j == 0 and kind == "tip") else "  " if kind == "tip" else "") + line, x, y + 12,
                    0.46 if kind != "cam" else 0.42, col, 1)
                y += 19

    def _draw_side_practice(self, ui, x, y, w):
        lab = self.stable_label
        put(ui, "YOU ARE SHOWING", x, y, 0.5, C_MUTED, 1)
        if lab != gg.NOTHING_LABEL:
            for i, line in enumerate(wrap(gg.pretty_label(lab), w, 0.8, 2)):
                put(ui, line, x, y + 34 + i * 30, 0.8, C_GREEN, 2)
        else:
            put(ui, "(no signal yet)", x, y + 34, 0.8, C_MUTED, 2)
        y += 100
        put(ui, "Model confidence", x, y, 0.5, C_BLUE, 1)
        probs = self.last_rec["probs"] if self.last_rec is not None else np.zeros(len(self.be.real_labels))
        for i, lab_i in enumerate(self.be.real_labels):
            yy = y + 16 + i * 22
            put(ui, fit(gg.short_label(lab_i), 190, 0.42), x, yy + 11, 0.42, C_TEXT if lab_i == lab else C_MUTED, 1)
            bar(ui, x + 195, yy + 2, w - 245, 9, probs[i], C_GREEN if probs[i] >= 0.5 else C_MUTED)
            put(ui, f"{probs[i]:.2f}", x + w - 40, yy + 11, 0.42, C_MUTED, 1)
        y += 16 + len(self.be.real_labels) * 22 + 14
        if self.practice_checks:
            put(ui, "Live FIVB checklist", x, y, 0.5, C_BLUE, 1)
            for c in self.practice_checks:
                y += 21
                draw_icon(ui, c.status, x + 8, y - 4, 7)
                put(ui, fit(c.label, w - 30, 0.43), x + 24, y, 0.43, C_TEXT if c.status == "pass" else C_MUTED, 1)
            y += 30
        if self.mode in ("match_test", "practice") and self.mt_last_result is not None:
            r = self.mt_last_result
            col = {gg.VERDICT_CORRECT: C_GREEN, gg.VERDICT_ALMOST: C_AMBER}.get(r.verdict, C_RED)
            put(ui, "Last graded hold", x, y, 0.5, C_BLUE, 1)
            y += 22
            tail = f"+{r.points} pts" if self.mode == "match_test" else "not counted, this is Practice"
            put(ui, fit(f"{gg.pretty_label(r.target)}: {r.verdict}  {r.score}/100  {tail}", w, 0.46), x, y, 0.46, col, 1)
            for f in r.feedback[:2]:
                y += 19
                put(ui, fit("- " + f, w, 0.4), x, y, 0.4, C_MUTED, 1)

    def _draw_overlays(self, ui, t, placement):
        ox, oy, nw, nh = placement
        cx, cy = ox + nw // 2, oy + nh // 2
        if self.phase == "countdown":
            left = self.cd_seconds - (t - self.phase_t0)
            txt = str(int(np.ceil(left))) if left > 0.05 else "GO"
            sc = 5.0
            w = tw(txt, sc, 10)
            put(ui, txt, cx - w // 2, cy + 60, sc, OV_AMBER, 10)
        elif self.phase == "capture":
            cv2.circle(ui, (ox + 30, oy + 30), 9, OV_RED, -1, cv2.LINE_AA)
            put(ui, "REC  GO", ox + 48, oy + 38, 0.7, OV_RED, 2)
        if t < self.whistle_flash_until:
            txt = "WHISTLE!"
            w = tw(txt, 1.4, 3)
            panel(ui, cx - w // 2 - 20, oy + 10, cx + w // 2 + 20, oy + 70, OV_AMBER, 0.9)
            put(ui, txt, cx - w // 2, oy + 55, 1.4, OV_ON, 3)
        if self.banner:
            put(ui, self.banner, ox + 20, oy + nh - 20, 0.7, OV_RED, 2)

    def _draw_summary(self, ui):
        s = self.summary
        panel(ui, 0, TOP_H, UI_W, UI_H, C_PANEL)
        cv2.line(ui, (SIDE_X, TOP_H), (SIDE_X, UI_H), C_LINE, 1)
        y = TOP_H + 48
        put(ui, "SESSION COMPLETE", 40, y, 1.0, C_GREEN, 3)
        put(ui, f"{self.trainee['display_name']}  |  {s['mode'].upper()}  |  {self.cfg.name}", 40, y + 28, 0.55, C_MUTED, 1)
        y += 70
        stat = [("Trainee score", f"{s['points']}/{s['max_points']}"), ("Accuracy", f"{s['accuracy_pct']:.0f}%"),
                ("Correct", f"{s['correct']}/{s['attempts']}"), ("Avg signal score", f"{s['avg_gesture_score']:.0f}/100")]
        for i, (k, v) in enumerate(stat):
            xx = 40 + i * 215
            put(ui, k, xx, y, 0.5, C_MUTED, 1)
            put(ui, v, xx, y + 36, 0.95, C_TEXT, 2)
        y += 70
        per_txt = "   ".join(f"{gg.short_label(k)} {d['correct']}/{d['n']}" for k, d in s["per_signal"].items()
                               if k != "whistle")
        if per_txt:
            put(ui, fit("Correct per signal:  " + per_txt, SIDE_X - 80, 0.47), 40, y, 0.47, C_TEXT, 1)
            y += 22
            needs = ", ".join(gg.short_label(k) for k in s.get("needs_practice", [])) or "none, well done"
            put(ui, fit("Needs more practice:  " + needs, SIDE_X - 80, 0.47), 40, y, 0.47, C_AMBER, 1)
            y += 30
        if s.get("team_points"):
            put(ui, f"Scoreboard from your correct calls:  team on your LEFT {s['team_points']['left']}  -  {s['team_points']['right']} RIGHT",
                40, y, 0.55, C_AMBER, 1)
            y += 28
        put(ui, "Attempt by attempt", 40, y, 0.55, C_BLUE, 1)
        y += 8
        max_rows = max(1, (UI_H - 70 - y) // 26)
        rows = list(enumerate(self.attempts))
        first = 0
        if self.review_i is not None and len(rows) > max_rows:
            first = min(max(0, self.review_i - max_rows // 2), len(rows) - max_rows)
        for idx, a in rows[first:first + max_rows]:
            y += 26
            sel = idx == self.review_i
            if sel:
                panel(ui, 30, y - 18, SIDE_X - 20, y + 6, C_PANEL2)
            name = "Whistle" if a["kind"] == "whistle" else gg.short_label(a["target"])
            put(ui, f"{idx + 1}", 40, y, 0.5, C_MUTED, 1)
            put(ui, fit(name, 330, 0.5), 80, y, 0.5, C_TEXT, 1)
            put(ui, a["verdict"], 430, y, 0.5, VERDICT_COLOR.get(a["verdict"], C_TEXT), 1)
            put(ui, f"{a['score']}/100  +{a['points']}", 590, y, 0.5, C_TEXT, 1)
        again = "     R  do it again" if self.mode != "practice" else ""
        put(ui, "A / D  review each attempt" + again + "     SPACE  back to the menu", 40, UI_H - 26, 0.55, C_AMBER, 1)
        self._draw_disclaimer(ui, 40, SIDE_X - 80, UI_H - 50)
        # right column: full breakdown of the selected attempt
        if self.review_i is not None:
            a = self.attempts[self.review_i]
            res = self.attempt_results[self.review_i]
            put(ui, f"Attempt {self.review_i + 1}: {fit(gg.short_label(a['target']), SIDE_W - 130, 0.5)}",
                SIDE_X + 18, TOP_H + 26, 0.5, C_BLUE, 1)
            if res is not None:
                self._draw_side_result(ui, SIDE_X + 18, TOP_H + 60, SIDE_W - 36, res)

    # ---- outputs ----
    def _whistle_rows(self):
        """Every whistle event of this session with its time, source (auto = microphone detector, manual = W key), the
        confidence (auto only), and how it was used. Like the whistle logs of the live system."""
        getter = getattr(self.whistle, "events_since", None)
        events = getter(self._t0 if self._t0 is not None else 0.0) if getter else []
        rows = []
        judged = set()
        for ev in events:
            win = next((w for w in self.capture_windows if w["start"] <= ev["t"] <= w["end"] + 0.5), None)
            try:
                when = datetime.datetime.fromtimestamp(ev["t"], PH_TZ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            except (OverflowError, OSError, ValueError):
                when = ""
            row = {"ph_time": when, "seconds_from_session_start": round(ev["t"] - (self._t0 or 0.0), 2),
                   "source": ev["source"], "confidence": "" if ev.get("confidence") is None else round(float(ev["confidence"]), 3),
                   "attempt_no": "", "signal": "", "seconds_after_go": "", "signal_onset_estimate_s": "",
                   "judged_as": "outside a capture (not used)"}
            if win is not None:
                row.update(attempt_no=win["step"], signal=win["labels"], seconds_after_go=round(ev["t"] - win["start"], 2),
                           signal_onset_estimate_s="" if win["onset"] is None else round(win["onset"] - win["start"], 2))
                if win["step"] in judged:
                    row["judged_as"] = "extra whistle in the same capture"
                else:
                    judged.add(win["step"])
                    row["judged_as"] = {"ok": "heard (10 points)", "late": "late (5 points)",
                                        "none": "not heard"}.get(win["state"], "heard")
            rows.append(row)
        return rows

    def _write_whistle_log(self):
        rows = self._whistle_rows()
        self.whistle_rows = rows
        if not rows:
            return
        cols = ["ph_time", "seconds_from_session_start", "source", "confidence", "attempt_no", "signal", "seconds_after_go",
                "signal_onset_estimate_s", "judged_as"]
        with open(os.path.join(self.dir, "whistle_events.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    def _write_practice_log(self):
        if not self.practice_log:
            return
        cols = ["ph_time", "seconds_from_start", "detected_label", "detected_name", "best_prob", "checks_passed",
                "checks_total"]
        with open(os.path.join(self.dir, "practice_log.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(self.practice_log)
        with open(os.path.join(self.dir, "practice_summary.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["signal", "times_detected", "avg_checks_passed", "avg_checks_total"])
            for row in practice_summary_rows(self.practice_log):
                w.writerow(row)

    def _write_practice_scores(self):
        if not self.practice_scores:
            return
        cols = ["ph_time", "seconds_from_start", "target", "verdict", "score", "hold_s",
                "score_beginner", "verdict_beginner", "score_standard", "verdict_standard",
                "score_referee", "verdict_referee"]
        with open(os.path.join(self.dir, "practice_scores.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(self.practice_scores)

    def _save_outputs(self):
        try:
            self._write_whistle_log()
        except Exception:
            self.error_trace += "\n" + traceback.format_exc()
        try:
            self._write_practice_log()
        except Exception:
            self.error_trace += "\n" + traceback.format_exc()
        try:
            self._write_practice_scores()
        except Exception:
            self.error_trace += "\n" + traceback.format_exc()
        with open(os.path.join(self.dir, "attempts.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLS)
            w.writeheader()
            w.writerows(self.attempts)
        with open(os.path.join(self.dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({**self.summary, "trainee_id": self.trainee["trainee_id"], "intent": self.intent, "note": self.note,
                       "consent_version": trainer_ui.CONSENT_VERSION}, f, indent=2)
        if self.error_trace:
            with open(os.path.join(self.dir, "error.log"), "w", encoding="utf-8") as f:
                f.write(self.error_trace)
        write_report(os.path.join(self.dir, "report.html"), self.trainee, self.summary, self.attempts,
                     getattr(self, "whistle_rows", []), self.practice_log, self.practice_scores)


def practice_summary_rows(practice_log):
    """One row per detected signal: how many times it showed up, and how thoroughly formed it was, on average, when
    it did. This is the table to paste into the paper for Practice mode, rather than the raw per-event log."""
    per = {}
    for r in practice_log:
        lab = r["detected_label"]
        if lab == gg.NOTHING_LABEL:
            continue
        d = per.setdefault(lab, {"n": 0, "passed": 0, "total": 0})
        d["n"] += 1
        d["passed"] += r["checks_passed"]
        d["total"] += r["checks_total"]
    rows = []
    for lab, d in sorted(per.items(), key=lambda kv: -kv[1]["n"]):
        rows.append([gg.pretty_label(lab), d["n"], round(d["passed"] / max(1, d["n"]), 2),
                    round(d["total"] / max(1, d["n"]), 2)])
    return rows


def practice_score_summary_rows(practice_scores):
    """One row per signal from the fully graded Practice holds: how many times, and the average score at each
    difficulty level. Unlike practice_summary_rows (checks_passed/total from a live snapshot), this uses a real
    grade_attempt() score for the WHOLE hold, at all three levels."""
    per = {}
    for r in practice_scores:
        d = per.setdefault(r["target"], {"n": 0, "beginner": 0, "standard": 0, "referee": 0})
        d["n"] += 1
        for lv in gg.LEVELS:
            d[lv] += r[f"score_{lv}"]
    rows = []
    for lab, d in sorted(per.items(), key=lambda kv: -kv[1]["n"]):
        rows.append([gg.pretty_label(lab), d["n"]] + [round(d[lv] / d["n"], 1) for lv in gg.LEVELS])
    return rows


def write_report(path, trainee, summary, attempts, whistle_rows=None, practice_log=None, practice_scores=None):
    e = html.escape
    whistle_html = ""
    if whistle_rows:
        wr = "".join(
            f"<tr><td>{e(str(w['ph_time']))}</td><td>{e(str(w['source']))}</td><td>{e(str(w['confidence']))}</td>"
            f"<td>{e(str(w['attempt_no']))}</td><td>{e(str(w['signal']))}</td><td>{e(str(w['seconds_after_go']))}</td>"
            f"<td>{e(str(w['judged_as']))}</td></tr>" for w in whistle_rows)
        auto = sum(1 for w in whistle_rows if w["source"] == "auto")
        whistle_html = (f"<h2>Whistle log</h2><p>{len(whistle_rows)} whistle event(s): {auto} heard by the microphone detector "
                        f"(auto), {len(whistle_rows) - auto} from the W key (manual). Also saved as whistle_events.csv.</p>"
                        "<table><tr><th>Time (PH)</th><th>Source</th><th>Confidence</th><th>Attempt</th><th>Signal</th>"
                        f"<th>Seconds after GO</th><th>Judged as</th></tr>{wr}</table>")
    def all_levels_cell(a):
        parts = []
        for lv, tag in (("beginner", "B"), ("standard", "S"), ("referee", "R")):
            sc, vd = a.get(f"score_{lv}"), a.get(f"verdict_{lv}")
            if sc not in (None, "") and vd not in (None, ""):
                parts.append(f"{tag}:{sc} {vd[0]}")           # e.g. "B:95 C" (C/A/I for Correct/Almost/Incorrect)
        return " &nbsp; ".join(parts)

    rows = "".join(
        f"<tr><td>{e(a['ph_time'])}</td><td>{e(a['target'])}</td><td>{e(str(a.get('detected', '')))}</td>"
        f"<td style='font-weight:bold;color:{ {'CORRECT': '#2e7d32', 'ALMOST': '#e65100', 'INCORRECT': '#c62828'}.get(a['verdict'], '#333') }'>{e(a['verdict'])}</td>"
        f"<td>{a['score']}</td><td>{a['points']}</td><td>{e(str(a['hold_s']))}</td>"
        f"<td style='font-size:12px'>{all_levels_cell(a)}</td>"
        f"<td>{e(a['failed_checks'])}</td>"
        f"<td>{e(a['feedback'])}</td><td>{e(str(a.get('feedback_latency_s', '')))}</td></tr>" for a in attempts)         or "<tr><td colspan='11'>(no graded attempts)</td></tr>"
    per = "".join(f"<tr><td>{e(k)}</td><td>{v['correct']}/{v['n']}</td><td>{v['score_sum'] / max(1, v['n']):.0f}</td></tr>"
                  for k, v in summary["per_signal"].items())
    team = ""
    if summary.get("team_points"):
        t = summary["team_points"]
        team = f"<p><b>Scoreboard (from the trainee's correct calls):</b> team on your LEFT {t['left']} - {t['right']} RIGHT</p>"
    practice_scores_html = ""
    if practice_scores:
        score_summary = practice_score_summary_rows(practice_scores)
        ssr = "".join(f"<tr><td>{e(str(row[0]))}</td><td>{row[1]}</td><td>{row[2]}</td><td>{row[3]}</td><td>{row[4]}</td></tr>"
                     for row in score_summary)
        psr = "".join(
            f"<tr><td>{e(str(r['ph_time']))}</td><td>{e(str(r['seconds_from_start']))}</td>"
            f"<td>{e(gg.pretty_label(r['target']))}</td>"
            f"<td style='font-weight:bold;color:{ {'CORRECT': '#2e7d32', 'ALMOST': '#e65100', 'INCORRECT': '#c62828'}.get(r['verdict'], '#333') }'>{e(r['verdict'])}</td>"
            f"<td>{r['score']}</td><td>{r['hold_s']}</td>"
            f"<td>B:{r['score_beginner']} {e(r['verdict_beginner'][0])} &nbsp; S:{r['score_standard']} {e(r['verdict_standard'][0])} "
            f"&nbsp; R:{r['score_referee']} {e(r['verdict_referee'][0])}</td></tr>" for r in practice_scores)
        practice_scores_html = (
            "<h2>Practice: fully graded holds</h2>"
            "<p>Every signal you held long enough to grade during Practice, scored the same way as Drill (recognition, "
            "distinctness, hold, FIVB form, ready position), at all three difficulty levels. Not shown as a live score "
            "during Practice itself, but recorded here for documentation.</p>"
            "<h3>By signal (average score per level)</h3>"
            "<table><tr><th>Signal</th><th>Times</th><th>Avg Beginner</th><th>Avg Standard</th><th>Avg Referee</th></tr>"
            f"{ssr}</table>"
            "<p style=\"color:#666;font-size:12px\">Also saved as practice_scores.csv.</p>"
            "<details><summary>Every graded hold</summary>"
            "<table><tr><th>Time</th><th>Seconds from start</th><th>Signal</th><th>Verdict (session level)</th>"
            f"<th>Score</th><th>Hold (s)</th><th>All levels</th></tr>{psr}</table></details>")
    practice_html = ""
    if practice_log:
        summary_rows = practice_summary_rows(practice_log)
        sr = "".join(f"<tr><td>{e(str(row[0]))}</td><td>{row[1]}</td><td>{row[2]}</td><td>{row[3]}</td></tr>"
                    for row in summary_rows) or "<tr><td colspan='4'>(no named signal was detected)</td></tr>"
        pr = "".join(
            f"<tr><td>{e(str(r['ph_time']))}</td><td>{e(str(r['seconds_from_start']))}</td>"
            f"<td>{e(str(r['detected_name']))}</td><td>{e(str(r['best_prob']))}</td>"
            f"<td>{r['checks_passed']}/{r['checks_total']}</td></tr>" for r in practice_log)
        n_events = len(practice_log)
        n_signals = sum(1 for r in practice_log if r["detected_label"] != gg.NOTHING_LABEL)
        practice_html = (f"<h2>Practice log</h2><p>{n_events} detection event(s) logged during free practice "
                         f"({n_signals} of them a named signal, the rest 'no signal'/arms down). No score is kept in "
                         f"Practice mode; this is a record of what the system detected and when, for documentation.</p>"
                         "<h3>Signals detected (summary table, for the paper)</h3>"
                         "<table><tr><th>Signal</th><th>Times detected</th><th>Avg. checks passed</th>"
                         f"<th>Avg. checks total</th></tr>{sr}</table>"
                         "<p style=\"color:#666;font-size:12px\">Also saved as practice_summary.csv (this table) and "
                         "practice_log.csv (every detection event).</p>"
                         "<details><summary>Full event-by-event log</summary>"
                         "<table><tr><th>Time (PH)</th><th>Seconds from start</th><th>Detected</th>"
                         f"<th>Model confidence</th><th>Checks passed</th></tr>{pr}</table></details>")
    p = summary.get("performance") or {}
    perf_html = ""
    if p:
        perf_html = (f"<h2>Performance of this session</h2><p>{p.get('frames', 0)} frames in {p.get('seconds', 0)} s. "
                     f"Speed {p.get('fps_mean', 0)} frames per second (first quarter {p.get('fps_first_quarter', 0)}, "
                     f"last quarter {p.get('fps_last_quarter', 0)}). Per frame: feature extraction {p.get('extract_ms_mean', 0)} ms, "
                     f"model {p.get('classify_ms_mean', 0)} ms per inference. Verdict computed in {p.get('grade_ms_mean', 0)} ms "
                     f"(max {p.get('grade_ms_max', 0)} ms). "
                     + (f"CPU of this program {p.get('cpu_process_mean_pct', 0)}% (max {p.get('cpu_process_max_pct', 0)}%) of one core, "
                        f"whole computer {p.get('cpu_system_mean_pct', 0)}%, memory {p.get('ram_mean_mb', 0)} MB (max {p.get('ram_max_mb', 0)} MB)."
                        if p.get("psutil") else "CPU and memory were not measured (install psutil).") + "</p>")
    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Training report</title>
<style>body{{font-family:Arial,sans-serif;margin:30px;background:#fafafa;color:#222}}
table{{border-collapse:collapse;width:100%;margin-top:10px}}th,td{{border:1px solid #ccc;padding:6px 10px;font-size:14px;text-align:left}}
th{{background:#eee}}.box{{background:#fff;border:1px solid #ddd;border-radius:6px;padding:14px 20px;margin-bottom:18px}}</style></head><body>
<h1>Officiating Training Report</h1>
<div class="box"><p><b>Trainee:</b> {e(trainee['display_name'])} ({e(trainee['trainee_id'])})</p>
<p><b>Mode:</b> {e(summary['mode'])} &nbsp; <b>Level:</b> {e(summary['level'])} &nbsp; <b>Generated:</b> {e(now_ph_str())} (PH time)</p>
<p><b>Trainee score:</b> {summary['points']} / {summary['max_points']} &nbsp; <b>Accuracy:</b> {summary['accuracy_pct']}%
&nbsp; <b>Correct:</b> {summary['correct']} of {summary['attempts']} &nbsp; <b>Average signal score:</b> {summary['avg_gesture_score']}/100</p>{team}
<p style="color:#666;font-size:12px">Consent version {e(trainer_ui.CONSENT_VERSION)}. Grading is based on the FIVB referee hand signals as measured by a single 2D camera
(see gesture_grader.py). CORRECT = 10 pts, ALMOST = 5 pts.</p>
<p style="color:#666;font-size:12px"><b>{e(gg.DISCLAIMER)}</b></p></div>
{perf_html}
{whistle_html}
{practice_scores_html}
{practice_html}
<h2>By signal</h2><table><tr><th>Signal</th><th>Correct</th><th>Avg score</th></tr>{per}</table>
<p><b>Needs more practice</b> (fewer than half CORRECT): {e(", ".join(gg.short_label(k) for k in summary.get("needs_practice", [])) or "none")}</p>
<h2>Attempts</h2><p style="color:#666;font-size:12px">"All levels" shows how the SAME captured movement would be judged at every difficulty
(B=Beginner, S=Standard, R=Referee; C/A/I = Correct/Almost/Incorrect), not just the level this session used.</p>
<table><tr><th>Time</th><th>Target</th><th>Detected</th><th>Verdict</th><th>Score</th><th>Points</th><th>Hold (s)</th><th>All levels</th><th>Failed checks</th><th>Feedback</th><th>Verdict shown (s after arms down)</th></tr>{rows}</table>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)


# ============================================================================
# Entry point
# ============================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description="Volleyball Officiating Training Tool")
    ap.add_argument("--camera", type=int, default=None,
                    help="camera index (default: the one chosen in the setup screen, else trainer_config.py)")
    ap.add_argument("--list-cameras", action="store_true", help="show which camera indexes work, then exit")
    args = ap.parse_args(argv)
    if args.list_cameras:
        print("Checking cameras (close other programs that use the camera first)...")
        found = devices.list_cameras()
        for c in found:
            print(f"  camera {c['index']}: works ({c['width']}x{c['height']})")
        if not found:
            print("  no working camera found")
        return

    trainee = trainer_ui.run_welcome()
    if not trainee:
        print("Consent not given. Exiting.")
        return

    # first run: let the trainee pick the camera and test the microphone
    if args.camera is None and not trainer_ui.read_settings().get("setup_done"):
        import device_setup
        device_setup.run_device_setup()

    while True:
        try:
            backend = build_backend(args.camera)
            break
        except Exception as exc:
            print(f"Could not start the recognition backend: {exc}")
            if not trainer_ui.ask_yes_no(
                    "The tool could not start",
                    "The camera or the recognition model could not be started, so nothing was recorded.\n\n"
                    f"{exc}\n\nCheck that the camera is plugged in (or Camo is running) and that no other program is "
                    "using it. Open the camera and microphone check to pick a camera and try again?"):
                return
            import device_setup
            device_setup.run_device_setup()
    whistle = WhistleHub()
    whistle.device, mic_warning = devices.resolve_saved_mic(trainer_ui.read_settings())
    if mic_warning:
        trainer_ui.show_message("Microphone not found", mic_warning, error=True)
    last_summary = ""
    try:
        while True:
            choice = trainer_ui.run_menu(trainee, backend.real_labels, last_summary)
            if choice["action"] == "devices":
                import device_setup
                backend.release_camera()
                device_setup.run_device_setup()
                settings = trainer_ui.read_settings()
                whistle.stop()
                whistle.device, mic_warning = devices.resolve_saved_mic(settings)
                if mic_warning:
                    trainer_ui.show_message("Microphone not found", mic_warning, error=True)
                try:
                    backend.reopen_camera(pick_camera_index(args.camera, settings.get("camera_index") or 0))
                except Exception as exc:
                    print(f"Could not reopen the camera: {exc}")
                    trainer_ui.show_message("Camera not available",
                                           f"The camera could not be opened: {exc}\n\nChoose another camera under "
                                           "Camera and mic, or check the connection.", error=True)
                continue
            if choice["action"] in ("quit", "deleted"):
                break
            set_cv_theme(trainer_ui.current_theme())
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            fit_cv_window(WINDOW_NAME, 1280, 720)
            while True:
                summary = Session(backend, trainee, choice, whistle).run()
                last_summary = summary["one_line"]
                if not summary.get("restart") or summary.get("error") or summary.get("camera_lost"):
                    break                         # R on the summary screen ("do it again") runs the same session again
            cv2.destroyAllWindows()
            for _ in range(3):
                cv2.waitKey(1)
            if summary.get("error"):
                trainer_ui.show_message("Something went wrong",
                                        "The session stopped because of an error. Everything recorded before it was "
                                        "saved (see error.log in the session folder).\n\n" + summary["error"], error=True)
            elif summary.get("camera_lost"):
                trainer_ui.show_message("Camera stopped",
                                        "The camera stopped working during the session and could not be reconnected. "
                                        "Your results so far were saved. Check the camera, then start again.")
    finally:
        whistle.stop()
        backend.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()