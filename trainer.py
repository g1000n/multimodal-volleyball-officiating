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
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

import devices
import gesture_grader as gg
import trainer_ui
from devices import open_camera   # noqa: F401  (re-exported: tests and other tools use trainer.open_camera)

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
TIMINGS = {"drill": (2.5, 3.0, 3.0), "combo": (2.5, 3.5, 3.0), "challenge": (1.8, 2.2, 2.0)}
AUTO_INTRO_SECONDS = 2.5
AUTO_RESULT_SECONDS = 3.0
WINDOW_NAME = "Volleyball Officiating Trainer"

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

    def release_camera(self):
        try:
            self.cap.release()
        except Exception:
            pass

    def reopen_camera(self, preferred):
        """Reopen after the setup screen. Falls back to another camera if `preferred` is unavailable."""
        self.cap, used = devices.open_camera(preferred)
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
        self.detector = None
        self.device = None          # microphone index chosen in the setup screen (None = use trainer_config)

    def _on_whistle(self, timestamp, confidence=None):
        with self._lock:
            self._times.append(time.time())

    def manual(self):
        self._on_whistle(time.time())

    def start(self):
        if self.detector is not None:
            return
        try:
            from whistle_detector import WhistleDetector
            device = self.device if self.device is not None else WHISTLE_DEVICE_INDEX
            self.detector = WhistleDetector(on_whistle_callback=self._on_whistle, device=device)
            self.detector.start()
            print("Whistle detection ACTIVE (mic). Press W as a backup.")
        except Exception as exc:  # missing model, no mic, sounddevice error ...
            self.detector = None
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
            f"The team on your {W} wins the rally and serves next.")


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
    """A short narrated set: the trainee authorises the serve, then makes the call at the end of each rally,
    then signals the end of the set. Everything comes from the same 8 signals and one set (project scope)."""
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
                      f"Blow the whistle, then authorise the serve.")
        if sa in have:
            queue.append({"kind": "whistle", "text": serve_text, "scenario": r, "n_scenarios": total})
            queue.append({"kind": "gesture", "label": sa, "text": serve_text, "scenario": r, "n_scenarios": total})
        choices = [x for x in reasons if x != last] or reasons
        reason = rng.choice(choices)
        last = reason
        winner, fault_side, story = make_rally_end(rng, reason, server)
        if f"team_to_serve_{winner}" not in have:
            continue
        text = f"Rally {r} of {total} is over. " + story
        queue.append({"kind": "whistle", "text": text + " Blow the whistle, then show the team to serve and the reason.",
                      "scenario": r, "n_scenarios": total})
        queue.append(make_pair_step(winner, reason, fault_side, text, scenario=r, n_scenarios=total))
        server = winner
    if "end_of_set" in have:
        end_text = "The set is over (practice set). Blow the whistle and signal the end of the set."
        queue.append({"kind": "whistle", "text": end_text, "scenario": total, "n_scenarios": total})
        queue.append({"kind": "gesture", "label": "end_of_set", "text": end_text, "scenario": total,
                      "n_scenarios": total})
    return queue


def add_whistle_steps(queue, choice):
    """Optional: Team to Serve and Service Authorization must be preceded by the whistle (FIVB 22.2)."""
    if not choice.get("whistle"):
        return queue
    out = []
    for st in queue:
        first = st.get("label") or (st.get("labels") or [""])[0]
        if st["kind"] in ("gesture", "pair") and needs_whistle(first):
            out.append({"kind": "whistle", "text": WHISTLE_NOTE})
            if st.get("repeat"):
                st = dict(st, loop_to=len(out) - 1)
        out.append(st)
    return out


def build_queue(mode, choice, real_labels, rng):
    if mode == "sim":
        return build_sim_queue(rng, real_labels)
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
        self.uses_whistle = self.mode == "sim" or any(q["kind"] == "whistle" for q in self.queue)
        self.step_i = 0
        self.phase = "practice" if self.mode == "practice" else "intro"
        self.phase_t0 = 0.0
        self.cap_frames, self.cap_records = [], []
        self.result = None
        self.last_rec = None
        self.stable_label = gg.NOTHING_LABEL
        self._prev_label = gg.NOTHING_LABEL
        self.practice_checks = []
        self.hint = False
        self.attempts = []
        self.attempt_results = []       # parallel to attempts: AttemptResult (gestures) or None (whistle)
        self.review_i = None
        self.reps = int(choice.get("reps", 1) or 1)
        self.auto = self.mode in ("combo", "challenge") or (self.mode == "drill" and self.reps > 1)
        self.t_intro, self.t_result, self.t_countdown = TIMINGS.get(self.mode, (2.5, 3.0, COUNTDOWN_SECONDS))
        self.cd_seconds = self.t_countdown
        self.cap_seconds = CAPTURE_SECONDS
        self._ref_cache = {}
        self.results = []               # AttemptResult per signal of the current step (1, or 2 for a pair)
        self.attempt_no = 0
        self.auto_go = False            # becomes True after the trainee presses SPACE once
        self.points = 0
        self.max_points = 0
        self.team = {"left": 0, "right": 0}
        self.whistle_flash_until = 0.0
        self.whistle_ok = None
        self.banner = ""
        self.summary = None

    # ---- hooks overridable for tests ----
    def _show(self, ui):
        show_letterboxed(WINDOW_NAME, ui)

    def _key(self):
        return cv2.waitKey(1) & 0xFF

    # ---- helpers ----
    @property
    def step(self):
        return self.queue[self.step_i] if self.step_i < len(self.queue) else None

    def _flush_camera(self):
        for _ in range(4):
            self.be.cap.read()

    def _set_phase(self, phase):
        self.phase = phase
        self.phase_t0 = self.clock()

    def _begin_step(self):
        """Start the current step: the whistle wait, or the countdown before a signal."""
        step = self.step
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
    def _log_attempt(self, step, res=None, whistle_ok=None):
        kind = step["kind"]
        if kind == "whistle":
            verdict = gg.VERDICT_CORRECT if whistle_ok else gg.VERDICT_INCORRECT
            pts = 10 if whistle_ok else 0
            row = {"ph_time": now_ph_str(), "kind": "whistle", "target": "whistle", "level": self.level,
                   "verdict": verdict, "score": 100 if whistle_ok else 0, "points": pts, "best_prob": "",
                   "margin": "", "hold_s": "", "confused_with": "", "failed_checks": "", "check_values": "",
                   "intent": self.intent, "note": self.note,
                   "feedback": "" if whistle_ok else "No whistle detected in time."}
        else:
            failed = [c.id for c in res.checks if c.status == "fail"]
            row = {"ph_time": now_ph_str(), "kind": "gesture", "target": step["label"], "level": self.level,
                   "verdict": res.verdict, "score": res.score, "points": res.points,
                   "best_prob": f"{res.best_prob:.3f}", "margin": f"{res.margin:.3f}",
                   "hold_s": f"{res.hold_seconds:.2f}", "confused_with": res.confused_with or "",
                   "failed_checks": ";".join(failed), "intent": self.intent, "note": self.note,
                   "check_values": ";".join(
                       f"{c.id}={'' if c.value is None else round(c.value, 2)}[{c.need}]:{c.status}" for c in res.checks),
                   "feedback": " | ".join(res.feedback)}
            pts = res.points
        if row["verdict"] != gg.VERDICT_NO_READING:
            self.points += pts
            self.max_points += 10
            self.attempts.append(row)
            self.attempt_results.append(res if kind == "gesture" else None)
        return row

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
        acc = 100.0 * correct / n if n else 0.0
        one = (f"{self.mode} ({self.cfg.name}): {correct}/{n} correct, {self.points}/{self.max_points} points"
               if n else f"{self.mode}: no graded attempts")
        return {"mode": self.mode, "level": self.level, "attempts": n, "correct": correct, "almost": almost,
                "incorrect": n - correct - almost, "points": self.points, "max_points": self.max_points,
                "accuracy_pct": round(acc, 1), "avg_gesture_score": round(avg, 1), "per_signal": per,
                "team_points": dict(self.team) if self.mode == "sim" else None, "one_line": one}

    # ---- main loop ----
    def run(self):
        os.makedirs(self.dir, exist_ok=True)
        writer = cv2.VideoWriter(os.path.join(self.dir, "session.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 RECORD_FPS, (UI_W, UI_H))
        self._flush_camera()
        if self.uses_whistle:
            self.whistle.start()
        if self.queue and self.mode != "practice":
            self._set_phase("intro")

        try:
            while True:
                ok, frame = self.be.cap.read()
                if not ok:
                    self.banner = "Camera stopped."
                    break
                if INPUT_ALREADY_MIRRORED:
                    frame = cv2.flip(frame, 1)
                self.aspect = frame.shape[0] / max(1, frame.shape[1])
                t = self.clock()
                self.fps_times.append(t)

                feats = np.zeros(122)
                rec = None
                if not self.paused:
                    feats = self.be.extract(frame)
                    self.rolling.append(feats)
                    self.frame_counter += 1
                    if (self.phase in ("practice", "capture") and len(self.rolling) == ROLLING_WINDOW_FRAMES
                            and self.frame_counter % INFERENCE_EVERY_N_FRAMES == 0):
                        label, _conf, probs = self.be.classify(list(self.rolling))
                        rec = {"label": label, "probs": np.array(probs, dtype=float),
                               "frames": np.array(self.rolling), "t": t}
                        self.last_rec = rec
                    self._update(t, feats, rec)
                else:
                    feats = np.array(self.rolling[-1]) if self.rolling else feats

                ui = self._render(frame, feats, t)
                writer.write(ui)
                self._show(ui)
                key = self._key()
                if self._handle_key(key):
                    break
                if self.phase == "done":
                    break
        finally:
            writer.release()
            if self.uses_whistle:
                self.whistle.stop()

        if self.summary is None:
            self.summary = self._make_summary()
        self._save_outputs()
        return self.summary

    # ---- state machine ----
    def _update(self, t, feats, rec):
        step = self.step
        if self.phase == "practice":
            if rec is not None:
                lab = rec["label"]
                if lab == gg.NOTHING_LABEL:
                    self.stable_label = gg.NOTHING_LABEL
                elif lab == self._prev_label:          # same signal on two consecutive windows = stable
                    self.stable_label = lab
                self._prev_label = lab
                self.practice_checks = (gg.grade_form_only(self.stable_label, rec["frames"], self.level, self.aspect)
                                        if self.stable_label in gg.RULES else [])
        elif self.phase == "intro":
            if self.auto and self.auto_go and step is not None and t - self.phase_t0 >= self.t_intro:
                self._begin_step()
        elif self.phase == "result":
            if self.auto and t - self.phase_t0 >= self.t_result:
                if any(r.verdict == gg.VERDICT_NO_READING for r in self.results):
                    self._set_phase("intro")      # not counted: automatically try the same step again
                else:
                    self._advance()
        elif self.phase == "countdown":
            if t - self.phase_t0 >= self.cd_seconds:
                self.cap_frames, self.cap_records = [], []
                self.cap_seconds = CAPTURE_SECONDS_PAIR if step is not None and step["kind"] == "pair" else CAPTURE_SECONDS
                self._set_phase("capture")
        elif self.phase == "capture":
            self.cap_frames.append(feats)
            if rec is not None:
                rec["end"] = len(self.cap_frames)      # frame index in the capture where this window ends
                self.cap_records.append(rec)
            if t - self.phase_t0 >= self.cap_seconds:
                self._grade_current()
        elif self.phase == "wait_whistle":
            if self.whistle.heard_since(self.phase_t0):
                self._whistle_done(True)
            elif t - self.phase_t0 >= WHISTLE_WAIT_SECONDS:
                self._whistle_done(False)
        elif self.phase == "whistle_result":
            if t - self.phase_t0 >= 1.6:
                self._advance()

    def _grade_current(self):
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
        else:
            labels = [step["label"]]
            ctxs = [step.get("ctx")]
            results = [gg.grade_attempt(step["label"], self.cap_frames, recs, self.be.label_to_idx,
                                        level=self.level, aspect=self.aspect, step_seconds=step_s, context=step.get("ctx"))]
        self.results, self.result = results, results[0]
        self._save_attempt_npz(labels, ctxs, results, step_s)
        if not any(r.verdict == gg.VERDICT_NO_READING for r in results):
            for label, res in zip(labels, results):
                self._log_attempt({"kind": "gesture", "label": label}, res=res)
                if (self.mode == "sim" and res.verdict == gg.VERDICT_CORRECT and label.startswith("team_to_serve_")):
                    self.team[label.rsplit("_", 1)[1]] += 1
        self._set_phase("result")

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
        elif ch == "p" and self.phase in ("practice", "intro", "result"):
            self.paused = not self.paused
        elif ch == "t":
            new = "light" if _cv_theme_name == "dark" else "dark"
            trainer_ui.set_theme(new)
            set_cv_theme(new)
        elif ch == "w":
            self.whistle.manual()
            self.whistle_flash_until = self.clock() + 1.2
        elif ch == "h":
            self.hint = not self.hint
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
            self._set_phase("intro")
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
            if self.phase in ("intro", "countdown"):
                self._draw_reference(ui, self.step)
        if self.paused:
            panel(ui, 250, 300, 650, 360, OV_AMBER, 0.9)
            put(ui, "PAUSED (press P)", 300, 340, 0.9, OV_ON, 2)
        return ui

    def _draw_top(self, ui):
        panel(ui, 0, 0, UI_W, TOP_H, C_PANEL)
        title = {"practice": "PRACTICE", "drill": "DRILL", "combo": "COMBO DRILL", "challenge": "CHALLENGE", "sim": "MATCH SIMULATION"}[self.mode]
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
        if self.auto and self.queue and self.phase != "summary":
            prog = f"Attempt {min(self.step_i + 1, len(self.queue))}/{len(self.queue)}"
            put(ui, prog, x - tw(prog, 0.6, 1) - 40, 36, 0.6, C_AMBER, 1)
        if self.mode == "sim":
            team = f"Team on your LEFT {self.team['left']}  -  {self.team['right']} RIGHT"
            put(ui, team, x - tw(team, 0.6, 1) - 40, 36, 0.6, C_AMBER, 1)

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

    def _draw_bottom(self, ui, t):
        bx, by, bw, bh = BOTTOM_BOX
        panel(ui, bx, by, bx + bw, by + bh, C_PANEL2)
        step = self.step
        y = by + 32
        if self.phase == "practice":
            put(ui, "Do any signal. The system will tell you which one it sees.", 20, y, 0.65, C_TEXT, 1)
            lab = self.stable_label
            if lab in gg.SIGNALS:
                for i, line in enumerate(wrap("FIVB: " + gg.SIGNALS[lab]["fivb"], bw - 40, 0.55)):
                    put(ui, line, 20, y + 34 + i * 24, 0.55, C_AMBER, 1)
        elif self.phase == "intro" and step is not None:
            self._draw_intro_bottom(ui, by, bw, y, step)
        elif self.phase == "countdown":
            put(ui, "Get into the ready position: arms relaxed, facing the camera.", 20, y, 0.65, C_TEXT, 1)
            two = step is not None and step["kind"] == "pair"
            put(ui, ("Show the first signal at GO, hold about 2 s, then go straight into the second."
                     if two else "Perform the signal when you see GO, hold it, then lower your arms."),
                20, y + 28, 0.58, C_MUTED, 1)
            if step is not None and step.get("text"):
                for i, line in enumerate(wrap(step["text"], bw - 40, 0.52)[:2]):
                    put(ui, line, 20, y + 58 + i * 21, 0.52, C_TEXT, 1)
        elif self.phase == "capture":
            two = step is not None and step["kind"] == "pair"
            put(ui, ("Team to serve, hold 2 s, then go straight into the reason."
                     if two else "Perform the signal now. Hold it steady, then lower your arms."),
                20, y, 0.62, C_GREEN, 2)
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
            notes = [r.note for r in self.results if r.note]
            msg = notes[0] if notes else "Check the panel on the right for details."
            if step is not None and step["kind"] == "pair":
                msg += "   Correct call: " + self._step_names(step) + "."
            elif self.mode == "sim" and step is not None and step["kind"] == "gesture":
                msg += "   Correct call: " + self._step_names(step) + "."
            if self.auto:
                msg += "  The next step starts automatically; every score is listed at the end."
            for i, line in enumerate(wrap(msg, bw - 40, 0.55)[:3]):
                put(ui, line, 20, y + i * 24, 0.55, C_TEXT, 1)
        keys = {"practice": "Q end practice   P pause   M mirror   T theme",
                "intro": "SPACE start   Q end session   M mirror   T theme" + ("   H hint" if self.mode == "sim" else ""),
                "result": "SPACE next" + ("   R retry" if self.mode == "drill" and not self.auto else "") + "   Q end session",
                "countdown": "Q end session", "capture": "Q end session",
                "wait_whistle": "W = manual whistle   Q end session", "whistle_result": ""}.get(self.phase, "")
        put(ui, keys, 20, by + bh - 14, 0.5, C_MUTED, 1)

    def _draw_intro_bottom(self, ui, by, bw, y, step):
        lines = wrap(step["text"], bw - 40, 0.56)[:3] if step.get("text") else []
        for i, line in enumerate(lines):
            put(ui, line, 20, y + i * 23, 0.56, C_TEXT, 1)
        cur = y + 23 * len(lines) + (4 if lines else 0)
        if step["kind"] == "whistle":
            msg = "Step: blow the whistle." if not lines else "Then: blow the whistle."
            prompt = None
        elif step["kind"] == "pair":
            msg = (f"Signals: {self._step_names(step)}" if self._reveal()
                   else "Give the team to serve, then the reason. Press H for a hint.")
        elif self.mode == "sim":
            msg = (f"Signal needed: {self._step_names(step)}" if self._reveal()
                   else "Give the correct signal. Press H for a hint.")
        elif self.mode == "challenge":
            msg = f"Signal {self.step_i + 1} of {len(self.queue)}: {self._step_names(step)}. One attempt each."
        elif self.auto:
            msg = f"Repetition {step.get('rep', 1)} of {step.get('n_reps', 1)}"
        else:
            msg = "Read the card on the right, then press SPACE when you are ready."
        put(ui, msg, 20, cur + 18, 0.6, C_AMBER, 1)
        if self.auto:
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
            elif step["kind"] == "pair":
                self._draw_side_pair_card(ui, x0 + pad, y, w, step)
            else:
                self._draw_side_card(ui, x0 + pad, y, w, step)
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
            self._draw_disclaimer(ui, x0 + pad, w)

    def _draw_side_card(self, ui, x, y, w, step):
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
            if not compact and s.get("not_graded"):
                y += 24
                put(ui, "Not graded by the camera", x, y, 0.43, C_AMBER, 1)
                for line in wrap(s["not_graded"], w, 0.42):
                    y += 16
                    put(ui, line, x, y, 0.42, C_MUTED, 1)
        if self.phase == "capture":
            self._draw_capture_meter(ui, x, UI_H - 150, w, label)

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

    def _draw_pair_meter(self, ui, x, y, w, step):
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
            put(ui, f"{r.score}/100  +{r.points} pts", x + 175, y + 32, 0.5, C_TEXT, 1)
            yy = y + 52
            if r.verdict != gg.VERDICT_NO_READING:
                for c in r.checks[:6]:
                    draw_icon(ui, c.status, x + 8, yy + 6, 6)
                    put(ui, fit(c.label, w - 26, 0.42), x + 22, yy + 10, 0.42, C_TEXT if c.status == "pass" else C_MUTED, 1)
                    yy += 18
            for line_src in r.feedback[:2]:
                for j, line in enumerate(wrap(line_src, w, 0.44)[:2]):
                    put(ui, ("- " if j == 0 else "  ") + line, x, yy + 12, 0.44, C_AMBER, 1)
                    yy += 17
            y = max(yy + 26, y + 150)
            cv2.line(ui, (x, y - 16), (x + w, y - 16), C_LINE, 1)
        for i, line in enumerate(wrap("Correct call: " + self._step_names(step), w, 0.42)[:2]):
            put(ui, line, x, min(y + 6 + i * 16, UI_H - 62), 0.42, C_MUTED, 1)

    def _draw_side_result(self, ui, x, y, w, r=None):
        r = r or self.result
        color = VERDICT_COLOR[r.verdict]
        put(ui, VERDICT_TEXT[r.verdict], x, y + 6, 1.15, color, 3)
        if r.verdict == gg.VERDICT_NO_READING:
            for i, line in enumerate(wrap(r.feedback[0], w, 0.55)):
                put(ui, line, x, y + 50 + i * 22, 0.55, C_TEXT, 1)
            return
        put(ui, f"Score {r.score}/100   +{r.points} pts", x, y + 40, 0.65, C_TEXT, 2)
        y += 64
        rows = [("Recognition", r.parts["recognition"], gg.W_RECOGNITION), ("Distinctness", r.parts["distinctness"], gg.W_DISTINCT),
                ("Hold", r.parts["hold"], gg.W_HOLD), ("FIVB form", r.parts["form"], gg.W_FORM),
                ("Ready pos.", r.parts["ready_position"], gg.W_READY)]
        for name, val, mx in rows:
            put(ui, f"{name}", x, y + 12, 0.45, C_MUTED, 1)
            bar(ui, x + 110, y + 2, w - 170, 10, val / mx, color)
            put(ui, f"{val:.0f}/{mx}", x + w - 52, y + 12, 0.45, C_TEXT, 1)
            y += 22
        y += 8
        put(ui, "Checklist", x, y + 8, 0.5, C_BLUE, 1)
        y += 18
        for c in r.checks:
            draw_icon(ui, c.status, x + 8, y + 8, 7)
            put(ui, fit(c.label, w - 30, 0.45), x + 24, y + 12, 0.45, C_TEXT if c.status == "pass" else C_MUTED, 1)
            y += 21
        y += 10
        put(ui, "How to improve", x, y + 8, 0.5, C_AMBER, 1)
        y += 20
        for line_src in r.feedback[:4]:
            for j, line in enumerate(wrap(line_src, w, 0.46)):
                if y > UI_H - 62:
                    return
                put(ui, ("- " if j == 0 else "  ") + line, x, y + 12, 0.46, C_TEXT, 1)
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
        put(ui, "A / D  review each attempt     SPACE  back to the menu", 40, UI_H - 26, 0.55, C_AMBER, 1)
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
    def _save_outputs(self):
        cols = ["ph_time", "kind", "target", "level", "intent", "note", "verdict", "score", "points", "best_prob", "margin",
                "hold_s", "confused_with", "failed_checks", "check_values", "feedback"]
        with open(os.path.join(self.dir, "attempts.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(self.attempts)
        with open(os.path.join(self.dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({**self.summary, "trainee_id": self.trainee["trainee_id"], "intent": self.intent, "note": self.note,
                       "consent_version": trainer_ui.CONSENT_VERSION}, f, indent=2)
        write_report(os.path.join(self.dir, "report.html"), self.trainee, self.summary, self.attempts)


def write_report(path, trainee, summary, attempts):
    e = html.escape
    rows = "".join(
        f"<tr><td>{e(a['ph_time'])}</td><td>{e(a['target'])}</td>"
        f"<td style='font-weight:bold;color:{ {'CORRECT': '#2e7d32', 'ALMOST': '#e65100', 'INCORRECT': '#c62828'}.get(a['verdict'], '#333') }'>{e(a['verdict'])}</td>"
        f"<td>{a['score']}</td><td>{a['points']}</td><td>{e(str(a['hold_s']))}</td><td>{e(a['failed_checks'])}</td>"
        f"<td>{e(a['feedback'])}</td></tr>" for a in attempts) or "<tr><td colspan='8'>(no graded attempts)</td></tr>"
    per = "".join(f"<tr><td>{e(k)}</td><td>{v['correct']}/{v['n']}</td><td>{v['score_sum'] / max(1, v['n']):.0f}</td></tr>"
                  for k, v in summary["per_signal"].items())
    team = ""
    if summary.get("team_points"):
        t = summary["team_points"]
        team = f"<p><b>Scoreboard (from the trainee's correct calls):</b> team on your LEFT {t['left']} - {t['right']} RIGHT</p>"
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
<h2>By signal</h2><table><tr><th>Signal</th><th>Correct</th><th>Avg score</th></tr>{per}</table>
<h2>Attempts</h2><table><tr><th>Time</th><th>Target</th><th>Verdict</th><th>Score</th><th>Points</th><th>Hold (s)</th><th>Failed checks</th><th>Feedback</th></tr>{rows}</table>
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

    try:
        backend = build_backend(args.camera)
    except Exception as exc:
        print(f"Could not start the recognition backend: {exc}")
        return
    whistle = WhistleHub()
    whistle.device = trainer_ui.read_settings().get("mic_index")
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
                whistle.device = settings.get("mic_index")
                try:
                    backend.reopen_camera(pick_camera_index(args.camera, settings.get("camera_index") or 0))
                except Exception as exc:
                    print(f"Could not reopen the camera: {exc}")
                    return
                continue
            if choice["action"] in ("quit", "deleted"):
                break
            set_cv_theme(trainer_ui.current_theme())
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 1280, 720)
            summary = Session(backend, trainee, choice, whistle).run()
            last_summary = summary["one_line"]
            cv2.destroyAllWindows()
            for _ in range(3):
                cv2.waitKey(1)
    finally:
        whistle.stop()
        backend.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()