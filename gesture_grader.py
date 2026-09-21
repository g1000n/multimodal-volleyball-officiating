"""
gesture_grader.py

FIVB-based grading layer for the Multimodal Officiating TRAINING Tool.

WHY THIS FILE EXISTS
--------------------
The recognition model (CNN-LSTM) answers "WHAT gesture is this?" and was
deliberately built to be lenient, because for live match officiating a missed
call is worse than a generous one. A trainee tool needs the opposite question:
"Did you perform THIS gesture CORRECTLY?" This module answers that WITHOUT
touching the model, the dataset, or any weights. It sits on top of the model's
existing outputs (per-class sigmoid probabilities) and the raw MediaPipe
keypoint features that extract_keypoints.py already produces.

LEFT / RIGHT CONVENTION (trainee UI)
------------------------------------
"left" always means THE TRAINEE'S OWN LEFT ARM. The class names in the dataset
(team_to_serve_left, ...) are tied to the referee's own left/right, and
MediaPipe's LEFT_* landmarks are also the subject's own left, so the class
suffix maps 1:1 onto "your left arm" with NO flipping. (decision_engine.py's
GESTURE_TO_SCORE_SIDE flip is an AUDIENCE-perspective convention for the
match scoreboard; the trainer does not use it.)

HOW AN ATTEMPT IS SCORED (100 points)
-------------------------------------
    Recognition   35   best sigmoid probability of the TARGET class vs the
                       difficulty level's threshold
    Distinctness  10   lead of the target over the strongest other class
    Hold          10   how long the system kept recognising the signal
    Form          40   FIVB-derived geometric checks (per gesture, below)
    Ready pos.     5   arms returned to the ready position afterwards
Verdict: CORRECT / ALMOST / INCORRECT (+ NO_READING when the body was not seen).
A failed CRITICAL form check caps the verdict at ALMOST.

DIFFICULTY LEVELS = YOUR LENIENCY, MADE ADJUSTABLE
--------------------------------------------------
Beginner / Standard / Referee change only thresholds (probability cutoff,
hold time, joint-angle tolerances). Nothing is retrained.

WHERE THE RULES COME FROM (be honest about this in the paper)
-------------------------------------------------------------
Each form check is tagged with a `basis`:
  "FIVB"     - derived from the wording of the FIVB Official Volleyball Rules,
               "Referee Hand Signals" (paraphrased in SIGNALS[...]["fivb"]).
               Verify the wording against the edition your referee validator
               uses; the SIGNALS text is easy to edit.
  "TRAINING" - a pedagogical requirement (hold time, returning to the ready
               position). NOT stated by FIVB as a numeric rule.
The numeric tolerances (degrees etc.) are ENGINEERING choices that translate the
FIVB wording into what a single 2D camera can measure. They are collected in
one place (the `pick(level, beginner, standard, referee)` calls in the _rules_*
functions) and can be checked against your own dataset with
tools/grader_sanity_check.py.

LIMITS OF A SINGLE 2D CAMERA (also worth stating in the paper)
--------------------------------------------------------------
Depth and palm orientation cannot be measured reliably, so e.g. "palms toward
the body" (Ball Out) is NOT graded; see SIGNALS[...]["not_graded"].
Ball In is the weakest class of the recognition model (recall 0.57), so its
recognition score may partly reflect the model, not only the trainee.

COORDINATES
-----------
Raw feature layout per frame (extract_keypoints.py): 8 pose landmarks x
(x, y, visibility) in cols 0-23 [LS, RS, LE, RE, LW, RW, LH, RH], hand
coordinates 24-107, hand-detected flags 108/109, finger-extension flags
110-114 (left) / 115-119 (right), elbow-angle features 120/121.
All geometry here is ASPECT-CORRECTED (y is scaled by frame_height/frame_width)
so angles are true image-plane angles regardless of camera resolution.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np

try:                                   # your own adjustments live in trainer_config.py (see GRADER_OVERRIDES there)
    import trainer_config as _tc
except ImportError:
    _tc = None

NOTHING_LABEL = "nothing"

VERDICT_CORRECT = "CORRECT"
VERDICT_ALMOST = "ALMOST"
VERDICT_INCORRECT = "INCORRECT"
VERDICT_NO_READING = "NO_READING"
POINTS = {VERDICT_CORRECT: 10, VERDICT_ALMOST: 5, VERDICT_INCORRECT: 0, VERDICT_NO_READING: 0}

# Score weights (sum = 100)
W_RECOGNITION = 38
W_DISTINCT = 10
W_HOLD = 10
W_FORM = 40
W_READY = 2

MIN_POSE_FRACTION = 0.6      # below this, the body was not seen well enough to grade
MIN_ARM_VISIBLE_FRAC = 0.5   # an arm checked by a rule must be visible in >= this fraction of frames
MIN_HAND_COVERAGE = 0.3      # hand-detection coverage needed to trust finger checks

# ----------------------------------------------------------------------------
# Difficulty levels
# ----------------------------------------------------------------------------

LEVELS = ("beginner", "standard", "referee")


@dataclass(frozen=True)
class LevelConfig:
    key: str
    name: str
    rec_threshold: float       # best target sigmoid prob needed for full recognition points
    margin_required: float     # lead over the strongest other class for full distinctness points
    hold_seconds: float        # time the signal should stay recognised for full hold points
    correct_cut: int           # min total score for CORRECT
    almost_cut: int            # min total score for ALMOST


LEVEL_CONFIG = {
    "beginner": LevelConfig("beginner", "Beginner", 0.55, 0.15, 0.6, 60, 40),
    "standard": LevelConfig("standard", "Standard", 0.75, 0.30, 1.0, 75, 55),
    "referee": LevelConfig("referee", "Referee", 0.90, 0.45, 1.5, 90, 68),
}
# LEVEL_OVERRIDES in trainer_config.py, e.g. {"referee": {"correct_cut": 92}}. Keys: rec_threshold, margin_required,
# hold_seconds, correct_cut, almost_cut.
for _lv, _d in (getattr(_tc, "LEVEL_OVERRIDES", None) or {}).items():
    if _lv in LEVEL_CONFIG:
        LEVEL_CONFIG[_lv] = replace(LEVEL_CONFIG[_lv], **{k: v for k, v in _d.items() if k in (
            "rec_threshold", "margin_required", "hold_seconds", "correct_cut", "almost_cut")})


def pick(level: str, beginner, standard, referee):
    """One place to read a threshold for the current difficulty level."""
    return {"beginner": beginner, "standard": standard, "referee": referee}[level]


# ----------------------------------------------------------------------------
# Signal reference material (used by the Learn screen, the drill intro card,
# and practice-mode hints). Edit freely to match the rulebook edition your
# referee validator uses.
# ----------------------------------------------------------------------------

SIGNALS: Dict[str, dict] = {
    "team_to_serve_left": {
        "title": "Team to Serve (your LEFT arm)",
        "short": "Team to Serve - left arm",
        "fivb": "Extend the arm to the side of team that will serve.",
        "howto": [
            "Stand about 2 m from the camera, facing it, arms relaxed. Head, shoulders and hips should be in view.",
            "Swing your LEFT arm out to your left in ONE smooth movement, straight from your side to the pointing position. Do not bend the elbow first.",
            "Keep the arm as straight as you can (a slight bend is fine) and about level with your shoulder or a little below, hand open with the fingers together (not a fist).",
            "Keep your other arm relaxed. Hold about 2 seconds (FIVB: signals are maintained for a moment), then lower the arm.",
        ],
        "mistakes": ["Elbow bent, or only the forearm raised", "Folding the arm in first, then pointing",
                     "Hand closed into a fist", "Using the wrong arm", "Raising both arms"],
        "not_graded": "A perfectly straight arm is hard for one camera to see, so a slight bend is accepted. The FIVB illustration shows an open hand; a closed hand only lowers the score a little.",
    },
    "team_to_serve_right": {
        "title": "Team to Serve (your RIGHT arm)",
        "short": "Team to Serve - right arm",
        "fivb": "Extend the arm to the side of team that will serve.",
        "howto": [
            "Stand about 2 m from the camera, facing it, arms relaxed. Head, shoulders and hips should be in view.",
            "Swing your RIGHT arm out to your right in ONE smooth movement, straight from your side to the pointing position. Do not bend the elbow first.",
            "Keep the arm straight and about level with your shoulder or a little below, hand open with the fingers together (not a fist).",
            "Keep your other arm relaxed. Hold about 2 seconds (FIVB: signals are maintained for a moment), then lower the arm.",
        ],
        "mistakes": ["Elbow bent, or only the forearm raised", "Folding the arm in first, then pointing",
                     "Hand closed into a fist", "Using the wrong arm", "Raising both arms"],
        "not_graded": "The FIVB illustration shows an open hand. A closed hand only lowers the score a little; the camera can misread fingers.",
    },
    "service_authorization_left": {
        "title": "Authorization to Serve (your LEFT hand)",
        "short": "Authorization - left hand",
        "fivb": "Move the hand to indicate direction of service.",
        "howto": [
            "Stand about 2 m from the camera, facing it, arms relaxed. In a match the whistle comes first (FIVB 12.3, 22.2.1.1).",
            "Bend your LEFT elbow and bring your OPEN hand in front of your body at chest height, about the level of your upper arm (bicep), forearm roughly level.",
            "Sweep the hand smoothly toward the serving team on your left, once or twice. Keep the sweep at that chest height.",
            "Keep the hand open (not a fist) and the other arm relaxed.",
            "Hold the final position for about a second, then lower the arm.",
        ],
        "mistakes": ["Hand above the shoulders or down at the belly", "Hand not moving (static pose)",
                     "Arm fully straight like Team to Serve", "Hand closed into a fist", "Using the wrong hand"],
        "not_graded": "FIVB gives no height for the sweep. The chest (upper arm) level is referee guidance used by this trainer, not FIVB text.",
    },
    "service_authorization_right": {
        "title": "Authorization to Serve (your RIGHT hand)",
        "short": "Authorization - right hand",
        "fivb": "Move the hand to indicate direction of service.",
        "howto": [
            "Stand about 2 m from the camera, facing it, arms relaxed. In a match the whistle comes first (FIVB 12.3, 22.2.1.1).",
            "Bend your RIGHT elbow and bring your OPEN hand in front of your body at chest height, about the level of your upper arm (bicep), forearm roughly level.",
            "Sweep the hand smoothly toward the serving team on your right, once or twice. Keep the sweep at that chest height.",
            "Keep the hand open (not a fist) and the other arm relaxed.",
            "Hold the final position for about a second, then lower the arm.",
        ],
        "mistakes": ["Hand above the shoulders or down at the belly", "Hand not moving (static pose)",
                     "Arm fully straight like Team to Serve", "Hand closed into a fist", "Using the wrong hand"],
        "not_graded": "FIVB gives no height for the sweep. The chest (upper arm) level is referee guidance used by this trainer, not FIVB text.",
    },
    "ball_in": {
        "title": "Ball In",
        "short": "Ball In",
        "fivb": "Point the arm and fingers toward the floor.",
        "howto": [
            "Stand about 2 m from the camera, facing it, arms relaxed.",
            "Straighten one arm and point it, with the fingers open and extended, down toward the floor.",
            "Keep the other arm relaxed.",
            "Hold about 2 seconds, then lower the arm.",
        ],
        "mistakes": ["Elbow bent", "Arm raised out to the side instead of pointing down", "Hand closed into a fist"],
        "not_graded": "Which spot of the floor you point at (depth) cannot be measured by one 2D camera. Ball In is also the model's weakest class.",
    },
    "ball_out": {
        "title": "Ball Out",
        "short": "Ball Out",
        "fivb": "Raise the forearms vertically, hands open, palms towards the body.",
        "howto": [
            "Stand about 2 m from the camera, square to it, so BOTH hands stay visible and do not overlap.",
            "Raise both arms to the same height. For the full form lift the elbows out to about shoulder height so the armpits are open (referee guidance); elbows kept nearer the body still count, with fewer points.",
            "Bend both elbows so both forearms point straight up, hands open with the fingers extended, palms toward your face.",
            "Hold about 2 seconds, then lower both arms.",
        ],
        "mistakes": ["Only one arm raised", "Elbows tucked against the body (armpits closed)",
                     "Forearms leaning instead of vertical", "Arms straight overhead", "Hands closed"],
        "not_graded": "Palm direction (palms toward the body) cannot be measured reliably by one 2D camera; check it with your instructor. The open armpits are referee guidance, not FIVB text, and only earn marks.",
    },
    "double_contact": {
        "title": "Double Contact",
        "short": "Double Contact",
        "fivb": "Raise two fingers, spread open.",
        "howto": [
            "Stand about 2 m from the camera, facing it, arms relaxed.",
            "Raise ONE hand to about shoulder height or higher, facing the camera. In a match use the hand on the side of the team that made the fault (FIVB 30.1).",
            "Show exactly two fingers (index and middle), spread apart like a V. Curl the ring and little fingers.",
            "Keep the other arm down. Hold about 2 seconds, then lower the hand.",
        ],
        "mistakes": ["Hand too low", "One, three or four fingers", "Two fingers held together instead of spread",
                     "Hand turned away from the camera"],
        "not_graded": "How far apart the two fingers are spread is not measured.",
    },
    "end_of_set": {
        "title": "End of Set",
        "short": "End of Set",
        "fivb": "Cross the forearms in front of the chest, hands open.",
        "howto": [
            "Stand about 2 m from the camera, square to it. Keep your hands visible: do not let one hand cover the other.",
            "Bend your elbows, bring both forearms up and cross them in front of your CHEST (not at the belly, not overhead). Overlapping forearms are fine.",
            "Keep both hands open with the fingers extended, and turn them so the camera can see them.",
            "Hold about 2 seconds, then lower both arms.",
        ],
        "mistakes": ["Hugging yourself (hands on elbows) low on the body", "Forearms not actually crossing",
                     "Arms too low or too high", "Hands closed or hidden behind each other"],
        "not_graded": "",
    },
}

# Shown in the Learn window: the similar-looking pairs that beginners (and scorekeepers) mix up.
_COMPARE = {
    "team_to_serve_left": "Do not confuse it with Authorization to Serve: Team to Serve is ONE straight arm held out to the side; Authorization is a bent arm whose hand sweeps at chest height.",
    "team_to_serve_right": "Do not confuse it with Authorization to Serve: Team to Serve is ONE straight arm held out to the side; Authorization is a bent arm whose hand sweeps at chest height.",
    "service_authorization_left": "Do not confuse it with Team to Serve: Authorization is a bent arm whose open hand SWEEPS at chest height; Team to Serve is a straight arm HELD out to the side.",
    "service_authorization_right": "Do not confuse it with Team to Serve: Authorization is a bent arm whose open hand SWEEPS at chest height; Team to Serve is a straight arm HELD out to the side.",
    "ball_in": "Do not confuse it with Team to Serve: Ball In points DOWN toward the floor; Team to Serve points out to the side at shoulder level.",
    "ball_out": "Do not confuse it with End of Set: Ball Out raises BOTH forearms straight up; End of Set crosses the forearms in front of the chest.",
    "double_contact": "Do not confuse it with Team to Serve or Authorization: Double Contact is ONE raised hand showing exactly two spread fingers.",
    "end_of_set": "Do not confuse it with Ball Out: End of Set crosses the forearms in front of the chest; Ball Out raises both forearms straight up.",
}
for _label, _txt in _COMPARE.items():
    SIGNALS[_label]["compare"] = _txt

# Every graded check is taught in one of the numbered "how to" steps above (tests/test_gesture_grader.py enforces it).
_TTS_TAUGHT = {"arm_extended": 3, "points_to_side": 2, "no_contraction": 2, "hands_open": 3, "other_arm_down": 4,
               "ready_position": 4}
_SA_TAUGHT = {"arm_bent": 2, "hand_moves": 3, "at_chest": 2, "hands_open": 4, "toward_side": 3, "other_arm_down": 4,
              "ready_position": 5}
_TAUGHT = {
    "team_to_serve_left": _TTS_TAUGHT, "team_to_serve_right": _TTS_TAUGHT,
    "service_authorization_left": _SA_TAUGHT, "service_authorization_right": _SA_TAUGHT,
    "ball_in": {"arm_extended": 2, "arm_lowered": 2, "hand_open": 2, "other_arm_relaxed": 3, "ready_position": 4},
    "ball_out": {"forearms_vertical": 3, "arms_raised": 2, "arms_symmetric": 2, "hands_open": 3, "elbows_bent": 3,
                 "ready_position": 4},
    "double_contact": {"hand_raised": 2, "hand_side": 2, "two_fingers": 3, "other_arm_down": 4, "ready_position": 4},
    "end_of_set": {"forearms_crossed": 2, "at_chest": 2, "elbows_bent": 2, "hands_open": 3, "ready_position": 4},
}
for _label, _t in _TAUGHT.items():
    SIGNALS[_label]["taught"] = _t

# Shown first in the Learn window and referred to on the instruction cards.
SETUP_TIPS = [
    "Stand about 2 meters (6 feet) from the camera and face it squarely. Do not stand sideways.",
    "Your head, shoulders, arms and hips should all be in view, with room on both sides for your arms.",
    "Keep the camera still, at about chest to eye height.",
    "Use even light from the front. Avoid a bright window behind you.",
    "Wear a top that contrasts with the background, and avoid loose sleeves that hide the elbows.",
    "Keep both hands visible and not overlapping. The open hand and finger checks need the camera to see your hands.",
    "Raise your LEFT arm once: the blue arm on screen should be the one you raised (see trainer_config.py if not).",
]

DISCLAIMER = ("Training guide only. The system uses one camera and an AI model, so it can make mistakes. "
              "It is not an official FIVB judgement and does not replace a coach.")

KNOWN_WEAK = {
    "ball_in": "Ball In is the weakest signal for this system (lower recognition rate, and it was trained on the "
               "older straight-down form), so a low score can partly reflect the model and not only your form.",
}

# Ball Out with open armpits (elbows out to about shoulder height) is referee guidance, not FIVB text (which only says raise the
# forearms vertically, hands open, palms towards the body). Measured on the team's own attempts the elbows are usually
# kept near the body, so by default open armpits only earn marks. True = tucked elbows cap the verdict at ALMOST.
ARMPITS_REQUIRED = bool(getattr(_tc, "ARMPITS_REQUIRED", False))

# FIVB 30.1: signals are "maintained for a moment". Service Authorization is a moving signal, so the
# time the model keeps recognising it is shorter; its hold requirement is scaled down.
HOLD_SCALE = {"service_authorization_left": 0.6, "service_authorization_right": 0.6,
              "ball_in": 0.5}          # Ball In is recognised only briefly (measured 0.2 to 0.7 s)

# While moving into a signal, the arm passes through poses that look like a neighbouring signal (for example the sweep
# of Authorization to Serve or the arm going down for Ball In passes through Team to Serve). The system does not commit
# anything for those in-between poses, so they are ignored and never shown as "looks like ...".
TRANSITION_CONFUSIONS = {
    "team_to_serve_left": {"service_authorization_left", "ball_in"},
    "team_to_serve_right": {"service_authorization_right", "ball_in"},
    "service_authorization_left": {"team_to_serve_left"},
    "service_authorization_right": {"team_to_serve_right"},
    "ball_in": {"team_to_serve_left", "team_to_serve_right"},
    "ball_out": {"end_of_set", "double_contact"},
    "double_contact": {"team_to_serve_left", "team_to_serve_right", "service_authorization_left",
                       "service_authorization_right", "end_of_set"},
    "end_of_set": {"ball_out"},
}

# Open hands: the FIVB text names them for Ball Out and End of Set, and the FIVB illustrations show an open hand for the
# other signals too. Reading fingers from one camera can be wrong, so by default a closed hand only costs a few points
# and shows a "keep your hand open" tip; it does NOT change the verdict. True = a visible fist caps the verdict at ALMOST.
OPEN_HAND_STRICT = bool(getattr(_tc, "OPEN_HAND_STRICT", False))


def pretty_label(label: str) -> str:
    return SIGNALS.get(label, {}).get("title", label.replace("_", " ").title())


def short_label(label: str) -> str:
    return SIGNALS.get(label, {}).get("short", label.replace("_", " ").title())


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------

def _pct(values, mask, q):
    v = np.asarray(values, dtype=float)[mask]
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if len(v) else float("nan")


def _deg(cos_value):
    return np.degrees(np.arccos(np.clip(cos_value, -1.0, 1.0)))


class Arm:
    """Per-frame, aspect-corrected geometry for ONE arm (side = 'left'/'right')."""

    def __init__(self, geo: "Geo", side: str):
        self.side = side
        p = geo.p
        s, e, w = ("ls", "le", "lw") if side == "left" else ("rs", "re", "rw")
        sx, sy, _ = p[s]
        ex, ey, ev = p[e]
        wx, wy, wv = p[w]
        out_sign = geo.sgn if side == "left" else -geo.sgn

        ax, ay = wx - sx, wy - sy
        alen = np.maximum(np.hypot(ax, ay), 1e-6)
        v1x, v1y = sx - ex, sy - ey
        v2x, v2y = wx - ex, wy - ey
        n1 = np.maximum(np.hypot(v1x, v1y), 1e-6)
        n2 = np.maximum(np.hypot(v2x, v2y), 1e-6)

        ulen = np.maximum(np.hypot(ex - sx, ey - sy), 1e-6)
        # upper-arm angle from hanging straight down: ~0 arm at the side, ~90 elbow out at shoulder height
        # (armpit open), larger = elbow raised above the shoulder line
        self.upper_from_down = _deg((ey - sy) / ulen)
        # 180 = perfectly straight arm
        self.elbow = _deg((v1x * v2x + v1y * v2y) / (n1 * n2))
        # angle of the whole arm from "hanging straight down": 0 down, 90 horizontal, 180 straight up
        self.from_down = _deg(ay / alen)
        # outward component of the arm direction, away from the body (0..1 for an arm held to the side)
        self.outward = out_sign * ax / alen
        # forearm angle from vertical-up: 0 = forearm perfectly vertical, wrist above elbow
        self.forearm_up = _deg(-v2y / n2)
        # wrist height above the shoulder line, in shoulder widths (negative = below the shoulder)
        self.elev = (geo.sh_mid_y - wy) / geo.sw
        # wrist position outward of its own shoulder, in shoulder widths
        self.wr_out = out_sign * (wx - sx) / geo.sw
        # wrist vertical position: 0 = shoulder line, 1 = hip line
        self.wr_frac = (wy - geo.sh_mid_y) / geo.torso
        # >0 once the wrist has crossed the body midline to the OTHER side (shoulder widths)
        self.cross = -(wx - geo.mid_x) * out_sign / geo.sw
        # wrist position relative to the body, used for motion (sweep) measurement
        self.pos_x = out_sign * (wx - geo.mid_x) / geo.sw
        self.pos_y = (wy - geo.sh_mid_y) / geo.sw

        self.mask = geo.valid & (ev >= 0.4) & (wv >= 0.4)
        self.vis_frac = float(self.mask.sum() / max(1, geo.valid.sum()))

    def hi(self, arr):
        return _pct(arr, self.mask, 65)   # "mostly held": tolerant of onset / release frames

    def lo(self, arr):
        return _pct(arr, self.mask, 35)

    def mid(self, arr):
        return _pct(arr, self.mask, 50)

    def pct(self, arr, q):
        return _pct(arr, self.mask, q)

    def motion(self):
        """Size of the wrist's sweep in shoulder widths (robust range, not path length,
        so tracking jitter does not count as movement)."""
        dx = _pct(self.pos_x, self.mask, 90) - _pct(self.pos_x, self.mask, 10)
        dy = _pct(self.pos_y, self.mask, 90) - _pct(self.pos_y, self.mask, 10)
        if not (math.isfinite(dx) and math.isfinite(dy)):
            return float("nan")
        return float(math.hypot(dx, dy))


class Hand:
    """Finger-extension summary for one hand (flags come from extract_keypoints.py)."""

    def __init__(self, geo: "Geo", side: str):
        raw = geo.raw
        det_col = 108 if side == "left" else 109
        f0 = 110 if side == "left" else 115
        det = (raw[:, det_col] > 0.5) & geo.valid
        self.coverage = float(det.sum() / max(1, geo.valid.sum()))
        self.ext = raw[det, f0:f0 + 5].mean(axis=0) if det.any() else None  # thumb, index, middle, ring, pinky

    def extended_count(self):
        """Non-thumb fingers extended (0-4). None when the hand was not seen well enough."""
        if self.ext is None or self.coverage < MIN_HAND_COVERAGE:
            return None
        return int((self.ext[1:] > 0.5).sum())


class Geo:
    """Aspect-corrected geometry for a raw window of frames (N x >= 24)."""

    def __init__(self, raw, aspect: float = 9.0 / 16.0):
        raw = np.asarray(raw, dtype=float)
        if raw.ndim != 2 or raw.shape[1] < 24:
            raise ValueError("raw window must be (frames x >=24 features)")
        if raw.shape[1] < 122:  # allow pose-only arrays (hand checks then report 'unverified')
            pad = np.zeros((raw.shape[0], 122 - raw.shape[1]))
            raw = np.concatenate([raw, pad], axis=1)
        self.raw = raw
        self.n = len(raw)
        self.aspect = aspect

        def lm(i):
            return raw[:, i * 3], raw[:, i * 3 + 1] * aspect, raw[:, i * 3 + 2]

        self.p = {name: lm(i) for i, name in enumerate(["ls", "rs", "le", "re", "lw", "rw", "lh", "rh"])}
        lsx, rsx = self.p["ls"][0], self.p["rs"][0]
        self.sgn = np.where(lsx - rsx >= 0, 1.0, -1.0)   # +1 when the subject's left is on the image right
        self.sw = np.maximum(np.abs(lsx - rsx), 1e-4)
        self.mid_x = (lsx + rsx) / 2.0
        self.sh_mid_y = (self.p["ls"][1] + self.p["rs"][1]) / 2.0

        hips_vis = float(np.median((self.p["lh"][2] + self.p["rh"][2]) / 2.0))
        self.hips_reliable = hips_vis >= 0.4
        est_hip_y = self.sh_mid_y + 1.3 * self.sw   # torso ~ 1.3 shoulder widths when hips are out of frame
        hip_y = (self.p["lh"][1] + self.p["rh"][1]) / 2.0
        self.hip_y = hip_y if self.hips_reliable else est_hip_y
        self.torso = np.maximum(self.hip_y - self.sh_mid_y, 0.5 * self.sw)

        pose_present = (raw[:, :24] != 0).any(axis=1)
        shoulders_seen = (self.p["ls"][2] >= 0.3) & (self.p["rs"][2] >= 0.3)
        self.valid = pose_present & shoulders_seen
        self._arms: Dict[str, Arm] = {}
        self._hands: Dict[str, Hand] = {}

    def arm(self, side: str) -> Arm:
        if side not in self._arms:
            self._arms[side] = Arm(self, side)
        return self._arms[side]

    def hand(self, side: str) -> Hand:
        if side not in self._hands:
            self._hands[side] = Hand(self, side)
        return self._hands[side]


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------

@dataclass
class Check:
    id: str
    label: str               # what is being checked, in trainee wording
    basis: str               # "FIVB" or "TRAINING"
    weight: int
    critical: bool
    status: str              # "pass" | "fail" | "unverified"
    value: Optional[float]
    need: str                # human-readable requirement, e.g. ">= 150 deg"
    tip: str                 # what to change when it fails
    strict: bool = False     # a FAILED strict check caps the verdict at ALMOST; an unverified one does not

    def feedback(self) -> str:
        if self.status == "fail":
            shown = "" if self.value is None or not math.isfinite(self.value) else f" (you: {self.value:.2f}, need {self.need})"
            return f"{self.tip}{shown}"
        if self.status == "unverified":
            return f"Could not verify: {self.label}. Make sure that part of you is clearly visible to the camera."
        return ""


def _fmt_need(ge, le, unit):
    if ge is not None and le is not None:
        return f"{ge:g} to {le:g}{unit}"
    if ge is not None:
        return f">= {ge:g}{unit}"
    if le is not None:
        return f"<= {le:g}{unit}"
    return ""


# Threshold / effect overrides typed in trainer_config.py:
#   GRADER_OVERRIDES = {"service_authorization.hand_moves": {"ge": (0.5, 1.0, 1.4)},          # beginner, standard, referee
#                       "service_authorization_left.arm_bent": {"le": 140, "effect": "important"}}
# The key is "<signal>.<check id>": the signal without _left/_right applies to both sides, with it to that side only.
# "ge" = minimum, "le" = maximum (a number for all levels, or a (beginner, standard, referee) tuple),
# "effect" = "required" | "important" | "scored", "weight" = points. See tools/export_rubric.py --print for the ids.
OVERRIDES = dict(getattr(_tc, "GRADER_OVERRIDES", None) or {})
_CTX = {"label": None, "level": "standard"}


def _override(cid, key, default):
    if not OVERRIDES or _CTX["label"] is None:
        return default
    label = _CTX["label"]
    base = label[:-5] if label.endswith(("_left", "_right")) and not label.startswith("ball") else label
    if label.endswith("_left"):
        base = label[:-5]
    elif label.endswith("_right"):
        base = label[:-6]
    for k in (f"{label}.{cid}", f"{base}.{cid}"):
        v = OVERRIDES.get(k)
        if v is not None and key in v:
            val = v[key]
            return pick(_CTX["level"], *val) if isinstance(val, (tuple, list)) else val
    return default


def _c(cid, label, weight, critical, value, *, ge=None, le=None, arm: Optional[Arm] = None,
       basis="FIVB", tip="", unit="", verifiable=True, strict=False) -> Check:
    ge = _override(cid, "ge", ge)
    le = _override(cid, "le", le)
    weight = _override(cid, "weight", weight)
    effect = _override(cid, "effect", None)
    if effect is not None:
        critical, strict = effect == "required", effect == "important"
    status = "unverified"
    arm_ok = arm is None or arm.vis_frac >= MIN_ARM_VISIBLE_FRAC
    if verifiable and arm_ok and value is not None and math.isfinite(value):
        ok = (ge is None or value >= ge) and (le is None or value <= le)
        status = "pass" if ok else "fail"
    return Check(cid, label, basis, weight, critical, status,
                 None if value is None else float(value), _fmt_need(ge, le, unit), tip, strict)


def _other(side):
    return "right" if side == "left" else "left"


# ----------------------------------------------------------------------------
# FIVB-derived rules, one function per signal. Every rule set sums to 40 points.
# Signature: (g, lv, gcap=None, ctx=None)
#   g    Geo of the analysis window (middle of the recognised hold)
#   gcap Geo of the whole capture (or the part of it that belongs to this signal); used for motion checks
#   ctx  optional context, e.g. {"side": "left"} = the team at fault is on the trainee's left
# Official wording used (Official Volleyball Rules 2025-2028, Diagram 11, checked against the 2025-2028 text): Authorisation to serve "Move the hand to indicate
# direction of service"; Team to serve "Extend the arm to the side of team that will serve"; Ball in "Point the
# arm and fingers toward the floor"; Ball out "Raise the forearms vertically, hands open, palms towards the
# body"; Double contact "Raise two fingers, spread open"; End of set "Cross the forearms in front of the chest,
# hands open". Rule 30.1: signals are maintained for a moment; a one-handed signal uses the hand on the side of
# the team at fault or making the request.
# ----------------------------------------------------------------------------

def _rules_team_to_serve(g: Geo, lv: str, side: str, gcap=None, ctx=None) -> List[Check]:
    A, O = g.arm(side), g.arm(_other(side))
    Ac = (gcap or g).arm(side)
    S = side.upper()
    n_open = g.hand(side).extended_count()
    return [
        _c("arm_extended", f"{S} arm extended (FIVB: extend the arm)", 12, True, A.lo(A.elbow),
           ge=pick(lv, 130, 145, 155), unit=" deg", arm=A,
           tip=f"Straighten your {side} arm. Do not keep the elbow clearly bent or raise only the forearm"),
        _c("points_to_side", f"{S} arm pointing out to the side", 12, True,
           A.hi(A.outward), ge=pick(lv, 0.45, 0.65, 0.85), arm=A,
           tip=f"Swing your {side} arm out to the side, about level with your shoulder"),
        _c("no_contraction", "Arm swings out straight (no folding in first)", 8, False,
           Ac.pct(Ac.elbow, 10), ge=pick(lv, 100, 120, 132), unit=" deg", arm=Ac, strict=True, basis="TRAINING",
           tip=f"Swing your {side} arm from your side to the pointing position in one movement; do not fold it in first"),
        _c("hands_open", "Hand open (as in the FIVB illustration)", 4, False,
           None if n_open is None else float(n_open), ge=pick(lv, 2, 3, 4), verifiable=n_open is not None,
           strict=OPEN_HAND_STRICT, basis="TRAINING",
           tip="Keep your hand open with the fingers extended, not a fist"),
        _c("other_arm_down", "Other arm kept down (one-arm signal)", 4, False,
           O.mid(O.from_down), le=pick(lv, 55, 40, 30), unit=" deg", arm=O,
           tip=f"Keep your {_other(side)} arm relaxed at your side"),
    ]


def _rules_authorization(g: Geo, lv: str, side: str, gcap=None, ctx=None) -> List[Check]:
    A, O = g.arm(side), g.arm(_other(side))
    S = side.upper()
    n_open = g.hand(side).extended_count()
    return [
        _c("arm_bent", f"{S} elbow bent for the sweeping motion", 8, False, A.mid(A.elbow),
           le=pick(lv, 150, 145, 140), unit=" deg", arm=A, basis="TRAINING", strict=(lv != "beginner"),
           tip=f"Bend your {side} elbow; a nearly straight arm is sloppy and reads as Team to Serve"),
        _c("hand_moves", "Hand moves (FIVB: move the hand to indicate the direction)", 12, True,
           A.motion(), ge=pick(lv, 0.5, 1.0, 1.4), arm=A,
           tip="Make a clearer sweeping motion with your hand; a still pose is not this signal"),
        _c("at_chest", "Sweep at chest (upper arm) level", 9, False, A.mid(A.wr_frac),
           ge=pick(lv, -0.15, 0.0, 0.05), le=pick(lv, 0.80, 0.65, 0.55), arm=A, strict=True, basis="TRAINING",
           tip="Keep the sweep at chest height, about the level of your upper arm: not above the shoulders, not down at the belly"),
        _c("hands_open", "Hand open (as in the FIVB illustration)", 6, False,
           None if n_open is None else float(n_open), ge=pick(lv, 2, 3, 4), verifiable=n_open is not None,
           strict=OPEN_HAND_STRICT, basis="TRAINING",
           tip="Keep your hand open with the fingers extended, not a fist"),
        _c("toward_side", f"Hand travels toward the {side} side (direction of service)", 3, False,
           A.hi(A.wr_out), ge=pick(lv, -0.4, -0.2, 0.0), arm=A,
           tip=f"Sweep your hand further toward your {side} side"),
        _c("other_arm_down", "Other arm kept down", 2, False, O.mid(O.from_down),
           le=pick(lv, 55, 40, 30), unit=" deg", arm=O,
           tip=f"Keep your {_other(side)} arm relaxed at your side"),
    ]


def _rules_ball_in(g: Geo, lv: str, gcap=None, ctx=None) -> List[Check]:
    """FIVB: point the arm and fingers toward the floor. Where the arm points in DEPTH cannot be measured by one
    frontal 2D camera, so an extended arm is only scored, while an arm that is not pointing down (raised out to the
    side) caps the verdict. The recognition model is also weakest on this class."""
    L, R = g.arm("left"), g.arm("right")

    def key(a: Arm):
        o, e = a.mid(a.outward), a.mid(a.elbow)
        if not (math.isfinite(o) and math.isfinite(e)) or a.vis_frac < MIN_ARM_VISIBLE_FRAC:
            return -9.0
        return o + 0.5 * e / 180.0

    sig_side = "left" if key(L) >= key(R) else "right"
    A, O = g.arm(sig_side), g.arm(_other(sig_side))
    n_open = g.hand(sig_side).extended_count()
    return [
        _c("arm_extended", "Pointing arm extended (FIVB: point the arm)", 14, False, A.hi(A.elbow),
           ge=pick(lv, 130, 145, 155), unit=" deg", arm=A,
           tip="Straighten the pointing arm; do not leave the elbow bent"),
        _c("arm_lowered", "Arm pointing toward the floor", 14, False, A.mid(A.from_down),
           le=pick(lv, 70, 55, 50), unit=" deg", arm=A, strict=True,
           tip="Point the arm lower, toward the floor, not out to the side"),
        _c("hand_open", "Fingers extended (FIVB: arm and fingers)", 8, False,
           None if n_open is None else float(n_open), ge=pick(lv, 2, 3, 4), verifiable=n_open is not None,
           tip="Keep your fingers open and straight, do not make a fist"),
        _c("other_arm_relaxed", "Other arm relaxed", 4, False, O.mid(O.from_down),
           le=pick(lv, 55, 40, 30), unit=" deg", arm=O,
           tip="Keep your other arm relaxed at your side"),
    ]


def _rules_ball_out(g: Geo, lv: str, gcap=None, ctx=None) -> List[Check]:
    L, R = g.arm("left"), g.arm("right")

    def worst(fn, agg):
        a, b = fn(L), fn(R)
        return agg(a, b) if (math.isfinite(a) and math.isfinite(b)) else float("nan")

    worst_fore = worst(lambda A: A.mid(A.forearm_up), max)          # least vertical forearm
    least_raised = worst(lambda A: A.mid(A.upper_from_down), min)   # lowest upper arm (armpit least open)
    worst_elbow = worst(lambda A: A.mid(A.elbow), max)
    wl, wr = L.mid(L.elev), R.mid(R.elev)
    asym = abs(wl - wr) if (math.isfinite(wl) and math.isfinite(wr)) else float("nan")
    nl, nr = g.hand("left").extended_count(), g.hand("right").extended_count()
    seen = [n for n in (nl, nr) if n is not None]
    worst_open = float(min(seen)) if seen else None
    both_vis = L.vis_frac >= MIN_ARM_VISIBLE_FRAC and R.vis_frac >= MIN_ARM_VISIBLE_FRAC
    return [
        _c("forearms_vertical", "BOTH forearms vertical (FIVB: raise the forearms vertically)", 12, True,
           worst_fore, le=pick(lv, 40, 30, 20), unit=" deg", verifiable=both_vis,
           tip="Raise BOTH forearms straight up; keep them vertical, not leaning"),
        _c("arms_raised", "Elbows lifted out to the sides, armpits open (full marks)", 12, ARMPITS_REQUIRED, least_raised,
           ge=pick(lv, 20, 40, 60), unit=" deg", verifiable=both_vis, basis="TRAINING",
           tip="For the full form lift your elbows up and out to about shoulder height so your armpits are open"),
        _c("arms_symmetric", "Both arms at the same height", 5, False, asym,
           le=pick(lv, 0.6, 0.4, 0.3), verifiable=both_vis,
           tip="Raise both arms to the same height"),
        _c("hands_open", "Hands open (FIVB: hands open)", 6, False, worst_open,
           ge=pick(lv, 2, 3, 4), verifiable=worst_open is not None, strict=OPEN_HAND_STRICT,
           tip="Keep both hands open with the fingers extended"),
        _c("elbows_bent", "Elbows bent, forearms up (not arms straight overhead)", 5, True, worst_elbow,
           le=pick(lv, 150, 135, 125), unit=" deg", verifiable=both_vis,
           tip="Keep your elbows bent with the forearms vertical; do not stretch your arms overhead"),
    ]


def _rules_double_contact(g: Geo, lv: str, gcap=None, ctx=None) -> List[Check]:
    L, R = g.arm("left"), g.arm("right")

    def key(a: Arm):
        v = a.hi(a.elev)
        return v if (math.isfinite(v) and a.vis_frac >= MIN_ARM_VISIBLE_FRAC) else -99.0

    sig_side = "left" if key(L) >= key(R) else "right"
    A, O = g.arm(sig_side), g.arm(_other(sig_side))
    H = g.hand(sig_side)
    want_side = (ctx or {}).get("side")

    if H.ext is None or H.coverage < MIN_HAND_COVERAGE:
        two = Check("two_fingers", "Exactly two fingers shown (FIVB: raise two fingers)", "FIVB",
                    18 if want_side else 20, True, "unverified", None, "index + middle only",
                    "Show exactly two fingers to the camera")
    else:
        f = H.ext
        idx_mid = bool(f[1] > 0.5 and f[2] > 0.5)
        curled = int(f[3] <= 0.5) + int(f[4] <= 0.5)
        ok = idx_mid and curled >= pick(lv, 1, 2, 2)
        n_ext = float((f[1:] > 0.5).sum())
        two = Check("two_fingers", "Exactly two fingers shown (FIVB: raise two fingers)", "FIVB",
                    18 if want_side else 20, True, "pass" if ok else "fail", n_ext, "index + middle only",
                    "Show exactly two fingers (index and middle), spread apart, and curl the ring and little fingers")
    checks = [
        _c("hand_raised", "Hand raised (FIVB: raise two fingers)", 10 if want_side else 12, True, A.hi(A.elev),
           ge=pick(lv, -0.6, -0.3, 0.0), arm=A,
           tip="Raise your hand higher, to about shoulder height or above"),
        two,
        _c("other_arm_down", "Other arm kept down", 6 if want_side else 8, False, O.mid(O.from_down),
           le=pick(lv, 55, 40, 30), unit=" deg", arm=O,
           tip="Keep your other arm relaxed at your side"),
    ]
    if want_side:
        used = A.vis_frac >= MIN_ARM_VISIBLE_FRAC or O.vis_frac >= MIN_ARM_VISIBLE_FRAC
        checks.append(_c("hand_side", f"Used your {want_side.upper()} hand (side of the team at fault, FIVB 30.1)", 6, False,
                         1.0 if sig_side == want_side else 0.0, ge=1.0, verifiable=used, strict=True,
                         tip=f"Use your {want_side} hand: a one-handed signal uses the hand on the side of the team at fault"))
    return checks


def _rules_end_of_set(g: Geo, lv: str, gcap=None, ctx=None) -> List[Check]:
    L, R = g.arm("left"), g.arm("right")
    cl, cr = L.hi(L.cross), R.hi(R.cross)
    worst_cross = min(cl, cr) if (math.isfinite(cl) and math.isfinite(cr)) else float("nan")
    fl, fr = L.mid(L.wr_frac), R.mid(R.wr_frac)
    lo_b, hi_b = pick(lv, -0.3, -0.1, 0.0), pick(lv, 1.1, 0.95, 0.8)
    if math.isfinite(fl) and math.isfinite(fr):
        # distance outside the allowed chest band (0 when both wrists are inside it)
        chest_value = max(0.0, lo_b - min(fl, fr), max(fl, fr) - hi_b)
    else:
        chest_value = float("nan")
    el_l, el_r = L.mid(L.elbow), R.mid(R.elbow)
    worst_elbow = max(el_l, el_r) if (math.isfinite(el_l) and math.isfinite(el_r)) else float("nan")
    nl, nr = g.hand("left").extended_count(), g.hand("right").extended_count()
    seen = [n for n in (nl, nr) if n is not None]
    worst_open = float(min(seen)) if seen else None
    both_vis = L.vis_frac >= MIN_ARM_VISIBLE_FRAC and R.vis_frac >= MIN_ARM_VISIBLE_FRAC
    # Crossed = each wrist has passed the middle of the body, OR the two wrists are on top of each other near the middle.
    # The second form matters because the tracker often loses or swaps the wrists when the forearms overlap.
    thr_cross, thr_inner = pick(lv, 0.0, 0.05, 0.15), pick(lv, 0.45, 0.35, 0.30)
    pl, pr = L.mid(L.pos_x), R.mid(R.pos_x)
    inner = max(abs(pl), abs(pr)) if (math.isfinite(pl) and math.isfinite(pr)) else float("nan")
    crossed_now = math.isfinite(worst_cross) and worst_cross >= thr_cross
    overlapped = math.isfinite(inner) and inner <= thr_inner
    if not both_vis or not (math.isfinite(worst_cross) or math.isfinite(inner)):
        c_status = "unverified"
    else:
        c_status = "pass" if (crossed_now or overlapped) else "fail"
    crossed = Check("forearms_crossed", "Forearms crossed (FIVB: cross the forearms)", "FIVB", 16, True, c_status,
                    None if not math.isfinite(worst_cross) else float(worst_cross),
                    f">= {thr_cross:g}, or both wrists close together at the middle",
                    "Cross your forearms so the wrists pass each other in front of your chest")
    return [
        crossed,
        _c("at_chest", "Crossed in front of the chest (FIVB)", 10, False, chest_value, le=0.0, verifiable=both_vis,
           strict=(lv == "referee"),          # only the strictest level caps the verdict; otherwise it only costs points
           tip="Cross your forearms at chest height: not low at the belly and not overhead"),
        _c("elbows_bent", "Elbows bent", 6, False, worst_elbow, le=pick(lv, 150, 135, 125), unit=" deg",
           verifiable=both_vis, tip="Bend your elbows to bring the forearms across your chest"),
        _c("hands_open", "Hands open (FIVB: hands open)", 8, False, worst_open,
           ge=pick(lv, 2, 3, 4), verifiable=worst_open is not None, strict=OPEN_HAND_STRICT,
           tip="Keep your hands open with the fingers extended and turned toward the camera"),
    ]


RULES = {
    "team_to_serve_left": lambda g, lv, gcap=None, ctx=None: _rules_team_to_serve(g, lv, "left", gcap, ctx),
    "team_to_serve_right": lambda g, lv, gcap=None, ctx=None: _rules_team_to_serve(g, lv, "right", gcap, ctx),
    "service_authorization_left": lambda g, lv, gcap=None, ctx=None: _rules_authorization(g, lv, "left", gcap, ctx),
    "service_authorization_right": lambda g, lv, gcap=None, ctx=None: _rules_authorization(g, lv, "right", gcap, ctx),
    "ball_in": _rules_ball_in,
    "ball_out": _rules_ball_out,
    "double_contact": _rules_double_contact,
    "end_of_set": _rules_end_of_set,
}


def ready_position_check(tail_frames, lv: str, aspect: float) -> Check:
    """TRAINING requirement: after the signal, arms return to the ready position."""
    tip = "Lower both arms to the ready position after the signal"
    need = f"wrists at or below {pick(lv, 0.45, 0.55, 0.65):g} of the shoulder-to-hip distance"
    try:
        g = Geo(tail_frames, aspect)
    except ValueError:
        return Check("ready_position", "Arms back at the ready position", "TRAINING", W_READY, False,
                     "unverified", None, need, tip)
    if g.valid.sum() < 3:
        return Check("ready_position", "Arms back at the ready position", "TRAINING", W_READY, False,
                     "unverified", None, need, tip)
    fl, fr = g.arm("left").mid(g.arm("left").wr_frac), g.arm("right").mid(g.arm("right").wr_frac)
    worst = min(fl, fr) if (math.isfinite(fl) and math.isfinite(fr)) else float("nan")
    # wr_frac: 0 = shoulder line, 1 = hip line; hanging arms sit around 1.0 or lower on screen
    return _c("ready_position", "Arms back at the ready position", W_READY, False, worst,
              ge=pick(lv, 0.45, 0.55, 0.65), basis="TRAINING", tip=tip)


def rules_for(label: str, g, lv: str, gcap=None, ctx=None):
    """Run the rule set of a signal with the user's overrides (GRADER_OVERRIDES) applied."""
    _CTX["label"], _CTX["level"] = label, lv
    try:
        return RULES[label](g, lv, gcap, ctx)
    finally:
        _CTX["label"] = None


def grade_form_only(label: str, frames, level: str = "standard", aspect: float = 9.0 / 16.0,
                    capture_frames=None, context: Optional[dict] = None) -> List[Check]:
    """Runs ONLY the FIVB-derived geometric checks on a window of raw frames.
    `capture_frames` (optional) is the whole capture for motion checks; `context` e.g. {"side": "left"}.
    Used by grade_attempt() and by tools/grader_sanity_check.py."""
    if label not in RULES:
        return []
    g = Geo(frames, aspect)
    gcap = None
    if capture_frames is not None and len(capture_frames) >= 6:
        gcap = Geo(capture_frames, aspect)
    return rules_for(label, g, level, gcap, context)


# ----------------------------------------------------------------------------
# Whole-attempt grading
# ----------------------------------------------------------------------------

@dataclass
class AttemptResult:
    target: str
    level: str
    verdict: str
    score: int
    points: int
    parts: Dict[str, float]
    checks: List[Check]
    feedback: List[str]
    recognized: bool
    best_prob: float
    margin: float
    hold_seconds: float
    confused_with: Optional[str]
    pose_fraction: float
    note: str = ""
    hold_required: float = 0.0


def _form_points(checks: List[Check]):
    verifiable = sum(c.weight for c in checks if c.status != "unverified")
    passed = sum(c.weight for c in checks if c.status == "pass")
    if verifiable <= 0:
        return 0.0
    return W_FORM * passed / verifiable


def _arms_stayed_down(capture, aspect) -> bool:
    """True when neither wrist was ever raised above about the hips during the capture."""
    try:
        g = Geo(capture, aspect)
    except ValueError:
        return False
    lows = []
    for side in ("left", "right"):
        a = g.arm(side)
        vals = np.asarray(a.wr_frac, dtype=float)[a.mask]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            lows.append(float(vals.min()))          # smallest wr_frac = highest the wrist ever got
    return bool(lows) and min(lows) >= 0.75


def missing_result(target: str, level: str, msg: str) -> AttemptResult:
    """A signal the trainee never showed (used when grading a sequence)."""
    return AttemptResult(target, level, VERDICT_INCORRECT, 0, 0,
                         {"recognition": 0.0, "distinctness": 0.0, "hold": 0.0, "form": 0.0, "ready_position": 0.0},
                         [], [msg], False, 0.0, 0.0, 0.0, None, 1.0, note=msg)


def grade_attempt(target: str, capture_frames, window_records, label_to_idx: Dict[str, int],
                  level: str = "standard", aspect: float = 9.0 / 16.0,
                  step_seconds: float = 0.3, context: Optional[dict] = None,
                  check_ready: bool = True) -> AttemptResult:
    """
    target          : gesture label the trainee was asked to perform
    capture_frames  : (N x 122) raw features recorded during the capture period
    window_records  : list of dicts, one per model inference during the capture:
                      {"label": predicted label after tie-breaker, "probs": sigmoid vector,
                       "frames": (24 x 122) raw window that produced it}
    label_to_idx    : real-class label -> index in `probs`
    step_seconds    : average time between two consecutive inferences
    context         : optional, e.g. {"side": "left"} (team at fault on the trainee's left)
    check_ready     : False for a signal that is followed directly by another one (sequence)
    """
    cfg = LEVEL_CONFIG[level]
    capture = np.asarray(capture_frames, dtype=float) if len(capture_frames) else np.zeros((0, 122))
    pose_fraction = float((capture[:, :24] != 0).any(axis=1).mean()) if len(capture) else 0.0

    def no_reading(msg):
        return AttemptResult(target, level, VERDICT_NO_READING, 0, 0, {}, [], [msg], False, 0.0, 0.0, 0.0,
                             None, pose_fraction, note=msg)

    if len(capture) == 0 or pose_fraction < MIN_POSE_FRACTION:
        return no_reading("We could not see you clearly. Step back so your head, shoulders and arms are "
                          "in the frame, face the camera, and try again. This attempt is not counted.")
    if not window_records:
        return no_reading("The capture was too short to analyse. Try again. This attempt is not counted.")
    if target not in label_to_idx:
        return no_reading(f"'{target}' is not one of this model's signals.")

    t_idx = label_to_idx[target]
    hold_req = cfg.hold_seconds * HOLD_SCALE.get(target, 1.0)
    labels = [r["label"] for r in window_records]
    p_target = np.array([float(r["probs"][t_idx]) for r in window_records])
    best_i = int(np.argmax(p_target))
    best_p = float(p_target[best_i])
    others = np.array(window_records[best_i]["probs"], dtype=float).copy()
    others[t_idx] = -1.0
    margin = float(best_p - max(0.0, others.max()))

    is_target = [lab == target for lab in labels]
    n_target = int(sum(is_target))
    longest = run = 0
    longest_end = -1
    for i, flag in enumerate(is_target):
        run = run + 1 if flag else 0
        if run > longest:
            longest, longest_end = run, i
    hold_seconds = longest * step_seconds
    # Form is measured on the MIDDLE window of the longest recognised run: that window sits
    # inside the hold, so onset / release frames do not pollute the geometry.
    analysis_i = (longest_end - longest // 2) if longest >= 1 else best_i

    recognized = n_target >= 1 and best_p >= 0.5
    # While the target IS recognised, windows of a neighbouring signal are just the movement passing through and are ignored.
    # When the target was NOT recognised, the neighbour is reported ("the system saw X instead").
    ignored = TRANSITION_CONFUSIONS.get(target, set()) if recognized else set()
    other_counts = Counter(lab for lab in labels if lab not in (target, NOTHING_LABEL) and lab not in ignored)
    dom_other, dom_n = (other_counts.most_common(1)[0] if other_counts else (None, 0))

    # --- FIVB form checks on the middle of the recognised hold (motion checks use the whole capture) ---
    motion_cap = capture
    rec_end = window_records[analysis_i].get("end")
    if rec_end is not None:
        motion_cap = capture[:max(6, min(int(rec_end), len(capture)))]   # up to the end of the hold, not the release
    checks = grade_form_only(target, window_records[analysis_i]["frames"], level, aspect,
                             capture_frames=motion_cap, context=context)
    ready = None
    if check_ready:
        tail = capture[-6:] if len(capture) >= 6 else capture
        ready = ready_position_check(tail, level, aspect)

    rec_pts = W_RECOGNITION * float(np.clip((best_p - 0.3) / max(1e-6, cfg.rec_threshold - 0.3), 0, 1))
    dist_pts = W_DISTINCT * float(np.clip(margin / cfg.margin_required, 0, 1))
    hold_pts = W_HOLD * float(np.clip(hold_seconds / hold_req, 0, 1))
    form_pts = _form_points(checks)
    if ready is None:
        ready_pts = W_READY                       # not applicable: the next signal follows directly
    elif ready.status == "pass":
        ready_pts = W_READY
    elif ready.status == "unverified":
        ready_pts = W_READY * 0.5                 # do not punish what the camera could not see
    else:
        ready_pts = 0.0
    total = int(round(rec_pts + dist_pts + hold_pts + form_pts + ready_pts))

    parts = {"recognition": round(rec_pts, 1), "distinctness": round(dist_pts, 1), "hold": round(hold_pts, 1),
             "form": round(form_pts, 1), "ready_position": round(ready_pts, 1)}

    critical_failed = any((c.critical or c.strict) and c.status == "fail" for c in checks)
    critical_unverified = any(c.critical and c.status == "unverified" for c in checks)
    hold_ok = hold_seconds >= 0.35 * hold_req
    other_dominates = dom_other is not None and dom_n >= 2 and dom_n > n_target

    note = ""
    if not recognized:
        if other_dominates:
            verdict = VERDICT_INCORRECT
            note = f"The system saw {short_label(dom_other)} instead."
        elif best_p < 0.35:
            verdict = VERDICT_INCORRECT
            note = ("No signal seen: your arms stayed down. Raise your arm(s) and perform the signal."
                    if _arms_stayed_down(capture, aspect) else "No clear signal was recognised.")
        else:
            verdict = VERDICT_ALMOST if total >= cfg.almost_cut else VERDICT_INCORRECT
            note = "The system only partly recognised the signal."
    elif other_dominates:
        verdict = VERDICT_ALMOST if total >= cfg.almost_cut else VERDICT_INCORRECT
        note = f"Part of your movement looked like {short_label(dom_other)}."
    elif total >= cfg.correct_cut and hold_ok and not critical_failed and not critical_unverified:
        verdict = VERDICT_CORRECT
    elif total >= cfg.almost_cut:
        verdict = VERDICT_ALMOST
        if critical_unverified and not critical_failed and total >= cfg.correct_cut:
            note = "Good attempt, but a required part could not be verified by the camera."
    else:
        verdict = VERDICT_INCORRECT

    # --- feedback lines: most valuable first ---
    fb: List[str] = []
    if note:
        fb.append(note)
    ordered = sorted([c for c in checks if c.status in ("fail", "unverified")],
                     key=lambda c: (c.status != "fail", not (c.critical or c.strict), -c.weight))
    for c in ordered:
        text = c.feedback()
        if text:
            fb.append(text)
    if hold_seconds < hold_req and recognized:
        fb.append(f"Hold the signal a little longer (FIVB: maintained for a moment). About {hold_req:.1f}s at "
                  f"{cfg.name} level; the system kept seeing it for ~{hold_seconds:.1f}s.")
    if ready is not None and ready.status == "fail":
        fb.append(ready.feedback())
    if verdict != VERDICT_CORRECT and target in KNOWN_WEAK:
        fb.append(KNOWN_WEAK[target])
    if verdict == VERDICT_CORRECT and not fb:
        fb.append("Clean signal. Nice work.")

    all_checks = checks + ([ready] if ready is not None else [])
    return AttemptResult(target, level, verdict, total, POINTS[verdict], parts, all_checks, fb[:6],
                         recognized, best_p, margin, hold_seconds, dom_other, pose_fraction, note, hold_req)


def grade_sequence(targets, capture_frames, window_records, label_to_idx: Dict[str, int],
                   level: str = "standard", aspect: float = 9.0 / 16.0, step_seconds: float = 0.3,
                   contexts=None,
                   order_note: str = "Show the signals in the required order (FIVB 22.2.3.1: the team to serve first, then the reason).") -> List[AttemptResult]:
    """
    Grades signals performed back to back in ONE capture (for example Team to Serve, then Ball Out).
    The model windows are split into one part per signal, then each part is graded like a single attempt.
    Only the last signal is checked for the return to the ready position. If the signals were shown in the wrong
    order, a CORRECT verdict is capped at ALMOST.
    """
    n = len(targets)
    contexts = contexts or [None] * n
    capture = np.asarray(capture_frames, dtype=float) if len(capture_frames) else np.zeros((0, 122))
    recs = list(window_records)
    if not recs:
        return [grade_attempt(t, capture, [], label_to_idx, level, aspect, step_seconds, contexts[i],
                              check_ready=(i == n - 1)) for i, t in enumerate(targets)]

    labels = [r["label"] for r in recs]
    first = [next((k for k, lab in enumerate(labels) if lab == t), None) for t in targets]
    order_ok = True
    cuts = []
    for i in range(n - 1):
        fi, fj = first[i], first[i + 1]
        if fi is not None and fj is not None:
            if fj > fi:
                li = max(k for k in range(fi, fj) if labels[k] == targets[i])
                cuts.append((li + fj) // 2)
            else:
                order_ok = False
                cuts.append((fi + fj) // 2)
        elif fi is not None:
            cuts.append(max(k for k, lab in enumerate(labels) if lab == targets[i]))
        elif fj is not None:
            cuts.append(fj - 1)
        else:
            cuts.append(len(recs) // 2 - 1)
    for i in range(1, len(cuts)):
        cuts[i] = max(cuts[i], cuts[i - 1])

    results = []
    start = 0
    frame_start = 0
    have_ends = all("end" in r for r in recs)
    for i, target in enumerate(targets):
        end = cuts[i] if i < n - 1 else len(recs) - 1
        end = min(max(end, start - 1), len(recs) - 1)
        part = recs[start:end + 1]
        if have_ends and i < n - 1 and part:
            frame_end = max(int(recs[end]["end"]), frame_start + 1)
        else:
            frame_end = len(capture)
        seg = capture[frame_start:frame_end]
        if len(seg) < 6:
            seg = capture
        if not part:
            results.append(missing_result(target, level, f"{short_label(target)} was not seen."))
        else:
            seg_start = frame_start if len(seg) < len(capture) else 0
            rel = [dict(r, end=int(r["end"]) - seg_start) for r in part] if have_ends else part
            results.append(grade_attempt(target, seg, rel, label_to_idx, level, aspect, step_seconds,
                                         contexts[i], check_ready=(i == n - 1)))
        start = end + 1
        frame_start = frame_end if (have_ends and i < n - 1) else frame_start

    if not order_ok:
        for res in results:
            if res.verdict == VERDICT_CORRECT:
                res.verdict, res.points = VERDICT_ALMOST, POINTS[VERDICT_ALMOST]
            res.feedback.insert(0, order_note)
            res.feedback = res.feedback[:6]
            res.note = res.note or order_note
    return results


# ----------------------------------------------------------------------------
# Summaries for the screens (what is checked, and how the score works)
# ----------------------------------------------------------------------------

def graded_summary(label: str, ctx: Optional[dict] = None):
    """What is checked for a signal, in trainee wording. Effect: Required (caps at ALMOST if failed or not visible),
    Important (caps at ALMOST if failed), Scored (only costs points)."""
    if label not in RULES:
        return []
    dummy = np.zeros((10, 122))
    checks = rules_for(label, Geo(dummy), "standard", None, ctx) + [ready_position_check(np.zeros((6, 122)), "standard", 9.0 / 16.0)]
    out = []
    for c in checks:
        effect = "Required" if c.critical else "Important" if c.strict else "Scored"
        out.append({"id": c.id, "label": c.label, "weight": c.weight, "basis": c.basis, "effect": effect})
    return out


def scoring_note(level: str = "standard") -> str:
    cfg = LEVEL_CONFIG[level]
    return (f"Also scored: the system must recognise the signal ({W_RECOGNITION} points) and how clearly ({W_DISTINCT}), "
            f"how long you hold it ({W_HOLD}), and lowering your arms afterwards ({W_READY}, minor). "
            f"You do not need 100/100: {cfg.correct_cut} or more counts as CORRECT at {cfg.name} level, "
            f"{cfg.almost_cut} or more is ALMOST.")