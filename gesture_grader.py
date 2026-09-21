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
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

NOTHING_LABEL = "nothing"

VERDICT_CORRECT = "CORRECT"
VERDICT_ALMOST = "ALMOST"
VERDICT_INCORRECT = "INCORRECT"
VERDICT_NO_READING = "NO_READING"
POINTS = {VERDICT_CORRECT: 10, VERDICT_ALMOST: 5, VERDICT_INCORRECT: 0, VERDICT_NO_READING: 0}

# Score weights (sum = 100)
W_RECOGNITION = 35
W_DISTINCT = 10
W_HOLD = 10
W_FORM = 40
W_READY = 5

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
    "beginner": LevelConfig("beginner", "Beginner", 0.55, 0.15, 0.6, 65, 45),
    "standard": LevelConfig("standard", "Standard", 0.75, 0.30, 1.0, 80, 60),
    "referee": LevelConfig("referee", "Referee", 0.90, 0.45, 1.5, 90, 70),
}


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
        "fivb": "Extend the arm toward the side of the team that will serve.",
        "howto": [
            "Face the camera with your arms relaxed.",
            "Extend your LEFT arm out to your left, toward the serving team.",
            "Keep the elbow straight and the other arm down.",
            "Hold about 1 to 2 seconds, then lower your arm.",
        ],
        "mistakes": ["Elbow still bent", "Arm hanging too close to the body",
                     "Using the wrong arm", "Raising both arms"],
        "not_graded": "",
    },
    "team_to_serve_right": {
        "title": "Team to Serve (your RIGHT arm)",
        "short": "Team to Serve - right arm",
        "fivb": "Extend the arm toward the side of the team that will serve.",
        "howto": [
            "Face the camera with your arms relaxed.",
            "Extend your RIGHT arm out to your right, toward the serving team.",
            "Keep the elbow straight and the other arm down.",
            "Hold about 1 to 2 seconds, then lower your arm.",
        ],
        "mistakes": ["Elbow still bent", "Arm hanging too close to the body",
                     "Using the wrong arm", "Raising both arms"],
        "not_graded": "",
    },
    "service_authorization_left": {
        "title": "Authorization to Serve (your LEFT hand)",
        "short": "Authorization - left hand",
        "fivb": "Move the hand to indicate the direction of service.",
        "howto": [
            "Face the camera with your arms relaxed.",
            "Bend your LEFT elbow and raise your hand to about chest height.",
            "Sweep the hand toward the serving team on your left, with a clear beckoning motion.",
            "Keep the other arm down, then lower your hand.",
        ],
        "mistakes": ["Hand not moving (static pose)", "Arm fully straight like Team to Serve",
                     "Using the wrong hand", "Movement too small to see"],
        "not_graded": "",
    },
    "service_authorization_right": {
        "title": "Authorization to Serve (your RIGHT hand)",
        "short": "Authorization - right hand",
        "fivb": "Move the hand to indicate the direction of service.",
        "howto": [
            "Face the camera with your arms relaxed.",
            "Bend your RIGHT elbow and raise your hand to about chest height.",
            "Sweep the hand toward the serving team on your right, with a clear beckoning motion.",
            "Keep the other arm down, then lower your hand.",
        ],
        "mistakes": ["Hand not moving (static pose)", "Arm fully straight like Team to Serve",
                     "Using the wrong hand", "Movement too small to see"],
        "not_graded": "",
    },
    "ball_in": {
        "title": "Ball In",
        "short": "Ball In",
        "fivb": "Point the arm and hand toward the floor (the court, where the ball landed).",
        "howto": [
            "Face the camera with your arms relaxed.",
            "Straighten one arm and point it, with an open hand, down toward the court.",
            "Keep the other arm relaxed.",
            "Hold about 1 to 2 seconds, then lower your arm.",
        ],
        "mistakes": ["Elbow bent", "Arm raised instead of pointing down", "Hand closed into a fist"],
        "not_graded": "Current practice (per our referee validator) points at the court line where the ball landed "
                      "(attack line / center line). This system was trained on the older straight-down form, and "
                      "where you point in depth cannot be measured by one 2D camera, so Ball In is graded leniently.",
    },
    "ball_out": {
        "title": "Ball Out",
        "short": "Ball Out",
        "fivb": "Raise both forearms vertically, hands open, palms toward the body.",
        "howto": [
            "Face the camera with your arms relaxed.",
            "Lift both arms out to the sides so your elbows are about shoulder height and your armpits are open.",
            "Bend both elbows and raise BOTH forearms straight up, hands open, palms toward your body.",
            "Hold about 1 to 2 seconds, then lower your arms.",
        ],
        "mistakes": ["Only one arm raised", "Elbows tucked against the body (armpits closed)",
                     "Forearms leaning instead of vertical", "Arms fully overhead", "Hands closed"],
        "not_graded": "Palm direction (palms toward the body) cannot be measured reliably by one 2D camera; check it with your instructor.",
    },
    "double_contact": {
        "title": "Double Contact",
        "short": "Double Contact",
        "fivb": "Raise two fingers of one hand.",
        "howto": [
            "Face the camera with your arms relaxed.",
            "Raise one hand to about shoulder height or higher, facing the camera.",
            "Show exactly two fingers (index and middle), curl the others.",
            "Hold about 1 to 2 seconds, then lower your hand.",
        ],
        "mistakes": ["Hand too low", "Three fingers or one finger", "Hand turned away from the camera"],
        "not_graded": "",
    },
    "end_of_set": {
        "title": "End of Set",
        "short": "End of Set",
        "fivb": "Cross the forearms in front of the chest, with the hands open.",
        "howto": [
            "Face the camera with your arms relaxed.",
            "Bring both forearms up and cross them in front of your chest.",
            "Keep your hands open.",
            "Hold about 1 to 2 seconds, then lower your arms.",
        ],
        "mistakes": ["Forearms not actually crossing", "Arms too low or too high", "Hands closed"],
        "not_graded": "",
    },
}

KNOWN_WEAK = {
    "ball_in": "Ball In is the weakest signal for this system (lower recognition rate, and it was trained on the "
               "older straight-down form), so a low score can partly reflect the model and not only your form.",
}


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


def _c(cid, label, weight, critical, value, *, ge=None, le=None, arm: Optional[Arm] = None,
       basis="FIVB", tip="", unit="", verifiable=True, strict=False) -> Check:
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
# ----------------------------------------------------------------------------

def _rules_team_to_serve(g: Geo, lv: str, side: str) -> List[Check]:
    A, O = g.arm(side), g.arm(_other(side))
    S = side.upper()
    return [
        _c("arm_extended", f"{S} arm extended (FIVB: extend the arm)", 15, True, A.hi(A.elbow),
           ge=pick(lv, 135, 150, 160), unit=" deg", arm=A,
           tip=f"Straighten your {side} arm; do not leave the elbow bent"),
        _c("points_to_side", f"{S} arm pointing out to the side", 15, True,
           A.hi(A.outward), ge=pick(lv, 0.40, 0.55, 0.70), arm=A,
           tip=f"Swing your {side} arm further out to the side, away from your body"),
        _c("other_arm_down", "Other arm kept down (one-arm signal)", 10, False,
           O.mid(O.from_down), le=pick(lv, 55, 40, 30), unit=" deg", arm=O,
           tip=f"Keep your {_other(side)} arm relaxed at your side"),
    ]


def _rules_authorization(g: Geo, lv: str, side: str) -> List[Check]:
    A, O = g.arm(side), g.arm(_other(side))
    S = side.upper()
    return [
        _c("arm_bent", f"{S} elbow bent for the beckoning motion", 10, True, A.mid(A.elbow),
           le=pick(lv, 150, 145, 140), unit=" deg", arm=A,
           tip=f"Bend your {side} elbow; a straight arm reads as Team to Serve"),
        _c("hand_moves", "Hand moves (FIVB: move the hand to indicate the direction)", 14, True,
           A.motion(), ge=pick(lv, 0.15, 0.25, 0.40), arm=A,
           tip="Make a clearer sweeping motion with your hand; a still pose is not this signal"),
        _c("at_chest", "Hand sweeps at chest level, in front of the body", 8, False, A.mid(A.wr_frac),
           ge=pick(lv, -0.2, -0.05, 0.0), le=pick(lv, 0.9, 0.8, 0.65), arm=A,
           tip="Bring your hand up to chest height (between shoulder and waist) before you sweep",
           basis="TRAINING"),
        _c("toward_side", f"Hand travels toward the {side} side (direction of service)", 5, False,
           A.hi(A.wr_out), ge=pick(lv, -0.4, -0.2, 0.0), arm=A,
           tip=f"Sweep your hand further toward your {side} side"),
        _c("other_arm_down", "Other arm kept down", 3, False, O.mid(O.from_down),
           le=pick(lv, 55, 40, 30), unit=" deg", arm=O,
           tip=f"Keep your {_other(side)} arm relaxed at your side"),
    ]


def _rules_ball_in(g: Geo, lv: str, side=None) -> List[Check]:
    """Ball In is graded LENIENTLY on purpose. The model was trained on the older straight-down form,
    while current practice (per the referee validator) points the arm and hand toward the court line
    where the ball landed. Where the arm points in depth cannot be measured by one frontal 2D camera,
    so no Ball In check is critical and only the parts a 2D camera CAN see are scored."""
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
        _c("arm_extended", "Pointing arm extended (FIVB: point the arm)", 16, False, A.hi(A.elbow),
           ge=pick(lv, 130, 145, 155), unit=" deg", arm=A,
           tip="Straighten the pointing arm; do not leave the elbow bent"),
        _c("arm_lowered", "Arm pointing down toward the court, not raised", 12, False, A.mid(A.from_down),
           le=pick(lv, 90, 75, 60), unit=" deg", arm=A,
           tip="Point the arm lower, toward the court, not out to the side or up"),
        _c("hand_open", "Open hand pointing (FIVB: arm and hand)", 8, False,
           None if n_open is None else float(n_open), ge=pick(lv, 2, 3, 4), verifiable=n_open is not None,
           tip="Keep your fingers open and straight, do not make a fist"),
        _c("other_arm_relaxed", "Other arm relaxed", 4, False, O.mid(O.from_down),
           le=pick(lv, 55, 40, 30), unit=" deg", arm=O,
           tip="Keep your other arm relaxed at your side"),
    ]


def _rules_ball_out(g: Geo, lv: str, side=None) -> List[Check]:
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
        _c("forearms_vertical", "BOTH forearms vertical (FIVB: raise both forearms vertically)", 12, True,
           worst_fore, le=pick(lv, 40, 30, 20), unit=" deg", verifiable=both_vis,
           tip="Raise BOTH forearms straight up; keep them vertical, not leaning"),
        _c("arms_raised", "Arms raised out to the sides, armpits open", 12, True, least_raised,
           ge=pick(lv, 45, 60, 70), unit=" deg", verifiable=both_vis,
           tip="Lift your elbows up and out to about shoulder height so your armpits are open; do not just bend the elbows at your sides"),
        _c("arms_symmetric", "Both arms at the same height", 5, False, asym,
           le=pick(lv, 0.6, 0.4, 0.3), verifiable=both_vis,
           tip="Raise both arms to the same height"),
        _c("hands_open", "Hands open (FIVB: hands open)", 6, False, worst_open,
           ge=pick(lv, 2, 3, 4), verifiable=worst_open is not None, strict=True,
           tip="Open both hands with the fingers extended"),
        _c("elbows_bent", "Elbows bent, forearms up (not arms straight overhead)", 5, True, worst_elbow,
           le=pick(lv, 150, 135, 125), unit=" deg", verifiable=both_vis,
           tip="Keep your elbows bent with the forearms vertical; do not stretch your arms overhead"),
    ]


def _rules_double_contact(g: Geo, lv: str, side=None) -> List[Check]:
    L, R = g.arm("left"), g.arm("right")

    def key(a: Arm):
        v = a.hi(a.elev)
        return v if (math.isfinite(v) and a.vis_frac >= MIN_ARM_VISIBLE_FRAC) else -99.0

    sig_side = "left" if key(L) >= key(R) else "right"
    A, O = g.arm(sig_side), g.arm(_other(sig_side))
    H = g.hand(sig_side)

    if H.ext is None or H.coverage < MIN_HAND_COVERAGE:
        two = Check("two_fingers", "Exactly two fingers shown (FIVB: raise two fingers)", "FIVB", 20, True,
                    "unverified", None, "index + middle only", "Show exactly two fingers to the camera")
    else:
        f = H.ext
        idx_mid = bool(f[1] > 0.5 and f[2] > 0.5)
        curled = int(f[3] <= 0.5) + int(f[4] <= 0.5)
        ok = idx_mid and curled >= pick(lv, 1, 2, 2)
        n_ext = float((f[1:] > 0.5).sum())
        two = Check("two_fingers", "Exactly two fingers shown (FIVB: raise two fingers)", "FIVB", 20, True,
                    "pass" if ok else "fail", n_ext, "index + middle only",
                    "Show exactly two fingers (index and middle) and curl the ring and little fingers")
    return [
        _c("hand_raised", "Hand raised (FIVB: raise two fingers)", 12, True, A.hi(A.elev),
           ge=pick(lv, -0.6, -0.3, 0.0), arm=A,
           tip="Raise your hand higher, to about shoulder height or above"),
        two,
        _c("other_arm_down", "Other arm kept down", 8, False, O.mid(O.from_down),
           le=pick(lv, 55, 40, 30), unit=" deg", arm=O,
           tip="Keep your other arm relaxed at your side"),
    ]


def _rules_end_of_set(g: Geo, lv: str, side=None) -> List[Check]:
    L, R = g.arm("left"), g.arm("right")
    cl, cr = L.hi(L.cross), R.hi(R.cross)
    worst_cross = min(cl, cr) if (math.isfinite(cl) and math.isfinite(cr)) else float("nan")
    fl, fr = L.mid(L.wr_frac), R.mid(R.wr_frac)
    lo_b, hi_b = pick(lv, -0.3, -0.1, 0.0), pick(lv, 1.1, 0.95, 0.8)
    if math.isfinite(fl) and math.isfinite(fr):
        # distance outside the allowed chest band (0 when both wrists are inside it)
        out_of_band = max(0.0, lo_b - min(fl, fr), max(fl, fr) - hi_b)
        chest_value = 0.0 if out_of_band == 0.0 else out_of_band
    else:
        chest_value = float("nan")
    el_l, el_r = L.mid(L.elbow), R.mid(R.elbow)
    worst_elbow = max(el_l, el_r) if (math.isfinite(el_l) and math.isfinite(el_r)) else float("nan")
    nl, nr = g.hand("left").extended_count(), g.hand("right").extended_count()
    seen = [n for n in (nl, nr) if n is not None]
    worst_open = float(min(seen)) if seen else None
    both_vis = L.vis_frac >= MIN_ARM_VISIBLE_FRAC and R.vis_frac >= MIN_ARM_VISIBLE_FRAC
    return [
        _c("forearms_crossed", "Forearms crossed (FIVB: cross the forearms)", 16, True, worst_cross,
           ge=pick(lv, 0.0, 0.10, 0.25), verifiable=both_vis,
           tip="Cross your forearms so each wrist passes the middle of your body"),
        _c("at_chest", "Crossed in front of the chest", 10, False, chest_value, le=0.0, verifiable=both_vis,
           tip="Cross your forearms at chest height, not too low or too high"),
        _c("elbows_bent", "Elbows bent", 6, False, worst_elbow, le=pick(lv, 150, 135, 125), unit=" deg",
           verifiable=both_vis, tip="Bend your elbows to bring the forearms across your chest"),
        _c("hands_open", "Hands open (FIVB: with the hands open)", 8, False, worst_open,
           ge=pick(lv, 2, 3, 4), verifiable=worst_open is not None, strict=True,
           tip="Keep your hands open with the fingers extended"),
    ]


RULES = {
    "team_to_serve_left": lambda g, lv: _rules_team_to_serve(g, lv, "left"),
    "team_to_serve_right": lambda g, lv: _rules_team_to_serve(g, lv, "right"),
    "service_authorization_left": lambda g, lv: _rules_authorization(g, lv, "left"),
    "service_authorization_right": lambda g, lv: _rules_authorization(g, lv, "right"),
    "ball_in": _rules_ball_in,
    "ball_out": _rules_ball_out,
    "double_contact": _rules_double_contact,
    "end_of_set": _rules_end_of_set,
}


def ready_position_check(tail_frames, lv: str, aspect: float) -> Check:
    """TRAINING requirement: after the signal, arms return to the ready position."""
    tip = "Lower both arms to the ready position after the signal"
    need = f"wrists at or below {pick(lv, 0.55, 0.70, 0.80):g} of the shoulder-to-hip distance"
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
              ge=pick(lv, 0.55, 0.70, 0.80), basis="TRAINING", tip=tip)


def grade_form_only(label: str, frames, level: str = "standard", aspect: float = 9.0 / 16.0) -> List[Check]:
    """Runs ONLY the FIVB-derived geometric checks on a window of raw frames.
    Used by grade_attempt() and by tools/grader_sanity_check.py."""
    if label not in RULES:
        return []
    return RULES[label](Geo(frames, aspect), level)


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


def _form_points(checks: List[Check]):
    verifiable = sum(c.weight for c in checks if c.status != "unverified")
    passed = sum(c.weight for c in checks if c.status == "pass")
    if verifiable <= 0:
        return 0.0
    return W_FORM * passed / verifiable


def grade_attempt(target: str, capture_frames, window_records, label_to_idx: Dict[str, int],
                  level: str = "standard", aspect: float = 9.0 / 16.0,
                  step_seconds: float = 0.3) -> AttemptResult:
    """
    target          : gesture label the trainee was asked to perform
    capture_frames  : (N x 122) raw features recorded during the capture period
    window_records  : list of dicts, one per model inference during the capture:
                      {"label": predicted label after tie-breaker, "probs": sigmoid vector,
                       "frames": (24 x 122) raw window that produced it}
    label_to_idx    : real-class label -> index in `probs`
    step_seconds    : average time between two consecutive inferences
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

    other_counts = Counter(lab for lab in labels if lab not in (target, NOTHING_LABEL))
    dom_other, dom_n = (other_counts.most_common(1)[0] if other_counts else (None, 0))

    recognized = n_target >= 1 and best_p >= 0.5

    # --- FIVB form checks on the middle of the recognised hold ---
    checks = grade_form_only(target, window_records[analysis_i]["frames"], level, aspect)
    tail = capture[-6:] if len(capture) >= 6 else capture
    ready = ready_position_check(tail, level, aspect)

    rec_pts = W_RECOGNITION * float(np.clip((best_p - 0.3) / max(1e-6, cfg.rec_threshold - 0.3), 0, 1))
    dist_pts = W_DISTINCT * float(np.clip(margin / cfg.margin_required, 0, 1))
    hold_pts = W_HOLD * float(np.clip(hold_seconds / cfg.hold_seconds, 0, 1))
    form_pts = _form_points(checks)
    ready_pts = W_READY if ready.status == "pass" else 0.0
    if ready.status == "unverified":
        ready_pts = W_READY * 0.5   # do not punish what the camera could not see
    total = int(round(rec_pts + dist_pts + hold_pts + form_pts + ready_pts))

    parts = {"recognition": round(rec_pts, 1), "distinctness": round(dist_pts, 1), "hold": round(hold_pts, 1),
             "form": round(form_pts, 1), "ready_position": round(ready_pts, 1)}

    critical_failed = any((c.critical or c.strict) and c.status == "fail" for c in checks)
    critical_unverified = any(c.critical and c.status == "unverified" for c in checks)
    hold_ok = hold_seconds >= 0.5 * cfg.hold_seconds
    other_dominates = dom_other is not None and dom_n >= 2 and dom_n > n_target

    note = ""
    if not recognized:
        if other_dominates:
            verdict = VERDICT_INCORRECT
            note = f"The system saw {short_label(dom_other)} instead."
        elif best_p < 0.35:
            verdict = VERDICT_INCORRECT
            note = "No clear signal was recognised."
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
                     key=lambda c: (c.status != "fail", not c.critical, -c.weight))
    for c in ordered:
        text = c.feedback()
        if text:
            fb.append(text)
    if hold_seconds < cfg.hold_seconds and recognized:
        fb.append(f"Hold the signal longer (about {cfg.hold_seconds:.1f}s at {cfg.name} level; you held ~{hold_seconds:.1f}s).")
    if ready.status == "fail":
        fb.append(ready.feedback())
    if verdict != VERDICT_CORRECT and target in KNOWN_WEAK:
        fb.append(KNOWN_WEAK[target])
    if verdict == VERDICT_CORRECT and not fb:
        fb.append("Clean signal. Nice work.")

    return AttemptResult(target, level, verdict, total, POINTS[verdict], parts, checks + [ready], fb[:6],
                         recognized, best_p, margin, hold_seconds, dom_other, pose_fraction, note)