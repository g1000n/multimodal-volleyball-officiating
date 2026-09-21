"""
tests/test_gesture_grader.py

Synthetic-skeleton tests for gesture_grader.py. No camera, model, or dataset needed.
Run:  python tests/test_gesture_grader.py      (or: pytest tests/test_gesture_grader.py)

A tiny stick-figure generator builds raw 122-feature frames in the SAME layout as
extract_keypoints.py, so the geometry code is exercised end to end, including
the mirrored-camera case.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gesture_grader as gg  # noqa: E402

ASPECT = 9.0 / 16.0
SW = 0.20                    # shoulder width in frame-width units
SH_Y = 0.20                  # shoulder line in aspect-corrected units
UP, FORE = 0.16, 0.16        # upper arm / forearm length
HIP_Y = SH_Y + 1.3 * SW
LS = (0.5 + SW / 2, SH_Y)    # subject's LEFT shoulder appears on the image RIGHT (un-mirrored camera)
RS = (0.5 - SW / 2, SH_Y)


def out_dir(side):
    return 1.0 if side == "left" else -1.0


def straight_arm(side, deg_from_down):
    """Elbow and wrist of a straight arm angled `deg_from_down` outward from hanging."""
    s = LS if side == "left" else RS
    t = np.radians(deg_from_down)
    d = np.array([out_dir(side) * np.sin(t), np.cos(t)])
    e = np.array(s) + UP * d
    w = np.array(s) + (UP + FORE) * d
    return e, w


def hanging(side):
    return straight_arm(side, 4)


def frame(le, re, lw, rw, left_fingers=None, right_fingers=None, mirror=False):
    """One raw feature frame. Points are in aspect-corrected (x, y) units."""
    f = np.zeros(122)
    pts = [LS, RS, tuple(le), tuple(re), tuple(lw), tuple(rw),
           (LS[0] - 0.02, HIP_Y), (RS[0] + 0.02, HIP_Y)]
    for i, (x, ys) in enumerate(pts):
        if mirror:
            x = 1.0 - x
        f[i * 3] = x
        f[i * 3 + 1] = ys / ASPECT
        f[i * 3 + 2] = 1.0
    if left_fingers is not None:
        f[108] = 1.0
        f[110:115] = left_fingers
    if right_fingers is not None:
        f[109] = 1.0
        f[115:120] = right_fingers
    return f


OPEN = [1, 1, 1, 1, 1]
TWO = [0, 1, 1, 0, 0]
THREE = [0, 1, 1, 1, 0]


def idle(mirror=False):
    le, lw = hanging("left")
    re, rw = hanging("right")
    return frame(le, re, lw, rw, mirror=mirror)


def seq(fn, n, mirror=False):
    return [fn(i, mirror) for i in range(n)]


# ---- pose generators: fn(i, mirror) -> frame ----

def g_tts(side, deg=70, bend=False, fingers=None):
    def fn(i, m):
        e, w = straight_arm(side, deg)
        if bend:  # fold the forearm up so the elbow angle is ~100 deg
            s = np.array(LS if side == "left" else RS)
            e = s + UP * np.array([out_dir(side) * np.sin(np.radians(deg)), np.cos(np.radians(deg))])
            w = e + FORE * np.array([0.0, -1.0])
        oe, ow = hanging("right" if side == "left" else "left")
        args = (e, oe, w, ow) if side == "left" else (oe, e, ow, w)
        kw = {"left_fingers": fingers} if side == "left" else {"right_fingers": fingers}
        return frame(*args, mirror=m, **(kw if fingers is not None else {}))
    return fn


def g_auth(side, moving=True, straight=False, fingers=None, drop=0.0, amp=1.0):
    def fn(i, m):
        s = np.array(LS if side == "left" else RS)
        e = s + np.array([out_dir(side) * 0.10, 0.10])
        sweep = amp * SW * np.sin(i / 3.0) if moving else 0.0
        w = e + np.array([out_dir(side) * 0.10 + out_dir(side) * sweep, -0.06 + drop])
        if straight:
            e, w = straight_arm(side, 60)
        oe, ow = hanging("right" if side == "left" else "left")
        args = (e, oe, w, ow) if side == "left" else (oe, e, ow, w)
        kw = {"left_fingers": fingers} if side == "left" else {"right_fingers": fingers}
        return frame(*args, mirror=m, **(kw if fingers is not None else {}))
    return fn


def g_auth_low(side):
    """Beckoning sweep but the hand stays down at hip level (not at chest)."""
    def fn(i, m):
        s = np.array(LS if side == "left" else RS)
        e = s + np.array([out_dir(side) * 0.10, 0.20])
        w = e + np.array([out_dir(side) * (0.10 + 0.06 * np.sin(i / 3.0)), 0.10])
        oe, ow = hanging("right" if side == "left" else "left")
        args = (e, oe, w, ow) if side == "left" else (oe, e, ow, w)
        return frame(*args, mirror=m)
    return fn


def g_ball_in(deg=15, fingers=OPEN):
    def fn(i, m):
        e, w = straight_arm("right", deg)
        le, lw = hanging("left")
        return frame(le, e, lw, w, right_fingers=fingers, mirror=m)
    return fn


def g_ball_out(one_arm=False, fingers=OPEN):
    def fn(i, m):
        out = []
        for side in ("left", "right"):
            s = np.array(LS if side == "left" else RS)
            e = s + np.array([out_dir(side) * 0.8 * SW, 0.2 * SW])
            w = e + np.array([0.0, -FORE])
            out.append((e, w))
        if one_arm:
            out[1] = hanging("right")
        (le, lw), (re, rw) = out
        return frame(le, re, lw, rw, left_fingers=fingers, right_fingers=fingers, mirror=m)
    return fn


def g_ball_out_tucked(i, m):
    """Forearms vertical but elbows tucked at the sides (armpits closed): the older / incomplete form."""
    out = []
    for side in ("left", "right"):
        s = np.array(LS if side == "left" else RS)
        e = s + np.array([out_dir(side) * 0.3 * SW, 0.7 * SW])
        out.append((e, e + np.array([0.0, -FORE])))
    (le, lw), (re, rw) = out
    return frame(le, re, lw, rw, left_fingers=OPEN, right_fingers=OPEN, mirror=m)


def g_double(fingers=TWO, low=False, side="right"):
    def fn(i, m):
        s = np.array(RS if side == "right" else LS)
        o = -1.0 if side == "right" else 1.0            # outward direction for this hand
        e = s + np.array([o * 0.05, 0.10 if not low else 0.12])
        w = e + np.array([o * 0.02, -0.16 if not low else 0.02])
        oe, ow = hanging("left" if side == "right" else "right")
        if side == "right":
            return frame(oe, e, ow, w, right_fingers=fingers, mirror=m)
        return frame(e, oe, w, ow, left_fingers=fingers, mirror=m)
    return fn


def g_end(cross=True, low=0.0):
    def fn(i, m):
        mid = 0.5
        dx = 0.06 if cross else -0.10
        le = (LS[0] - 0.02, SH_Y + 0.14 + low)
        re = (RS[0] + 0.02, SH_Y + 0.14 + low)
        lw = (mid - dx, SH_Y + 0.10 + low)
        rw = (mid + dx, SH_Y + 0.10 + low)
        return frame(le, re, lw, rw, left_fingers=OPEN, right_fingers=OPEN, mirror=m)
    return fn


# ---- helpers to run a whole attempt ----

def build_attempt(target_fn, target, mirror=False, labels_fn=None, probs_target=0.95, other_label=None):
    label_to_idx = {l: i for i, l in enumerate(sorted(gg.RULES))}
    frames = seq(lambda i, m: idle(m), 12, mirror) + seq(target_fn, 34, mirror) + seq(lambda i, m: idle(m), 10, mirror)
    frames = np.array(frames)
    records = []
    for end in range(24, len(frames) + 1, 3):
        win = frames[end - 24:end]
        gesture_share = np.mean([12 <= k < 46 for k in range(end - 24, end)])
        probs = np.full(len(label_to_idx), 0.02)
        if gesture_share >= 0.5:
            lab = other_label or target
            probs[label_to_idx[lab]] = probs_target
        else:
            lab = gg.NOTHING_LABEL
        records.append({"label": lab, "probs": probs, "frames": win})
    return frames, records, label_to_idx


def run(target_fn, target, level="standard", **kw):
    mirror = kw.pop("mirror", False)
    frames, records, l2i = build_attempt(target_fn, target, mirror=mirror, **kw)
    return gg.grade_attempt(target, frames, records, l2i, level=level, aspect=ASPECT, step_seconds=0.3)


def build_sequence(fns_labels, gap=4, lead=8, tail=10, share=0.6):
    """Back-to-back gestures in ONE capture. fns_labels = [(pose_fn, label), ...]."""
    label_to_idx = {l: i for i, l in enumerate(sorted(gg.RULES))}
    frames, owner = [], []
    for _ in range(lead):
        frames.append(idle())
        owner.append(None)
    for k, (fn, label) in enumerate(fns_labels):
        for i in range(24):
            frames.append(fn(i, False))
            owner.append(label)
        if k < len(fns_labels) - 1:
            for _ in range(gap):
                frames.append(idle())
                owner.append(None)
    for _ in range(tail):
        frames.append(idle())
        owner.append(None)
    frames = np.array(frames)
    records = []
    for end in range(24, len(frames) + 1, 3):
        window_owner = owner[end - 24:end]
        probs = np.full(len(label_to_idx), 0.02)
        lab = gg.NOTHING_LABEL
        for _, label in fns_labels:
            if window_owner.count(label) / 24.0 >= share:
                lab = label
                probs[label_to_idx[label]] = 0.95
        records.append({"label": lab, "probs": probs, "frames": frames[end - 24:end], "end": end})
    return frames, records, label_to_idx


def seq_run(fns_labels, targets, level="standard", contexts=None, **kw):
    frames, records, l2i = build_sequence(fns_labels, **kw)
    return gg.grade_sequence(targets, frames, records, l2i, level=level, aspect=ASPECT, step_seconds=0.3,
                             contexts=contexts)


def statuses(res):
    return {c.id: c.status for c in res.checks}


# ---------------------------------------------------------------- tests

def test_team_to_serve_correct_all_levels():
    for lv in gg.LEVELS:
        for side in ("left", "right"):
            r = run(g_tts(side), f"team_to_serve_{side}", lv)
            assert r.verdict == gg.VERDICT_CORRECT, (lv, side, r.verdict, r.score, r.feedback)


def test_team_to_serve_bent_elbow_is_not_correct():
    r = run(g_tts("left", bend=True), "team_to_serve_left")
    assert statuses(r)["arm_extended"] == "fail"
    assert r.verdict != gg.VERDICT_CORRECT


def test_team_to_serve_wrong_arm():
    def wrong(i, m):
        e, w = straight_arm("right", 50)
        le, lw = hanging("left")
        return frame(le, e, lw, w, mirror=m)
    r = run(wrong, "team_to_serve_left")
    st = statuses(r)
    assert st["points_to_side"] == "fail" and st["other_arm_down"] == "fail"
    assert r.verdict != gg.VERDICT_CORRECT


def test_authorization_moving_vs_static():
    ok = run(g_auth("left"), "service_authorization_left")
    assert statuses(ok)["hand_moves"] == "pass" and ok.verdict == gg.VERDICT_CORRECT, (ok.verdict, ok.feedback)
    static = run(g_auth("left", moving=False), "service_authorization_left")
    assert statuses(static)["hand_moves"] == "fail"
    assert static.verdict != gg.VERDICT_CORRECT
    low = run(g_auth_low("left"), "service_authorization_left")
    assert statuses(low)["at_chest"] == "fail" and statuses(ok)["at_chest"] == "pass", (statuses(low), statuses(ok))
    straight = run(g_auth("left", straight=True), "service_authorization_left")
    assert statuses(straight)["arm_bent"] == "fail"


def test_ball_in():
    ok = run(g_ball_in(), "ball_in")
    assert ok.verdict == gg.VERDICT_CORRECT, (ok.verdict, ok.feedback)
    high = run(g_ball_in(deg=100), "ball_in")
    assert statuses(high)["arm_lowered"] == "fail"
    assert not any(c.critical for c in ok.checks)   # Ball In has no critical check by design
    fist = run(g_ball_in(fingers=[0, 0, 0, 0, 0]), "ball_in")
    assert statuses(fist)["hand_open"] == "fail"


def test_ball_out():
    ok = run(g_ball_out(), "ball_out")
    assert ok.verdict == gg.VERDICT_CORRECT, (ok.verdict, ok.feedback)
    def overhead(i, m):
        e1, w1 = straight_arm("left", 175)
        e2, w2 = straight_arm("right", 175)
        return frame(e1, e2, w1, w2, left_fingers=OPEN, right_fingers=OPEN, mirror=m)
    over = run(overhead, "ball_out")
    assert statuses(over)["elbows_bent"] == "fail" and over.verdict != gg.VERDICT_CORRECT, over.feedback
    fists = run(g_ball_out(fingers=[0, 0, 0, 0, 0]), "ball_out")
    assert statuses(fists)["hands_open"] == "fail" and fists.score < ok.score, fists.feedback
    unseen = run(g_ball_out(fingers=None), "ball_out")     # hands not detected: NOT punished
    assert statuses(unseen)["hands_open"] == "unverified" and unseen.verdict == gg.VERDICT_CORRECT, unseen.feedback
    tucked = run(g_ball_out_tucked, "ball_out")
    # measured on the team's own attempts: elbows are usually kept near the body, so this only costs points by default
    assert statuses(tucked)["arms_raised"] == "fail" and tucked.score < ok.score, tucked.feedback
    assert tucked.verdict == gg.VERDICT_CORRECT and gg.ARMPITS_REQUIRED is False
    gg.ARMPITS_REQUIRED = True
    try:
        assert run(g_ball_out_tucked, "ball_out").verdict != gg.VERDICT_CORRECT
    finally:
        gg.ARMPITS_REQUIRED = False
    one = run(g_ball_out(one_arm=True), "ball_out")
    assert statuses(one)["forearms_vertical"] == "fail" and one.verdict != gg.VERDICT_CORRECT


def test_double_contact():
    ok = run(g_double(), "double_contact")
    assert ok.verdict == gg.VERDICT_CORRECT, (ok.verdict, ok.feedback)
    three = run(g_double(fingers=THREE), "double_contact")
    assert statuses(three)["two_fingers"] == "fail" and three.verdict != gg.VERDICT_CORRECT
    low = run(g_double(low=True), "double_contact")
    assert statuses(low)["hand_raised"] == "fail"
    nohand = run(g_double(fingers=None), "double_contact")
    assert statuses(nohand)["two_fingers"] == "unverified" and nohand.verdict != gg.VERDICT_CORRECT


def test_end_of_set():
    ok = run(g_end(), "end_of_set")
    assert ok.verdict == gg.VERDICT_CORRECT, (ok.verdict, ok.feedback)
    bad = run(g_end(cross=False), "end_of_set")
    assert statuses(bad)["forearms_crossed"] == "fail" and bad.verdict != gg.VERDICT_CORRECT


def test_mirrored_camera_gives_identical_results():
    for target, fn in [("team_to_serve_left", g_tts("left")), ("ball_out", g_ball_out()),
                       ("end_of_set", g_end()), ("service_authorization_right", g_auth("right"))]:
        a = run(fn, target)
        b = run(fn, target, mirror=True)
        assert a.verdict == b.verdict and a.score == b.score, (target, a.score, b.score, b.feedback)


def test_wrong_gesture_recognised_is_incorrect():
    r = run(g_tts("left"), "ball_out", other_label="team_to_serve_left")
    assert r.verdict == gg.VERDICT_INCORRECT and r.confused_with == "team_to_serve_left"


def test_low_confidence_not_correct():
    r = run(g_tts("left"), "team_to_serve_left", probs_target=0.40)
    assert r.verdict != gg.VERDICT_CORRECT


def test_no_reading_when_body_not_seen():
    l2i = {l: i for i, l in enumerate(sorted(gg.RULES))}
    r = gg.grade_attempt("ball_out", np.zeros((40, 122)), [], l2i)
    assert r.verdict == gg.VERDICT_NO_READING and r.points == 0


def test_difficulty_is_monotonic():
    # a sloppy but recognisable attempt: shallow arm angle
    scores = {lv: run(g_tts("left", deg=30), "team_to_serve_left", lv) for lv in gg.LEVELS}
    order = {gg.VERDICT_INCORRECT: 0, gg.VERDICT_ALMOST: 1, gg.VERDICT_CORRECT: 2}
    assert order[scores["beginner"].verdict] >= order[scores["standard"].verdict] >= order[scores["referee"].verdict]


def test_every_rule_set_sums_to_form_weight():
    for label, rule in gg.RULES.items():
        checks = rule(gg.Geo(np.array([idle()] * 30), ASPECT), "standard")
        assert sum(c.weight for c in checks) == gg.W_FORM, (label, sum(c.weight for c in checks))


def test_team_to_serve_elbow_down_forearm_up_is_wrong():
    def folded(i, m):      # elbow at the side, only the forearm up
        s = np.array(LS)
        e = s + UP * np.array([0.05, 1.0])
        w = e + FORE * np.array([0.0, -1.0])
        oe, ow = hanging("right")
        return frame(e, oe, w, ow, mirror=m)
    r = run(folded, "team_to_serve_left")
    assert statuses(r)["arm_extended"] == "fail" and r.verdict != gg.VERDICT_CORRECT, r.feedback


def test_team_to_serve_contract_then_point_is_wrong():
    """Fold the arm in first, then point: the swing must be straight from neutral to the point."""
    label_to_idx = {l: i for i, l in enumerate(sorted(gg.RULES))}
    tts = g_tts("left")
    fold = g_tts("left", bend=True)
    frames = ([idle() for _ in range(8)] + [fold(i, False) for i in range(8)] + [tts(i, False) for i in range(30)]
              + [idle() for _ in range(8)])
    frames = np.array(frames)
    recs = []
    for end in range(24, len(frames) + 1, 3):
        good = sum(1 for k in range(end - 24, end) if 16 <= k < 46) / 24.0
        probs = np.full(len(label_to_idx), 0.02)
        lab = gg.NOTHING_LABEL
        if good >= 0.5:
            lab = "team_to_serve_left"
            probs[label_to_idx[lab]] = 0.95
        recs.append({"label": lab, "probs": probs, "frames": frames[end - 24:end], "end": end})
    bad = gg.grade_attempt("team_to_serve_left", frames, recs, label_to_idx, level="standard", aspect=ASPECT)
    assert statuses(bad)["no_contraction"] == "fail" and bad.verdict != gg.VERDICT_CORRECT, bad.feedback
    good = run(g_tts("left"), "team_to_serve_left")
    assert statuses(good)["no_contraction"] == "pass" and good.verdict == gg.VERDICT_CORRECT, good.feedback


def test_open_hand_is_scored_but_does_not_change_the_verdict_by_default():
    OPEN5, FIST = [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]
    assert gg.OPEN_HAND_STRICT is False
    for label, mk, side in (("team_to_serve_left", g_tts, "left"), ("service_authorization_left", g_auth, "left")):
        ok = run(mk("left", fingers=OPEN5), label)
        fist = run(mk("left", fingers=FIST), label)
        assert statuses(ok)["hands_open"] == "pass" and ok.verdict == gg.VERDICT_CORRECT, ok.feedback
        assert statuses(fist)["hands_open"] == "fail", statuses(fist)
        assert fist.score < ok.score                                  # points are lost ...
        assert fist.verdict == gg.VERDICT_CORRECT                    # ... but the gesture still counts
        assert any("open" in f.lower() for f in fist.feedback)       # and the trainee is told to keep the hand open
    fists = run(g_ball_out(fingers=FIST), "ball_out")
    assert statuses(fists)["hands_open"] == "fail" and fists.verdict == gg.VERDICT_CORRECT
    gg.OPEN_HAND_STRICT = True                                        # the strict option still works
    try:
        assert run(g_tts("left", fingers=FIST), "team_to_serve_left").verdict != gg.VERDICT_CORRECT
        assert run(g_ball_out(fingers=FIST), "ball_out").verdict != gg.VERDICT_CORRECT
        assert run(g_end(), "end_of_set").verdict == gg.VERDICT_CORRECT
    finally:
        gg.OPEN_HAND_STRICT = False


def test_disclaimer_exists():
    assert "mistakes" in gg.DISCLAIMER


def test_authorization_height_band():
    ok = run(g_auth("left"), "service_authorization_left")
    assert statuses(ok)["at_chest"] == "pass"
    high = run(g_auth("left", drop=-0.16), "service_authorization_left")      # hand above the shoulders
    assert statuses(high)["at_chest"] == "fail" and high.verdict != gg.VERDICT_CORRECT, high.feedback
    low = run(g_auth_low("left"), "service_authorization_left")               # belly
    assert statuses(low)["at_chest"] == "fail" and low.verdict != gg.VERDICT_CORRECT


def test_end_of_set_hugging_low_is_wrong():
    ok = run(g_end(), "end_of_set")
    assert statuses(ok)["at_chest"] == "pass"
    low = run(g_end(low=0.16), "end_of_set")
    assert statuses(low)["at_chest"] == "fail" and low.score < ok.score          # points are lost ...
    assert low.verdict == gg.VERDICT_CORRECT                                     # ... but Standard still counts it
    low_ref = run(g_end(low=0.16), "end_of_set", "referee")
    assert low_ref.verdict != gg.VERDICT_CORRECT                                 # the strictest level caps it


def test_end_of_set_overlapping_wrists_count_as_crossed():
    def overlap(i, m):        # forearms lying on top of each other: the tracker puts both wrists near the middle
        mid = 0.5
        le = (LS[0] - 0.02, SH_Y + 0.14)
        re = (RS[0] + 0.02, SH_Y + 0.14)
        return frame(le, re, (mid + 0.015, SH_Y + 0.10), (mid - 0.015, SH_Y + 0.10),
                     left_fingers=OPEN, right_fingers=OPEN, mirror=m)
    for lv in gg.LEVELS:
        r = run(overlap, "end_of_set", lv)
        assert statuses(r)["forearms_crossed"] == "pass" and r.verdict == gg.VERDICT_CORRECT, (lv, r.feedback)
    apart = run(g_end(cross=False), "end_of_set")          # hands side by side, not crossed
    assert statuses(apart)["forearms_crossed"] == "fail" and apart.verdict != gg.VERDICT_CORRECT


def test_close_attempts_still_pass_and_100_is_not_needed():
    for lv in gg.LEVELS:
        cfg = gg.LEVEL_CONFIG[lv]
        assert cfg.correct_cut <= 90 and cfg.almost_cut < cfg.correct_cut       # 100 is never required
    assert gg.LEVEL_CONFIG["standard"].correct_cut <= 75
    r = run(g_tts("left", deg=70), "team_to_serve_left")
    assert r.score < 100 or r.verdict == gg.VERDICT_CORRECT


def test_team_to_serve_slightly_bent_arm_is_accepted():
    def slightly_bent(i, m):        # about 150 degrees at the elbow: the camera cannot see a perfectly straight arm
        s = np.array(LS)
        d1 = np.array([np.sin(np.radians(50)), np.cos(np.radians(50))])
        e = s + UP * d1
        d2 = np.array([np.sin(np.radians(50 + 30)), np.cos(np.radians(50 + 30))])
        w = e + FORE * d2
        oe, ow = hanging("right")
        return frame(e, oe, w, ow, mirror=m)
    r = run(slightly_bent, "team_to_serve_left")
    assert statuses(r)["arm_extended"] == "pass" and r.verdict == gg.VERDICT_CORRECT, r.feedback
    clearly_bent = run(g_tts("left", bend=True), "team_to_serve_left")
    assert statuses(clearly_bent)["arm_extended"] == "fail"


def test_service_authorization_transition_through_team_to_serve_is_ignored():
    """The sweep passes through Team to Serve like poses; that must not become 'looks like Team to Serve'."""
    label_to_idx = {l: i for i, l in enumerate(sorted(gg.RULES))}
    frames = np.array([idle()] * 12 + [g_auth("left")(i, False) for i in range(34)] + [idle()] * 10)
    recs = []
    for end in range(24, len(frames) + 1, 3):
        share = np.mean([12 <= k < 46 for k in range(end - 24, end)])
        probs = np.full(len(label_to_idx), 0.02)
        lab = gg.NOTHING_LABEL
        if share >= 0.5:
            # the model calls the first windows Team to Serve and the later ones Authorization
            lab = "team_to_serve_left" if end < 36 else "service_authorization_left"
            probs[label_to_idx[lab]] = 0.9
            probs[label_to_idx["service_authorization_left"]] = max(probs[label_to_idx["service_authorization_left"]], 0.8)
        recs.append({"label": lab, "probs": probs, "frames": frames[end - 24:end], "end": end})
    r = gg.grade_attempt("service_authorization_left", frames, recs, label_to_idx, aspect=ASPECT)
    assert r.confused_with is None and not any("looked like" in f or "saw" in f for f in r.feedback), r.feedback
    # the same confusion is still reported for a signal that is not allowed to pass through it
    r2 = gg.grade_attempt("ball_out", frames, recs, label_to_idx, aspect=ASPECT)
    assert r2.verdict != gg.VERDICT_CORRECT


def test_double_contact_hand_side_context():
    frames, records, l2i = build_attempt(g_double(), "double_contact")        # RIGHT hand raised
    right = gg.grade_attempt("double_contact", frames, records, l2i, aspect=ASPECT, context={"side": "right"})
    left = gg.grade_attempt("double_contact", frames, records, l2i, aspect=ASPECT, context={"side": "left"})
    assert right.verdict == gg.VERDICT_CORRECT, right.feedback
    assert {c.id: c.status for c in left.checks}["hand_side"] == "fail" and left.verdict != gg.VERDICT_CORRECT
    for ctx in (None, {"side": "left"}):
        checks = gg.RULES["double_contact"](gg.Geo(np.array([idle()] * 30), ASPECT), "standard", None, ctx)
        assert sum(c.weight for c in checks) == gg.W_FORM


def test_sequence_team_to_serve_then_ball_out():
    good = seq_run([(g_tts("right"), "team_to_serve_right"), (g_ball_out(), "ball_out")],
                   ["team_to_serve_right", "ball_out"])
    assert [r.verdict for r in good] == [gg.VERDICT_CORRECT] * 2, [(r.verdict, r.feedback) for r in good]
    # only the LAST signal is checked for returning to the ready position
    assert "ready_position" not in {c.id for c in good[0].checks} and "ready_position" in {c.id for c in good[1].checks}
    wrong_order = seq_run([(g_ball_out(), "ball_out"), (g_tts("right"), "team_to_serve_right")],
                          ["team_to_serve_right", "ball_out"])
    assert all(r.verdict != gg.VERDICT_CORRECT for r in wrong_order)
    assert any("order" in f.lower() for f in wrong_order[0].feedback), wrong_order[0].feedback
    missing = seq_run([(g_tts("right"), "team_to_serve_right")], ["team_to_serve_right", "ball_out"])
    assert missing[1].verdict == gg.VERDICT_INCORRECT and missing[1].points == 0


def test_authorization_hold_is_scaled():
    assert gg.HOLD_SCALE["service_authorization_left"] < 1.0
    r = run(g_auth("left"), "service_authorization_left")
    assert r.hold_required < gg.LEVEL_CONFIG["standard"].hold_seconds


def test_every_graded_check_is_taught_in_the_instruction_card():
    for label in gg.RULES:
        ctx = {"side": "left"} if label == "double_contact" else None
        ids = {c["id"] for c in gg.graded_summary(label, ctx)}
        taught = gg.SIGNALS[label]["taught"]
        n_steps = len(gg.SIGNALS[label]["howto"])
        missing = ids - set(taught)
        stale = set(taught) - ids
        assert not missing, (label, "graded but not taught:", missing)
        assert not stale, (label, "taught but not graded:", stale)
        assert all(1 <= step <= n_steps for step in taught.values()), (label, taught, n_steps)


def test_neighbouring_signal_is_ignored_only_while_the_target_is_recognised():
    label_to_idx = {l: i for i, l in enumerate(sorted(gg.RULES))}
    frames = np.array([idle()] * 12 + [g_ball_out()(i, False) for i in range(34)] + [idle()] * 10)

    def records(target_share_label, other_label):
        recs = []
        for end in range(24, len(frames) + 1, 3):
            share = np.mean([12 <= k < 46 for k in range(end - 24, end)])
            probs = np.full(len(label_to_idx), 0.02)
            lab = gg.NOTHING_LABEL
            if share >= 0.5:
                lab = target_share_label if end < 33 else other_label
                probs[label_to_idx[lab]] = 0.95
            recs.append({"label": lab, "probs": probs, "frames": frames[end - 24:end], "end": end})
        return recs
    # Ball Out passes through End of Set on the way: mostly End of Set windows but the target IS seen -> no complaint
    only_first_ball_out = records("ball_out", "end_of_set")
    r = gg.grade_attempt("ball_out", frames, only_first_ball_out, label_to_idx, aspect=ASPECT)
    assert r.confused_with is None and not any("looked like" in f for f in r.feedback), r.feedback
    # the trainee did End of Set instead of Ball Out: the target is never recognised -> the system says what it saw
    wrong = gg.grade_attempt("ball_out", frames, records("end_of_set", "end_of_set"), label_to_idx, aspect=ASPECT)
    assert wrong.verdict == gg.VERDICT_INCORRECT and wrong.confused_with == "end_of_set", (wrong.verdict, wrong.feedback)
    assert any("End of Set" in f for f in wrong.feedback), wrong.feedback


def test_ball_in_short_recognition_is_accepted():
    assert gg.HOLD_SCALE["ball_in"] <= 0.5
    r = run(g_ball_in(deg=35), "ball_in")            # the measured natural angle is 33 to 46 degrees from vertical
    assert statuses(r)["arm_lowered"] == "pass"
    for lv in gg.LEVELS:
        assert gg.RULES["ball_in"](gg.Geo(np.array([idle()] * 30), ASPECT), lv)             # builds for every level


def test_measured_team_to_serve_values_separate_good_from_folded():
    # numbers measured on real attempts: good swings 138 to 162 deg, deliberately folded ones 25 to 107 deg
    for lv, thr in (("beginner", 100), ("standard", 120), ("referee", 132)):
        checks = gg.RULES["team_to_serve_left"](gg.Geo(np.array([idle()] * 30), ASPECT), lv)
        need = {c.id: c.need for c in checks}["no_contraction"]
        assert f">= {thr}" in need, (lv, need)
    for lv, thr in (("beginner", 130), ("standard", 145), ("referee", 155)):
        need = {c.id: c.need for c in gg.RULES["team_to_serve_left"](gg.Geo(np.array([idle()] * 30), ASPECT), lv)}["arm_extended"]
        assert f">= {thr}" in need, (lv, need)


def test_scoring_note_says_a_perfect_score_is_not_needed():
    for lv in gg.LEVELS:
        note = gg.scoring_note(lv)
        assert "100" in note and str(gg.LEVEL_CONFIG[lv].correct_cut) in note


def test_sloppy_service_authorization_is_not_correct_at_standard_and_referee():
    for lv in ("standard", "referee"):
        good = run(g_auth("left"), "service_authorization_left", lv)
        assert good.verdict == gg.VERDICT_CORRECT, (lv, good.feedback)
        tiny = run(g_auth("left", amp=0.25), "service_authorization_left", lv)          # a small wiggle
        assert statuses(tiny)["hand_moves"] == "fail" and tiny.verdict != gg.VERDICT_CORRECT, (lv, tiny.feedback)
        straight = run(g_auth("left", straight=True), "service_authorization_left", lv)  # arm out, not a sweep
        assert straight.verdict != gg.VERDICT_CORRECT
    # a nearly straight arm that still sweeps: only the bent-elbow check catches it (Important at Standard and Referee)
    def stiff(i, m):
        s = np.array(LS)
        e = s + np.array([0.13, 0.09])
        sweep = 1.0 * SW * np.sin(i / 3.0)
        w = e + np.array([0.14 + sweep, -0.02])           # forearm almost in line with the upper arm
        oe, ow = hanging("right")
        return frame(e, oe, w, ow, mirror=m)
    r = run(stiff, "service_authorization_left", "referee")
    assert statuses(r)["arm_bent"] == "fail" and r.verdict != gg.VERDICT_CORRECT, (r.feedback, statuses(r))


def test_thresholds_can_be_overridden_from_the_config():
    tiny = g_auth("left", amp=0.25)
    assert run(tiny, "service_authorization_left").verdict != gg.VERDICT_CORRECT
    old = dict(gg.OVERRIDES)
    gg.OVERRIDES.update({"service_authorization.hand_moves": {"ge": (0.05, 0.05, 0.05)}})       # both sides
    try:
        assert statuses(run(tiny, "service_authorization_left"))["hand_moves"] == "pass"
        assert statuses(run(tiny, "service_authorization_right"))["hand_moves"] in ("pass", "fail")
        gg.OVERRIDES.clear()
        gg.OVERRIDES.update({"service_authorization_right.hand_moves": {"ge": 0.05}})            # right side only
        assert statuses(run(tiny, "service_authorization_left"))["hand_moves"] == "fail"
        gg.OVERRIDES.clear()
        gg.OVERRIDES.update({"service_authorization.at_chest": {"effect": "scored"}})            # not strict any more
        chk = {c.id: c for c in gg.grade_form_only("service_authorization_left",
                                                   np.array([g_auth_low("left")(i, False) for i in range(30)]),
                                                   "standard", ASPECT)}
        assert chk["at_chest"].status == "fail" and chk["at_chest"].strict is False and chk["at_chest"].critical is False
        summ = {c["id"]: c["effect"] for c in gg.graded_summary("service_authorization_left")}
        assert summ["at_chest"] == "Scored"
        gg.OVERRIDES.clear()
        assert {c["id"]: c["effect"] for c in gg.graded_summary("service_authorization_left")}["at_chest"] == "Important"
    finally:
        gg.OVERRIDES.clear()
        gg.OVERRIDES.update(old)
    assert gg._CTX["label"] is None                                                             # context is cleaned up


def test_no_signal_says_so_plainly():
    label_to_idx = {l: i for i, l in enumerate(sorted(gg.RULES))}

    def records(frames):
        return [{"label": gg.NOTHING_LABEL, "probs": np.full(len(label_to_idx), 0.02), "frames": frames[i:i + 24],
                 "end": i + 24} for i in range(0, len(frames) - 23, 3)]
    down = np.array([idle()] * 50)                      # standing still, arms never raised
    r = gg.grade_attempt("ball_out", down, records(down), label_to_idx, aspect=ASPECT)
    assert r.verdict == gg.VERDICT_INCORRECT and "arms stayed down" in r.note and r.points == 0, (r.verdict, r.note)
    raised = np.array([g_ball_out()(i, False) for i in range(50)])   # arms up, but the model saw nothing
    r2 = gg.grade_attempt("ball_out", raised, records(raised), label_to_idx, aspect=ASPECT)
    assert r2.verdict == gg.VERDICT_INCORRECT and "arms stayed down" not in r2.note and "No clear signal" in r2.note, r2.note
    r3 = gg.grade_attempt("ball_out", np.zeros((40, 122)), records(down), label_to_idx, aspect=ASPECT)   # not in frame
    assert r3.verdict == gg.VERDICT_NO_READING and "could not see you" in r3.note


def test_learn_notes_and_disclaimer():
    assert all(gg.SIGNALS[l].get("compare", "").startswith("Do not confuse") for l in gg.SIGNALS)
    assert "coach" in gg.DISCLAIMER and "mistakes" in gg.DISCLAIMER


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL  {name}: {e}")
    print("\nAll good." if not failed else f"\n{failed} test(s) failed.")
    sys.exit(1 if failed else 0)