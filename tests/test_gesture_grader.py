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

def g_tts(side, deg=50, bend=False):
    def fn(i, m):
        e, w = straight_arm(side, deg)
        if bend:  # fold the forearm up so the elbow angle is ~100 deg
            s = np.array(LS if side == "left" else RS)
            e = s + UP * np.array([out_dir(side) * np.sin(np.radians(deg)), np.cos(np.radians(deg))])
            w = e + FORE * np.array([0.0, -1.0])
        oe, ow = hanging("right" if side == "left" else "left")
        args = (e, oe, w, ow) if side == "left" else (oe, e, ow, w)
        return frame(*args, mirror=m)
    return fn


def g_auth(side, moving=True, straight=False):
    def fn(i, m):
        s = np.array(LS if side == "left" else RS)
        e = s + np.array([out_dir(side) * 0.10, 0.10])
        sweep = 0.30 * SW * np.sin(i / 3.0) if moving else 0.0
        w = e + np.array([out_dir(side) * 0.10 + out_dir(side) * sweep, -0.06])
        if straight:
            e, w = straight_arm(side, 60)
        oe, ow = hanging("right" if side == "left" else "left")
        args = (e, oe, w, ow) if side == "left" else (oe, e, ow, w)
        return frame(*args, mirror=m)
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


def g_double(fingers=TWO, low=False):
    def fn(i, m):
        s = np.array(RS)
        e = s + np.array([-0.05, 0.10 if not low else 0.12])
        w = e + np.array([-0.02, -0.16 if not low else 0.02])
        le, lw = hanging("left")
        return frame(le, e, lw, w, right_fingers=fingers, mirror=m)
    return fn


def g_end(cross=True):
    def fn(i, m):
        mid = 0.5
        dx = 0.06 if cross else -0.10
        le = (LS[0] - 0.02, SH_Y + 0.14)
        re = (RS[0] + 0.02, SH_Y + 0.14)
        lw = (mid - dx, SH_Y + 0.10)
        rw = (mid + dx, SH_Y + 0.10)
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
    assert statuses(fists)["hands_open"] == "fail" and fists.verdict != gg.VERDICT_CORRECT, fists.feedback
    unseen = run(g_ball_out(fingers=None), "ball_out")     # hands not detected: NOT punished
    assert statuses(unseen)["hands_open"] == "unverified" and unseen.verdict == gg.VERDICT_CORRECT, unseen.feedback
    tucked = run(g_ball_out_tucked, "ball_out")
    assert statuses(tucked)["arms_raised"] == "fail" and tucked.verdict != gg.VERDICT_CORRECT, tucked.feedback
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