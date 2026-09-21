"""
tests/test_progress.py

Progress summary from saved sessions (progress.py). Builds fake sessions in a temporary folder.
Run:  python tests/test_progress.py
"""

import csv
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import progress  # noqa: E402

COLS = ["ph_time", "kind", "target", "level", "intent", "note", "verdict", "score", "points", "best_prob", "margin",
        "hold_s", "confused_with", "failed_checks", "check_values", "feedback"]


def write_session(root, name, rows):
    d = os.path.join(root, name)
    os.makedirs(d)
    with open(os.path.join(d, "attempts.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            base = {c: "" for c in COLS}
            base.update({"kind": "gesture", "level": "standard", "intent": "normal", "points": 0})
            base.update(r)
            w.writerow(base)


def make():
    tmp = tempfile.mkdtemp()
    root = os.path.join(tmp, "ana")
    os.makedirs(root)
    write_session(root, "20260920_100000_drill", [
        {"target": "double_contact", "verdict": "INCORRECT", "score": 30, "failed_checks": "two_fingers"},
        {"target": "double_contact", "verdict": "ALMOST", "score": 60, "failed_checks": "two_fingers"},
        {"target": "double_contact", "verdict": "INCORRECT", "score": 40, "failed_checks": "two_fingers;hand_raised"},
        {"target": "ball_out", "verdict": "CORRECT", "score": 90},
    ])
    write_session(root, "20260921_100000_challenge", [
        {"target": "double_contact", "verdict": "CORRECT", "score": 85},
        {"target": "ball_out", "verdict": "CORRECT", "score": 95},
        {"target": "end_of_set", "verdict": "CORRECT", "score": 88},
        {"kind": "whistle", "target": "whistle", "verdict": "CORRECT", "score": 100},
    ])
    # a labeled test session must not count as training
    write_session(root, "20260921_120000_drill_wrong", [
        {"target": "ball_out", "verdict": "INCORRECT", "score": 10, "intent": "wrong"},
    ])
    return tmp, root


def test_totals_and_exclusions():
    tmp, root = make()
    try:
        d = progress.build_progress(root)
        assert len(d["sessions"]) == 2                       # the test session is left out
        assert d["total_attempts"] == 7 and d["total_correct"] == 4, d
        assert abs(d["accuracy"] - 100.0 * 4 / 7) < 1e-6
        assert d["per_signal"]["ball_out"]["attempts"] == 2 and d["per_signal"]["ball_out"]["accuracy"] == 100.0
        assert d["whistle"]["attempts"] == 1 and d["whistle"]["pct"] == 100.0
    finally:
        shutil.rmtree(tmp)


def test_weakest_signal_and_issues():
    tmp, root = make()
    try:
        d = progress.build_progress(root)
        assert d["weakest"][0] == "double_contact", d["weakest"]
        top = d["per_signal"]["double_contact"]["issues"][0]
        assert "two fingers" in top[0].lower() and top[1] == 3, top
        lines = progress.focus_lines(d)
        assert any("Double Contact" in l for l in lines), lines
    finally:
        shutil.rmtree(tmp)


def test_trend_and_report():
    tmp, root = make()
    try:
        d = progress.build_progress(root)
        assert "up" in d["trend_text"], d["trend_text"]           # first attempts were worse than the latest
        html = progress.progress_html(d, "Ana <test>")
        assert "Ana &lt;test&gt;" in html and "Double Contact" in html and "mistakes" in html
    finally:
        shutil.rmtree(tmp)


def test_empty_folder_is_fine():
    tmp = tempfile.mkdtemp()
    try:
        d = progress.build_progress(os.path.join(tmp, "nobody"))
        assert d["sessions"] == [] and d["total_attempts"] == 0 and d["weakest"] == []
        assert "No sessions yet" in progress.progress_html(d, "X")
    finally:
        shutil.rmtree(tmp)


def test_list_sessions_newest_first_with_sizes_and_test_flag():
    tmp, root = make()
    try:
        for name in os.listdir(root):
            open(os.path.join(root, name, "session.mp4"), "wb").write(b"x" * 1024)
        s = progress.list_sessions(root)
        assert [x["session"] for x in s] == ["20260921_120000_drill_wrong", "20260921_100000_challenge",
                                             "20260920_100000_drill"], [x["session"] for x in s]
        assert s[0]["test"] is True and s[1]["test"] is False
        assert all(x["has_video"] and x["size_mb"] > 0 for x in s)
        assert s[2]["mode"] == "drill" and s[2]["when"].year == 2026
        assert progress.list_sessions(os.path.join(tmp, "nobody")) == []
    finally:
        shutil.rmtree(tmp)


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