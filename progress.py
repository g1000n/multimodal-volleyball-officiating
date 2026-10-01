"""
progress.py

Progress of ONE trainee over all their saved sessions (data/trainer_sessions/<trainee>/...). No Tk, no camera: it only
reads the attempts.csv files the trainer already writes. Used by the "My progress" screen in the menu and by the saved
progress report (progress.html in the trainee's folder).

Test sessions (labeled "correct on purpose" / "wrong on purpose" in TEST_MODE) are left out, because they are not
real training.
"""

import csv
import datetime
import glob
import html
import os
from collections import Counter, defaultdict

import gesture_grader as gg


def _when(session_name):
    try:
        return datetime.datetime.strptime(session_name[:15], "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def check_label(signal, check_id):
    """Trainee wording for a check id of a signal (e.g. 'hands_open' -> 'Hand open (as in the FIVB illustration)')."""
    ctx = {"side": "left"} if signal == "double_contact" else None
    for it in gg.graded_summary(signal, ctx):
        if it["id"] == check_id:
            return it["label"]
    return check_id.replace("_", " ")


def _pct(a, b):
    return 100.0 * a / b if b else 0.0


def build_progress(trainee_dir):
    """Returns a dict describing the trainee's progress, or one with sessions == [] when there is nothing yet."""
    rows = []
    sessions = []
    for f in sorted(glob.glob(os.path.join(trainee_dir, "*", "attempts.csv"))):
        session = os.path.basename(os.path.dirname(f))
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("intent", "normal") in ("correct", "wrong"):
                    continue
                r["session"] = session
                r["when"] = _when(session)
                rows.append(r)
    by_session = defaultdict(list)
    for r in rows:
        by_session[r["session"]].append(r)
    for name in sorted(by_session):
        g = [r for r in by_session[name] if r["kind"] == "gesture"]
        if not g:
            continue
        correct = sum(1 for r in g if r["verdict"] == gg.VERDICT_CORRECT)
        parts = name.split("_", 2)
        sessions.append({"session": name, "when": _when(name), "mode": parts[2] if len(parts) > 2 else "",
                         "level": g[0].get("level", ""), "attempts": len(g), "correct": correct,
                         "accuracy": _pct(correct, len(g)), "avg_score": sum(float(r["score"]) for r in g) / len(g)})
    gest = [r for r in rows if r["kind"] == "gesture"]
    whistle = [r for r in rows if r["kind"] == "whistle"]
    out = {"sessions": sessions, "total_attempts": len(gest),
           "total_correct": sum(1 for r in gest if r["verdict"] == gg.VERDICT_CORRECT),
           "avg_score": (sum(float(r["score"]) for r in gest) / len(gest)) if gest else 0.0,
           "per_signal": {}, "weakest": [], "trend_text": "", "whistle": None}
    out["accuracy"] = _pct(out["total_correct"], out["total_attempts"])

    per = defaultdict(list)
    for r in gest:
        per[r["target"]].append(r)                      # rows are already in time order (sessions sorted by name)
    for label, rs in per.items():
        n = len(rs)
        correct = sum(1 for r in rs if r["verdict"] == gg.VERDICT_CORRECT)
        almost = sum(1 for r in rs if r["verdict"] == gg.VERDICT_ALMOST)
        scores = [float(r["score"]) for r in rs]
        trend = None
        if n >= 4:
            half = n // 2
            trend = sum(scores[half:]) / len(scores[half:]) - sum(scores[:half]) / half
        issues = Counter(c for r in rs for c in (r.get("failed_checks") or "").split(";") if c)
        out["per_signal"][label] = {"attempts": n, "correct": correct, "almost": almost,
                                    "accuracy": _pct(correct, n), "avg_score": sum(scores) / n, "trend": trend,
                                    "issues": [(check_label(label, c), k) for c, k in issues.most_common(3)]}
    weak = [(v["accuracy"], v["avg_score"], k) for k, v in out["per_signal"].items() if v["attempts"] >= 3]
    out["weakest"] = [k for _a, _s, k in sorted(weak)[:3] if out["per_signal"][k]["accuracy"] < 100]

    if len(gest) >= 6:
        third = len(gest) // 3
        first = _pct(sum(1 for r in gest[:third] if r["verdict"] == gg.VERDICT_CORRECT), third)
        last = _pct(sum(1 for r in gest[-third:] if r["verdict"] == gg.VERDICT_CORRECT), third)
        diff = last - first
        out["trend_text"] = (f"Your first attempts were {first:.0f}% correct and your latest are {last:.0f}% correct "
                             f"({'up' if diff > 0 else 'down' if diff < 0 else 'no change'}"
                             f"{f' {abs(diff):.0f} points' if diff else ''}).")
    elif gest:
        out["trend_text"] = "Do a few more attempts to see your trend."
    if whistle:
        ok = sum(1 for r in whistle if r["verdict"] == gg.VERDICT_CORRECT)
        out["whistle"] = {"attempts": len(whistle), "on_time": ok, "pct": _pct(ok, len(whistle))}
    return out


def focus_lines(data):
    """Plain sentences: what to work on next."""
    lines = []
    for label in data["weakest"]:
        v = data["per_signal"][label]
        why = f" Most missed: {v['issues'][0][0]}." if v["issues"] else ""
        lines.append(f"{gg.short_label(label)}: {v['accuracy']:.0f}% correct over {v['attempts']} tries.{why}")
    if not lines and data["total_attempts"]:
        lines.append("Nothing stands out. Keep practising and try a harder difficulty level.")
    return lines


def progress_html(data, display_name):
    e = html.escape
    def trend_cell(v):
        return "" if v["trend"] is None else "%+.0f" % v["trend"]

    rows = "".join(
        f"<tr><td>{e(gg.short_label(k))}</td><td>{v['attempts']}</td><td>{v['correct']}</td><td>{v['accuracy']:.0f}%</td>"
        f"<td>{v['avg_score']:.0f}</td><td>{trend_cell(v)}</td>"
        f"<td>{e('; '.join(n + ' (' + str(c) + 'x)' for n, c in v['issues']))}</td></tr>"
        for k, v in sorted(data["per_signal"].items(), key=lambda kv: kv[1]["accuracy"])
    ) or "<tr><td colspan='7'>No attempts yet</td></tr>"
    srows = "".join(
        f"<tr><td>{e(s['when'].strftime('%Y-%m-%d %H:%M') if s['when'] else s['session'])}</td><td>{e(s['mode'])}</td>"
        f"<td>{e(s['level'])}</td><td>{s['attempts']}</td><td>{s['correct']}</td><td>{s['accuracy']:.0f}%</td>"
        f"<td>{s['avg_score']:.0f}</td></tr>" for s in data["sessions"]) or "<tr><td colspan='7'>No sessions yet</td></tr>"
    focus = "".join(f"<li>{e(t)}</li>" for t in focus_lines(data)) or "<li>Nothing yet.</li>"
    wh = ""
    if data["whistle"]:
        w = data["whistle"]
        wh = f"<p><b>Whistle:</b> on time {w['on_time']} of {w['attempts']} ({w['pct']:.0f}%)</p>"
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Progress</title>
<style>body{{font-family:Arial,sans-serif;margin:30px;background:#fafafa;color:#222}}
table{{border-collapse:collapse;width:100%;margin:8px 0 22px}}th,td{{border:1px solid #ccc;padding:6px 10px;font-size:14px;text-align:left}}
th{{background:#eee}}.box{{background:#fff;border:1px solid #ddd;border-radius:6px;padding:14px 20px;margin-bottom:18px}}</style></head><body>
<h1>Progress: {e(display_name)}</h1>
<div class="box"><p><b>Sessions:</b> {len(data['sessions'])} &nbsp; <b>Attempts:</b> {data['total_attempts']} &nbsp;
<b>Correct:</b> {data['total_correct']} ({data['accuracy']:.0f}%) &nbsp; <b>Average score:</b> {data['avg_score']:.0f}/100</p>
<p>{e(data['trend_text'])}</p>{wh}
<p style="color:#666;font-size:12px">{e(gg.DISCLAIMER)} Test sessions are not included.</p></div>
<h2>What to work on</h2><ul>{focus}</ul>
<h2>By signal (weakest first)</h2><table><tr><th>Signal</th><th>Tries</th><th>Correct</th><th>Accuracy</th><th>Avg score</th><th>Trend</th><th>Most missed checks</th></tr>{rows}</table>
<h2>Sessions</h2><table><tr><th>When</th><th>Mode</th><th>Level</th><th>Attempts</th><th>Correct</th><th>Accuracy</th><th>Avg score</th></tr>{srows}</table>
</body></html>"""


def _session_signals(session_dir):
    """Short text of the signals performed in a session, read from its attempts.csv (works for older sessions too)."""
    try:
        with open(os.path.join(session_dir, "attempts.csv"), newline="", encoding="utf-8") as f:
            seen = []
            for r in csv.DictReader(f):
                if r.get("kind") == "gesture" and r.get("target") and r["target"] not in seen:
                    seen.append(r["target"])
    except OSError:
        return ""
    if not seen:
        return ""
    names = [gg.short_label(s) for s in seen]
    if len(names) <= 2:
        return " + ".join(names)
    return f"{len(names)} signals"


def list_sessions(trainee_dir):
    """Every saved session of one trainee, newest first (used by the My sessions window)."""
    import json
    out = []
    if not os.path.isdir(trainee_dir):
        return out
    for name in sorted(os.listdir(trainee_dir), reverse=True):
        d = os.path.join(trainee_dir, name)
        if not os.path.isdir(d):
            continue
        info = {"session": name, "path": d, "when": _when(name), "mode": "", "level": "", "attempts": 0, "correct": 0,
                "points": 0, "max_points": 0}
        parts = name.split("_", 2)
        if len(parts) > 2:
            info["mode"] = parts[2]
        try:
            with open(os.path.join(d, "summary.json"), encoding="utf-8") as f:
                s = json.load(f)
            info.update(mode=s.get("mode", info["mode"]), level=s.get("level", ""), attempts=s.get("attempts", 0),
                        correct=s.get("correct", 0), points=s.get("points", 0), max_points=s.get("max_points", 0))
        except (OSError, ValueError):
            pass
        info["signals"] = _session_signals(d)
        size = 0
        for base, _dirs, files in os.walk(d):
            for f in files:
                try:
                    size += os.path.getsize(os.path.join(base, f))
                except OSError:
                    pass
        info.update(size_mb=size / (1024 * 1024), has_video=os.path.exists(os.path.join(d, "session.mp4")),
                    has_report=os.path.exists(os.path.join(d, "report.html")),
                    test=name.endswith(("_correct", "_wrong")))
        out.append(info)
    return out