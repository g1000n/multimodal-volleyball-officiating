"""
tools/performance_report.py

Measured speed and load of the training tool, per mode, from every saved session. Every session records these figures
in summary.json ("performance"): frames per second, time for feature extraction and for the model, time to compute a
verdict, and (when the `psutil` package is installed) CPU and memory. Use it for the performance items of the expert
evaluation form (time behaviour, resource utilization, capacity).

    python tools/performance_report.py
    python tools/performance_report.py --since 20260921_150000
    python tools/performance_report.py --min-seconds 60          # ignore very short sessions

"Capacity" evidence: compare the speed in the first and the last quarter of the session (fps first vs last). If they are
about the same, the tool keeps its speed over a full session.

Writes data/performance_summary.csv. Run from the project root.
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.join("data", "trainer_sessions")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="")
    ap.add_argument("--min-seconds", type=float, default=20.0)
    args = ap.parse_args()

    by_mode = defaultdict(list)
    n_files = 0
    for f in sorted(glob.glob(os.path.join(ROOT, "*", "*", "summary.json"))):
        session = os.path.basename(os.path.dirname(f))
        if session < args.since:
            continue
        try:
            with open(f, encoding="utf-8") as fh:
                s = json.load(fh)
        except (OSError, ValueError):
            continue
        p = s.get("performance")
        n_files += 1
        if not p or p.get("seconds", 0) < args.min_seconds:
            continue
        by_mode[s.get("mode", "?")].append(p)

    if not by_mode:
        print(f"No sessions with performance figures found ({n_files} sessions read). Sessions recorded with an older "
              f"version have none; run a new session of at least {args.min_seconds:.0f} seconds.")
        return 1

    cols = ["mode", "sessions", "seconds", "fps_mean", "fps_first_quarter", "fps_last_quarter", "extract_ms", "model_ms",
            "verdict_ms", "cpu_program_pct", "cpu_computer_pct", "ram_mb", "worst_frame_ms",
            "feedback_attempts", "feedback_within_3s", "feedback_mean_s", "feedback_max_s"]
    rows = []
    print(f"{'mode':10s} {'sess':>4s} {'sec':>6s} {'fps':>6s} {'fps 1st':>8s} {'fps last':>9s} {'extract':>8s} {'model':>7s} "
          f"{'verdict':>8s} {'cpu prog':>9s} {'cpu pc':>7s} {'ram MB':>7s} {'worst fr':>9s} {'<=3s':>7s} {'fb mean':>8s} {'fb max':>7s}")
    for mode, ps in sorted(by_mode.items()):
        r = [mode, len(ps), sum(p["seconds"] for p in ps), mean([p["fps_mean"] for p in ps]),
             mean([p["fps_first_quarter"] for p in ps]), mean([p["fps_last_quarter"] for p in ps]),
             mean([p["extract_ms_mean"] for p in ps]), mean([p["classify_ms_mean"] for p in ps]),
             mean([p["grade_ms_mean"] for p in ps]),
             mean([p["cpu_process_mean_pct"] for p in ps if p.get("psutil")]),
             mean([p["cpu_system_mean_pct"] for p in ps if p.get("psutil")]),
             mean([p["ram_mean_mb"] for p in ps if p.get("psutil")]),
             max(p.get("frame_ms_max", 0.0) for p in ps),
             # the 3-second rule (sessions recorded before feedback timing was added have none)
             sum(p.get("feedback_n", 0) for p in ps), sum(p.get("feedback_within_3s", 0) for p in ps),
             (sum(p.get("feedback_mean_s", 0.0) * p.get("feedback_n", 0) for p in ps)
              / max(1, sum(p.get("feedback_n", 0) for p in ps))),
             max(p.get("feedback_max_s", 0.0) for p in ps)]
        rows.append(r)
        print(f"{mode:10s} {r[1]:4d} {r[2]:6.0f} {r[3]:6.1f} {r[4]:8.1f} {r[5]:9.1f} {r[6]:7.1f}ms {r[7]:6.1f}ms "
              f"{r[8]:7.1f}ms {r[9]:8.1f}% {r[10]:6.1f}% {r[11]:7.0f} {r[12]:7.0f}ms {r[14]:3d}/{r[13]:<3d} "
              f"{r[15]:7.2f}s {r[16]:6.2f}s")
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "performance_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows([[round(x, 1) if isinstance(x, float) else x for x in r] for r in rows])
    print("\nfps 1st / fps last = speed in the first and the last quarter of each session (about equal means no slowdown).")
    print("worst fr = slowest single frame (a visible stutter is roughly 200 ms or more).")
    print("<=3s = graded attempts whose verdict appeared within 3 s of the arms coming down (the forms' 3-second rule);")
    print("       fb mean / fb max = that delay in seconds.")
    print("cpu prog = share of ONE core used by this program; cpu pc = whole computer. 0 means psutil is not installed.")
    print("Saved data/performance_summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())