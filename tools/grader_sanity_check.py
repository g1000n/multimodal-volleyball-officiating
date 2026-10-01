"""
tools/grader_sanity_check.py

Measures how often the FIVB-derived form checks in gesture_grader.py ACCEPT the clips in your own
dataset, per gesture, per check, per difficulty level. No model and no retraining involved.

Why you want this
-----------------
* It is the evidence for "the strict rules are sensible": if expert-validated reference clips pass
  most checks at Standard level, the thresholds are reasonable.
* It shows which threshold is too strict or too lenient BEFORE a trainee is told they are wrong.
  Edit the `pick(level, beginner, standard, referee)` values in gesture_grader.py and re-run.
* Where a check rejects many dataset clips, that is a real finding: either the threshold is off, or
  the recorded form differs from the FIVB wording (your referee validator already flagged this
  for Ball In). Report it honestly.

What the columns mean
---------------------
* ALL_CRITICAL_PASS: no Required check failed (this is the old summary; it ignores Important checks and Required
  checks that could not be seen).
* NO_VERDICT_CAP: what actually decides CORRECT in a live attempt: no Required check failed or was unseen, and no
  Important check failed. Use THIS number for "how many reference clips would pass at this level".
* grader_sanity_values.csv: the measured value of every check (median, 10th, 25th, 75th, 90th percentile) per gesture.
  Compare the left and right versions of a signal here before calling a difference in pass rates a real asymmetry.

Usage
-----
    python tools/grader_sanity_check.py --keypoints-dir data/keypoints --aspect 0.5625
    python tools/grader_sanity_check.py --keypoints-dir data/keypoints --max-clips 200

Assumptions (adjust if your layout differs):
  * clips are .npy arrays of shape (frames, 122) in the extract_keypoints.py raw-feature layout,
    stored somewhere under --keypoints-dir, with the gesture label as a parent folder name
    (or as the start of the file name);
  * --aspect is frame_height / frame_width of the ORIGINAL videos (1080p landscape = 0.5625;
    a portrait phone video would be 1.7778). It matters because elbow/arm angles are measured in
    true image geometry.
  * each clip is analysed as ONE window (onset and release included), so the numbers are a
    little harsher than a live attempt, which is analysed on the middle of the hold.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gesture_grader as gg  # noqa: E402


def infer_label(path, labels):
    parts = os.path.normpath(path).split(os.sep)
    for p in reversed(parts[:-1]):
        if p in labels:
            return p
    stem = os.path.splitext(parts[-1])[0]
    for lab in sorted(labels, key=len, reverse=True):
        if stem.startswith(lab):
            return lab
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keypoints-dir", default=os.path.join("data", "keypoints"))
    ap.add_argument("--aspect", type=float, default=9.0 / 16.0)
    ap.add_argument("--max-clips", type=int, default=0, help="max clips per gesture (0 = all)")
    ap.add_argument("--out", default=os.path.join("data", "grader_sanity_check.csv"))
    ap.add_argument("--dt", type=float, default=0.0,
                    help="seconds between two frames of the dataset clips (0.1 = 10 fps). Needed only for the pace check of "
                         "Service Authorization; leave at 0 to skip it.")
    args = ap.parse_args()

    labels = set(gg.RULES)
    clips = defaultdict(list)
    skipped = 0
    for root, _, files in os.walk(args.keypoints_dir):
        for fn in sorted(files):
            if not fn.endswith(".npy"):
                continue
            path = os.path.join(root, fn)
            lab = infer_label(path, labels)
            if lab is None:
                continue
            clips[lab].append(path)
    if not clips:
        print(f"No .npy clips with a known gesture label found under {args.keypoints_dir}.")
        print("Known labels:", ", ".join(sorted(labels)))
        return

    rows = []
    values = defaultdict(lambda: defaultdict(list))      # gesture -> check id -> measured values (level independent)
    units = {}
    ctx = {"dt": args.dt} if args.dt > 0 else None
    print(f"aspect = {args.aspect:.4f}" + (f", frame interval {args.dt} s" if ctx else ", pace check skipped (no --dt)") + "\n")
    for lab in sorted(clips):
        paths = clips[lab][: args.max_clips] if args.max_clips else clips[lab]
        data = []
        for p in paths:
            try:
                arr = np.load(p)
            except Exception:
                skipped += 1
                continue
            if arr.ndim != 2 or arr.shape[1] < 24 or len(arr) < 6:
                skipped += 1
                continue
            data.append(arr)
        if not data:
            continue
        print(f"=== {lab}  ({len(data)} clips) ===")
        for lv in gg.LEVELS:
            per = defaultdict(lambda: {"pass": 0, "fail": 0, "unverified": 0, "critical": False})
            all_critical_ok = 0
            no_cap = 0
            for arr in data:
                checks = gg.grade_form_only(lab, arr, lv, args.aspect, context=ctx)
                if not any(c.critical and c.status == "fail" for c in checks):
                    all_critical_ok += 1
                capped = (any(c.critical and c.status in ("fail", "unverified") for c in checks)
                          or any(c.strict and c.status == "fail" for c in checks))
                if not capped:
                    no_cap += 1
                for c in checks:
                    per[c.id][c.status] += 1
                    per[c.id]["critical"] = c.critical
                    if lv == "standard" and c.value is not None and np.isfinite(c.value):
                        values[lab][c.id].append(float(c.value))
            print(f"  {gg.LEVEL_CONFIG[lv].name:9s} clips passing every CRITICAL check: "
                  f"{100.0 * all_critical_ok / len(data):5.1f}%   with no verdict-capping check failing: "
                  f"{100.0 * no_cap / len(data):5.1f}%")
            for cid, d in per.items():
                n = d["pass"] + d["fail"] + d["unverified"]
                rate = 100.0 * d["pass"] / n
                flag = " (critical)" if d["critical"] else ""
                print(f"      {cid:20s} pass {rate:5.1f}%   fail {d['fail']:4d}   unverified {d['unverified']:4d}{flag}")
                rows.append([lab, lv, cid, int(d["critical"]), n, d["pass"], d["fail"], d["unverified"], round(rate, 1)])
            rows.append([lab, lv, "ALL_CRITICAL_PASS", 1, len(data), all_critical_ok, len(data) - all_critical_ok, 0,
                         round(100.0 * all_critical_ok / len(data), 1)])
            rows.append([lab, lv, "NO_VERDICT_CAP", 1, len(data), no_cap, len(data) - no_cap, 0,
                         round(100.0 * no_cap / len(data), 1)])
        print()

    # measured values per check, and left against right for the signals that have both
    vpath = os.path.join(os.path.dirname(args.out) or ".", "grader_sanity_values.csv")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(vpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gesture", "check", "n", "min", "p10", "p25", "median", "p75", "p90", "max"])
        for lab in sorted(values):
            for cid, vs in values[lab].items():
                a = np.asarray(vs)
                w.writerow([lab, cid, len(a), *[round(float(x), 3) for x in (a.min(), *np.percentile(a, [10, 25, 50, 75, 90]), a.max())]])
    pairs = sorted({lab[:-5] for lab in values if lab.endswith("_left") and lab[:-5] + "_right" in values})
    if pairs:
        print("Left against right (median of the measured value, standard level):")
        for base in pairs:
            for cid in values[base + "_left"]:
                if cid in values[base + "_right"]:
                    ml, mr = np.median(values[base + "_left"][cid]), np.median(values[base + "_right"][cid])
                    print(f"  {base:28s} {cid:18s} left {ml:8.2f}  right {mr:8.2f}  difference {ml - mr:+7.2f}")
        print()
    print(f"Saved {vpath}")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gesture", "level", "check", "critical", "clips", "pass", "fail", "unverified", "pass_rate_pct"])
        w.writerows(rows)
    print(f"Saved {args.out}" + (f"  ({skipped} unreadable/too-short files skipped)" if skipped else ""))
    print("\nHow to read it: at Standard level, a check that rejects most of your reference clips is a "
          "threshold to loosen (or a finding to report); a check that accepts wrong attempts is one to tighten.")


if __name__ == "__main__":
    main()