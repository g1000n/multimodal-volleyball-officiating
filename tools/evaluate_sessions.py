"""
tools/evaluate_sessions.py

Turns your labeled TEST sessions into the evaluation table for the paper.

How to collect the data
-----------------------
In the menu, set "Session label" to
    "TEST: I will perform correctly"          for attempts you do properly on purpose, and
    "TEST: I will perform WRONG on purpose"   for attempts you do wrongly on purpose
(wrong arm, elbow bent, arm too low, hands closed, a different signal ...). Use Drill with 5 or 10
repetitions. Do this at each difficulty level you want to report.

Run (from the project root):
    python tools/evaluate_sessions.py

It reads data/trainer_sessions/*/*/attempts.csv and reports, per difficulty level and per signal:
  * acceptance rate of correct attempts   (verdict CORRECT)          -> the grader is not too strict
  * rejection rate of wrong attempts      (verdict not CORRECT)      -> the grader is not too lenient
  * overall accuracy and Cohen's kappa between the LABEL you intended and the system's verdict
and lists the wrong attempts that were accepted (and correct attempts that were rejected) together with
the measured values of every check, so you can see which threshold to tune, or which mistake a single
2D camera simply cannot see (that goes into the delimitations).

Writes data/evaluation_summary.csv.

Honest note for the paper: the label is what the tester INTENDED, not an expert's judgement. Say so,
and add the agreement between independent human judges if you can.
"""

import csv
import glob
import os
import sys
from collections import defaultdict

ROOT = os.path.join("data", "trainer_sessions")


def kappa(tp, fn, fp, tn):
    n = tp + fn + fp + tn
    if n == 0:
        return float("nan")
    po = (tp + tn) / n
    pe = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def pct(a, b):
    return "n/a" if b == 0 else f"{100.0 * a / b:5.1f}%"


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "*", "*", "attempts.csv"))):
        session = os.path.basename(os.path.dirname(f))
        trainee = os.path.basename(os.path.dirname(os.path.dirname(f)))
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("kind") == "gesture" and r.get("intent") in ("correct", "wrong"):
                    r["trainee"], r["session"] = trainee, session
                    rows.append(r)
    if not rows:
        print("No labeled test attempts found. In the menu set 'Session label' to a TEST option, run a Drill,")
        print("then run this again from the project root.")
        return

    out = []
    levels = sorted({r["level"] for r in rows}, key=lambda x: ["beginner", "standard", "referee"].index(x)
                    if x in ("beginner", "standard", "referee") else 9)
    for lv in levels:
        sub = [r for r in rows if r["level"] == lv]
        print(f"\n=== {lv.upper()} level ===")
        print(f"{'signal':30s} {'correct: n':>10s} {'accepted':>9s} {'wrong: n':>9s} {'rejected':>9s}")
        by_sig = defaultdict(lambda: {"c": [], "w": []})
        for r in sub:
            by_sig[r["target"]]["c" if r["intent"] == "correct" else "w"].append(r)
        tp = fn = fp = tn = 0
        for sig, d in sorted(by_sig.items()):
            c_ok = sum(1 for r in d["c"] if r["verdict"] == "CORRECT")
            w_rej = sum(1 for r in d["w"] if r["verdict"] != "CORRECT")
            tp += c_ok
            fn += len(d["c"]) - c_ok
            fp += len(d["w"]) - w_rej
            tn += w_rej
            print(f"{sig:30s} {len(d['c']):10d} {pct(c_ok, len(d['c'])):>9s} {len(d['w']):9d} {pct(w_rej, len(d['w'])):>9s}")
            out.append([lv, sig, len(d["c"]), c_ok, len(d["w"]), w_rej])
        n = tp + fn + fp + tn
        print(f"{'ALL':30s} {tp + fn:10d} {pct(tp, tp + fn):>9s} {fp + tn:9d} {pct(tn, fp + tn):>9s}")
        print(f"overall accuracy vs intended label: {pct(tp + tn, n).strip()}   Cohen's kappa: {kappa(tp, fn, fp, tn):.2f}")
        out.append([lv, "ALL", tp + fn, tp, fp + tn, tn])

        bad_acc = [r for r in sub if r["intent"] == "wrong" and r["verdict"] == "CORRECT"]
        bad_rej = [r for r in sub if r["intent"] == "correct" and r["verdict"] != "CORRECT"]
        if bad_acc:
            print("\n  WRONG attempts that were accepted (grader too lenient, or a mistake the camera cannot see):")
            for r in bad_acc[:12]:
                print(f"    {r['trainee']}/{r['session']}  {r['target']}  score {r['score']}\n      {r['check_values']}")
        if bad_rej:
            print("\n  CORRECT attempts that were rejected (grader too strict, or the model misread you):")
            for r in bad_rej[:12]:
                why = r["confused_with"] and f"looked like {r['confused_with']}; " or ""
                print(f"    {r['trainee']}/{r['session']}  {r['target']}  {r['verdict']} {r['score']}  {why}failed: {r['failed_checks']}\n      {r['check_values']}")

    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "evaluation_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["level", "signal", "correct_attempts", "correct_accepted", "wrong_attempts", "wrong_rejected"])
        w.writerows(out)
    print("\nSaved data/evaluation_summary.csv")


if __name__ == "__main__":
    sys.exit(main())