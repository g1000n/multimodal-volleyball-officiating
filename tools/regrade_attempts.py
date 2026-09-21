"""
tools/regrade_attempts.py

Re-grades the movements you already recorded with the CURRENT rules in gesture_grader.py, without redoing them.

Every graded attempt is saved as data/trainer_sessions/<trainee>/<session>/attempts/attempt_NNN_<signal>.npz
(your real keypoints, the model's answers, and the verdict you got at the time). After you change a threshold
in gesture_grader.py, run:

    python tools/regrade_attempts.py                     # everything, at the level it was recorded
    python tools/regrade_attempts.py --level standard    # re-grade everything at one level
    python tools/regrade_attempts.py --since 20260921_150000 --intent wrong
    python tools/regrade_attempts.py --verbose           # one line per attempt with the failed checks

It shows what changed (old verdict -> new verdict) and, for attempts labeled "correct on purpose" or
"wrong on purpose" (TEST_MODE), how many correct attempts are now accepted and wrong ones rejected.
It writes data/regrade_results.csv.

Run it from the project root (the folder that contains `data/`).
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gesture_grader as gg  # noqa: E402

RANK = {gg.VERDICT_INCORRECT: 0, gg.VERDICT_ALMOST: 1, gg.VERDICT_CORRECT: 2, gg.VERDICT_NO_READING: -1}


def load(path):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    labels = [str(x) for x in d["labels"]]
    recs = []
    for i, lab in enumerate(labels):
        rec = {"label": lab, "probs": d["probs"][i].astype(float), "frames": d["windows"][i].astype(float)}
        if int(d["ends"][i]) >= 0:
            rec["end"] = int(d["ends"][i])
        recs.append(rec)
    return meta, d["capture"].astype(float), recs


def regrade(meta, capture, recs, level):
    l2i = {k: int(v) for k, v in meta["label_to_idx"].items()}
    targets, ctxs = meta["targets"], meta.get("contexts") or [None] * len(meta["targets"])
    kw = dict(level=level, aspect=meta["aspect"], step_seconds=meta["step_seconds"])
    if len(targets) > 1:
        return gg.grade_sequence(targets, capture, recs, l2i, contexts=ctxs, **kw)
    return [gg.grade_attempt(targets[0], capture, recs, l2i, context=ctxs[0], **kw)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("data", "trainer_sessions"))
    ap.add_argument("--level", choices=gg.LEVELS, default=None, help="re-grade at this level (default: as recorded)")
    ap.add_argument("--intent", choices=["normal", "correct", "wrong"], default=None)
    ap.add_argument("--since", default="", help="only sessions whose folder name is >= this (YYYYmmdd_HHMMSS)")
    ap.add_argument("--trainee", default="")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=os.path.join("data", "regrade_results.csv"))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, "*", "*", "attempts", "*.npz")))
    rows, unchanged, better, worse = [], 0, 0, 0
    by_intent = defaultdict(lambda: [0, 0])          # intent -> [attempts, accepted (CORRECT)]
    for path in files:
        attempt_dir = os.path.dirname(path)
        session = os.path.basename(os.path.dirname(attempt_dir))
        trainee = os.path.basename(os.path.dirname(os.path.dirname(attempt_dir)))
        if session < args.since or (args.trainee and trainee != args.trainee):
            continue
        try:
            meta, capture, recs = load(path)
        except Exception as exc:
            print(f"skipped {path}: {exc}")
            continue
        if args.intent and meta.get("intent", "normal") != args.intent:
            continue
        level = args.level or meta["level"]
        new = regrade(meta, capture, recs, level)
        for k, (target, res) in enumerate(zip(meta["targets"], new)):
            old_v = meta["verdicts"][k]
            changed = res.verdict != old_v or level != meta["level"]
            if res.verdict == old_v:
                unchanged += 1
            elif RANK.get(res.verdict, 0) > RANK.get(old_v, 0):
                better += 1
            else:
                worse += 1
            failed = [c.id for c in res.checks if c.status == "fail"]
            rows.append([trainee, session, os.path.basename(path), target, meta.get("intent", "normal"),
                         meta.get("note", ""), meta["level"], level, old_v, res.verdict, meta["scores"][k], res.score,
                         ";".join(failed)])
            b = by_intent[meta.get("intent", "normal")]
            b[0] += 1
            b[1] += res.verdict == gg.VERDICT_CORRECT
            if args.verbose or (changed and res.verdict != old_v):
                print(f"{session}/{os.path.basename(path)[:40]:40s} {target:26s} {old_v:9s} -> {res.verdict:9s} "
                      f"{meta['scores'][k]:3d} -> {res.score:3d}  {('failed: ' + ','.join(failed)) if failed else ''}")

    if not rows:
        print("No saved attempts found. Run a session first (movement files are saved automatically), then run this "
              "again from the project root.")
        return
    print(f"\n{len(rows)} graded signals: {unchanged} unchanged, {better} better, {worse} worse.")
    for intent, (n, acc) in sorted(by_intent.items()):
        if intent == "correct":
            print(f"  labeled 'correct on purpose': {acc}/{n} now CORRECT ({100.0 * acc / n:.0f}%)")
        elif intent == "wrong":
            print(f"  labeled 'wrong on purpose':   {n - acc}/{n} now rejected ({100.0 * (n - acc) / n:.0f}%)")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trainee", "session", "file", "target", "intent", "note", "level_recorded", "level_regraded",
                    "verdict_before", "verdict_now", "score_before", "score_now", "failed_checks_now"])
        w.writerows(rows)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()