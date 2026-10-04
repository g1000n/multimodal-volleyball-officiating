"""
tools/pilot_check.py

One pass/fail table for the measurable items of the evaluation forms, from your own saved sessions -- a dry run
before the real evaluators come. Every number traces back to a session folder, so you can show where it came from.

    python tools/pilot_check.py --since 20261003_090000
    python tools/pilot_check.py --since 20261003_090000 --noise-session 20261003_101500_match_test

What it checks (IT-form item numbers in brackets):
  * recognition: per signal, attempts where the model saw the signal that was asked for (detected == target)
    -> at least 8 of 10                                                          [1, 6]
  * consistency: every block of 5 repetitions of one signal in a Drill session, how many got the most common verdict
    -> at least 4 of 5                                                           [17]
  * feedback time: graded attempts whose verdict appeared within 3 s of the arms coming down
    -> at least 4 of 5 (here: of all timed attempts)                             [13, 15]
  * stability: sessions that ended with an error or a lost camera, slowest single frame, speed first vs last quarter
                                                                                 [14, 15, 16]
  * whistle false alarms: microphone ("auto") whistles in the session you ran with NO whistle blown (--noise-session)
    -> 0                                                                         [2]

Only labelled "TEST: I will perform correctly" sessions (folder name ends in _correct) are used for recognition and
consistency, so deliberately wrong attempts do not count against the tool. Run from the project root.
"""
import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict

ROOT = os.path.join("data", "trainer_sessions")
FEEDBACK_TARGET_SECONDS = 3.0


def read_csv(path):
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return []


def verdict(ok):
    return "PASS" if ok else "FAIL"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default="", help="only sessions whose folder name sorts at or after this (YYYYMMDD_HHMMSS)")
    ap.add_argument("--noise-session", default="", help="folder name of the session run with background noise and NO whistle")
    args = ap.parse_args()

    sessions = sorted(d for d in glob.glob(os.path.join(ROOT, "*", "*")) if os.path.isdir(d)
                      and os.path.basename(d) >= args.since)
    if not sessions:
        print(f"No sessions found under {ROOT} since '{args.since}'.")
        return 1

    recog = defaultdict(lambda: [0, 0])          # signal -> [detected == target, attempts]
    blocks = []                                  # (session, signal, [verdicts]) for each block of 5 drill reps
    latencies = []
    bad_sessions, worst_frame, slowdowns = [], 0.0, []
    for d in sessions:
        name = os.path.basename(d)
        rows = [r for r in read_csv(os.path.join(d, "attempts.csv")) if r.get("kind") == "gesture"]
        try:
            with open(os.path.join(d, "summary.json"), encoding="utf-8") as fh:
                summary = json.load(fh)
        except (OSError, ValueError):
            summary = {}
        if summary.get("error") or summary.get("camera_lost"):
            bad_sessions.append(f"{name}: {summary.get('error') or 'camera lost'}")
        perf = summary.get("performance") or {}
        worst_frame = max(worst_frame, perf.get("frame_ms_max", 0.0))
        if perf.get("seconds", 0) >= 60 and perf.get("fps_first_quarter"):
            slowdowns.append((name, perf["fps_first_quarter"], perf.get("fps_last_quarter", 0.0)))
        for r in rows:
            if r.get("feedback_latency_s"):
                latencies.append(float(r["feedback_latency_s"]))
        if not name.endswith("_correct"):
            continue
        for r in rows:
            if "detected" in r:
                recog[r["target"]][0] += r["detected"] == r["target"]
                recog[r["target"]][1] += 1
        if "_drill" in name:
            by_signal = defaultdict(list)
            for r in rows:
                by_signal[r["target"]].append(r["verdict"])
            for sig, vs in by_signal.items():
                for i in range(0, len(vs) - 4, 5):
                    blocks.append((name, sig, vs[i:i + 5]))

    print(f"{len(sessions)} session(s) since '{args.since or 'the beginning'}'\n")

    print("RECOGNITION per signal (labelled-correct sessions; needs >= 8 of 10)   [items 1, 6]")
    if not recog:
        print("  no labelled 'correct' sessions with a 'detected' column -- set TEST_MODE and the session label")
    for sig, (hit, n) in sorted(recog.items()):
        rate = hit / n if n else 0.0
        enough = "" if n >= 10 else f"  (only {n} attempts, do at least 10)"
        print(f"  {sig:28s} {hit:3d}/{n:<3d} {rate:6.0%}  {verdict(rate >= 0.8 and n >= 10)}{enough}")

    print("\nCONSISTENCY: blocks of 5 drill repetitions (needs >= 4 of 5 the same verdict)   [item 17]")
    if not blocks:
        print("  no drill blocks of 5 found -- run Drill with 5 repetitions, labelled 'correct'")
    ok_blocks = 0
    for name, sig, vs in blocks:
        top, count = Counter(vs).most_common(1)[0]
        ok_blocks += count >= 4
        print(f"  {name:38s} {sig:26s} {count}/5 {top:9s} {verdict(count >= 4)}")
    if blocks:
        print(f"  -> {ok_blocks}/{len(blocks)} blocks pass")

    print(f"\nFEEDBACK TIME: verdict within {FEEDBACK_TARGET_SECONDS:.0f} s of the arms coming down   [items 13, 15]")
    if latencies:
        within = sum(1 for x in latencies if x <= FEEDBACK_TARGET_SECONDS)
        print(f"  {within}/{len(latencies)} attempts ({within / len(latencies):.0%}), mean {sum(latencies) / len(latencies):.2f} s, "
              f"max {max(latencies):.2f} s   {verdict(within / len(latencies) >= 0.8)}")
    else:
        print("  no timed attempts (sessions recorded before feedback timing was added have none)")

    print("\nSTABILITY   [items 14, 15, 16]")
    print(f"  sessions ending in an error or a lost camera: {len(bad_sessions)}   {verdict(not bad_sessions)}")
    for b in bad_sessions:
        print(f"    {b}")
    print(f"  slowest single frame: {worst_frame:.0f} ms   {verdict(worst_frame < 200)}  (about 200 ms or more is a visible stutter)")
    for name, first, last in slowdowns:
        drop = (first - last) / first if first else 0.0
        print(f"  {name:38s} fps {first:.1f} -> {last:.1f} ({drop:+.0%} drop)   {verdict(drop <= 0.10)}")

    if args.noise_session:
        print("\nWHISTLE FALSE ALARMS in the no-whistle noise session (needs 0)   [item 2]")
        matches = [d for d in sessions if os.path.basename(d) == args.noise_session]
        if not matches:
            print(f"  session '{args.noise_session}' not found among the sessions above")
        else:
            start = [r.get("detail", "") for r in read_csv(os.path.join(matches[0], "session_log.csv"))
                     if r.get("event") == "session_start"]
            if not start or "whistle_detector=not running" in start[0] or "whistle_detector=" not in start[0]:
                print("  the microphone whistle detector was NOT running in that session, so 0 false alarms proves nothing"
                      "   FAIL\n  (set WHISTLE_DEVICE_INDEX / pick the mic under Camera and mic, then run the noise test again)")
                return 0
            ev = read_csv(os.path.join(matches[0], "whistle_events.csv"))
            auto = [e for e in ev if e.get("source") == "auto"]
            print(f"  {len(auto)} microphone whistle(s) detected   {verdict(not auto)}")
            for e in auto:
                print(f"    at {e.get('seconds_from_session_start')} s, confidence {e.get('confidence')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
