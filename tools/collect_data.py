"""
tools/collect_data.py

Packs everything one laptop recorded during the evaluation into ONE zip to upload to the team's Google Drive folder.
Run it from the project folder on every laptop that ran participants:

    py tools/collect_data.py --name alexia                  # data only (small, a few MB): enough for the results
    py tools/collect_data.py --name alexia --with-videos    # also the app's own session.mp4 videos (large)

The zip (collect_<name>_<date>.zip, in the project folder) holds:
  - trainer_sessions/<code>/...  every participant folder whose code starts with G- (attempts.csv, session_log.csv,
    summary.json, whistle_events.csv, report.html, and the attempts/*.npz movement files used for re-grading)
  - consent_records.csv          the consent log, only the rows of those codes
  - whistle_logs/                the whistle detector's own logs from the testing days
  - manifest.txt                 who packed it, on which computer and code version, which codes and sessions, and
                                 anything that looks wrong (a session the app did not finish, an old app version)
Nothing on the laptop is changed or deleted.
"""

import argparse
import csv
import datetime
import glob
import io
import json
import os
import platform
import subprocess
import sys
import zipfile

ROOT = os.path.join("data", "trainer_sessions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="your name or the laptop's name, e.g. alexia")
    ap.add_argument("--prefix", default="G-", help="participant codes start with this (default G-)")
    ap.add_argument("--since", default="20261005", help="testing started on this day, YYYYMMDD (for the whistle logs)")
    ap.add_argument("--with-videos", action="store_true", help="also pack each session's session.mp4 (large)")
    args = ap.parse_args()
    if not os.path.isdir(ROOT):
        sys.exit("Run this from the project folder (the one that contains data\\trainer_sessions).")

    codes = sorted(os.path.basename(d) for d in glob.glob(os.path.join(ROOT, args.prefix + "*")) if os.path.isdir(d))
    if not codes:
        sys.exit(f"No participant folders starting with {args.prefix} in {ROOT}. Did this laptop run participants?")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out = f"collect_{args.name}_{stamp}.zip"
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    lines = [f"Packed by: {args.name}", f"Computer: {platform.node()}", f"Code version (git): {commit or 'unknown'}",
             f"Packed at: {stamp}", f"Videos included: {'yes' if args.with_videos else 'no'}", "", "Participants:"]
    warnings = []
    n_files = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for code in codes:
            sessions = sorted(d for d in glob.glob(os.path.join(ROOT, code, "*")) if os.path.isdir(d))
            lines.append(f"  {code}: {len(sessions)} sessions: " + ", ".join(os.path.basename(s) for s in sessions))
            for s in sessions:
                if not os.path.exists(os.path.join(s, "summary.json")):
                    warnings.append(f"{code}/{os.path.basename(s)}: no summary.json (the app stopped before it ended)")
                att = os.path.join(s, "attempts.csv")
                if os.path.exists(att):
                    with open(att, encoding="utf-8") as f:
                        head = f.readline()
                    if "step_no" not in head:
                        warnings.append(f"{code}/{os.path.basename(s)}: recorded by an older app version (no step_no)")
            for path in glob.glob(os.path.join(ROOT, code, "**", "*"), recursive=True):
                if os.path.isdir(path) or (path.endswith(".mp4") and not args.with_videos):
                    continue
                z.write(path, os.path.relpath(path, "data"))
                n_files += 1
        # consent log: only the participants' rows
        log = os.path.join("data", "consent_records.csv")
        if os.path.exists(log):
            with open(log, newline="", encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f) if str(r.get("trainee_id", "")).startswith(args.prefix)]
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=["ph_time", "trainee_id", "event", "consent_version"], extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
            z.writestr("consent_records.csv", buf.getvalue())
            lines.append(f"\nConsent log rows: {len(rows)}")
        # whistle detector logs from the testing days
        since = datetime.datetime.strptime(args.since, "%Y%m%d").timestamp()
        wl = [p for p in glob.glob(os.path.join("data", "whistle_logs", "*")) if os.path.getmtime(p) >= since]
        for p in wl:
            z.write(p, os.path.relpath(p, "data"))
        lines.append(f"Whistle logs: {len(wl)} files")
        lines += ["", "Check these:" if warnings else "Nothing looks wrong."] + [f"  - {x}" for x in warnings]
        z.writestr("manifest.txt", "\n".join(lines) + "\n")

    size = os.path.getsize(out) / 1e6
    print("\n".join(lines))
    print(f"\nSaved {out} ({size:.1f} MB, {n_files} session files). Upload it to the team's Google Drive folder.")


if __name__ == "__main__":
    main()
