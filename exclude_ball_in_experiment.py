"""
exclude_ball_in_experiment.py

Non-destructive way to test "does the standing-still false positive
disappear if ball_in isn't a class at all?" without permanently losing
anything. dataset_split.py and train.py both hardcode
MANIFEST_PATH = "data/dataset_manifest.csv" -- rather than editing
those files, this script backs up the real manifest, swaps in a
filtered version with ball_in rows removed, and gives you a one-command
way to put the original back once the experiment's done.

USAGE:

    Step 1 -- start the experiment (backs up + filters the manifest):
        python exclude_ball_in_experiment.py start

    Step 2 -- run the normal pipeline as usual:
        python dataset_split.py
        python train.py

    Step 3 -- test (live_deployment.py / replay_recorded_footage.py),
    specifically checking whether standing-still still gets misread.

    Step 4 -- put the original manifest (with ball_in) back:
        python exclude_ball_in_experiment.py restore

Safe to run 'start' only once without restoring first -- it refuses to
overwrite an existing backup, so you can't accidentally lose the
original by running 'start' twice in a row.
"""

import sys
import csv
import shutil
import os

MANIFEST_PATH = "data/dataset_manifest.csv"
BACKUP_PATH = "data/dataset_manifest_backup_with_ball_in.csv"
EXCLUDED_LABEL = "ball_in"


def start():
    if os.path.exists(BACKUP_PATH):
        print(f"A backup already exists at {BACKUP_PATH} -- looks like the experiment is already running.")
        print("Run 'python exclude_ball_in_experiment.py restore' first if you want to start over.")
        return

    if not os.path.exists(MANIFEST_PATH):
        print(f"No manifest found at {MANIFEST_PATH}. Nothing to do.")
        return

    shutil.copy(MANIFEST_PATH, BACKUP_PATH)
    print(f"Backed up original manifest to: {BACKUP_PATH}")

    with open(MANIFEST_PATH, "r") as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys() if rows else []

    kept_rows = [r for r in rows if r.get("gesture_label") != EXCLUDED_LABEL]
    removed_count = len(rows) - len(kept_rows)

    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    print(f"Removed {removed_count} '{EXCLUDED_LABEL}' rows from the manifest.")
    print(f"Manifest now has {len(kept_rows)} rows across the remaining classes.")
    print("\nNext steps:")
    print("  python dataset_split.py")
    print("  python train.py")
    print("\nWhen you're done testing, restore the original with:")
    print("  python exclude_ball_in_experiment.py restore")


def restore():
    if not os.path.exists(BACKUP_PATH):
        print(f"No backup found at {BACKUP_PATH} -- nothing to restore. "
              f"(Did you already restore, or never run 'start'?)")
        return

    shutil.copy(BACKUP_PATH, MANIFEST_PATH)
    os.remove(BACKUP_PATH)
    print(f"Restored the original manifest (with '{EXCLUDED_LABEL}') to: {MANIFEST_PATH}")
    print("Backup file removed. You'll need to re-run dataset_split.py and train.py")
    print("again if you want the real 8-class model back as your active models/final_model.pt.")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("start", "restore"):
        print("Usage:")
        print("  python exclude_ball_in_experiment.py start     -- back up + remove ball_in rows")
        print("  python exclude_ball_in_experiment.py restore   -- put the original manifest back")
        return

    if sys.argv[1] == "start":
        start()
    else:
        restore()


if __name__ == "__main__":
    main()