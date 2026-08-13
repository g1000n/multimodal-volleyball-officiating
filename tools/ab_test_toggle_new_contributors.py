"""
ab_test_toggle_new_contributors.py

Temporarily moves Dwayne (p08) and Liam (p09)'s ball_out and
double_contact clips OUT of data/raw_clips/ into a holding folder, so
you can rerun the pipeline and test_reference_clips.py without them —
to check whether their addition (specifically the tighter/no-neutral-
padding recording style) is what caused the reference-clip accuracy
regression.

This is REVERSIBLE — clips are moved, not deleted, and can be restored
with --restore.

USAGE:
    Remove them (before re-running the pipeline to test "without them"):
        python ab_test_toggle_new_contributors.py

    Put them back afterward (to test "with them" again, or once you've
    decided what to do):
        python ab_test_toggle_new_contributors.py --restore

WORKFLOW:
    1. python ab_test_toggle_new_contributors.py          (remove)
    2. python build_manifest.py
    3. python extract_keypoints.py
    4. python dataset_split.py
    5. python train.py
    6. python test_reference_clips.py                      <- compare this number
    7. python ab_test_toggle_new_contributors.py --restore (put them back)
    8. Re-run steps 2-6 again to confirm the "with them" number, if needed
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import os
import re
import shutil
import sys

RAW_CLIPS_DIR = "data/raw_clips"
TEMP_HOLD_DIR = "data/ab_test_temp_removed"

TARGET_PEOPLE = {"p08", "p09"}
TARGET_CLASSES = {"ball_out", "double_contact"}

# Same filename convention as build_manifest.py: <gesture>_<personid>_<take>.ext
FILENAME_PATTERN = re.compile(r"^(.+)_(p\d+)_(\d+)\.(mp4|mov)$", re.IGNORECASE)


def remove_mode():
    moved_count = 0

    for gesture_folder in TARGET_CLASSES:
        folder_path = os.path.join(RAW_CLIPS_DIR, gesture_folder)
        if not os.path.isdir(folder_path):
            continue

        for filename in sorted(os.listdir(folder_path)):
            match = FILENAME_PATTERN.match(filename)
            if not match:
                continue

            _, person_id, _, _ = match.groups()
            if person_id.lower() not in TARGET_PEOPLE:
                continue

            src = os.path.join(folder_path, filename)
            dest_dir = os.path.join(TEMP_HOLD_DIR, gesture_folder)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, filename)

            shutil.move(src, dest)
            moved_count += 1
            print(f"  Moved out: {src} -> {dest}")

    print(f"\nDone. {moved_count} clip(s) moved to {TEMP_HOLD_DIR}/.")
    print("Now run the pipeline fresh to test WITHOUT p08/p09's ball_out/double_contact clips:")
    print("  python build_manifest.py")
    print("  python extract_keypoints.py")
    print("  python dataset_split.py")
    print("  python train.py")
    print("  python test_reference_clips.py")
    print("\nWhen done comparing, restore with:")
    print("  python ab_test_toggle_new_contributors.py --restore")


def restore_mode():
    if not os.path.isdir(TEMP_HOLD_DIR):
        print(f"No holding folder found at {TEMP_HOLD_DIR}/ — nothing to restore.")
        return

    restored_count = 0

    for gesture_folder in os.listdir(TEMP_HOLD_DIR):
        folder_path = os.path.join(TEMP_HOLD_DIR, gesture_folder)
        if not os.path.isdir(folder_path):
            continue

        for filename in sorted(os.listdir(folder_path)):
            src = os.path.join(folder_path, filename)
            dest_dir = os.path.join(RAW_CLIPS_DIR, gesture_folder)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, filename)

            shutil.move(src, dest)
            restored_count += 1
            print(f"  Restored: {src} -> {dest}")

    print(f"\nDone. {restored_count} clip(s) restored to {RAW_CLIPS_DIR}/.")
    print("Re-run the pipeline again to get back to the full dataset (with p08/p09):")
    print("  python build_manifest.py")
    print("  python extract_keypoints.py")
    print("  python dataset_split.py")
    print("  python train.py")


if __name__ == "__main__":
    if "--restore" in sys.argv:
        restore_mode()
    else:
        remove_mode()