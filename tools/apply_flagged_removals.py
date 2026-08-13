"""
apply_flagged_removals.py

Reads the progress file saved by clip_reviewer.py, lists every clip you
flagged for removal, asks for confirmation, then MOVES (not deletes)
those clips into data/flagged_removed/<gesture_label>/ — same relative
structure as raw_clips, so it's easy to review or restore later if you
change your mind.

After running this, you'll need to rerun the pipeline from the top so
the manifest/keypoints reflect the removed clips:
    python build_manifest.py
    python extract_keypoints.py
    python dataset_split.py
    python train.py

Run from your project root:
    python apply_flagged_removals.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import os
import json
import shutil

PROGRESS_PATH = "data/clip_review_progress.json"
QUARANTINE_DIR = "data/flagged_removed"


def main():
    if not os.path.exists(PROGRESS_PATH):
        print("No review progress found. Run clip_reviewer.py first.")
        return

    with open(PROGRESS_PATH, "r") as f:
        progress = json.load(f)

    flagged_paths = [path for path, status in progress["reviewed"].items() if status == "flagged"]

    if not flagged_paths:
        print("No clips are currently flagged for removal.")
        return

    print(f"{len(flagged_paths)} clip(s) flagged for removal:\n")
    for path in flagged_paths:
        print(f"  {path}")

    print(f"\nThese will be MOVED (not deleted) to '{QUARANTINE_DIR}/', preserving folder structure.")
    print("You can restore them manually later if needed.")
    confirm = input("\nType 'yes' to proceed, anything else to cancel: ").strip().lower()

    if confirm != "yes":
        print("Cancelled. No files were moved.")
        return

    moved_count = 0
    missing_count = 0

    for clip_path in flagged_paths:
        if not os.path.exists(clip_path):
            print(f"  MISSING (already moved or deleted?): {clip_path}")
            missing_count += 1
            continue

        # Preserve the gesture-label subfolder structure inside quarantine
        rel_path = os.path.relpath(clip_path, "data/raw_clips")
        dest_path = os.path.join(QUARANTINE_DIR, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        shutil.move(clip_path, dest_path)
        moved_count += 1
        print(f"  Moved: {clip_path} -> {dest_path}")

    print(f"\nDone. {moved_count} clip(s) moved to quarantine, {missing_count} were already missing.")
    print("\nNext steps — rerun the pipeline so these removals take effect:")
    print("  python build_manifest.py")
    print("  python extract_keypoints.py")
    print("  python dataset_split.py")
    print("  python train.py")


if __name__ == "__main__":
    main()