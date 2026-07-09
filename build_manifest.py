"""
build_manifest.py

Scans the raw_clips/ folder and builds a manifest CSV listing every clip,
its gesture label, and which participant performed it — parsed straight
from the filename convention:

    <gesture_name>_<person_id>_<take_number>.mp4

Example: team_to_serve_right_p01_01.mp4
    -> gesture_label = "team_to_serve_right"
    -> person_id     = "p01"
    -> take_number   = "01"

Run this any time you add new clips — it will regenerate the manifest
from scratch based on whatever is currently in raw_clips/.
"""

import os
import re
import csv

RAW_CLIPS_DIR = "data/raw_clips"
OUTPUT_CSV = "data/dataset_manifest.csv"

# Matches: <gesture_name>_<pXX>_<take>.mp4  (gesture_name can contain underscores)
FILENAME_PATTERN = re.compile(r"^(.+)_(p\d+)_(\d+)\.(mp4|mov)$", re.IGNORECASE)


def build_manifest():
    rows = []
    skipped = []

    for gesture_folder in sorted(os.listdir(RAW_CLIPS_DIR)):
        folder_path = os.path.join(RAW_CLIPS_DIR, gesture_folder)
        if not os.path.isdir(folder_path):
            continue

        for filename in sorted(os.listdir(folder_path)):
            match = FILENAME_PATTERN.match(filename)
            if not match:
                skipped.append(os.path.join(gesture_folder, filename))
                continue

            gesture_name_from_file, person_id, take_number, ext = match.groups()
            clip_path = os.path.join(folder_path, filename)

            rows.append({
                "clip_path": clip_path,
                "gesture_label": gesture_folder,   # trust the folder name as the label
                "person_id": person_id.lower(),
                "take_number": take_number,
                "source": "volunteer",             # change manually later for external clips
            })

    # Write manifest
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "clip_path", "gesture_label", "person_id", "take_number", "source"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest written: {OUTPUT_CSV}")
    print(f"Total clips found: {len(rows)}")

    # Quick per-class, per-person summary — useful sanity check
    from collections import defaultdict
    class_counts = defaultdict(int)
    person_counts = defaultdict(int)
    for row in rows:
        class_counts[row["gesture_label"]] += 1
        person_counts[row["person_id"]] += 1

    print("\nClips per gesture class:")
    for label, count in sorted(class_counts.items()):
        print(f"  {label}: {count}")

    print("\nClips per participant:")
    for person, count in sorted(person_counts.items()):
        print(f"  {person}: {count}")

    if skipped:
        print(f"\nWARNING: {len(skipped)} files did not match the naming convention and were skipped:")
        for s in skipped:
            print(f"  - {s}")


if __name__ == "__main__":
    build_manifest()