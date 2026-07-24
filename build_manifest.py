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

--------------------------------------------------------------------
FIX (this version): previously, every rebuild wrote out ONLY 5 columns
(clip_path, gesture_label, person_id, take_number, source) -- silently
dropping keypoint_path, frame_count, and split for EVERY clip, not just
new ones, every single time this ran. That meant extract_keypoints.py's
incremental skip-logic (skip clips that already have a valid
keypoint_path) could never actually work: right after build_manifest.py
ran, every row looked "not yet extracted" again, even clips that had
been extracted hours ago -- forcing a full re-extraction every time you
added even one new clip/folder (like a new "nothing" class).

Fixed by reading the OLD manifest first (if it exists) into a lookup
keyed by clip_path, then carrying forward keypoint_path/frame_count/
split for any clip_path that already existed before. Only genuinely
NEW clip_paths get blank keypoint_path/frame_count/split (correctly
signaling "not yet processed" to extract_keypoints.py and
dataset_split.py). Existing clips' extraction/split state now survives
manifest rebuilds.
--------------------------------------------------------------------
"""

import os
import re
import csv
from collections import defaultdict

RAW_CLIPS_DIR = "data/raw_clips"
OUTPUT_CSV = "data/dataset_manifest.csv"

# All columns the manifest can carry. clip_path/gesture_label/person_id/
# take_number/source are always freshly derived from the filesystem scan.
# keypoint_path/frame_count/split are carried forward from the OLD
# manifest when a clip_path already existed, and left blank for new ones.
ALL_FIELDNAMES = [
    "clip_path", "gesture_label", "person_id", "take_number", "source",
    "keypoint_path", "frame_count", "split",
]

# Matches: <gesture_name>_<pXX>_<take>.mp4  (gesture_name can contain underscores)
FILENAME_PATTERN = re.compile(r"^(.+)_(p\d+)_(\d+)\.(mp4|mov)$", re.IGNORECASE)


def load_old_manifest_lookup():
    """
    Reads the EXISTING manifest (if any) into a dict keyed by clip_path,
    so we can carry forward keypoint_path/frame_count/split for clips
    that already existed before this rebuild. Returns {} if no old
    manifest exists yet (first-ever run).
    """
    if not os.path.exists(OUTPUT_CSV):
        return {}

    lookup = {}
    with open(OUTPUT_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lookup[row["clip_path"]] = row
    return lookup


def build_manifest():
    old_lookup = load_old_manifest_lookup()
    carried_forward_count = 0

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

            row = {
                "clip_path": clip_path,
                "gesture_label": gesture_folder,   # trust the folder name as the label
                "person_id": person_id.lower(),
                "take_number": take_number,
                "source": "volunteer",             # change manually later for external clips
                "keypoint_path": "",
                "frame_count": "",
                "split": "",
            }

            # FIX: carry forward extraction/split state if this clip_path
            # already existed in the previous manifest. Only genuinely
            # new clip_paths stay blank, correctly signaling to
            # extract_keypoints.py / dataset_split.py that they still
            # need processing.
            if clip_path in old_lookup:
                old_row = old_lookup[clip_path]
                row["keypoint_path"] = old_row.get("keypoint_path", "")
                row["frame_count"] = old_row.get("frame_count", "")
                row["split"] = old_row.get("split", "")
                if row["keypoint_path"]:
                    carried_forward_count += 1

            rows.append(row)

    # Write manifest with the full column set (existing extraction/split
    # data for old clips preserved, new clips blank as expected)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest written: {OUTPUT_CSV}")
    print(f"Total clips found: {len(rows)}")
    print(f"Carried forward existing keypoint_path for {carried_forward_count} previously-extracted clips.")
    print(f"{len(rows) - carried_forward_count} clips are new or not yet extracted.")

    # Quick per-class, per-person summary — useful sanity check
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

    gesture_person_counts = defaultdict(lambda: defaultdict(int))

    for row in rows:
        gesture_person_counts[row["gesture_label"]][row["person_id"]] += 1

    print("\nClips per gesture per participant:")
    for gesture in sorted(gesture_person_counts):
        print(f"\n{gesture}:")
        for person in sorted(gesture_person_counts[gesture]):
            print(f"  {person}: {gesture_person_counts[gesture][person]}")

    if skipped:
        print(f"\nWARNING: {len(skipped)} files did not match the naming convention and were skipped:")
        for s in skipped:
            print(f"  - {s}")


if __name__ == "__main__":
    build_manifest()