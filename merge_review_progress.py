"""
merge_review_progress.py

Combines multiple teammates' clip_review_progress.json files into one.
Each person reviews a different slice of clips locally (progress files
are gitignored on purpose, so they have to be shared manually — Discord,
Drive, etc.). This script merges them into a single file you can use
with apply_flagged_removals.py.

HOW TO USE:
1. Ask each teammate for their data/clip_review_progress.json file.
2. Rename each one distinctly (e.g. gion_progress.json, anya_progress.json,
   anouc_progress.json) and put them all in one folder — by default,
   this script looks in: data/team_review_jsons/
3. Run: python merge_review_progress.py
4. It merges everyone's "reviewed" decisions into one combined file,
   printing any conflicts (same clip marked differently by two people)
   along the way so you can manually double-check those specific clips.
5. The merged result is saved to data/clip_review_progress.json — this
   is what apply_flagged_removals.py reads from.

CONFLICT RULE: if two people disagree on the same clip (one said keep,
another said flag), this defaults to FLAG — safer to double-check a
clip than to silently keep one that someone found a real problem with.
Every conflict is printed so you can review that decision if you want
to override it.
"""

import os
import json
import glob

TEAM_JSONS_DIR = "data/team_review_jsons"
OUTPUT_PATH = "data/clip_review_progress.json"


def main():
    json_files = glob.glob(os.path.join(TEAM_JSONS_DIR, "*.json"))

    if not json_files:
        print(f"No JSON files found in {TEAM_JSONS_DIR}/")
        print("Put your teammates' clip_review_progress.json files there first")
        print("(rename each one distinctly, e.g. gion_progress.json, anya_progress.json).")
        return

    print(f"Found {len(json_files)} file(s) to merge:")
    for f in json_files:
        print(f"  {f}")

    merged_reviewed = {}
    conflicts = []

    for filepath in json_files:
        with open(filepath, "r") as f:
            data = json.load(f)

        reviewed = data.get("reviewed", {})
        for clip_path, status in reviewed.items():
            if clip_path not in merged_reviewed:
                merged_reviewed[clip_path] = status
            elif merged_reviewed[clip_path] != status:
                conflicts.append((clip_path, merged_reviewed[clip_path], status, filepath))
                # Conflict rule: flagged wins (safer default)
                if status == "flagged" or merged_reviewed[clip_path] == "flagged":
                    merged_reviewed[clip_path] = "flagged"
                else:
                    merged_reviewed[clip_path] = status  # last one wins if neither is "flagged"

    if conflicts:
        print(f"\n{len(conflicts)} conflict(s) found (defaulted to 'flagged' where applicable):")
        for clip_path, existing, new, source_file in conflicts:
            print(f"  {clip_path}")
            print(f"    existing: {existing}  |  from {source_file}: {new}")
    else:
        print("\nNo conflicts — everyone agreed on every clip they both reviewed.")

    kept_count = sum(1 for v in merged_reviewed.values() if v == "kept")
    flagged_count = sum(1 for v in merged_reviewed.values() if v == "flagged")
    print(f"\nMerged total: {kept_count} kept, {flagged_count} flagged, "
          f"{len(merged_reviewed)} clips reviewed overall.")

    if os.path.exists(OUTPUT_PATH):
        confirm = input(f"\n{OUTPUT_PATH} already exists. Overwrite with merged result? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Cancelled. Nothing was written.")
            return

    merged_output = {
        "reviewed": merged_reviewed,
        "last_index_by_filter": {},  # not meaningful after merging, left empty
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(merged_output, f, indent=2)

    print(f"\nMerged progress written to {OUTPUT_PATH}")
    print("Ready to run apply_flagged_removals.py")


if __name__ == "__main__":
    main()