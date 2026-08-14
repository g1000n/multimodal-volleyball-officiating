"""
sample_validation_clips.py

Builds a stratified sample of clips for volleyball-expert dataset
validation, instead of asking anyone to review the entire dataset
(1,000+ clips).

Sampling rule: up to --per-person clips per (person, gesture_label)
combination, drawn from the manifest. Plus: every clip listed in a
"must include" source is ALWAYS included on top of that, regardless of
the per-person cap -- use this for clips already flagged during
internal review, or to guarantee specific known-important clips make
it into the sample.

Copies the actual RAW VIDEO files (not keypoints) into one output
folder, organized by class, so they're ready to hand off directly --
no need to point the expert at the full data/raw_clips/ tree.

USAGE:
    python sample_validation_clips.py
    python sample_validation_clips.py --per-person 8
    python sample_validation_clips.py --must-include flagged_pmax_clips.txt

NOTE ON ball_in SPECIFICALLY: this project has a known execution-style
split within ball_in (roughly half "official FIVB-standard" style, half
imitating an earlier referee's non-standard style) that isn't recorded
as its own column in the manifest -- person_id alone doesn't reliably
separate the two. If you have the two styles organized into separate
source folders/lists, pass BOTH as --must-include so the sample is
guaranteed to include real examples of each side of the split, e.g.:

    python sample_validation_clips.py --must-include path/to/official_style/ --must-include path/to/old_style/

--must-include accepts either a text file (one filename per line, e.g.
flagged_pmax_clips.txt) or a folder of clips -- pass it as many times
as needed.
"""

import os
import csv
import random
import shutil
import argparse

MANIFEST_PATH = "data/dataset_manifest.csv"
OUTPUT_DIR = "data/validation_sample"
DEFAULT_PER_PERSON_PER_CLASS = 10


def load_manifest_rows():
    with open(MANIFEST_PATH) as f:
        return list(csv.DictReader(f))


def load_must_include(paths):
    """Reads one or more text files (one filename per line) or plain
    directories, returns a set of basenames that must always be
    included regardless of the per-person cap."""
    must_include = set()
    for path in paths:
        if not path:
            continue
        if os.path.isdir(path):
            for name in os.listdir(path):
                must_include.add(name)
        elif os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    name = line.strip()
                    if name:
                        must_include.add(os.path.basename(name))
        else:
            print(f"  WARNING: --must-include path not found, skipping: {path}")
    return must_include


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-person", type=int, default=DEFAULT_PER_PERSON_PER_CLASS,
                         help="Max clips to sample per (person, class) combination (default: 10)")
    parser.add_argument("--must-include", action="append", default=[],
                         help="A text file (one filename per line) or a folder of clips that must "
                              "always be included on top of the random sample. Can be passed "
                              "multiple times.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed, for a reproducible sample")
    args = parser.parse_args()

    random.seed(args.seed)

    rows = load_manifest_rows()
    rows = [r for r in rows if r.get("clip_path") and os.path.exists(r["clip_path"])]
    if not rows:
        print("No rows with a valid clip_path found in the manifest.")
        print("(Note: this samples RAW VIDEO, not keypoints -- if data/raw_clips/ isn't")
        print("populated locally right now, there's nothing to sample from.)")
        return

    must_include = load_must_include(args.must_include)
    if must_include:
        print(f"Loaded {len(must_include)} must-include filenames from {len(args.must_include)} source(s).")

    groups = {}
    for r in rows:
        key = (r["gesture_label"], r["person_id"])
        groups.setdefault(key, []).append(r)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    selected_rows = []
    summary = {}

    for (gesture_label, person_id), group_rows in sorted(groups.items()):
        forced = [r for r in group_rows if os.path.basename(r["clip_path"]) in must_include]
        remaining_pool = [r for r in group_rows if r not in forced]
        random_needed = max(0, args.per_person - len(forced))
        random_sample = random.sample(remaining_pool, min(random_needed, len(remaining_pool)))

        chosen = forced + random_sample
        selected_rows.extend(chosen)
        summary[(gesture_label, person_id)] = (len(chosen), len(forced), len(group_rows))

    print(f"\n{'class':<30} {'person':<8} {'sampled':>8} {'forced':>8} {'total avail':>12}")
    for (gesture_label, person_id), (chosen_count, forced_count, total_count) in summary.items():
        print(f"{gesture_label:<30} {person_id:<8} {chosen_count:>8} {forced_count:>8} {total_count:>12}")

    print(f"\nTotal clips selected: {len(selected_rows)} (out of {len(rows)} total available)")

    for r in selected_rows:
        class_dir = os.path.join(OUTPUT_DIR, r["gesture_label"])
        os.makedirs(class_dir, exist_ok=True)
        dest = os.path.join(class_dir, os.path.basename(r["clip_path"]))
        if not os.path.exists(dest):
            shutil.copy(r["clip_path"], dest)

    print(f"\nDone. Sample copied to: {OUTPUT_DIR}/ (organized by class folder)")
    print("Hand this folder to the volleyball expert -- no need to touch the full dataset.")


if __name__ == "__main__":
    main()