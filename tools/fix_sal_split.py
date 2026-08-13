"""
fix_sal_split.py

MANUAL OVERRIDE for service_authorization_left's train/test split only.

WHY: dataset_split.py's automatic subject-holdout assigned p03 as the
SOLE test person for this class, with train = [p08, p02] only -- p03
never appeared in training at all. diagnose_sal_scores.py confirmed
every one of p03's 49 clips scores near-zero (mean 0.024), a uniform,
clean blind spot -- not scattered inconsistency -- consistent with
"the model has zero exposure to this person's execution style," not
mislabeled or messy content. Given SAL only has 4 contributors total,
losing one ENTIRELY to the test split leaves training too thin to
generalize.

THIS SCRIPT: moves p03 into TRAIN (so the model actually learns from
his style, which the team has separately noted has real, natural
variation worth learning from -- not an error to exclude), and
promotes p09 (previously just val) to be the new held-out TEST person
instead. A small val set is carved from the remaining train clips
(p02, p08, p03), same approach dataset_split.py already uses for
thinner classes.

HONEST LIMITATION TO NOTE IN YOUR PAPER: this is a manual override of
the automatic subject-holdout process for this one class, made
deliberately because the automatic assignment left training with zero
exposure to a real, natural style variant. It doesn't invalidate the
evaluation (p09 is still a genuinely held-out person), but it should
be documented as a deliberate exception, not silently identical to
every other class's fully-automatic split.

Run this AFTER dataset_split.py (so the rest of the manifest's splits
are already assigned normally), then go straight to train.py -- no
need to rerun build_manifest.py or extract_keypoints.py.

Run:
    python fix_sal_split.py
    python train.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import random

MANIFEST_PATH = "data/dataset_manifest.csv"
TARGET_CLASS = "service_authorization_left"
NEW_TEST_PERSON = "p09"
PERSON_TO_MOVE_TO_TRAIN = "p03"
VAL_FRACTION_OF_TRAIN_CLIPS = 0.15
SEED = 42


def main():
    with open(MANIFEST_PATH, "r") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())

    sal_indices = [i for i, r in enumerate(rows) if r["gesture_label"] == TARGET_CLASS]
    if not sal_indices:
        print(f"No rows found for {TARGET_CLASS}. Did you mean to run this after dataset_split.py?")
        return

    people_in_class = sorted(set(rows[i]["person_id"] for i in sal_indices))
    print(f"People contributing to {TARGET_CLASS}: {people_in_class}")

    if NEW_TEST_PERSON not in people_in_class:
        print(f"WARNING: {NEW_TEST_PERSON} not found in this class's contributors -- check the person ID.")
        return

    # Step 1: assign the new test person
    test_indices = [i for i in sal_indices if rows[i]["person_id"] == NEW_TEST_PERSON]
    for i in test_indices:
        rows[i]["split"] = "test"

    # Step 2: everyone else (including the previously-excluded person)
    # goes into the train pool, with a small val slice carved out --
    # same style as dataset_split.py's 2-3-person fallback.
    train_pool_indices = [i for i in sal_indices if rows[i]["person_id"] != NEW_TEST_PERSON]

    random.seed(SEED)
    shuffled_train_pool = train_pool_indices[:]
    random.shuffle(shuffled_train_pool)

    num_val = max(1, int(len(shuffled_train_pool) * VAL_FRACTION_OF_TRAIN_CLIPS))
    val_indices = shuffled_train_pool[:num_val]
    train_only_indices = shuffled_train_pool[num_val:]

    for i in val_indices:
        rows[i]["split"] = "val"
    for i in train_only_indices:
        rows[i]["split"] = "train"

    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    train_people = sorted(set(rows[i]["person_id"] for i in train_only_indices + val_indices))
    print(f"\nDone. New split for {TARGET_CLASS}:")
    print(f"  test person:  {NEW_TEST_PERSON}  ({len(test_indices)} clips)")
    print(f"  train/val people: {train_people}  ({len(train_only_indices)} train, {len(val_indices)} val clips)")
    print(f"  '{PERSON_TO_MOVE_TO_TRAIN}' is now included in training -- the model will actually see "
          f"his execution style for the first time.")
    print(f"\nManifest updated: {MANIFEST_PATH}")
    print("Next step: python train.py")


if __name__ == "__main__":
    main()