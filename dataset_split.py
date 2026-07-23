"""
dataset_split.py

Automatically assigns each clip in the manifest to "train", "val", or "test".

Core rule: split by PERSON first, not by clip. Whole people are held out
for validation/testing so the model is evaluated on someone it never
trained on. This is what proves the model generalizes to a new referee,
not just to people it already partially memorized.

Handles real-world messiness automatically:
- If a gesture class has 4+ contributing people: hold out 1 person for
  test, 1 different person for val, rest go to train.
- If a gesture class has only 2-3 people: hold out 1 person for test,
  no separate val person (val is carved from the train people's clips
  instead) — printed as a notice, not silently done.
- If a gesture class has only 1 person: true subject-holdout isn't
  possible. Falls back to a clip-level split for that class only, and
  prints a clear warning so you can mention this as a known limitation.

--------------------------------------------------------------------
FIX (this version): previously, ONE random.seed(SEED) call at the top
seeded a single shared random stream, and every class's shuffle drew
from that same stream sequentially. That meant adding/removing people
from ONE class (e.g. adding new volunteers to ball_out) changed how
many random numbers that class's shuffle consumed -- which silently
shifted the random state for every class processed AFTER it in the
loop, even though their data never changed. In practice this could
reassign which person gets held out for test/val in completely
unrelated, untouched classes, causing accuracy swings that look like a
regression in a class you didn't even touch.

Fixed by giving each class its OWN independent random.Random() instance,
seeded deterministically from (SEED, gesture_label). Adding data to one
class can now only ever affect that class's own held-out assignment --
never any other class's.
--------------------------------------------------------------------

Safe to re-run any time you add more clips — it recalculates from
scratch based on whatever is currently in the manifest.
"""

import csv
import random
from collections import defaultdict

MANIFEST_PATH = "data/dataset_manifest.csv"
SEED = 42
VAL_FRACTION_OF_TRAIN_CLIPS = 0.15   # only used in the clip-level fallback


def load_manifest():
    with open(MANIFEST_PATH, "r") as f:
        return list(csv.DictReader(f))


def save_manifest(rows, fieldnames):
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def assign_splits(rows):
    # REMOVED: random.seed(SEED) -- no longer used. Each class now gets
    # its own independent RNG below instead of sharing one global stream.

    # Group rows by gesture class -> person -> list of row indices
    class_to_people = defaultdict(lambda: defaultdict(list))
    for idx, row in enumerate(rows):
        class_to_people[row["gesture_label"]][row["person_id"]].append(idx)

    for row in rows:
        row["split"] = ""  # will fill in below

    print("=" * 60)
    # sorted() so class processing order is always the same regardless
    # of manifest row order or dict iteration order
    for gesture_label, people in sorted(class_to_people.items()):
        # FIX: independent RNG per class, seeded from the class name
        # itself. This is the actual isolation -- no class's shuffle can
        # ever consume random draws that affect another class's outcome.
        class_rng = random.Random(f"{SEED}-{gesture_label}")

        person_ids = list(people.keys())
        class_rng.shuffle(person_ids)
        num_people = len(person_ids)

        print(f"\nClass: {gesture_label}  ({num_people} contributing people)")

        if num_people >= 4:
            test_person = person_ids[0]
            val_person = person_ids[1]
            train_people = person_ids[2:]

            for idx in people[test_person]:
                rows[idx]["split"] = "test"
            for idx in people[val_person]:
                rows[idx]["split"] = "val"
            for p in train_people:
                for idx in people[p]:
                    rows[idx]["split"] = "train"

            print(f"  test person:  {test_person}")
            print(f"  val person:   {val_person}")
            print(f"  train people: {train_people}")

        elif num_people in (2, 3):
            test_person = person_ids[0]
            train_people = person_ids[1:]

            for idx in people[test_person]:
                rows[idx]["split"] = "test"

            # Carve val out of the train people's clips (clip-level),
            # since there's no spare whole person to dedicate to val.
            train_indices = []
            for p in train_people:
                train_indices.extend(people[p])
            class_rng.shuffle(train_indices)  # FIX: was random.shuffle(...)
            num_val = max(1, int(len(train_indices) * VAL_FRACTION_OF_TRAIN_CLIPS))
            val_indices = train_indices[:num_val]
            train_only_indices = train_indices[num_val:]

            for idx in val_indices:
                rows[idx]["split"] = "val"
            for idx in train_only_indices:
                rows[idx]["split"] = "train"

            print(f"  test person:  {test_person}")
            print(f"  train people: {train_people}  (val carved from their clips: {num_val} clips)")
            print(f"  NOTE: only {num_people} people for this class — no dedicated val person.")

        else:
            # Only 1 person for this class — true subject holdout impossible.
            only_person = person_ids[0]
            indices = people[only_person][:]
            class_rng.shuffle(indices)  # FIX: was random.shuffle(...)

            n = len(indices)
            n_test = max(1, int(n * 0.15))
            n_val = max(1, int(n * 0.15))

            test_indices = indices[:n_test]
            val_indices = indices[n_test:n_test + n_val]
            train_indices = indices[n_test + n_val:]

            for idx in test_indices:
                rows[idx]["split"] = "test"
            for idx in val_indices:
                rows[idx]["split"] = "val"
            for idx in train_indices:
                rows[idx]["split"] = "train"

            print(f"  WARNING: only 1 person ({only_person}) for this class.")
            print(f"  Falling back to CLIP-LEVEL split (not subject-based).")
            print(f"  This class's test accuracy will NOT prove generalization")
            print(f"  to a new person — note this as a limitation in your paper.")

    print("\n" + "=" * 60)
    return rows


def print_summary(rows):
    counts = defaultdict(lambda: defaultdict(int))
    for row in rows:
        counts[row["gesture_label"]][row["split"]] += 1

    print("\nFinal split counts per class:")
    print(f"{'class':<30} {'train':>7} {'val':>7} {'test':>7}")
    for gesture_label, split_counts in counts.items():
        print(f"{gesture_label:<30} {split_counts.get('train',0):>7} "
              f"{split_counts.get('val',0):>7} {split_counts.get('test',0):>7}")


if __name__ == "__main__":
    rows = load_manifest()
    if not rows:
        print("Manifest is empty — run build_manifest.py first.")
        exit()

    rows = assign_splits(rows)

    fieldnames = list(rows[0].keys())
    save_manifest(rows, fieldnames)

    print_summary(rows)
    print(f"\nManifest updated with 'split' column: {MANIFEST_PATH}")