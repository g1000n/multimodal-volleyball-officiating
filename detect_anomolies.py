"""
detect_anomalies.py

For each gesture class, computes an "average" (centroid) version of the
clip across all contributing clips, then measures how far each individual
clip is from that average. Clips far from the norm are flagged — these
are your most likely mislabeled or badly-executed clips, worth manually
reviewing against reference footage.

Also breaks the distances down by person, so you can see if one specific
person's clips are consistently unusual within a class (possible
execution issue) versus scattered outliers (possible one-off mislabeling
or sloppy takes).

Run this AFTER extract_keypoints.py (needs the .npy keypoint files).
Does not modify anything — read-only diagnostic.
"""

import csv
import numpy as np
from collections import defaultdict

from train import normalize_sequence, resample_sequence, SEQUENCE_LENGTH

MANIFEST_PATH = "data/dataset_manifest.csv"
TOP_N_OUTLIERS_PER_CLASS = 5  # how many worst offenders to print per class


def load_manifest_rows():
    with open(MANIFEST_PATH, "r") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("keypoint_path") and r["keypoint_path"] != ""]
    return rows


def get_clip_vector(keypoint_path):
    raw = np.load(keypoint_path)
    normalized = normalize_sequence(raw)
    resampled = resample_sequence(normalized, SEQUENCE_LENGTH)
    return resampled.flatten()  # single flat vector per clip, easy to compare


def main():
    rows = load_manifest_rows()

    class_to_rows = defaultdict(list)
    for row in rows:
        class_to_rows[row["gesture_label"]].append(row)

    print("=" * 70)
    print("ANOMALY DETECTION — clips farthest from their class's average pattern")
    print("=" * 70)

    for gesture_label, class_rows in sorted(class_to_rows.items()):
        vectors = []
        valid_rows = []
        for row in class_rows:
            try:
                vec = get_clip_vector(row["keypoint_path"])
                vectors.append(vec)
                valid_rows.append(row)
            except Exception as e:
                print(f"  Could not process {row['clip_path']}: {e}")

        if len(vectors) < 3:
            print(f"\n{gesture_label}: too few clips ({len(vectors)}) to compute meaningful anomalies, skipping.")
            continue

        vectors = np.array(vectors)
        centroid = vectors.mean(axis=0)

        distances = np.linalg.norm(vectors - centroid, axis=1)

        mean_dist = distances.mean()
        std_dist = distances.std()

        print(f"\n{gesture_label}  ({len(valid_rows)} clips)")
        print(f"  mean distance to class average: {mean_dist:.3f}  (std: {std_dist:.3f})")

        # Sort clips by distance, descending — worst offenders first
        order = np.argsort(distances)[::-1]
        print(f"  Top {TOP_N_OUTLIERS_PER_CLASS} outliers (furthest from the norm):")
        for idx in order[:TOP_N_OUTLIERS_PER_CLASS]:
            row = valid_rows[idx]
            dist = distances[idx]
            z_score = (dist - mean_dist) / std_dist if std_dist > 0 else 0
            flag = "  <-- FLAG (>2 std from average)" if z_score > 2 else ""
            print(f"    {row['clip_path']:<55} person={row['person_id']:<5} "
                  f"dist={dist:.3f}  z={z_score:+.2f}{flag}")

        # Per-person average distance — flags a person whose clips are
        # consistently unusual within this class (possible execution
        # difference, not just one bad take)
        person_distances = defaultdict(list)
        for idx, row in enumerate(valid_rows):
            person_distances[row["person_id"]].append(distances[idx])

        print(f"  Per-person average distance (higher = more consistently unusual):")
        person_avgs = [(p, np.mean(d)) for p, d in person_distances.items()]
        person_avgs.sort(key=lambda x: -x[1])
        for person, avg_dist in person_avgs:
            print(f"    {person:<6} avg_dist={avg_dist:.3f}  ({len(person_distances[person])} clips)")

    print("\n" + "=" * 70)
    print("Done. Clips flagged above are worth manually reviewing against")
    print("reference footage — they may be mislabeled, sloppy takes, or")
    print("genuinely different (but valid) execution styles.")
    print("=" * 70)


if __name__ == "__main__":
    main()