"""
check_unreviewed.py
Scans the full manifest and reports any clip that has no decision
(kept/flagged) in clip_review_progress.json — i.e. slipped through
review entirely. Run this AFTER merging everyone's progress, BEFORE
apply_flagged_removals.py.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import csv
import json
from collections import defaultdict

with open("data/dataset_manifest.csv") as f:
    rows = list(csv.DictReader(f))

with open("data/clip_review_progress.json") as f:
    progress = json.load(f)

reviewed = progress.get("reviewed", {})
missing = [r for r in rows if r["clip_path"] not in reviewed]

print(f"{len(missing)} / {len(rows)} clips have NO review decision yet.\n")

by_class = defaultdict(list)
for r in missing:
    by_class[r["gesture_label"]].append(r["clip_path"])

for gesture, paths in sorted(by_class.items()):
    print(f"{gesture}: {len(paths)} unreviewed")
    for p in paths:
        print(f"   {p}")