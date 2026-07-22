"""
check_class_balance.py
Shows, per gesture class, how many clips are kept vs flagged, plus a
side-by-side comparison for left/right paired classes so an imbalance
(e.g. team_to_serve_right=50 vs team_to_serve_left=130) is obvious.
Run BEFORE apply_flagged_removals.py.
"""
import csv
import json
from collections import defaultdict

with open("data/dataset_manifest.csv") as f:
    rows = list(csv.DictReader(f))

with open("data/clip_review_progress.json") as f:
    progress = json.load(f)

reviewed = progress.get("reviewed", {})

class_counts = defaultdict(lambda: {"kept": 0, "flagged": 0})
class_person_counts = defaultdict(lambda: defaultdict(lambda: {"kept": 0, "flagged": 0}))

for row in rows:
    status = reviewed.get(row["clip_path"])
    if status not in ("kept", "flagged"):
        continue
    class_counts[row["gesture_label"]][status] += 1
    class_person_counts[row["gesture_label"]][row["person_id"]][status] += 1

print(f"{'class':<30} {'kept':>6} {'flagged':>8} {'% removed':>10}")
for gesture, counts in sorted(class_counts.items()):
    kept, flagged = counts["kept"], counts["flagged"]
    total = kept + flagged
    pct_removed = (flagged / total * 100) if total else 0
    flag_note = "  <-- CHECK THIS" if pct_removed > 30 else ""
    print(f"{gesture:<30} {kept:>6} {flagged:>8} {pct_removed:>9.1f}%{flag_note}")

# --- Left/right and paired-class comparison ---
print("\n" + "=" * 60)
print("PAIRED CLASS COMPARISON (remaining clips after removal)")
print("=" * 60)

remaining = {g: c["kept"] for g, c in class_counts.items()}
seen = set()
for gesture in sorted(remaining):
    if gesture in seen:
        continue
    base = gesture.replace("_left", "").replace("_right", "")
    left_key = f"{base}_left"
    right_key = f"{base}_right"
    if left_key in remaining and right_key in remaining:
        l, r = remaining[left_key], remaining[right_key]
        diff = abs(l - r)
        pct_diff = (diff / max(l, r) * 100) if max(l, r) else 0
        flag_note = "  <-- IMBALANCED" if pct_diff > 20 else ""
        print(f"{base:<25} left={l:<6} right={r:<6} diff={diff} ({pct_diff:.0f}%){flag_note}")
        seen.add(left_key)
        seen.add(right_key)

# Any classes with no left/right pair (ball_out, double_contact, end_of_set)
unpaired = [g for g in remaining if g not in seen]
if unpaired:
    print("\nUnpaired classes:")
    for g in sorted(unpaired):
        print(f"  {g}: {remaining[g]} remaining")

print("\nPer-person breakdown for classes >30% removed:")
for gesture, counts in sorted(class_counts.items()):
    total = counts["kept"] + counts["flagged"]
    pct_removed = (counts["flagged"] / total * 100) if total else 0
    if pct_removed <= 30:
        continue
    print(f"\n  {gesture}:")
    for person, p_counts in sorted(class_person_counts[gesture].items()):
        p_total = p_counts["kept"] + p_counts