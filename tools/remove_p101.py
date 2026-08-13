
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import csv
rows = list(csv.DictReader(open("data/dataset_manifest.csv")))
kept = [r for r in rows if r["person_id"] != "p101"]
print(f"Removed {len(rows) - len(kept)} p101 rows")
with open("data/dataset_manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(kept)