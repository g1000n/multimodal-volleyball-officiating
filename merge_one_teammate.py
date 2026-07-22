"""
merge_one_teammate.py
Merges a groupmate's clip_review_progress.json into your own,
preserving BOTH people's last_index_by_filter (unlike
merge_review_progress.py, which wipes it).
"""
import json
import shutil

MY_PROGRESS = "data/clip_review_progress.json"
THEIR_PROGRESS = "data/ashley_progress.json"  # rename to whatever her file is called

# backup first, just in case
shutil.copy(MY_PROGRESS, MY_PROGRESS + ".bak")

with open(MY_PROGRESS) as f:
    mine = json.load(f)
with open(THEIR_PROGRESS) as f:
    theirs = json.load(f)

conflicts = []
for clip_path, status in theirs.get("reviewed", {}).items():
    if clip_path in mine["reviewed"] and mine["reviewed"][clip_path] != status:
        conflicts.append((clip_path, mine["reviewed"][clip_path], status))
        # flagged wins if either says flagged
        mine["reviewed"][clip_path] = "flagged" if "flagged" in (status, mine["reviewed"][clip_path]) else status
    else:
        mine["reviewed"][clip_path] = status

# bring over her resume position(s) for whatever filter(s) she was working on,
# without touching yours
for filter_key, idx in theirs.get("last_index_by_filter", {}).items():
    if filter_key not in mine["last_index_by_filter"]:
        mine["last_index_by_filter"][filter_key] = idx
    else:
        # keep whichever is further along
        mine["last_index_by_filter"][filter_key] = max(mine["last_index_by_filter"][filter_key], idx)

with open(MY_PROGRESS, "w") as f:
    json.dump(mine, f, indent=2)

print(f"Merged {len(theirs.get('reviewed', {}))} of her clips.")
print(f"{len(conflicts)} conflicts (defaulted to flagged):")
for c in conflicts:
    print(" ", c)