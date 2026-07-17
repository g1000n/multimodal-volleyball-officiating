"""
Step 1: Audit the reanchored whistle annotations before extracting anything.

Run this first. It doesn't touch audio yet -- it just tells you what you're
working with, so the windowing decisions in step 2 are informed rather than
guessed.

Place raw_data/volleylitics/whistles_all_reanchored.json and the 14 match
.wav files before running.
"""
import json
from collections import Counter
from pathlib import Path

JSON_PATH = Path(__file__).parent.parent / "raw_data" / "volleylitics" / "whistles_all_reanchored.json"


def main():
    with open(JSON_PATH) as f:
        whistles = json.load(f)

    print(f"Total whistle annotations: {len(whistles)}")

    types = Counter(w.get("type", "MISSING") for w in whistles)
    print("\nType values found (kept for reference -- not used for labeling yet):")
    for t, c in types.items():
        print(f"  {t}: {c}")

    counts_per_match = Counter(w["match_id"] for w in whistles)
    print("\nWhistles per match:")
    for match_id, c in sorted(counts_per_match.items()):
        print(f"  {match_id}: {c}")

    # Gap analysis per match: helps spot potential long whistles
    # (two closely-spaced anchors might actually be one long whistle logged twice,
    # or two fast consecutive short blasts -- worth listening to a few of these)
    print("\nGap analysis (anchors < 0.5s apart within the same match):")
    by_match = {}
    for w in whistles:
        by_match.setdefault(w["match_id"], []).append(w["t_anchor"])

    for match_id, times in by_match.items():
        times = sorted(times)
        close_pairs = sum(
            1 for a, b in zip(times, times[1:]) if (b - a) < 0.5
        )
        if close_pairs:
            print(f"  {match_id}: {close_pairs} pairs < 0.5s apart")

    print(
        "\nNext step: manually extract and LISTEN to ~20-30 random whistle "
        "timestamps before deciding your window size in step 2. "
        "Adjust PRE/POST padding in 02_extract_whistle_clips.py based on what you hear."
    )


if __name__ == "__main__":
    main()
