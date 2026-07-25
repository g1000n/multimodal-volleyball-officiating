"""
Step 1 helper: generates the random calibration sample for manual listening,
and pinpoints the match2 edge case flagged by 01_audit_json.py.

Run this AFTER 01_audit_json.py, before opening anything in VLC.

Output: processed/calibration_sample.csv -- one row per timestamp to check,
with a ready-to-paste VLC start time and blank columns for your notes.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
JSON_PATH = ROOT / "raw_data" / "volleylitics" / "whistles_all_reanchored.json"
OUT_CSV = ROOT / "processed" / "calibration_sample.csv"
N_SAMPLES = 27          # ~20-30 as requested
SEED = 42               # fixed so the sample is reproducible if you rerun this


def to_mmss(t):
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def main():
    with open(JSON_PATH) as f:
        whistles = json.load(f)

    # --- Part A: stratified random sample across match_id x type ---
    random.seed(SEED)
    by_type = defaultdict(list)
    for w in whistles:
        by_type[w.get("type", "MISSING")].append(w)

    types = list(by_type.keys())
    per_type_quota = max(1, N_SAMPLES // len(types))

    sample = []
    for t in types:
        pool = by_type[t]
        random.shuffle(pool)
        sample.extend(pool[:per_type_quota])

    # top up / trim to hit N_SAMPLES, keeping match spread reasonable
    random.shuffle(sample)
    sample = sample[:N_SAMPLES]

    rows = []
    for w in sample:
        rows.append({
            "match_id": w["match_id"],
            "type": w.get("type", "MISSING"),
            "t_anchor_sec": w["t_anchor"],
            "vlc_jump_time": to_mmss(w["t_anchor"]),
            "vlc_cli_start_time": f"{max(0, w['t_anchor'] - 1):.2f}",  # 1s before
            "onset_before_at_after": "",
            "duration_short_or_long": "",
            "single_or_double": "",
            "silence_after_sec": "",
            "notes": "",
        })

    # --- Part B: the flagged match2 edge case (<0.5s apart) ---
    match2_times = sorted(w["t_anchor"] for w in whistles if w["match_id"] == "match2")
    close_pairs = [(a, b) for a, b in zip(match2_times, match2_times[1:]) if (b - a) < 0.5]

    for a, b in close_pairs:
        rows.append({
            "match_id": "match2",
            "type": "EDGE_CASE_PAIR",
            "t_anchor_sec": a,
            "vlc_jump_time": to_mmss(max(0, a - 1)),
            "vlc_cli_start_time": f"{max(0, a - 1):.2f}",
            "onset_before_at_after": "",
            "duration_short_or_long": "",
            "single_or_double": "",
            "silence_after_sec": f"gap_to_next={b - a:.3f}s",
            "notes": "EDGE CASE: is this one long whistle logged twice, or two real rapid blasts?",
        })

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"{len(sample)} calibration samples + {len(close_pairs)} match2 edge case(s)")
    print(f"Saved -> {OUT_CSV}")
    print("\nOpen this CSV in Excel/Sheets and work through it row by row in VLC.")


if __name__ == "__main__":
    main()
