"""
Step 4c: Generate synthetic hard-negative court sounds (shoe squeaks, ball
bounces) and wire them directly into the real pipeline.

Unlike an earlier version of this idea, this writes straight into
processed/clips/non_whistle/ and produces processed/synthetic_negative_index.csv
in the SAME format as every other index file, so 05_extract_features.py picks
it up automatically. These are synthetic approximations, not real recordings --
useful as a supplementary hard-negative source, not a replacement for real
court-noise recordings.
"""
from pathlib import Path

import numpy as np
import soundfile as sf
import pandas as pd

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "processed" / "clips" / "non_whistle"

SR = 22050
DURATION = 1.5   # matches TARGET_LEN/SEG_LEN/WINDOW_SEC used everywhere else
SAMPLES = int(SR * DURATION)
N_PER_TYPE = 150  # Increased to 150 squeaks + 150 bounces (300 total samples)


def generate_shoe_squeak():
    t = np.linspace(0, DURATION, SAMPLES)
    freq_sweep = 2500 + 1700 * np.sin(2 * np.pi * 15 * t) + np.random.normal(0, 200, SAMPLES)
    phase = 2 * np.pi * np.cumsum(freq_sweep) / SR
    signal = 0.5 * np.sin(phase)
    envelope = np.exp(-t * 6)
    return (signal * envelope).astype(np.float32)


def generate_ball_bounce():
    t = np.linspace(0, DURATION, SAMPLES)
    freq = 180 * np.exp(-t * 10)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    signal = 0.8 * np.sin(phase) + 0.2 * np.random.normal(0, 1, SAMPLES)
    envelope = np.exp(-t * 12)
    return (signal * envelope).astype(np.float32)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_rows = []

    print(f"Generating {N_PER_TYPE} squeak + {N_PER_TYPE} bounce synthetic negatives...")
    for i in range(N_PER_TYPE):
        squeak = generate_shoe_squeak()
        fname = f"synthetic_squeak_{i:03d}.wav"
        sf.write(OUT_DIR / fname, squeak, SR)
        index_rows.append({
            "match_id": "synthetic_court_noise",
            "filename": fname,
            "start_time": 0.0,
            "label": 0,
            "source": "synthetic_squeak",
        })

        bounce = generate_ball_bounce()
        fname = f"synthetic_bounce_{i:03d}.wav"
        sf.write(OUT_DIR / fname, bounce, SR)
        index_rows.append({
            "match_id": "synthetic_court_noise",
            "filename": fname,
            "start_time": 0.0,
            "label": 0,
            "source": "synthetic_bounce",
        })

    df = pd.DataFrame(index_rows)
    out_csv = ROOT / "processed" / "synthetic_negative_index.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} synthetic negative clips -> {out_csv}")


if __name__ == "__main__":
    main()