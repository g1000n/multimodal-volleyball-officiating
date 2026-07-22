"""
Step 4: Process long (5-10 min) iPhone voice memo recordings into usable
negative clips.

Pipeline: convert -> segment (non-overlapping) -> filter near-silence ->
subsample with minimum time spacing so selected clips aren't clustered.

Target: ~100-150 clips total across all your recordings (diversity matters
more than volume here -- this is a supplementary domain, not the primary
negative source).
"""
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import ffmpeg
import pandas as pd

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "raw_data" / "iphone_recordings"
OUT_DIR = ROOT / "processed" / "clips" / "non_whistle"

SR = 22050            # downsample to match Volleylitics
WINDOW_SEC = 1.0        # match your whistle clip length
SILENCE_PERCENTILE = 20  # drop the quietest 20% of segments
MIN_GAP_SEC = 5.0        # minimum spacing between selected clips
TARGET_PER_FILE = 30     # adjust based on how many recordings you have --
                          # exact split vs. online negatives not decided yet,
                          # revisit once iPhone recordings are ready


def convert_to_wav(src_path, dst_path, sr=SR):
    (
        ffmpeg.input(str(src_path))
        .output(str(dst_path), ar=sr, ac=1)
        .run(overwrite_output=True, quiet=True)
    )


def segment_long_recording(y, sr, window_sec=WINDOW_SEC):
    win_len = int(window_sec * sr)
    segments = []
    for start in range(0, len(y) - win_len, win_len):
        segments.append((start / sr, y[start:start + win_len]))
    return segments


def energy_filter(segments, percentile_cutoff=SILENCE_PERCENTILE):
    energies = [np.sqrt(np.mean(clip ** 2)) for _, clip in segments]
    threshold = np.percentile(energies, percentile_cutoff)
    return [(t, clip) for (t, clip), e in zip(segments, energies) if e > threshold]


def subsample_spaced(segments, n_target, min_gap_sec=MIN_GAP_SEC):
    indices = list(range(len(segments)))
    np.random.shuffle(indices)

    chosen = []
    chosen_times = []
    for idx in indices:
        t = segments[idx][0]
        if all(abs(t - ct) >= min_gap_sec for ct in chosen_times):
            chosen.append(segments[idx])
            chosen_times.append(t)
        if len(chosen) >= n_target:
            break
    return chosen


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_rows = []

    source_files = list(RAW_DIR.glob("*.m4a")) + list(RAW_DIR.glob("*.wav"))
    if not source_files:
        print(f"No recordings found in {RAW_DIR}. Add your .m4a/.wav files there first.")
        return

    for src in source_files:
        print(f"Processing {src.name}...")

        if src.suffix == ".m4a":
            tmp_wav = ROOT / "raw_data" / "iphone_recordings" / f"{src.stem}_converted.wav"
            convert_to_wav(src, tmp_wav)
            load_path = tmp_wav
        else:
            load_path = src

        y, sr = librosa.load(load_path, sr=SR)

        segments = segment_long_recording(y, sr)
        segments = energy_filter(segments)
        chosen = subsample_spaced(segments, TARGET_PER_FILE)

        for i, (t, clip) in enumerate(chosen):
            fname = f"iphone_{src.stem}_{i}.wav"
            out_path = OUT_DIR / fname
            sf.write(out_path, clip, sr)

            index_rows.append({
                "match_id": f"iphone_{src.stem}",  # treat each recording as its own "group"
                "filename": fname,
                "start_time": t,
                "label": 0,
                "source": "iphone_negative",
            })

        print(f"  -> kept {len(chosen)} clips after filtering/subsampling")

    df = pd.DataFrame(index_rows)
    out_csv = ROOT / "processed" / "iphone_negative_index.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nExtracted {len(df)} iPhone negative clips -> {out_csv}")


if __name__ == "__main__":
    main()
