# Whistle Detection Pipeline

Audio whistle-detection module for the Multimodal Real-Time Officiating System
(gesture recognition + whistle detection + automated scoring). This README covers
setup and the current run order after recent fixes.

## Getting the latest code (for groupmates)

```powershell
git pull origin main
```
If you get conflicts on scripts you haven't touched locally, just accept the incoming
version (`git checkout --theirs scripts/<file>.py`) and re-run `git pull`.

## Environment setup (one-time)

Use **Python 3.12** specifically — Python 3.14 has known DLL compatibility issues with
numba/librosa on Windows.

```powershell
py -3.12 -m venv whistle_env
whistle_env\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Every new terminal session needs `whistle_env\Scripts\activate` run again before any
`python`/`pip` command.

## Data setup (one-time)

```powershell
python -m pip install -U huggingface_hub
hf download GYdevy/volleyball-whistles --repo-type dataset --local-dir raw_data/volleylitics
```

For iPhone recordings: drop `.m4a`/`.wav` files into `raw_data/iphone_recordings/`.
**Keep whistle-containing recordings and pure-negative recordings as separate files**
(see Data Notes below) — do not mix whistles and negatives in the same file.

## Full pipeline (run in order)

```powershell
python scripts/01_audit_json.py
python scripts/09_sample_calibration_timestamps.py   # manual QC listening pass, see notes below
python scripts/02_extract_whistle_clips.py            # Volleylitics whistle clips
python scripts/03_extract_match_negatives.py          # Volleylitics negative clips

# iPhone data (only once recordings are ready):
python scripts/04a_detect_iphone_whistle_candidates.py  # auto-detect candidates, then confirm y/n in the CSV
python scripts/04b_extract_iphone_whistles.py           # extract confirmed whistle clips
python scripts/04_process_iphone_negatives.py           # run on NEGATIVE-ONLY recordings

python scripts/05_extract_features.py                   # combines all sources -> features.csv
python scripts/06_train_model.py                        # edit TEST_MATCHES first, see below
python scripts/07_evaluate.py                            # run once, don't re-tune on this result

python scripts/08_realtime_test.py live                  # live mic test
```

## Important settings to check before training

- **`scripts/06_train_model.py` → `TEST_MATCHES`**: must list match_ids that actually
  exist in your data. Once iPhone data is included, add at least one
  `iphone_<filename>` group here too — otherwise the reported accuracy only reflects
  Volleylitics-style audio, not real device conditions.
- **`scripts/02_extract_whistle_clips.py` → `PRE`/`POST`/`TARGET_LEN`**: currently
  `0.3s` / `1.2s` / `1.5s`, set from a manual listening QC pass (see
  `processed/calibration_sample.csv`). If you re-run QC and find different patterns,
  update these three values consistently — `04b_extract_iphone_whistles.py` and
  `08_realtime_test.py`'s `WINDOW_SEC` must match `TARGET_LEN` exactly, or you'll get
  train/inference mismatch (this caused erratic real-time detection before it was fixed).

## Data notes

- Audio loading uses `soundfile` + `scipy` resampling, not `librosa`, to avoid
  Windows DLL import failures (numba/soxr). Don't reintroduce `librosa.load` calls.
- iPhone data is a **co-primary** source alongside Volleylitics, not merely
  supplementary — see the thesis Methods chapter for the current framing.
- Whistle-containing and negative-only iPhone recordings must stay in separate files,
  since `04_process_iphone_negatives.py` has no whistle-location awareness and would
  mislabel whistle audio as negative if they're mixed.
