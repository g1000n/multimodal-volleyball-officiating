# Whistle Detection Pipeline

Audio whistle-detection module for the Multimodal Real-Time Officiating System
(gesture recognition + whistle detection + automated scoring).

**Scope note:** whistle detection is binary (Whistle / No Whistle) only. It does
NOT classify blast duration or call type -- that distinction is not reliable from
audio alone and isn't needed, since the gesture recognition module determines what
the call means. The whistle's only job is to trigger/validate that a call happened.

## Getting the latest code (for groupmates)

```powershell
git pull origin audio
```

This now pulls all the small/structural files too -- scripts, README,
requirements.txt, and (as of this update) every CSV under `processed/`
(index files, `features.csv`, `test_set.csv`, and critically
`iphone_whistle_candidates.csv` with everyone's confirmed y/n labels). Only
the actual audio `.wav` clips and the trained model `.pkl` are excluded from
git -- see "Getting the processed audio clips" below for those.

## Environment setup (one-time)

Use **Python 3.12** -- Python 3.14 has known DLL compatibility issues with
numba/librosa on Windows.

```powershell
py -3.12 -m venv whistle_env
whistle_env\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Every new terminal session needs `whistle_env\Scripts\activate` run again first.

## Data setup (one-time)

**Volleylitics (public match audio):**
```powershell
python -m pip install -U huggingface_hub
hf download GYdevy/volleyball-whistles --repo-type dataset --local-dir raw_data/volleylitics
```
If `hf` isn't recognized as a command, try one of these instead:
```powershell
huggingface-cli download GYdevy/volleyball-whistles --repo-type dataset --local-dir raw_data/volleylitics
```
or, as a fallback that should always work:
```powershell
python -m huggingface_hub download GYdevy/volleyball-whistles --repo-type dataset --local-dir raw_data/volleylitics
```

**iPhone recordings (co-primary source, private -- not on Hugging Face):**
1. Download from this shared Drive folder:
   https://drive.google.com/drive/folders/1QsFHW6hpkjNp6-fj4D7KCkzXxZdK81Wn?usp=sharing
2. Place the folders into these exact paths:

```
raw_data/iphone_recordings/
├── positive_audio/          <- clean whistle-only session(s), .m4a or .wav
├── positive_negative_audio/ <- noisy session(s), whistles + background noise mixed in
└── negative_audio/          <- pure ambient noise, NO whistles at all
```

Do not mix whistle-containing and whistle-free audio in the same file --
`04_process_iphone_negatives.py` has no whistle-location awareness and would
mislabel whistle audio as negative if they're combined.

## Getting the processed audio clips (skip re-extraction)

`processed/clips/whistle/` and `processed/clips/non_whistle/` (the actual
`.wav` clips, as opposed to the CSVs that index them) are **not in git** --
they're large and fully regenerable, so keeping them out of version control
keeps the repo lightweight. Two ways to get them:

**Option A -- pull from Drive (fastest, recommended for groupmates):**
1. Download the `processed_clips` folder from the same shared Drive link
   above.
2. Place its contents at `processed/clips/whistle/` and
   `processed/clips/non_whistle/` respectively.
3. Since the matching index CSVs (`whistle_index.csv`,
   `match_negative_index.csv`, etc.) are already in git, you now have a
   complete `processed/` folder without running any extraction scripts.
   You can jump straight to `05_extract_features.py`.

**Option B -- regenerate from raw_data (if Drive is unavailable or you
changed extraction settings):**
Run `02` through `04c` as in "Full pipeline" below. Slower, but always
produces clips consistent with whatever's currently in the scripts.

**If you generate new/updated clips locally, please re-upload
`processed/clips/` to the same Drive folder** so the rest of the team stays
in sync -- git won't do this for you since those files are intentionally
ignored.

## Full pipeline (run in order)

```powershell
python scripts/01_audit_json.py
python scripts/09_sample_calibration_timestamps.py     # manual QC listening pass
python scripts/02_extract_whistle_clips.py              # Volleylitics whistle clips
python scripts/03_extract_match_negatives.py            # Volleylitics negative clips

# iPhone data:
python scripts/04a_detect_iphone_whistle_candidates.py    # auto-detect candidates in positive_audio/ + positive_negative_audio/
#  -> open processed/iphone_whistle_candidates.csv, confirm each row y/n in VLC (Ctrl+T to jump to vlc_jump_time), save
#  -> this file IS tracked in git now -- commit it once you've filled it in, don't let this work live only on your machine
python scripts/04b_extract_iphone_whistles.py             # extracts confirmed whistle clips -> iphone_whistle_index.csv
python scripts/04_process_iphone_negatives.py             # only reads negative_audio/ -> iphone_negative_index.csv
python scripts/04c_synthetic_hard_negatives.py             # synthetic squeak/bounce hard negatives -> synthetic_negative_index.csv

python scripts/05_extract_features.py                     # combines ALL sources -> features.csv
python scripts/06_train_model.py                          # edit TEST_MATCHES first, see below
python scripts/07_evaluate.py                              # run once, don't re-tune on this result

python scripts/08_realtime_ui.py                           # live mic test -- GUI: confidence bar, live mic energy, adjustable threshold
```

## Important settings to check before training

- **`TEST_MATCHES` appears in BOTH `05_extract_features.py` and `06_train_model.py`
  and must match in both files.** `05` uses it to decide which clips get skipped
  from data augmentation (time-shift/noise variants); `06` uses it to hold out the
  actual test split. If they don't match, augmented copies of your "held-out" data
  leak into training, quietly inflating your reported accuracy.
  Current setting (once iPhone data is included):
```python
  TEST_MATCHES = ["match9", "match7", "match13", "iphone_positive_audio"]
```
  Check `processed/features.csv` `match_id` counts to confirm exact group names
  before setting this -- they must match exactly (e.g. `iphone_positive_audio`,
  not `iphone_positive_audio_wav` or similar). Note iPhone `match_id`s come from
  the recording's filename stem, not its subfolder name.

- **`02_extract_whistle_clips.py` → `PRE`/`POST`/`TARGET_LEN`**: `0.3s` / `1.2s` /
  `1.5s`, set from manual listening QC (`processed/calibration_sample.csv`).
  `04b_extract_iphone_whistles.py` and `08_realtime_ui.py`'s `WINDOW_SEC` must
  match `TARGET_LEN` exactly -- a mismatch here previously caused a single long
  whistle to be reported as 2-3 separate triggers in real-time testing.

- **`04_process_iphone_negatives.py` → `TARGET_PER_FILE`**: currently `120`
  (increased from `30`). This is a **per-recording-file** target, not a total
  across all recordings -- if you add more negative recordings, total clip count
  scales up. Check the script's console output (`kept N clips after filtering/
  subsampling`) per file to see actual totals, since `MIN_GAP_SEC` spacing can
  cap shorter recordings below the target anyway.

## Data notes

- Audio loading uses `soundfile` + `scipy` resampling, not `librosa` -- avoids
  Windows DLL import failures (numba/soxr). Don't reintroduce `librosa.load`.
- iPhone data is co-primary alongside Volleylitics, not merely supplementary.
- Negative clips come from three sources: Volleylitics hard negatives (`03`),
  real iPhone ambient recordings (`04`), and synthetic squeak/bounce clips
  (`04c`, 300 total: 150 squeak + 150 bounce) -- synthetic clips are a
  supplementary hard-negative source, not a replacement for real court-noise
  recordings.
- Latest held-out evaluation (`match9`, `match7`, `match13`, and one held-out
  iPhone recording; 555 test samples): **98.2% accuracy**, 0.96-0.99
  precision/recall across both classes (non_whistle: 0.96 precision / 0.98
  recall; whistle: 0.99 precision / 0.98 recall). Note in the paper that
  iPhone test performance reflects close-mic recording conditions, not
  broadcast match audio -- avoid overstating this as a strict like-for-like
  improvement over the Volleylitics-only baseline.
- Real-time detection (`08_realtime_ui.py`) is intentionally binary
  (whistle / no whistle trigger only). No blast-duration or call-type
  classification is done here -- that logic was removed to match the thesis's
  stated scope; call interpretation is the gesture recognition module's job.
- Note: `06_train_model.py`'s printed "Individual Validation Scores" (SVM/RF/GB)
  use leave-one-group-out cross-validation, where some groups (e.g. purely
  synthetic or single-class recordings) can produce artificially low/undefined
  F1 scores for a single fold -- this is a known quirk of that CV setup, not a
  sign of a broken model. Trust `07_evaluate.py`'s held-out test numbers above,
  not the CV scores printed during training.

## Before committing: a quick size sanity check

Now that `processed/*.csv` files are tracked in git (not just ignored), it's
worth confirming none of them have quietly grown too large for a normal git
push -- `features.csv` in particular, since it includes the augmented
variants (6x per training clip).

```powershell
Get-ChildItem processed\*.csv | Sort-Object Length -Descending | Select-Object Name, @{N='MB';E={[math]::Round($_.Length/1MB,2)}}
```

If any single file is creeping past ~50MB, flag it before pushing -- either
it needs Git LFS, or it should join the clips/model on the ignore list and
get distributed via Drive instead. For a dataset this size it's unlikely to
be an issue, but worth a 10-second check rather than finding out from a
failed push.

## Utility scripts

- `scripts/force_rebuild.py` -- deletes `processed/features.csv`,
  `processed/test_set.csv`, and `models/whistle_svm_model.pkl`, then reruns `05`
  and `06` from scratch. Use after changing feature-extraction code to guarantee
  no stale cached features leak into a new model. Does **not** regenerate
  clip/index files -- rerun `02`-`04c` first if raw data or extraction windows
  changed.