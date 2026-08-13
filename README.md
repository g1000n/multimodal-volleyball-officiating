# Multimodal Volleyball Officiating System

An automated volleyball officiating assistant that watches a referee through a camera
and listens for their whistle, then uses that to keep score automatically. It recognizes
the referee's official hand signals in real time (using MediaPipe pose tracking and a
CNN-LSTM gesture classifier), listens for whistle blasts, and runs both through a decision
engine that follows the real FIVB officiating sequence — so a point is only added when a
recognized "team to serve" signal happens at the right point in that sequence, not just
whenever a gesture looks similar to one.

Built as a BS Computer Science thesis project at Holy Angel University.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

python main.py                # starts the system directly
```

That's the whole interface — `python main.py` with no arguments launches the live camera
system immediately. See `HOW_TO_RUN.md` for a plain-language walkthrough (including on-screen
controls) if you're running a demo for someone unfamiliar with the codebase.

For the two-iPhone camera/microphone hardware setup specifically, see `HardwareSetup.md`.


## Project Structure

```
.
├── main.py                          # interactive menu -- run this first
├── train.py                         # trains the gesture classifier
├── decision_engine.py               # sequences gestures into scoring events (FIVB rules)
├── model.py                         # CNN-LSTM architecture
├── extract_keypoints.py             # video -> MediaPipe keypoints
├── build_manifest.py                # scans data/raw_clips/ -> dataset_manifest.csv
├── dataset_split.py                 # person-based (subject-holdout) train/val/test split
├── convert_maxlsb_nothing_data.py   # converts MaxLSB/volley-judge's external dataset
├── whistle_detector.py              # real-time whistle detection (audio)
├── scoreboard_gui.py                # scoreboard display
├── live_deployment.py               # real-time deployment (camera + live inference)
├── replay_recorded_footage.py       # replays a saved recording through the same pipeline
│
├── diagnostics/                     # standalone data-quality / model-behavior checks
├── tools/                           # one-off data-fix / maintenance scripts
├── tests/                           # decision_engine unit tests, reference-clip validation
│
├── data/                            # NOT fully in git -- see Data section below
├── models/                          # active model files (final_model.pt, etc.) -- gitignored
├── model_checkpoints/                # archived model snapshot per training run -- gitignored
└── training_logs/                    # full console output per pipeline run -- gitignored
```

## Full Training Pipeline

Run in this order (or use `python main.py train`, which runs them for you):

```bash
python build_manifest.py               # scans data/raw_clips/, rebuilds the manifest
python convert_maxlsb_nothing_data.py  # re-adds pmax's converted external data
python extract_keypoints.py            # extracts MediaPipe keypoints for any new clips
python dataset_split.py                # assigns train/val/test (person-based holdout)
python train.py                        # trains the model, saves models/final_model.pt
```

**Important:** `build_manifest.py` does a full rescan of `data/raw_clips/`, which wipes
`pmax`'s converted rows (they point at external paths, not files under `raw_clips/`).
**Always run `convert_maxlsb_nothing_data.py` again immediately after `build_manifest.py`**,
before `dataset_split.py` — otherwise `pmax`'s contribution to `nothing`, `ball_out`,
`double_contact`, and `team_to_serve_left/right` silently disappears from that run.

## Live Usage

```bash
python live_deployment.py              # real-time camera + decision engine + scoreboard
python replay_recorded_footage.py <path>   # re-run a saved session through the same pipeline
```

`live_deployment.py` records two videos every session into `data/raw_recordings/`
(`raw_<timestamp>.mp4`, no overlay; `fullwindow_<timestamp>.mp4`, everything shown live) —
useful for later replaying a real session against a code change via `replay_recorded_footage.py`.

## Data

`data/raw_clips/` (original video) and `data/maxlsb_source/` (external MaxLSB source data) are
**not** tracked in git. `models/*.pt`, `models/*.pkl`, and `data/keypoints/` **are** tracked —
small enough to commit directly, and having them means you can run the system or retrain
without needing a camera, re-extracting from video, or re-downloading external data.

`data/dataset_manifest.csv` is gitignored (rebuilt by `build_manifest.py`) — run the pipeline
above to regenerate it locally.

**Attribution:** some training data for `nothing`, `ball_out`, `double_contact`, and
`team_to_serve_left/right` was supplemented from [MaxLSB/volley-judge](https://github.com/MaxLSB/volley-judge)
(MIT licensed), converted via `convert_maxlsb_nothing_data.py`. The already-converted keypoints
are included in this repo's `data/keypoints/` — you don't need to re-run the conversion or
download his source data unless extending the dataset further.

## Current Model Status

8 real gesture classes: `ball_in`, `ball_out`, `double_contact`, `end_of_set`,
`service_authorization_left`, `service_authorization_right`, `team_to_serve_left`,
`team_to_serve_right`. `nothing` is used in training (as an all-zero label) but is not a
model output class.

Internal held-out (subject-holdout) test accuracy has ranged 86–99% depending on exact
manifest composition — see `training_logs/` for the manifest snapshot + full results of
any specific run. External reference-clip accuracy (`tests/test_reference_clips.py`,
real out-of-domain footage) is meaningfully lower — a known, documented generalization gap,
not a bug.

Known open items and design decisions are tracked in the team's handoff notes — check with
the team for the latest status before assuming any particular metric is current.

## Controls (live_deployment.py)

| Key | Action |
|---|---|
| `Q` / `ESC` | Quit |
| `P` | Pause/resume (nothing processed while paused) |
| `S` | Toggle skeleton overlay |
| `W` | Manual whistle |
| `[` / `]` | Left score −1 / +1 |
| `-` / `+` | Right score −1 / +1 |
| `R` | Clear the last-attached reason for the current point |

`replay_recorded_footage.py` adds: `SPACE` pause/resume, `A`/`D` seek ±5s, `J`/`L` seek ±30s.