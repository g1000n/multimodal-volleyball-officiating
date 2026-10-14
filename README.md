# Multimodal Officiating Training Tool for Volleyball

A training tool for volleyball officials: a trainee practises the official FIVB referee hand signals and whistle
cues in front of a camera and gets graded instantly. It recognises the signal (MediaPipe pose and hand tracking +
a CNN-LSTM gesture classifier), checks the form against an FIVB-derived checklist, listens for the whistle, and
keeps a simulated scoreboard.

Built as a BS Computer Science thesis project (team NaviTalamas), School of Computing, Holy Angel University.

The project started as a **live officiating system** that kept score for a real referee during a match. After the
final defense it became a **training tool** built on the same models. Both are in this repository; the training
tool is the main program. **`docs/HANDOFF.md` is the full, current description** of the system, every design
decision with its evidence, and the evaluation.

## Download

| What | Link |
|---|---|
| Source code (this repository) | https://github.com/g1000n/multimodal-volleyball-officiating |
| Release v1.1 (latest: also starts without a camera, showing "No camera detected") | https://github.com/g1000n/multimodal-volleyball-officiating/releases/tag/v1.1 |
| Release v1.0 (the version used in the final evaluation, as a zip) | https://github.com/g1000n/multimodal-volleyball-officiating/releases/tag/v1.0 |
| Docker image | https://hub.docker.com/r/g1000n/multimodal-volleyball-officiating |

## Quick start (Windows, with camera and microphone)

Python **3.11** (the project venv uses 3.11.9):

```bash
git clone https://github.com/g1000n/multimodal-volleyball-officiating
cd multimodal-volleyball-officiating
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

`python main.py` opens the training tool. On first start it shows the welcome screen and the recording notice;
then choose the camera and microphone under **Camera and mic** (the whistle test must report a whistle).

The pinned versions in `requirements.txt` matter: the whistle model needs **numpy 2.2.6**. With an older numpy
the whistle detector stops with "is not a known BitGenerator module".

For the iPhone (Camo) camera and microphone setup, see `docs/HardwareSetup.md`. The evaluation used an iPhone 13
through Camo Studio at 2 to 2.5 m, camera at chest height (1.2 to 1.4 m), head to hips in frame.

## Docker (any computer, no Python install)

```bash
docker pull g1000n/multimodal-volleyball-officiating:1.1
```

or build it from this repository with `docker compose build` and run the automated tests with
`docker compose run --rm tests` (expect 35 `OK` lines, `Smoke test passed.`, and `All good.`). The image contains the
code and the trained models, no dataset and no participant data. On Windows and macOS, Docker cannot use the camera
or microphone, so use the Quick start above for the live tool; inside Docker the tool opens, says "No camera detected" and still shows the menu, Learn the signals and the progress screens. Full instructions: `docker.md`.

## The training tool

### Six modes

| Mode | What the trainee does |
|---|---|
| **Practice** | Any signal; the tool names what it sees and shows a live checklist. No score |
| **Drill** | One chosen signal, 1 (free practice with retries), 3, 5 or 10 repetitions |
| **Combo drill** | The end of a rally: Team to Serve, then the reason (Ball In, Ball Out, Double Contact) |
| **Challenge** | Every signal once, in random order |
| **Match simulation** | A narrated set: Service Authorization, then Team to Serve + reason per rally, with a two-team scoreboard |
| **Match testing** | Continuous play with no set order; every signal is detected, graded and scored on a live scoreboard |

Eight signals are recognised: Team to Serve (left / right), Service Authorization (left / right), Ball In,
Ball Out, Double Contact, End of Set.

### Grading

Every attempt is graded by `gesture_grader.py` on an FIVB-derived rubric: Recognition 38, Distinctness 10, Hold 10,
Form 40, Ready position 2 (out of 100), with a checklist of form checks and a tip for each failed one. The verdict
is CORRECT, ALMOST or INCORRECT; a NO READING means the trainee could not be seen and the attempt is not counted.

| Level | CORRECT from | ALMOST from |
|---|---|---|
| Beginner | 60 | 40 |
| Standard | 75 | 55 |
| Referee | 100 | 68 |

### Whistle

**Require the whistle** (menu option) adds the whistle to Drill, Combo and Challenge, and makes Match simulation and
Match testing count a point or a serve authorisation only with a whistle: those two modes use the live system's
`decision_engine.py` (a whistle up to 10 s before the signal, or up to 6 s after it). In Drill, Combo and Challenge
the whistle step listens during the capture (1.5 s longer) and up to 1.5 s after the signal, and is graded as heard
(10 points) or not heard (0); it never changes the gesture's verdict. W is a manual whistle when no microphone is
available.

The training tool's whistle threshold is **0.55** (`trainer_config.py`); the live system keeps **0.70**. On
2026-10-04, replaying the detector's logs of 51 clean whistles from a phone microphone at 1 to 2 m, 0.70 confirmed
8 of 51 and 0.55 confirmed 45 of 51 (most whistles scored 0.55 to 0.68). The whistle model itself is unchanged.

In a pair, the reason only counts after its Team to Serve (the FIVB order, as in the live system), and a capture
ends early only after a signal was seen in 3 model windows in a row.

### Keys

| Key | Action |
|---|---|
| `SPACE` | Start / next |
| `Q` / `ESC` | End the session (the summary opens) |
| `P` | Pause / resume; an attempt paused mid-way is dropped and repeated |
| `R` | Retry (1-repetition Drill); on the summary, do the same session again |
| `D` / `A` | Next / previous attempt on the summary |
| `H` | Hint: show the required signal (Match simulation) |
| `W` | Manual whistle |
| `Up` / `Down`, `PgUp` / `PgDn`, mouse wheel | Scroll the signal card on the right |
| `M` / `T` | Mirror the camera view / light or dark theme |
| `[` `]` / `-` `+` | Match testing: left / right score −1 / +1 |

### What every session saves

`data/trainer_sessions/<name or code>/<date_time>_<mode>/`:

| File | Contents |
|---|---|
| `attempts.csv` | One row per graded attempt: signal, what the model saw, verdict, score, failed checks, feedback time, the on-screen attempt number, and the same attempt graded at all three levels |
| `session_log.csv` | Every event, including NO READING attempts and pauses |
| `whistle_events.csv` | Every whistle detected, with its time |
| `summary.json` | Totals and performance (fps, CPU, RAM, feedback time) |
| `attempts/*.npz` | The movement data of each attempt, for re-grading |
| `report.html`, `session.mp4` | The session report and the training screen recording |

**Progress and sessions** on the menu shows the history, reports and videos; **Delete my data** removes a trainee's
data.

## Evaluation tools

| Command | Purpose |
|---|---|
| `py tools/performance_report.py --since 20261007_090000` | Speed and load per mode: fps, first vs last quarter, slowest frame, verdicts within 3 s, CPU, RAM |
| `py tools/regrade_attempts.py --level standard` | Re-grade saved attempts with the current rules (evidence for any grading change) |
| `py tools/pilot_check.py --since <time>` | Pass / fail table for a dry run |
| `py tools/collect_data.py --name <researcher>` | Pack one laptop's participant data (codes `G-…`, no videos) into a zip for the team folder |
| `py tools/compile_results.py --sheet <records.xlsx> --root <folder> …` | The evaluation's results from every laptop's logs + the records sheet: recognition per signal, NO READING, consistency, feedback time, whistle detection, help, crashes, and a list of problems to fix first |

## The live officiating system

```bash
python main.py live                        # live_deployment.py: camera + decision engine + scoreboard
python main.py replay <path>               # replay a saved session through the same pipeline
```

It recognises the referee's signals, listens for the whistle, and runs both through `decision_engine.py`, which
follows the FIVB sequence: a point is added only when a recognised Team to Serve comes with its whistle at the right
moment. It records `raw_<timestamp>.mp4` and `fullwindow_<timestamp>.mp4` per session into `data/raw_recordings/`.
Fixes made in the training tool are not automatically in the live system, and the other way round.

| Key (live system) | Action |
|---|---|
| `Q` / `ESC` | Quit |
| `P` | Pause / resume |
| `S` | Toggle skeleton overlay |
| `W` | Manual whistle |
| `[` / `]`, `-` / `+` | Left / right score −1 / +1 |
| `R` | Clear the last-attached reason for the current point |

## Project structure

```
.
├── main.py                 # front door: training tool by default; live, replay, train, menu as subcommands
├── trainer.py              # the training tool (modes, capture, scoring, reports)
├── trainer_ui.py           # welcome / consent, menu, Learn the signals, progress screens
├── gesture_grader.py       # FIVB-derived rubric, checklists, levels, verdicts
├── devices.py, device_setup.py   # camera and microphone choice and test
├── progress.py             # "My progress" from saved attempts
├── trainer_config.py       # settings (whistle threshold, TEST_MODE, overrides)
├── whistle_detector.py     # real-time whistle detection (audio)
├── decision_engine.py      # FIVB sequencing of whistle + signals into points (live system, Simulation, Match testing)
├── live_deployment.py      # the original live officiating system
├── replay_recorded_footage.py
├── model.py, train.py, extract_keypoints.py, build_manifest.py, dataset_split.py   # the gesture model
├── assets/signals/         # signal pictures for Learn the signals and the drill card
├── tools/                  # evaluation and maintenance scripts (see above)
├── tests/                  # test_trainer_smoke.py, test_gesture_grader.py, decision-engine tests
├── docs/                   # HANDOFF.md (start here), HardwareSetup.md, whistle pipeline notes
├── models/                 # final_model.pt (gestures), whistle_svm_model.pkl (whistle)
├── Dockerfile, docker-compose.yml, .dockerignore, docker.md   # the Docker image and how to use it
└── data/                   # trainer_sessions/, keypoints/, whistle_logs/ (raw video not in git)
```

## Tests

Run before every commit, with the project venv:

```bash
.venv\Scripts\python.exe tests\test_trainer_smoke.py      # expect 35 "OK" lines, "Smoke test passed."
.venv\Scripts\python.exe tests\test_gesture_grader.py     # expect all PASS, "All good."
```

Keep `TEST_MODE = False` in `trainer_config.py` when committing.

## Training the gesture model

Run in this order (or `python main.py train`):

```bash
python build_manifest.py               # scans data/raw_clips/, rebuilds the manifest
python convert_maxlsb_nothing_data.py  # re-adds the converted external data
python extract_keypoints.py            # MediaPipe keypoints for any new clips
python dataset_split.py                # person-based (subject-holdout) train/val/test split
python train.py                        # trains the model, saves models/final_model.pt
```

`build_manifest.py` does a full rescan of `data/raw_clips/`, which removes the converted external rows. **Always
run `convert_maxlsb_nothing_data.py` again right after `build_manifest.py`**, before `dataset_split.py`.

The training tool did not retrain the model: it uses the same `models/final_model.pt` as the live system.

## Data and attribution

The trained models (`models/*.pt`, `models/*.pkl`) are in git, so the system runs without any dataset. The dataset
itself is not: `data/raw_clips/` (original video), `data/maxlsb_source/` and `data/keypoints/` (the extracted
keypoints, about 280 MB) are kept by the team, and retraining needs them. `data/dataset_manifest.csv` is rebuilt
by `build_manifest.py`. Participants' session data (`data/trainer_sessions/`) stays on the laptops and the team's
private storage; it is not committed.

Some training data for `nothing`, `ball_out`, `double_contact` and `team_to_serve_left/right` was supplemented from
[MaxLSB/volley-judge](https://github.com/MaxLSB/volley-judge) (MIT licensed), converted via
`convert_maxlsb_nothing_data.py` into `data/keypoints/`.

## Model status

Eight gesture classes: `ball_in`, `ball_out`, `double_contact`, `end_of_set`, `service_authorization_left`,
`service_authorization_right`, `team_to_serve_left`, `team_to_serve_right`. `nothing` is used in training but is not
an output class.

| Model | Test set | Result |
|---|---|---|
| Gesture (CNN-LSTM), `models/final_model.pt` = checkpoint `2026-08-11_235000` | 617 held-out clips (subject-holdout split) | **92.54%** accuracy; weakest: Ball In recall 0.57 (37 of 100 read as Team to Serve left) |
| Whistle (SVM + Random Forest + HistGradientBoosting, 92 acoustic features) | 435 held-out samples, threshold 0.30 (offline) | **98.16%** accuracy (non-whistle 117/120, whistle 310/315) |

`training_logs/train_2026-08-11_235000.log` holds the gesture model's test output, and `reproduce_table5_eval.py`
reprints it from the saved model (it needs the team's `data/keypoints/` and manifest). Accuracy on out-of-domain reference clips (`tests/test_reference_clips.py`) is
lower, a documented generalisation gap. See `docs/HANDOFF.md` for the known limitations.

## Evaluation results (training tool, October 2026)

Eleven general participants (codes G-01 to G-04, G-11, G-12, G-13 to G-17) each did a guided session at Standard
difficulty on an iPhone 13 through Camo Studio:

| Measure | Result |
|---|---|
| Gesture recognition, Drill + Challenge | 178/198 = **89.9%** |
| Combo / Match simulation signal recognition | 81.8% / 75.5% |
| Whistle detection, Combo + Match simulation | 95/110 = **86.4%**, 0 false alarms |
| Verdict within 3 s of the arms coming down | 358/360 = 99.4% (Drill: 102/102) |
| Crashes or freezes | 0 |
| ISO/IEC 25010:2023 questionnaire | participants 4.64, experts 4.52 (both Strongly Agree) |

The results are compiled with `tools/compile_results.py` from the session logs; participant data is not in this
repository.
