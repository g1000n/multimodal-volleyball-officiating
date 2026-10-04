# Handoff — Multimodal Officiating Training Tool for Volleyball

Team NaviTalamas, BS Computer Science thesis, Holy Angel University. Written 2026-10-04, the day before the
system evaluation. This is the single document to read before touching the code, the evaluation, or the parts of
the manuscript that describe what the system does. It is written for teammates **and** for an AI assistant
(Claude Code reads `CLAUDE.md` at the repo root, which points here).

Contents

1. What this system is (and the pivot)
2. How to run it
3. Files and what each one does
4. The six modes, exactly as they behave now
5. Grading
6. Whistle detection
7. What every session saves
8. Tools and tests
9. Decisions made, with the evidence (cite these in the paper)
10. Evaluation: forms, procedure, what to record
11. Manuscript alignment: what the paper must say
12. Known limitations and open items
13. Rules for changing the code

---

## 1. What this system is

The thesis originally built a **live officiating system** (`live_deployment.py` + `decision_engine.py`): a camera
across the court recognised a real referee's hand signals and whistle and kept the score of a real match. After
the final defense a panelist required a pivot to a **training tool**: a trainee practises the signals alone and
is graded. The panelist's requests were:

- a welcome screen for the trainee, since the focus is now teaching how to officiate;
- a "training tool" setup instead of an official match;
- choose a gesture and be told whether it was performed properly or not (plus a practice mode that detects
  whatever gesture is performed);
- scoring tied to how well the trainee performs the gestures;
- a GDPR-style agreement that sessions are filmed and saved.

The recognition models were **not retrained** for the pivot. The training tool (`trainer.py` and friends) is a new
layer on top of the same gesture model (CNN-LSTM) and whistle model (SVM / Random Forest / HistGradientBoosting
soft-voting ensemble). It reuses `decision_engine.py` only for whistle-gated scoring in Match Simulation and Match
Testing.

**Both systems still exist in the repo. A fix in one is never automatically in the other.** In particular the
training tool deliberately differs from the live system in: left/right convention, whistle threshold, and
scoring rules (see §9).

## 2. How to run it

```
# Windows, from the project folder; the project's real environment is .venv (Python 3.11.9)
.venv\Scripts\activate
py main.py                 # opens the training tool (trainer.py)
py main.py live            # the original live officiating system
py main.py menu            # menu: trainer, live, replay, training pipeline, diagnostics, tools, tests
```

The system Python (3.14) lacks `joblib`, `mediapipe`, etc. — always use the venv (with it active, `py` uses it).

Settings live in `trainer_config.py`:

| Setting | Value | Meaning |
|---|---|---|
| `CONTACT`, `RETENTION` | team email, 1 year | shown in the consent notice |
| `CAMERA_INDEX`, `WHISTLE_DEVICE_INDEX` | `None` | fallbacks; normally chosen in the app under **Camera and mic** |
| `INPUT_ALREADY_MIRRORED` / `MIRROR_DISPLAY` | `False` / `True` | display only; grading always uses the trainee's true left/right |
| `WHISTLE_THRESHOLD` | **0.55** | training tool's whistle operating point (live system stays 0.70) — §9 |
| `TEST_MODE` | **False** for evaluators | `True` shows a "Session label" box (correct / wrong on purpose) for dry runs |
| `THEME` | `dark` | |

**Hardware used for testing:** iPhone through **Camo Studio** as camera and microphone. In **Camera and mic**
choose the Camo camera and **"Microphone (Camo) [Windows DirectSound]"**. The WASAPI entry never works (it only
accepts 48 kHz; the detector needs 22,050 Hz) and the WDM-KS entry fails whenever another program holds the mic.
The app remembers the mic **by name**, finds it again after the phone reconnects, and warns if it is missing.

## 3. Files

| File | Role |
|---|---|
| `trainer.py` | the training tool: session state machine for all six modes, grading calls, scoreboard, whistle hub, logs, report |
| `trainer_ui.py` | Tkinter screens: welcome + consent, menu (modes, options, "Require the whistle"), Learn the signals, progress, sessions, delete data |
| `gesture_grader.py` | FIVB-derived rubric: recognition, distinctness, hold, form checks, ready position; three levels; instruction cards |
| `decision_engine.py` | the live system's call validator (whistle windows, pending gesture, settle window); reused by Simulation/Testing |
| `whistle_detector.py` | the audio team's detector wrapper (band-pass, energy gate, ensemble, physics gate, 2-window confirm) |
| `devices.py`, `device_setup.py` | camera/mic listing, level meter, whistle test, saved device resolution |
| `progress.py` | "My progress" from saved attempts |
| `main.py` | front door; no arguments = trainer |
| `live_deployment.py` | the original live officiating system (unchanged by the training-tool work except where stated) |
| `tools/pilot_check.py` | pass/fail table for a dry run (recognition, consistency, feedback time, stability, whistle false alarms) |
| `tools/performance_report.py` | fps, slowest frame, verdicts within 3 s, CPU, RAM per mode |
| `tools/evaluate_sessions.py`, `tools/regrade_attempts.py` | grader accuracy vs labelled intent; re-grade saved attempts with current rules |
| `tests/test_trainer_smoke.py`, `tests/test_gesture_grader.py` | automated tests (§8) |

## 4. The six modes, exactly as they behave now

All graded modes use the same capture: countdown → capture window (5 s; 9 s for a Team-to-Serve-plus-reason pair;
+1.5 s when the step needs the whistle). **The capture ends early** once the signal was recognised (held for two
model windows in a row) and the arms are back at the ready position; Ball In, held low, ends when the model stops
seeing it. If the step needs the whistle and none was heard, it waits at most 1.5 s after the signal ends.

| Mode | What happens |
|---|---|
| **Practice** | free signalling; "You are showing" names the detected signal live with confidence bars and a live checklist; no score; Q returns to the menu (report saved, no summary screen) |
| **Drill** | one chosen signal × 1/3/5/10 reps; 1 rep = free practice with **R retry** (adds a new attempt, keeps the old one) |
| **Combo** | end-of-rally pair: Team to Serve then the reason, in one capture |
| **Challenge** | every signal once, random order |
| **Match Simulation** | narrated set (3/5/10 rallies): Service Authorization, then Team to Serve + reason per rally; H shows the required signal at higher levels |
| **Match Testing** | continuous, unscripted grading; live scoreboard window, sequence bar, decision chip; shows a win at 25 (win by 2) but never stops scoring (same as the live system); `[ ] - +` manual score keys |

**"Require the whistle"** (menu checkbox, available in Drill, Combo, Challenge, Simulation, Testing):

- Drill / Combo / Challenge: Team to Serve and Service Authorization steps also expect a whistle (FIVB order:
  whistle, then signal). The whistle is graded as its own attempt: **heard anywhere in the attempt = 10 points,
  not heard = 0**. There is no timing grade (§9).
- Match Simulation: each call goes through `decision_engine.py`; a point or serve authorisation needs **both** a
  whistle within the allowed window (up to 10 s before the signal or 6 s after it) **and** a passing form verdict
  (CORRECT or ALMOST). Unticked: form alone decides.
- Match Testing: same engine and same double requirement. A finished Team to Serve is **held pending 1.5 s** so a
  same-side Service Authorization (whose beckon starts from a Team-to-Serve-like pose) can cancel it — the same
  rule as `live_deployment.py`'s `TEAM_TO_SERVE_CONFIRM_DELAY_SECONDS`.

**Scoring side:** left/right always means the **trainee's own arm**. Team to Serve with the left arm gives the
point to the team on the trainee's left. (The live system flips this for an audience-facing scoreboard; the trainer
deliberately does not.) A wrong-arm Team to Serve in Simulation is graded INCORRECT and scores nothing.

**Serving indicator:** set from the story at the start, then moves only when the trainee's Team to Serve is
actually committed.

**Other behaviour evaluators will see:**

- Result screen: verdict, score, "Recognised as" + confidence bar, score breakdown, checklist, tips.
- NO READING (body seen in under 60% of frames, or nothing to analyse): "step back so your head, shoulders and
  arms are in frame", not counted, step repeats. **After 3 NO READINGs in a row** it stops and asks to check the
  camera (Camo keeps sending a placeholder picture when the phone is unplugged, so a lost camera looks like "no
  person", not a failed read). A truly failed camera read shows "Camera disconnected… reconnect", then "Camera
  stopped… results saved".
- P pauses (also mid-attempt: that attempt is dropped, the step repeats); Q ends early and keeps attempts.
- Summary: per attempt, "Correct per signal", "Needs more practice" (signals under half CORRECT); A/D reviews each
  attempt.
- Report (`report.html`): Target, Detected, verdict, score, All levels, failed checks, feedback, seconds from arms
  down to verdict; whistle log; performance.
- Welcome screen states the purpose and how to start; consent box must be ticked; "Delete my data" on the menu.

## 5. Grading (`gesture_grader.py`)

Score out of 100 = Recognition 38 + Distinctness 10 + Hold 10 + FIVB form 40 + Ready position 2. Form points are
normalised to 40 over the checks the camera could verify.

| Level | Model prob for full recognition | Lead over next class | Hold | CORRECT from | ALMOST from |
|---|---|---|---|---|---|
| Beginner | 0.55 | 0.15 | 0.6 s | 60 | 40 |
| Standard | 0.75 | 0.30 | 1.0 s | 75 | 55 |
| Referee | 0.90 | 0.45 | 1.5 s | **100** | 68 |

- A failed critical/strict check caps the verdict at ALMOST. Session points: CORRECT 10, ALMOST 5, INCORRECT 0.
- Every attempt is also graded at all three levels (report column "All levels").
- Not recognised: if another signal dominates → INCORRECT "The system saw X instead"; arms never raised →
  INCORRECT "No signal seen: raise your arm(s)".
- Pairs (Combo/Simulation) are split by first appearance of each signal; wrong order caps CORRECT at ALMOST.

## 6. Whistle detection (`whistle_detector.py`)

22,050 Hz; 1.5 s windows every 0.5 s; band-pass 1.8–4.8 kHz; energy gate RMS ≥ 0.002; ensemble probability ≥
threshold; physics gate (whistle-band ratio ≥ 0.35, pitch instability ≤ 20); **two windows in a row** confirm one
whistle; 2 s cooldown before the next. It confirms about **0.75 s after** the whistle starts (measured, §9). The
trainer subtracts that delay when it needs a whistle's start time.

`start(confirm_open=True)` (used by the trainer and the setup test) waits until the mic stream actually opened and
raises a clear error otherwise; before this, an unopenable mic failed silently while the app said "blow your
whistle". `live_deployment.py` still calls `start()` the old way.

## 7. What every session saves

`data/trainer_sessions/<name or evaluator code>/<YYYYMMDD_HHMMSS>_<mode>[_label]/`

| File | Contents |
|---|---|
| `attempts.csv` | one row per graded attempt: target, **detected**, level, verdict, score, points, best prob, margin, hold, confused_with, failed checks, check values, feedback, **feedback_latency_s**, verdict at all levels |
| `session_log.csv` | timeline: session start (mode, level, whistle option, detector running/error, **whistle_threshold**), captures, verdicts, whistles, points awarded/withheld, pauses, camera events |
| `whistle_events.csv` | each whistle: time, source (auto / manual W), confidence, attempt, judged |
| `summary.json` | totals, per signal, needs practice, performance (fps, slowest frame, CPU, RAM, feedback within 3 s) |
| `report.html`, `session.mp4`, `attempts/*.npz` | readable report; screen recording (no audio, timing drifts); raw keypoints per attempt (for re-grading) |

Also `data/consent_records.csv` (consent and deletions) and `data/whistle_logs/` (every audio window the detector
scored — useful for calibrating the threshold). **Back up `data/trainer_sessions/` after every testing day.**

## 8. Tools and tests

```
py tools/pilot_check.py --since 20261004_180000 --noise-session <folder> --expected-whistles 5
py tools/performance_report.py --since <YYYYMMDD_HHMMSS>
py tools/regrade_attempts.py --level standard
.venv\Scripts\python.exe tests\test_trainer_smoke.py      # expect 34 "OK" lines, no traceback
.venv\Scripts\python.exe tests\test_gesture_grader.py     # expect all PASS (45)
```

PowerShell: never type `<...>` literally; it is a placeholder. Running the grader tests regenerates
`data/grading_rubric.md/.csv`. `pilot_check` only counts recognition/consistency from sessions labelled "I will
perform correctly" (TEST_MODE); evaluator sessions need a separate compile step (open item, §12).

## 9. Decisions made, with the evidence

Cite these when the manuscript explains a number. All measured on this project's own recordings.

| Decision | Evidence |
|---|---|
| **Whistle threshold 0.55** for the training tool (model unchanged; live stays 0.70) | Replaying the detector's own per-window logs of 51 clean whistles from a phone mic at 1–2 m: 0.70 confirmed 8/51, 0.60 39/51, **0.55 45/51**, 0.50 the same 45/51 (lower adds only noise risk). Remaining misses were whistles inside the 2 s cooldown. Dry run: 5/5 intentional whistles detected, **0 false alarms** in ~60 s of talking/clapping/music. |
| scikit-learn version mismatch is harmless | Model saved with 1.9.0, venv has 1.5.1: identical probabilities on the same features (max difference 0.0000). |
| **Whistle graded as present / absent**, no timing | FIVB 22.2.3 sets the order (whistle, then signals) but no time limit; the panel did not ask for timing. Previously the detector's confirmation delay plus a 10-fps assumption made on-time whistles look late (4/29 judged on time; 21/29 after correcting both). |
| Whistle start = detection − 0.75 s; signal start from the real frame rate | Median 0.5 s from first whistle window to confirmation + ~0.25 s inside it (37 whistles). Trainer runs at 15–18 fps on the test laptop, not 10. |
| **Referee level needs 100** for CORRECT | Team decision. Effect on all 382 recorded attempts: CORRECT 66% Beginner, 55% Standard, **27% Referee**. Present Referee level as an aspirational, textbook-strict target. Test at Standard. |
| **Ball Out "elbows not too far apart"** check (strict at Standard/Referee) kept | Team decision: an over-spread Ball Out is ALMOST at Standard. On 61 real Ball Outs: median spread 1.56× shoulder width, 49 pass; 9 CORRECT → ALMOST. Learn card now teaches it ("less than about twice your shoulder width"). |
| Ball Out form weights 10/10/4/7/5/4 (= 40) | Previously 48 after the new check. Re-grading all 61 real Ball Outs at every level: identical scores and verdicts. |
| Team to Serve pending 1.5 s in Match Testing | Same rule as the live system; a Service Authorization beckon was scoring as a point. |
| Early capture finish + per-attempt latency | Drill dry run: 42/42 verdicts within 3 s of arms down (mean 0.34 s). Match Testing grades when the model stops seeing a signal, so it is slower (mean ~3.6 s) — time feedback in Drill. |
| Match Testing reads only its own session's whistles | Bug found in the dry run: earlier sessions' whistles were replayed at session start. Regression test added. |
| Simulation whistle gating made optional (checkbox) | Previously Simulation added points from form alone, never checking the whistle. |

## 10. Evaluation

**Forms.** `new_evaluation_forms_REVISED_clean.docx` (give to evaluators) and `…_REVISED_tracked-changes.docx`
(for the adviser / validator), next to the original `new_evaluation_forms.docx` — currently in Gion's Downloads,
share them. 18 items were reworded after content validation to describe the final system (IT 2, 3, 4, 7, 9, 11, 12,
18, 19, 35; Referee 5, 9, 11; Scorekeeper 5, 6; General 5, 10), keeping item numbers, sub-characteristics and the
scale; an Evaluation Protocol was added. Only the IT forms were validated by Dr. Collo; the other three were adapted
from them — say exactly that in the manuscript. The IT form has **37** items (the manuscript still says 29).

**Procedure.** The testing-day page (Claude artifact, ask Gion for the link): setup, participant session B1–B12,
expert steps C1–C7, record sheets, and every item of all four forms mapped to the step where it is observed.
Design: each general participant uses the tool as a trainee and answers the General form from their own use;
experts watch participant sessions plus a researcher-performed Part C, and answer from observation (IT evaluators
read "I" as "the performer"). If an expert cannot attend: one unedited continuous recording of a researcher doing
Parts B and C (screen + mic audio + a phone filming the body), never a compilation of participants' clips, and not
participants' videos (consent covers research use, not outside viewing).

**Record by hand (cannot be recovered from logs):** referee tally sheet (own call per attempt), which attempts were
deliberately wrong, whistles actually blown and the no-whistle minute's times, the facilitator log (date, place,
face-to-face/remote, help given, problems), respondent background, signed qualification sheets, setup photos.

## 11. Manuscript alignment

Claims checked against the code. Line numbers refer to the current revision ("Copy of Ma'am Rivera for the
REVISIONS_MANUSCRIPT…").

| Manuscript says | Code / status | Action |
|---|---|---|
| IT questionnaire has 29 items (Instruments, line 316) | 37 items | update |
| Whistle threshold 0.70 (lines 311, 392) | true for the live system; training tool uses **0.55** | add the training-tool threshold and its calibration (§9) |
| Laptop webcam/mic at 2 m (line 309) | evaluation uses iPhone via Camo, DirectSound mic | describe the setup actually used |
| Evaluators "used or observed" the six modes (line 329) | matches the procedure | fill "[face-to-face / remotely]" and "[dates]" |
| Simulation "always" whistle-gated (older text / CLAUDE.md) | only when "Require the whistle" is ticked | correct wherever stated |
| System Performance Analysis (lines 393–402) | describes the **live-match** system only | add a training-tool performance subsection: recognition rate (target vs detected), grader agreement with the referee, whistle detection rate + false alarms, scoring correctness, feedback within 3 s, fps/CPU/RAM, consistency |
| Results means at line 425 | from the earlier live-match version | new results after evaluation |
| A distinct no-signal message in graded modes | yes: NO READING when the body is not seen (not counted); visible-but-no-signal is INCORRECT with a "raise your arm" tip | describe both cases |
| Camera interruption shows a message, allows restart, keeps attempts | yes (failed read: disconnected message; Camo placeholder: after 3 NO READINGs) | — |
| Whistle "timing" graded | no: presence only (FIVB order, no time limit) | state it |
| Referee level | CORRECT only at 100 | state it; frame as aspirational |
| Left/right | trainee's own arm (not the live system's flipped convention) | state it if scoring sides are described |

Live-system tables (7A/7B response time, 11 score match, etc.) describe `live_deployment.py` and stay as live-system
results; do not present them as training-tool results.

## 12. Known limitations and open items

- Service Authorization vs Team to Serve confusion is real in the model; Match Testing's pending window hides it,
  Drill/Combo/Challenge do not.
- Background objects can be read as a person (a false Service Authorization was seen once): plain wall needed.
- Match Testing feedback is slower than the 3 s rule by design (grades when the signal ends).
- Single 2D camera: open hand, palm direction and some form checks are often "unverified".
- Ball In has the weakest recall (two contributing volunteers).
- `session.mp4` has no audio and drifts in timing.
- Practice has no summary screen (report is saved).
- Open: a tool that compiles evaluator-session numbers into the Results tables (pilot_check only handles labelled
  dry runs); the forms' embedded protocol still describes each role testing alone (the testing-day page supersedes
  it).
- Untracked repo files (appendix tools, diagnostics, `data/grading_rubric.*`) were left out of commits on purpose.

## 13. Rules for changing the code

- Run both test suites with the venv before committing; they must pass.
- Read `gesture_grader.py` and `trainer_ui.py` fresh before editing; teammates edit them directly.
- Any grading change: re-grade the saved real attempts (`tools/regrade_attempts.py`) and note the effect here (§9).
- Any whistle change: check with the detector logs in `data/whistle_logs/` and a no-whistle noise test.
- Do not change `live_deployment.py` behaviour as a side effect; it is a separate, documented system.
- Keep `TEST_MODE = False` in commits.
