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
10. Evaluation: forms, design, setup, participant session, demonstration, expert clips, item maps, referee tally, records to results
11. How the system fulfils the study's objectives
12. Manuscript alignment: what the paper must say
13. Known limitations and open items
14. Rules for changing the code

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
| `tools/compile_results.py` | **the evaluation's results**: every laptop's session logs + the records sheet (.xlsx) → recognition per signal (all / with exclusions / per setup), NO READING, consistency, feedback time, P9/P10, whistle detection, help, crashes, and a list of problems to fix first |
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
py tools/compile_results.py --sheet Evaluation_Records.xlsx --root data/trainer_sessions --root <other laptop's copy>
py tools/regrade_attempts.py --level standard
.venv\Scripts\python.exe tests\test_trainer_smoke.py      # expect 34 "OK" lines, no traceback
.venv\Scripts\python.exe tests\test_gesture_grader.py     # expect all PASS (45)
```

PowerShell: never type `<...>` literally; it is a placeholder. Running the grader tests regenerates
`data/grading_rubric.md/.csv`. `pilot_check` only counts recognition/consistency from sessions labelled "I will
perform correctly" (TEST_MODE); evaluator sessions need a separate compile step (open item, §13).

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
| **Service Authorization: "elbow bent" caps only at Referee; Standard chest band 0.15 → 0.20** | A trainee's correct-on-purpose SA-left Drill (Gop, 2026-10-04 16:46, 10 attempts) got 2/10 CORRECT: 8 failed the elbow check at 148–161° (Standard limit 145°), because one frontal camera sees the elbow open as the hand sweeps out. 144 real Team to Serve attempts measured the same way span 58–178°, so elbow alone cannot separate the two; a straight arm is still caught by the required "hand moves" check and the model. Now 9/10 CORRECT. All 103 SA attempts at Standard: CORRECT 55 → 77, ALMOST 37 → 15, INCORRECT 10 → 10; Referee unchanged; no other signal changed. A failed elbow check still costs its 7 points and shows the tip. |
| Team to Serve pending 1.5 s in Match Testing | Same rule as the live system; a Service Authorization beckon was scoring as a point. |
| Early capture finish + per-attempt latency | Drill dry run: 42/42 verdicts within 3 s of arms down (mean 0.34 s). Match Testing grades when the model stops seeing a signal, so it is slower (mean ~3.6 s) — time feedback in Drill. |
| Match Testing reads only its own session's whistles | Bug found in the dry run: earlier sessions' whistles were replayed at session start. Regression test added. |
| Simulation whistle gating made optional (checkbox) | Previously Simulation added points from form alone, never checking the whistle. |

## 10. Evaluation

### 10.1 The forms

- Files (in Gion's Downloads; share them): `new_evaluation_forms_REVISED_clean.docx` (give to evaluators) and
  `new_evaluation_forms_REVISED_tracked-changes.docx` (for the adviser / validator), next to the original
  `new_evaluation_forms.docx`.
- Four forms, all on a 1–5 Likert scale, ISO/IEC 25010:2023 characteristics (Functional Suitability, Performance
  Efficiency, Reliability, Interaction Capability): **IT Professional 37 items, Volleyball Referee 21, Volleyball
  Scorekeeper 17, General Audience 21**, each with open-ended questions and an overall validation result.
- Shared criteria used in items: 3-second rule (verdict within 3 s of finishing the signal = arms back down);
  5/4 consistency (same signal 5 times → same verdict at least 4 times); 10/8 accuracy; 10-minute / 20-attempt
  capacity; allowed whistle window (up to 10 s before the signal or 6 s after it).
- Validation: Dr. Dexter Collo validated the **IT** forms ("Validated with Minor Revisions"); the referee,
  scorekeeper and general forms were **adapted** from it, not independently validated. After validation, 18 items
  were reworded to match the final system (IT 2, 3, 4, 7, 9, 11, 12, 18, 19, 35; Referee 5, 9, 11; Scorekeeper 5,
  6; General 5, 10), keeping item numbers, sub-characteristics and the scale. The forms' header note says so. Do
  not add new items (a proposed "wrong arm scores for the opposite team" item was rejected: it describes the live
  system, not the trainer).
- The revised forms also contain an embedded protocol written for "each role uses the tool alone". **The testing
  procedure below supersedes it.**

### 10.2 Design

- **One researcher** runs every session (facilitator, recorder, laptop).
- **General participants** (volleyball players, fans, beginners) each use the tool **as trainees**, one at a time,
  and answer the General Audience form from their own use. Never from watching someone else.
- **One demonstration session** (researcher or one consenting volunteer, code `DEMO`) shows the few things only
  experts are asked about.
- **Expert evaluators** (IT, referee, scorekeeper) answer **later, from clips** cut from the recordings. IT
  evaluators read "I" in their items as "the performer in the clip".
- **No blank items:** every item on every form maps to a step or clip (10.6). Before collecting a form, check every
  item is answered; if someone is unsure, they redo / rewatch the mapped step, then answer.
- Participants are identified by a **code** (`G-01`, `G-02`, …) typed in the app instead of a name, so all their
  data lands in `data/trainer_sessions/G-01/`. Their real name appears only on the paper sign-in sheet.
- Locations may differ (rooms, a court) if the camera setup is identical and each session's location is recorded.

### 10.3 Setup (every location)

- iPhone on a tripod at **chest height** (1.2–1.4 m), **2–2.5 m** from the performer, straight on. Frame **head to
  hips** (the grader measures wrists against the shoulder-to-hip distance) with room above the head and to both
  sides. Tape mark on the floor. Plain wall, light from the front, nobody else in frame, clear space ~3 × 2 m.
- Laptop: Camo Studio running; `py main.py`; **Camera and mic** = Camo camera + **Microphone (Camo) [Windows
  DirectSound]**; whistle test passes; `TEST_MODE = False`; app restarted for every participant.
- **Per-location check (5 min):** framing; whistle test; 3-rep Drill with "Require the whistle"; **30 s of normal
  noise with no whistle — nothing may flash** (on a court, other courts' whistles can trigger it); setup photo.
- Recording: **OBS** (whole screen + microphone so whistles are audible) for every participant, from the consent
  screen to the summary, saved as `G-01_<date>.mkv`. A phone filming the full body is required for the
  demonstration's trainee script (the referee judges form from it).

### 10.4 Participant session P1–P13 (about 40–45 min), difficulty Standard throughout

| Step | What the participant does | Researcher records |
|---|---|---|
| P1 | Signs the sign-in sheet (code, name, date, signature, **yes/no: recording may be shown to the expert evaluators**); gets their code; briefing read aloud | code, background (player/fan/beginner/coach, years), location, start time |
| P2 | Researcher starts OBS and the app. Participant reads the welcome screen and consent notice, types their code, ticks, continues | read the notice? questions? |
| P3 | Learn the signals: looks through all of them | help needed? |
| P4 | Practice: three different signals (at least one left-arm, one right-arm), ~3 s each, arms down between; Q to end | name followed each signal? |
| P5 | Drill, the signal **assigned to the code** (rotation below), **10 reps**: countdown → GO → signal, hold ~2 s, arms down; result shows at once, next starts by itself. All 10 the same way. After **each** result read "Recognised as" and the first tip. On attempt 6 press P mid-signal, wait, P again (not counted, repeats). After 10: D/A to step through attempts, read "Correct per signal" and "Needs more practice", SPACE | assigned signal; tally of "Recognised as" correct out of 10; pause worked? |
| P6 | Drill "1 (free practice, retries)": one attempt, R, again, Q → summary shows both. Drill 5 reps: Q after the 2nd result → summary shows 2 | both checks |
| P7 | Drill 2 reps: attempt 1 deliberately sloppy (e.g. arm half raised); attempt 2 step out of frame at GO until a message appears, then come back and do it properly | **deliberate attempts** (this Drill, attempts 1–2); message shown |
| P8 | Challenge: every signal once | any never recognised? |
| P9 | Combo, random calls, 2 reps, **Require the whistle** ticked: whistle → Team to Serve (hold 2 s) → the reason | whistles blown; WHISTLE! each time? |
| P10 | Match Simulation, 3 rallies, **Require the whistle**, Continuous off: read story, SPACE, whistle, signal(s); watch bottom message and score | whistles blown; final score |
| P11 | Reads the summary; researcher stops OBS | end time |
| P12 | Answers the General Audience form alone, every item | items they wanted to retry |
| P13 | Researcher checks every item is answered; unsure items → redo the mapped step | steps redone |

**Assigned Drill signals (P5; P6 and P7 use the same):** G-01 Team to Serve left · G-02 Team to Serve right ·
G-03 Service Authorization left · G-04 Service Authorization right · G-05 Ball Out · G-06 Ball In · G-07 Double
Contact · G-08 End of Set · G-09 Ball In (weakest in Table 6, recall 0.57; repeated) · G-10 Team to Serve left (second weakest in Table 6,
precision 0.72, F1 0.83; repeated);
more participants continue from G-01's signal.

**Deviations, logged live (P5–P10):** an attempt where the participant performed a different signal from the one
prompted, or was out of frame, is written on the participant record (step, attempt, what happened) and excluded
under the counting rules.

**Counting rules (fixed before any data):** recognition rate per signal = P5 + P8 Challenge attempts; excluded: P4,
P6, P7 and logged deviations; **NO READING counts as a miss** unless the participant was logged out of frame or not
performing (count it separately; it is in `session_log.csv`, not `attempts.csv`); report recognition **with and
without** the exclusions; Combo
(P9) and Simulation (P10) pairs reported separately; consistency = P5 attempts 1–5; whistle false alarms = the D4
no-whistle minute plus detections beyond whistles blown in P9–P10 for every participant; DRY-01 and DEMO data never
enter participant results.

**Before the first participant:** one dry session on a teammate (code DRY-01, excluded) to check timing, OBS audio
and the data folders; send the validator the **General Audience form first** (answered tomorrow), then the others
(37 IT items, 18 post-validation rewordings); disclose exactly what changed if re-confirmation is still pending.

After each participant: rename the recording; confirm `data/trainer_sessions/G-xx/` exists (that folder holds
everything they did); enter the record in the Google Sheet; wipe the whistle; restart the app.

### 10.5 Demonstration D1–D7 (once, ~25 min, code DEMO, recorded with OBS + phone)

| Step | Do |
|---|---|
| D1 | **Trainee script**: Drill, Team to Serve – left arm, 10 reps in this order: 1 clean · 2 clean · 3 elbow clearly bent · 4 arm well below shoulder · 5 clean · 6 right arm instead of left · 7 clean but lowered after under 0.5 s · 8 clean · 9 both arms raised · 10 clean |
| D2 | Open D1's report (Progress and sessions → My sessions → Open report): Detected, Verdict shown (s after arms down), All levels columns; then a 3-rep Drill at **Referee** level |
| D3 | Match Simulation, 3 rallies, whistle ticked: rally 1 proper; rally 2 Team to Serve **without** whistle; rally 3 whistle but wrong arm; press H once |
| D4 | Match Testing, whistle ticked: (a) whistle → Service Authorization (no point); (b) whistle → Team to Serve → Ball Out (point after ~1.5 s); (c) Team to Serve, no whistle, 10 s still (no point); (d) 5 whistles (5 flashes); (e) **1 minute without a whistle**: talk, clap, bounce a ball, signals (no flash) |
| D5 | Practice: alternate Service Authorization and Team to Serve with the same arm, 3 times each |
| D6 | Drill, unplug the iPhone mid-attempt; after 3 NO READINGs the tool asks to check the camera; reconnect, SPACE |
| D7 | Show "Delete my data" on the menu (don't press); run `py tools/performance_report.py --since <today>` on screen |

### 10.6 Expert clips and item maps

Each clip is **one continuous uncut stretch** of a step; participant clips only from participants who said "yes"
on the sign-in sheet, taken from the **first eligible participant** (a fixed rule, not the best-looking session).

| Clip | Source | Shows |
|---|---|---|
| E1 | P2–P3 | welcome, consent (cannot continue until ticked), Learn the signals |
| E2 | P4 | Practice name follows signal, left and right arm |
| E3 | P5 | Drill 10: options, results, Recognised as, tips, pause, summary review |
| E4 | P6 | retry keeps both attempts; ending early keeps attempts |
| E5 | P7 | sloppy attempt graded down; out of frame → NO READING |
| E6 | P8 | Challenge |
| E7 | P9 | Combo with whistle |
| E8 | P10 | Match Simulation: whistle, commits, score, serving |
| E9 | P11 | summary |
| E10 | D1 (screen + phone) | trainee script with known mistakes |
| E11 | D2 | report vs reality, all levels, Referee level |
| E12 | D3 | no point without whistle or with wrong arm; hint |
| E13 | D4 | Match Testing whistle gating, 5 whistles, no-whistle minute |
| E14 | D5 | Service Authorization vs Team to Serve |
| E15 | D6 | camera unplugged: message, reconnect, data kept |
| E16 | D7 | Delete my data visible; performance report |
| Full | one whole uncut participant recording + session numbers (10.8) | whole-session items |

**General Audience (participants, steps):** 1 P5 · 2 P5 · 3 P3, P4–P10 · 4 P5, P7 · 5 P5, P7 · 6 P10 · 7 P5 ·
8 whole session · 9 whole session · 10 P5 (attempts 1–5) · 11 P7 · 12 P7 · 13 P2 · 14 P2–P11 · 15 P4–P10 ·
16 P5, P11 · 17 P5 · 18 P2 · 19 P5, P6 · 20 P11 · 21 whole session.

**IT Professional (clips):** 1 E3 · 2 E13 · 3 E8, E12, E13 · 4 E12, E13 · 5 E11 · 6 E6 · 7 Full, E2–E9, E13 ·
8 E10 · 9 E10 · 10 E11 · 11 E8, E12, E13 · 12 E12 · 13 E3 + session numbers · 14 E16, Full · 15 E16, Full ·
16 Full + session numbers · 17 E3 · 18 E5 · 19 E5 · 20 E15 · 21 Full · 22 E3 · 23 E3 · 24 E1 · 25 E5, E10 ·
26 E10 · 27 E1 · 28 E3 · 29 E2 · 30 E6, E7 · 31 E8, E12, E13 · 32 E16 · 33 Full + help-given numbers · 34 E3 ·
35 E4 · 36 E4 · 37 E9.

**Volleyball Referee (clips):** 1 E1 · 2 E10 + tally sheet · 3 E10 · 4 E1 · 5 E7, E8 · 6 E11, Full · 7 E10 ·
8 E2, E3 · 9 E8 · 10 Full · 11 E3 · 12 E5, E13 · 13 E5 · 14 E1 · 15 E2, E3 · 16 E3, E10 · 17 E1 · 18 E2, E14 ·
19 Full, E4 · 20 E9 · 21 Full.

**Volleyball Scorekeeper (clips):** 1 E10 · 2 E8, E12 · 3 E8, E12 · 4 E1 · 5 E7 · 6 E10 · 7 E3 · 8 E8, E13 ·
9 Full · 10 E3 · 11 E5 · 12 E14 · 13 E3 · 14 E9 · 15 E6, E7, E8 · 16 E9 · 17 E1.

### 10.7 Referee tally (while the referee watches E10)

The expert check of the grader. Only the 10 scripted D1 attempts. For each attempt the researcher **pauses before
the verdict appears** (phone video preferred), the referee writes **Correct / Almost / Incorrect**, the researcher
resumes and writes the tool's verdict, then marks Match (yes if identical). Paper, then typed into the Google Sheet.
Becomes **agreement = matches ÷ 10 × 100** plus a 3 × 3 table (referee call × tool verdict). Referee item 2 is
answered from it.

### 10.8 Records → results

Paper during sessions, typed into one Google Sheet the same day:

| Tab | Columns | Result |
|---|---|---|
| Participants | code, date, **location** (+ noise/light note), background, years, start, end, help given, problems, steps redone | respondent count and composition; locations; help needed per step (learnability) |
| Whistles | code/DEMO, step, whistles blown, flashes seen, no-whistle minute start–end | detection rate = detected ÷ blown × 100; false alarms in the no-whistle minute |
| Deliberate attempts | code, session folder, attempt numbers, what was done | excluded from recognition accuracy |
| Referee tally | attempt, performed, referee call, tool verdict, match | grader agreement |
| Performance | output of `tools/performance_report.py` | FPS, slowest frame, verdicts within 3 s, CPU, RAM |
| Form responses | one row per respondent, one column per item | item, characteristic and overall means; validation result frequencies; open-ended themes (manuscript's Software Quality Evaluation Analysis) |

From the app logs across all participants (compile after testing — the compile tool is still an open item):
recognition rate per signal (target vs detected, deliberate attempts excluded), consistency of P5 attempts 1–5,
feedback within 3 s, crashes. All descriptive statistics, matching the manuscript's approach. Back up
`data/trainer_sessions/` and the recordings after every testing day.

## 11. How the system fulfils the study's objectives

General objective: a multimodal real-time training tool that helps referee and scorekeeper trainees learn and
practise official hand signals and whistle cues, using gesture recognition, whistle detection and automated score
tracking to give structured feedback and a training score.

| Objective | Fulfilled by | Evidence |
|---|---|---|
| 1. Classification model for a trainee's referee gestures | CNN-LSTM (8 classes + nothing) running live in every mode | offline metrics (existing Results tables); recognition rate per signal from participants' logs |
| 2. Model that detects a trainee's whistle cue | SVM / RF / HGB soft-voting ensemble with band-pass, energy and physics gates; training-tool threshold 0.55 (§9) | offline metrics (existing tables); detection rate and false alarms from the whistle records |
| 3. Multimodal decision engine validating service-related signals against a whistle within a temporal window, integrated with FIVB rule-based grading (correct / almost / incorrect) | `decision_engine.py` in Match Simulation and Match Testing with "Require the whistle" (10 s before / 6 s after; Team to Serve pending 1.5 s); `gesture_grader.py` three-level FIVB rubric | clips E8, E12, E13; referee tally agreement |
| 4. Automated score tracking interface showing recognised signal, verdict, feedback, and a scoreboard of the trainee's accuracy | result screen ("Recognised as" + confidence, verdict, breakdown, checklist, tips); top bar score and accuracy; team scoreboard and serving indicator in Simulation / Testing; summary and report | clips E3, E8, E9 |
| 5. Evaluate (a) models with accuracy, precision, recall, F1; (b) ISO/IEC 25010:2023 quality | (a) the existing offline evaluation; (b) the four forms via this procedure | form means plus the session numbers in 10.8 |

Keep "Require the whistle" ticked in Combo, Simulation and Testing during evaluation: objective 3 is only exercised
when it is on.

## 12. Manuscript alignment

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

## 13. Known limitations and open items

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

## 14. Rules for changing the code

- Run both test suites with the venv before committing; they must pass.
- Read `gesture_grader.py` and `trainer_ui.py` fresh before editing; teammates edit them directly.
- Any grading change: re-grade the saved real attempts (`tools/regrade_attempts.py`) and note the effect here (§9).
- Any whistle change: check with the detector logs in `data/whistle_logs/` and a no-whistle noise test.
- Do not change `live_deployment.py` behaviour as a side effect; it is a separate, documented system.
- Keep `TEST_MODE = False` in commits.
