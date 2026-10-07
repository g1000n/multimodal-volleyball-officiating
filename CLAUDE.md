# CLAUDE.md

Volleyball officiating thesis project (team NaviTalamas). **Read `docs/HANDOFF.md` first** — it is the full,
current description of the system, every behaviour and decision with its evidence, the evaluation setup, and how
the manuscript must line up with the code.

Essentials:

- Two systems live here: the **training tool** (`trainer.py`, `trainer_ui.py`, `gesture_grader.py`, `devices.py`,
  `device_setup.py`, `progress.py`, `trainer_config.py`; `py main.py`) and the original **live officiating system**
  (`live_deployment.py` + `decision_engine.py`; `py main.py live`). A fix in one is not in the other.
- Use the project venv: `.venv\Scripts\python.exe` (Python 3.11.9). The system Python lacks the model packages.
- Tests before every commit: `tests\test_trainer_smoke.py` (35 OK) and `tests\test_gesture_grader.py` (all PASS).
- `gesture_grader.py` and `trainer_ui.py` are edited directly by teammates: read them fresh, don't trust summaries.
- Grading or whistle changes need evidence from the saved recordings (`tools/regrade_attempts.py`,
  `data/whistle_logs/`) and a note in `docs/HANDOFF.md` §9 — the manuscript cites those numbers.
- When asked to check a manuscript claim against the code, treat it as a real correctness check.
- Keep `TEST_MODE = False` in commits.
