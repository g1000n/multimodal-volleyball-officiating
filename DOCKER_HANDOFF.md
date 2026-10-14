# Docker handoff: everything to know for the thesis paper

For the teammate who documents the Docker work in the thesis paper. It covers what was added, how it differs from
the original repository, how each problem was solved, how the dean installs and checks it, what was verified, and
what must be stated as a limit.

Repository: https://github.com/g1000n/multimodal-volleyball-officiating. The Docker files are on branch `docker`
and were merged into `main` on 15 Oct 2026, so a plain `git clone` gets them.
Companion files: `docker.md` (usage and troubleshooting), `DOCKER_TESTING.md` (verification on another computer).

**Rule for the paper:** only describe something as verified if section 7 says it was. Items marked "not yet
verified" are limits or future work.

---

## 1. Why Docker was done

The dean asked the team to dockerize the system. Goals:

1. The dean can install and run the system from Docker without installing Python, MediaPipe, OpenCV or PyTorch.
2. The dean can see it running with `docker ps` and check it with `docker exec`.
3. The original system is not changed. The application code must stay as it is on Windows.
4. `requirements.txt` is followed exactly.

## 2. Changes from the original repository

This is the full list. Everything is in the repository root.

| Change | Type | Notes |
|---|---|---|
| `Dockerfile` (new) | New file | Builds the image. |
| `docker-compose.yml` (new) | New file | Named commands: `tests`, `app`, `trainer`, `live`. |
| `.dockerignore` (new) | New file | Allowlist; keeps data, recordings and the virtual environment out of the image. |
| `docker.md`, `DOCKER_TESTING.md`, this handoff (optional) | New docs | Instructions, verification checklist and this handoff. |

**No existing source file was changed** (no edits to `extract_keypoints.py`, `trainer.py`, `requirements.txt` or any
other code) **when the stand-in approach in section 4 is used.** The application code is identical to the original.

Two things in the repository are worth knowing about but were **not** changed:

- `extract_keypoints.py` imports `winsound`, which exists only on Windows (section 4).
- The saved whistle model was created with a newer scikit-learn than `requirements.txt` pins (section 5).

## 3. What the Docker files contain and why

**`Dockerfile`**

- Base image `python:3.11-slim-bookworm`. Python 3.11 matches the team's venv (3.11.9). MediaPipe 0.10.14 does
  not support Python 3.12 or later.
- System libraries that the slim image lacks: OpenCV (`libgl1`, `libxcb-*`), Tkinter (`libtk8.6`, `libtcl8.6`),
  audio (`libportaudio2`, `libsndfile1`) and `xdg-utils` for the "open report" button.
- PyTorch is installed first from the CPU-only index (`torch==2.12.1`, matching the pin in `requirements.txt`).
  The default Linux wheel pulls about 2 GB of CUDA libraries that the project never uses. Everything else is then
  installed from `requirements.txt` unchanged (including scikit-learn 1.5.1 and numpy 2.2.6).
- `xvfb` and `xauth` (a virtual screen, section 6).
- A small stand-in `winsound` module (section 4).
- The code and trained models are copied in last, so changing code does not reinstall the heavy packages.
- Trainee data is not copied in; it lives outside the image.

**`docker-compose.yml`**

| Service | Purpose |
|---|---|
| `tests` | Runs the project's automated tests inside the container (needs no camera). Starts a virtual screen first. |
| `app` | A long-running container named `volleyball-officiating` (`sleep infinity`) so that `docker ps` lists it and `docker exec` can run checks in it. The training tool is a desktop window, not a server, so there is no port to show. |
| `trainer`, `live` | The training tool and the original live system as GUI programs. Linux host with camera and microphone only (profile `gui`). |

**`.dockerignore`** ignores everything and lets in only `requirements.txt`, `*.py`, `models`, `assets`, `tests`,
`tools`, `diagnostics`, `scripts` and `docs`. The file must be named exactly `.dockerignore`.

## 4. The winsound situation: the problem, and how it was solved

**Problem.** `extract_keypoints.py` starts with `import winsound` and uses it only for a finish beep at the end of
the offline extraction script (`winsound.Beep(...)`). `winsound` is a Windows-only module. The live system
(`live_deployment.py`) imports `extract_keypoints.py`, so in a Linux container (which is what Docker runs) the
live system would stop with `ModuleNotFoundError: No module named 'winsound'`. A search of the repository found
`winsound` used in only those two places in that one file.

**Option A (considered): edit the code.** Make the import optional (`try: import winsound / except ImportError:
winsound = None`) and skip the beep when it is missing. This works and does nothing on Windows, but it changes an
application file. A ready patch was prepared (`extract_keypoints_winsound.patch`) but is **not used**.

**Option B (used): a stand-in module inside the Docker image only.** The `Dockerfile` creates a tiny module:

```dockerfile
RUN printf 'def Beep(frequency, duration):\n    pass\n' > /usr/local/lib/python3.11/site-packages/winsound.py
```

In the container, `import winsound` now succeeds and `Beep` does nothing. The repository code is unchanged and
Windows is unaffected, because the stand-in exists only inside the image.

**Effect.** The live system and training tool start normally in the container. The only behavior difference is
that the finish beep of the offline extraction script is silent in Docker.

**Status.** Option B was added to the `Dockerfile` after the main test run. Confirm it with
`docker exec volleyball-officiating python -W ignore -c "import extract_keypoints; print('imports OK')"`, which
should print `imports OK`. See section 7.

## 5. The scikit-learn version warning

The saved whistle model (`models/whistle_svm_model.pkl`) was created with scikit-learn 1.9.0. `requirements.txt`
pins 1.5.1, and both the team's Windows venv and the Docker image install 1.5.1. Loading the model prints
`InconsistentVersionWarning` (scikit-learn says results are not guaranteed across versions).

- This mismatch is in the repository itself. It exists on the team's Windows setup too. Docker did not introduce it.
- `requirements.txt` was **not** changed to hide it.
- The project's own `docs/HANDOFF.md` already records it as harmless: "Model saved with 1.9.0, venv has 1.5.1:
  identical probabilities on the same features (max difference 0.0000)."
- **Checked for Docker.** The same synthetic test sound (a 3 kHz tone with low noise, fixed random seed) was
  passed through the model in both places. The Windows venv gave `[[0.87199647 0.12800353]]` and the Docker
  container gave `[[0.87199647 0.12800353]]`: identical. This is one test sound, not real whistle recordings,
  but it agrees with the project's own note.

## 6. Other problems met, and how each was solved

These are useful for a limitations or lessons-learned section.

| Problem | Cause | Solution |
|---|---|---|
| Tests aborted with `qt.qpa.xcb: could not connect to display` | The Match Testing check opens an OpenCV window, and a container has no screen | Install `xvfb` and start a virtual screen for the tests. It exists only inside the container; the application code is unaware of it. |
| Tests printed nothing and appeared to hang | The `xvfb-run` wrapper hung inside Docker Desktop | The `tests` command starts `Xvfb` directly. |
| Build tried to copy more than 1 GB, "read-only file system" | `.dockerignore` was misnamed (no dot) and then too permissive | Renamed it and switched to an allowlist. |
| Build failed with `Read-only file system` and `Bus error` | The laptop's C: drive was full; Docker's disk went read-only | Purged Docker data and moved Docker's disk image to D: (Docker Desktop, Settings, Resources, Advanced). |
| Docker Desktop bootstrap error after moving the disk | Leftover WSL state | Quit Docker, `wsl --shutdown`, restarted the computer. |
| `wsl --install` failed ("requires elevation", "Catastrophic failure") | Windows features not enabled; needs administrator | Administrator PowerShell, enabled the Windows features, `wsl --install --no-distribution --web-download`, restarted. |
| `docker version` showed an empty `Server:`; `hello-world` hung | Engine stuck | Quit Docker Desktop, `wsl --shutdown`, restarted it. |
| Very slow `apt-get` step | The container could not reach the Debian servers (network, VPN or DNS) | Retried. If it recurs: turn off any VPN or set Docker's DNS to 8.8.8.8 and 1.1.1.1. |

## 7. What was verified, and what was not

**Verified (with evidence from the author's Windows laptop, Docker Desktop):**

1. The image builds from the repository: `docker compose build` ended with `Image volleyball-officiating:latest Built`.
2. The project's own tests pass inside the container: `docker compose run --rm tests` printed all the smoke-test
   `OK` lines and `Smoke test passed.`, then every grader test as `PASS` and `All good.` (The README gives the
   expected count as 35 `OK` lines; count them in your own run before quoting a number.)
3. The whistle rule and timing tests passed inside the container (`sim gate`, `whistle`, `whistle st`, `logs`).
   These use simulated whistles, not real sound.
4. The whistle model loads in the container (`whistle model loaded: VotingClassifier`).
5. A container named `volleyball-officiating` appears in `docker ps`.
6. The scikit-learn version in the image (1.5.1) is the same as in the team's Windows venv.
7. The whistle model gives identical probabilities in Docker and in the Windows venv on the same test sound
   (`[[0.87199647 0.12800353]]` in both). Smoke test: 35 `OK` lines and `Smoke test passed.`; grader tests all
   `PASS` and `All good.`, re-run after the latest `Dockerfile` and compose changes.

**Not yet verified. State these as limits or future work, or fill them in once done:**

| Item | Status |
|---|---|
| The winsound stand-in works (`imports OK` check, then tests pass again after rebuilding) | Tests pass after rebuilding from a fresh clone (15 Oct 2026); the `imports OK` check is still to run |
| Docker files merged to `main` | Done (15 Oct 2026) |
| A clean install on another computer (the dean's setup) | Done on a second laptop: fresh clone, `docker compose build`, then `docker compose run --rm tests` passed (35 `OK`, `All good.`), 15 Oct 2026 |
| GitHub Actions check (optional, described in `DOCKER_TESTING.md`) | Not created or run |
| Image published to a registry | Done: `g1000n/multimodal-volleyball-officiating:1.0` on Docker Hub, https://hub.docker.com/r/g1000n/multimodal-volleyball-officiating (15 Oct 2026) |
| Image size and build time | Not recorded; measure with `docker images volleyball-officiating` |
| Real camera and microphone inside Docker | Not possible on Windows (section 9) |

## 8. How the dean installs and checks it

**Prerequisites:** Docker Desktop installed and running (Windows with the WSL 2 backend, or macOS), or Docker
Engine with the Compose plugin on Linux. Git. About 20 GB free on the drive Docker uses. An internet connection
for the first build (roughly 1 to 2 GB of downloads). No Python, MediaPipe or PyTorch is needed on the host.

**Install and start (any operating system):**

```
git clone -b docker https://github.com/g1000n/multimodal-volleyball-officiating
cd multimodal-volleyball-officiating
docker compose up -d --build app
docker ps
```

Build from a **fresh clone**, as above. The `Dockerfile` copies whatever is on disk (`COPY . .`), so building from a
working folder with uncommitted or untracked files (old model folders, evaluation records containing participant
codes) would put them in the image. Do not share an image built from a working folder.

The first build takes several minutes. `docker ps` should list a container named `volleyball-officiating`.

**Check that it works:**

```
docker compose run --rm tests
docker exec volleyball-officiating python tests/test_gesture_grader.py
docker exec volleyball-officiating python -W ignore -c "import extract_keypoints, whistle_detector; print('imports OK')"
```

- `docker compose run --rm tests` should end with `Smoke test passed.` and `All good.` It can stay silent for
  a few minutes; let it finish.
- The grader test should show every line as `PASS` and `All good.`
- The import check should print `imports OK`. It shows the whistle model and the winsound stand-in both work.

**Try the system without a camera.** The project's interactive decision-engine tester needs no camera, model or
audio. Type events and watch what the real `decision_engine.py` does (`w` stands in for a whistle):

```
docker exec -it volleyball-officiating python tests/interactive_decision_engine_test.py
```

Commands: `w`, `team_to_serve_left`, `ball_out`, `score left +1`, `status`, `help`, `quit`.

**Stop and remove:**

```
docker compose down
```

**Run the live system with a camera and microphone (Linux host only):**

```
ls /dev/video*                      # if it is not video0, change the devices: line in docker-compose.yml
xhost +local:docker
docker compose --profile gui run --rm trainer       # training tool
docker compose --profile gui run --rm live         # original live system
```

**Where data goes:** `./data` on the host is mounted into the container, so sessions, consent records and settings
are kept on the host. They are never part of the image, so a shared image contains no participant data.

**If something fails:**

| Symptom | Fix |
|---|---|
| `docker` not recognized, or cannot connect to the Docker API | Docker Desktop is not installed or not running. Open it and wait until it reports running; use a new terminal. |
| `Read-only file system` or `Bus error` during the build | The drive Docker uses is full. Free about 20 GB, or move Docker's disk image to another drive. |
| `apt-get` shows `Ign:` lines and stalls | No internet from inside the container. Turn off any VPN or set Docker's DNS to 8.8.8.8 and 1.1.1.1. |
| `load build context` shows hundreds of MB or GB | `.dockerignore` is missing or misnamed. It must be exactly `.dockerignore`. |
| Tests stop with a Qt `xcb` display error | The virtual screen is missing. Use `docker compose run --rm tests` (it starts one). |
| `--progress is a global compose flag` | Put it before the command: `docker compose --progress plain build`. |

## 9. Limits to state honestly in the paper

- **Camera and microphone.** A container has no camera or microphone of its own. On Windows, Docker Desktop cannot
  pass a webcam through (the default WSL 2 kernel has no webcam driver, and Camo is a virtual camera). The live
  window with the camera and real whistle detection works with Docker only on a Linux host. On Windows the system
  still runs natively with `python main.py`. Docker on Windows demonstrates the reproducible environment and the
  automated tests, not a live demonstration.
- **The tests do not use real whistle sound.** They check the whistle rules, timing and the model loading.
  Real-microphone whistle detection was not tested inside Docker.
- **The finish beep** of the offline extraction script is silent in Docker (the winsound stand-in).
- **The model/version mismatch** (section 5) exists in the repository and was not changed.
- **Single test machine.** Everything verified was run on one Windows laptop.

## 10. Suggested wording for the paper

Edit this to match section 7 at the time of writing.

> The system was packaged as a Docker image based on Python 3.11. The image installs the dependencies pinned in
> `requirements.txt` (with CPU-only PyTorch) and contains the application code and trained models; no application
> source file was modified. Because the code imports `winsound`, a Windows-only module, a minimal stand-in module
> was added inside the image so that the code runs unchanged on Linux. The image builds with `docker compose build`,
> and the project's automated tests (the trainer smoke test and the gesture-grader tests) pass inside the
> container. A virtual display is used inside the container so that the window-based tests can run without a
> screen. Camera and microphone access from a container is not available under Docker Desktop on Windows; live use
> with a camera and microphone is supported natively on Windows and through Docker on a Linux host. Real-microphone
> whistle detection was not tested inside the container.

## 11. Files to hand over

Branch `docker` should contain: `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `docker.md`, `DOCKER_TESTING.md`
(optionally this handoff and the GitHub Actions workflow in `DOCKER_TESTING.md`, Part 7). No existing source file is
changed. There is no `extract_keypoints_winsound.patch`; it is not used. Do not commit the `data/` folder.