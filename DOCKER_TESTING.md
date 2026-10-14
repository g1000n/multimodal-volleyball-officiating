# Testing the Docker setup

A checklist for verifying that the Docker setup (`Dockerfile`, `docker-compose.yml`, `.dockerignore`; the `Dockerfile`
includes a small winsound stand-in) really builds and passes the project's tests. It is meant for a computer that has
the disk space the original laptop did not, such as a teammate's laptop, a Linux machine, or GitHub's own servers
(Part 7).

Repository: https://github.com/g1000n/multimodal-volleyball-officiating

---

## 0. Where things stand

| Item | Status |
|---|---|
| Every pinned package in `requirements.txt` has a Linux wheel for Python 3.11 | Checked |
| `docker-compose.yml` is valid | Checked |
| On the original laptop the build got through the `apt` step and nearly all `pip` downloads | Seen working |
| Full image build to the end | **Done** (built successfully once Docker had enough disk space) |
| `docker compose run --rm tests` passing inside the container | **Done on one Windows laptop**: 35 `OK` lines, `Smoke test passed.`, all grader tests `PASS`, `All good.` Re-run it on the machine you are testing. |
| `import extract_keypoints` works in the container (winsound stand-in) | **To confirm** with the extra check in Part 5 |
| Training tool window with camera and microphone (Linux host only) | **Not yet done** |

The goal of this document is to finish the remaining rows and to leave evidence (Part 9). The `Dockerfile` must contain
the `xvfb xauth` install line and the `tests` service in `docker-compose.yml` starts `Xvfb :99`; without a virtual
screen the tests abort with a Qt "could not connect to display" error.

---

## 1. What you need

- **About 20 GB of free disk space** on the drive Docker uses (C: on Windows by default). Docker's disk is remounted
  read-only when the drive fills up, which stops a build with `Read-only file system` or `Bus error`.
- **Docker Desktop** (Windows, with WSL 2) or **Docker Engine + Compose plugin** (Linux). A reliable internet
  connection: the first build downloads roughly 1 to 2 GB.
- **Git**.

No Python, MediaPipe or PyTorch installation is needed on the host. That is the point of Docker.

Choose one place to run this:

| Option | Good for |
|---|---|
| A. A teammate's Windows laptop with enough free space | Quick, same setup as the team uses |
| B. A Linux computer or VM | Also lets you test the training tool window (Part 6) |
| C. GitHub Actions (Part 7) | No local space needed; leaves a public green check |

---

## 2. Get the files in place

If the Docker files are already pushed:

```powershell
git clone -b docker https://github.com/g1000n/multimodal-volleyball-officiating
cd multimodal-volleyball-officiating
```

The Docker files are on branch `docker` (the default branch `main` does not have them until it is merged). Always
test from a **fresh clone** like this, not from a working folder: the image copies whatever is on disk, so
uncommitted or untracked files (old model folders, evaluation records with participant codes) would be baked in.

Then confirm the things that went wrong the first time:

**a. The Docker files exist, and `.dockerignore` starts with a dot.**

```powershell
dir -Force | findstr /i docker
```

Expected: `.dockerignore` (a few hundred bytes), `docker-compose.yml`, `Dockerfile`. A file called `dockerignore`
(no dot) or `.dockerignore.txt` is ignored by Docker, and the build will try to copy gigabytes of data into the image.

**b. `.dockerignore` is the allowlist version** (it begins with `*` and then lists `!` exceptions):

```powershell
Get-Content .dockerignore
```

**c. The winsound stand-in is in the `Dockerfile`** (no change to `extract_keypoints.py` is needed):

```powershell
Select-String "winsound" Dockerfile
```

This should print the line that creates `winsound.py`. Without it the live system crashes in the container with
`No module named 'winsound'`.

---

## 3. Pre-flight

```powershell
docker --version
docker compose version
docker run --rm hello-world
```

`hello-world` must print "Hello from Docker!". If the command hangs, or `docker version` shows an empty `Server:`
section, the Docker engine is stuck. See the troubleshooting table in Part 8 before continuing.

Check the free space on the Docker drive:

```powershell
Get-PSDrive C | Select-Object Name,@{n='FreeGB';e={[math]::Round($_.Free/1GB,1)}}
```

---

## 4. Build the image

```powershell
docker compose --progress plain build
```

Note the order: `--progress plain` goes **before** `build`. The `DISPLAY variable is not set` warnings are harmless on
Windows.

What a healthy build looks like:

| Step | What you should see |
|---|---|
| `load build context` | Finishes in seconds, **under about 30 MB**. If it shows hundreds of MB or GB, `.dockerignore` is wrong (Part 2). Press Ctrl+C. |
| `apt-get install` (step 2/8) | A couple of minutes, ending in `DONE`. |
| `pip install torch ... whl/cpu` | Downloads a few hundred MB. |
| `pip install -r requirements.txt` | The longest step (MediaPipe, OpenCV, SciPy, scikit-learn, JAX). Quiet stretches are normal. |
| `COPY . .` and the final steps | A few seconds. |

The first build usually takes **10 to 20 minutes**. Later builds reuse cached layers and take seconds.

It passes if the build ends with no `ERROR` lines. Then check the size:

```powershell
docker images volleyball-officiating
```

---

## 5. Run the tests inside the container

```powershell
docker compose run --rm tests
```

Expected (from the project README):

- the smoke test prints **35 `OK` lines** and `Smoke test passed.`
- the grader test shows all `PASS` and `All good.`

This is the main proof that the container environment is correct.

### Extra checks (quick, and good evidence)

Run each one; the expected result is in the comment.

```powershell
# the winsound stand-in really works: the real recognition code imports on Linux
docker compose run --rm tests python -W ignore -c "import extract_keypoints, whistle_detector; print('imports OK')"
# expected: imports OK

# the interactive decision-engine tester (needs no camera); type: w, status, quit
docker compose up -d app
docker exec -it volleyball-officiating python tests/interactive_decision_engine_test.py

# the trainer starts and reports no camera (a container has none)
docker compose run --rm tests python trainer.py --list-cameras
# expected: "Checking cameras..." then "no working camera found"

# trained models are inside the image
docker compose run --rm tests ls -la models
# expected: final_model.pt, whistle_svm_model.pkl, label_map.json

# no participant data is inside the image
docker compose run --rm tests ls /app/data
# expected: nothing listed (or an empty folder), and never trainer_sessions
```

---

## 6. Training tool window (Linux host with camera and microphone only)

On Windows this is not possible (Docker Desktop cannot access the webcam or Camo). On a Linux computer:

```bash
ls /dev/video*                # note your camera, change the devices: line in docker-compose.yml if it is not video0
xhost +local:docker
docker compose --profile gui run --rm trainer
```

Check: the welcome and consent screen opens, **Camera and mic** lists the camera, and Practice mode shows the
skeleton overlay. Undo the permission afterwards with `xhost -local:docker`.

---

## 7. Option C: let GitHub test it (no local disk space needed)

Add this file to the repository as `.github/workflows/docker-test.yml`:

```yaml
name: docker-build-and-test

on:
  push:
    branches: [main, docker]
  pull_request:
  workflow_dispatch:

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build the image
        run: docker compose build tests

      - name: Run the project tests inside the container
        run: docker compose run --rm tests

      - name: Check the real code imports (winsound stand-in)
        run: docker compose run --rm tests python -c "import extract_keypoints, whistle_detector; print('imports OK')"
```

Commit and push it. Then open the repository on GitHub, click the **Actions** tab, and open the latest
`docker-build-and-test` run. A green check means the image builds and the tests pass on a clean machine. If it is red,
open the failed step and read the log. This workflow has not been run yet.

---

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `docker : The term 'docker' is not recognized` | Docker Desktop is not installed, or the terminal was opened before installing. Install it and open a **new** terminal. |
| `failed to connect to the docker API ... docker_engine` | Docker Desktop is not running. Open it and wait until the engine says running. |
| `docker version` shows an empty `Server:` section, or `hello-world` hangs | Engine stuck. Quit Docker Desktop, run `wsl --shutdown`, reopen it. If still stuck: Docker Desktop, bug icon (Troubleshoot), **Clean / Purge data**. |
| `Read-only file system`, `Bus error (core dumped)`, `failed to solve: write ...` | The Docker disk filled up (usually C:). Free 20 GB, run `docker builder prune -af` and `docker system prune -af`, then **Clean / Purge data**, then rebuild. |
| `load build context` shows hundreds of MB or GB | `.dockerignore` is missing, misnamed (`dockerignore` or `.dockerignore.txt`), or not the allowlist version. Fix it (Part 2) and rebuild. |
| `--progress is a global compose flag` | Wrong order. Use `docker compose --progress plain build`. |
| `apt-get update` shows `Ign:` lines and fails | The container cannot reach the internet. Turn off any VPN. In Docker Desktop, Settings, Docker Engine, add `"dns": ["8.8.8.8", "1.1.1.1"]`, Apply & restart. |
| `torch==2.12.1+cpu` not found | The CPU wheel for that exact version is not on the PyTorch index. Delete the first `pip install torch ... --index-url ...` line from the `Dockerfile` so torch comes from PyPI (bigger image, same behaviour). |
| `No module named 'winsound'` | The stand-in line is missing from the `Dockerfile` that was built. Use the current `Dockerfile` and rebuild. |
| Tests stop with `qt.qpa.xcb: could not connect to display` and `Aborted (core dumped)` | A test (Match Testing) opens a window and the container has no screen. The `Dockerfile` must install `xvfb xauth`, and the `tests` command in `docker-compose.yml` must start `Xvfb :99`. Rebuild. |
| `is not a known BitGenerator module` | Wrong numpy. `requirements.txt` pins `numpy==2.2.6`. Rebuild with `docker compose build --no-cache`. |

---

## 9. Report back

When finished, copy this block, fill it in, and send it to whoever needs the proof (the author, the adviser, the dean):

```
Computer / OS:
Docker version (docker --version):
Free disk space before the build:

[ ] .dockerignore starts with a dot and is the allowlist version
[ ] built from a fresh clone of branch docker; winsound stand-in works (imports OK)
[ ] docker run --rm hello-world worked
[ ] docker compose build finished with no ERROR   (time taken: ___ min)
[ ] Image size (docker images volleyball-officiating): ___ GB
[ ] docker compose run --rm tests: 35 OK + "Smoke test passed." + "All good."
[ ] imports OK check passed
[ ] no trainer_sessions inside the image
[ ] (Linux only) training tool window opened with the camera

Problems or errors (paste the first ERROR line and the lines before it):
```

---

## 10. After it passes

1. Merge to `main` if the work was on a branch.
2. Add a short "Docker" section to `README.md` pointing to `docs/docker.md`.
3. Give the dean the repository link and the three commands:

   ```bash
   git clone -b docker https://github.com/g1000n/multimodal-volleyball-officiating
   cd multimodal-volleyball-officiating
   docker compose up -d --build app && docker ps && docker compose run --rm tests
   ```