# Running the project with Docker

Instructions for packaging the Multimodal Officiating Training Tool as a Docker image, so anyone (including the
dean) can run it without installing Python, MediaPipe or PyTorch by hand.

Repository: https://github.com/g1000n/multimodal-volleyball-officiating (branch `main`)

---

## 0. Read this first: what Docker can and cannot do here

The project is a desktop app: Tkinter menus, an OpenCV camera window, a webcam and a microphone. A container has no
screen, camera or microphone of its own, so it has to borrow them from the host.

| Host computer | Tests / headless | Training tool window | Camera + microphone |
|---|---|---|---|
| **Linux** | Yes | Yes (X11) | Yes (`/dev/video0`, `/dev/snd`) |
| **Windows (Docker Desktop)** | Yes | Only through WSLg, see section 7 | **No** in practice (Docker Desktop cannot see a webcam, and Camo is a virtual camera, not a USB device) |
| **macOS** | Yes | Needs XQuartz | No |

What this means for the dean's requirement:

- **Reproducible environment, every OS:** yes. `docker compose run --rm tests` builds the exact environment and runs
  the project's own automated tests (35 smoke-test checks plus the grader tests).
- **Full live demo with camera and microphone:** best done on a Linux host, or natively on Windows as you do now.
  Say this plainly in your documentation; it is a property of Docker, not a defect in the project.

---

## 1. One-time code fix (required)

`live_deployment.py` imports `extract_keypoints.py`, which starts with `import winsound`. `winsound` exists only on
Windows, so the app would crash on startup inside a Linux container with
`ModuleNotFoundError: No module named 'winsound'`.

The fix makes that import optional (on Windows nothing changes; on Linux the finish beep is skipped). It is in
`extract_keypoints_winsound.patch`.

From the repository root:

```bash
git apply extract_keypoints_winsound.patch
python -c "import ast; ast.parse(open('extract_keypoints.py').read()); print('ok')"
```

This is a code change, so per `docs/HANDOFF.md` §14 run the tests before committing (section 4 below does this inside
Docker; you can also run the Windows venv tests as usual).

---

## 2. Add the Docker files to the repository

Copy these three files into the **repository root** (next to `main.py` and `requirements.txt`):

```
Dockerfile
docker-compose.yml
.dockerignore
```

Optionally also copy this file to `docs/DOCKER.md` and link it from `README.md`.

Commit and push to `main`:

```bash
git add Dockerfile docker-compose.yml .dockerignore extract_keypoints.py docs/DOCKER.md
git commit -m "Add Docker support (Dockerfile, compose file, winsound import guard)"
git push origin main
```

What the files do:

| File | Purpose |
|---|---|
| `Dockerfile` | Python 3.11 base, system libraries (OpenCV, Tk, audio), CPU-only PyTorch, `requirements.txt`, then the code and models |
| `docker-compose.yml` | Shortcuts: `tests`, `trainer` (the app), `live` (the original live system) |
| `.dockerignore` | Keeps `.git`, raw video, and **trainee data** out of the image |

---

## 3. Install Docker

- **Windows / macOS:** install Docker Desktop (https://www.docker.com/products/docker-desktop/). On Windows enable
  the WSL 2 backend when asked.
- **Linux:** install Docker Engine and the Compose plugin
  (https://docs.docker.com/engine/install/), then allow your user to run it without `sudo`:
  `sudo usermod -aG docker $USER` and log out and in again.

Check:

```bash
docker --version
docker compose version
```

You need about **20 GB of free disk space** on the drive Docker uses (C: on Windows). If the drive fills up, Docker's
disk turns read-only and builds fail with `Read-only file system` or `Bus error`. On Windows, open Docker Desktop and
wait until it says the engine is running before using any `docker` command.

`DOCKER_TESTING.md` is a step-by-step checklist for verifying the build on a machine with enough space.

---

## 4. Build and verify (any operating system)

From the repository root:

```bash
docker compose build
```

The first build downloads PyTorch and MediaPipe and takes several minutes. Later builds reuse the cached layers
unless `requirements.txt` changed.

Then run the project's automated tests inside the container:

```bash
docker compose run --rm tests
```

Expected output (from `README.md`): 35 `OK` lines and `Smoke test passed.`, then all grader tests `PASS` and
`All good.` If you see that, the container environment is correct. This is the command to show the dean.

---

## 5. Run the training tool (Linux host with camera and microphone)

1. Find your camera: `ls /dev/video*`. If it is not `/dev/video0`, change that line in `docker-compose.yml`.
2. Allow the container to open windows on your screen (once per login):

   ```bash
   xhost +local:docker
   ```
3. Start the training tool:

   ```bash
   docker compose --profile gui run --rm trainer
   ```

The welcome and consent screen appears first, then **Camera and mic** (choose the camera, and the microphone under
the ALSA / PulseAudio entries; the "DirectSound" advice in `docs/HANDOFF.md` is Windows-only).

The original live officiating system uses the same image:

```bash
docker compose --profile gui run --rm live
```

When finished, you can undo the X permission with `xhost -local:docker`.

---

## 6. Where your data goes

`docker-compose.yml` mounts the host folder `./data` onto `/app/data` in the container. Sessions
(`data/trainer_sessions/`), consent records and settings are therefore saved **on the host**, and survive the
container being deleted. They are deliberately excluded from the image by `.dockerignore`, so images you share
never contain participants' data.

The trained models (`models/final_model.pt`, `models/whistle_svm_model.pkl`) are baked into the image.

---

## 7. Windows notes

**Tests:** `docker compose run --rm tests` works on Windows exactly as in section 4.

**The window:** Docker Desktop with the WSL 2 backend includes WSLg, which can display Linux GUI windows. Run the
compose commands from inside a WSL 2 Ubuntu terminal (not PowerShell) with the repository cloned inside the WSL
filesystem, and add these to the `trainer` service:

```yaml
    environment:
      - DISPLAY=:0
      - WAYLAND_DISPLAY=wayland-0
      - XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir
    volumes:
      - /mnt/wslg:/mnt/wslg
      - /tmp/.X11-unix:/tmp/.X11-unix
```

**The camera:** not available this way. The default WSL 2 kernel has no webcam (UVC) driver, and Camo appears to
Windows as a virtual camera rather than a USB device. Getting a webcam into a container on Windows means building a
custom WSL kernel and using `usbipd-win`, which is not worth it for this project. For a live demo on Windows, run
`python main.py` natively as in `README.md`, and use Docker to show the reproducible build and tests.

---

## 8. Giving the image to the dean

**Option A: the dean builds it from GitHub (simplest, smallest)**

```bash
git clone https://github.com/g1000n/multimodal-volleyball-officiating
cd multimodal-volleyball-officiating
docker compose build
docker compose run --rm tests
```

**Option B: send a ready-made image file**

```bash
docker save volleyball-officiating:latest | gzip > volleyball-officiating.tar.gz
```

The dean loads and checks it with:

```bash
docker load < volleyball-officiating.tar.gz
docker run --rm volleyball-officiating:latest sh -c "python tests/test_trainer_smoke.py && python tests/test_gesture_grader.py"
```

The file will be large (likely a couple of GB, because of PyTorch, MediaPipe and OpenCV). Check it contains no participant data before sharing:
`docker run --rm volleyball-officiating:latest ls /app/data` should not list `trainer_sessions`.

**Option C: publish to a registry** (Docker Hub or GitHub Container Registry), then the dean only runs
`docker pull` and `docker run`.

---

## 9. Useful commands

| Task | Command |
|---|---|
| Build | `docker compose build` |
| Rebuild from scratch | `docker compose build --no-cache` |
| Run tests | `docker compose run --rm tests` |
| Training tool | `docker compose --profile gui run --rm trainer` |
| Open a shell in the container | `docker compose run --rm tests bash` |
| List cameras the container sees | `docker compose --profile gui run --rm trainer python trainer.py --list-cameras` |
| Remove the image | `docker rmi volleyball-officiating:latest` |

---

## 10. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'winsound'` | The patch in section 1 was not applied (or not committed before building). |
| `Could not start the recognition backend` window | Camera not passed in. Check `ls /dev/video*` and the `devices:` lines in `docker-compose.yml`. |
| `no matching manifest` / `torch==2.12.1+cpu` not found during build | The CPU wheel for that exact version is not on the PyTorch index. Remove the first `pip install torch ... --index-url` line from the `Dockerfile` so torch comes from PyPI (larger image, same behaviour). |
| `cannot connect to X server` / `Can't open display` | Run `xhost +local:docker` first, and check `echo $DISPLAY` is not empty in the terminal you launch from. |
| Window appears but is blank or crashes with a Qt xcb error | A system library is missing. Run `docker compose run --rm tests bash`, then `python -c "import cv2"` and read the error; add the named `libxcb-*` package to the `apt-get install` list in the `Dockerfile`. |
| `ImportError: libtk8.6.so` or `No module named _tkinter` | The Tk libraries are missing from the image; confirm `libtk8.6 libtcl8.6` are in the `Dockerfile`. |
| Whistle test never hears a whistle | The container is not seeing the right microphone. Check `ls /dev/snd`; on a PulseAudio/PipeWire desktop you may also need to mount the Pulse socket (`-v /run/user/1000/pulse:/run/user/1000/pulse`). `W` still works as a manual whistle. |
| `permission denied` on `/dev/video0` | Add your user to the `video` group on the host (`sudo usermod -aG video $USER`) and log in again. |
| `is not a known BitGenerator module` | The wrong numpy got installed; `requirements.txt` pins `numpy==2.2.6`. Rebuild with `--no-cache`. |
| Tests stop with `qt.qpa.xcb: could not connect to display` | A test opens a window and the container has no screen. The `Dockerfile` must install `xvfb xauth` and the `tests` command in `docker-compose.yml` must start with `xvfb-run -a`. |
| `Read-only file system` or `Bus error (core dumped)` during the build | Docker's disk filled up. Free about 20 GB, run `docker system prune -af`, then Docker Desktop, bug icon, **Clean / Purge data**, then rebuild. |
| `load build context` shows hundreds of MB or GB | `.dockerignore` is missing or misnamed (`dockerignore` without the dot, or `.dockerignore.txt`). It must be named exactly `.dockerignore`. |
| `--progress is a global compose flag` | Put it before the command: `docker compose --progress plain build`. |
| `docker version` shows an empty `Server:` / `hello-world` hangs | Engine stuck. Quit Docker Desktop, run `wsl --shutdown`, reopen it. If still stuck, Troubleshoot, **Clean / Purge data**. |