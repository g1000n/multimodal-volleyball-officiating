# Running the Training Tool with Docker

Docker packages the training tool with Python 3.11, every library it needs, and the trained models
(`models/final_model.pt` for gestures, `models/whistle_svm_model.pkl` for the whistle). No dataset or participant data
is inside the image: `.dockerignore` only lets in the code, the models, `assets/`, `tests/`, `tools/`, `scripts/`,
`diagnostics/` and `docs/`.

## What works where

| Host | Automated tests | Training tool window | Camera and microphone |
|---|---|---|---|
| **Windows / macOS** (Docker Desktop) | Yes | Yes on Windows through WSLg | Not directly: Docker Desktop cannot pass a webcam or microphone into a Linux container. **Yes through the host bridge** (next section) |
| **Linux** | Yes | Yes (X11) | Yes (`/dev/video0`, `/dev/snd`) |

On Windows the camera and microphone reach the container through `tools/av_bridge.py` (below). You can also run the
tool natively (see `README.md`: venv, `pip install -r requirements.txt`, `py main.py`).

## Windows / macOS: camera and microphone through the host bridge

Docker Desktop runs containers in a small Linux VM that cannot see USB devices. `tools/av_bridge.py` runs on the
**host** (your normal venv), reads the camera and the microphone, and serves them on two local ports; the container
reads them from there. Camo works, because the host already sees it as a camera and a microphone.

0. **On the host, once: get the bridge.** Two ways; the first needs no Python.

   **No Python (Windows):** the image carries a ready-made `av_bridge.exe`. Copy it out into the current folder:

   ```
   docker run --rm -v "${PWD}:/out" g1000n/multimodal-volleyball-officiating:1.2 cp bridge/av_bridge.exe /out/
   ```

   Windows may warn about an unknown publisher (More info, then Run anyway), and some antivirus programs flag
   programs built with PyInstaller. If so, use the Python route.

   **With Python 3.11:** the bridge needs three packages (the container does the recognition; the bridge is only a
   device relay): `pip install -r requirements-bridge.txt`. If you already use the project venv, you have them.
1. **On the host**, start the bridge (`.\av_bridge.exe`, or in the project folder with the venv active):

   ```
   py tools/av_bridge.py
   ```

   It lists this computer's cameras and microphones once at startup and serves all of them. You do **not** have to
   choose here: the app's **Camera and mic** screen lists every camera and microphone of the host and switches
   between them. `--camera N` and `--mic N` only set the defaults (`py tools/av_bridge.py --list` shows the
   numbers). In the app pick the microphone's **DirectSound** entry (for example `Microphone (Camo) [Windows
   DirectSound]`); the WASAPI and WDM-KS entries often fail to open, as in `docs/HANDOFF.md`. Allow the Windows
   Firewall prompt for private networks. Leave the window open. A camera that another program is using (Zoom,
   Camo's own app) cannot be opened; close that program first.
2. **In another terminal**, start the training tool in the container:

   ```
   docker compose --profile bridge run --rm trainer-bridge
   ```

   The `trainer-bridge` service sets `VB_CAMERA_URL=http://host.docker.internal:8765/video` and
   `VB_MIC_URL=tcp://host.docker.internal:8766`. In **Camera and mic** you will see the host's own cameras and
   microphones; choosing one switches the bridge to it, and the whistle test works as usual.
3. Stop the bridge with Ctrl+C when you are done. While it runs, anyone who can reach ports 8765 and 8766 on your
   computer sees the camera and hears the microphone, so use a trusted network. Safer: start the bridge with
   `--bind 127.0.0.1` and check that the container still connects (Docker Desktop usually forwards
   `host.docker.internal` to the host's loopback); if it does not, drop the flag. If the bridge stops during a
   session, the camera follows the usual "camera lost" handling and the whistle detector logs
   "the microphone bridge stopped" (the `W` key still works as a whistle).

Only the **training tool** (`main.py`) uses the bridge. The original live system (`main.py live`,
`live_deployment.py`) still opens the camera directly. `python tests/test_av_bridge.py` checks the bridge without any
camera. With the published image, use the same variables:
`docker run --rm -it -e VB_CAMERA_URL=http://host.docker.internal:8765/video -e VB_MIC_URL=tcp://host.docker.internal:8766 <display options> g1000n/multimodal-volleyball-officiating:<tag> python main.py`.
The published `1.0` and `1.1` images do not contain this support; it ships from `1.2` (build from this branch or
`main` until then). Feedback times quoted in the paper were measured natively, not through the bridge, which adds
roughly 0.1 s.

## Option A: use the published image (fastest)

```
docker pull g1000n/multimodal-volleyball-officiating:1.1
docker run --rm g1000n/multimodal-volleyball-officiating:1.1 sh -c "Xvfb :99 -screen 0 1280x720x24 -nolisten tcp & sleep 3 && DISPLAY=:99 python tests/test_trainer_smoke.py && DISPLAY=:99 python tests/test_gesture_grader.py"
```

Expected: 35 `OK` lines, `Smoke test passed.`, the grader `PASS` lines, then `All good.`

## Option B: build it yourself

```
git clone https://github.com/g1000n/multimodal-volleyball-officiating
cd multimodal-volleyball-officiating
docker compose build
docker compose run --rm tests
```

The first build downloads about 1 to 2 GB (torch is installed from the CPU-only index) and takes 15 to 40 minutes;
later builds reuse the cached layers. Build from a fresh clone, not from a working folder: the Dockerfile copies the
folder as it is on disk.

## Showing that it is installed and running

```
docker compose up -d app
docker ps
docker exec volleyball-officiating python -W ignore -c "import extract_keypoints, trainer; print('imports OK')"
docker exec -it volleyball-officiating python tests/interactive_decision_engine_test.py
```

`docker ps` lists the container `volleyball-officiating` as running. The interactive test needs no camera: at the
`>` prompt type `w` (a whistle), a signal name, `status`, then `quit`. Stop the container with
`docker compose down`.

## Linux only: the training tool with camera and microphone

```
xhost +local:docker
docker compose --profile gui run --rm trainer
```

The `trainer` service passes `/dev/video0` (camera) and `/dev/snd` (microphone, ALSA) into the container and shows the
window on the host screen. On desktops that use PulseAudio or PipeWire, add
`- /run/user/1000/pulse:/run/user/1000/pulse` under `volumes` and `- PULSE_SERVER=unix:/run/user/1000/pulse/native`
under `environment` if the whistle test hears nothing. Sessions are saved to `./data` on the host.

## How the image is built (Dockerfile)

- `python:3.11-slim-bookworm`, because mediapipe 0.10.14 does not support Python 3.12 or later.
- System libraries for OpenCV, Tkinter, PortAudio and libsndfile.
- `torch==2.12.1` from the CPU-only index, then `requirements.txt` (numpy 2.2.6 is pinned there; an older numpy stops
  the whistle detector).
- `xvfb`, a virtual screen for the automated tests (Match Testing opens an OpenCV window).
- A small `winsound` stand-in: `extract_keypoints.py` imports the Windows-only `winsound` module, and the stand-in lets
  the unchanged code import on Linux. No code change or patch is needed.

## Troubleshooting

| Problem | Fix |
|---|---|
| `docker compose build` fails on the torch line | Check the network, then build again; the CPU wheel is downloaded from download.pytorch.org |
| Tests hang or fail with a display error | Use `docker compose run --rm tests`, which starts the `Xvfb :99` virtual screen first |
| `DISPLAY` warning on Windows | Harmless; the GUI services are for Linux only |
| No camera inside the container on Windows | Expected; run the tool natively on Windows instead |
