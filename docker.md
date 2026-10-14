# Running the Training Tool with Docker

Docker packages the training tool with Python 3.11, every library it needs, and the trained models
(`models/final_model.pt` for gestures, `models/whistle_svm_model.pkl` for the whistle). No dataset or participant data
is inside the image: `.dockerignore` only lets in the code, the models, `assets/`, `tests/`, `tools/`, `scripts/`,
`diagnostics/` and `docs/`.

## What works where

| Host | Automated tests | Training tool window | Camera and microphone |
|---|---|---|---|
| **Windows / macOS** (Docker Desktop) | Yes | No | No: Docker Desktop cannot pass a webcam or microphone into a Linux container |
| **Linux** | Yes | Yes (X11) | Yes (`/dev/video0`, `/dev/snd`) |

To use the camera and microphone on Windows, run the tool natively instead (see `README.md`: venv,
`pip install -r requirements.txt`, `py main.py`).

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
