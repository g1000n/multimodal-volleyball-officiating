# Multimodal Officiating Training Tool for Volleyball
# Python 3.11 matches the project's venv (3.11.9); mediapipe 0.10.14 does not support 3.12+.
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System libraries the Python packages need at runtime:
#   OpenCV (cv2.imshow window) : libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 + the libxcb-* set
#   Tkinter (welcome/menu)     : libtk8.6 libtcl8.6 (slim images ship _tkinter but not the Tk libraries)
#   Whistle / microphone       : libportaudio2 (sounddevice), libsndfile1 (soundfile)
#   "Open report/video" button : xdg-utils
# ffmpeg is NOT installed: nothing the trainer runs needs the binary. Add `ffmpeg` here if a script in tools/ does.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxkbcommon-x11-0 \
        libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
        libxcb-shape0 libxcb-xfixes0 libxcb-xinerama0 \
        libtk8.6 libtcl8.6 fonts-dejavu-core \
        libportaudio2 libsndfile1 \
        xdg-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so Docker caches this layer until requirements.txt changes.
# torch from the CPU-only index: the default Linux wheel pulls ~2 GB of CUDA libraries the project never uses.
# requirements.txt pins torch==2.12.1, which the "+cpu" build satisfies.
COPY requirements.txt .
RUN pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# Application code and the trained models (models/*.pt and *.pkl are in git).
COPY . .

# Sessions, consent records and settings are written to /app/data; docker-compose mounts ./data there
# so they survive the container being removed.
RUN mkdir -p /app/data

# Same as running `python main.py` on Windows: opens the training tool.
CMD ["python", "main.py"]
