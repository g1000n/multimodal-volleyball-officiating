"""
av_bridge.py -- serve this computer's camera and microphone over the network, for the Docker container.

Docker Desktop on Windows/macOS cannot pass a webcam or microphone into a container. Run this on the HOST (the
normal Windows venv), then start the container with the two URLs it prints. Nothing else changes: the training
tool reads the camera and the microphone through devices.py, which uses the URLs when they are set.

    py tools/av_bridge.py                      camera 0 + default microphone
    py tools/av_bridge.py --list               list cameras and microphones, then exit
    py tools/av_bridge.py --camera 1 --mic 3   pick them (numbers from --list; Camo shows up as a camera and a mic)
    py tools/av_bridge.py --video clip.mp4     use a video file instead of a camera (rehearsal / tests)

Video : MJPEG over HTTP, http://<host>:8765/video       (also one picture at /frame.jpg)
Audio : raw mono float32 over TCP, <host>:8766           (the client sends the sample rate it wants first)

Privacy: the camera and microphone are served to anyone who can reach these ports. Use it on a trusted network
only, stop it when you are done (Ctrl+C), and use --bind 127.0.0.1 when the container is not involved.
"""

import argparse
import os
import socketserver
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import devices  # noqa: E402


class FrameSource:
    """Keeps the newest JPEG from a camera (or a looping video file) in memory."""

    def __init__(self, camera=0, video=None, fps=25.0, quality=80):
        self.camera, self.video, self.fps, self.quality = camera, video, fps, quality
        self.jpeg = None
        self.size = (0, 0)
        self._stop = threading.Event()
        self.cap = None

    def _open(self):
        if self.video:
            if not os.path.isfile(self.video):
                raise RuntimeError(f"video file not found: {self.video}")
            return cv2.VideoCapture(self.video)
        cap = devices.open_camera_index(self.camera)
        if cap is None:
            raise RuntimeError(f"camera {self.camera} could not be opened (try --list)")
        return cap

    def start(self):
        self.cap = self._open()
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("the camera/video gave no picture")
        self._store(frame)
        threading.Thread(target=self._loop, daemon=True).start()

    def _store(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if ok:
            self.jpeg = buf.tobytes()
            self.size = (frame.shape[1], frame.shape[0])

    def _loop(self):
        delay = 1.0 / self.fps
        while not self._stop.is_set():
            t0 = time.time()
            ok, frame = self.cap.read()
            if not ok:
                if self.video:                      # loop a video file
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.05)
                continue
            self._store(frame)
            if self.video:                          # a file would otherwise play as fast as it can be read
                time.sleep(max(0.0, delay - (time.time() - t0)))

    def stop(self):
        self._stop.set()
        try:
            self.cap.release()
        except Exception:
            pass


def make_http_handler(source):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/frame.jpg"):
                body = source.jpeg or b""
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if not self.path.startswith("/video"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            last = None
            try:
                while True:
                    jpeg = source.jpeg
                    if jpeg is not None and jpeg is not last:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                        last = jpeg
                    time.sleep(0.01)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
    return Handler


class AudioSource:
    """Feeds mono float32 blocks at the requested rate. `synthetic` makes a quiet tone (tests / rehearsal)."""

    def __init__(self, device=None, synthetic=False):
        self.device, self.synthetic = device, synthetic

    def stream(self, rate, blocksize, write):
        """Calls write(bytes) for every block until write raises. Blocks the calling thread."""
        if self.synthetic:
            t = 0
            while True:
                n = np.arange(t, t + blocksize)
                write((0.2 * np.sin(2 * np.pi * 440.0 * n / rate)).astype("<f4").tobytes())
                t += blocksize
                time.sleep(blocksize / rate)
        import sounddevice as sd
        failed = []

        def cb(indata, frames, ti, status):
            try:
                write(indata[:, 0].astype("<f4").tobytes())
            except Exception as exc:
                failed.append(exc)

        with sd.InputStream(device=self.device, channels=1, samplerate=rate, blocksize=blocksize, callback=cb):
            while not failed:
                time.sleep(0.1)


class AudioHandler(socketserver.BaseRequestHandler):
    source = None

    def handle(self):
        try:
            rate = struct.unpack("<I", self.request.recv(4))[0]
            rate = rate if 8000 <= rate <= 96000 else 22050
            sock = self.request
            self.source.stream(rate, 1024, lambda b: sock.sendall(b))
        except Exception:
            pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="Serve the camera and microphone to the Docker container.")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--mic", type=int, default=None, help="sounddevice input index (see --list)")
    ap.add_argument("--video", help="serve this video file instead of a camera")
    ap.add_argument("--synthetic-audio", action="store_true", help="a quiet test tone instead of the microphone")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--video-port", type=int, default=8765)
    ap.add_argument("--audio-port", type=int, default=8766)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        print("Cameras:")
        for c in devices.list_cameras():
            print(f"  --camera {c['index']}   {c['width']}x{c['height']}")
        mics, err = devices.list_microphones()
        print("Microphones:" + (f"  ({err})" if err else ""))
        for m in mics:
            print(f"  --mic {m['index']}   {m['name']}  [{m['hostapi']}]{'  (default)' if m['default'] else ''}")
        return 0

    source = FrameSource(args.camera, args.video)
    source.start()
    http = ThreadingHTTPServer((args.bind, args.video_port), make_http_handler(source))
    threading.Thread(target=http.serve_forever, daemon=True).start()
    print(f"Video : http://{args.bind}:{args.video_port}/video   ({source.size[0]}x{source.size[1]})")

    audio = None
    if not args.no_audio:
        AudioHandler.source = AudioSource(args.mic, args.synthetic_audio)
        socketserver.ThreadingTCPServer.allow_reuse_address = True
        audio = socketserver.ThreadingTCPServer((args.bind, args.audio_port), AudioHandler)
        audio.daemon_threads = True
        threading.Thread(target=audio.serve_forever, daemon=True).start()
        print(f"Audio : tcp://{args.bind}:{args.audio_port}")

    print("\nIn Docker use:")
    print(f"  VB_CAMERA_URL=http://host.docker.internal:{args.video_port}/video")
    if audio:
        print(f"  VB_MIC_URL=tcp://host.docker.internal:{args.audio_port}")
    print("\nAnyone who can reach these ports sees the camera and hears the microphone. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        source.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
