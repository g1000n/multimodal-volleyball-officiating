"""
av_bridge.py -- serve this computer's camera and microphone over the network, for the Docker container.

Docker Desktop on Windows/macOS cannot pass a webcam or microphone into a container. Run this on the HOST (the
normal Windows venv), then start the container. The training tool reads the camera and the microphone through
devices.py, which uses the bridge when VB_CAMERA_URL / VB_MIC_URL are set. The app's "Camera and mic" screen then
lists ALL of this computer's cameras and microphones and switches between them through the bridge.

    py tools/av_bridge.py                      defaults: camera 0 + default microphone
    py tools/av_bridge.py --list               list cameras and microphones, then exit
    py tools/av_bridge.py --camera 1 --mic 3   the defaults when the app has not chosen one
    py tools/av_bridge.py --video clip.mp4     use a video file instead of a camera (rehearsal / tests)

Video  : MJPEG over HTTP, http://<host>:8765/video?camera=N   (default camera when no ?camera=)
Devices: JSON at http://<host>:8765/devices                    (cameras and microphones of this computer)
Audio  : raw mono float32 over TCP, <host>:8766                (client sends the sample rate, then a microphone
                                                               index; -1 or nothing = the default)

Privacy: the camera and microphone are served to anyone who can reach these ports. Use it on a trusted network
only, stop it when you are done (Ctrl+C), and try --bind 127.0.0.1 (it often still works with Docker Desktop).
"""

import argparse
import json
import os
import socket
import socketserver
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import devices  # noqa: E402


class FrameSource:
    """Keeps the newest JPEG from the selected camera (or a looping video file) in memory.
    select(n) switches to camera n; every connected client then sees that camera."""

    def __init__(self, camera=0, video=None, fps=25.0, quality=80, opener=None):
        self.camera, self.video, self.fps, self.quality = camera, video, fps, quality
        self.opener = opener                     # opener(index) -> capture or None (tests inject fake cameras)
        self.jpeg = None
        self.size = (0, 0)
        self.error = ""
        self.cap = None
        self._lock = threading.Lock()
        self._generation = 0
        self._thread = None

    def _open(self, index):
        if self.video:
            if not os.path.isfile(self.video):
                raise RuntimeError(f"video file not found: {self.video}")
            return cv2.VideoCapture(self.video)
        cap = self.opener(index) if self.opener else devices.open_camera_index(index)
        if cap is None:
            raise RuntimeError(f"camera {index} could not be opened (busy, or not connected; try --list)")
        return cap

    def start(self):
        self.select(self.camera, force=True)

    def select(self, index, force=False):
        """Switch to camera `index`. Returns True when it is open and delivering pictures."""
        with self._lock:
            if not force and index == self.camera and self.cap is not None and self.jpeg is not None:
                return True
            self._stop_reader()
            try:
                cap = self._open(index)
                ok, frame = cap.read()
                if not ok:
                    cap.release()
                    raise RuntimeError(f"camera {index} opened but gave no picture")
            except Exception as exc:
                self.error = str(exc)
                self.jpeg = None
                return False
            self.error = ""
            self.camera, self.cap = index, cap
            self._store(frame)
            self._thread = threading.Thread(target=self._loop, args=(self._generation, cap), daemon=True)
            self._thread.start()
            return True

    def _stop_reader(self):
        """Stop the reader thread BEFORE releasing the camera: releasing a capture another thread is reading
        from can stall OpenCV for tens of seconds. Call with the lock held."""
        self._generation += 1                    # tells the old reader thread to stop
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=3.0)
        self._thread = None
        old, self.cap = self.cap, None
        if old is not None:
            try:
                old.release()
            except Exception:
                pass

    def _store(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if ok:
            self.jpeg = buf.tobytes()
            self.size = (frame.shape[1], frame.shape[0])

    def _loop(self, generation, cap):
        delay = 1.0 / self.fps
        while generation == self._generation:
            t0 = time.time()
            try:
                ok, frame = cap.read()
            except Exception:
                break
            if generation != self._generation:
                break
            if not ok:
                if self.video:                      # loop a video file
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.05)
                continue
            self._store(frame)
            if self.video:                          # a file would otherwise play as fast as it can be read
                time.sleep(max(0.0, delay - (time.time() - t0)))

    def stop(self):
        with self._lock:
            self._stop_reader()


def scan_devices(video=None, default_camera=0, default_mic=None, synthetic=False, skip_cameras=()):
    """What this computer has: {"cameras": [...], "mics": [...], "default_camera": n, "default_mic": n|None}."""
    if video:
        cams = [{"index": default_camera, "width": 0, "height": 0, "name": os.path.basename(video)}]
    else:
        # Opening a camera to read its size wakes virtual cameras (Camo starts its app), so skipped ones are
        # never opened and never offered.
        cams = devices.list_cameras(scan=[i for i in range(5) if i not in set(skip_cameras)])
    mics, _err = ([{"index": 0, "name": "Test tone", "hostapi": "synthetic", "default": True}], "") if synthetic \
        else devices.list_microphones()
    return {"cameras": cams, "mics": mics, "default_camera": default_camera, "default_mic": default_mic}


def make_http_handler(source, info=None):
    info = info if info is not None else {"cameras": [], "mics": [], "default_camera": 0, "default_mic": None}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if url.path == "/devices":
                self._send(200, json.dumps(info).encode("utf-8"), "application/json")
                return
            if url.path == "/frame.jpg":
                self._send(200, source.jpeg or b"", "image/jpeg")
                return
            if url.path != "/video":
                self.send_error(404)
                return
            if "camera" in query:
                try:
                    wanted = int(query["camera"][0])
                except ValueError:
                    self.send_error(400, "camera must be a number")
                    return
                if not source.select(wanted):
                    self.send_error(503, source.error or "camera not available")
                    return
            elif source.jpeg is None and not source.select(info.get("default_camera", 0)):
                self.send_error(503, source.error or "camera not available")
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
        self.last_device = None                 # the microphone the latest client asked for (tests)

    def stream(self, rate, blocksize, write, device=None):
        """Calls write(bytes) for every block until write raises. Blocks the calling thread.
        device: microphone index the client asked for; None/-1 = the bridge's default."""
        device = None if device is None or device < 0 else device
        use = self.device if device is None else device
        self.last_device = use
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

        with sd.InputStream(device=use, channels=1, samplerate=rate, blocksize=blocksize, callback=cb):
            while not failed:
                time.sleep(0.1)


class AudioHandler(socketserver.BaseRequestHandler):
    source = None

    def handle(self):
        try:
            sock = self.request
            sock.settimeout(5.0)
            rate = struct.unpack("<I", sock.recv(4))[0]
            rate = rate if 8000 <= rate <= 96000 else 22050
            device = None
            sock.settimeout(0.5)                  # an older client sends only the rate
            try:
                raw = sock.recv(4)
                if len(raw) == 4:
                    device = struct.unpack("<i", raw)[0]
            except socket.timeout:
                pass
            sock.settimeout(None)
            self.source.stream(rate, 1024, lambda b: sock.sendall(b), device)
        except Exception:
            pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="Serve the camera and microphone to the Docker container.")
    ap.add_argument("--camera", type=int, default=0, help="default camera (the app can choose another)")
    ap.add_argument("--mic", type=int, default=None, help="default sounddevice input index (see --list)")
    ap.add_argument("--video", help="serve this video file instead of a camera")
    ap.add_argument("--synthetic-audio", action="store_true", help="a quiet test tone instead of the microphone")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--video-port", type=int, default=8765)
    ap.add_argument("--audio-port", type=int, default=8766)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--skip-cameras", default="",
                    help="comma-separated camera numbers to never open or list, e.g. 0 (use it for a virtual "
                         "camera such as Camo, which launches its app when opened)")
    args = ap.parse_args(argv)
    skip = tuple(int(x) for x in args.skip_cameras.replace(" ", "").split(",") if x != "")
    if args.camera in skip:
        others = [i for i in range(5) if i not in skip]
        args.camera = others[0] if others else args.camera
        print(f"The default camera was in --skip-cameras; using camera {args.camera} instead.")

    if args.list:
        print("Cameras:")
        for c in devices.list_cameras(scan=[i for i in range(5) if i not in skip]):
            print(f"  --camera {c['index']}   {c['width']}x{c['height']}")
        mics, err = devices.list_microphones()
        print("Microphones:" + (f"  ({err})" if err else ""))
        for m in mics:
            print(f"  --mic {m['index']}   {m['name']}  [{m['hostapi']}]{'  (default)' if m['default'] else ''}")
        return 0

    print("Looking for cameras and microphones ...")
    info = scan_devices(args.video, args.camera, args.mic, args.synthetic_audio, skip)
    print(f"  {len(info['cameras'])} camera(s), {len(info['mics'])} microphone(s)")

    source = FrameSource(args.camera, args.video)
    source.start()
    http = ThreadingHTTPServer((args.bind, args.video_port), make_http_handler(source, info))
    threading.Thread(target=http.serve_forever, daemon=True).start()
    print(f"Video : http://{args.bind}:{args.video_port}/video   (camera {args.camera}, "
          f"{source.size[0]}x{source.size[1]})")
    print(f"Devices: http://{args.bind}:{args.video_port}/devices")

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
    print("\nThe app's Camera and mic screen lists this computer's cameras and microphones.")
    print("Anyone who can reach these ports sees the camera and hears the microphone. Ctrl+C to stop.")
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
