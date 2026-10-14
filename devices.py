"""
devices.py

Camera and microphone helpers shared by trainer.py and the setup screen (device_setup.py).

    open_camera(preferred)   open a camera, falling back to the others if it is not available
    open_camera_index(i)     open exactly camera i (default backend, then DirectShow), or None
    list_cameras()           every camera index that opens, with its resolution
    list_microphones()       every input microphone (needs the `sounddevice` package)
    MicMeter                 live input level of a microphone, for the "does my mic work" test
"""

import json
import math
import os
import socket
import struct
import threading
import time
import urllib.request
from urllib.parse import urlparse

import cv2
import numpy as np

# Docker on Windows/macOS cannot pass a webcam or microphone into a container. tools/av_bridge.py runs on the HOST
# and serves them over the network; these two environment variables point the tool at that bridge:
#   VB_CAMERA_URL  e.g. http://host.docker.internal:8765/video   (MJPEG video)
#   VB_MIC_URL     e.g. tcp://host.docker.internal:8766          (raw float32 audio)
# When they are NOT set (the normal native run) nothing in this file behaves differently.
ENV_CAMERA_URL = "VB_CAMERA_URL"
ENV_MIC_URL = "VB_MIC_URL"


def camera_url():
    return os.environ.get(ENV_CAMERA_URL, "").strip()


def mic_url():
    return os.environ.get(ENV_MIC_URL, "").strip()


def _bridge_base():
    """http://host:port of the bridge, from VB_CAMERA_URL."""
    u = urlparse(camera_url())
    return f"{u.scheme}://{u.netloc}"


def bridge_devices():
    """The host's cameras and microphones as the bridge reports them, or None when it cannot be reached.
    {"cameras": [{index, width, height}], "mics": [{index, name, hostapi, default}], "default_camera", "default_mic"}"""
    if not camera_url():
        return None
    try:
        with urllib.request.urlopen(_bridge_base() + "/devices", timeout=5) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def _bridge_video_url(index):
    try:
        return f"{_bridge_base()}/video?camera={int(index)}"
    except (TypeError, ValueError):
        return camera_url()


def _apis():
    return [None] + ([cv2.CAP_DSHOW] if hasattr(cv2, "CAP_DSHOW") else [])


def _try_open(index, api=None):
    cap = cv2.VideoCapture(index) if api is None else cv2.VideoCapture(index, api)
    try:
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
    except Exception:
        pass
    cap.release()
    return None


class LatestFrameCapture:
    """A network camera that always hands over the NEWEST frame.

    A plain cv2.VideoCapture on a stream queues every frame it receives. The trainer processes fewer frames per
    second than the camera sends, so the queue grows and the picture falls further and further behind (seconds of
    lag). A reader thread here drains the stream continuously and keeps only the latest frame; read() waits for a
    frame it has not returned yet, like a normal capture, and older frames are dropped.
    Same interface the code uses from cv2.VideoCapture: read, isOpened, release, get, set."""

    def __init__(self, cap):
        self._cap = cap
        self._cond = threading.Condition()
        self._frame, self._seq, self._returned = None, 0, 0
        self._alive = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._alive:
            try:
                ok, frame = self._cap.read()
            except Exception:
                ok, frame = False, None
            with self._cond:
                if ok:
                    self._frame, self._seq = frame, self._seq + 1
                else:
                    self._alive = False           # the stream ended or the bridge stopped
                self._cond.notify_all()
            if not ok:
                break

    def read(self):
        with self._cond:
            self._cond.wait_for(lambda: self._seq > self._returned or not self._alive, timeout=2.0)
            if self._seq > self._returned and self._frame is not None:
                self._returned = self._seq
                return True, self._frame
            return False, None

    def isOpened(self):
        return self._alive and self._cap.isOpened()

    def get(self, prop):
        return self._cap.get(prop)

    def set(self, prop, value):
        return self._cap.set(prop, value)

    def release(self):
        self._alive = False
        with self._cond:
            self._cond.notify_all()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        try:
            self._cap.release()
        except Exception:
            pass


def open_camera_index(index):
    if camera_url():                          # network camera from the host bridge: it opens camera `index`
        cap = _try_open(_bridge_video_url(index))
        return LatestFrameCapture(cap) if cap is not None else None
    for api in _apis():
        cap = _try_open(index, api)
        if cap is not None:
            return cap
    return None


def open_camera(preferred, scan=range(0, 5)):
    """Opens `preferred`; if Windows cannot open it (Camo not running, camera busy ...), tries the other
    indexes with both the default and the DirectShow backend. Returns (capture, index_used)."""
    if camera_url():
        info = bridge_devices()
        order = [preferred]
        for extra in ([info.get("default_camera")] if info else []) + [0]:
            if extra is not None and extra not in order:
                order.append(extra)
        for idx in order:
            cap = open_camera_index(idx)
            if cap is not None:
                if idx != preferred:
                    print(f"Camera {preferred} is not available on the host, using camera {idx} instead.")
                return cap, idx
        raise RuntimeError(
            f"The network camera {camera_url()} could not be opened. Start tools/av_bridge.py on the host computer "
            f"(it serves the camera) and check that the port is not blocked by the firewall.")
    for idx in [preferred] + [i for i in scan if i != preferred]:
        cap = open_camera_index(idx)
        if cap is not None:
            if idx != preferred:
                print(f"Camera {preferred} is not available, using camera {idx} instead.")
            return cap, idx
    raise RuntimeError(
        f"No camera could be opened (tried index {preferred} and 0-4). Check that the camera is plugged in / "
        f"Camo is running, and that no other program (Zoom, the Camera app, live_deployment.py) is using it.")


def list_cameras(scan=range(0, 5)):
    found = []
    if camera_url():
        info = bridge_devices()
        if info and info.get("cameras") is not None:
            return [{"index": c["index"], "width": c.get("width", 0), "height": c.get("height", 0)}
                    for c in info["cameras"]]
        scan = [0]                            # bridge too old to list devices: just the default camera
    for idx in scan:
        cap = open_camera_index(idx)
        if cap is not None:
            try:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            except Exception:
                w = h = 0
            cap.release()
            found.append({"index": idx, "width": w, "height": h})
    return found


def list_microphones():
    """Returns (devices, error_text). devices = [{index, name, hostapi, default}], possibly empty."""
    if mic_url():
        info = bridge_devices()
        if info and info.get("mics"):
            default_mic = info.get("default_mic")
            return [{"index": m["index"], "name": m["name"], "hostapi": m.get("hostapi", ""),
                     "default": (m["index"] == default_mic) if default_mic is not None else bool(m.get("default"))}
                    for m in info["mics"]], ""
        return [{"index": -1, "name": "Host microphone (av_bridge)", "hostapi": "network", "default": True}], ""
    try:
        import sounddevice as sd
    except Exception as exc:
        return [], f"the 'sounddevice' package is not available ({exc})"
    try:
        devs = sd.query_devices()
        apis = sd.query_hostapis()
    except Exception as exc:
        return [], f"the audio devices could not be listed ({exc})"
    try:
        default_in = int(sd.default.device[0])
    except Exception:
        default_in = -1
    out = []
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            try:
                api = apis[d["hostapi"]]["name"]
            except Exception:
                api = ""
            out.append({"index": i, "name": d["name"], "hostapi": api, "default": i == default_in})
    return out, ""


def resolve_saved_mic(settings):
    """The microphone chosen under "Camera and mic", found again by NAME. Device numbers can change when a phone
    microphone (e.g. an iPhone through WO Mic) or a USB mic is reconnected, so the saved number alone could silently
    point at a different input. Returns (index or None, warning text or "")."""
    if mic_url():
        mics, _err = list_microphones()
        want = settings.get("mic_name")
        same = [m for m in mics if m["name"] == want]
        if same:
            return same[0]["index"], ""
        default = next((m for m in mics if m["default"]), None)
        return (default["index"] if default else -1), ""
    want_name, want_api = settings.get("mic_name"), settings.get("mic_hostapi")
    index = settings.get("mic_index")
    if not want_name:
        return index, ""                                     # saved by an older version: number only
    mics, _err = list_microphones()
    same = [m for m in mics if m["name"] == want_name]
    match = (next((m for m in same if m["index"] == index), None)
             or next((m for m in same if m["hostapi"] == want_api), None)
             or (same[0] if same else None))
    if match is not None:
        return match["index"], ""
    return None, (f"The microphone chosen for whistle detection ({want_name}) is not connected. Connect it and check "
                  f"it under Camera and mic; until then the default microphone is used (W still works as a whistle).")


class NetworkInputStream:
    """Same interface the code uses from sounddevice.InputStream (start/stop/close, callback(indata, frames, time,
    status)), but the audio comes from tools/av_bridge.py on the host. Protocol: the client sends the sample rate as
    4 bytes (little-endian unsigned int) and the microphone number as 4 more (signed int, -1 = the bridge's
    default); the bridge answers with mono float32 samples at that rate."""

    def __init__(self, url, samplerate, blocksize, callback, device=None):
        self.device = -1 if device is None else int(device)      # microphone number on the host; -1 = its default
        rest = url.split("://", 1)[-1].rstrip("/")
        host, _, port = rest.rpartition(":")
        self.host, self.port = (host or rest), int(port or 8766)
        self.samplerate, self.blocksize, self.callback = int(samplerate), int(blocksize), callback
        self._sock = None
        self._thread = None
        self._running = False
        self.lost = False                 # True when the bridge stopped sending while we were still listening

    def start(self):
        self._sock = socket.create_connection((self.host, self.port), timeout=5.0)
        self._sock.sendall(struct.pack("<Ii", self.samplerate, self.device))
        self._sock.settimeout(2.0)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        need = self.blocksize * 4
        buf = b""
        while self._running:
            try:
                data = self._sock.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                self.lost = self._running
                break
            if not data:
                self.lost = self._running
                break
            buf += data
            while len(buf) >= need:
                block, buf = buf[:need], buf[need:]
                chunk = np.frombuffer(block, dtype="<f4").reshape(-1, 1).copy()
                self.callback(chunk, len(chunk), None, None)

    def stop(self):
        self._running = False
        try:
            self._sock.close()
        except Exception:
            pass
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)

    def close(self):
        self.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


def open_input_stream(sd, device, samplerate, blocksize, callback):
    """A microphone stream: the host bridge when VB_MIC_URL is set, otherwise a normal sounddevice.InputStream."""
    url = mic_url()
    if url:
        return NetworkInputStream(url, samplerate, blocksize, callback, device)
    return sd.InputStream(device=device, channels=1, samplerate=samplerate, blocksize=blocksize, callback=callback)


class MicMeter:
    """Opens an input stream and exposes a 0..1 loudness level (log scale, -60 dB to 0 dB)."""

    HEARD_LEVEL = 0.30

    def __init__(self):
        self.stream = None
        self._level = 0.0
        self.heard = False
        self.error = ""

    def _callback(self, indata, frames, time_info, status):
        rms = float(np.sqrt(np.mean(np.square(indata[:, 0])))) if len(indata) else 0.0
        db = 20.0 * math.log10(max(rms, 1e-6))
        self._level = float(min(1.0, max(0.0, (db + 60.0) / 60.0)))
        if self._level >= self.HEARD_LEVEL:
            self.heard = True

    def start(self, index):
        self.stop()
        self.error, self._level, self.heard = "", 0.0, False
        try:
            import sounddevice as sd
            rate = 22050 if mic_url() else int(sd.query_devices(index)["default_samplerate"])
            self.stream = open_input_stream(sd, index, rate, 1024, self._callback)
            self.stream.start()
        except Exception as exc:
            self.stream = None
            self.error = str(exc)

    def level(self):
        return self._level

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None


class WhistleProbe:
    """Starts the real whistle detector on a chosen microphone, only to TEST it (the setup screen)."""

    def __init__(self):
        self.detector = None
        self.count = 0
        self.last_time = None
        self.last_conf = None
        self.error = ""

    def _on_whistle(self, timestamp, confidence=None):
        self.count += 1
        self.last_time = time.time()
        self.last_conf = confidence

    def start(self, mic_index):
        self.stop()
        self.count, self.last_time, self.last_conf, self.error = 0, None, None, ""
        try:
            from whistle_detector import WhistleDetector
            import trainer_config as _tc
            self.detector = WhistleDetector(on_whistle_callback=self._on_whistle, device=mic_index,
                                            threshold=getattr(_tc, "WHISTLE_THRESHOLD", 0.70))   # same as sessions
            self.detector.start(confirm_open=True)      # the setup screen then shows why, instead of a dead test
        except Exception as exc:
            self.detector = None
            self.error = str(exc)

    @property
    def running(self):
        return self.detector is not None

    def stop(self):
        if self.detector is not None:
            try:
                self.detector.stop()
            except Exception:
                pass
            self.detector = None