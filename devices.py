"""
devices.py

Camera and microphone helpers shared by trainer.py and the setup screen (device_setup.py).

    open_camera(preferred)   open a camera, falling back to the others if it is not available
    open_camera_index(i)     open exactly camera i (default backend, then DirectShow), or None
    list_cameras()           every camera index that opens, with its resolution
    list_microphones()       every input microphone (needs the `sounddevice` package)
    MicMeter                 live input level of a microphone, for the "does my mic work" test
"""

import math

import cv2
import numpy as np


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


def open_camera_index(index):
    for api in _apis():
        cap = _try_open(index, api)
        if cap is not None:
            return cap
    return None


def open_camera(preferred, scan=range(0, 5)):
    """Opens `preferred`; if Windows cannot open it (Camo not running, camera busy ...), tries the other
    indexes with both the default and the DirectShow backend. Returns (capture, index_used)."""
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
            rate = int(sd.query_devices(index)["default_samplerate"])
            self.stream = sd.InputStream(device=index, channels=1, samplerate=rate, blocksize=1024,
                                         callback=self._callback)
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