"""
test_av_bridge.py -- checks tools/av_bridge.py and the network camera / microphone support in devices.py.

No camera, microphone or Docker is needed: the bridge serves generated videos (two fake cameras) and a synthetic
tone on local ports, and the client side (devices.py, whistle_detector.py) is pointed at them with VB_CAMERA_URL /
VB_MIC_URL, exactly as the container is.

Run:  python tests/test_av_bridge.py
"""

import os
import socket
import socketserver
import sys
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))

import devices          # noqa: E402
import av_bridge        # noqa: E402

failed = 0


def check(name, ok, detail=""):
    global failed
    print(("PASS  " if ok else "FAIL  ") + name + (f"  {detail}" if detail else ""))
    if not ok:
        failed += 1


class FakeCam:
    """An in-memory camera. (A real VideoCapture on a file would take OpenCV's global open lock while the client
    in this same process is opening the network stream, and the test would wait for itself.)"""

    def __init__(self, level, size):
        self.frame = np.full((size[1], size[0], 3), level, np.uint8)

    def read(self):
        time.sleep(0.02)
        return True, self.frame.copy()

    def set(self, *a):
        return True

    def release(self):
        pass


def mean_level(cap):
    ok, frame = cap.read()
    return float(frame.mean()) if ok and frame is not None else -1.0


def main():
    tmp = tempfile.mkdtemp()
    cams = {0: lambda: FakeCam(40, (320, 240)),             # a dark camera 0
            3: lambda: FakeCam(200, (640, 480))}            # a bright, larger camera 3

    src = av_bridge.FrameSource(camera=0, opener=lambda i: cams[i]() if i in cams else None)
    src.start()
    info = {"cameras": [{"index": 0, "width": 320, "height": 240}, {"index": 3, "width": 640, "height": 480}],
            "mics": [{"index": 1, "name": "Array mic", "hostapi": "MME", "default": True},
                     {"index": 5, "name": "USB mic", "hostapi": "Windows DirectSound", "default": False}],
            "default_camera": 0, "default_mic": 1}
    http = ThreadingHTTPServer(("127.0.0.1", 0), av_bridge.make_http_handler(src, info))
    threading.Thread(target=http.serve_forever, daemon=True).start()
    audio_source = av_bridge.AudioSource(device=1, synthetic=True)
    av_bridge.AudioHandler.source = audio_source
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    aud = socketserver.ThreadingTCPServer(("127.0.0.1", 0), av_bridge.AudioHandler)
    aud.daemon_threads = True
    threading.Thread(target=aud.serve_forever, daemon=True).start()
    vport, aport = http.server_address[1], aud.server_address[1]

    # native behaviour is untouched when the variables are not set
    os.environ.pop(devices.ENV_CAMERA_URL, None)
    os.environ.pop(devices.ENV_MIC_URL, None)
    check("no URLs set: bridge off", devices.camera_url() == "" and devices.mic_url() == "")

    # ---- camera ---------------------------------------------------------------------------------------------
    os.environ[devices.ENV_CAMERA_URL] = f"http://127.0.0.1:{vport}/video"
    found = devices.list_cameras()
    check("list_cameras shows ALL host cameras", [c["index"] for c in found] == [0, 3], str(found))
    check("camera sizes are reported", found[1]["width"] == 640 and found[1]["height"] == 480)

    cap = devices.open_camera_index(0)
    ok, frame = (False, None) if cap is None else cap.read()
    check("camera 0 opens and gives a picture", ok and frame is not None and frame.shape[:2] == (240, 320))
    dark = mean_level(cap) if cap is not None else -1
    if cap is not None:
        cap.release()
    cap = devices.open_camera_index(3)
    ok, frame = (False, None) if cap is None else cap.read()
    check("camera 3 opens: the bridge switched cameras",
          ok and frame is not None and frame.shape[:2] == (480, 640))
    bright = mean_level(cap) if cap is not None else -1
    check("camera 3 is the bright one", bright > dark + 100, f"{dark:.0f} -> {bright:.0f}")
    if cap is not None:
        cap.release()

    cap = devices.open_camera_index(7)                  # a camera the host does not have
    check("a camera the host does not have gives None", cap is None)
    cap, used = devices.open_camera(7)                  # saved camera 7 is missing: falls back to the default
    check("open_camera falls back to the host's default camera", cap is not None and used == 0, f"used {used}")
    cap.release()
    cap, used = devices.open_camera(3)
    check("open_camera opens the camera that was chosen", cap is not None and used == 3)
    cap.release()

    os.environ[devices.ENV_CAMERA_URL] = "http://127.0.0.1:9/video"
    try:
        devices.open_camera(0)
        check("unreachable bridge raises a clear error", False)
    except RuntimeError as exc:
        check("unreachable bridge raises a clear error", "av_bridge" in str(exc))
    os.environ[devices.ENV_CAMERA_URL] = f"http://127.0.0.1:{vport}/video"

    # ---- microphone -----------------------------------------------------------------------------------------
    os.environ[devices.ENV_MIC_URL] = f"tcp://127.0.0.1:{aport}"
    mics, err = devices.list_microphones()
    check("list_microphones shows ALL host microphones", [m["index"] for m in mics] == [1, 5] and not err, str(mics))
    check("the default microphone is marked", [m["default"] for m in mics] == [True, False])
    check("saved microphone is found again by name",
          devices.resolve_saved_mic({"mic_name": "USB mic"}) == (5, ""))
    check("an unknown saved microphone falls back to the default",
          devices.resolve_saved_mic({"mic_name": "Unplugged"}) == (1, ""))

    got = []
    st = devices.open_input_stream(None, 5, 22050, 1024, lambda d, n, t, s: got.append(d))
    st.start()
    time.sleep(1.0)
    st.stop()
    audio = np.concatenate(got) if got else np.zeros((0, 1))
    check("the bridge was asked for microphone 5", audio_source.last_device == 5, str(audio_source.last_device))
    check("network audio arrives in blocks of the right size", len(got) > 5 and all(len(g) == 1024 for g in got),
          f"{len(got)} blocks")
    check("the audio is the tone that was sent", len(audio) > 0 and 0.1 < float(np.sqrt(np.mean(audio ** 2))) < 0.2)

    got = []
    st = devices.open_input_stream(None, None, 22050, 1024, lambda d, n, t, s: got.append(d))
    st.start()
    time.sleep(0.5)
    st.stop()
    check("no microphone chosen: the bridge's default is used", audio_source.last_device == 1)

    meter = devices.MicMeter()
    meter.start(5)
    time.sleep(1.0)
    check("MicMeter hears the bridge", meter.error == "" and meter.heard, meter.error)
    meter.stop()

    # the bridge going away is reported (the microphone must not just go silent)
    dying = socket.socket()
    dying.bind(("127.0.0.1", 0))
    dying.listen(1)

    def _die():
        conn, _ = dying.accept()
        conn.recv(8)
        conn.sendall(np.zeros(2048, "<f4").tobytes())
        time.sleep(0.2)
        conn.close()
    threading.Thread(target=_die, daemon=True).start()
    st = devices.NetworkInputStream(f"tcp://127.0.0.1:{dying.getsockname()[1]}", 22050, 1024, lambda *a: None)
    st.start()
    time.sleep(1.0)
    check("a stopped bridge microphone is flagged as lost", st.lost is True)
    st.stop()
    dying.close()

    # the real whistle detector runs on the network microphone (a pure tone is not a whistle, so no event)
    try:
        from whistle_detector import WhistleDetector
        det = WhistleDetector(on_whistle_callback=lambda *a: None, device=5, log_dir=tmp)
        det.start(confirm_open=True)
        time.sleep(2.0)
        check("WhistleDetector runs on the network microphone", det.error == "")
        det.stop()
    except Exception as exc:
        check("WhistleDetector runs on the network microphone", False, str(exc))

    src.stop()
    http.shutdown()
    aud.shutdown()
    print("\nAll good." if not failed else f"\n{failed} check(s) failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
