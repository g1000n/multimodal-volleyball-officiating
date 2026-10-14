"""
test_av_bridge.py -- checks tools/av_bridge.py and the network camera / microphone support in devices.py.

No camera, microphone or Docker is needed: the bridge serves a generated video and a synthetic tone on local ports,
and the client side (devices.py, whistle_detector.py) is pointed at them with VB_CAMERA_URL / VB_MIC_URL, exactly as
the container is.

Run:  python tests/test_av_bridge.py
"""

import os
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


def main():
    tmp = tempfile.mkdtemp()
    video = os.path.join(tmp, "test.mp4")
    vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (320, 240))
    for i in range(50):
        img = np.full((240, 320, 3), (i * 5) % 255, np.uint8)
        vw.write(img)
    vw.release()

    src = av_bridge.FrameSource(video=video)
    src.start()
    http = ThreadingHTTPServer(("127.0.0.1", 0), av_bridge.make_http_handler(src))
    threading.Thread(target=http.serve_forever, daemon=True).start()
    av_bridge.AudioHandler.source = av_bridge.AudioSource(synthetic=True)
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    aud = socketserver.ThreadingTCPServer(("127.0.0.1", 0), av_bridge.AudioHandler)
    aud.daemon_threads = True
    threading.Thread(target=aud.serve_forever, daemon=True).start()
    vport, aport = http.server_address[1], aud.server_address[1]

    # native behaviour is untouched when the variables are not set
    os.environ.pop(devices.ENV_CAMERA_URL, None)
    os.environ.pop(devices.ENV_MIC_URL, None)
    check("no URLs set: bridge off", devices.camera_url() == "" and devices.mic_url() == "")

    # camera
    os.environ[devices.ENV_CAMERA_URL] = f"http://127.0.0.1:{vport}/video"
    cap = devices.open_camera_index(7)           # the index is ignored in bridge mode
    ok, frame = (False, None) if cap is None else cap.read()
    check("network camera opens and gives a picture", ok and frame is not None and frame.shape[:2] == (240, 320),
          "" if ok else "no frame")
    if cap is not None:
        cap.release()
    cap, used = devices.open_camera(3)
    check("open_camera uses the bridge", cap is not None and used == 0)
    cap.release()
    found = devices.list_cameras()
    check("list_cameras shows the one network camera", len(found) == 1 and found[0]["width"] == 320)
    os.environ[devices.ENV_CAMERA_URL] = "http://127.0.0.1:9/video"
    try:
        devices.open_camera(0)
        check("unreachable camera raises a clear error", False)
    except RuntimeError as exc:
        check("unreachable camera raises a clear error", "av_bridge" in str(exc))
    os.environ.pop(devices.ENV_CAMERA_URL)

    # microphone
    os.environ[devices.ENV_MIC_URL] = f"tcp://127.0.0.1:{aport}"
    mics, err = devices.list_microphones()
    check("list_microphones shows the bridge", len(mics) == 1 and mics[0]["hostapi"] == "network" and not err)
    check("saved microphone resolves to the bridge", devices.resolve_saved_mic({"mic_name": "x"}) == (0, ""))
    got = []
    st = devices.open_input_stream(None, 0, 22050, 1024, lambda d, n, t, s: got.append(d))
    st.start()
    time.sleep(1.0)
    st.stop()
    audio = np.concatenate(got) if got else np.zeros((0, 1))
    check("network audio arrives in blocks of the right size", len(got) > 5 and all(len(g) == 1024 for g in got),
          f"{len(got)} blocks")
    check("the audio is the tone that was sent", len(audio) > 0 and 0.1 < float(np.sqrt(np.mean(audio ** 2))) < 0.2)

    meter = devices.MicMeter()
    meter.start(0)
    time.sleep(1.0)
    check("MicMeter hears the bridge", meter.error == "" and meter.heard, meter.error)
    meter.stop()

    # the real whistle detector runs on the network microphone (a pure tone is not a whistle, so no event)
    try:
        from whistle_detector import WhistleDetector
        det = WhistleDetector(on_whistle_callback=lambda *a: None, device=0, log_dir=tmp)
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
