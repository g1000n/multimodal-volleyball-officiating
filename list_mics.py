"""
list_mics.py

Lists available audio input devices so you can find which index is your
WO Mic (iPhone 13) virtual microphone vs. your laptop's built-in mic or
any Camo virtual audio channel.

Companion to list_cameras.py -- same idea, for audio instead of video.

Run this AFTER connecting WO Mic (iPhone 13 app + Windows client, via USB).
Speak or blow a whistle near the iPhone 13 while this runs; whichever index
shows the highest peak level during its 2-second sample is your mic.

Set the result as WHISTLE_DEVICE_INDEX in live_deployment.py.

Re-run this any time after a reboot/reconnect -- Windows can reassign
device indices, and the code comment in live_deployment.py notes this
already happened once during testing.
"""
import sys

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("sounddevice not installed -- run: pip install sounddevice")
    sys.exit(1)

SAMPLE_SEC = 2.0
SR = 22050

print("Available audio devices:\n")
devices = sd.query_devices()
for i, d in enumerate(devices):
    if d["max_input_channels"] > 0:
        print(f"  [{i}] {d['name']}  (max input channels: {d['max_input_channels']})")

print("\nNow sampling each input device for 2 seconds -- speak, clap, or blow a")
print("whistle near the iPhone 13 to see which index picks it up.\n")

for i, d in enumerate(devices):
    if d["max_input_channels"] == 0:
        continue
    try:
        recording = sd.rec(int(SAMPLE_SEC * SR), samplerate=SR, channels=1,
                            device=i, dtype="float32")
        sd.wait()
        peak = float(np.max(np.abs(recording)))
        bar = "#" * int(peak * 50)
        print(f"  [{i}] {d['name']:<45} peak={peak:.3f}  {bar}")
    except Exception as e:
        print(f"  [{i}] {d['name']:<45} ERROR: {e}")

print("\nDone. Set WHISTLE_DEVICE_INDEX in live_deployment.py to whichever index")
print("showed a clear peak when you spoke/blew near the iPhone 13 (WO Mic).")