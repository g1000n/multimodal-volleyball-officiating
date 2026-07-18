"""
list_cameras.py

Lists available camera indices so you can find which one is your
Camo/iPhone virtual camera vs. your laptop's built-in webcam.

Run this, look at each preview window's title (it'll show the index),
note which one shows your iPhone's feed, then set that number as
CAMERA_INDEX in live_inference.py.

Press any key to move to the next camera. Press Q to stop early.
"""

import cv2

MAX_INDEX_TO_CHECK = 6

for i in range(MAX_INDEX_TO_CHECK):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f"Index {i}: not available")
        cap.release()
        continue

    success, frame = cap.read()
    if not success:
        print(f"Index {i}: opened but no frame received")
        cap.release()
        continue

    print(f"Index {i}: showing preview — press any key to continue, Q to stop")
    cv2.imshow(f"Camera index {i}", frame)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    cap.release()

    if key == ord('q'):
        break

print("\nDone. Set CAMERA_INDEX in live_inference.py to whichever index showed your iPhone's feed.")