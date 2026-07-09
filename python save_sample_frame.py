"""
save_sample_frame.py

Grabs one frame from a video (roughly the middle of the clip) and saves
it as a .png so you can open it and look directly — checking for
rotation issues, framing problems, or anything else MediaPipe might be
struggling with.

Usage:
    Edit VIDEO_PATH below, then run:
        python save_sample_frame.py
"""

import cv2

VIDEO_PATH = "data/raw_clips/service_authorization_right/authorization_to_serve_right_p02_50.mov"
OUTPUT_PATH = "sample_frame.png"


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Could not open {VIDEO_PATH}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    success, frame = cap.read()

    if not success:
        print("Could not read a frame from this video.")
        return

    height, width = frame.shape[:2]
    print(f"Frame size: {width}x{height}")
    if height > width:
        print("NOTE: this frame is TALLER than it is wide (portrait-shaped).")
    else:
        print("NOTE: this frame is WIDER than it is tall (landscape-shaped).")

    cv2.imwrite(OUTPUT_PATH, frame)
    print(f"Saved: {OUTPUT_PATH} — open this file and look at it directly.")


if __name__ == "__main__":
    main()