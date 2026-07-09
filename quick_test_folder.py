"""
quick_test_folder.py

Runs Pose + Hands on every clip inside a given folder and prints the
hand detection rate per clip. This is a throwaway diagnostic — it does
NOT save keypoints or touch the manifest. Use this to quickly check
whether your newly-converted .mp4 clips detect hands better than the
old .mov versions did, without re-running the full pipeline.

Usage:
    Edit FOLDER_TO_TEST below, then run:
        python quick_test_folder.py
"""

import os
import cv2
import numpy as np
import mediapipe as mp

FOLDER_TO_TEST = "data/raw_clips/service_authorization_right"

mp_hands = mp.solutions.hands


def detection_rate_for_clip(video_path, hands_model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0

    total_frames = 0
    left_detected = 0
    right_detected = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        total_frames += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_model.process(frame_rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            for handedness in results.multi_handedness:
                label = handedness.classification[0].label
                if label == "Left":
                    left_detected += 1
                else:
                    right_detected += 1

    cap.release()

    if total_frames == 0:
        return None, 0

    return (left_detected / total_frames, right_detected / total_frames), total_frames


def main():
    hands_model = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    files = [f for f in sorted(os.listdir(FOLDER_TO_TEST)) if f.lower().endswith((".mp4", ".mov"))]

    if not files:
        print(f"No video files found in {FOLDER_TO_TEST}")
        return

    print(f"Testing {len(files)} clips in {FOLDER_TO_TEST}\n")
    print(f"  {'filename':<45} {'left':>8} {'right':>8} {'frames':>8}")

    for filename in files:
        path = os.path.join(FOLDER_TO_TEST, filename)
        result, frame_count = detection_rate_for_clip(path, hands_model)

        if result is None:
            print(f"  {filename:<45} COULD NOT PROCESS")
            continue

        left_rate, right_rate = result
        print(f"  {filename:<45} {left_rate:>7.1%} {right_rate:>7.1%} {frame_count:>8}")

    hands_model.close()


if __name__ == "__main__":
    main()