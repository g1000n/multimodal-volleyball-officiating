"""
quick_test_crop.py

Tests the NEW wrist-crop hand detection (from extract_keypoints.py)
on just one folder, without touching the manifest or running the
full dataset. Reuses the actual functions from extract_keypoints.py
so this is a true test of the same logic that will run for real.

Usage:
    Edit FOLDER_TO_TEST below, then run:
        python quick_test_crop.py
"""

import os
import cv2
import numpy as np
import mediapipe as mp

from extract_keypoints import extract_pose_features, extract_hand_features

FOLDER_TO_TEST = "data/raw_clips/service_authorization_right"

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


def test_clip(video_path, pose_model, hands_model):
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
        pose_results = pose_model.process(frame_rgb)
        _, pose_landmarks = extract_pose_features(pose_results)

        _, left_det, right_det, _, _ = extract_hand_features(frame_rgb, pose_landmarks, hands_model)
        left_detected += left_det
        right_detected += right_det

    cap.release()

    if total_frames == 0:
        return None, 0
    return (left_detected / total_frames, right_detected / total_frames), total_frames


def main():
    pose_model = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hands_model = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    files = [f for f in sorted(os.listdir(FOLDER_TO_TEST)) if f.lower().endswith((".mp4", ".mov"))]
    print(f"Testing {len(files)} clips in {FOLDER_TO_TEST} with wrist-crop detection\n")
    print(f"  {'filename':<45} {'left':>8} {'right':>8} {'frames':>8}")

    for filename in files:
        path = os.path.join(FOLDER_TO_TEST, filename)
        result, frame_count = test_clip(path, pose_model, hands_model)

        if result is None:
            print(f"  {filename:<45} COULD NOT PROCESS")
            continue

        left_rate, right_rate = result
        print(f"  {filename:<45} {left_rate:>7.1%} {right_rate:>7.1%} {frame_count:>8}")

    pose_model.close()
    hands_model.close()


if __name__ == "__main__":
    main()