"""
check_reference_hand_detection.py

Diagnostic tool — NOT a classifier test. Runs the same pose/hand
extraction used in extract_keypoints.py on your reference clips and
reports the hand-detection rate for each one.

Purpose: isolate WHERE the reference-clip failures are coming from.
  - If detection rate is low (e.g. <50%) on the clips the model gets
    wrong, MediaPipe itself is struggling to find the hands at that
    distance/resolution/compression — this is an EXTRACTION problem,
    and no amount of retraining or augmentation will fix it. You'd
    need better source footage, or to detect the person/crop tighter
    before running Hands.
  - If detection rate is high (similar to your training clips, which
    ran 88-100% per your extraction logs) but the model still gets it
    wrong, the extraction is working fine and the problem is genuinely
    in what the model learned — a real classification/generalization
    issue, not a detection issue.

Run this AFTER organizing your reference clips exactly like
test_reference_clips.py expects (data/reference_clips/<label>/*.mp4).
"""

import os
import cv2
import numpy as np
import mediapipe as mp

from extract_keypoints import extract_pose_features, extract_hand_features

REFERENCE_DIR = "data/reference_clips"

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


def analyze_clip(video_path, pose_model, hands_model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    left_flags, right_flags, pose_flags = [], [], []

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose_model.process(frame_rgb)
        pose_features, pose_landmarks = extract_pose_features(pose_results)

        pose_detected = 0.0 if np.all(pose_features == 0) else 1.0
        pose_flags.append(pose_detected)

        _, left_det, right_det, _, _ = extract_hand_features(frame_rgb, pose_landmarks, hands_model)
        left_flags.append(left_det)
        right_flags.append(right_det)

    cap.release()

    if len(pose_flags) == 0:
        return None

    return {
        "pose_rate": np.mean(pose_flags),
        "left_hand_rate": np.mean(left_flags),
        "right_hand_rate": np.mean(right_flags),
        "frame_count": len(pose_flags),
    }


def main():
    pose_model = mp_pose.Pose(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    hands_model = mp_hands.Hands(
        static_image_mode=True, max_num_hands=1,
        min_detection_confidence=0.25, min_tracking_confidence=0.25,
    )

    results = []

    for expected_label in sorted(os.listdir(REFERENCE_DIR)):
        folder_path = os.path.join(REFERENCE_DIR, expected_label)
        if not os.path.isdir(folder_path):
            continue

        for filename in sorted(os.listdir(folder_path)):
            video_path = os.path.join(folder_path, filename)
            print(f"Analyzing: {video_path} ...")
            stats = analyze_clip(video_path, pose_model, hands_model)
            if stats is None:
                print(f"  Could not read clip, skipping.")
                continue
            stats["file"] = filename
            stats["label"] = expected_label
            results.append(stats)

    pose_model.close()
    hands_model.close()

    print("\n" + "=" * 90)
    print("HAND/POSE DETECTION RATES ON REFERENCE CLIPS")
    print("=" * 90)
    print(f"{'file':<45} {'label':<28} {'pose':>6} {'L-hand':>7} {'R-hand':>7}")

    low_detection_clips = []
    for r in results:
        flag = ""
        if r["left_hand_rate"] < 0.5 and r["right_hand_rate"] < 0.5:
            flag = "  <-- LOW HAND DETECTION"
            low_detection_clips.append(r)
        print(f"{r['file']:<45} {r['label']:<28} {r['pose_rate']:>5.0%} "
              f"{r['left_hand_rate']:>6.0%} {r['right_hand_rate']:>6.0%}{flag}")

    print("\n" + "=" * 90)
    if low_detection_clips:
        print(f"{len(low_detection_clips)} / {len(results)} clips have LOW hand detection (<50% both hands).")
        print("These clips are likely failing due to MediaPipe not finding hands at this")
        print("distance/resolution/compression, not because of the classifier itself.")
        print("Compare this to your training clips' detection rates (88-100% per your")
        print("extraction logs) to see how large the gap is.")
    else:
        print("No clips show critically low hand detection — extraction seems to be")
        print("working reasonably across your reference set. If the classifier is still")
        print("getting these wrong, the issue is more likely in what the model learned,")
        print("not in MediaPipe's ability to see the hands.")
    print("=" * 90)


if __name__ == "__main__":
    main()