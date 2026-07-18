"""
extract_keypoints.py (v5 — Pose + wrist-cropped Hands + finger extension)
[OPTIMIZED: multiprocessing + auto warm-up fix + timer + alarm]

Feature layout: 122 features per frame (24 pose + 84 hand coords +
2 hand-detected flags + 10 finger-extension + 2 elbow angles).

SAFE ON FRESH DEVICES (the actual fix):
Multiple worker processes starting at once can each try to download
MediaPipe's model file simultaneously, corrupting it ("Model provided
must have at least 7 bytes...", "should be 'TFL3'" errors). This
version runs ONE sequential warm-up (single Pose + Hands init) BEFORE
spinning up any parallel workers, guaranteeing the model file is
already fully downloaded and valid by the time workers start. This
has hit multiple team members' machines before -- this fix makes it a
non-issue going forward, on any device, first run or not.

SPEED:
- Clips processed in parallel across CPU cores via multiprocessing.Pool.
- Pose model_complexity lowered from 1 -> 0 (faster, small accuracy
  tradeoff -- check the hand/pose detection rate printout at the end;
  if it drops noticeably vs. a previous run, raise it back to 1 in
  _init_worker() below).
- Wall-clock time printed at the end.
- 3 beeps (Windows only) when extraction finishes.
"""

import os
import csv
import time
import cv2
import numpy as np
import mediapipe as mp
import winsound
from collections import defaultdict
from multiprocessing import Pool, cpu_count

MANIFEST_PATH = "data/dataset_manifest.csv"
KEYPOINTS_DIR = "data/keypoints"

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

UPPER_BODY_LANDMARKS = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
]

NUM_HAND_LANDMARKS = 21
FINGER_JOINTS = {
    "thumb":  (4, 2),
    "index":  (8, 6),
    "middle": (12, 10),
    "ring":   (16, 14),
    "pinky":  (20, 18),
}
EXTENSION_MARGIN = 1.1

CROP_MARGIN_MULTIPLIER = 1.9
MIN_VISIBILITY_FOR_CROP = 0.18


def extract_pose_features(pose_results):
    if pose_results.pose_landmarks is None:
        return np.zeros(len(UPPER_BODY_LANDMARKS) * 3), None
    landmarks = pose_results.pose_landmarks.landmark
    features = []
    for lm_id in UPPER_BODY_LANDMARKS:
        lm = landmarks[lm_id]
        features.extend([lm.x, lm.y, lm.visibility])
    return np.array(features), landmarks


def compute_finger_extension(hand_landmarks):
    lm = hand_landmarks.landmark
    wrist = np.array([lm[0].x, lm[0].y])

    def dist_to_wrist(idx):
        point = np.array([lm[idx].x, lm[idx].y])
        return np.linalg.norm(point - wrist)

    extensions = []
    for finger_name, (tip_idx, base_idx) in FINGER_JOINTS.items():
        tip_dist = dist_to_wrist(tip_idx)
        base_dist = dist_to_wrist(base_idx)
        extended = 1.0 if tip_dist > base_dist * EXTENSION_MARGIN else 0.0
        extensions.append(extended)
    return np.array(extensions)


def compute_elbow_angles(pose_landmarks):
    if pose_landmarks is None:
        return np.array([0.5, 0.5])

    def angle_for_arm(shoulder_id, elbow_id, wrist_id):
        shoulder = pose_landmarks[shoulder_id]
        elbow = pose_landmarks[elbow_id]
        wrist = pose_landmarks[wrist_id]

        if elbow.visibility < 0.3:
            return 0.5

        v1 = np.array([shoulder.x - elbow.x, shoulder.y - elbow.y])
        v2 = np.array([wrist.x - elbow.x, wrist.y - elbow.y])
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.5

        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        angle_radians = np.arccos(cos_angle)
        return angle_radians / np.pi

    left_angle = angle_for_arm(
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST
    )
    right_angle = angle_for_arm(
        mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST
    )
    return np.array([left_angle, right_angle])


def get_wrist_crop_box(wrist_landmark, shoulder_width_px, frame_width, frame_height):
    if wrist_landmark.visibility < MIN_VISIBILITY_FOR_CROP:
        return None

    cx = wrist_landmark.x * frame_width
    cy = wrist_landmark.y * frame_height
    half_size = max(shoulder_width_px * CROP_MARGIN_MULTIPLIER, 40)

    x1 = int(max(0, cx - half_size))
    x2 = int(min(frame_width, cx + half_size))
    y1 = int(max(0, cy - half_size))
    y2 = int(min(frame_height, cy + half_size))

    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    return (x1, y1, x2, y2)


def detect_hand_in_crop(frame_rgb, crop_box, hands_model):
    x1, y1, x2, y2 = crop_box
    crop = frame_rgb[y1:y2, x1:x2]

    results = hands_model.process(crop)
    if not results.multi_hand_landmarks:
        return None

    hand_landmarks = results.multi_hand_landmarks[0]

    crop_width = x2 - x1
    crop_height = y2 - y1
    frame_height, frame_width = frame_rgb.shape[:2]

    converted_coords = []
    for lm in hand_landmarks.landmark:
        full_x = (lm.x * crop_width + x1) / frame_width
        full_y = (lm.y * crop_height + y1) / frame_height
        converted_coords.append((full_x, full_y))

    return hand_landmarks, converted_coords


def extract_hand_features(frame_rgb, pose_landmarks, hands_model):
    left_hand = np.zeros(NUM_HAND_LANDMARKS * 2)
    right_hand = np.zeros(NUM_HAND_LANDMARKS * 2)
    left_detected = 0.0
    right_detected = 0.0
    left_fingers = np.zeros(5)
    right_fingers = np.zeros(5)

    if pose_landmarks is None:
        return np.concatenate([left_hand, right_hand]), left_detected, right_detected, left_fingers, right_fingers

    frame_height, frame_width = frame_rgb.shape[:2]

    left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    shoulder_width_px = abs(left_shoulder.x - right_shoulder.x) * frame_width
    if shoulder_width_px < 1:
        shoulder_width_px = frame_width * 0.15

    left_wrist_lm = pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist_lm = pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

    left_crop_box = get_wrist_crop_box(left_wrist_lm, shoulder_width_px, frame_width, frame_height)
    if left_crop_box is not None:
        result = detect_hand_in_crop(frame_rgb, left_crop_box, hands_model)
        if result is not None:
            hand_landmarks, converted_coords = result
            left_hand = np.array(converted_coords).flatten()
            left_detected = 1.0
            left_fingers = compute_finger_extension(hand_landmarks)

    right_crop_box = get_wrist_crop_box(right_wrist_lm, shoulder_width_px, frame_width, frame_height)
    if right_crop_box is not None:
        result = detect_hand_in_crop(frame_rgb, right_crop_box, hands_model)
        if result is not None:
            hand_landmarks, converted_coords = result
            right_hand = np.array(converted_coords).flatten()
            right_detected = 1.0
            right_fingers = compute_finger_extension(hand_landmarks)

    hand_coords = np.concatenate([left_hand, right_hand])
    return hand_coords, left_detected, right_detected, left_fingers, right_fingers


def extract_keypoints_from_video(video_path, pose_model, hands_model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    frame_sequence = []
    left_detected_flags = []
    right_detected_flags = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_results = pose_model.process(frame_rgb)
        pose_features, pose_landmarks = extract_pose_features(pose_results)

        hand_coords, left_det, right_det, left_fingers, right_fingers = extract_hand_features(
            frame_rgb, pose_landmarks, hands_model
        )

        elbow_angles = compute_elbow_angles(pose_landmarks)

        frame_features = np.concatenate([
            pose_features,
            hand_coords,
            np.array([left_det, right_det]),
            left_fingers,
            right_fingers,
            elbow_angles,
        ])

        frame_sequence.append(frame_features)
        left_detected_flags.append(left_det)
        right_detected_flags.append(right_det)

    cap.release()

    if len(frame_sequence) == 0:
        return None, None

    detection_stats = {
        "left_hand_rate": np.mean(left_detected_flags),
        "right_hand_rate": np.mean(right_detected_flags),
    }

    return np.array(frame_sequence), detection_stats


# ---------------------------------------------------------------------
# THE FIX: sequential warm-up before any parallel workers start
# ---------------------------------------------------------------------

def warm_up_mediapipe():
    """
    Runs ONE sequential Pose + Hands initialization before any worker
    processes are spawned. This forces MediaPipe to fully download and
    validate its model file in a single process, so that when the
    Pool's workers start afterward, they all read an already-good file
    instead of racing to download it simultaneously (which corrupts it).
    """
    print("Warming up MediaPipe (one-time, sequential, ensures model file is valid)...")
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=0)
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    pose.close()
    hands.close()
    print("Warm-up complete.\n")


# ---------------------------------------------------------------------
# Multiprocessing worker setup
# ---------------------------------------------------------------------

_worker_pose_model = None
_worker_hands_model = None


def _init_worker():
    global _worker_pose_model, _worker_hands_model
    _worker_pose_model = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _worker_hands_model = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.28,
        min_tracking_confidence=0.28,
    )


def _process_one_clip(row):
    clip_path = row["clip_path"]
    gesture_label = row["gesture_label"]

    keypoints, hand_stats = extract_keypoints_from_video(
        clip_path, _worker_pose_model, _worker_hands_model
    )

    if keypoints is None:
        row["keypoint_path"] = ""
        row["frame_count"] = 0
        return row, None

    pose_part = keypoints[:, :24]
    zero_frames = int(np.all(pose_part == 0, axis=1).sum())
    pose_detection_rate = 1 - (zero_frames / len(keypoints))

    out_dir = os.path.join(KEYPOINTS_DIR, gesture_label)
    os.makedirs(out_dir, exist_ok=True)
    clip_filename = os.path.splitext(os.path.basename(clip_path))[0]
    out_path = os.path.join(out_dir, f"{clip_filename}.npy")
    np.save(out_path, keypoints)

    row["keypoint_path"] = out_path
    row["frame_count"] = len(keypoints)

    diagnostic = {
        "gesture_label": gesture_label,
        "clip_path": clip_path,
        "pose_detection_rate": pose_detection_rate,
        "left_hand_rate": hand_stats["left_hand_rate"],
        "right_hand_rate": hand_stats["right_hand_rate"],
    }
    return row, diagnostic


def process_manifest():
    start_time = time.time()

    with open(MANIFEST_PATH, "r") as f:
        rows = list(csv.DictReader(f))

    # THE FIX: warm up sequentially BEFORE spinning up the parallel pool
    warm_up_mediapipe()

    num_workers = max(1, cpu_count() - 1)
    print(f"Starting extraction on {len(rows)} clips using {num_workers} worker processes...")

    updated_rows = []
    flagged_low_pose_detection = []
    hand_detection_by_class = defaultdict(list)

    with Pool(processes=num_workers, initializer=_init_worker) as pool:
        for i, (row, diagnostic) in enumerate(pool.imap(_process_one_clip, rows, chunksize=4)):
            print(f"[{i+1}/{len(rows)}] {row['clip_path']}")

            if diagnostic is None:
                print(f"  SKIPPED (unreadable): {row['clip_path']}")
                updated_rows.append(row)
                continue

            if diagnostic["pose_detection_rate"] < 0.7:
                flagged_low_pose_detection.append(
                    (diagnostic["clip_path"], diagnostic["pose_detection_rate"])
                )

            hand_detection_by_class[diagnostic["gesture_label"]].append(
                (diagnostic["left_hand_rate"], diagnostic["right_hand_rate"])
            )

            updated_rows.append(row)

    fieldnames = list(updated_rows[0].keys())
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)
    print(f"\nDone. Manifest updated: {MANIFEST_PATH}")
    print(f"Total extraction time: {int(minutes)}m {seconds:.1f}s")

    if flagged_low_pose_detection:
        print(f"\nWARNING: {len(flagged_low_pose_detection)} clips had low POSE detection rates (<70%):")
        for path, rate in flagged_low_pose_detection:
            print(f"  - {path}  (detected in {rate:.0%} of frames)")

    print("\n" + "=" * 60)
    print("HAND DETECTION RATE PER CLASS (diagnostic)")
    print("=" * 60)
    print(f"  {'class':<30} {'left hand':>12} {'right hand':>12}")
    for gesture_label, rate_pairs in sorted(hand_detection_by_class.items()):
        left_rates = [p[0] for p in rate_pairs]
        right_rates = [p[1] for p in rate_pairs]
        print(f"  {gesture_label:<30} {np.mean(left_rates):>11.1%} {np.mean(right_rates):>12.1%}")

    try:
        for _ in range(3):
            winsound.Beep(1000, 400)
            time.sleep(0.15)
    except RuntimeError:
        pass


if __name__ == "__main__":
    process_manifest()