"""
extract_keypoints.py (v5 — Pose + wrist-cropped Hands + finger extension)
[+ multiprocessing, timing, and completion alarm added]

WHY THIS CHANGED FROM v4:
Diagnostic testing showed 0.0% hand detection across an entire class,
regardless of .mov vs .mp4 (ruling out codec issues). Visual inspection
of a sample frame showed the actual person occupies only a narrow
strip of a pillarboxed 1920x1080 frame, and the raised hand was
blurred from motion. MediaPipe Hands needs a reasonably large, sharp
view of a hand to detect it reliably — a small, blurry hand in a
mostly-empty wide frame is close to a worst case for it.

THE FIX: instead of running Hands on the entire frame, we now:
  1. Run Pose on the full frame (unchanged) to find wrist positions.
  2. For each detected wrist, crop a small region directly around it
     (sized relative to shoulder width, so it adapts to how close/far
     the person is from the camera).
  3. Run Hands only on that small cropped region — the hand now
     occupies a much larger portion of what Hands actually looks at.
  4. Convert the resulting hand landmark coordinates back into the
     original frame's coordinate system, so everything downstream
     (normalization in train.py, finger-extension math) is unaffected
     and works exactly as before.

Feature layout: 122 features per frame (24 pose + 84 hand coords +
2 hand-detected flags + 10 finger-extension + 2 elbow angles).

SPEED-UP NOTES (new):
- Clips are now processed in parallel across CPU cores via
  multiprocessing.Pool, instead of one at a time.
- Pose model_complexity lowered from 1 to 0 (faster, small accuracy
  tradeoff — check the hand/pose detection rate printout at the end;
  if it drops noticeably from your last run, bump it back to 1).
- Total wall-clock time is printed at the end.
- 3 beeps play when extraction finishes (Windows only, via winsound).
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

CROP_MARGIN_MULTIPLIER = 1.4  # crop half-size = shoulder width (in pixels) * this
MIN_VISIBILITY_FOR_CROP = 0.3  # don't bother cropping if Pose barely saw this wrist


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
    """
    Returns [left_elbow_angle, right_elbow_angle], each normalized 0.0-1.0
    (0.0 = fully bent/folded, 1.0 = fully straight arm).
    """
    if pose_landmarks is None:
        return np.array([0.5, 0.5])  # neutral fallback, no information

    def angle_for_arm(shoulder_id, elbow_id, wrist_id):
        shoulder = pose_landmarks[shoulder_id]
        elbow = pose_landmarks[elbow_id]
        wrist = pose_landmarks[wrist_id]

        if elbow.visibility < 0.3:
            return 0.5  # not confidently visible, neutral fallback

        v1 = np.array([shoulder.x - elbow.x, shoulder.y - elbow.y])
        v2 = np.array([wrist.x - elbow.x, wrist.y - elbow.y])
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.5

        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        angle_radians = np.arccos(cos_angle)  # 0 (fully bent) to pi (straight)
        return angle_radians / np.pi  # normalize to 0.0-1.0

    left_angle = angle_for_arm(
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST
    )
    right_angle = angle_for_arm(
        mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST
    )
    return np.array([left_angle, right_angle])


def get_wrist_crop_box(wrist_landmark, shoulder_width_px, frame_width, frame_height):
    """Returns (x1, y1, x2, y2) pixel box around the wrist, or None if not usable."""
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
    """Runs Hands on a cropped region, returns landmarks converted back to
    full-frame normalized coordinates, or None if no hand found."""
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
# Multiprocessing worker setup
# ---------------------------------------------------------------------
# Each worker process needs its OWN Pose/Hands model instances (mediapipe
# model objects can't be pickled/shared across processes). _init_worker()
# runs once per worker when the Pool starts, creating that worker's models
# as module-level globals which _process_one_clip() then reuses.

_worker_pose_model = None
_worker_hands_model = None


def _init_worker():
    global _worker_pose_model, _worker_hands_model
    _worker_pose_model = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,   # lowered from 1 -> 0 for speed
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _worker_hands_model = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )


def _process_one_clip(row):
    """Runs inside a worker process. Returns (updated_row, diagnostic_or_None)."""
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

    num_workers = max(1, cpu_count() - 1)  # leave 1 core free for the OS
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

    # 3 beeps to signal completion (Windows only)
    try:
        for _ in range(3):
            winsound.Beep(1000, 400)
            time.sleep(0.15)
    except RuntimeError:
        pass  # non-Windows environment, skip silently


if __name__ == "__main__":
    process_manifest()