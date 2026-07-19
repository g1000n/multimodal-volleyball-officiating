"""
extract_keypoints.py (v6 — Pose + COMBINED-CROP two-hand detection + finger extension)
[OPTIMIZED: multiprocessing + auto warm-up fix + timer + alarm]

Feature layout: 122 features per frame (24 pose + 84 hand coords +
2 hand-detected flags + 10 finger-extension + 2 elbow angles).

v6 CHANGE — combined-crop, two-hand detection with nearest-wrist assignment:
Previously (v5), each wrist got its OWN independent crop, and each crop
searched for exactly one hand. When the referee's hands came close
together (e.g. during ball_out or double_contact), both crops could
overlap the SAME physical hand — so that one hand would get detected
twice (once labeled left, once labeled right), while the other hand
went undetected. This showed up during manual clip review as the
cyan/magenta hand overlays landing on top of each other, and one hand's
skeleton being missing for sustained stretches even though the video
itself looked fine.

The fix: take ONE combined crop that covers both wrists, ask MediaPipe
for up to 2 hands in that single region (letting MediaPipe's own
multi-hand detection logic tell them apart), then assign each detected
hand to whichever wrist it's physically closest to. This structurally
prevents the "two independent searches grab the same hand" failure
mode, instead of trying to patch around it after the fact.

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

CROP_MARGIN_MULTIPLIER = 1.0
# Single-wrist crops use a SMALLER margin than the combined crop — they
# only need to comfortably contain one hand, not account for spanning
# both wrists. Using the same large margin as the combined crop for
# both was the bug: it made single crops so big that the overlap check
# almost always concluded "would overlap," silently forcing the
# combined (lower-resolution) branch even for widely spread arms.
SINGLE_CROP_MARGIN_MULTIPLIER = 0.9
MIN_VISIBILITY_FOR_CROP = 0.18
# If the wrists are farther apart than this (relative to shoulder width),
# use two separate high-resolution crops instead of one wide combined
# crop. Close together = combined crop (avoids double-detecting the same
# hand). Far apart = separate crops (avoids resolution loss on wide poses).
WRIST_DISTANCE_COMBINED_THRESHOLD = 1.3  # in units of shoulder width


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


def get_combined_hand_crop_box(left_wrist, right_wrist, shoulder_width_px, frame_width, frame_height):
    """
    ONE crop covering both wrists. Only appropriate when the wrists are
    close together — otherwise the crop has to stretch wide, lowering
    effective resolution per hand once MediaPipe resizes it internally,
    which can cause a clearly-visible hand to go undetected.
    """
    margin = shoulder_width_px * CROP_MARGIN_MULTIPLIER

    xs, ys = [], []
    if left_wrist.visibility >= MIN_VISIBILITY_FOR_CROP:
        xs.append(left_wrist.x * frame_width)
        ys.append(left_wrist.y * frame_height)
    if right_wrist.visibility >= MIN_VISIBILITY_FOR_CROP:
        xs.append(right_wrist.x * frame_width)
        ys.append(right_wrist.y * frame_height)

    if not xs:
        return None

    x1 = int(max(0, min(xs) - margin))
    x2 = int(min(frame_width, max(xs) + margin))
    y1 = int(max(0, min(ys) - margin))
    y2 = int(min(frame_height, max(ys) + margin))

    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    return (x1, y1, x2, y2)


def get_single_wrist_crop_box(wrist_landmark, shoulder_width_px, frame_width, frame_height):
    """
    A tight crop around ONE wrist only — higher effective resolution
    for that hand than a combined crop would give. Only used when this
    is guaranteed NOT to overlap the other wrist's crop (checked by the
    caller via crops_would_overlap()) — so both crops searching
    independently can never land on the same physical hand.
    """
    if wrist_landmark.visibility < MIN_VISIBILITY_FOR_CROP:
        return None

    cx = wrist_landmark.x * frame_width
    cy = wrist_landmark.y * frame_height
    half_size = max(shoulder_width_px * SINGLE_CROP_MARGIN_MULTIPLIER, 35)

    x1 = int(max(0, cx - half_size))
    x2 = int(min(frame_width, cx + half_size))
    y1 = int(max(0, cy - half_size))
    y2 = int(min(frame_height, cy + half_size))

    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    return (x1, y1, x2, y2)


def detect_hands_in_combined_crop(frame_rgb, crop_box, hands_model):
    """Returns a list of (hand_landmarks, full_frame_coords) — 0, 1, or 2 hands."""
    x1, y1, x2, y2 = crop_box
    crop = frame_rgb[y1:y2, x1:x2]

    results = hands_model.process(crop)
    if not results.multi_hand_landmarks:
        return []

    crop_width, crop_height = x2 - x1, y2 - y1
    frame_height, frame_width = frame_rgb.shape[:2]

    detected = []
    for hand_landmarks in results.multi_hand_landmarks:
        converted_coords = []
        for lm in hand_landmarks.landmark:
            full_x = (lm.x * crop_width + x1) / frame_width
            full_y = (lm.y * crop_height + y1) / frame_height
            converted_coords.append((full_x, full_y))
        detected.append((hand_landmarks, converted_coords))

    return detected


def debug_get_crop_info(pose_landmarks, frame_width, frame_height):
    """
    DEBUG ONLY — replicates extract_hand_features()'s crop decision so
    review tools can visualize exactly which strategy was chosen and
    where the crop box(es) landed, without touching the real extraction
    function's signature or behavior. Returns:
        (mode, boxes) where mode is "combined" or "separate" or "none",
        and boxes is a list of (x1,y1,x2,y2) tuples (1 or 2 boxes).
    """
    if pose_landmarks is None:
        return "none", []

    left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    shoulder_width_px = abs(left_shoulder.x - right_shoulder.x) * frame_width
    if shoulder_width_px < 1:
        shoulder_width_px = frame_width * 0.15

    left_wrist_lm = pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist_lm = pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

    both_wrists_visible = (
        left_wrist_lm.visibility >= MIN_VISIBILITY_FOR_CROP
        and right_wrist_lm.visibility >= MIN_VISIBILITY_FOR_CROP
    )

    use_combined_crop = True
    if both_wrists_visible:
        dx = (left_wrist_lm.x - right_wrist_lm.x) * frame_width
        dy = (left_wrist_lm.y - right_wrist_lm.y) * frame_height
        wrist_dist_px = np.sqrt(dx * dx + dy * dy)
        single_crop_half_size = max(shoulder_width_px * SINGLE_CROP_MARGIN_MULTIPLIER, 35)
        use_combined_crop = wrist_dist_px < (2 * single_crop_half_size)

    if use_combined_crop:
        box = get_combined_hand_crop_box(left_wrist_lm, right_wrist_lm, shoulder_width_px, frame_width, frame_height)
        return "combined", ([box] if box else [])
    else:
        boxes = []
        lb = get_single_wrist_crop_box(left_wrist_lm, shoulder_width_px, frame_width, frame_height)
        rb = get_single_wrist_crop_box(right_wrist_lm, shoulder_width_px, frame_width, frame_height)
        if lb:
            boxes.append(lb)
        if rb:
            boxes.append(rb)
        return "separate", boxes


def extract_hand_features(frame_rgb, pose_landmarks, hands_model):
    """
    v7: ADAPTIVE crop strategy.
      - Wrists CLOSE together (e.g. ball_out, double_contact holds):
        one combined crop, up to 2 hands detected in it, assigned to
        the nearest wrist. Prevents the same physical hand being
        double-detected by two overlapping independent crops.
      - Wrists FAR apart (e.g. both arms raised wide, service
        authorization poses): two separate, tight, high-resolution
        crops, one per wrist. Prevents a clearly-visible hand going
        undetected because a wide combined crop lowered its effective
        resolution after MediaPipe resizes it internally.
    """
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

    # Decide which strategy to use based on whether two independent
    # single-wrist crops would actually overlap each other, given their
    # real size — NOT an arbitrary distance guess. This makes the two
    # branches mutually exclusive by construction: if separate crops
    # would overlap, we MUST use the combined (multi-hand) branch,
    # otherwise two independent searches can land on the same hand.
    both_wrists_visible = (
        left_wrist_lm.visibility >= MIN_VISIBILITY_FOR_CROP
        and right_wrist_lm.visibility >= MIN_VISIBILITY_FOR_CROP
    )

    use_combined_crop = True
    if both_wrists_visible:
        # Pixel-space distance, correctly scaling x and y by their own
        # frame dimensions (previous version incorrectly scaled both
        # by frame_width only, distorting the distance on non-square
        # frames).
        dx = (left_wrist_lm.x - right_wrist_lm.x) * frame_width
        dy = (left_wrist_lm.y - right_wrist_lm.y) * frame_height
        wrist_dist_px = np.sqrt(dx * dx + dy * dy)

        single_crop_half_size = max(shoulder_width_px * SINGLE_CROP_MARGIN_MULTIPLIER, 35)
        # Two crops of this half-size, centered on each wrist, would
        # touch/overlap once the wrists are closer than 2x half-size.
        crops_would_overlap = wrist_dist_px < (2 * single_crop_half_size)

        use_combined_crop = crops_would_overlap

    if use_combined_crop:
        crop_box = get_combined_hand_crop_box(left_wrist_lm, right_wrist_lm, shoulder_width_px, frame_width, frame_height)
        if crop_box is None:
            return np.concatenate([left_hand, right_hand]), left_detected, right_detected, left_fingers, right_fingers

        detected_hands = detect_hands_in_combined_crop(frame_rgb, crop_box, hands_model)
        if not detected_hands:
            return np.concatenate([left_hand, right_hand]), left_detected, right_detected, left_fingers, right_fingers

        left_wrist_full = np.array([left_wrist_lm.x, left_wrist_lm.y])
        right_wrist_full = np.array([right_wrist_lm.x, right_wrist_lm.y])
        hand_centroids = [np.array(coords).mean(axis=0) for _, coords in detected_hands]

        assignments = {}
        remaining_hand_indices = list(range(len(detected_hands)))
        remaining_wrists = {"left": left_wrist_full, "right": right_wrist_full}

        while remaining_hand_indices and remaining_wrists:
            best = None
            for hand_idx in remaining_hand_indices:
                for wrist_key, wrist_pos in remaining_wrists.items():
                    dist = np.linalg.norm(hand_centroids[hand_idx] - wrist_pos)
                    if best is None or dist < best[0]:
                        best = (dist, hand_idx, wrist_key)
            _, hand_idx, wrist_key = best
            assignments[wrist_key] = hand_idx
            remaining_hand_indices.remove(hand_idx)
            del remaining_wrists[wrist_key]

        if "left" in assignments:
            hand_landmarks, converted_coords = detected_hands[assignments["left"]]
            left_hand = np.array(converted_coords).flatten()
            left_detected = 1.0
            left_fingers = compute_finger_extension(hand_landmarks)

        if "right" in assignments:
            hand_landmarks, converted_coords = detected_hands[assignments["right"]]
            right_hand = np.array(converted_coords).flatten()
            right_detected = 1.0
            right_fingers = compute_finger_extension(hand_landmarks)

    else:
        # Wrists far apart — use two separate, tighter, higher-resolution crops
        left_crop_box = get_single_wrist_crop_box(left_wrist_lm, shoulder_width_px, frame_width, frame_height)
        if left_crop_box is not None:
            results = detect_hands_in_combined_crop(frame_rgb, left_crop_box, hands_model)
            if results:
                hand_landmarks, converted_coords = results[0]
                left_hand = np.array(converted_coords).flatten()
                left_detected = 1.0
                left_fingers = compute_finger_extension(hand_landmarks)

        right_crop_box = get_single_wrist_crop_box(right_wrist_lm, shoulder_width_px, frame_width, frame_height)
        if right_crop_box is not None:
            results = detect_hands_in_combined_crop(frame_rgb, right_crop_box, hands_model)
            if results:
                hand_landmarks, converted_coords = results[0]
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
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
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
        max_num_hands=2,
        min_detection_confidence=0.1,
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