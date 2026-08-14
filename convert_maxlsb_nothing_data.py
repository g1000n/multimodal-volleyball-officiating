"""
convert_maxlsb_nothing_data.py

Converts MaxLSB/volley-judge's raw "Nothing" class keypoint sequences
(collected via MediaPipe Holistic, 258 values/frame: 33 pose landmarks
x [x,y,z,visibility] + 21 left-hand x [x,y,z] + 21 right-hand x [x,y,z])
into THIS project's feature format (122 raw features/frame: 24 pose +
84 hand coords + 2 hand-detected flags + 10 finger-extension + 2 elbow
angles), so his real "Nothing" data can be added as a second,
independent contributing person -- without needing to wait on
volunteers, since the landmark indices are the same standard MediaPipe
Pose/Hands layout in both projects, just extracted via Holistic
(his) vs. separate Pose+Hands (ours).

WHAT THIS DOES:
  1. Reads each numbered sequence folder under SOURCE_DIR (his raw
     per-frame .npy files: 0.npy, 1.npy, ... one per frame).
  2. Converts each frame's 258 raw values into our 122-feature layout,
     using the same landmark indices/definitions as extract_keypoints.py
     (so the SAME normalize_sequence()/ablate() pipeline in train.py
     works on this data unmodified).
  3. Saves the converted sequence directly into data/keypoints/nothing/
     (skipping build_manifest.py and extract_keypoints.py entirely for
     this batch, since we're going straight from his raw keypoints to
     our keypoint format -- there's no video file to extract from).
  4. Appends corresponding rows to data/dataset_manifest.csv with a new
     person_id (default "pmax") so dataset_split.py treats this as a
     genuinely distinct contributing person for subject-diversity
     purposes -- not just more of your own single-person data.

Run this ONCE. It's safe to re-run (skips already-converted sequences
by checking if the output .npy already exists), but you'll want to
manually double check the manifest afterward if you rerun it.

After running this, skip straight to:
    python dataset_split.py
    python train.py
(no need to run build_manifest.py or extract_keypoints.py for this data)
"""

import os
import csv
import numpy as np

# ---------------------------------------------------------------------
# EDIT THIS: point at the ROOT of his "data" folder (the one containing
# Nothing/, DbHit/, OutofB/, pointL/, pointR/, Substi/ as subfolders).
# Use a raw string (r"...") or forward slashes to avoid backslash
# escape errors on Windows paths.
# ---------------------------------------------------------------------
SOURCE_DATA_ROOT = r"C:\Users\anouc\Downloads\data\data"  # EDIT THIS locally to your actual path -- don't commit your real path

# Maps his folder names to YOUR gesture_label names. "Substi" has no
# equivalent in your 7-class set and is intentionally left out --
# importing it as anything (including as more "nothing" data) would be
# wrong, since it's a real, distinct signal you don't classify, not an
# absence of one.
CLASS_MAPPING = {
    "Nothing": "nothing",
    "DbHit": "double_contact",
    "OutofB": "ball_out",
    "pointL": "team_to_serve_left",
    "pointR": "team_to_serve_right",
    # "Substi" intentionally omitted -- no equivalent class
}

MANIFEST_PATH = "data/dataset_manifest.csv"
NEW_PERSON_ID = "pmax"   # a new, distinct person_id -- NOT one of your existing p01-p09/p101 codes
SOURCE_TAG = "external_maxlsb"

# --- His raw layout (from his extraction.py) ---
HIS_POSE_LANDMARKS = 33
HIS_POSE_VALUES_PER_LANDMARK = 4   # x, y, z, visibility
HIS_HAND_LANDMARKS = 21
HIS_HAND_VALUES_PER_LANDMARK = 3   # x, y, z
HIS_TOTAL_VALUES = (HIS_POSE_LANDMARKS * HIS_POSE_VALUES_PER_LANDMARK
                    + HIS_HAND_LANDMARKS * HIS_HAND_VALUES_PER_LANDMARK * 2)  # 258

# --- Our layout (from extract_keypoints.py / mp_pose.PoseLandmark) ---
# Standard MediaPipe Pose landmark indices -- same in his Holistic pose
# output and our standalone Pose output, since both use the same
# underlying 33-point Pose model.
OUR_LANDMARK_IDS = {
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
    "LEFT_HIP": 23, "RIGHT_HIP": 24,
}
UPPER_BODY_ORDER = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
                    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP"]

FINGER_JOINTS = {
    "thumb":  (4, 2),
    "index":  (8, 6),
    "middle": (12, 10),
    "ring":   (16, 14),
    "pinky":  (20, 18),
}
EXTENSION_MARGIN = 1.1


def convert_one_frame(raw_258):
    """
    raw_258: flat array of 258 values from ONE of his frame .npy files.
    Layout: [0:132] = pose (33 x [x,y,z,vis]), [132:195] = left hand
    (21 x [x,y,z]), [195:258] = right hand (21 x [x,y,z]).
    Returns a 122-value array matching our extract_keypoints.py layout.
    """
    pose_flat = raw_258[:132].reshape(33, 4)
    lh_flat = raw_258[132:132 + 63].reshape(21, 3)
    rh_flat = raw_258[132 + 63:258].reshape(21, 3)

    # --- Pose: our 8 landmarks, (x, y, visibility) each, dropping z ---
    pose_features = []
    for name in UPPER_BODY_ORDER:
        idx = OUR_LANDMARK_IDS[name]
        x, y, z, vis = pose_flat[idx]
        pose_features.extend([x, y, vis])
    pose_features = np.array(pose_features)  # 24 values

    # --- Hands: (x, y) only, dropping z, same left-then-right order ---
    left_hand_xy = lh_flat[:, :2].flatten()   # 42 values
    right_hand_xy = rh_flat[:, :2].flatten()  # 42 values

    # Detection flags: his code writes np.zeros(21*3) when a hand wasn't
    # detected that frame -- same convention we can check for here.
    left_detected = 0.0 if np.all(lh_flat == 0) else 1.0
    right_detected = 0.0 if np.all(rh_flat == 0) else 1.0

    # --- Finger extension: same tip/base distance-ratio logic as
    # compute_finger_extension() in extract_keypoints.py, adapted to
    # work on raw (21,2) arrays instead of MediaPipe landmark objects ---
    def finger_extension_from_xy(hand_xy_21x2):
        wrist = hand_xy_21x2[0]
        extensions = []
        for _, (tip_idx, base_idx) in FINGER_JOINTS.items():
            tip_dist = np.linalg.norm(hand_xy_21x2[tip_idx] - wrist)
            base_dist = np.linalg.norm(hand_xy_21x2[base_idx] - wrist)
            extended = 1.0 if tip_dist > base_dist * EXTENSION_MARGIN else 0.0
            extensions.append(extended)
        return np.array(extensions)

    left_fingers = finger_extension_from_xy(lh_flat[:, :2]) if left_detected else np.zeros(5)
    right_fingers = finger_extension_from_xy(rh_flat[:, :2]) if right_detected else np.zeros(5)

    # --- Elbow angles: same vector-angle logic as compute_elbow_angles()
    # in extract_keypoints.py, adapted to raw pose array indices ---
    def elbow_angle(shoulder_idx, elbow_idx, wrist_idx):
        shoulder = pose_flat[shoulder_idx][:2]
        elbow_xy = pose_flat[elbow_idx][:2]
        elbow_vis = pose_flat[elbow_idx][3]
        wrist = pose_flat[wrist_idx][:2]

        if elbow_vis < 0.3:
            return 0.5

        v1 = shoulder - elbow_xy
        v2 = wrist - elbow_xy
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.5

        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        return np.arccos(cos_angle) / np.pi

    left_elbow_angle = elbow_angle(OUR_LANDMARK_IDS["LEFT_SHOULDER"],
                                    OUR_LANDMARK_IDS["LEFT_ELBOW"],
                                    OUR_LANDMARK_IDS["LEFT_WRIST"])
    right_elbow_angle = elbow_angle(OUR_LANDMARK_IDS["RIGHT_SHOULDER"],
                                     OUR_LANDMARK_IDS["RIGHT_ELBOW"],
                                     OUR_LANDMARK_IDS["RIGHT_WRIST"])

    return np.concatenate([
        pose_features,                              # 24
        left_hand_xy, right_hand_xy,                 # 84
        np.array([left_detected, right_detected]),   # 2
        left_fingers, right_fingers,                 # 10
        np.array([left_elbow_angle, right_elbow_angle]),  # 2
    ])  # total: 122


def convert_one_sequence(sequence_folder):
    """Reads all frame .npy files in one of his numbered sequence
    folders, in order, and converts the whole sequence."""
    frame_files = sorted(
        [f for f in os.listdir(sequence_folder) if f.endswith(".npy")],
        key=lambda f: int(os.path.splitext(f)[0])
    )
    if not frame_files:
        return None

    converted_frames = []
    for frame_file in frame_files:
        raw = np.load(os.path.join(sequence_folder, frame_file))
        if raw.shape[0] != HIS_TOTAL_VALUES:
            print(f"  WARNING: {sequence_folder}/{frame_file} has {raw.shape[0]} values, "
                  f"expected {HIS_TOTAL_VALUES} -- skipping this sequence.")
            return None
        converted_frames.append(convert_one_frame(raw))

    return np.array(converted_frames)


def main():
    if not os.path.isdir(SOURCE_DATA_ROOT):
        print(f"SOURCE_DATA_ROOT not found: {SOURCE_DATA_ROOT}")
        print("Edit SOURCE_DATA_ROOT at the top of this script to point at his 'data' folder.")
        return

    # Load existing manifest to check for duplicates on rerun, and to
    # append to rather than overwrite.
    existing_rows = []
    fieldnames = ["clip_path", "gesture_label", "person_id", "take_number",
                  "source", "keypoint_path", "frame_count", "split"]
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            # if reader.fieldnames:
                # fieldnames = reader.fieldnames

    already_converted_paths = {row["keypoint_path"] for row in existing_rows if row.get("keypoint_path")}

    all_new_rows = []
    total_converted = 0
    total_skipped = 0

    for his_folder_name, our_gesture_label in CLASS_MAPPING.items():
        source_dir = os.path.join(SOURCE_DATA_ROOT, his_folder_name)
        if not os.path.isdir(source_dir):
            print(f"WARNING: {source_dir} not found -- skipping this class.")
            continue

        output_dir = os.path.join("data", "keypoints", our_gesture_label)
        os.makedirs(output_dir, exist_ok=True)

        sequence_folders = sorted(
            [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))],
            key=lambda d: int(d) if d.isdigit() else d
        )

        print(f"\n{his_folder_name} -> {our_gesture_label}: found {len(sequence_folders)} sequence folders.")

        for seq_name in sequence_folders:
            seq_folder_path = os.path.join(source_dir, seq_name)
            out_filename = f"maxlsb_{our_gesture_label}_{seq_name}.npy"
            out_path = os.path.join(output_dir, out_filename)

            if out_path in already_converted_paths and os.path.exists(out_path):
                total_skipped += 1
                continue

            converted = convert_one_sequence(seq_folder_path)
            if converted is None:
                continue

            np.save(out_path, converted)

            all_new_rows.append({
                "clip_path": f"external/maxlsb_volley-judge/{his_folder_name}/{seq_name}",
                "gesture_label": our_gesture_label,
                "person_id": NEW_PERSON_ID,
                "take_number": seq_name,
                "source": SOURCE_TAG,
                "keypoint_path": out_path,
                "frame_count": len(converted),
                "split": "",  # left blank -- dataset_split.py will assign
            })
            total_converted += 1
            print(f"  Converted sequence {seq_name}: {len(converted)} frames -> {out_path}")

    if all_new_rows:
        all_rows = existing_rows + all_new_rows
        with open(MANIFEST_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    print(f"\nDone. {total_converted} new sequences converted across all classes, "
          f"{total_skipped} already done (skipped).")
    print(f"Manifest updated: {MANIFEST_PATH}")
    print(f"\nNext steps -- skip build_manifest.py and extract_keypoints.py for this batch, just run:")
    print("  python dataset_split.py")
    print("  python train.py")


if __name__ == "__main__":
    main()