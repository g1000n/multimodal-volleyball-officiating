"""
test_reference_clips.py [OPTIMIZED - parallel version]

Scans data/reference_clips/<gesture_name>/*.mp4 (or .mov) and runs each
one through the ALREADY-TRAINED model, IN PARALLEL across CPU cores.

Prints a summary table: expected label, predicted label, confidence,
and match, for every reference clip.

SAFE ON FRESH DEVICES: does a one-time SEQUENTIAL warm-up (single
process) to make sure MediaPipe's model file is fully downloaded and
valid BEFORE spinning up parallel workers. This avoids the race
condition where multiple workers try to download the same model file
at once and corrupt it (the "Model provided must have at least 7
bytes" / "should be 'TFL3'" errors seen before).

This is a DIAGNOSTIC tool only -- these clips are never added to
data/raw_clips/, the manifest, or training.

--------------------------------------------------------------------
PATCH (this version): _init_worker()'s hands_model settings were stale
-- they did NOT match extract_keypoints.py's actual worker settings
used to build the training data (max_num_hands=2,
min_detection_confidence=0.1, min_tracking_confidence=0.28). This file
was still using max_num_hands=1, 0.4/0.4, which is the same class of
bug Anya found and fixed in live_auto_inference.py -- just never
applied here too.

This matters even with ABLATE_HAND_COORDS=True: ablation only drops
the raw 84 hand x/y coordinates, but the model still trains on
hand-detected flags (2 features) and finger-extension (10 features),
both of which come from this hands_model. Evaluating reference clips
with a stricter, single-hand detector than what training data was
extracted with is a real train/eval mismatch, separate from any
labeling issue -- it can independently hurt reference-clip accuracy
and confidence readings.

Fixed to match extract_keypoints.py's worker exactly.
--------------------------------------------------------------------

Usage:
    python test_reference_clips.py
"""

import os
import json
import numpy as np
import torch
from multiprocessing import Pool, cpu_count

from model import GestureCNNLSTM
from extract_keypoints import mp_pose, mp_hands, extract_keypoints_from_video
from train import (
    normalize_sequence,
    resample_sequence,
    SEQUENCE_LENGTH,
    TOTAL_FEATURES,
    ablate,                  # NEW
    ABLATE_HAND_COORDS,      # NEW
    ABLATED_FEATURE_COUNT,   # NEW
)

REFERENCE_DIR = "data/reference_clips"
MODEL_PATH = "models/final_model.pt"
LABEL_MAP_PATH = "models/label_map.json"
VALID_EXTENSIONS = (".mp4", ".mov")


def warm_up_mediapipe():
    """
    Runs ONE sequential Pose + Hands initialization before any parallel
    workers start, to force a clean, single-process model download.
    Prevents the multi-worker race condition that corrupts the model file.
    """
    print("Warming up MediaPipe models (one-time, sequential)...")
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=0)
    # PATCH: max_num_hands=2 to match extract_keypoints.py's worker --
    # this warm-up just needs to trigger the model download, but keeping
    # it consistent avoids any confusion when reading this file later.
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
    pose.close()
    hands.close()
    print("Warm-up complete.\n")


# ---------------------------------------------------------------------
# Worker setup -- each process gets its own MediaPipe + PyTorch models
# ---------------------------------------------------------------------

_worker_pose_model = None
_worker_hands_model = None
_worker_gesture_model = None
_worker_idx_to_label = None
_worker_device = None


def _init_worker():
    global _worker_pose_model, _worker_hands_model
    global _worker_gesture_model, _worker_idx_to_label, _worker_device

    _worker_pose_model = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    # PATCH: these three values were previously max_num_hands=1,
    # min_detection_confidence=0.4, min_tracking_confidence=0.4 -- stale
    # and inconsistent with extract_keypoints.py's actual training-time
    # worker settings. Now matches exactly, so hand-detected flags and
    # finger-extension features are computed under the same conditions
    # the model was trained on.
    _worker_hands_model = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.1,
        min_tracking_confidence=0.28,
    )

    with open(LABEL_MAP_PATH, "r") as f:
        label_to_idx = json.load(f)
    _worker_idx_to_label = {v: k for k, v in label_to_idx.items()}

    _worker_device = torch.device("cpu")  # CPU is safest across multiple processes

    # CHANGED: input_size must match whatever train.py actually trained
    # with. If ABLATE_HAND_COORDS was True during training, the saved
    # checkpoint expects 38 features, not 122 -- loading it with the
    # wrong size will error out or silently produce garbage.
    input_size = ABLATED_FEATURE_COUNT if ABLATE_HAND_COORDS else TOTAL_FEATURES
    _worker_gesture_model = GestureCNNLSTM(
        input_size=input_size, num_classes=len(label_to_idx)
    ).to(_worker_device)
    _worker_gesture_model.load_state_dict(torch.load(MODEL_PATH, map_location=_worker_device))
    _worker_gesture_model.eval()


def _process_one_reference_clip(task):
    """task = (video_path, expected_label, filename)"""
    video_path, expected_label, filename = task

    keypoints, hand_stats = extract_keypoints_from_video(
        video_path, _worker_pose_model, _worker_hands_model
    )
    if keypoints is None:
        return {"file": filename, "expected": expected_label, "predicted": None,
                "confidence": 0.0, "correct": False, "skipped": True}

    normalized = normalize_sequence(keypoints)
    normalized = ablate(normalized)   # NEW LINE — must match what the model was trained on
    resampled = resample_sequence(normalized, SEQUENCE_LENGTH)
    x = torch.tensor(resampled, dtype=torch.float32).unsqueeze(0).to(_worker_device)

    with torch.no_grad():
        logits = _worker_gesture_model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = int(np.argmax(probs))
    predicted_label = _worker_idx_to_label[top_idx]
    confidence = float(probs[top_idx])

    return {
        "file": filename,
        "expected": expected_label,
        "predicted": predicted_label,
        "confidence": confidence,
        "correct": predicted_label == expected_label,
        "skipped": False,
    }


def main():
    if not os.path.isdir(REFERENCE_DIR):
        print(f"No reference clips found. Create '{REFERENCE_DIR}/<gesture_name>/clip.mp4' first.")
        return

    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_MAP_PATH):
        print(f"Model files not found ({MODEL_PATH} / {LABEL_MAP_PATH}). Run train.py first, or copy them from a trained device.")
        return

    warm_up_mediapipe()

    # Build the full list of (video_path, expected_label, filename) tasks
    tasks = []
    for expected_label in sorted(os.listdir(REFERENCE_DIR)):
        folder_path = os.path.join(REFERENCE_DIR, expected_label)
        if not os.path.isdir(folder_path):
            continue
        for filename in sorted(os.listdir(folder_path)):
            if not filename.lower().endswith(VALID_EXTENSIONS):
                continue
            video_path = os.path.join(folder_path, filename)
            tasks.append((video_path, expected_label, filename))

    if not tasks:
        print("No reference clips found to test.")
        return

    num_workers = min(len(tasks), max(1, cpu_count() - 1))
    print(f"Testing {len(tasks)} reference clips using {num_workers} worker process(es)...\n")
    if ABLATE_HAND_COORDS:
        print(f"NOTE: ABLATE_HAND_COORDS is True -- evaluating the hand-coordinate-ablated model "
              f"({ABLATED_FEATURE_COUNT} features).\n")

    results = []
    with Pool(processes=num_workers, initializer=_init_worker) as pool:
        for i, result in enumerate(pool.imap(_process_one_reference_clip, tasks)):
            status = "SKIPPED" if result["skipped"] else ("YES" if result["correct"] else "NO")
            print(f"[{i+1}/{len(tasks)}] {result['file']:<45} -> {status}")
            results.append(result)

    valid_results = [r for r in results if not r["skipped"]]
    skipped_results = [r for r in results if r["skipped"]]

    if not valid_results:
        print("\nNo clips were successfully processed.")
        return

    print("\n" + "=" * 90)
    print("REFERENCE CLIP TEST RESULTS")
    print("=" * 90)
    print(f"{'file':<45} {'expected':<28} {'predicted':<28} {'conf':>6} {'match'}")
    for r in valid_results:
        match_symbol = "YES" if r["correct"] else "NO"
        print(f"{r['file']:<45} {r['expected']:<28} {r['predicted']:<28} "
              f"{r['confidence']:>5.1%} {match_symbol}")

    if skipped_results:
        print(f"\n{len(skipped_results)} clip(s) skipped (unreadable):")
        for r in skipped_results:
            print(f"  - {r['file']}")

    num_correct = sum(1 for r in valid_results if r["correct"])
    print("=" * 90)
    print(f"Overall: {num_correct}/{len(valid_results)} reference clips correctly predicted")


if __name__ == "__main__":
    main()