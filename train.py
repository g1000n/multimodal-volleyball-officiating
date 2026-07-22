"""
train.py

Loads keypoint sequences according to the manifest's train/val/test
split (produced by dataset_split.py), normalizes and resamples them
to a fixed length, trains the CNN-LSTM model, and evaluates on the
held-out test set.

Run order (full pipeline):
    1. build_manifest.py
    2. extract_keypoints.py
    3. dataset_split.py
    4. train.py   <- this script
"""

import csv
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from model import GestureCNNLSTM

MANIFEST_PATH = "data/dataset_manifest.csv"
SEQUENCE_LENGTH = 50       # fixed frame count after resampling
POSE_FEATURES = 24         # 8 pose landmarks x (x, y, visibility)
HAND_COORD_FEATURES = 84   # 2 hands x 21 landmarks x (x, y)
HAND_FLAG_FEATURES = 2     # left_detected, right_detected
FINGER_FEATURES = 10       # 2 hands x 5 fingers (thumb, index, middle, ring, pinky)
ELBOW_ANGLE_FEATURES = 2   # left_elbow_angle, right_elbow_angle
TOTAL_FEATURES = POSE_FEATURES + HAND_COORD_FEATURES + HAND_FLAG_FEATURES + FINGER_FEATURES + ELBOW_ANGLE_FEATURES  # 122
BATCH_SIZE = 16

# --- NEW: hand-coordinate ablation experiment ---
# Tests the hypothesis that the model over-relies on the 84 raw hand
# x/y coordinates, which are reliable in training footage (85-100%
# detection) but frequently missing/garbage on real external footage
# (seen as low as 0-35% detection). If True, those 84 columns are
# dropped before the model ever sees them, leaving only pose (24) +
# hand-detected flags (2) + finger-extension (10) + elbow angles (2)
# = 38 features. Nothing on disk changes -- this only affects what
# gets fed into the model at train/eval time.
ABLATE_HAND_COORDS = True
ABLATED_FEATURE_COUNT = POSE_FEATURES + HAND_FLAG_FEATURES + FINGER_FEATURES + ELBOW_ANGLE_FEATURES  # 38

# Column ranges for the tie-breaker logic (see below)
LEFT_FINGERS_START = POSE_FEATURES + HAND_COORD_FEATURES + HAND_FLAG_FEATURES        # 110
RIGHT_FINGERS_START = LEFT_FINGERS_START + 5                                        # 115
# Within each 5-value finger block: [thumb, index, middle, ring, pinky]
INDEX_FINGER_OFFSET = 1
MIDDLE_FINGER_OFFSET = 2
RING_FINGER_OFFSET = 3
PINKY_FINGER_OFFSET = 4

# Classes involved in the known confusable pair — tie-breaker only activates
# when the model's top-2 predictions fall inside this set and are close in
# probability. Adjust if your dataset's confusable pair changes.
CONFUSABLE_CLASSES = {"double_contact", "service_authorization_left", "service_authorization_right"}
TIE_BREAKER_PROB_MARGIN = 0.20   # only override when top-2 probabilities are this close
PEACE_SIGN_THRESHOLD = 0.5       # fraction of frames needed to count a finger as "extended" overall
EPOCHS = 100
PATIENCE = 10              # early stopping
LEARNING_RATE = 5e-4
GRAD_CLIP_NORM = 1.0       # caps how large a single update step can be, prevents loss spikes
MODEL_SAVE_PATH = "models/final_model.pt"
LABEL_MAP_SAVE_PATH = "models/label_map.json"


# ---------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------

def load_manifest_rows():
    with open(MANIFEST_PATH, "r") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("keypoint_path") and r["keypoint_path"] != ""]
    if not rows:
        raise RuntimeError("No rows with extracted keypoints found. Run extract_keypoints.py first.")
    if "split" not in rows[0] or rows[0]["split"] == "":
        raise RuntimeError("Manifest has no split assignments. Run dataset_split.py first.")
    return rows


def normalize_sequence(seq):
    """
    seq shape: (frames, TOTAL_FEATURES) = (frames, 122)
      [0:24]     pose:        8 landmarks x (x, y, visibility)
      [24:66]    left hand:  21 landmarks x (x, y)
      [66:108]   right hand: 21 landmarks x (x, y)
      [108]      left_hand_detected   (0.0 or 1.0)
      [109]      right_hand_detected  (0.0 or 1.0)
      [110:115]  left hand finger extension  (0.0-1.0 each)
      [115:120]  right hand finger extension (0.0-1.0 each)
      [120:122]  elbow angles

    Pose and hand COORDINATES are normalized relative to shoulder
    midpoint/width. Detection flags and finger-extension values are
    NOT normalized — they're already clean 0/1-ish signals, and
    scaling them would distort that meaning.
    """
    pose_part = seq[:, :POSE_FEATURES].reshape(seq.shape[0], 8, 3)
    left_hand_part = seq[:, POSE_FEATURES:POSE_FEATURES + 42].reshape(seq.shape[0], 21, 2)
    right_hand_part = seq[:, POSE_FEATURES + 42:POSE_FEATURES + 84].reshape(seq.shape[0], 21, 2)
    hand_flags = seq[:, POSE_FEATURES + 84:POSE_FEATURES + 86]        # (frames, 2)
    finger_features = seq[:, POSE_FEATURES + 86:POSE_FEATURES + 96]   # (frames, 10)
    elbow_angles = seq[:, POSE_FEATURES + 96:]                        # (frames, 2) — already 0.0-1.0, no normalization needed

    left_shoulder = pose_part[:, 0, :2]
    right_shoulder = pose_part[:, 1, :2]
    mid_shoulder = (left_shoulder + right_shoulder) / 2.0

    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=1, keepdims=True)
    shoulder_width[shoulder_width < 1e-6] = 1e-6

    pose_xy = pose_part[:, :, :2]
    pose_xy_norm = (pose_xy - mid_shoulder[:, None, :]) / shoulder_width[:, None, :]
    pose_visibility = pose_part[:, :, 2:3]
    pose_normalized = np.concatenate([pose_xy_norm, pose_visibility], axis=2)

    left_hand_norm = (left_hand_part - mid_shoulder[:, None, :]) / shoulder_width[:, None, :]
    right_hand_norm = (right_hand_part - mid_shoulder[:, None, :]) / shoulder_width[:, None, :]

    normalized = np.concatenate([
        pose_normalized.reshape(seq.shape[0], -1),
        left_hand_norm.reshape(seq.shape[0], -1),
        right_hand_norm.reshape(seq.shape[0], -1),
        hand_flags,        # untouched
        finger_features,   # untouched
        elbow_angles,      # untouched
    ], axis=1)

    return normalized


def ablate(normalized_seq):
    """
    NEW FUNCTION.
    Drops the 84 raw hand-coordinate columns [24:108] from an already-
    normalized sequence, keeping only pose (0:24) + flags/fingers/elbow
    (108:122) -- 38 columns total. No-op if ABLATE_HAND_COORDS is False.
    Only ever called on the NORMALIZED tensor that feeds the model --
    never on the raw array used by the tie-breaker.
    """
    if not ABLATE_HAND_COORDS:
        return normalized_seq
    keep_cols = list(range(0, POSE_FEATURES)) + list(range(POSE_FEATURES + HAND_COORD_FEATURES, TOTAL_FEATURES))
    return normalized_seq[:, keep_cols]


def resample_sequence(seq, target_len):
    """Linearly interpolates a (frames, features) sequence to a fixed length."""
    orig_len = seq.shape[0]
    if orig_len == target_len:
        return seq
    if orig_len < 2:
        # Degenerate clip (basically no frames) — pad by repeating.
        return np.repeat(seq, target_len, axis=0)[:target_len]

    orig_idx = np.linspace(0, 1, orig_len)
    target_idx = np.linspace(0, 1, target_len)

    resampled = np.zeros((target_len, seq.shape[1]))
    for feature_i in range(seq.shape[1]):
        resampled[:, feature_i] = np.interp(target_idx, orig_idx, seq[:, feature_i])
    return resampled


def augment_sequence(seq):
    """
    Applies small random perturbations to a normalized keypoint sequence
    to make the model more tolerant of real-world variation: different
    distances/scales, slightly different body proportions, gesture speed
    differences, and minor camera-angle differences. Only applied to
    TRAINING clips — never to val/test, which must stay unmodified to
    give an honest accuracy reading.
    """
    seq = seq.copy()

    # 1. Random spatial scaling (simulates different camera distances)
    scale_factor = np.random.uniform(0.9, 1.1)
    seq[:, :] *= scale_factor

    # 2. Small random spatial jitter/noise (simulates body proportion
    #    differences and minor detection noise)
    noise = np.random.normal(0, 0.01, seq.shape)
    seq = seq + noise

    # 3. Random time-warping (simulates different gesture speeds/rhythms)
    #    Stretches or compresses the sequence slightly before it gets
    #    resampled back to SEQUENCE_LENGTH.
    warp_factor = np.random.uniform(0.85, 1.15)
    orig_len = seq.shape[0]
    warped_len = max(2, int(orig_len * warp_factor))
    orig_idx = np.linspace(0, 1, orig_len)
    warped_idx = np.linspace(0, 1, warped_len)
    warped = np.zeros((warped_len, seq.shape[1]))
    for feature_i in range(seq.shape[1]):
        warped[:, feature_i] = np.interp(warped_idx, orig_idx, seq[:, feature_i])

    return warped


class GestureDataset(Dataset):
    def __init__(self, rows, label_to_idx, augment=False):
        self.rows = rows
        self.label_to_idx = label_to_idx
        self.augment = augment  # only True for the training set

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        raw = np.load(row["keypoint_path"])
        normalized = normalize_sequence(raw)
        normalized = ablate(normalized)   # NEW LINE — drops hand coords if ABLATE_HAND_COORDS is True

        if self.augment:
            normalized = augment_sequence(normalized)

        resampled = resample_sequence(normalized, SEQUENCE_LENGTH)

        x = torch.tensor(resampled, dtype=torch.float32)
        y = self.label_to_idx[row["gesture_label"]]
        return x, y


def is_peace_sign(finger_block_avg):
    """
    finger_block_avg: length-5 array, [thumb, index, middle, ring, pinky],
    each value is the AVERAGE extension (0.0-1.0) across all frames in the clip.
    Returns True if this looks like a peace sign: index + middle extended,
    ring + pinky curled (thumb is ambiguous/unreliable, not used as a deciding factor).
    """
    index_extended = finger_block_avg[INDEX_FINGER_OFFSET] > PEACE_SIGN_THRESHOLD
    middle_extended = finger_block_avg[MIDDLE_FINGER_OFFSET] > PEACE_SIGN_THRESHOLD
    ring_curled = finger_block_avg[RING_FINGER_OFFSET] <= PEACE_SIGN_THRESHOLD
    pinky_curled = finger_block_avg[PINKY_FINGER_OFFSET] <= PEACE_SIGN_THRESHOLD
    return index_extended and middle_extended and ring_curled and pinky_curled


def apply_tie_breaker(raw_sequence, top1_label, top2_label, top1_prob, top2_prob, idx_to_label):
    """
    raw_sequence: the ORIGINAL (pre-normalization, pre-ablation) keypoint
    sequence for one clip, shape (frames, 122) — needed because finger
    features are easiest to read directly here rather than re-deriving
    from the normalized/ablated tensor. This function is UNCHANGED by
    the ablation experiment: it always reads from the full raw array,
    regardless of what the model itself was trained on.

    Only overrides the model's prediction when:
      1. The top-2 predicted classes are both in CONFUSABLE_CLASSES, AND
      2. Their probabilities are close enough that the model is genuinely
         unsure (not just slightly preferring one).
    Otherwise, returns the model's original top-1 prediction unchanged —
    this tie-breaker is a safety net, not a replacement for the model.
    """
    if top1_label not in CONFUSABLE_CLASSES or top2_label not in CONFUSABLE_CLASSES:
        return top1_label  # not a relevant confusion, leave the model's answer alone

    if (top1_prob - top2_prob) > TIE_BREAKER_PROB_MARGIN:
        return top1_label  # model isn't actually torn, no need to intervene

    # Average finger extension across all frames, separately for each hand
    left_fingers_avg = raw_sequence[:, LEFT_FINGERS_START:LEFT_FINGERS_START + 5].mean(axis=0)
    right_fingers_avg = raw_sequence[:, RIGHT_FINGERS_START:RIGHT_FINGERS_START + 5].mean(axis=0)

    peace_sign_detected = is_peace_sign(left_fingers_avg) or is_peace_sign(right_fingers_avg)

    candidates = {top1_label, top2_label}
    if peace_sign_detected and "double_contact" in candidates:
        return "double_contact"
    else:
        # Pick whichever candidate isn't double_contact
        non_double_contact = [c for c in candidates if c != "double_contact"]
        return non_double_contact[0] if non_double_contact else top1_label




def train():
    rows = load_manifest_rows()

    all_labels = sorted(set(r["gesture_label"] for r in rows))
    label_to_idx = {label: i for i, label in enumerate(all_labels)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}

    with open(LABEL_MAP_SAVE_PATH, "w") as f:
        json.dump(label_to_idx, f, indent=2)

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]

    print(f"Train clips: {len(train_rows)} | Val clips: {len(val_rows)} | Test clips: {len(test_rows)}")
    print(f"Classes: {all_labels}")
    if ABLATE_HAND_COORDS:
        print(f"ABLATION ACTIVE: dropping {HAND_COORD_FEATURES} raw hand-coordinate features. "
              f"Model input size: {ABLATED_FEATURE_COUNT} (instead of {TOTAL_FEATURES}).")

    train_ds = GestureDataset(train_rows, label_to_idx, augment=True)
    val_ds = GestureDataset(val_rows, label_to_idx, augment=False)
    test_ds = GestureDataset(test_rows, label_to_idx, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    input_size = ABLATED_FEATURE_COUNT if ABLATE_HAND_COORDS else TOTAL_FEATURES   # CHANGED
    num_classes = len(all_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureCNNLSTM(input_size=input_size, num_classes=num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Class weighting: rarer classes get a higher weight, so misclassifying
    # them costs more during training. Without this, classes with far more
    # clips (e.g. double_contact) can dominate what the model learns to
    # favor, just because it saw more examples of them.
    class_counts = np.array([sum(1 for r in train_rows if r["gesture_label"] == label) for label in all_labels])
    class_weights = 1.0 / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.sum() * len(all_labels)  # normalize so weights average to ~1
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print("\nClass weights (higher = rarer, weighted more heavily):")
    for label, weight, count in zip(all_labels, class_weights, class_counts):
        print(f"  {label:<30} weight={weight:.3f}  (train clips: {count})")

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            total_train_loss += loss.item() * x_batch.size(0)
        avg_train_loss = total_train_loss / max(1, len(train_ds))

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                total_val_loss += loss.item() * x_batch.size(0)
        avg_val_loss = total_val_loss / max(1, len(val_ds))

        print(f"Epoch {epoch:3d} | train_loss: {avg_train_loss:.4f} | val_loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs).")
                break

    # ---------------------------------------------------------------
    # Final evaluation on held-out test set, using the best checkpoint
    # ---------------------------------------------------------------
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    all_preds, all_true = [], []
    tie_breaker_overrides = 0

    with torch.no_grad():
        for row in test_rows:
            raw = np.load(row["keypoint_path"])
            normalized = normalize_sequence(raw)
            normalized = ablate(normalized)   # NEW LINE — must match what the model was trained on
            resampled = resample_sequence(normalized, SEQUENCE_LENGTH)
            x = torch.tensor(resampled, dtype=torch.float32).unsqueeze(0).to(device)

            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            top2_idx = np.argsort(probs)[-2:][::-1]  # descending: [top1_idx, top2_idx]
            top1_idx, top2_idx_ = top2_idx[0], top2_idx[1]
            top1_label = idx_to_label[top1_idx]
            top2_label = idx_to_label[top2_idx_]
            top1_prob, top2_prob = probs[top1_idx], probs[top2_idx_]

            # Use the RAW (pre-normalized, pre-ablation, pre-resampled)
            # sequence for the tie-breaker so finger-extension values are
            # easy to read directly. UNCHANGED — always the full 122-column
            # raw array regardless of ABLATE_HAND_COORDS.
            raw_resampled_for_tiebreak = resample_sequence(raw, SEQUENCE_LENGTH)
            final_label = apply_tie_breaker(
                raw_resampled_for_tiebreak, top1_label, top2_label, top1_prob, top2_prob, idx_to_label
            )
            if final_label != top1_label:
                tie_breaker_overrides += 1

            all_preds.append(label_to_idx[final_label])
            all_true.append(label_to_idx[row["gesture_label"]])

    true_labels = [idx_to_label[i] for i in all_true]
    pred_labels = [idx_to_label[i] for i in all_preds]

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(all_true, all_preds):.4f}")
    print(f"Tie-breaker overrides applied: {tie_breaker_overrides} / {len(test_rows)} test clips")
    print("\nPer-class report:")
    print(classification_report(true_labels, pred_labels, zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(f"Labels order: {all_labels}")
    print(confusion_matrix(true_labels, pred_labels, labels=all_labels))

    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print(f"Label map saved to: {LABEL_MAP_SAVE_PATH}")


if __name__ == "__main__":
    train()