"""
train.py (MULTI-LABEL ARCHITECTURE — matches MaxLSB/volley-judge's approach)

Loads keypoint sequences according to the manifest's train/val/test
split (produced by dataset_split.py), normalizes and resamples them
to a fixed length, trains the CNN-LSTM model, and evaluates on the
held-out test set.

--------------------------------------------------------------------
ARCHITECTURE CHANGE (this version): switched from single-label
CrossEntropyLoss/softmax (8-way mutually-exclusive classification,
"nothing" as its own competing output neuron) to multi-label
BCEWithLogitsLoss/sigmoid (7-way INDEPENDENT per-class detection,
"nothing" represented as an all-zero label vector -- not a neuron at
all). This directly mirrors MaxLSB/volley-judge's design.

WHY: under softmax, every class (including "nothing") competes for a
shared probability budget that sums to 1 -- so even during genuine
ambiguity (idle standing, mid-gesture transitions), the model is
FORCED to push relative confidence toward SOME class, and an
underrepresented "nothing" class's inflated weight could win that
competition and steal confidence from real gestures. Under
independent sigmoid outputs, each of the 7 REAL gesture classes gets
its own yes/no confidence, with no shared budget -- so during genuine
non-gesture content, ALL 7 outputs can genuinely be low simultaneously,
and "nothing" naturally falls out as "no class was confident enough,"
rather than needing to be predicted by a dedicated neuron competing
against everything else.

Run order (full pipeline, UNCHANGED):
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
import random
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

# --- REPRODUCIBILITY FIX: train.py never seeded torch/numpy/random,
# which caused meaningfully different results (accuracy, confusion
# patterns) across retrains on IDENTICAL data. Seeding here makes
# retrains on unchanged data reproducible. ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

ABLATE_HAND_COORDS = True
ABLATED_FEATURE_COUNT = POSE_FEATURES + HAND_FLAG_FEATURES + FINGER_FEATURES + ELBOW_ANGLE_FEATURES  # 38

# "nothing" is a real label in the manifest/data (used for filtering
# training clips), but it is NOT one of the model's output neurons --
# it's represented as an all-zero prediction across the 7 real classes.
NOTHING_LABEL = "nothing"

# Column ranges for the tie-breaker logic (unchanged from before)
LEFT_FINGERS_START = POSE_FEATURES + HAND_COORD_FEATURES + HAND_FLAG_FEATURES        # 110
RIGHT_FINGERS_START = LEFT_FINGERS_START + 5                                        # 115
INDEX_FINGER_OFFSET = 1
MIDDLE_FINGER_OFFSET = 2
RING_FINGER_OFFSET = 3
PINKY_FINGER_OFFSET = 4

CONFUSABLE_CLASSES = {"double_contact", "service_authorization_left", "service_authorization_right"}
TIE_BREAKER_PROB_MARGIN = 0.20
PEACE_SIGN_THRESHOLD = 0.5

# NEW: elbow-angle offsets, for the team_to_serve vs. service_authorization
# safety check below. Layout: [...110 fingers...][120: left_elbow][121: right_elbow]
LEFT_ELBOW_ANGLE_IDX = POSE_FEATURES + HAND_COORD_FEATURES + HAND_FLAG_FEATURES + FINGER_FEATURES        # 120
RIGHT_ELBOW_ANGLE_IDX = LEFT_ELBOW_ANGLE_IDX + 1                                                          # 121

# NEW SAFETY CHECK: team_to_serve_X (a SCORING gesture) vs.
# service_authorization_X (a non-scoring gesture) on the SAME side. A
# slow/deliberate service_authorization performance risks being read as
# team_to_serve, which would incorrectly award a real point -- a genuine
# scoring-integrity risk, not just a display glitch, since
# service_authorization legitimately happens right after a whistle in
# real play too (whistle-gating does NOT protect against this specific
# confusion). team_to_serve is a straight, extended-arm point;
# service_authorization involves bending the elbow -- elbow_angle
# (already a real feature, 0=fully bent, 1=fully straight) is the
# natural signal to disambiguate them.
SAME_SIDE_SCORE_AUTH_PAIRS = {
    frozenset({"team_to_serve_left", "service_authorization_left"}): "left",
    frozenset({"team_to_serve_right", "service_authorization_right"}): "right",
}
SCORE_AUTH_TIE_BREAKER_PROB_MARGIN = 0.25  # slightly wider than the double_contact one -- err toward caution given this affects real scoring
STRAIGHT_ARM_ELBOW_THRESHOLD = 0.5  # elbow_angle above this = leaning "straight arm" (team_to_serve); below = "bent" (service_authorization). STARTING VALUE -- tune with real data if it misfires.

# DECISION_THRESHOLD: a class's sigmoid output must clear this to be
# considered a confident detection at all. If NO class clears it,
# the prediction is "nothing" -- this is the multi-label equivalent of
# MaxLSB's backend.py: `if proba > threshold: predictions.append(...)`.
DECISION_THRESHOLD = 0.5

EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 5e-4
GRAD_CLIP_NORM = 1.0
MODEL_SAVE_PATH = "models/final_model.pt"
LABEL_MAP_SAVE_PATH = "models/label_map.json"


# ---------------------------------------------------------------------
# Data loading and preprocessing (UNCHANGED from before)
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
    """UNCHANGED. See original docstring -- pose/hand coordinates
    normalized relative to shoulder midpoint/width; detection flags
    and finger-extension values left untouched."""
    pose_part = seq[:, :POSE_FEATURES].reshape(seq.shape[0], 8, 3)
    left_hand_part = seq[:, POSE_FEATURES:POSE_FEATURES + 42].reshape(seq.shape[0], 21, 2)
    right_hand_part = seq[:, POSE_FEATURES + 42:POSE_FEATURES + 84].reshape(seq.shape[0], 21, 2)
    hand_flags = seq[:, POSE_FEATURES + 84:POSE_FEATURES + 86]
    finger_features = seq[:, POSE_FEATURES + 86:POSE_FEATURES + 96]
    elbow_angles = seq[:, POSE_FEATURES + 96:]

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
        hand_flags,
        finger_features,
        elbow_angles,
    ], axis=1)

    return normalized


def ablate(normalized_seq):
    """UNCHANGED. Drops the 84 raw hand-coordinate columns if
    ABLATE_HAND_COORDS is True."""
    if not ABLATE_HAND_COORDS:
        return normalized_seq
    keep_cols = list(range(0, POSE_FEATURES)) + list(range(POSE_FEATURES + HAND_COORD_FEATURES, TOTAL_FEATURES))
    return normalized_seq[:, keep_cols]


def resample_sequence(seq, target_len):
    """UNCHANGED."""
    orig_len = seq.shape[0]
    if orig_len == target_len:
        return seq
    if orig_len < 2:
        return np.repeat(seq, target_len, axis=0)[:target_len]

    orig_idx = np.linspace(0, 1, orig_len)
    target_idx = np.linspace(0, 1, target_len)

    resampled = np.zeros((target_len, seq.shape[1]))
    for feature_i in range(seq.shape[1]):
        resampled[:, feature_i] = np.interp(target_idx, orig_idx, seq[:, feature_i])
    return resampled


def augment_sequence(seq):
    """UNCHANGED."""
    seq = seq.copy()
    scale_factor = np.random.uniform(0.9, 1.1)
    seq[:, :] *= scale_factor

    noise = np.random.normal(0, 0.01, seq.shape)
    seq = seq + noise

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
    """
    CHANGED: y is now a multi-hot FLOAT vector of length
    len(real_labels) (7), not a single class index. A "nothing" clip
    gets an all-zero vector; a real gesture clip gets a single 1 at
    its own index -- everything else 0.
    """
    def __init__(self, rows, real_label_to_idx, augment=False):
        self.rows = rows
        self.real_label_to_idx = real_label_to_idx
        self.num_real_classes = len(real_label_to_idx)
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        raw = np.load(row["keypoint_path"])
        normalized = normalize_sequence(raw)
        normalized = ablate(normalized)

        if self.augment:
            normalized = augment_sequence(normalized)

        resampled = resample_sequence(normalized, SEQUENCE_LENGTH)

        x = torch.tensor(resampled, dtype=torch.float32)

        y = np.zeros(self.num_real_classes, dtype=np.float32)
        gesture_label = row["gesture_label"]
        if gesture_label != NOTHING_LABEL:
            y[self.real_label_to_idx[gesture_label]] = 1.0
        y = torch.tensor(y, dtype=torch.float32)

        return x, y


def is_peace_sign(finger_block_avg):
    """UNCHANGED."""
    index_extended = finger_block_avg[INDEX_FINGER_OFFSET] > PEACE_SIGN_THRESHOLD
    middle_extended = finger_block_avg[MIDDLE_FINGER_OFFSET] > PEACE_SIGN_THRESHOLD
    ring_curled = finger_block_avg[RING_FINGER_OFFSET] <= PEACE_SIGN_THRESHOLD
    pinky_curled = finger_block_avg[PINKY_FINGER_OFFSET] <= PEACE_SIGN_THRESHOLD
    return index_extended and middle_extended and ring_curled and pinky_curled


def apply_tie_breaker(raw_sequence, top1_label, top2_label, top1_prob, top2_prob):
    """
    Two independent tie-breaker checks, tried in order:

    1. NEW SAFETY CHECK: team_to_serve_X vs. service_authorization_X
       (same side) -- uses elbow angle (straight arm = team_to_serve,
       bent elbow = service_authorization) to prevent a slow/deliberate
       service_authorization from being misread as a SCORING gesture.
       This is checked FIRST and with its own (wider) margin, since a
       wrong call here means an incorrectly awarded point, not just a
       display mixup.

    2. UNCHANGED: double_contact vs. service_authorization_left/right,
       using peace-sign finger detection. Only fires when BOTH top1 and
       top2 are in CONFUSABLE_CLASSES and close in score.

    Neither ever fires when the predicted label is "nothing" -- nothing
    to break a tie against.
    """
    label_pair = frozenset({top1_label, top2_label})
    if label_pair in SAME_SIDE_SCORE_AUTH_PAIRS and (top1_prob - top2_prob) <= SCORE_AUTH_TIE_BREAKER_PROB_MARGIN:
        side = SAME_SIDE_SCORE_AUTH_PAIRS[label_pair]
        elbow_idx = LEFT_ELBOW_ANGLE_IDX if side == "left" else RIGHT_ELBOW_ANGLE_IDX
        avg_elbow_angle = raw_sequence[:, elbow_idx].mean()

        scoring_label = f"team_to_serve_{side}"
        auth_label = f"service_authorization_{side}"

        if avg_elbow_angle >= STRAIGHT_ARM_ELBOW_THRESHOLD:
            return scoring_label
        else:
            return auth_label

    if top1_label not in CONFUSABLE_CLASSES or top2_label not in CONFUSABLE_CLASSES:
        return top1_label

    if (top1_prob - top2_prob) > TIE_BREAKER_PROB_MARGIN:
        return top1_label

    left_fingers_avg = raw_sequence[:, LEFT_FINGERS_START:LEFT_FINGERS_START + 5].mean(axis=0)
    right_fingers_avg = raw_sequence[:, RIGHT_FINGERS_START:RIGHT_FINGERS_START + 5].mean(axis=0)

    peace_sign_detected = is_peace_sign(left_fingers_avg) or is_peace_sign(right_fingers_avg)

    candidates = {top1_label, top2_label}
    if peace_sign_detected and "double_contact" in candidates:
        return "double_contact"
    else:
        non_double_contact = [c for c in candidates if c != "double_contact"]
        return non_double_contact[0] if non_double_contact else top1_label


def decide_label(sigmoid_probs, idx_to_real_label):
    """
    NEW: the core multi-label decision rule, mirroring MaxLSB's
    backend.py (`predicted_label = argmax; if proba > threshold: ...`).

    Returns (predicted_label, top1_label, top2_label, top1_prob, top2_prob)
    -- predicted_label is "nothing" if no class cleared DECISION_THRESHOLD,
    otherwise it's the tie-breaker-adjusted top1 real class.
    """
    sorted_idx = np.argsort(sigmoid_probs)[::-1]
    top1_idx, top2_idx = sorted_idx[0], sorted_idx[1]
    top1_label = idx_to_real_label[top1_idx]
    top2_label = idx_to_real_label[top2_idx]
    top1_prob = sigmoid_probs[top1_idx]
    top2_prob = sigmoid_probs[top2_idx]

    if top1_prob <= DECISION_THRESHOLD:
        # No class was confident enough -- this IS how "nothing" gets
        # predicted, same as MaxLSB's `if proba > threshold` gate.
        return NOTHING_LABEL, top1_label, top2_label, top1_prob, top2_prob

    return top1_label, top1_label, top2_label, top1_prob, top2_prob


def train():
    rows = load_manifest_rows()

    all_labels = sorted(set(r["gesture_label"] for r in rows))          # includes "nothing", for reporting only
    real_labels = sorted(l for l in all_labels if l != NOTHING_LABEL)    # 7 real gestures -- these are the model's outputs
    real_label_to_idx = {label: i for i, label in enumerate(real_labels)}
    idx_to_real_label = {i: label for label, i in real_label_to_idx.items()}

    # Save the FULL label set (including "nothing") for downstream
    # scripts that need to know all possible labels for reporting --
    # but also save which ones are real model outputs, since that's
    # what determines the model's actual output layer size now.
    with open(LABEL_MAP_SAVE_PATH, "w") as f:
        json.dump({"real_label_to_idx": real_label_to_idx, "nothing_label": NOTHING_LABEL}, f, indent=2)

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]

    print(f"Train clips: {len(train_rows)} | Val clips: {len(val_rows)} | Test clips: {len(test_rows)}")
    print(f"Real gesture classes (model outputs): {real_labels}")
    print(f"'{NOTHING_LABEL}' clips are used in training but represented as an all-zero label, not a model output.")
    if ABLATE_HAND_COORDS:
        print(f"ABLATION ACTIVE: dropping {HAND_COORD_FEATURES} raw hand-coordinate features. "
              f"Model input size: {ABLATED_FEATURE_COUNT} (instead of {TOTAL_FEATURES}).")

    train_ds = GestureDataset(train_rows, real_label_to_idx, augment=True)
    val_ds = GestureDataset(val_rows, real_label_to_idx, augment=False)
    test_ds = GestureDataset(test_rows, real_label_to_idx, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    input_size = ABLATED_FEATURE_COUNT if ABLATE_HAND_COORDS else TOTAL_FEATURES
    num_real_classes = len(real_labels)   # 7, NOT 8 -- "nothing" is not an output neuron

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureCNNLSTM(input_size=input_size, num_classes=num_real_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # CHANGED: pos_weight for BCEWithLogitsLoss, one value PER REAL
    # CLASS ONLY -- there is no "nothing" weight to worry about
    # anymore, since nothing isn't a competing output neuron. This
    # structurally eliminates the whole "nothing's weight got
    # cranked up and stole confidence from real classes" failure mode.
    class_counts = np.array([sum(1 for r in train_rows if r["gesture_label"] == label) for label in real_labels])
    pos_weight = torch.tensor(
        np.sqrt(len(train_rows) / np.maximum(class_counts, 1)), dtype=torch.float32
    ).to(device)

    print("\nPer-class pos_weight (higher = rarer positive examples, weighted more in loss):")
    for label, weight, count in zip(real_labels, pos_weight.cpu().numpy(), class_counts):
        print(f"  {label:<30} pos_weight={weight:.3f}  (train clips: {count})")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
            normalized = ablate(normalized)
            resampled = resample_sequence(normalized, SEQUENCE_LENGTH)
            x = torch.tensor(resampled, dtype=torch.float32).unsqueeze(0).to(device)

            logits = model(x)
            # CHANGED: sigmoid instead of softmax -- each class's score
            # is INDEPENDENT, doesn't have to sum to 1 with the others.
            probs = torch.sigmoid(logits).cpu().numpy()[0]

            predicted_label, top1_label, top2_label, top1_prob, top2_prob = decide_label(probs, idx_to_real_label)

            raw_resampled_for_tiebreak = resample_sequence(raw, SEQUENCE_LENGTH)
            final_label = apply_tie_breaker(raw_resampled_for_tiebreak, top1_label, top2_label, top1_prob, top2_prob)
            # Only apply the tie-breaker's override if we actually predicted
            # a real class -- if predicted_label came back "nothing", there's
            # no tie to break.
            if predicted_label != NOTHING_LABEL:
                if final_label != top1_label:
                    tie_breaker_overrides += 1
                predicted_label = final_label

            all_preds.append(predicted_label)
            all_true.append(row["gesture_label"])

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(all_true, all_preds):.4f}")
    print(f"Tie-breaker overrides applied: {tie_breaker_overrides} / {len(test_rows)} test clips")
    print("\nPer-class report:")
    print(classification_report(all_true, all_preds, labels=all_labels, zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(f"Labels order: {all_labels}")
    print(confusion_matrix(all_true, all_preds, labels=all_labels))

    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print(f"Label map saved to: {LABEL_MAP_SAVE_PATH}")
    print(f"Decision threshold used: {DECISION_THRESHOLD} (a class's sigmoid score must clear this to count)")


if __name__ == "__main__":
    train()