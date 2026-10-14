"""
reproduce_table5_eval.py

STANDALONE -- run this from your project root (same folder as train.py, model.py).

Reproduces Table 5 (and the confusion matrix / Table 6) by RELOADING the already-trained
model checkpoint (models/final_model.pt) and re-running just the test-set evaluation loop --
the exact same loop at the end of train.py's train() function, reusing its own functions
(normalize_sequence, ablate, resample_sequence, decide_label, apply_tie_breaker) rather than
re-implementing them, so nothing can drift out of sync with what actually produced Table 5.

WHY NOT JUST RE-RUN train.py: full training involves randomness (data loader shuffling,
augmentation, weight init) that can shift the final numbers slightly run to run, even with a
fixed seed if anything upstream isn't perfectly deterministic. This script never retrains --
it loads the SAME saved model weights and evaluates them on the SAME test split, so the
printed numbers are guaranteed to match Table 5 exactly (not a fresh, possibly-different run).

This is what to screenshot for the tracker's "include an actual screenshot of the generated
result directly from Python" item (Table 5) -- run it, screenshot the terminal output.

Run:
    python reproduce_table5_eval.py
"""

import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from model import GestureCNNLSTM
from train import (
    ABLATE_HAND_COORDS,
    ABLATED_FEATURE_COUNT,
    NOTHING_LABEL,
    TOTAL_FEATURES,
    ablate,
    apply_tie_breaker,
    decide_label,
    load_manifest_rows,
    normalize_sequence,
    resample_sequence,
    SEQUENCE_LENGTH,
)

MODEL_PATH = "models/final_model.pt"
LABEL_MAP_PATH = "models/label_map.json"


def main():
    with open(LABEL_MAP_PATH) as f:
        label_map_data = json.load(f)
    real_label_to_idx = label_map_data["real_label_to_idx"]
    idx_to_real_label = {int(v): k for k, v in real_label_to_idx.items()}

    input_size = ABLATED_FEATURE_COUNT if ABLATE_HAND_COORDS else TOTAL_FEATURES
    model = GestureCNNLSTM(input_size=input_size, num_classes=len(real_label_to_idx))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    print(f"Loaded {MODEL_PATH} (no retraining -- reusing the saved weights that produced Table 5)\n")

    rows = load_manifest_rows()
    test_rows = [r for r in rows if r["split"] == "test"]
    all_labels = sorted(set(r["gesture_label"] for r in rows))   # includes "nothing", for reporting only

    all_preds, all_true = [], []
    tie_breaker_overrides = 0

    with torch.no_grad():
        for row in test_rows:
            raw = np.load(row["keypoint_path"])
            normalized = normalize_sequence(raw)
            normalized = ablate(normalized)
            resampled = resample_sequence(normalized, SEQUENCE_LENGTH)
            x = torch.tensor(resampled, dtype=torch.float32).unsqueeze(0)

            logits = model(x)
            probs = torch.sigmoid(logits).numpy()[0]

            predicted_label, top1_label, top2_label, top1_prob, top2_prob = decide_label(probs, idx_to_real_label)

            raw_resampled_for_tiebreak = resample_sequence(raw, SEQUENCE_LENGTH)
            final_label = apply_tie_breaker(raw_resampled_for_tiebreak, top1_label, top2_label, top1_prob, top2_prob)
            if predicted_label != NOTHING_LABEL:
                if final_label != top1_label:
                    tie_breaker_overrides += 1
                predicted_label = final_label

            all_preds.append(predicted_label)
            all_true.append(row["gesture_label"])

    print("=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(all_true, all_preds):.4f}")
    print(f"Tie-breaker overrides applied: {tie_breaker_overrides} / {len(test_rows)} test clips")
    print("\nPer-class report:")
    print(classification_report(all_true, all_preds, labels=all_labels, zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(f"Labels order: {all_labels}")
    cm = confusion_matrix(all_true, all_preds, labels=all_labels)
    print(cm)


if __name__ == "__main__":
    main()
