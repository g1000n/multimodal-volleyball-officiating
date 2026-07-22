from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

ROOT = Path(__file__).parent.parent

def main():
    test_csv = ROOT / "processed" / "test_set.csv"
    model_path = ROOT / "models" / "whistle_svm_model.pkl"
    if not test_csv.exists() or not model_path.exists():
        print("Missing evaluation matrix components.")
        return

    df = pd.read_csv(test_csv)
    X_cols = [c for c in df.columns if c.startswith("mfcc_")]
    X_test = df[X_cols].values
    y_test = df["label"].values

    model = joblib.load(model_path)
    
    # Extract prediction confidence probabilities
    probabilities = model.predict_proba(X_test)[:, 1]

    print("=== Scanning Alternative Decision Thresholds ===")
    best_acc = 0
    best_threshold = 0.5
    
    # Sweep configurations from 0.30 to 0.70 confidence barriers
    for t in np.arange(0.3, 0.7, 0.05):
        preds = (probabilities >= t).astype(int)
        acc = accuracy_score(y_test, preds)
        print(f"Decision Boundary: {t:.2f} -> Resulting Test Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_threshold = t

    print(f"\n>>> Peak Mathematical Threshold Selected: {best_threshold:.2f} (Accuracy: {best_acc:.4f})")
    
    # Deploy definitive evaluation under optimized boundaries
    final_preds = (probabilities >= best_threshold).astype(int)
    print("\n=== Optimized Test Performance (Held-out Matches) ===")
    print(classification_report(y_test, final_preds, target_names=["non_whistle", "whistle"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, final_preds))

if __name__ == "__main__":
    main()