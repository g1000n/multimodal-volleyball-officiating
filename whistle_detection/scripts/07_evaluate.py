"""
Step 7: Final evaluation on the held-out test matches. Run this ONCE, after
you're done tuning in step 6 -- don't go back and re-tune based on these
numbers, or the test set stops being a true held-out estimate.

Also breaks down performance by negative source (match audio vs. iPhone),
which is worth reporting honestly in your thesis as a discussion point.
"""
from pathlib import Path

import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

ROOT = Path(__file__).parent.parent


def main():
    model = joblib.load(ROOT / "models" / "whistle_svm_model.pkl")
    test_df = pd.read_csv(ROOT / "processed" / "test_set.csv")
    mfcc_cols = [c for c in test_df.columns if c.startswith("mfcc_")]

    X_test = test_df[mfcc_cols].values
    y_test = test_df["label"].values

    y_pred = model.predict(X_test)

    print("=== Overall test performance (held-out matches) ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["non_whistle", "whistle"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Breakdown by source -- only meaningful if your test matches include
    # iPhone-derived negatives too; if not, this will just show one source.
    print("\n=== Breakdown by negative source ===")
    for source in test_df["source"].unique():
        mask = test_df["source"] == source
        if mask.sum() == 0:
            continue
        sub_X = test_df.loc[mask, mfcc_cols].values
        sub_y = test_df.loc[mask, "label"].values
        sub_pred = model.predict(sub_X)
        acc = accuracy_score(sub_y, sub_pred)
        print(f"{source}: n={mask.sum()}, accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
