"""
Step 6: Match-based train/test split + SVM training with GroupKFold
cross-validation for hyperparameter tuning.

IMPORTANT: edit TEST_MATCHES below to pick which 2-3 matches you're holding
out. These should never be touched again until step 7.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).parent.parent

# --- EDIT THIS: pick 2-3 match_ids to hold out entirely as your test set ---
TEST_MATCHES = ["match9", "match7", "match13"]
# iPhone recordings are their own "match_id" groups (iphone_<filename>) --
# decide whether to hold out one or more of those too, or keep them all in
# training since they're a supplementary domain. Default here: keep all
# iPhone groups in training.


def main():
    df = pd.read_csv(ROOT / "processed" / "features.csv")
    mfcc_cols = [c for c in df.columns if c.startswith("mfcc_")]

    # Guard against typos / stale match names silently vanishing from the
    # test set (e.g. TEST_MATCHES referencing a match that isn't in the data)
    available_matches = set(df["match_id"].unique())
    missing = [m for m in TEST_MATCHES if m not in available_matches]
    if missing:
        raise ValueError(
            f"TEST_MATCHES contains match_id(s) not present in features.csv: {missing}\n"
            f"Available match_ids are: {sorted(available_matches)}\n"
            f"Update TEST_MATCHES at the top of this script to match your actual data."
        )

    train_df = df[~df["match_id"].isin(TEST_MATCHES)].reset_index(drop=True)
    test_df = df[df["match_id"].isin(TEST_MATCHES)].reset_index(drop=True)

    print(f"Train: {len(train_df)} clips from {train_df['match_id'].nunique()} groups")
    print(f"Test:  {len(test_df)} clips from {test_df['match_id'].nunique()} groups")
    print("\nTrain label balance:")
    print(train_df["label"].value_counts())

    X_train = train_df[mfcc_cols].values
    y_train = train_df["label"].values
    groups = train_df["match_id"].values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced")),
    ])

    param_grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", 0.01, 0.001],
    }

    n_groups = train_df["match_id"].nunique()
    n_splits = min(5, n_groups)  # can't have more folds than groups
    group_kfold = GroupKFold(n_splits=n_splits)

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=group_kfold.split(X_train, y_train, groups=groups),
        scoring="f1",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    print(f"\nBest params: {grid.best_params_}")
    print(f"Best CV F1 score: {grid.best_score_:.4f}")

    best_model = grid.best_estimator_

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(best_model, models_dir / "whistle_svm_model.pkl")

    # save the test split too so step 7 uses exactly this data
    test_df.to_csv(ROOT / "processed" / "test_set.csv", index=False)

    print(f"\nModel saved -> {models_dir / 'whistle_svm_model.pkl'}")
    print("Test set (untouched until step 7) saved -> processed/test_set.csv")


if __name__ == "__main__":
    main()
