from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier

ROOT = Path(__file__).parent.parent
TEST_MATCHES = ["match9", "match7", "match13"]
# CO-PRIMARY DATA NOTE: once iPhone whistle/negative clips are extracted,
# add at least one iphone_<filename> group here too, e.g.:
#   TEST_MATCHES = ["match9", "match7", "match13", "iphone_recording2"]

def main():
    feat_csv = ROOT / "processed" / "features.csv"
    if not feat_csv.exists():
        print("Features file not found!")
        return

    df = pd.read_csv(feat_csv)
    X_cols = [c for c in df.columns if c.startswith("mfcc_")]

    train_df = df[~df["match_id"].isin(TEST_MATCHES)].copy()
    test_df = df[df["match_id"].isin(TEST_MATCHES)].copy()

    X_train = train_df[X_cols].values
    y_train = train_df["label"].values
    groups = train_df["match_id"].values

    test_df.to_csv(ROOT / "processed" / "test_set.csv", index=False)
    gkf = GroupKFold(n_splits=len(np.unique(groups)))

    # 1. Optimize Base SVM Core
    pipe_svm = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', probability=True, random_state=42))])
    param_svm = {'svm__C': [1, 10, 100], 'svm__gamma': ['scale', 0.01, 0.1]}
    print("Hyper-tuning SVM parameters...")
    grid_svm = GridSearchCV(pipe_svm, param_svm, cv=gkf, scoring='f1', n_jobs=-1).fit(X_train, y_train, groups=groups)
    
    # 2. Optimize Base Random Forest Core
    pipe_rf = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(random_state=42))])
    param_rf = {'rf__n_estimators': [100, 200], 'rf__max_depth': [15, None]}
    print("Hyper-tuning Random Forest structures...")
    grid_rf = GridSearchCV(pipe_rf, param_rf, cv=gkf, scoring='f1', n_jobs=-1).fit(X_train, y_train, groups=groups)

    # 3. Optimize Base Gradient Boosting Core (New Layer)
    pipe_gb = Pipeline([('scaler', StandardScaler()), ('gb', HistGradientBoostingClassifier(random_state=42))])
    param_gb = {'gb__max_iter': [100, 150], 'gb__learning_rate': [0.05, 0.1]}
    print("Hyper-tuning HistGradientBoosting paths...")
    grid_gb = GridSearchCV(pipe_gb, param_gb, cv=gkf, scoring='f1', n_jobs=-1).fit(X_train, y_train, groups=groups)

    print(f"\nIndividual Validation Scores -> SVM: {grid_svm.best_score_:.4f} | RF: {grid_rf.best_score_:.4f} | GB: {grid_gb.best_score_:.4f}")

    # Build a diverse 3-Way soft voting engine
    print("\nFusing into a Triple-Ensemble Meta-Classifier...")
    ensemble = VotingClassifier(
        estimators=[
            ('svm', grid_svm.best_estimator_),
            ('rf', grid_rf.best_estimator_),
            ('gb', grid_gb.best_estimator_)
        ],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(ensemble, models_dir / "whistle_svm_model.pkl")
    print(f"Triple-Ensemble core successfully saved -> {models_dir / 'whistle_svm_model.pkl'}")

if __name__ == "__main__":
    main()