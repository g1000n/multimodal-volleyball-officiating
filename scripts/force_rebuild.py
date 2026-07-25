"""
force_rebuild.py
Force-deletes feature caches, extracts 92 physical features, and retrains the model.

FIX: was pointing at "data/processed" (the old, disconnected pipeline's cache
folder). The real pipeline writes to "processed/" -- fixed below so cleanup
actually clears the files that 05/06 regenerate.
"""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent if Path(__file__).parent.name == "scripts" else Path(__file__).parent

# 1. Clean out processed caches and old models
print("🧹 Cleaning old feature caches and model files...")
cache_file = ROOT / "processed" / "features.csv"      # FIX: was data/processed dir
test_file = ROOT / "processed" / "test_set.csv"
model_file = ROOT / "models" / "whistle_svm_model.pkl"

for f in (cache_file, test_file, model_file):
    if f.exists():
        os.remove(f)
        print(f"   Deleted: {f}")

# 2. Run feature extraction
print("\n🎵 Extracting 92 features from dataset...")
extract_script = ROOT / "scripts" / "05_extract_features.py"
os.system(f'python "{extract_script}"')

# 3. Retrain model
print("\n🧠 Training SVM model...")
train_script = ROOT / "scripts" / "06_train_model.py"
os.system(f'python "{train_script}"')

# 4. Verify trained model feature count
print("\n🔍 Verifying saved model feature expectations...")
import joblib
if model_file.exists():
    model = joblib.load(model_file)

    n_features = None
    if hasattr(model, "n_features_in_"):
        n_features = model.n_features_in_
    elif hasattr(model, "estimators_"):
        n_features = model.estimators_[0].named_steps['scaler'].n_features_in_

    print(f"✅ Success! Saved model is now expecting EXACTLY {n_features} features.")
    if n_features != 92:
        print("⚠️ Warning: Feature count mismatch! Check that 05_extract_features.py returns 92 features.")
else:
    print("❌ Error: Model file was not created. Check 06_train_model.py output.")