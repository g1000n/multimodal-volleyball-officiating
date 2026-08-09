import csv
import numpy as np

MANIFEST_PATH = "data/dataset_manifest.csv"

def main():
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))

    rows = [r for r in rows if r.get("keypoint_path") and r["person_id"] != "pmax"]
    sample = rows[::5][:200]

    all_frames = []
    for r in sample:
        arr = np.load(r["keypoint_path"])
        all_frames.append(arr)
    combined = np.concatenate(all_frames, axis=0)

    print("Feature index : std across whole dataset")
    print(f"  pose[0:24] mean std:        {combined[:, 0:24].std(axis=0).mean():.4f}")
    print(f"  hand_flags[108:110] std:    {combined[:, 108:110].std(axis=0).mean():.4f}")
    print(f"  fingers[110:120] std:       {combined[:, 110:120].std(axis=0).mean():.4f}")
    print(f"  elbow_angles[120:122] std:  {combined[:, 120:122].std(axis=0).mean():.4f}")

if __name__ == "__main__":
    main()
