from pathlib import Path
import uuid

ROOT = Path(r"C:\Users\notgi\Downloads\drive-download-20260803T105039Z-1-001")

PERSON = "p08"
PREFIX = "ball_in"

FOLDERS = [
    "ball_in_right_dwayne_",
    "ball_in_right_version2_dwayne",
    "ball_in_left_dwayne",
    "ball_in_left_ver2_dwayne",
]

next_number = 1

for folder_name in FOLDERS:

    folder = ROOT / folder_name

    if not folder.exists():
        print(f"Skipping {folder_name}")
        continue

    files = sorted(f for f in folder.iterdir() if f.is_file())

    # PASS 1: rename to temporary names
    temp_files = []

    for file in files:
        temp = folder / f"__tmp__{uuid.uuid4().hex}{file.suffix.lower()}"
        file.rename(temp)
        temp_files.append(temp)

    # PASS 2: rename to final names
    start = next_number

    for temp in temp_files:
        final = folder / f"{PREFIX}_{PERSON}_{next_number}{temp.suffix.lower()}"
        temp.rename(final)
        next_number += 1

    print(f"{folder_name}: {start} -> {next_number - 1}")

print(f"\nTotal renamed: {next_number - 1}")