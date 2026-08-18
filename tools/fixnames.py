from pathlib import Path
import uuid

ROOT = Path(
    r"C:\Users\notgi\Downloads\drive-download-20260817T170623Z-1-001"
)

PERSON = "p02"
PREFIX = "ball_in"

FOLDERS = [
    "ball_in_amare_left",
    "ball_in_amare_right",
]

next_number = 1

for folder_name in FOLDERS:

    folder = ROOT / folder_name

    if not folder.exists():
        print(f"Skipping: {folder_name}")
        continue

    files = sorted(
        f for f in folder.iterdir()
        if f.is_file()
    )

    print(f"\n{folder_name}")
    print(f"Files: {len(files)}")
    print(f"Starting number: {next_number}")

    # PASS 1: temporary names
    temp_files = []

    for file in files:
        temp_name = f"__tmp__{uuid.uuid4().hex}{file.suffix.lower()}"
        temp_path = folder / temp_name

        file.rename(temp_path)
        temp_files.append(temp_path)

    # PASS 2: final names
    start = next_number

    for temp_file in temp_files:

        final_name = (
            f"{PREFIX}_{PERSON}_{next_number}"
            f"{temp_file.suffix.lower()}"
        )

        temp_file.rename(folder / final_name)

        next_number += 1

    print(f"Renamed: {start} -> {next_number - 1}")

print("\n" + "=" * 50)
print(f"TOTAL BALL_IN P02 FILES: {next_number - 1}")
print("=" * 50)
print("Done.")