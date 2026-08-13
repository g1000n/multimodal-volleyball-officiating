from pathlib import Path
import uuid

FOLDER = Path(
    r"C:\Users\notgi\Downloads\drive-download-20260810T173103Z-1-001\steady_liam_"
)

PERSON = "p09"
PREFIX = "nothing"

files = sorted(
    f for f in FOLDER.iterdir()
    if f.is_file()
)

print(f"Found {len(files)} files.")

# PASS 1: temporary names
temp_files = []

for file in files:
    temp_name = f"__tmp__{uuid.uuid4().hex}{file.suffix.lower()}"
    temp_path = FOLDER / temp_name

    file.rename(temp_path)
    temp_files.append(temp_path)

# PASS 2: final names
for i, temp_file in enumerate(temp_files, start=1):

    new_name = f"{PREFIX}_{PERSON}_{i}{temp_file.suffix.lower()}"

    temp_file.rename(FOLDER / new_name)

print(f"Renamed {len(files)} files.")

if files:
    print(f"First: {PREFIX}_{PERSON}_1{files[0].suffix.lower()}")
    print(f"Last:  {PREFIX}_{PERSON}_{len(files)}{files[-1].suffix.lower()}")