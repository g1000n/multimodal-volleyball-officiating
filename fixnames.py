from pathlib import Path
import re

# Folder to process
FOLDER = Path(r"C:\Users\notgi\Downloads\wtv html\data\raw_clips\nothing")

GESTURE = "nothing"
PERSON = "p101"

# Matches files already named like:
# ball_out_p08_1.mp4
# ball_out_p08_25.mov
pattern = re.compile(
    rf"^{re.escape(GESTURE)}_{PERSON}_(\d+)(\.[^.]+)$",
    re.IGNORECASE,
)

# Find the highest existing number
highest = 0

for file in FOLDER.iterdir():
    if not file.is_file():
        continue

    match = pattern.match(file.name)
    if match:
        number = int(match.group(1))
        highest = max(highest, number)

print(f"Highest existing {PERSON} number: {highest}")

# Rename only files that are NOT already correctly named
next_number = highest + 1

for file in sorted(FOLDER.iterdir()):
    if not file.is_file():
        continue

    # Skip files already in the correct format
    if pattern.match(file.name):
        continue

    new_name = f"{GESTURE}_{PERSON}_{next_number}{file.suffix}"

    print(f"{file.name} -> {new_name}")
    file.rename(FOLDER / new_name)

    next_number += 1

print("\nDone!")