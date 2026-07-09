from pathlib import Path
import re

folder = Path(r"C:\Users\notgi\Downloads\holfer")

person_code = "p03"   # Charlene
gesture_name = "service_to_authorization_left"

def get_number(file):
    numbers = re.findall(r"\d+", file.stem)
    return int(numbers[-1]) if numbers else 0

files = sorted(
    folder.glob("*.mp4"),
    key=get_number
)

for i, file in enumerate(files, start=1):
    new_name = f"{gesture_name}_{person_code}_{i}{file.suffix}"
    file.rename(folder / new_name)

print(f"Renamed {len(files)} files.")