from pathlib import Path

folder = Path(r"C:\Users\notgi\Downloads\gestures\end of set")

# Get all Timeline files
files = sorted(
    folder.glob("Timeline*"),
    key=lambda f: int(''.join(filter(str.isdigit, f.stem)))
)

# Rename sequentially
for i, file in enumerate(files, start=1):
    new_name = f"end_of_set_{i}{file.suffix}"
    file.rename(folder / new_name)

print(f"Renamed {len(files)} files.")