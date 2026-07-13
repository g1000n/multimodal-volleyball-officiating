from pathlib import Path

TEMP_BIN = Path(r"C:\Users\notgi\Downloads\temp bin")

LEFT_FOLDER = TEMP_BIN / "amare service left"
RIGHT_FOLDER = TEMP_BIN / "amare service right"


def rename_prefix(folder, old_prefix, new_prefix):
    count = 0

    for file in folder.iterdir():
        if not file.is_file():
            continue

        if file.name.startswith(old_prefix):
            new_name = file.name.replace(old_prefix, new_prefix, 1)
            file.rename(folder / new_name)
            count += 1

    print(f"{folder.name}: Renamed {count} files.")


# Files currently in the RIGHT folder are incorrectly named "...left..."
rename_prefix(
    RIGHT_FOLDER,
    "authorization_to_serve_left",
    "authorization_to_serve_right",
)

# Files currently in the LEFT folder are incorrectly named "...right..."
rename_prefix(
    LEFT_FOLDER,
    "authorization_to_serve_right",
    "authorization_to_serve_left",
)

print("\nDone!")