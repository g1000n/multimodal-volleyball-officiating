"""
main.py

Default behavior: `python main.py` with no arguments just runs the real
system -- live_deployment.py -- directly. That's the actual "main" use
case, so it shouldn't require picking an option first.

Everything else (replay, full training pipeline, diagnostics/tools/
tests) is available as an explicit subcommand, or via the interactive
menu (`python main.py menu`) if you'd rather browse than remember a
subcommand name.

Wraps the existing scripts as subprocess calls -- doesn't reimplement
any of their logic. `python train.py`, `python live_deployment.py`,
etc. still work exactly as before; this is just a friendlier front
door, especially useful for anyone new to the repo.

USAGE:
    python main.py                       # runs live_deployment.py directly
    python main.py replay <path>         # runs replay_recorded_footage.py
    python main.py train                 # runs the full training pipeline
    python main.py diagnostics           # pick a script from diagnostics/
    python main.py tools                 # pick a script from tools/
    python main.py tests                 # pick a script from tests/
    python main.py menu                  # interactive menu instead
"""

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def run(cmd_parts, cwd=REPO_ROOT):
    print(f"\n$ {' '.join(cmd_parts)}\n")
    subprocess.run(cmd_parts, cwd=cwd)


def pick_from_folder(folder, prompt):
    """Lists .py files in a folder, lets the user pick one to run."""
    folder_path = os.path.join(REPO_ROOT, folder)
    if not os.path.isdir(folder_path):
        print(f"  (no '{folder}/' folder found -- has the repo reorg run yet?)")
        return

    scripts = sorted(f for f in os.listdir(folder_path) if f.endswith(".py"))
    if not scripts:
        print(f"  (no scripts found in {folder}/)")
        return

    print(f"\n{prompt}")
    for i, name in enumerate(scripts, start=1):
        print(f"  {i}. {name}")
    choice = input(f"Pick a script (1-{len(scripts)}, or Enter to cancel): ").strip()
    if not choice:
        return
    try:
        selected = scripts[int(choice) - 1]
    except (ValueError, IndexError):
        print("  Invalid choice.")
        return
    run([sys.executable, os.path.join(folder, selected)])


def run_full_pipeline():
    print("\nFull training pipeline: build_manifest -> convert_maxlsb -> extract_keypoints "
          "-> dataset_split -> train")
    confirm = input("This can take a while (extraction + training). Continue? (y/n): ").strip().lower()
    if confirm != "y":
        return
    steps = [
        "build_manifest.py",
        "convert_maxlsb_nothing_data.py",
        "extract_keypoints.py",
        "dataset_split.py",
        "train.py",
    ]
    for step in steps:
        if not os.path.exists(os.path.join(REPO_ROOT, step)):
            print(f"  (skip) {step} not found at repo root -- has it moved?")
            continue
        run([sys.executable, step])
        cont = input(f"\n'{step}' finished. Continue to the next step? (y/n): ").strip().lower()
        if cont != "y":
            print("Stopped pipeline early.")
            return
    print("\nFull pipeline complete.")


def run_replay():
    recordings_dir = os.path.join(REPO_ROOT, "data", "raw_recordings")
    if os.path.isdir(recordings_dir):
        clips = sorted(f for f in os.listdir(recordings_dir) if f.endswith(".mp4"))
        if clips:
            print("\nAvailable recordings:")
            for i, name in enumerate(clips, start=1):
                print(f"  {i}. {name}")
            choice = input(f"Pick a recording (1-{len(clips)}), or paste a full path, or Enter to cancel: ").strip()
            if not choice:
                return
            try:
                idx = int(choice) - 1
                video_path = os.path.join("data", "raw_recordings", clips[idx])
            except (ValueError, IndexError):
                video_path = choice  # treat as a raw path if not a valid index
        else:
            video_path = input("No recordings found in data/raw_recordings/. Paste a full path, or Enter to cancel: ").strip()
            if not video_path:
                return
    else:
        video_path = input("data/raw_recordings/ not found. Paste a full path to a recording, or Enter to cancel: ").strip()
        if not video_path:
            return

    run([sys.executable, "replay_recorded_footage.py", video_path])


MENU = """
==================================================
  Volleyball Officiating System -- Main Menu
==================================================
  1. Live deployment (needs a camera connected)
  2. Replay a recorded session
  3. Run the full training pipeline
  4. Run a diagnostic script (diagnostics/)
  5. Run a tool/utility script (tools/)
  6. Run a test (tests/)
  0. Exit
==================================================
"""


def run_menu():
    while True:
        print(MENU)
        choice = input("Choose an option: ").strip()

        if choice == "1":
            run([sys.executable, "live_deployment.py"])
        elif choice == "2":
            run_replay()
        elif choice == "3":
            run_full_pipeline()
        elif choice == "4":
            pick_from_folder("diagnostics", "Diagnostic scripts:")
        elif choice == "5":
            pick_from_folder("tools", "Tool/utility scripts:")
        elif choice == "6":
            pick_from_folder("tests", "Test scripts:")
        elif choice == "0":
            print("Bye!")
            break
        else:
            print("Invalid choice, try again.")


def main():
    args = sys.argv[1:]

    # DEFAULT: no arguments -- just run the real system directly.
    # This is the actual "main" use case, shouldn't need a menu first.
    if not args:
        run([sys.executable, "live_deployment.py"])
        return

    command = args[0]

    if command == "replay":
        if len(args) >= 2:
            run([sys.executable, "replay_recorded_footage.py", args[1]])
        else:
            run_replay()  # no path given -- prompt/pick interactively
    elif command == "train":
        run_full_pipeline()
    elif command == "diagnostics":
        pick_from_folder("diagnostics", "Diagnostic scripts:")
    elif command == "tools":
        pick_from_folder("tools", "Tool/utility scripts:")
    elif command == "tests":
        pick_from_folder("tests", "Test scripts:")
    elif command == "menu":
        run_menu()
    else:
        print(f"Unrecognized command: '{command}'")
        print("Usage: python main.py [replay <path> | train | diagnostics | tools | tests | menu]")
        print("(no arguments = run live_deployment.py directly)")


if __name__ == "__main__":
    main()