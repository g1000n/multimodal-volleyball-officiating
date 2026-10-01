import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = Path(__file__).parent

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "data",
    "raw_data",
    "appendix_p_grids",
    "appendix_r_invalid_candidates",
    "appendix_r_screenshots",
    "appendix_sample_screenshots",
    "model_checkpoints",
    "training_logs",
}

EXCLUDED_FILES = {
    "appendix_r_valid_example.jpg",
    "test_audio_recording.wav",
}

def show_tree(path, prefix=""):
    items = sorted(
        [
            p for p in path.iterdir()
            if p.name not in EXCLUDED_DIRS
            and p.name not in EXCLUDED_FILES
        ],
        key=lambda p: (not p.is_dir(), p.name.lower())
    )

    for i, item in enumerate(items):
        last = i == len(items) - 1
        branch = "└── " if last else "├── "

        print(prefix + branch + item.name)

        if item.is_dir():
            show_tree(
                item,
                prefix + ("    " if last else "│   ")
            )

print(ROOT.name)
show_tree(ROOT)