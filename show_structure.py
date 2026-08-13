from pathlib import Path

ROOT = Path(__file__).parent

def show_tree(path, prefix=""):
    items = sorted(
        [p for p in path.iterdir() if p.name != ".venv"],
        key=lambda p: (not p.is_dir(), p.name.lower())
    )

    for i, item in enumerate(items):
        last = i == len(items) - 1
        branch = "└── " if last else "├── "

        print(prefix + branch + item.name)

        # Don't show anything inside data
        if item.is_dir() and item.name.lower() != "data":
            show_tree(item, prefix + ("    " if last else "│   "))


print(ROOT.name)
show_tree(ROOT)