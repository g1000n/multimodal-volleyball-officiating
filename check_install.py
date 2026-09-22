"""
check_install.py

Checks that the training tool is installed correctly BEFORE you run it. Put this file in the project root
(next to trainer.py) and run:

    python check_install.py

It tells you, in plain words:
  * which files are missing or in the wrong folder (and where it found them),
  * which Python packages are missing,
  * whether your existing live_deployment.py and whistle_detector.py have the functions the trainer relies on
    (checked by reading the source, nothing is started, no camera or microphone is used).

If anything is reported, fix it and run this again. Paste the output to whoever is helping you.
"""

import ast
import importlib.util
import os
import sys

ROOT_FILES = {
    "trainer.py": "the training tool (run this)",
    "trainer_ui.py": "welcome, consent and menu screens",
    "gesture_grader.py": "the FIVB based grading rules",
    "devices.py": "camera and microphone helpers",
    "progress.py": "the My progress summary",
    "device_setup.py": "camera and microphone check screen",
    "trainer_config.py": "your settings (email, camera, theme)",
    "live_deployment.py": "YOUR existing recognition code (used unchanged)",
    "whistle_detector.py": "YOUR existing whistle detector (used unchanged)",
}
OPTIONAL_FILES = {
    os.path.join("tools", "evaluate_sessions.py"): "evaluation table from labeled test sessions",
    os.path.join("tools", "regrade_attempts.py"): "re-grade saved attempts after changing a threshold",
    os.path.join("tools", "export_rubric.py"): "export the grading rubric as a table",
    os.path.join("tools", "grader_sanity_check.py"): "check the rules against your dataset clips",
    os.path.join("tests", "test_gesture_grader.py"): "grader tests",
    os.path.join("tests", "test_trainer_smoke.py"): "trainer tests",
}
PACKAGES = [("numpy", True), ("cv2", True), ("tkinter", True), ("mediapipe", True), ("torch", True),
            ("sounddevice", False), ("psutil", False)]

problems = []
notes = []


def find_elsewhere(name):
    hits = []
    for base, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in (".venv", "venv", "__pycache__", ".git", "data", "node_modules")]
        if os.path.basename(name) in files and os.path.normpath(os.path.join(base, os.path.basename(name))) != os.path.normpath(name):
            hits.append(os.path.normpath(os.path.join(base, os.path.basename(name))))
    return hits


def check_files():
    print("1. Files in the project root")
    for name, what in ROOT_FILES.items():
        if os.path.exists(name):
            print(f"   ok       {name}")
        else:
            elsewhere = find_elsewhere(name)
            if elsewhere:
                problems.append(f"{name} is in the wrong folder ({elsewhere[0]}). Move it next to trainer.py.")
                print(f"   WRONG    {name}  found at {elsewhere[0]}, move it to the project root")
            else:
                problems.append(f"{name} is missing ({what}). Download it and put it next to trainer.py.")
                print(f"   MISSING  {name}  ({what})")
    for name, what in OPTIONAL_FILES.items():
        print(f"   {'ok      ' if os.path.exists(name) else 'optional'} {name}" + ("" if os.path.exists(name) else f"  ({what})"))


def check_packages():
    print("\n2. Python packages (this Python: " + sys.executable + ")")
    for pkg, required in PACKAGES:
        found = importlib.util.find_spec(pkg) is not None
        print(f"   {'ok      ' if found else ('MISSING ' if required else 'optional')} {pkg}")
        if not found and required:
            hint = "it comes with Python on Windows; reinstall Python with tcl/tk" if pkg == "tkinter" else \
                   f"pip install {'opencv-python' if pkg == 'cv2' else pkg}"
            problems.append(f"package '{pkg}' is missing ({hint}).")
        if not found and not required:
            if pkg == "psutil":
                notes.append("'psutil' is not installed: CPU and memory are not measured for the performance figures "
                             "(pip install psutil). Speed (frames per second) is still measured.")
            else:
                notes.append(f"'{pkg}' is not installed: the microphone test and the whistle will not work "
                             f"(pip install {pkg}); press W as the whistle instead.")


def parse(path):
    try:
        with open(path, encoding="utf-8") as f:
            return ast.parse(f.read())
    except (OSError, SyntaxError) as exc:
        problems.append(f"{path} could not be read ({exc}).")
        return None


def check_live_deployment():
    print("\n3. live_deployment.py (your existing code, read only)")
    tree = parse("live_deployment.py") if os.path.exists("live_deployment.py") else None
    if tree is None:
        return
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    consts = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    want = {"load_model": 0, "classify_window": 3, "extract_frame_features": 3}
    for name, min_args in want.items():
        fn = funcs.get(name)
        if fn is None:
            problems.append(f"live_deployment.py has no function {name}() which the trainer calls.")
            print(f"   MISSING  {name}()")
            continue
        n_args = len(fn.args.args)
        ok = n_args >= min_args
        print(f"   {'ok      ' if ok else 'CHECK   '} {name}({', '.join(a.arg for a in fn.args.args)})")
        if not ok:
            problems.append(f"{name}() takes {n_args} arguments, the trainer passes {min_args}.")
    if "load_model" in funcs:
        lens = [len(r.value.elts) for r in ast.walk(funcs["load_model"])
                if isinstance(r, ast.Return) and isinstance(r.value, ast.Tuple)]
        ok = 3 in lens
        print(f"   {'ok      ' if ok else 'CHECK   '} load_model() returns {lens or 'something else'} values (the trainer expects 3: "
              f"model, idx_to_real_label, labels)")
        if not ok:
            problems.append("load_model() should return (model, idx_to_real_label, real_labels).")
    print(f"   {'ok      ' if 'CAMERA_INDEX' in consts else 'CHECK   '} CAMERA_INDEX defined")
    if "CAMERA_INDEX" not in consts:
        notes.append("live_deployment.CAMERA_INDEX not found: set CAMERA_INDEX in trainer_config.py instead.")


def check_whistle():
    print("\n4. whistle_detector.py (your existing code, read only)")
    tree = parse("whistle_detector.py") if os.path.exists("whistle_detector.py") else None
    if tree is None:
        return
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WhistleDetector"), None)
    if cls is None:
        problems.append("whistle_detector.py has no class WhistleDetector.")
        print("   MISSING  class WhistleDetector")
        return
    init = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
    args = [a.arg for a in init.args.args] + [a.arg for a in init.args.kwonlyargs] if init else []
    methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    for arg in ("on_whistle_callback", "device"):
        print(f"   {'ok      ' if arg in args else 'CHECK   '} WhistleDetector(..., {arg}=...)")
        if arg not in args:
            notes.append(f"WhistleDetector.__init__ has no '{arg}' argument: the whistle in the trainer will not start "
                         f"(the W key still works). Send the class to whoever is helping you.")
    for m in ("start", "stop"):
        print(f"   {'ok      ' if m in methods else 'CHECK   '} WhistleDetector.{m}()")
        if m not in methods:
            notes.append(f"WhistleDetector has no {m}() method: the whistle in the trainer may not work (use W).")


def check_models():
    print("\n5. Model files")
    print(f"   {'ok      ' if os.path.isdir('models') else 'MISSING '} models/")
    lm = os.path.join("models", "label_map.json")
    print(f"   {'ok      ' if os.path.exists(lm) else 'CHECK   '} {lm}")
    if not os.path.isdir("models"):
        problems.append("the models/ folder is missing: run the trainer from the project root.")


def main():
    print(f"Checking: {os.path.abspath('.')}\n")
    if not os.path.exists("trainer.py"):
        print("trainer.py is not in this folder. Open a terminal IN the project folder (where live_deployment.py is) "
              "and run this again.\n")
    check_files()
    check_packages()
    check_live_deployment()
    check_whistle()
    check_models()
    print("\n" + "=" * 60)
    if not problems:
        print("Everything needed was found. Start with:  python trainer.py")
    else:
        print("FIX THESE FIRST:")
        for p in problems:
            print("  - " + p)
    if notes:
        print("\nGood to know:")
        for n in notes:
            print("  - " + n)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())