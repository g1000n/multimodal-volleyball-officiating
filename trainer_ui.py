"""
trainer_ui.py

Tkinter screens that run BEFORE / BETWEEN the OpenCV training sessions:

    1. Welcome + data-privacy / recording consent   (run_welcome)
    2. Main menu: pick a mode, signal and difficulty (run_menu)
    3. Learn window: FIVB signal reference sheets    (open from the menu)

Tkinter ships with Python on Windows, so there is nothing extra to install.

LIGHT / DARK MODE
-----------------
Every screen has a theme button in the top-right corner. The choice is remembered in
data/ui_settings.json and is shared with the training screen (press T there to switch too).
Buttons are custom flat buttons, so a disabled button still looks like a button (and says what is
missing) instead of just greying out its text.

CONSENT (panel requirement: an agreement that sessions are filmed and saved)
---------------------------------------------------------------------------
The notice is framed under the Philippine Data Privacy Act of 2012 (RA 10173), applied together with
the GDPR principles the panelist asked for (purpose limitation, data minimisation, storage
limitation, data-subject rights). The tool cannot be used without accepting. Acceptance is logged
(trainee id, timestamp, consent-text version) to data/consent_records.csv and to
data/trainer_sessions/<trainee_id>/consent.json.

Edit CONTACT and RETENTION in trainer_config.py (not here). This is a good-faith notice, not legal
advice; have your adviser read it.

"Delete my data" (menu) removes the trainee's whole session folder and logs a `data_deleted` row in
the consent log (the log keeps only id + timestamp).
"""

import csv
import datetime
import json
import os
import re
import shutil
import tkinter as tk
from tkinter import messagebox, ttk

import gesture_grader as gg

try:
    import trainer_config as _cfg
except ImportError:  # keep working if the config file is missing
    _cfg = None


def _conf(name, default):
    return getattr(_cfg, name, default)


APP_TITLE = "Volleyball Officiating Training Tool"
CONSENT_VERSION = "2026-09-v2"
DATA_ROOT = os.path.join("data", "trainer_sessions")
CONSENT_LOG = os.path.join("data", "consent_records.csv")
SETTINGS_PATH = os.path.join("data", "ui_settings.json")
SIGNAL_IMAGE_DIR = os.path.join("assets", "signals")   # optional: assets/signals/<label>.png

# Evaluation-test labels (Session label + Note in the menu). Off for real use; switch on in trainer_config.py
# while collecting the "correct on purpose" / "wrong on purpose" attempts.
TEST_MODE = bool(_conf("TEST_MODE", False))

CONTACT = _conf("CONTACT", "[TEAM CONTACT EMAIL]")
RETENTION = _conf("RETENTION", "[RETENTION PERIOD]")

PH_TZ = datetime.timezone(datetime.timedelta(hours=8))

CONSENT_TEXT = f"""DATA PRIVACY AND RECORDING CONSENT

This notice follows the Philippine Data Privacy Act of 2012 (Republic Act No. 10173), applied together with the principles of the EU GDPR: purpose limitation, data minimisation, storage limitation, and your rights over your own data.

WHAT THIS TOOL DOES
It uses your webcam and microphone to check the referee hand signals you perform and to detect whistle blasts, and it gives you feedback and a training score.

WHAT WILL BE COLLECTED
- Video of your training sessions, recorded from your webcam. You will appear in it.
- Your session results: which signals you attempted, your scores, timestamps, and a session report.
- Movement data: the body and hand keypoint positions (a stick-figure of your movement, not a picture) that the tool extracts from your video for each graded attempt, so your attempts can be graded and re-checked.
- The name or nickname you typed on this screen.
- Microphone audio is analysed live to detect whistles. The audio itself is not recorded; only whistle-detection events and timestamps are logged.

WHY
To give you training feedback and to evaluate this tool for our thesis (research purposes only).

WHERE IT IS STORED AND FOR HOW LONG
On this computer only (folder: data/trainer_sessions/). Access is limited to the research team and the thesis adviser. Retention: {RETENTION}.

YOUR RIGHTS
Taking part is voluntary. You may ask to see, correct, or delete your data at any time, and you may withdraw your consent at any time by choosing "Delete my data" in the menu or by contacting {CONTACT}. The tool cannot run without recording your sessions, so if you decline you will not be able to use it. If you are under 18, a parent or guardian must give consent on your behalf.

By ticking the box below you confirm that you have read this notice and that you agree your training sessions will be filmed and saved as described above.
"""

MODES = [
    ("practice", "Practice",
     "Do any signal. The system tells you which one it sees and shows a live checklist. No score."),
    ("drill", "Drill",
     "Choose ONE signal and repeat it. With several repetitions they run back to back and you see every score "
     "at the end. Choose 1 to practise freely, with retries."),
    ("combo", "Combo drill",
     "The end of a rally, like a real match: a short story, then Team to Serve and the reason performed one right "
     "after the other. Runs automatically; review every score at the end."),
    ("challenge", "Challenge",
     "Every signal once, in random order, one after the other with no stopping. Your score is the number of "
     "signals you perform correctly."),
    ("sim", "Match simulation",
     "A short narrated set like a real game: for each call you blow the whistle and give the signal right away. "
     "A message at the bottom shows what was committed. Optional continuous mode runs the whole set by itself."),
]

COMBO_CHOICES = [
    ("random", "Random rally-end calls"),
    ("ball_out", "Team to Serve, then Ball Out"),
    ("ball_in", "Team to Serve, then Ball In"),
    ("double_contact", "Team to Serve, then Double Contact"),
]

REP_CHOICES = ["1 (free practice, retries)", "3", "5", "10"]

# Labels for evaluation tests: lets the team log deliberately correct / wrong attempts as evidence.
INTENT_CHOICES = [
    ("normal", "Normal training"),
    ("correct", "TEST: I will perform correctly"),
    ("wrong", "TEST: I will perform WRONG on purpose"),
]

LEVEL_INFO = {   # built from the real pass marks, so the text can never disagree with the grader
    "beginner": f"Forgiving: wide tolerances, shorter hold. {gg.LEVEL_CONFIG['beginner'].correct_cut} or more counts as correct.",
    "standard": f"Balanced: a clear, recognisable signal. {gg.LEVEL_CONFIG['standard'].correct_cut} or more counts as correct (100 is not needed).",
    "referee": f"Strict: close to textbook form, longer hold. {gg.LEVEL_CONFIG['referee'].correct_cut} or more counts as correct.",
}



# ----------------------------------------------------------------------------
# Trainee data helpers (also imported by trainer.py)
# ----------------------------------------------------------------------------

def sanitize_id(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "").strip()).strip("_")
    return cleaned[:40]


def trainee_dir(trainee_id: str) -> str:
    return os.path.join(DATA_ROOT, trainee_id)


def _now_ph() -> str:
    return datetime.datetime.now(PH_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _append_consent_log(trainee_id: str, event: str):
    os.makedirs(os.path.dirname(CONSENT_LOG), exist_ok=True)
    new_file = not os.path.exists(CONSENT_LOG)
    with open(CONSENT_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["ph_time", "trainee_id", "event", "consent_version"])
        w.writerow([_now_ph(), trainee_id, event, CONSENT_VERSION])


def record_consent(trainee_id: str, display_name: str) -> dict:
    os.makedirs(trainee_dir(trainee_id), exist_ok=True)
    record = {
        "trainee_id": trainee_id,
        "display_name": display_name,
        "consent_given": True,
        "consent_version": CONSENT_VERSION,
        "ph_time": _now_ph(),
    }
    with open(os.path.join(trainee_dir(trainee_id), "consent.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    _append_consent_log(trainee_id, "consent_given")
    return record


def trainee_data_summary(trainee_id: str):
    """(number of saved sessions, size in MB) for this trainee."""
    path = trainee_dir(trainee_id)
    sessions = 0
    size = 0
    if os.path.isdir(path):
        sessions = sum(1 for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))
        for base, _dirs, files in os.walk(path):
            for f in files:
                try:
                    size += os.path.getsize(os.path.join(base, f))
                except OSError:
                    pass
    return sessions, size / (1024 * 1024)


def open_path(path: str) -> bool:
    """Open a file or folder with the computer's default program (report in the browser, video in the player)."""
    try:
        if hasattr(os, "startfile"):
            os.startfile(path)                                   # Windows
        else:
            import subprocess
            import sys
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])
        return True
    except Exception:
        return False


def delete_session(trainee_id: str, session: str) -> bool:
    """Delete ONE session folder of a trainee (the rest of their data stays). Logged in the consent log."""
    session = os.path.basename(session)                          # never a path
    path = os.path.join(trainee_dir(trainee_id), session)
    existed = os.path.isdir(path)
    if existed:
        shutil.rmtree(path, ignore_errors=True)
        _append_consent_log(trainee_id, f"session_deleted:{session}")
    return existed


def delete_trainee_data(trainee_id: str) -> bool:
    path = trainee_dir(trainee_id)
    existed = os.path.isdir(path)
    if existed:
        shutil.rmtree(path, ignore_errors=True)
    _append_consent_log(trainee_id, "data_deleted")
    return existed



# ----------------------------------------------------------------------------
# Themes
# ----------------------------------------------------------------------------

PALETTES = {
    "dark": dict(bg="#14181d", card="#1e242c", card_hi="#28313b", fg="#f2f4f7", muted="#9aa4b0",
                 green="#6edc78", amber="#fab13c", blue="#5ab2eb", border="#3d4854",
                 primary_bg="#6edc78", primary_hover="#8ef096", primary_fg="#0b1a0d",
                 off_bg="#232a33", off_fg="#8b96a3",
                 sec_bg="#28313b", sec_hover="#36414d", sec_fg="#f2f4f7"),
    "light": dict(bg="#f3f5f8", card="#ffffff", card_hi="#e4e9ef", fg="#1b2430", muted="#5b6675",
                  green="#178a48", amber="#b56a00", blue="#1d6fb8", border="#c2cbd6",
                  primary_bg="#178a48", primary_hover="#127a3e", primary_fg="#ffffff",
                  off_bg="#e3e7ec", off_fg="#6f7a88",
                  sec_bg="#e4e9ef", sec_hover="#d3dbe4", sec_fg="#1b2430"),
}


def read_settings() -> dict:
    """data/ui_settings.json: theme, chosen camera and microphone."""
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_settings(patch: dict):
    data = read_settings()
    data.update(patch)
    try:
        os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def _load_saved_theme():
    name = read_settings().get("theme")
    if name in PALETTES:
        return name
    default = _conf("THEME", "dark")
    return default if default in PALETTES else "dark"


_theme = _load_saved_theme()


def current_theme() -> str:
    return _theme


def palette() -> dict:
    return PALETTES[_theme]


def set_theme(name: str, save: bool = True):
    global _theme
    if name not in PALETTES:
        return
    _theme = name
    if save:
        write_settings({"theme": name})


def _alive(widget) -> bool:
    try:
        return bool(widget.winfo_exists())
    except tk.TclError:
        return False


class Skin:
    """Registry of widgets and the palette keys they use, so the whole window can be re-themed live."""

    def __init__(self, root):
        self.root = root
        self.items = []        # (widget, {option: palette_key})
        self.buttons = []
        self.callbacks = []
        self.style = ttk.Style(root)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

    def add(self, widget, **opts):
        self.items.append((widget, opts))
        self._apply_one(widget, opts)
        return widget

    def _apply_one(self, widget, opts):
        p = palette()
        try:
            widget.configure(**{o: p[k] for o, k in opts.items()})
        except tk.TclError:
            pass

    def apply(self):
        p = palette()
        try:
            self.root.configure(bg=p["bg"])
        except tk.TclError:
            return
        self.style.configure("TCombobox", fieldbackground=p["card_hi"], background=p["card_hi"],
                             foreground=p["fg"], arrowcolor=p["fg"], bordercolor=p["border"],
                             lightcolor=p["card_hi"], darkcolor=p["card_hi"], selectbackground=p["card_hi"],
                             selectforeground=p["fg"])
        self.style.map("TCombobox", fieldbackground=[("readonly", p["card_hi"]), ("disabled", p["off_bg"])],
                       foreground=[("readonly", p["fg"]), ("disabled", p["off_fg"])])
        self.style.configure("Treeview", background=p["card"], fieldbackground=p["card"], foreground=p["fg"],
                             rowheight=26, borderwidth=0, font=("Segoe UI", 10))
        self.style.configure("Treeview.Heading", background=p["card_hi"], foreground=p["fg"], relief="flat",
                             font=("Segoe UI", 10, "bold"))
        self.style.map("Treeview", background=[("selected", p["green"])], foreground=[("selected", p["primary_fg"])])
        self.style.map("Treeview.Heading", background=[("active", p["card_hi"])])
        self.root.option_add("*TCombobox*Listbox.background", p["card"])
        self.root.option_add("*TCombobox*Listbox.foreground", p["fg"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", p["green"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", p["primary_fg"])
        self.items = [(w, o) for w, o in self.items if _alive(w)]          # forget closed windows (Learn ...)
        self.buttons = [b for b in self.buttons if _alive(b)]
        for w, opts in self.items:
            self._apply_one(w, opts)
        for b in self.buttons:
            b.retheme()
        for cb in list(self.callbacks):
            try:
                cb()
            except tk.TclError:                       # a widget the callback used has been destroyed
                self.callbacks.remove(cb)


class FlatButton(tk.Label):
    """A button drawn as a Label: consistent look everywhere, clear enabled / disabled states, hover."""

    def __init__(self, parent, skin: Skin, text, command, kind="secondary", big=False):
        pad = (30, 13) if big else (18, 9)
        super().__init__(parent, text=text, cursor="hand2", padx=pad[0], pady=pad[1], bd=0,
                         font=("Segoe UI", 13 if big else 11, "bold"), highlightthickness=1)
        self.kind, self.command = kind, command
        self.enabled, self._hover = True, False
        self.bind("<Button-1>", lambda _e: self.invoke())
        self.bind("<Enter>", lambda _e: self._set_hover(True))
        self.bind("<Leave>", lambda _e: self._set_hover(False))
        skin.buttons.append(self)
        self.retheme()

    def _set_hover(self, flag):
        self._hover = flag
        self.retheme()

    def set_enabled(self, flag: bool):
        self.enabled = bool(flag)
        self.configure(cursor="hand2" if self.enabled else "arrow")
        self.retheme()

    def set_text(self, text):
        self.configure(text=text)

    def invoke(self):
        if self.enabled and self.command:
            self.command()

    def retheme(self):
        p = palette()
        if self.kind == "primary":
            if self.enabled:
                bg = p["primary_hover"] if self._hover else p["primary_bg"]
                fg, border = p["primary_fg"], bg
            else:
                bg, fg, border = p["off_bg"], p["off_fg"], p["border"]
        else:
            bg = p["sec_hover"] if (self._hover and self.enabled) else p["sec_bg"]
            fg, border = p["sec_fg"], p["border"]
        try:
            self.configure(bg=bg, fg=fg, highlightbackground=border, highlightcolor=border)
        except tk.TclError:
            pass


def _base_root(title, w, h):
    root = tk.Tk()
    root.title(title)
    h = min(h, max(560, root.winfo_screenheight() - 90))   # never taller than the screen
    root.geometry(f"{w}x{h}")
    root.minsize(min(w, 760), min(h, 560))
    try:
        root.tk.call("tk", "scaling", 1.25)
    except tk.TclError:
        pass
    skin = Skin(root)
    skin.apply()
    return root, skin


def ask_yes_no(title, text):
    """A simple yes / no window. Returns True for yes (False if no window can be shown)."""
    try:
        root = tk.Tk()
        root.withdraw()
        answer = messagebox.askyesno(title, text, parent=root)
        root.destroy()
        return bool(answer)
    except tk.TclError:
        print(f"{title}: {text}")
        return False


def show_message(title, text, error=False):
    """A simple message window (used when the camera stops or a session fails)."""
    try:
        root = tk.Tk()
        root.withdraw()
        (messagebox.showerror if error else messagebox.showinfo)(title, text, parent=root)
        root.destroy()
    except tk.TclError:
        print(f"{title}: {text}")


def _theme_button(parent, skin):
    """Small light/dark switch for the top-right corner."""
    def label():
        return "Switch to light mode" if current_theme() == "dark" else "Switch to dark mode"

    def toggle():
        set_theme("light" if current_theme() == "dark" else "dark")
        skin.apply()
        btn.set_text(label())

    btn = FlatButton(parent, skin, label(), toggle)
    return btn


# ----------------------------------------------------------------------------
# 1. Welcome + consent
# ----------------------------------------------------------------------------

def run_welcome():
    """Returns {"trainee_id", "display_name"} once consent is given, else None."""
    result = {"value": None}
    root, skin = _base_root(APP_TITLE, 920, 820)

    head = skin.add(tk.Frame(root), bg="bg")
    head.pack(fill="x", padx=32, pady=(22, 0))
    _theme_button(head, skin).pack(side="right")
    skin.add(tk.Label(head, text="Welcome, future referee!", font=("Segoe UI", 24, "bold"), anchor="w"),
             bg="bg", fg="fg").pack(side="left")
    skin.add(tk.Label(root, text="Learn and practise the official FIVB referee hand signals, with instant feedback "
                                 "on whether you performed each one correctly. It is a training aid and does not replace a coach.",
                      font=("Segoe UI", 12), wraplength=850, justify="left"),
             bg="bg", fg="muted").pack(anchor="w", padx=32, pady=(4, 14))

    row = skin.add(tk.Frame(root), bg="bg")
    row.pack(fill="x", padx=32)
    skin.add(tk.Label(row, text="Your name or nickname:", font=("Segoe UI", 11, "bold")), bg="bg", fg="fg").pack(side="left")
    name_var = tk.StringVar()
    entry = skin.add(tk.Entry(row, textvariable=name_var, font=("Segoe UI", 12), relief="flat", width=28,
                              highlightthickness=1),
                     bg="card_hi", fg="fg", insertbackground="fg", highlightbackground="border", highlightcolor="green")
    entry.pack(side="left", padx=12, ipady=5)
    entry.focus_set()

    box = skin.add(tk.Frame(root, highlightthickness=1), bg="card", highlightbackground="border")
    box.pack(fill="both", expand=True, padx=32, pady=14)
    text = skin.add(tk.Text(box, wrap="word", font=("Segoe UI", 10), relief="flat", padx=16, pady=12,
                            highlightthickness=0), bg="card", fg="fg")
    scroll = tk.Scrollbar(box, command=text.yview)
    text.configure(yscrollcommand=scroll.set)
    scroll.pack(side="right", fill="y")
    text.pack(side="left", fill="both", expand=True)
    text.insert("1.0", CONSENT_TEXT)
    text.configure(state="disabled")

    agree_var = tk.BooleanVar(value=False)
    skin.add(tk.Checkbutton(root, variable=agree_var, font=("Segoe UI", 11), wraplength=840, justify="left",
                            highlightthickness=0, command=lambda: refresh(),
                            text="I have read this notice. I agree that my training sessions will be filmed and saved "
                                 "as described, and I am 18 or older (or have a parent/guardian's consent)."),
             bg="bg", fg="fg", selectcolor="card_hi", activebackground="bg",
             activeforeground="fg").pack(anchor="w", padx=32)

    footer = skin.add(tk.Frame(root), bg="bg")
    footer.pack(fill="x", padx=32, pady=16)
    status = skin.add(tk.Label(footer, text="", font=("Segoe UI", 10), anchor="w"), bg="bg", fg="muted")
    status.pack(side="left")

    def accept():
        display = name_var.get().strip()
        trainee_id = sanitize_id(display)
        record_consent(trainee_id, display)
        result["value"] = {"trainee_id": trainee_id, "display_name": display}
        root.destroy()

    go_btn = FlatButton(footer, skin, "I agree, continue", accept, kind="primary", big=True)
    go_btn.pack(side="right")
    FlatButton(footer, skin, "Decline and exit", root.destroy).pack(side="right", padx=10)

    def refresh(*_):
        has_name = bool(sanitize_id(name_var.get()))
        agreed = agree_var.get()
        ready = has_name and agreed
        go_btn.set_enabled(ready)
        missing = []
        if not has_name:
            missing.append("type your name")
        if not agreed:
            missing.append("tick the agreement box")
        status.configure(text="Ready to go." if ready else "To continue: " + " and ".join(missing) + ".",
                         fg=palette()["green"] if ready else palette()["muted"])

    name_var.trace_add("write", refresh)
    skin.callbacks.append(refresh)
    refresh()
    root.bind("<Return>", lambda _e: go_btn.invoke())

    root.mainloop()
    return result["value"]


# ----------------------------------------------------------------------------
# 3. Learn window (FIVB reference sheets)
# ----------------------------------------------------------------------------

def open_learn(parent, skin, start_label=None):
    win = tk.Toplevel(parent)
    win.title("Learn the signals")
    win.geometry("980x640")
    skin.add(win, bg="bg")

    labels = ["__setup__"] + list(gg.SIGNALS.keys())
    left = skin.add(tk.Frame(win, highlightthickness=1), bg="card", highlightbackground="border")
    left.pack(side="left", fill="y", padx=(16, 8), pady=16)
    skin.add(tk.Label(left, text="Signals", font=("Segoe UI", 10, "bold")), bg="card", fg="muted").pack(
        anchor="w", padx=12, pady=(10, 4))
    lb = skin.add(tk.Listbox(left, font=("Segoe UI", 11), relief="flat", highlightthickness=0, width=34,
                             activestyle="none"),
                  bg="card", fg="fg", selectbackground="green", selectforeground="primary_fg")
    for lab in labels:
        lb.insert("end", "How to set up (camera and position)" if lab == "__setup__" else gg.pretty_label(lab))
    lb.pack(fill="y", expand=True, padx=8, pady=(0, 10))

    right = skin.add(tk.Frame(win), bg="bg")
    right.pack(side="left", fill="both", expand=True, padx=(8, 16), pady=16)
    img_label = skin.add(tk.Label(right), bg="bg")
    img_label.pack(anchor="w")
    body = skin.add(tk.Text(right, wrap="word", font=("Segoe UI", 11), relief="flat", padx=16, pady=14,
                            highlightthickness=1), bg="card", fg="fg", highlightbackground="border")
    body.pack(fill="both", expand=True)
    skin.add(tk.Label(right, text=gg.DISCLAIMER, font=("Segoe UI", 9), wraplength=640, justify="left", anchor="w"),
             bg="bg", fg="muted").pack(anchor="w", pady=(6, 0))

    def retag():
        p = palette()
        body.tag_configure("h", font=("Segoe UI", 16, "bold"), foreground=p["green"], spacing3=6)
        body.tag_configure("k", font=("Segoe UI", 11, "bold"), foreground=p["blue"], spacing1=10)
        body.tag_configure("m", foreground=p["muted"])

    retag()
    skin.callbacks.append(retag)

    def on_close(event):
        if event.widget is win and retag in skin.callbacks:
            skin.callbacks.remove(retag)

    win.bind("<Destroy>", on_close)
    win._img_ref = None

    def show(_evt=None):
        sel = lb.curselection()
        if not sel:
            return
        label = labels[sel[0]]
        body.configure(state="normal")
        body.delete("1.0", "end")
        if label == "__setup__":
            body.insert("end", "How to set up\n", "h")
            body.insert("end", "For the best results\n", "k")
            for i, tip in enumerate(gg.SETUP_TIPS, 1):
                body.insert("end", f"  {i}. {tip}\n")
            body.insert("end", "\nThe camera reads a single 2D picture. Hands that overlap, sleeves that hide the elbows or "
                               "standing sideways make the checks unreliable, and the tool then says it could not verify "
                               "that part instead of guessing.\n", "m")
            body.configure(state="disabled")
            img_label.configure(image="")
            return
        s = gg.SIGNALS[label]
        body.insert("end", s["title"] + "\n", "h")
        body.insert("end", "FIVB signal\n", "k")
        body.insert("end", s["fivb"] + "\n")
        body.insert("end", "How to perform it\n", "k")
        for i, step in enumerate(s["howto"], 1):
            body.insert("end", f"  {i}. {step}\n")
        body.insert("end", "What is checked\n", "k")
        ctx = {"side": "left"} if label == "double_contact" else None
        for it in gg.graded_summary(label, ctx):
            body.insert("end", f"  [{it['effect']}] {it['label']} ({it['weight']} pts)\n")
        body.insert("end", "  Required: a miss limits the result to ALMOST. Important: same, but only when the camera "
                           "sees it. Scored: a miss only costs points.\n", "m")
        body.insert("end", "  " + gg.scoring_note("standard") + "\n", "m")
        if s.get("compare"):
            body.insert("end", "Easy to confuse\n", "k")
            body.insert("end", "  " + s["compare"] + "\n")
        body.insert("end", "Common mistakes\n", "k")
        for m in s["mistakes"]:
            body.insert("end", f"  - {m}\n")
        if s.get("not_graded"):
            body.insert("end", "Not graded by the camera\n", "k")
            body.insert("end", s["not_graded"] + "\n")
        body.insert("end", "\nWording follows the FIVB Official Volleyball Rules, Diagram 11 (Referees' Official Hand "
                           "Signals) and rule 30.1 (a signal is maintained for a moment). Left and right always mean YOUR "
                           "left and right.\n", "m")
        body.configure(state="disabled")
        img_path = os.path.join(SIGNAL_IMAGE_DIR, f"{label}.png")
        if os.path.exists(img_path):
            try:
                img = tk.PhotoImage(file=img_path)
                if img.height() > 240:                       # keep room for the text below the picture
                    img = img.subsample(-(-img.height() // 240))
                win._img_ref = img
                img_label.configure(image=img)
            except tk.TclError:
                img_label.configure(image="")
        else:
            img_label.configure(image="")

    lb.bind("<<ListboxSelect>>", show)
    idx = labels.index(start_label) if start_label in labels else 0
    lb.selection_set(idx)
    show()
    return win


# ----------------------------------------------------------------------------
# 4. My progress
# ----------------------------------------------------------------------------

def open_progress(parent, skin, trainee, on_practice=None):
    """Window with the trainee's progress over all their saved sessions (see progress.py)."""
    import webbrowser

    import progress

    data = progress.build_progress(trainee_dir(trainee["trainee_id"]))
    win = tk.Toplevel(parent)
    win.title("My progress")
    win.geometry("900x820")
    skin.add(win, bg="bg")

    skin.add(tk.Label(win, text=f"My progress: {trainee['display_name']}", font=("Segoe UI", 20, "bold"), anchor="w"),
             bg="bg", fg="fg").pack(fill="x", padx=28, pady=(18, 4))
    if not data["sessions"]:
        skin.add(tk.Label(win, text="No sessions yet. Do a Drill, Combo drill, Challenge or Match simulation and your "
                                    "progress will appear here.", font=("Segoe UI", 12), wraplength=820, justify="left",
                          anchor="w"), bg="bg", fg="muted").pack(fill="x", padx=28, pady=10)
        FlatButton(win, skin, "My sessions", lambda: open_sessions(win, skin, trainee)).pack(pady=(14, 4))
        FlatButton(win, skin, "Close", win.destroy).pack(pady=6)
        return win

    cards = skin.add(tk.Frame(win), bg="bg")
    cards.pack(fill="x", padx=28, pady=(6, 6))
    stats = [("Sessions", str(len(data["sessions"]))), ("Attempts", str(data["total_attempts"])),
             ("Correct", f"{data['accuracy']:.0f}%"), ("Average score", f"{data['avg_score']:.0f}/100")]
    for k, (name, val) in enumerate(stats):
        card = skin.add(tk.Frame(cards, highlightthickness=1), bg="card", highlightbackground="border")
        card.grid(row=0, column=k, padx=(0 if k == 0 else 8, 0), sticky="ew")
        cards.grid_columnconfigure(k, weight=1)
        skin.add(tk.Label(card, text=name, font=("Segoe UI", 9)), bg="card", fg="muted").pack(anchor="w", padx=12, pady=(8, 0))
        skin.add(tk.Label(card, text=val, font=("Segoe UI", 18, "bold")), bg="card", fg="fg").pack(anchor="w", padx=12, pady=(0, 8))
    skin.add(tk.Label(win, text=data["trend_text"], font=("Segoe UI", 11), anchor="w", wraplength=840, justify="left"),
             bg="bg", fg="green").pack(fill="x", padx=28)
    if data["whistle"]:
        w = data["whistle"]
        skin.add(tk.Label(win, text=f"Whistle: on time {w['on_time']} of {w['attempts']} ({w['pct']:.0f}%)",
                          font=("Segoe UI", 10), anchor="w"), bg="bg", fg="muted").pack(fill="x", padx=28)

    # ---- chart: average score per session
    skin.add(tk.Label(win, text="Average score per session", font=("Segoe UI", 10, "bold"), anchor="w"),
             bg="bg", fg="blue").pack(fill="x", padx=28, pady=(10, 2))
    canvas = tk.Canvas(win, height=170, highlightthickness=0, bd=0)
    canvas.pack(fill="x", padx=28)

    def draw_chart(_evt=None):
        p = palette()
        try:
            canvas.delete("all")
            canvas.configure(bg=p["card"])
        except tk.TclError:
            return
        W = canvas.winfo_width() or 840
        H = 170
        left, right, top, bottom = 40, 16, 14, 26
        for v in (0, 50, 100):
            y = top + (H - top - bottom) * (1 - v / 100.0)
            canvas.create_line(left, y, W - right, y, fill=p["border"])
            canvas.create_text(left - 6, y, text=str(v), fill=p["muted"], anchor="e", font=("Segoe UI", 8))
        sess = data["sessions"]
        n = len(sess)
        pts = []
        for i, s in enumerate(sess):
            x = left + (W - left - right) * (0.5 if n == 1 else i / (n - 1))
            y = top + (H - top - bottom) * (1 - s["avg_score"] / 100.0)
            pts.append((x, y))
        if n > 1:
            canvas.create_line(*[c for pt in pts for c in pt], fill=p["green"], width=2)
        for x, y in pts:
            canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill=p["green"], outline=p["green"])
        first, last = sess[0]["when"], sess[-1]["when"]
        canvas.create_text(left, H - 10, anchor="w", fill=p["muted"], font=("Segoe UI", 8),
                           text=first.strftime("%b %d %H:%M") if first else "first")
        canvas.create_text(W - right, H - 10, anchor="e", fill=p["muted"], font=("Segoe UI", 8),
                           text=last.strftime("%b %d %H:%M") if last else "latest")

    canvas.bind("<Configure>", draw_chart)
    skin.callbacks.append(draw_chart)

    # ---- table by signal
    skin.add(tk.Label(win, text="By signal (weakest first)", font=("Segoe UI", 10, "bold"), anchor="w"),
             bg="bg", fg="blue").pack(fill="x", padx=28, pady=(10, 2))
    cols = ("signal", "tries", "correct", "acc", "avg", "trend")
    tree = ttk.Treeview(win, columns=cols, show="headings", height=min(8, max(3, len(data["per_signal"]))))
    for c, txt, wd in (("signal", "Signal", 300), ("tries", "Tries", 60), ("correct", "Correct", 70),
                       ("acc", "Accuracy", 80), ("avg", "Avg score", 80), ("trend", "Trend", 80)):
        tree.heading(c, text=txt)
        tree.column(c, width=wd, anchor="w" if c == "signal" else "center")
    for label, v in sorted(data["per_signal"].items(), key=lambda kv: kv[1]["accuracy"]):
        trend = "" if v["trend"] is None else ("up " if v["trend"] > 2 else "down " if v["trend"] < -2 else "same ") + "%+.0f" % v["trend"]
        tree.insert("", "end", values=(gg.short_label(label), v["attempts"], v["correct"], f"{v['accuracy']:.0f}%",
                                       f"{v['avg_score']:.0f}", trend))
    tree.pack(fill="x", padx=28)

    # ---- what to work on
    skin.add(tk.Label(win, text="What to work on", font=("Segoe UI", 10, "bold"), anchor="w"),
             bg="bg", fg="blue").pack(fill="x", padx=28, pady=(10, 2))
    for line in progress.focus_lines(data) or ["Nothing yet."]:
        skin.add(tk.Label(win, text="- " + line, font=("Segoe UI", 10), anchor="w", wraplength=840, justify="left"),
                 bg="bg", fg="fg").pack(fill="x", padx=28)

    bar = skin.add(tk.Frame(win), bg="bg")
    bar.pack(fill="x", padx=28, pady=16, side="bottom")

    def practise():
        if on_practice and data["weakest"]:
            on_practice(data["weakest"][0])
            win.destroy()

    def save_report():
        path = os.path.join(trainee_dir(trainee["trainee_id"]), "progress.html")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(progress.progress_html(data, trainee["display_name"]))
            webbrowser.open("file:///" + os.path.abspath(path).replace("\\", "/"))
        except OSError as exc:
            messagebox.showerror("Progress report", f"The report could not be saved: {exc}", parent=win)

    FlatButton(bar, skin, "Close", win.destroy).pack(side="right")
    FlatButton(bar, skin, "My sessions", lambda: open_sessions(win, skin, trainee)).pack(side="right", padx=(0, 8))
    FlatButton(bar, skin, "Save report (HTML)", save_report).pack(side="right", padx=(0, 8))
    practise_btn = FlatButton(bar, skin, "Practise my weakest signal", practise, kind="primary")
    practise_btn.pack(side="right")
    practise_btn.set_enabled(bool(on_practice and data["weakest"]))

    def on_close(event):
        if event.widget is win and draw_chart in skin.callbacks:
            skin.callbacks.remove(draw_chart)

    win.bind("<Destroy>", on_close)
    skin.apply()
    return win


def open_sessions(parent, skin, trainee):
    """List of the trainee's saved sessions: open the report, play the video, or delete just that session."""
    import progress

    win = tk.Toplevel(parent)
    win.title("My sessions")
    win.geometry("900x560")
    skin.add(win, bg="bg")
    skin.add(tk.Label(win, text=f"My sessions: {trainee['display_name']}", font=("Segoe UI", 20, "bold"), anchor="w"),
             bg="bg", fg="fg").pack(fill="x", padx=28, pady=(18, 2))
    skin.add(tk.Label(win, text="Select a session, then open its report, play its video, or delete only that session. "
                                "To delete ALL your data, use Delete my data in the main menu.",
                      font=("Segoe UI", 10), wraplength=840, justify="left", anchor="w"),
             bg="bg", fg="muted").pack(fill="x", padx=28, pady=(0, 8))

    cols = ("when", "mode", "level", "attempts", "correct", "points", "size")
    tree = ttk.Treeview(win, columns=cols, show="headings", height=12, selectmode="browse")
    for c, txt, wd in (("when", "Date and time", 190), ("mode", "Mode", 130), ("level", "Level", 80),
                       ("attempts", "Attempts", 80), ("correct", "Correct", 80), ("points", "Points", 90),
                       ("size", "Size (MB)", 90)):
        tree.heading(c, text=txt)
        tree.column(c, width=wd, anchor="w" if c in ("when", "mode") else "center")
    tree.pack(fill="both", expand=True, padx=28)
    rows = {}

    def fill():
        tree.delete(*tree.get_children())
        rows.clear()
        for s in progress.list_sessions(trainee_dir(trainee["trainee_id"])):
            when = s["when"].strftime("%Y-%m-%d %H:%M") if s["when"] else s["session"]
            mode = s["mode"] + (" (test)" if s["test"] else "")
            iid = tree.insert("", "end", values=(when, mode, s["level"], s["attempts"], s["correct"],
                                                 f"{s['points']}/{s['max_points']}", f"{s['size_mb']:.1f}"))
            rows[iid] = s
        empty.configure(text="" if rows else "No saved sessions yet.")

    empty = skin.add(tk.Label(win, text="", font=("Segoe UI", 11)), bg="bg", fg="muted")
    empty.pack()

    def selected():
        sel = tree.selection()
        return rows.get(sel[0]) if sel else None

    def need_selection():
        messagebox.showinfo("My sessions", "Select a session in the list first.", parent=win)

    def open_report():
        s = selected()
        if not s:
            return need_selection()
        path = os.path.join(s["path"], "report.html")
        if not (os.path.exists(path) and open_path(path)):
            messagebox.showinfo("My sessions", "This session has no report.", parent=win)

    def play_video():
        s = selected()
        if not s:
            return need_selection()
        path = os.path.join(s["path"], "session.mp4")
        if not (os.path.exists(path) and open_path(path)):
            messagebox.showinfo("My sessions", "This session has no video, or it could not be opened. It is saved as "
                                               "session.mp4 in the session folder.", parent=win)

    def open_folder():
        s = selected()
        if not s:
            return need_selection()
        open_path(s["path"])

    def delete_one():
        s = selected()
        if not s:
            return need_selection()
        when = s["when"].strftime("%Y-%m-%d %H:%M") if s["when"] else s["session"]
        if messagebox.askyesno("Delete this session",
                               f"Delete the session from {when} ({s['mode']}, {s['size_mb']:.1f} MB)? Its video, report and "
                               "results are removed for good. Your other sessions are not touched.", parent=win):
            delete_session(trainee["trainee_id"], s["session"])
            fill()

    bar = skin.add(tk.Frame(win), bg="bg")
    bar.pack(fill="x", padx=28, pady=14)
    FlatButton(bar, skin, "Close", win.destroy).pack(side="right")
    FlatButton(bar, skin, "Delete this session", delete_one).pack(side="right", padx=(0, 8))
    FlatButton(bar, skin, "Open folder", open_folder).pack(side="right", padx=(0, 8))
    FlatButton(bar, skin, "Play video", play_video).pack(side="right", padx=(0, 8))
    FlatButton(bar, skin, "Open report", open_report, kind="primary").pack(side="right", padx=(0, 8))
    fill()
    skin.apply()
    return win


# ----------------------------------------------------------------------------
# 2. Main menu
# ----------------------------------------------------------------------------

def run_menu(trainee: dict, real_labels, last_summary: str = ""):
    """
    Returns one of:
        {"action": "start", "mode", "gesture", "level", "reps", "combo", "intent", "note", "whistle", "continuous"}
        {"action": "quit"}
        {"action": "deleted"}
        {"action": "devices"}     (open the camera and microphone setup)
    """
    result = {"value": {"action": "quit"}}
    root, skin = _base_root(APP_TITLE, 1000, 1080 if TEST_MODE else 900)

    head = skin.add(tk.Frame(root), bg="bg")
    head.pack(fill="x", padx=32, pady=(22, 0))
    _theme_button(head, skin).pack(side="right")
    FlatButton(head, skin, "Quit", root.destroy).pack(side="right", padx=(0, 8))
    skin.add(tk.Label(head, text=f"Hi, {trainee['display_name']}!", font=("Segoe UI", 22, "bold"), anchor="w"),
             bg="bg", fg="fg").pack(side="left")
    skin.add(tk.Label(root, text="What would you like to do today?", font=("Segoe UI", 12), anchor="w"),
             bg="bg", fg="muted").pack(anchor="w", padx=32, pady=(0, 8))
    if last_summary:
        skin.add(tk.Label(root, text=f"Last session: {last_summary}", font=("Segoe UI", 10), anchor="w"),
                 bg="bg", fg="amber").pack(anchor="w", padx=32, pady=(0, 6))

    bar = skin.add(tk.Frame(root), bg="bg")
    bar.pack(fill="x", padx=32, pady=14, side="bottom")
    skin.add(tk.Label(root, text=gg.DISCLAIMER, font=("Segoe UI", 9), wraplength=830, justify="left", anchor="w"),
             bg="bg", fg="muted").pack(fill="x", padx=32, side="bottom")

    last = read_settings().get("last_choice") or {}
    mode_var = tk.StringVar(value=last.get("mode") if last.get("mode") in [m[0] for m in MODES] else "drill")
    cards = skin.add(tk.Frame(root), bg="bg")
    cards.pack(fill="x", padx=32)
    for key, title, desc in MODES:
        f = skin.add(tk.Frame(cards, highlightthickness=1), bg="card", highlightbackground="border")
        f.pack(fill="x", pady=2)
        skin.add(tk.Radiobutton(f, variable=mode_var, value=key, text=title, font=("Segoe UI", 12, "bold"),
                                highlightthickness=0, command=lambda: update_state()),
                 bg="card", fg="fg", selectcolor="card_hi", activebackground="card",
                 activeforeground="fg").pack(anchor="w", padx=12, pady=(6, 0))
        skin.add(tk.Label(f, text=desc, font=("Segoe UI", 10), wraplength=740, justify="left"),
                 bg="card", fg="muted").pack(anchor="w", padx=40, pady=(0, 6))

    opts = skin.add(tk.Frame(root), bg="bg")
    opts.pack(fill="x", padx=32, pady=(10, 0))

    sig_names = [gg.pretty_label(l) for l in real_labels if l in gg.SIGNALS]
    sig_labels = [l for l in real_labels if l in gg.SIGNALS]
    combos = [(k, n) for k, n in COMBO_CHOICES if k == "random" or k in real_labels]

    def opt_label(text, row):
        skin.add(tk.Label(opts, text=text, font=("Segoe UI", 11, "bold")), bg="bg", fg="fg").grid(
            row=row, column=0, sticky="w")

    opt_label("Signal (Drill):", 0)
    last_sig = gg.pretty_label(last["gesture"]) if last.get("gesture") in sig_labels else None
    sig_var = tk.StringVar(value=last_sig or (sig_names[0] if sig_names else ""))
    sig_box = ttk.Combobox(opts, textvariable=sig_var, values=sig_names, state="readonly", width=42, font=("Segoe UI", 11))
    sig_box.grid(row=0, column=1, sticky="w", padx=12, pady=3)

    opt_label("Repetitions (Drill, Combo):", 1)
    last_reps = str(last.get("reps", 5))
    reps_var = tk.StringVar(value=next((r for r in REP_CHOICES if r.split()[0] == last_reps), "5"))
    reps_box = ttk.Combobox(opts, textvariable=reps_var, values=REP_CHOICES, state="readonly", width=42, font=("Segoe UI", 11))
    reps_box.grid(row=1, column=1, sticky="w", padx=12, pady=3)

    opt_label("Combo:", 2)
    combo_names = [n for _, n in combos]
    combo_var = tk.StringVar(value=next((n for k, n in combos if k == last.get("combo")), combo_names[0]))
    combo_box = ttk.Combobox(opts, textvariable=combo_var, values=combo_names, state="readonly", width=42, font=("Segoe UI", 11))
    combo_box.grid(row=2, column=1, sticky="w", padx=12, pady=3)

    opt_label("Difficulty:", 3)
    level_var = tk.StringVar(value=last.get("level") if last.get("level") in gg.LEVELS else "standard")
    lv_frame = skin.add(tk.Frame(opts), bg="bg")
    lv_frame.grid(row=3, column=1, sticky="w", padx=12)
    level_desc = skin.add(tk.Label(opts, text=LEVEL_INFO[level_var.get()], font=("Segoe UI", 10)), bg="bg", fg="muted")
    level_desc.grid(row=4, column=1, sticky="w", padx=12)

    whistle_var = tk.BooleanVar(value=bool(last.get("whistle", False)))
    whistle_box = skin.add(tk.Checkbutton(opts, variable=whistle_var, highlightthickness=0, font=("Segoe UI", 11),
                                          text="Require the whistle first (Team to Serve and Authorization to Serve)"),
                           bg="bg", fg="fg", selectcolor="card_hi", activebackground="bg", activeforeground="fg")
    whistle_box.grid(row=8, column=0, columnspan=2, sticky="w", pady=(8, 0))
    continuous_var = tk.BooleanVar(value=bool(last.get("continuous", False)))
    continuous_box = skin.add(tk.Checkbutton(opts, variable=continuous_var, highlightthickness=0, font=("Segoe UI", 11),
                                             text="Continuous (Match simulation runs the whole set by itself, timed to be readable)"),
                              bg="bg", fg="fg", selectcolor="card_hi", activebackground="bg", activeforeground="fg")
    continuous_box.grid(row=9, column=0, columnspan=2, sticky="w", pady=(2, 0))

    intent_var = tk.StringVar(value=INTENT_CHOICES[0][1])
    note_var = tk.StringVar()
    if TEST_MODE:
        opt_label("Session label:", 5)
        intent_box = ttk.Combobox(opts, textvariable=intent_var, values=[n for _, n in INTENT_CHOICES],
                                  state="readonly", width=42, font=("Segoe UI", 11))
        intent_box.grid(row=5, column=1, sticky="w", padx=12, pady=(8, 3))

        opt_label("Note (test sessions):", 6)
        note_entry = skin.add(tk.Entry(opts, textvariable=note_var, font=("Segoe UI", 11), relief="flat", width=44,
                                       highlightthickness=1),
                              bg="card_hi", fg="fg", insertbackground="fg", highlightbackground="border",
                              highlightcolor="green")
        note_entry.grid(row=6, column=1, sticky="w", padx=12, pady=3, ipady=3)
        skin.add(tk.Label(opts, text="TEST MODE. What you will do differently, e.g. 'elbow bent' or 'one arm only'. "
                                     "Saved with each attempt.", font=("Segoe UI", 9)),
                 bg="bg", fg="muted").grid(row=7, column=1, sticky="w", padx=12)

    def on_level():
        level_desc.configure(text=LEVEL_INFO[level_var.get()])

    for lv in gg.LEVELS:
        skin.add(tk.Radiobutton(lv_frame, variable=level_var, value=lv, text=gg.LEVEL_CONFIG[lv].name,
                                font=("Segoe UI", 11), highlightthickness=0, command=on_level),
                 bg="bg", fg="fg", selectcolor="card_hi", activebackground="bg",
                 activeforeground="fg").pack(side="left", padx=(0, 14))

    def update_state():
        m = mode_var.get()
        sig_box.configure(state="readonly" if m == "drill" else "disabled")
        reps_box.configure(state="readonly" if m in ("drill", "combo") else "disabled")
        combo_box.configure(state="readonly" if m == "combo" else "disabled")
        whistle_box.configure(state="normal" if m in ("drill", "combo", "challenge") else "disabled")
        continuous_box.configure(state="normal" if m == "sim" else "disabled")

    update_state()

    def start():
        label = None
        if mode_var.get() == "drill":
            name = sig_var.get()
            label = sig_labels[sig_names.index(name)] if name in sig_names else None
        combo_key = next((k for k, n in combos if n == combo_var.get()), "random")
        try:
            reps = int(reps_var.get().split()[0])
        except ValueError:
            reps = 1
        intent = next((k for k, n in INTENT_CHOICES if n == intent_var.get()), "normal") if TEST_MODE else "normal"
        result["value"] = {"action": "start", "mode": mode_var.get(), "gesture": label, "level": level_var.get(),
                           "reps": reps, "combo": combo_key, "intent": intent,
                           "note": note_var.get().strip() if TEST_MODE else "",
                           "whistle": bool(whistle_var.get()) and mode_var.get() in ("drill", "combo", "challenge"),
                           "continuous": bool(continuous_var.get()) and mode_var.get() == "sim"}
        # remember the choices (also for the next time the app is opened); the test label is never remembered
        remembered = {k: v for k, v in result["value"].items() if k not in ("action", "intent", "note")}
        remembered["gesture"] = label or last.get("gesture")
        write_settings({"last_choice": remembered})
        root.destroy()

    def learn():
        name = sig_var.get()
        label = sig_labels[sig_names.index(name)] if name in sig_names else None
        open_learn(root, skin, label)

    def devices():
        result["value"] = {"action": "devices"}
        root.destroy()

    def practise(label):
        mode_var.set("drill")
        sig_var.set(gg.pretty_label(label))
        update_state()

    def progress_window():
        open_progress(root, skin, trainee, practise)

    def delete():
        n, mb = trainee_data_summary(trainee["trainee_id"])
        if messagebox.askyesno(
                "Delete my data",
                f"This permanently deletes ALL saved data for '{trainee['display_name']}': {n} session(s), "
                f"{mb:.1f} MB (videos, results and movement files), and withdraws your consent. It is not only the "
                "latest session.\n\nOther people's data is not touched, and the consent log keeps a line saying that "
                "the data was deleted. You will need to accept the notice again to use the tool.\n\nDelete now?",
                parent=root):
            delete_trainee_data(trainee["trainee_id"])
            result["value"] = {"action": "deleted"}
            root.destroy()

    FlatButton(bar, skin, "Start session", start, kind="primary", big=True).pack(side="right")
    FlatButton(bar, skin, "Learn the signals", learn).pack(side="right", padx=(0, 8))
    FlatButton(bar, skin, "Progress and sessions", progress_window).pack(side="right", padx=(0, 8))
    FlatButton(bar, skin, "Camera and mic", devices).pack(side="right", padx=(0, 8))
    FlatButton(bar, skin, "Delete my data", delete).pack(side="left")

    skin.apply()
    root.mainloop()
    return result["value"]


if __name__ == "__main__":
    # Quick preview of the screens. The real program is:  python trainer.py
    t = run_welcome()
    print("welcome ->", t)
    while t:
        choice = run_menu(t, list(gg.SIGNALS.keys()))
        print("menu ->", choice)
        if choice.get("action") == "devices":
            import device_setup
            print("camera and mic ->", device_setup.run_device_setup())
            continue
        break