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
CONSENT_VERSION = "2026-09-v1"
DATA_ROOT = os.path.join("data", "trainer_sessions")
CONSENT_LOG = os.path.join("data", "consent_records.csv")
SETTINGS_PATH = os.path.join("data", "ui_settings.json")
SIGNAL_IMAGE_DIR = os.path.join("assets", "signals")   # optional: assets/signals/<label>.png

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
     "Back-to-back calls like the end of a rally: Team to Serve, then the reason. Runs automatically; "
     "review every score at the end."),
    ("challenge", "Challenge",
     "Every signal once, in random order. Your score is the number of signals you perform correctly."),
    ("sim", "Match simulation",
     "Scenarios like a real rally: blow the whistle, then give the right signals in order."),
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

LEVEL_INFO = {
    "beginner": "Forgiving: wide angle tolerances, shorter hold.",
    "standard": "Balanced: a clear, recognisable signal.",
    "referee": "Strict: close to textbook form, longer hold.",
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


def _load_saved_theme():
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            name = json.load(f).get("theme")
        if name in PALETTES:
            return name
    except (OSError, ValueError):
        pass
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
        try:
            os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump({"theme": name}, f)
        except OSError:
            pass


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
        self.root.option_add("*TCombobox*Listbox.background", p["card"])
        self.root.option_add("*TCombobox*Listbox.foreground", p["fg"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", p["green"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", p["primary_fg"])
        for w, opts in self.items:
            self._apply_one(w, opts)
        for b in self.buttons:
            b.retheme()
        for cb in self.callbacks:
            cb()


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
    root.geometry(f"{w}x{h}")
    root.minsize(min(w, 760), min(h, 560))
    try:
        root.tk.call("tk", "scaling", 1.25)
    except tk.TclError:
        pass
    skin = Skin(root)
    skin.apply()
    return root, skin


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
                                 "on whether you performed each one correctly.",
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

    labels = list(gg.SIGNALS.keys())
    left = skin.add(tk.Frame(win, highlightthickness=1), bg="card", highlightbackground="border")
    left.pack(side="left", fill="y", padx=(16, 8), pady=16)
    skin.add(tk.Label(left, text="Signals", font=("Segoe UI", 10, "bold")), bg="card", fg="muted").pack(
        anchor="w", padx=12, pady=(10, 4))
    lb = skin.add(tk.Listbox(left, font=("Segoe UI", 11), relief="flat", highlightthickness=0, width=34,
                             activestyle="none"),
                  bg="card", fg="fg", selectbackground="green", selectforeground="primary_fg")
    for lab in labels:
        lb.insert("end", gg.pretty_label(lab))
    lb.pack(fill="y", expand=True, padx=8, pady=(0, 10))

    right = skin.add(tk.Frame(win), bg="bg")
    right.pack(side="left", fill="both", expand=True, padx=(8, 16), pady=16)
    img_label = skin.add(tk.Label(right), bg="bg")
    img_label.pack(anchor="w")
    body = skin.add(tk.Text(right, wrap="word", font=("Segoe UI", 11), relief="flat", padx=16, pady=14,
                            highlightthickness=1), bg="card", fg="fg", highlightbackground="border")
    body.pack(fill="both", expand=True)

    def retag():
        p = palette()
        body.tag_configure("h", font=("Segoe UI", 16, "bold"), foreground=p["green"], spacing3=6)
        body.tag_configure("k", font=("Segoe UI", 11, "bold"), foreground=p["blue"], spacing1=10)
        body.tag_configure("m", foreground=p["muted"])

    retag()
    skin.callbacks.append(retag)
    win._img_ref = None

    def show(_evt=None):
        sel = lb.curselection()
        if not sel:
            return
        label = labels[sel[0]]
        s = gg.SIGNALS[label]
        body.configure(state="normal")
        body.delete("1.0", "end")
        body.insert("end", s["title"] + "\n", "h")
        body.insert("end", "FIVB signal\n", "k")
        body.insert("end", s["fivb"] + "\n")
        body.insert("end", "How to perform it\n", "k")
        for i, step in enumerate(s["howto"], 1):
            body.insert("end", f"  {i}. {step}\n")
        body.insert("end", "Common mistakes\n", "k")
        for m in s["mistakes"]:
            body.insert("end", f"  - {m}\n")
        if s.get("not_graded"):
            body.insert("end", "Not graded by the camera\n", "k")
            body.insert("end", s["not_graded"] + "\n")
        body.insert("end", "\nWording paraphrased from the FIVB Official Volleyball Rules (Referee Hand Signals). "
                           "Left and right always mean YOUR left and right.\n", "m")
        body.configure(state="disabled")
        img_path = os.path.join(SIGNAL_IMAGE_DIR, f"{label}.png")
        if os.path.exists(img_path):
            try:
                win._img_ref = tk.PhotoImage(file=img_path)
                img_label.configure(image=win._img_ref)
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
# 2. Main menu
# ----------------------------------------------------------------------------

def run_menu(trainee: dict, real_labels, last_summary: str = ""):
    """
    Returns one of:
        {"action": "start", "mode", "gesture", "level", "reps", "combo", "intent"}
        {"action": "quit"}
        {"action": "deleted"}
    """
    result = {"value": {"action": "quit"}}
    root, skin = _base_root(APP_TITLE, 900, 980)

    head = skin.add(tk.Frame(root), bg="bg")
    head.pack(fill="x", padx=32, pady=(22, 0))
    _theme_button(head, skin).pack(side="right")
    skin.add(tk.Label(head, text=f"Hi, {trainee['display_name']}!", font=("Segoe UI", 22, "bold"), anchor="w"),
             bg="bg", fg="fg").pack(side="left")
    skin.add(tk.Label(root, text="What would you like to do today?", font=("Segoe UI", 12), anchor="w"),
             bg="bg", fg="muted").pack(anchor="w", padx=32, pady=(0, 8))
    if last_summary:
        skin.add(tk.Label(root, text=f"Last session: {last_summary}", font=("Segoe UI", 10), anchor="w"),
                 bg="bg", fg="amber").pack(anchor="w", padx=32, pady=(0, 6))

    mode_var = tk.StringVar(value="drill")
    cards = skin.add(tk.Frame(root), bg="bg")
    cards.pack(fill="x", padx=32)
    for key, title, desc in MODES:
        f = skin.add(tk.Frame(cards, highlightthickness=1), bg="card", highlightbackground="border")
        f.pack(fill="x", pady=3)
        skin.add(tk.Radiobutton(f, variable=mode_var, value=key, text=title, font=("Segoe UI", 12, "bold"),
                                highlightthickness=0, command=lambda: update_state()),
                 bg="card", fg="fg", selectcolor="card_hi", activebackground="card",
                 activeforeground="fg").pack(anchor="w", padx=12, pady=(6, 0))
        skin.add(tk.Label(f, text=desc, font=("Segoe UI", 10), wraplength=780, justify="left"),
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
    sig_var = tk.StringVar(value=sig_names[0] if sig_names else "")
    sig_box = ttk.Combobox(opts, textvariable=sig_var, values=sig_names, state="readonly", width=42, font=("Segoe UI", 11))
    sig_box.grid(row=0, column=1, sticky="w", padx=12, pady=3)

    opt_label("Repetitions (Drill, Combo):", 1)
    reps_var = tk.StringVar(value="5")
    reps_box = ttk.Combobox(opts, textvariable=reps_var, values=REP_CHOICES, state="readonly", width=42, font=("Segoe UI", 11))
    reps_box.grid(row=1, column=1, sticky="w", padx=12, pady=3)

    opt_label("Combo:", 2)
    combo_names = [n for _, n in combos]
    combo_var = tk.StringVar(value=combo_names[0])
    combo_box = ttk.Combobox(opts, textvariable=combo_var, values=combo_names, state="readonly", width=42, font=("Segoe UI", 11))
    combo_box.grid(row=2, column=1, sticky="w", padx=12, pady=3)

    opt_label("Difficulty:", 3)
    level_var = tk.StringVar(value="standard")
    lv_frame = skin.add(tk.Frame(opts), bg="bg")
    lv_frame.grid(row=3, column=1, sticky="w", padx=12)
    level_desc = skin.add(tk.Label(opts, text=LEVEL_INFO["standard"], font=("Segoe UI", 10)), bg="bg", fg="muted")
    level_desc.grid(row=4, column=1, sticky="w", padx=12)

    opt_label("Session label:", 5)
    intent_var = tk.StringVar(value=INTENT_CHOICES[0][1])
    intent_box = ttk.Combobox(opts, textvariable=intent_var, values=[n for _, n in INTENT_CHOICES], state="readonly",
                              width=42, font=("Segoe UI", 11))
    intent_box.grid(row=5, column=1, sticky="w", padx=12, pady=(8, 3))

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
        intent = next((k for k, n in INTENT_CHOICES if n == intent_var.get()), "normal")
        result["value"] = {"action": "start", "mode": mode_var.get(), "gesture": label, "level": level_var.get(),
                           "reps": reps, "combo": combo_key, "intent": intent}
        root.destroy()

    def learn():
        name = sig_var.get()
        label = sig_labels[sig_names.index(name)] if name in sig_names else None
        open_learn(root, skin, label)

    def delete():
        if messagebox.askyesno(
                "Delete my data",
                "This permanently deletes all saved videos, reports and results for you, and withdraws your "
                "consent. You will need to accept the notice again to use the tool.\n\nDelete now?", parent=root):
            delete_trainee_data(trainee["trainee_id"])
            result["value"] = {"action": "deleted"}
            root.destroy()

    bar = skin.add(tk.Frame(root), bg="bg")
    bar.pack(fill="x", padx=32, pady=18, side="bottom")
    FlatButton(bar, skin, "Start session", start, kind="primary", big=True).pack(side="right")
    FlatButton(bar, skin, "Learn the signals", learn).pack(side="right", padx=10)
    FlatButton(bar, skin, "Delete my data", delete).pack(side="left")
    FlatButton(bar, skin, "Quit", root.destroy).pack(side="left", padx=10)

    skin.apply()
    root.mainloop()
    return result["value"]


if __name__ == "__main__":
    t = run_welcome()
    print("welcome ->", t)
    if t:
        print("menu ->", run_menu(t, list(gg.SIGNALS.keys())))