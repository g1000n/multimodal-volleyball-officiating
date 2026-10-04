"""
trainer_config.py

The ONLY file you need to edit for your own settings. Updates to trainer.py / trainer_ui.py will not
overwrite it, so your values stay put.

Keep the quotes around text values. Do not keep the square brackets from the old placeholders.
"""

# ---- Consent notice (shown on the welcome screen) ---------------------------------------------
CONTACT = "rblobo@student.hau.edu.ph"                      
RETENTION = "for one (1) year after the completion of this study, after which recordings are deleted"  # edit if not accurate

# ---- Camera -----------------------------------------------------------------------------------
# Which camera to open. None = use CAMERA_INDEX from live_deployment.py. If that camera cannot be
# opened, the trainer automatically tries the others and tells you which one it used.
# See which numbers work with:  python trainer.py --list-cameras
CAMERA_INDEX = None
# True if your camera source already mirrors the picture (for example a Camo mirror option).
# Test: raise your LEFT arm; if the skeleton highlights the RIGHT arm, set this to True.
INPUT_ALREADY_MIRRORED = False
# Show yourself like a mirror on screen (display only; grading always uses your true left/right).
MIRROR_DISPLAY = True

# ---- Whistle (Match simulation) ---------------------------------------------------------------
# sounddevice index of your microphone; None = system default (same idea as in live_deployment.py).
WHISTLE_DEVICE_INDEX = None
# Model probability a half-second window needs to count as whistle (two in a row confirm one whistle). The whistle
# model's own default, used by live_deployment.py, is 0.70. The training tool uses 0.55: on 2026-10-04, replaying the
# detector's own logs of 51 clean whistles from a phone mic at 1-2 m (most scored 0.55-0.68), 0.70 confirmed 8/51,
# 0.60 confirmed 39/51 and 0.55 confirmed 45/51; 0.50 confirmed the same 45/51, so going lower only adds noise risk.
# The remaining misses were whistles blown within the 2 s cooldown of the previous one. The model is unchanged.
# Check false alarms with the no-whistle noise test (tools/pilot_check.py --noise-session); raise it if it fires.
WHISTLE_THRESHOLD = 0.55

# ---- Test sessions (your own dry run only) ----------------------------------------------------
# True shows a "Session label" box in the menu ("TEST: I will perform correctly" / "...WRONG on purpose")
# and a note field, so tools/pilot_check.py and tools/evaluate_sessions.py can score your dry run.
# Set back to False before real evaluators use the tool.
TEST_MODE = False

# ---- Look -------------------------------------------------------------------------------------
# Default theme on a fresh install: "dark" or "light". Once you switch in the app, your choice is
# remembered in data/ui_settings.json (press T during a session, or use the button in the menus).
THEME = "dark"