"""
trainer_config.py

The ONLY file you need to edit for your own settings. Updates to trainer.py / trainer_ui.py will not
overwrite it, so your values stay put.

Keep the quotes around text values. Do not keep the square brackets from the old placeholders.
"""

# ---- Consent notice (shown on the welcome screen) ---------------------------------------------
CONTACT = "rblobo@student.hau.edu.ph"                      # TODO: put your real school email here
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

# ---- Look -------------------------------------------------------------------------------------
# Default theme on a fresh install: "dark" or "light". Once you switch in the app, your choice is
# remembered in data/ui_settings.json (press T during a session, or use the button in the menus).
THEME = "dark"