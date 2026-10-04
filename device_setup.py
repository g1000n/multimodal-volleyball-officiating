"""
device_setup.py

"Camera and microphone check" screen (Tkinter).

* Camera: lists every camera that opens, shows a live preview of the selected one, so you can pick the
  right one (built in webcam, Camo, external ...).
* Microphone: lists every input microphone and shows a live level bar. Clap or blow your whistle: if the
  bar moves, the microphone works. If no microphone is found, everything except the whistle in the match
  simulation still works (press W instead).

The choice is saved in data/ui_settings.json. It is shown automatically the first time the trainer runs and
can be reopened from the menu ("Camera and mic").
"""

import base64
import time
import tkinter as tk
from tkinter import ttk

import cv2

import devices
import trainer_ui
from trainer_ui import FlatButton, palette


def run_device_setup():
    """Returns {"camera": int|None, "mic": int|None, "saved": bool}."""
    saved = trainer_ui.read_settings()
    root, skin = trainer_ui._base_root("Camera and microphone check", 900, 800)
    result = {"camera": None, "mic": None, "saved": False}
    st = {"cap": None, "photo": None, "cams": [], "mics": []}
    meter = devices.MicMeter()
    mirror = bool(getattr(trainer_ui._cfg, "MIRROR_DISPLAY", True)) if trainer_ui._cfg else True

    head = skin.add(tk.Frame(root), bg="bg")
    head.pack(fill="x", padx=32, pady=(22, 0))
    trainer_ui._theme_button(head, skin).pack(side="right")
    skin.add(tk.Label(head, text="Camera and microphone check", font=("Segoe UI", 22, "bold"), anchor="w"),
             bg="bg", fg="fg").pack(side="left")
    skin.add(tk.Label(root, text="Pick the camera you will use and check that your microphone hears you. "
                                 "You can change this later from the menu.",
                      font=("Segoe UI", 11), wraplength=830, justify="left"),
             bg="bg", fg="muted").pack(anchor="w", padx=32, pady=(4, 12))

    # ---- buttons first, so they never get pushed off small screens ----
    bar = skin.add(tk.Frame(root), bg="bg")
    bar.pack(fill="x", padx=32, pady=16, side="bottom")

    # ---- the two cards scroll when the window is shorter than them (laptops); the buttons above stay pinned
    area = trainer_ui.ScrollArea(root, skin).pack(fill="both", expand=True)

    # ---- camera card ----
    cam_card = skin.add(tk.Frame(area.inner, highlightthickness=1), bg="card", highlightbackground="border")
    cam_card.pack(fill="x", padx=32, pady=(0, 10))
    row = skin.add(tk.Frame(cam_card), bg="card")
    row.pack(fill="x", padx=16, pady=(12, 4))
    skin.add(tk.Label(row, text="Camera", font=("Segoe UI", 12, "bold")), bg="card", fg="fg").pack(side="left")
    cam_var = tk.StringVar()
    cam_box = ttk.Combobox(row, textvariable=cam_var, state="readonly", width=34, font=("Segoe UI", 11))
    cam_box.pack(side="left", padx=12)
    refresh_btn = FlatButton(row, skin, "Refresh", lambda: refresh_cameras())
    refresh_btn.pack(side="left")
    preview = skin.add(tk.Label(cam_card, text="", font=("Segoe UI", 11), compound="center", width=64, height=15),
                       bg="card_hi", fg="muted")
    preview.pack(padx=16, pady=6)
    cam_status = skin.add(tk.Label(cam_card, text="", font=("Segoe UI", 10), anchor="w"), bg="card", fg="muted")
    cam_status.pack(fill="x", padx=16, pady=(0, 12))

    # ---- microphone card ----
    mic_card = skin.add(tk.Frame(area.inner, highlightthickness=1), bg="card", highlightbackground="border")
    mic_card.pack(fill="x", padx=32, pady=(0, 10))
    row2 = skin.add(tk.Frame(mic_card), bg="card")
    row2.pack(fill="x", padx=16, pady=(12, 4))
    skin.add(tk.Label(row2, text="Microphone", font=("Segoe UI", 12, "bold")), bg="card", fg="fg").pack(side="left")
    mic_var = tk.StringVar()
    mic_box = ttk.Combobox(row2, textvariable=mic_var, state="readonly", width=50, font=("Segoe UI", 11))
    mic_box.pack(side="left", padx=12)
    meter_canvas = tk.Canvas(mic_card, height=18, highlightthickness=0, bd=0)
    meter_canvas.pack(fill="x", padx=16, pady=6)
    mic_status = skin.add(tk.Label(mic_card, text="", font=("Segoe UI", 10), anchor="w", justify="left",
                                   wraplength=800), bg="card", fg="muted")
    mic_status.pack(fill="x", padx=16, pady=(0, 6))

    # ---- whistle test: the real whistle detector on the chosen microphone
    probe = devices.WhistleProbe()
    wrow = skin.add(tk.Frame(mic_card), bg="card")
    wrow.pack(fill="x", padx=16, pady=(0, 6))
    whistle_status = skin.add(tk.Label(wrow, text="Whistle test: press the button, then blow your whistle.",
                                       font=("Segoe UI", 10), anchor="w", justify="left", wraplength=560),
                              bg="card", fg="muted")

    def toggle_whistle_test():
        if probe.running:
            probe.stop()
            whistle_btn.set_text("Start whistle test")
            start_selected_mic()                       # give the microphone back to the level meter
            whistle_status.configure(text=f"Whistle test stopped. {probe.count} whistle(s) detected.", fg=palette()["muted"])
            return
        i = mic_box.current()
        if i < 0 or i >= len(st["mics"]):
            whistle_status.configure(text="Choose a microphone first.", fg=palette()["amber"])
            return
        meter.stop()                                   # the detector opens the microphone itself
        probe.start(st["mics"][i]["index"])
        if probe.error:
            whistle_status.configure(
                text=f"The whistle detector could not start ({probe.error}). In the match simulation you can press W "
                     "instead of blowing the whistle.", fg=palette()["amber"])
            start_selected_mic()
            return
        whistle_btn.set_text("Stop whistle test")
        whistle_status.configure(text="Listening... blow your whistle now (about 20 to 50 cm from the microphone).",
                                 fg=palette()["blue"])

    whistle_btn = FlatButton(wrow, skin, "Start whistle test", toggle_whistle_test)
    whistle_btn.pack(side="left", padx=(0, 12))
    whistle_status.pack(side="left", fill="x")

    # ------------------------------------------------------------------ camera logic
    def close_camera():
        if st["cap"] is not None:
            try:
                st["cap"].release()
            except Exception:
                pass
            st["cap"] = None

    def open_selected_camera(_evt=None):
        close_camera()
        i = cam_box.current()
        if i < 0 or i >= len(st["cams"]):
            return
        idx = st["cams"][i]["index"]
        cap = devices.open_camera_index(idx)
        st["cap"] = cap
        if cap is None:
            cam_status.configure(text=f"Camera {idx} could not be opened right now.", fg=palette()["amber"])
        else:
            cam_status.configure(text=f"Camera {idx} works. Move your hand to check the picture.", fg=palette()["green"])

    def refresh_cameras():
        close_camera()
        preview.configure(image="", text="Scanning cameras... this can take a few seconds")
        st["photo"] = None
        cam_status.configure(text="", fg=palette()["muted"])
        root.update()
        st["cams"] = devices.list_cameras()
        if not st["cams"]:
            cam_box.configure(values=[])
            cam_var.set("")
            preview.configure(image="", text="No camera found")
            cam_status.configure(
                text="No camera was found. Plug it in (or start Camo), close other programs that use the camera "
                     "(Zoom, the Camera app), then press Refresh.", fg=palette()["amber"])
            return
        names = [f"Camera {c['index']}" + (f"  ({c['width']}x{c['height']})" if c["width"] else "") for c in st["cams"]]
        cam_box.configure(values=names)
        want = saved.get("camera_index")
        pos = next((k for k, c in enumerate(st["cams"]) if c["index"] == want), 0)
        cam_box.current(pos)
        preview.configure(text="")
        open_selected_camera()

    cam_box.bind("<<ComboboxSelected>>", open_selected_camera)

    # ------------------------------------------------------------------ microphone logic
    def start_selected_mic(_evt=None):
        meter.stop()
        i = mic_box.current()
        if i < 0 or i >= len(st["mics"]):
            return
        meter.start(st["mics"][i]["index"])
        if meter.error:
            mic_status.configure(text=f"This microphone could not be opened ({meter.error}). "
                                      "Try another entry in the list.", fg=palette()["amber"])
        else:
            mic_status.configure(text="Clap, speak or blow your whistle. The bar should move.", fg=palette()["muted"])

    def load_mics():
        st["mics"], err = devices.list_microphones()
        if not st["mics"]:
            mic_box.configure(values=[])
            mic_var.set("")
            why = f" ({err})" if err else ""
            mic_status.configure(
                text="No input microphone was found" + why + ". You can still use every mode. In the match simulation "
                     "press W instead of blowing the whistle.", fg=palette()["amber"])
            return
        names = [f"{m['name']}  [{m['hostapi']}]  #{m['index']}" + ("  (default)" if m["default"] else "")
                 for m in st["mics"]]
        mic_box.configure(values=names)
        want = saved.get("mic_index")
        # find the saved mic by name first: device numbers change when a phone or USB mic is reconnected
        pos = next((k for k, m in enumerate(st["mics"]) if saved.get("mic_name") and m["name"] == saved["mic_name"]
                    and m["hostapi"] == saved.get("mic_hostapi")), None)
        if pos is None:
            pos = next((k for k, m in enumerate(st["mics"]) if m["index"] == want), None)
        if pos is None:
            pos = next((k for k, m in enumerate(st["mics"]) if m["default"]), 0)
        mic_box.current(pos)
        start_selected_mic()

    mic_box.bind("<<ComboboxSelected>>", start_selected_mic)

    # ------------------------------------------------------------------ live loop
    def tick():
        p = palette()
        cap = st["cap"]
        if cap is not None:
            try:
                ok, frame = cap.read()
            except Exception:
                ok = False
            if ok:
                if mirror:
                    frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                s = min(520 / w, 292 / h)
                small = cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))))
                good, buf = cv2.imencode(".png", small, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                if good:
                    try:
                        st["photo"] = tk.PhotoImage(data=base64.b64encode(buf.tobytes()))
                        preview.configure(image=st["photo"], text="", width=small.shape[1], height=small.shape[0])
                    except tk.TclError:
                        pass
        # microphone level bar
        lvl = meter.level()
        meter_canvas.configure(bg=p["card_hi"])
        meter_canvas.delete("all")
        width = meter_canvas.winfo_width() or 800
        meter_canvas.create_rectangle(0, 0, int(width * lvl), 18, fill=p["green"] if lvl >= meter.HEARD_LEVEL else p["blue"],
                                      width=0)
        if meter.heard and not meter.error and st["mics"]:
            mic_status.configure(text="Microphone works: it is picking up sound.", fg=p["green"])
        if probe.running:
            if probe.count:
                ago = time.time() - probe.last_time
                conf = "" if probe.last_conf is None else f", confidence {float(probe.last_conf):.2f}"
                whistle_status.configure(text=f"WHISTLE DETECTED ({probe.count} so far, last {ago:.0f} s ago{conf}).",
                                         fg=p["green"])
        st["job"] = root.after(120, tick)

    def finish(save):
        if st.get("job"):
            try:
                root.after_cancel(st["job"])
            except tk.TclError:
                pass
        probe.stop()
        meter.stop()
        close_camera()
        if save:
            ci = cam_box.current()
            mi = mic_box.current()
            if 0 <= ci < len(st["cams"]):
                result["camera"] = st["cams"][ci]["index"]
            if 0 <= mi < len(st["mics"]):
                result["mic"] = st["mics"][mi]["index"]
            patch = {"setup_done": True}
            if result["camera"] is not None:
                patch["camera_index"] = result["camera"]
            if result["mic"] is not None:
                patch["mic_index"] = result["mic"]
                patch["mic_name"] = st["mics"][mi]["name"]          # so it can be found again after a reconnect
                patch["mic_hostapi"] = st["mics"][mi]["hostapi"]
            trainer_ui.write_settings(patch)
            result["saved"] = True
        else:
            trainer_ui.write_settings({"setup_done": True})
        root.destroy()

    FlatButton(bar, skin, "Save and continue", lambda: finish(True), kind="primary", big=True).pack(side="right")
    FlatButton(bar, skin, "Skip for now", lambda: finish(False)).pack(side="right", padx=10)
    root.protocol("WM_DELETE_WINDOW", lambda: finish(False))

    skin.apply()
    root.update()
    refresh_cameras()
    load_mics()
    tick()
    root.mainloop()
    meter.stop()
    close_camera()
    return result


if __name__ == "__main__":
    print(run_device_setup())