"""
scoreboard_gui.py

Simple desktop scoreboard display for the volleyball officiating
system. Runs as its own Tkinter window, separate from the camera/
skeleton-debug window (live_auto_inference.py) -- this is the
spectator/scorekeeper-facing view, like a real volleyball scoreboard.

No .exe or .apk needed -- runs directly from a cloned repo with:
    python scoreboard_gui.py
"""

import tkinter as tk


class ScoreboardGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Volleyball Officiating Scoreboard")
        self.root.configure(bg="#0b1e0f")
        self.root.geometry("640x360")

        self.set_label = tk.Label(
            self.root, text="SET 1", font=("Helvetica", 24, "bold"),
            fg="white", bg="#0b1e0f"
        )
        self.set_label.pack(pady=(20, 10))

        score_frame = tk.Frame(self.root, bg="#0b1e0f")
        score_frame.pack(pady=10)

        self.left_score_label = tk.Label(
            score_frame, text="LEFT\n0", font=("Helvetica", 48, "bold"),
            fg="#4fc3f7", bg="#0b1e0f", width=6
        )
        self.left_score_label.grid(row=0, column=0, padx=30)

        self.right_score_label = tk.Label(
            score_frame, text="RIGHT\n0", font=("Helvetica", 48, "bold"),
            fg="#ff8a65", bg="#0b1e0f", width=6
        )
        self.right_score_label.grid(row=0, column=1, padx=30)

        self.gesture_label = tk.Label(
            self.root, text="Last gesture: -", font=("Helvetica", 16),
            fg="white", bg="#0b1e0f"
        )
        self.gesture_label.pack(pady=(20, 5))

        self.whistle_label = tk.Label(
            self.root, text="Whistle: waiting...", font=("Helvetica", 16),
            fg="#ffd54f", bg="#0b1e0f"
        )
        self.whistle_label.pack(pady=5)

    def update_display(self, left_score=None, right_score=None,
                        gesture_text=None, whistle_status=None, set_number=None):
        """Call this whenever any of these values change."""
        if left_score is not None:
            self.left_score_label.config(text=f"LEFT\n{left_score}")
        if right_score is not None:
            self.right_score_label.config(text=f"RIGHT\n{right_score}")
        if gesture_text is not None:
            self.gesture_label.config(text=f"Last gesture: {gesture_text}")
        if whistle_status is not None:
            self.whistle_label.config(text=f"Whistle: {whistle_status}")
        if set_number is not None:
            self.set_label.config(text=f"SET {set_number}")

    def tick(self):
        """
        Call this once per loop iteration INSTEAD OF root.mainloop().
        Lets this window update alongside your camera/inference loop
        in the same process, without threads or blocking either one.
        """
        self.root.update_idletasks()
        self.root.update()

    def close(self):
        self.root.destroy()


if __name__ == "__main__":
    # Standalone demo -- simulates a match without a real camera, so
    # you can see the scoreboard working on its own.
    import time
    from decision_engine import DecisionEngine

    engine = DecisionEngine()
    gui = ScoreboardGUI()

    demo_events = [
        ("whistle", None),
        ("gesture", "team_to_serve_right"),
        ("gesture", "ball_out"),
        ("whistle", None),
        ("gesture", "team_to_serve_left"),
        ("gesture", "double_contact"),
    ]

    for kind, label in demo_events:
        if kind == "whistle":
            result = engine.on_whistle_detected()
            gui.update_display(whistle_status="detected")
        else:
            result = engine.on_gesture_detected(label)
            if result["event"] == "point_awarded":
                gui.update_display(
                    left_score=engine.score["left"],
                    right_score=engine.score["right"],
                    gesture_text=label,
                    whistle_status="waiting...",
                )
            elif result["event"] == "reason_attached":
                gui.update_display(gesture_text=f"{label} ({result['side']})")

        print(result)
        for _ in range(30):
            gui.tick()
            time.sleep(1 / 30)

    time.sleep(3)
    gui.close()