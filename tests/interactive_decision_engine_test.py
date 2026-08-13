"""
interactive_decision_engine_test.py

Isolated, interactive decision_engine.py tester. No camera, no trained
model, no audio -- just you, typing commands one at a time, starting
fresh at 0-0, watching exactly what decision_engine.py does in response.
Uses REAL wall-clock time (not simulated timestamps), so settle windows,
cooldowns, and the temporal window all behave exactly like they would
during a real game -- if you type too fast, you'll see the same
rejections you'd see live.

Run:
    python interactive_decision_engine_test.py

COMMANDS:
    w                          -- whistle
    team_to_serve_left         -- (or _right)
    ball_out                   -- (or double_contact, service_authorization_left,
                                    service_authorization_right, end_of_set)
    score left +1              -- manual score adjustment (also: score left -1,
                                    score right +1, score right -1)
    clear                      -- manually clear the current point's attached reason
    status                     -- show current score/state without sending an event
    reset                      -- start a completely fresh engine at 0-0
    help                       -- show this list again
    quit / exit                -- stop
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
from decision_engine import DecisionEngine

GESTURE_LABELS = {
    "team_to_serve_left", "team_to_serve_right",
    "ball_out", "double_contact",
    "service_authorization_left", "service_authorization_right",
    "end_of_set",
}

HELP_TEXT = """
COMMANDS:
  w                     -- whistle
  <gesture label>       -- team_to_serve_left, team_to_serve_right, ball_out,
                            double_contact, service_authorization_left,
                            service_authorization_right, end_of_set
  score left +1         -- manual score adjustment (left/right, +1/-1)
  score right -1
  clear                 -- manually clear the current point's reason
  status                -- show current score/state
  reset                 -- start a completely fresh engine at 0-0
  help                  -- show this again
  quit / exit           -- stop
"""


def print_status(engine):
    print(f"  SCORE: LEFT {engine.score['left']} - {engine.score['right']} RIGHT   "
          f"(first to {engine.win_score}, win by {engine.win_by_margin})   "
          f"set_over={engine.set_over}   last_reason={engine.last_reason}")


def main():
    engine = DecisionEngine()  # real defaults: win_score=25, win_by_margin=2
    print("Isolated decision_engine tester -- starting fresh at 0-0.")
    print(HELP_TEXT)
    print_status(engine)

    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue

        cmd = raw.lower()

        if cmd in ("quit", "exit"):
            print("Exiting.")
            break

        elif cmd == "help":
            print(HELP_TEXT)

        elif cmd == "status":
            print_status(engine)

        elif cmd == "reset":
            engine = DecisionEngine()
            print("Engine reset -- fresh start at 0-0.")
            print_status(engine)

        elif cmd == "clear":
            engine.manual_clear_reason()
            print("Cleared the current point's attached reason.")
            print_status(engine)

        elif cmd == "w" or cmd == "whistle":
            now = time.time()
            engine.on_whistle_detected(now)
            print(f"  WHISTLE at t={now:.3f}")
            print_status(engine)

        elif cmd.startswith("score "):
            parts = cmd.split()
            if len(parts) != 3 or parts[1] not in ("left", "right") or parts[2] not in ("+1", "-1"):
                print("  Usage: score left +1   |   score right -1")
                continue
            side = parts[1]
            delta = 1 if parts[2] == "+1" else -1
            engine.manual_override_score(side, delta)
            print(f"  MANUAL: {side} score {'+1' if delta > 0 else '-1'}")
            print_status(engine)

        elif raw in GESTURE_LABELS:
            now = time.time()
            result = engine.on_gesture_detected(raw, now)
            print(f"  t={now:.3f}  {raw} -> {result}")
            print_status(engine)

        else:
            print(f"  Unrecognized command: '{raw}'. Type 'help' for the command list.")


if __name__ == "__main__":
    main()