"""
decision_engine.py

Validates and sequences referee calls for automated scorekeeping.

Sequence per point:
  1. Whistle detected
  2. team_to_serve_left/right gesture within TEMPORAL_WINDOW of the whistle
     -> awards a point to that side, sets them as next server
  3. (optional) a fault gesture (ball_out, double_contact,
     service_authorization_left/right, end_of_set) within
     TEMPORAL_WINDOW of the scoring gesture -> attached as the
     reason/context for that point, does NOT change score itself
"""

import time

TEMPORAL_WINDOW = 2.0  # seconds -- adjust based on real referee timing

SCORING_GESTURES = {"team_to_serve_left", "team_to_serve_right"}
FAULT_REASON_GESTURES = {"ball_out", "double_contact",
                          "service_authorization_left", "service_authorization_right",
                          "end_of_set"}


class DecisionEngine:
    def __init__(self):
        self.score = {"left": 0, "right": 0}
        self.server = None
        self.last_whistle_time = None
        self.last_point_time = None
        self.last_point_side = None
        self.last_reason = None

    def on_whistle_detected(self, timestamp=None):
        self.last_whistle_time = timestamp or time.time()

    def on_gesture_detected(self, label, timestamp=None):
        timestamp = timestamp or time.time()

        if label in SCORING_GESTURES:
            if self.last_whistle_time is None or (timestamp - self.last_whistle_time) > TEMPORAL_WINDOW:
                return {"event": "ignored", "reason": "no recent whistle"}

            side = "right" if label == "team_to_serve_right" else "left"
            self.score[side] += 1
            self.server = side
            self.last_point_time = timestamp
            self.last_point_side = side
            self.last_reason = None
            return {"event": "point_awarded", "side": side, "score": dict(self.score)}

        elif label in FAULT_REASON_GESTURES:
            if self.last_point_time is None or (timestamp - self.last_point_time) > TEMPORAL_WINDOW:
                return {"event": "ignored", "reason": "no recent point to attach reason to"}

            self.last_reason = label
            return {"event": "reason_attached", "reason": label, "side": self.last_point_side}

        return {"event": "ignored", "reason": "unrecognized label"}