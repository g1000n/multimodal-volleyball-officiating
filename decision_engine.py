"""
decision_engine.py

Validates and sequences referee calls for automated scorekeeping,
following the real FIVB-confirmed sequence:
  1. Whistle
  2. team_to_serve_left/right -> awards the point, sets next server
  3. (optional) a fault/context gesture -> attached as the reason,
     does NOT change score itself
  4. A real PAUSE before the next cycle -- confirmed directly from the
     FIVB Referereeing Guidelines: "whistle, indicate the nature of the
     fault, indicate the player at fault (if necessary), PAUSE, then
     follow the 1st Referee's signal for side to serve next." This
     project's live testing observed sustained (multi-second) spurious
     spikes of OTHER classes immediately after a real gesture finishes
     (e.g. double_contact spiking right after ball_out, service_
     authorization_right spiking right after team_to_serve_right) --
     this version enforces that real pause at the decision layer,
     which structurally suppresses exactly that failure mode without
     needing to fix the underlying model/data at all.

--------------------------------------------------------------------
NEW (this version), on top of the previous win-condition/whistle fixes:

4. SETTLE WINDOW AFTER ANY COMMIT: after ANY accepted event (point
   awarded OR reason attached), further gesture detections are ignored
   for SETTLE_WINDOW_SECONDS -- mirroring the real, FIVB-confirmed
   pause referees take before the next signal. A fresh whistle is
   NEVER blocked by this (whistles always get through), since that's
   the deliberate start of the next cycle, not noise from the gesture
   that just finished.

5. ONE FAULT-REASON PER POINT: previously, a second fault-reason
   gesture (even a spurious one) could silently OVERWRITE the first,
   correct reason for the same point. Now, once a reason is attached,
   further fault-reason attempts for the SAME point are rejected
   (ignored) rather than overwriting -- this is what SETTLE_WINDOW
   mostly prevents anyway, but this is a second, independent guard in
   case a spurious spike happens to land just outside the settle
   window but still within TEMPORAL_WINDOW of the same point.
--------------------------------------------------------------------
"""

import time

TEMPORAL_WINDOW = 10.0  # seconds -- CHANGED from 5.0. Live testing showed even
# 5.0s was too tight: a genuinely natural (not rushed) team_to_serve -> ball_out
# attempt measured 5.752s total and got rejected. Real pipeline latency alone
# (SETTLE_WINDOW_SECONDS 1.5s + ~1.5s minimum streak-build time) plus normal
# human repositioning/reaction time easily approaches 4-6s even when nothing
# is going wrong -- this value now has real margin above that, rather than a
# tight fit that keeps failing on ordinary pacing.

# How long to ignore further gesture detections after ANY accepted
# commit (point or reason). Grounded in the FIVB guideline's confirmed
# "pause" before the next signal. Whistles are NEVER blocked by this.
SETTLE_WINDOW_SECONDS = 1.5

SCORING_GESTURES = {"team_to_serve_left", "team_to_serve_right"}
FAULT_REASON_GESTURES = {"ball_out", "double_contact",
                          "service_authorization_left", "service_authorization_right"}
# end_of_set handled separately below since it needs the win-condition check

WIN_SCORE = 25          # DEFAULT for the real game -- can be overridden per-instance (see __init__)
WIN_BY_MARGIN = 2        # win by 2, no cap (deuce continues past WIN_SCORE)


class DecisionEngine:
    def __init__(self, win_score=WIN_SCORE, win_by_margin=WIN_BY_MARGIN):
        # NEW: win_score/win_by_margin are now per-instance, not fixed
        # module constants -- lets a shorter practice simulation (e.g.
        # first to 7) share this exact same engine/logic as the real
        # 25-point game, instead of needing a separate copy of the file.
        self.win_score = win_score
        self.win_by_margin = win_by_margin
        self.score = {"left": 0, "right": 0}
        self.server = None
        self.last_whistle_time = None
        self.last_point_time = None
        self.last_point_side = None
        self.last_reason = None
        self.set_over = False
        self.last_settle_start_time = None  # NEW: when the current settle window began

    def manual_override_score(self, side, delta):
        """
        Lets a human operator correct the score directly if the
        automated system gets something wrong live -- a necessary
        safety net for any real deployed officiating aid, not just a
        demo. Re-checks the win condition after the edit so a manual
        correction can also end (or un-end) the set correctly.
        """
        self.score[side] = max(0, self.score[side] + delta)
        self.set_over = self._check_set_over()
        return dict(self.score)

    def manual_clear_reason(self):
        """Lets a human operator clear an incorrectly-attached fault
        reason for the current point, so a new one can be attached."""
        self.last_reason = None

    def on_whistle_detected(self, timestamp=None):
        # Whistles are never blocked by the settle window -- they're
        # the deliberate start of the next cycle, not tail-end noise.
        self.last_whistle_time = timestamp or time.time()

    def _check_set_over(self):
        """Returns True if the current score satisfies the win condition."""
        left, right = self.score["left"], self.score["right"]
        if left >= self.win_score and (left - right) >= self.win_by_margin:
            return True
        if right >= self.win_score and (right - left) >= self.win_by_margin:
            return True
        return False

    def _in_settle_window(self, timestamp):
        if self.last_settle_start_time is None:
            return False
        return (timestamp - self.last_settle_start_time) < SETTLE_WINDOW_SECONDS

    def reset_for_new_set(self):
        """Call this to start a fresh set after the current one ends."""
        self.score = {"left": 0, "right": 0}
        self.server = None
        self.last_whistle_time = None
        self.last_point_time = None
        self.last_point_side = None
        self.last_reason = None
        self.set_over = False
        self.last_settle_start_time = None

    def on_gesture_detected(self, label, timestamp=None):
        timestamp = timestamp or time.time()

        if self.set_over:
            return {"event": "ignored", "reason": "set already over -- call reset_for_new_set() to continue"}

        # NEW: settle window check -- applies to ALL gesture types (not
        # whistles). This is the direct fix for the observed sustained
        # tail-end spikes: a spurious detection landing right after a
        # real, just-accepted commit gets rejected here before it can
        # do anything, exactly matching the real refereeing pause.
        if self._in_settle_window(timestamp):
            remaining = SETTLE_WINDOW_SECONDS - (timestamp - self.last_settle_start_time)
            return {"event": "ignored",
                    "reason": f"settle window active ({remaining:.1f}s remaining) -- "
                              f"suppressing likely tail-end noise from the last gesture"}

        if label in SCORING_GESTURES:
            if self.last_whistle_time is None or (timestamp - self.last_whistle_time) > TEMPORAL_WINDOW:
                return {"event": "ignored", "reason": "no recent whistle"}

            side = "right" if label == "team_to_serve_right" else "left"
            self.score[side] += 1
            self.server = side
            self.last_point_time = timestamp
            self.last_point_side = side
            self.last_reason = None
            self.last_settle_start_time = timestamp  # NEW: start the settle window

            self.last_whistle_time = None

            self.set_over = self._check_set_over()

            return {
                "event": "point_awarded",
                "side": side,
                "score": dict(self.score),
                "set_over": self.set_over,
            }

        elif label == "end_of_set":
            if not self._check_set_over():
                return {
                    "event": "ignored",
                    "reason": f"end_of_set predicted but score ({self.score['left']}-{self.score['right']}) "
                              f"doesn't satisfy the win condition -- likely a misclassification, discarding",
                }

            if self.last_point_time is None or (timestamp - self.last_point_time) > TEMPORAL_WINDOW:
                return {"event": "ignored", "reason": "no recent point to attach end_of_set reason to"}

            # NEW: one reason per point -- don't let a second (possibly
            # spurious) reason gesture overwrite an already-attached one.
            if self.last_reason is not None:
                return {"event": "ignored", "reason": f"a reason ('{self.last_reason}') was already attached to this point"}

            self.last_reason = label
            self.last_settle_start_time = timestamp  # NEW
            self.set_over = True
            return {"event": "reason_attached", "reason": label, "side": self.last_point_side, "set_over": True}

        elif label in FAULT_REASON_GESTURES:
            if self.last_point_time is None or (timestamp - self.last_point_time) > TEMPORAL_WINDOW:
                return {"event": "ignored", "reason": "no recent point to attach reason to"}

            # NEW: one reason per point -- see above.
            if self.last_reason is not None:
                return {"event": "ignored", "reason": f"a reason ('{self.last_reason}') was already attached to this point"}

            self.last_reason = label
            self.last_settle_start_time = timestamp  # NEW
            return {"event": "reason_attached", "reason": label, "side": self.last_point_side}

        return {"event": "ignored", "reason": "unrecognized label"}