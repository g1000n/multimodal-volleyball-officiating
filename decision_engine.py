"""
decision_engine.py

Validates and sequences referee calls for automated scorekeeping,
following the REAL confirmed FIVB sequence (confirmed directly against
the official FIVB description, and against this project's own live
testing):

  1. Whistle #1 (start of play)
  2. service_authorization_left/right (beckon -- signals direction of
     serve). Informational for scoring purposes -- does NOT award a
     point, does NOT need to match the eventual scoring side to be
     "correct" gesture-recognition-wise, but its side IS recorded so
     it can be compared against the eventual team_to_serve side for
     accuracy reporting (see `authorization_side` /
     `authorization_match` in the point_awarded event below).
  3. Rally (no gesture expected -- players playing)
  4. Whistle #2 (end of play -- fault or end of rally)
  5. team_to_serve_left/right (point/team-to-serve signal) -> THIS is
     what awards the point and sets the next server.
  6. (optional) fault/nature-of-fault gesture (ball_out, double_contact)
     -> attached as the reason for the point just awarded. Does NOT
     change score itself.
  7. A real PAUSE before the next cycle -- confirmed directly from the
     FIVB guidelines' "pause" before the next signal. This project's
     live testing observed sustained (multi-second) spurious spikes of
     OTHER classes immediately after a real gesture finishes -- the
     settle window below enforces that real pause at the decision
     layer, suppressing that failure mode without needing to fix the
     underlying model/data.

--------------------------------------------------------------------
CHANGED (previous version): service_authorization now has its OWN
bucket (SERVICE_AUTHORIZATION_GESTURES), separate from
FAULT_REASON_GESTURES, and is validated against last_whistle_time
(whistle #1), exactly like team_to_serve is validated against its own
whistle (whistle #2) -- NOT against last_point_time (the old, buggy
anchor). Accepting a service_authorization gesture CONSUMES that
whistle, forcing a genuine SECOND, distinct whistle before
team_to_serve can be accepted -- structurally enforcing the real
two-whistle cycle instead of letting one whistle silently authorize
both steps. The authorized side is tracked and compared against the
eventual scoring side, returned as `authorization_side` /
`authorization_match` in the point_awarded event -- purely
informational, doesn't gate or block scoring.

CHANGED (this version): LEFT/RIGHT FIX. The gesture class names
(team_to_serve_left, service_authorization_left, etc.) are tied to the
REFEREE's own left/right -- how the training data was filmed and
labeled. That's confusing to watch live: the referee's left-arm
gesture would score/display as "left," which is actually the wrong
side from the audience/court-facing perspective everyone is actually
watching from. GESTURE_TO_SCORE_SIDE below is the single place that
translates a recognized gesture's label into the side that actually
gets scored -- deliberately flipped (referee-left -> scored-right and
vice versa) so what's on the scoreboard matches what a spectator
watching the match would expect. This does NOT touch the model, the
training data, or any class names -- those stay exactly as filmed and
labeled; only the meaning assigned to "which side scored" changes,
in this one table.
--------------------------------------------------------------------

SETTLE WINDOW: after ANY accepted event (authorization, point, or
reason), further gesture detections are ignored for
SETTLE_WINDOW_SECONDS -- mirroring the real FIVB-confirmed pause
referees take before the next signal. A fresh whistle is NEVER blocked
by this, since a whistle is the deliberate start of the next step, not
tail-end noise from the gesture that just finished.

ONE FAULT-REASON PER POINT: once a reason is attached to a point,
further fault-reason attempts for the SAME point are rejected
(ignored) rather than overwriting.
"""

import time

TEMPORAL_WINDOW = 10.0  # seconds -- how long a whistle stays "valid" for the
# gesture that's supposed to follow it (service_authorization after whistle #1,
# team_to_serve after whistle #2). Live testing showed even 5.0s was too tight: a
# genuinely natural (not rushed) attempt measured 5.752s total and got rejected.
# Real pipeline latency alone (SETTLE_WINDOW_SECONDS 1.5s + ~1.5s minimum
# streak-build time) plus normal human repositioning/reaction time easily
# approaches 4-6s even when nothing is going wrong.

# NEW: separate, SHORTER window for how long a just-awarded point stays open for
# a fault-reason gesture (ball_out, double_contact, ball_in) or end_of_set to
# attach to. Previously this reused TEMPORAL_WINDOW (10.0s), which was
# deliberately widened for whistle-to-gesture timing tolerance -- but a real
# reason gesture happens right after the point, not 8-9 seconds later. Reusing
# the wide window here unnecessarily extended how long a spurious sustained
# gesture (e.g. resting arms on a rail during idle time) could get incorrectly
# attached as that point's reason. Shrinks the exposure window without touching
# the whistle-timing tolerance TEMPORAL_WINDOW was calibrated for.
REASON_ATTACH_WINDOW = 4.0

# How long to ignore further gesture detections after ANY accepted
# commit (authorization, point, or reason). Grounded in the FIVB
# guideline's confirmed "pause" before the next signal. Whistles are
# NEVER blocked by this.
SETTLE_WINDOW_SECONDS = 1.5

SERVICE_AUTHORIZATION_GESTURES = {"service_authorization_left", "service_authorization_right"}
SCORING_GESTURES = {"team_to_serve_left", "team_to_serve_right"}
# FIX: ball_in is a real, trained, deployed model class now -- without it
# here, a genuine ball_in recognition fell through every branch of
# on_gesture_detected() and was silently rejected as "unrecognized label".
FAULT_REASON_GESTURES = {"ball_out", "double_contact", "ball_in"}
# end_of_set handled separately below since it needs the win-condition check

# LEFT/RIGHT FIX: the gesture class names are tied to the REFEREE's own
# left/right (how the data was filmed and labeled). This table
# deliberately flips that into the audience/court-facing side that
# actually gets scored and displayed -- referee-left gestures score as
# "right," referee-right gestures score as "left." This is the ONLY
# place that assigns scoring-side meaning; everything else (the model,
# the training data, the class names themselves) is untouched.
GESTURE_TO_SCORE_SIDE = {
    "team_to_serve_left": "right",
    "team_to_serve_right": "left",
    "service_authorization_left": "right",
    "service_authorization_right": "left",
}

WIN_SCORE = 25          # DEFAULT for the real game -- can be overridden per-instance (see __init__)
WIN_BY_MARGIN = 2        # win by 2, no cap (deuce continues past WIN_SCORE)


class DecisionEngine:
    def __init__(self, win_score=WIN_SCORE, win_by_margin=WIN_BY_MARGIN):
        # win_score/win_by_margin are per-instance, not fixed module
        # constants -- lets a shorter practice simulation (e.g. first
        # to 7) share this exact same engine/logic as the real
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
        self.last_settle_start_time = None

        # Authorization-phase tracking
        self.last_authorization_side = None
        self.last_authorization_time = None

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
        self.last_whistle_time = timestamp if timestamp is not None else time.time()

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
        self.last_authorization_side = None
        self.last_authorization_time = None

    def on_gesture_detected(self, label, timestamp=None):
        timestamp = timestamp if timestamp is not None else time.time()

        if self.set_over:
            return {"event": "ignored", "reason": "set already over -- call reset_for_new_set() to continue"}

        # Settle window check -- applies to ALL gesture types (not
        # whistles). Direct fix for observed sustained tail-end spikes:
        # a spurious detection landing right after a real, just-accepted
        # commit gets rejected here before it can do anything, matching
        # the real refereeing pause.
        if self._in_settle_window(timestamp):
            remaining = SETTLE_WINDOW_SECONDS - (timestamp - self.last_settle_start_time)
            return {"event": "ignored",
                    "reason": f"settle window active ({remaining:.1f}s remaining) -- "
                              f"suppressing likely tail-end noise from the last gesture"}

        # --------------------------------------------------------
        # PHASE 1: service_authorization (whistle #1 -> beckon)
        # --------------------------------------------------------
        if label in SERVICE_AUTHORIZATION_GESTURES:
            if self.last_whistle_time is None or (timestamp - self.last_whistle_time) > TEMPORAL_WINDOW:
                return {"event": "ignored", "reason": "no recent whistle for service authorization"}

            side = GESTURE_TO_SCORE_SIDE[label]  # LEFT/RIGHT FIX -- see table above
            self.last_authorization_side = side
            self.last_authorization_time = timestamp
            self.last_settle_start_time = timestamp

            # Consume this whistle -- forces a genuine SECOND, distinct
            # whistle before team_to_serve can be accepted, enforcing
            # the real two-whistle cycle instead of one whistle silently
            # covering both steps.
            self.last_whistle_time = None

            return {"event": "authorization_acknowledged", "side": side}

        # --------------------------------------------------------
        # PHASE 2: team_to_serve (whistle #2 -> point/team-to-serve signal)
        # --------------------------------------------------------
        if label in SCORING_GESTURES:
            if self.last_whistle_time is None or (timestamp - self.last_whistle_time) > TEMPORAL_WINDOW:
                return {"event": "ignored", "reason": "no recent whistle"}

            side = GESTURE_TO_SCORE_SIDE[label]  # LEFT/RIGHT FIX -- see table above

            # Informational only -- compare the beckoned side against the
            # side that ultimately scored. None if no (recent) authorization
            # was ever recorded, so this doesn't get compared against a
            # stale authorization from several points ago.
            authorization_match = None
            if (self.last_authorization_time is not None
                    and (timestamp - self.last_authorization_time) <= TEMPORAL_WINDOW):
                authorization_match = (self.last_authorization_side == side)

            self.score[side] += 1
            self.server = side
            self.last_point_time = timestamp
            self.last_point_side = side
            self.last_reason = None
            self.last_settle_start_time = timestamp

            self.last_whistle_time = None
            self.set_over = self._check_set_over()

            result = {
                "event": "point_awarded",
                "side": side,
                "score": dict(self.score),
                "set_over": self.set_over,
                "authorization_side": self.last_authorization_side,
                "authorization_match": authorization_match,
            }

            # Clear authorization state now that it's been consumed/reported
            # for this point -- next point needs its own fresh authorization.
            self.last_authorization_side = None
            self.last_authorization_time = None

            return result

        # --------------------------------------------------------
        # end_of_set -- same win-condition-gated handling as before
        # --------------------------------------------------------
        elif label == "end_of_set":
            if not self._check_set_over():
                return {
                    "event": "ignored",
                    "reason": f"end_of_set predicted but score ({self.score['left']}-{self.score['right']}) "
                              f"doesn't satisfy the win condition -- likely a misclassification, discarding",
                }

            if self.last_point_time is None or (timestamp - self.last_point_time) > REASON_ATTACH_WINDOW:
                return {"event": "ignored", "reason": "no recent point to attach end_of_set reason to"}

            if self.last_reason is not None:
                return {"event": "ignored", "reason": f"a reason ('{self.last_reason}') was already attached to this point"}

            self.last_reason = label
            self.last_settle_start_time = timestamp
            self.set_over = True
            return {"event": "reason_attached", "reason": label, "side": self.last_point_side, "set_over": True}

        # --------------------------------------------------------
        # PHASE 3 (optional): fault/nature-of-fault gesture -> reason
        # --------------------------------------------------------
        elif label in FAULT_REASON_GESTURES:
            if self.last_point_time is None or (timestamp - self.last_point_time) > REASON_ATTACH_WINDOW:
                return {"event": "ignored", "reason": "no recent point to attach reason to"}

            if self.last_reason is not None:
                return {"event": "ignored", "reason": f"a reason ('{self.last_reason}') was already attached to this point"}

            self.last_reason = label
            self.last_settle_start_time = timestamp
            return {"event": "reason_attached", "reason": label, "side": self.last_point_side}

        return {"event": "ignored", "reason": "unrecognized label"}