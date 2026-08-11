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
  6. (optional) fault/nature-of-fault gesture (ball_out, double_contact,
     ball_in) -> attached as the reason for the point just awarded.
     Does NOT change score itself.
  7. A real PAUSE before the next cycle -- confirmed directly from the
     FIVB guidelines' "pause" before the next signal. This project's
     live testing observed sustained (multi-second) spurious spikes of
     OTHER classes immediately after a real gesture finishes -- the
     settle window below enforces that real pause at the decision
     layer, suppressing that failure mode without needing to fix the
     underlying model/data.

--------------------------------------------------------------------
CHANGED (this version): AUTOMATIC SET-ENDING REMOVED. This project's
core purpose is automating POINT-ADDING -- fault/reason gestures
(ball_out, double_contact, ball_in, end_of_set) are informational only,
not the thing being validated for correctness. Previously, EVERY
team_to_serve commit auto-checked the win condition and could silently
set self.set_over = True the moment score crossed the win threshold --
this happened purely from the SCORING gesture, with no dependency on
end_of_set ever being detected. Confirmed via a real live session:
this caused scoring to freeze while a real rally was still ongoing,
because the model isn't perfect and the automatic win-condition check
doesn't know that.

Now: score is tracked with NO cap and NO automatic stop, regardless of
value. end_of_set, if detected, is logged as an informational reason
attached to the current point (same as ball_out/double_contact/ball_in)
but does NOT set self.set_over and does NOT block further scoring.
self.set_over still EXISTS as an attribute (for any future manual-only
end-of-set control, e.g. an operator explicitly ending a set), but
nothing in this file sets it automatically anymore. _check_set_over()
is kept as a plain query method (still useful for external code that
wants to know "has the win condition technically been met"), it's just
no longer wired to actually stop anything.

--------------------------------------------------------------------
PREVIOUS CHANGES (still in effect):

service_authorization has its OWN bucket (SERVICE_AUTHORIZATION_GESTURES),
separate from FAULT_REASON_GESTURES, and is validated against
last_whistle_time (whistle #1), exactly like team_to_serve is validated
against its own whistle (whistle #2) -- NOT against last_point_time (the
old, buggy anchor). Accepting a service_authorization gesture CONSUMES
that whistle, forcing a genuine SECOND, distinct whistle before
team_to_serve can be accepted -- structurally enforcing the real
two-whistle cycle instead of letting one whistle silently authorize
both steps. The authorized side is tracked and compared against the
eventual scoring side, returned as `authorization_side` /
`authorization_match` in the point_awarded event -- purely
informational, doesn't gate or block scoring.

LEFT/RIGHT FIX: the gesture class names (team_to_serve_left,
service_authorization_left, etc.) are tied to the REFEREE's own
left/right -- how the training data was filmed and labeled. That's
confusing to watch live: the referee's left-arm gesture would
score/display as "left," which is actually the wrong side from the
audience/court-facing perspective everyone is actually watching from.
GESTURE_TO_SCORE_SIDE below is the single place that translates a
recognized gesture's label into the side that actually gets scored --
deliberately flipped (referee-left -> scored-right and vice versa) so
what's on the scoreboard matches what a spectator watching the match
would expect. This does NOT touch the model, the training data, or any
class names -- those stay exactly as filmed and labeled; only the
meaning assigned to "which side scored" changes, in this one table.
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

# Separate, SHORTER window for how long a just-awarded point stays open for
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
# ball_in is a real, trained, deployed model class -- without it here, a
# genuine ball_in recognition would fall through every branch of
# on_gesture_detected() and be silently rejected as "unrecognized label".
FAULT_REASON_GESTURES = {"ball_out", "double_contact", "ball_in"}
# end_of_set handled separately below -- informational only now, see
# module docstring (no longer gates on win condition, no longer stops
# scoring).

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

WIN_SCORE = 25          # kept for reference / any future manual-only win check.
WIN_BY_MARGIN = 2        # No longer used to automatically stop scoring -- see
# module docstring. _check_set_over() still uses these if called directly.


class DecisionEngine:
    def __init__(self, win_score=WIN_SCORE, win_by_margin=WIN_BY_MARGIN):
        # win_score/win_by_margin are per-instance, not fixed module
        # constants -- kept for any future manual/optional win-condition
        # querying, even though nothing in this class auto-triggers on
        # them anymore.
        self.win_score = win_score
        self.win_by_margin = win_by_margin
        self.score = {"left": 0, "right": 0}
        self.server = None
        self.last_whistle_time = None
        self.last_point_time = None
        self.last_point_side = None
        self.last_reason = None
        self.set_over = False  # NEVER set automatically anymore -- see
        # module docstring. Left in place for potential future manual
        # end-of-set control (e.g. an operator explicitly calling it).
        self.last_settle_start_time = None

        # Authorization-phase tracking
        self.last_authorization_side = None
        self.last_authorization_time = None

    def manual_override_score(self, side, delta):
        """
        Lets a human operator correct the score directly if the
        automated system gets something wrong live -- a necessary
        safety net for any real deployed officiating aid, not just a
        demo. No longer touches self.set_over -- score can be corrected
        freely regardless of value, matching the "no automatic
        stopping" design.
        """
        self.score[side] = max(0, self.score[side] + delta)
        return dict(self.score)

    def manual_clear_reason(self):
        """Lets a human operator clear an incorrectly-attached fault
        reason for the current point, so a new one can be attached."""
        self.last_reason = None

    def manual_end_set(self):
        """
        NEW: the only remaining way self.set_over becomes True -- an
        explicit, deliberate call, e.g. a human operator pressing an
        "end set" control. Nothing in on_gesture_detected() calls this
        automatically anymore.
        """
        self.set_over = True
        return self.set_over

    def on_whistle_detected(self, timestamp=None):
        # Whistles are never blocked by the settle window -- they're
        # the deliberate start of the next step, not tail-end noise.
        self.last_whistle_time = timestamp if timestamp is not None else time.time()

    def _check_set_over(self):
        """
        Returns True if the current score WOULD satisfy a standard win
        condition (score >= win_score, margin >= win_by_margin). This
        is a plain query only -- nothing in this class calls it to
        automatically set self.set_over anymore. Kept available for
        any external code (e.g. a UI) that wants to show "win condition
        met" as an informational hint without it forcing a stop.
        """
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
            # Only reachable now via manual_end_set() -- nothing in
            # this method sets self.set_over automatically anymore.
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
            # CHANGED: no longer auto-checks/sets self.set_over here --
            # see module docstring. Score just keeps counting, no cap.

            result = {
                "event": "point_awarded",
                "side": side,
                "score": dict(self.score),
                "set_over": self.set_over,  # will simply stay False unless manual_end_set() was called
                "authorization_side": self.last_authorization_side,
                "authorization_match": authorization_match,
            }

            # Clear authorization state now that it's been consumed/reported
            # for this point -- next point needs its own fresh authorization.
            self.last_authorization_side = None
            self.last_authorization_time = None

            return result

        # --------------------------------------------------------
        # end_of_set -- CHANGED: now purely informational, like any
        # other fault/reason gesture. No longer requires the win
        # condition to be met, no longer forces self.set_over = True.
        # --------------------------------------------------------
        elif label == "end_of_set":
            if self.last_point_time is None or (timestamp - self.last_point_time) > REASON_ATTACH_WINDOW:
                return {"event": "ignored", "reason": "no recent point to attach end_of_set reason to"}

            if self.last_reason is not None:
                return {"event": "ignored", "reason": f"a reason ('{self.last_reason}') was already attached to this point"}

            self.last_reason = label
            self.last_settle_start_time = timestamp
            # CHANGED: no longer sets self.set_over = True here.
            return {"event": "reason_attached", "reason": label, "side": self.last_point_side}

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