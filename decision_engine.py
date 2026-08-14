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
NEW THIS VERSION -- PENDING WHISTLE CONFIRMATION:

Previously, a service_authorization or team_to_serve gesture arriving
with no valid recent whistle was rejected outright ("no recent
whistle" / "no recent whistle for service authorization"). That
turned out to be too rigid for how referees actually perform the
sequence in real match footage: sometimes the gesture is HELD and the
whistle is blown a moment AFTER the gesture starts, not strictly
before it. A strict "the whistle must already exist" check would
wrongly reject a completely genuine call just because of that
ordering.

FIX: a gesture that arrives with no valid recent whistle is no longer
instantly rejected. It's parked as `self.pending_gesture` (its label
plus its OWN original timestamp) and the caller gets back
`{"event": "awaiting_whistle_confirmation", ...}` instead of
`"ignored"`. If a whistle then arrives within
WHISTLE_CONFIRMATION_GRACE_SECONDS, `on_whistle_detected()`
retroactively confirms it -- it runs the EXACT SAME commit logic
(`_commit_authorization` / `_commit_scoring`) that the immediate path
uses, using the GESTURE's own original timestamp (not the whistle's)
for the actual commit. This matters: it means `last_point_time`, the
settle-window start, etc. all reflect when the gesture itself
happened, not when the whistle happened to arrive -- so downstream
timing behaves identically to the immediate-whistle-first case. If the
grace window passes with no whistle arriving, the pending gesture is
simply discarded -- the same end result as the old immediate
rejection, just delayed instead of instant, so a referee who never
actually blows the whistle still doesn't get a false commit.

Only one gesture can be pending confirmation at a time -- a new,
different unconfirmed gesture arriving replaces whatever was pending
before (the previous one presumably either already timed out or was
superseded by whatever's happening now).

Deliberately NOT implemented as a simple bidirectional time window
(`abs(gesture_time - whistle_time) <= window`), because that would let
ANY whistle within a loose window validate ANY gesture, including ones
they had nothing to do with -- e.g. a whistle blown for an unrelated
reason seconds earlier could wrongly confirm a totally different
gesture that happens to land nearby in time. The pending-state design
keeps a tight, one-to-one pairing instead: THIS specific gesture is
waiting on THIS specific upcoming whistle, and nothing else can
satisfy it. A whistle that arrives with nothing pending just behaves
exactly as it always did -- an ordinary whistle, refreshing
last_whistle_time for whatever gesture comes next.

--------------------------------------------------------------------
CHANGED (previous version): AUTOMATIC SET-ENDING REMOVED. This
project's core purpose is automating POINT-ADDING -- fault/reason
gestures (ball_out, double_contact, ball_in, end_of_set) are
informational only, not the thing being validated for correctness.
Previously, EVERY team_to_serve commit auto-checked the win condition
and could silently set self.set_over = True the moment score crossed
the win threshold -- this happened purely from the SCORING gesture,
with no dependency on end_of_set ever being detected. Confirmed via a
real live session: this caused scoring to freeze while a real rally
was still ongoing, because the model isn't perfect and the automatic
win-condition check doesn't know that.

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

# NEW: how long a gesture with no valid recent whistle stays "pending,"
# waiting for a whistle to arrive AFTER the fact, before being
# discarded. This is deliberately a SEPARATE constant from
# TEMPORAL_WINDOW -- TEMPORAL_WINDOW governs whistle-THEN-gesture
# ordering (how long a whistle stays valid for a gesture that follows
# it); this one governs the opposite ordering, gesture-THEN-whistle.
# They don't have to be the same value, and starting narrower here is
# deliberate -- a "waiting on a whistle that might never come" state
# should time out faster than an already-heard whistle stays valid.
# Starting value -- tune against real match logs the same way
# TEMPORAL_WINDOW was calibrated (see above), if this ever misfires by
# being too eager (unrelated whistle confirms a stale gesture) or too
# easily missed (a real, slightly slow whistle doesn't make it in time).
WHISTLE_CONFIRMATION_GRACE_SECONDS = 3.0

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

        # NEW: the one gesture (if any) currently waiting on a whistle
        # to arrive after the fact. Either None, or a dict of the shape
        # {"label": <gesture label>, "timestamp": <when the gesture
        # itself was detected>}. Only one gesture can be pending at a
        # time -- see module docstring for why.
        self.pending_gesture = None

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
        The only remaining way self.set_over becomes True -- an
        explicit, deliberate call, e.g. a human operator pressing an
        "end set" control. Nothing in on_gesture_detected() calls this
        automatically anymore.
        """
        self.set_over = True
        return self.set_over

    def _pending_gesture_expired(self, now):
        """True if there's no pending gesture at all, OR the one that's
        pending has sat longer than WHISTLE_CONFIRMATION_GRACE_SECONDS
        without a whistle arriving to confirm it."""
        if self.pending_gesture is None:
            return True
        return (now - self.pending_gesture["timestamp"]) > WHISTLE_CONFIRMATION_GRACE_SECONDS

    def on_whistle_detected(self, timestamp=None):
        """
        Whistles are never blocked by the settle window -- they're
        the deliberate start of the next step, not tail-end noise.

        NEW: also checks whether a gesture is currently sitting in
        self.pending_gesture waiting for exactly this. If one is, and
        it hasn't expired, this whistle retroactively confirms it --
        runs the identical commit logic the immediate-whistle-first
        path would have run, using the GESTURE's own original
        timestamp (not this whistle's timestamp) so all downstream
        timing (last_point_time, the settle window, etc.) reflects
        when the gesture itself actually happened.

        Returns the confirmation result dict (same shape
        on_gesture_detected() would have returned for that gesture,
        plus a `confirmed_by_late_whistle: True` marker) if a pending
        gesture was just confirmed, or None if this was just an
        ordinary whistle with nothing pending on it.
        """
        timestamp = timestamp if timestamp is not None else time.time()

        if self.pending_gesture is not None and not self._pending_gesture_expired(timestamp):
            pending_label = self.pending_gesture["label"]
            pending_timestamp = self.pending_gesture["timestamp"]
            self.pending_gesture = None

            if pending_label in SERVICE_AUTHORIZATION_GESTURES:
                confirmation_result = self._commit_authorization(pending_label, pending_timestamp)
                confirmation_result["confirmed_by_late_whistle"] = True
                # Consumed by the authorization commit itself (see
                # _commit_authorization) -- do NOT also fall through to
                # setting last_whistle_time below, same as the
                # immediate-path behavior. This still forces a genuine
                # SECOND, distinct whistle before team_to_serve can be
                # accepted next.
                return confirmation_result
            elif pending_label in SCORING_GESTURES:
                confirmation_result = self._commit_scoring(pending_label, pending_timestamp)
                confirmation_result["confirmed_by_late_whistle"] = True
                return confirmation_result
        else:
            # Pending gesture (if any) expired before this whistle
            # arrived -- discard it. Same end result as the old
            # immediate rejection used to produce, just delayed by up
            # to WHISTLE_CONFIRMATION_GRACE_SECONDS instead of instant.
            self.pending_gesture = None

        self.last_whistle_time = timestamp
        return None

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
        self.pending_gesture = None

    def _commit_authorization(self, label, timestamp):
        """
        The actual service_authorization commit logic -- factored out
        into its own method so both the immediate path (a valid whistle
        was already present when the gesture arrived) and the new
        confirmed-later path (whistle arrives after the gesture, inside
        on_whistle_detected()) run EXACTLY the same code, rather than
        two versions that could quietly drift apart from each other.
        """
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

    def _commit_scoring(self, label, timestamp):
        """
        The actual team_to_serve (point-award) commit logic -- same
        factoring-out rationale as _commit_authorization above: one
        shared implementation for both the immediate-whistle-present
        path and the confirmed-later-via-pending-gesture path.
        """
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
        # No auto-check/set of self.set_over here -- see module
        # docstring. Score just keeps counting, no cap.

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
            has_valid_whistle = (self.last_whistle_time is not None
                                  and (timestamp - self.last_whistle_time) <= TEMPORAL_WINDOW)
            if has_valid_whistle:
                return self._commit_authorization(label, timestamp)

            # NEW: no valid whistle YET -- rather than rejecting
            # outright, park this as pending in case the whistle
            # arrives a moment after the gesture (a real ordering
            # observed in actual match footage -- see module
            # docstring). on_whistle_detected() will retroactively
            # confirm this if a whistle shows up within
            # WHISTLE_CONFIRMATION_GRACE_SECONDS.
            self.pending_gesture = {"label": label, "timestamp": timestamp}
            return {"event": "awaiting_whistle_confirmation",
                    "reason": "gesture recognized, waiting for whistle",
                    "grace_seconds": WHISTLE_CONFIRMATION_GRACE_SECONDS}

        # --------------------------------------------------------
        # PHASE 2: team_to_serve (whistle #2 -> point/team-to-serve signal)
        # --------------------------------------------------------
        if label in SCORING_GESTURES:
            has_valid_whistle = (self.last_whistle_time is not None
                                  and (timestamp - self.last_whistle_time) <= TEMPORAL_WINDOW)
            if has_valid_whistle:
                return self._commit_scoring(label, timestamp)

            # NEW: same pending treatment as the authorization branch
            # above -- see module docstring.
            self.pending_gesture = {"label": label, "timestamp": timestamp}
            return {"event": "awaiting_whistle_confirmation",
                    "reason": "gesture recognized, waiting for whistle",
                    "grace_seconds": WHISTLE_CONFIRMATION_GRACE_SECONDS}

        # --------------------------------------------------------
        # end_of_set -- purely informational, like any other
        # fault/reason gesture. No longer requires the win condition
        # to be met, no longer forces self.set_over = True. Also
        # unrelated to whistle timing entirely -- no pending-
        # confirmation behavior applies here.
        # --------------------------------------------------------
        elif label == "end_of_set":
            if self.last_point_time is None or (timestamp - self.last_point_time) > REASON_ATTACH_WINDOW:
                return {"event": "ignored", "reason": "no recent point to attach end_of_set reason to"}

            if self.last_reason is not None:
                return {"event": "ignored", "reason": f"a reason ('{self.last_reason}') was already attached to this point"}

            self.last_reason = label
            self.last_settle_start_time = timestamp
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