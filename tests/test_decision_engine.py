"""
test_decision_engine.py

Pure logic test harness for decision_engine.py -- NO camera, NO trained
model, NO audio. Just decision_engine.py's actual rules, fed a sequence of
events you define, so you can verify the logic matches what you expect
before trusting it live.

IMPORTANT -- LEFT/RIGHT FLIP: decision_engine.py's GESTURE_TO_SCORE_SIDE
deliberately flips the model's own class-name side into the audience/
court-facing scored side. team_to_serve_right now scores LEFT, and
team_to_serve_left now scores RIGHT. Every expected_final_score below has
been updated to reflect this -- do NOT "fix" these back to the old
unflipped values, that would be reintroducing the bug the flip was meant
to correct.

HOW TO ADD YOUR OWN SCENARIOS: copy one of the dicts in SCENARIOS below and
edit it. Each event is (timestamp_seconds, event_type), where event_type is
either "whistle" or a real gesture label. Timestamps are NOT real time --
they're simulated seconds, passed explicitly into every engine call, so you
can test a 10-second gap instantly without waiting 10 real seconds.

Run:
    python test_decision_engine.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from decision_engine import DecisionEngine

SCENARIOS = [
    {
        "name": "Basic point + reason, normal timing",
        "events": [
            (0.0, "whistle"),
            (0.5, "team_to_serve_right"),   # FLIPPED: scores LEFT now
            (3.0, "ball_out"),
        ],
        "expected_final_score": {"left": 1, "right": 0},
    },
    {
        "name": "Scoring gesture with NO whistle first -- should be rejected",
        "events": [
            (0.0, "team_to_serve_left"),
        ],
        "expected_final_score": {"left": 0, "right": 0},
    },
    {
        "name": "Whistle too long ago (past TEMPORAL_WINDOW) -- should be rejected",
        "events": [
            (0.0, "whistle"),
            (15.0, "team_to_serve_left"),  # TEMPORAL_WINDOW is 10.0s -- gap is 15s
        ],
        "expected_final_score": {"left": 0, "right": 0},
        "note": "UNRESOLVED as of this rewrite: live test output showed this gesture "
                "being ACCEPTED (point_awarded) despite the 15s gap exceeding the "
                "10.0s TEMPORAL_WINDOW. This should not be possible per the whistle-"
                "recency check in on_gesture_detected() -- needs the current live "
                "decision_engine.py inspected directly to find the actual cause "
                "before trusting this scenario's expected value either way.",
    },
    {
        "name": "Reason gesture arrives right at the settle-window boundary",
        "events": [
            (0.0, "whistle"),
            (0.5, "team_to_serve_right"),   # FLIPPED: scores LEFT now
            (2.0, "ball_out"),   # 1.5s after point -- exactly at SETTLE_WINDOW_SECONDS
        ],
        "expected_final_score": {"left": 1, "right": 0},
        "note": "Whether the reason attaches depends on exact settle-window timing -- "
                "watch the printed event result rather than assuming.",
    },
    {
        "name": "Second reason gesture should NOT overwrite the first (one-reason-per-point)",
        "events": [
            (0.0, "whistle"),
            (0.5, "team_to_serve_right"),   # FLIPPED: scores LEFT now
            (3.0, "ball_out"),
            (5.5, "double_contact"),  # should be ignored -- reason already attached
        ],
        "expected_final_score": {"left": 1, "right": 0},
    },
    {
        "name": "Two full points in a row, both sides",
        "events": [
            (0.0, "whistle"),
            (0.5, "team_to_serve_right"),   # FLIPPED: scores LEFT now
            (3.0, "ball_out"),
            (6.0, "whistle"),
            (6.5, "team_to_serve_left"),    # FLIPPED: scores RIGHT now
            (9.0, "double_contact"),
        ],
        "expected_final_score": {"left": 1, "right": 1},
    },
    {
        "name": "end_of_set predicted but score doesn't satisfy win condition -- should be rejected",
        "events": [
            (0.0, "whistle"),
            (0.5, "team_to_serve_right"),   # FLIPPED: scores LEFT now
            (3.0, "end_of_set"),  # score is only 1-0, nowhere near a real win
        ],
        "expected_final_score": {"left": 1, "right": 0},
        "note": "end_of_set itself should be REJECTED as a reason here (score doesn't "
                "satisfy win condition) -- watch the printed event.",
    },
    {
        "name": "Real win condition met -- end_of_set SHOULD be accepted (using manual_override_score to fast-forward the score)",
        "events": [
            ("manual_score", "left", 24),   # FIXED: was 25, which already satisfies
            # win-by-2 on its own (set_over becomes True immediately, before the
            # whistle/team_to_serve even get a chance to fire) -- 24 correctly leaves
            # the set open so the sequence tests a REAL point crossing the threshold.
            (0.0, "whistle"),
            (0.5, "team_to_serve_right"),   # FIXED: team_to_serve_right scores LEFT
            # (flip), correctly pushing left from 24 -> 25, satisfying win-by-2.
            (3.0, "end_of_set"),
        ],
        "expected_final_score": {"left": 25, "right": 0},
        "note": "Score set to 24-0 via manual override before the sequence starts, "
                "then team_to_serve_right (scores LEFT, per the flip) pushes left to "
                "25 -- satisfies win-by-2, end_of_set should be ACCEPTED as the reason.",
    },
    {
        "name": "Full two-phase cycle: authorization -> second whistle -> point",
        "events": [
            (0.0, "whistle"),                       # whistle #1
            (0.5, "service_authorization_left"),    # GESTURE_TO_SCORE_SIDE maps this
            # to "right" -- beckons right (per flip: service_authorization_left -> right)
            (6.0, "whistle"),                       # whistle #2, well after rally
            (6.5, "team_to_serve_right"),           # scores LEFT per flip -- different
            # side than the authorization on purpose, to confirm authorization_match
            # correctly reports False rather than assuming they always agree.
        ],
        "expected_final_score": {"left": 1, "right": 0},
        "note": "Confirms the full real sequence works end-to-end: whistle #1 is "
                "consumed by service_authorization (not available to team_to_serve), "
                "whistle #2 is required and present, and the point is awarded. Also "
                "watch authorization_side/authorization_match on the point_awarded "
                "event -- authorization_side should be 'right' (service_authorization_left "
                "maps to right per the flip table) while the point itself scores 'left', "
                "so authorization_match should print False.",
    },
    {
        "name": "team_to_serve with only ONE whistle (no second whistle) -- should be rejected",
        "events": [
            (0.0, "whistle"),                       # whistle #1
            (0.5, "service_authorization_left"),    # consumes whistle #1
            (3.0, "team_to_serve_left"),             # NO second whistle was ever fired --
            # this must be rejected, proving one whistle can't silently cover both
            # phases (the exact bug the two-phase redesign was meant to fix).
        ],
        "expected_final_score": {"left": 0, "right": 0},
        "note": "Confirms the two-whistle enforcement is real: after "
                "service_authorization consumes whistle #1, team_to_serve MUST be "
                "rejected with 'no recent whistle' since no second whistle was fired. "
                "If this scenario ever shows point_awarded, the two-phase redesign "
                "has regressed back to the old single-whistle behavior.",
    },
]


def run_scenario(scenario):
    print(f"\n{'=' * 70}")
    print(f"SCENARIO: {scenario['name']}")
    if "note" in scenario:
        print(f"NOTE: {scenario['note']}")
    print("=" * 70)

    engine = DecisionEngine()  # uses real default win_score=25, win_by_margin=2

    for event in scenario["events"]:
        if event[0] == "manual_score":
            _, side, delta = event
            engine.manual_override_score(side, delta)
            print(f"  [manual_score] {side} += {delta}  -> score={engine.score}")
            continue

        timestamp, event_type = event
        if event_type == "whistle":
            engine.on_whistle_detected(timestamp)
            print(f"  t={timestamp:>5.1f}  WHISTLE")
        else:
            result = engine.on_gesture_detected(event_type, timestamp)
            print(f"  t={timestamp:>5.1f}  {event_type:<30} -> {result}")

    final = engine.score
    expected = scenario["expected_final_score"]
    match = final == expected
    status = "PASS" if match else "*** MISMATCH ***"
    print(f"\n  Final score: {final}   Expected: {expected}   [{status}]")
    return match


def main():
    print(f"Running {len(SCENARIOS)} decision_engine scenarios...\n")
    results = [run_scenario(s) for s in SCENARIOS]

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {sum(results)}/{len(results)} scenarios matched expected final score")
    print("=" * 70)
    if not all(results):
        print("\nAny MISMATCH above needs a closer look before deployment -- either")
        print("the expected_final_score was wrong, or decision_engine.py's logic")
        print("did something you didn't intend. Read the printed per-event results")
        print("above the summary to see exactly which step diverged.")


if __name__ == "__main__":
    main()