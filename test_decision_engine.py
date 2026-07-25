"""
test_decision_engine.py

Pure logic test harness for decision_engine.py -- NO camera, NO trained
model, NO audio. Just decision_engine.py's actual rules, fed a sequence of
events you define, so you can verify the logic matches what you expect
before trusting it live tomorrow.

HOW TO ADD YOUR OWN SCENARIOS: copy one of the dicts in SCENARIOS below and
edit it. Each event is (timestamp_seconds, event_type), where event_type is
either "whistle" or a real gesture label. Timestamps are NOT real time --
they're simulated seconds, so you can test a 10-second gap instantly without
waiting 10 real seconds. This lets you directly test edge cases like "what
happens if the reason gesture arrives exactly at the settle-window boundary"
without needing to physically perform anything.

Run:
    python test_decision_engine.py
"""

from decision_engine import DecisionEngine

SCENARIOS = [
    {
        "name": "Basic point + reason, normal timing",
        "events": [
            (0.0, "whistle"),
            (0.5, "team_to_serve_right"),
            (3.0, "ball_out"),
        ],
        "expected_final_score": {"left": 0, "right": 1},
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
            (15.0, "team_to_serve_left"),  # TEMPORAL_WINDOW is 10.0s
        ],
        "expected_final_score": {"left": 0, "right": 0},
    },
    {
        "name": "Reason gesture arrives right at the settle-window boundary",
        "events": [
            (0.0, "whistle"),
            (0.5, "team_to_serve_right"),
            (2.0, "ball_out"),   # 1.5s after point -- exactly at SETTLE_WINDOW_SECONDS
        ],
        "expected_final_score": {"left": 0, "right": 1},
        "note": "Whether the reason attaches depends on exact settle-window timing -- "
                "watch the printed event result rather than assuming.",
    },
    {
        "name": "Second reason gesture should NOT overwrite the first (one-reason-per-point)",
        "events": [
            (0.0, "whistle"),
            (0.5, "team_to_serve_right"),
            (3.0, "ball_out"),
            (5.5, "double_contact"),  # should be ignored -- reason already attached
        ],
        "expected_final_score": {"left": 0, "right": 1},
    },
    {
        "name": "Two full points in a row, both sides",
        "events": [
            (0.0, "whistle"),
            (0.5, "team_to_serve_right"),
            (3.0, "ball_out"),
            (6.0, "whistle"),
            (6.5, "team_to_serve_left"),
            (9.0, "double_contact"),
        ],
        "expected_final_score": {"left": 1, "right": 1},
    },
    {
        "name": "end_of_set predicted but score doesn't satisfy win condition -- should be rejected",
        "events": [
            (0.0, "whistle"),
            (0.5, "team_to_serve_right"),
            (3.0, "end_of_set"),  # score is only 0-1, nowhere near a real win
        ],
        "expected_final_score": {"left": 0, "right": 1},
        "note": "end_of_set itself should be REJECTED as a reason here (score doesn't satisfy win condition) -- watch the printed event.",
    },
    {
        "name": "Real win condition met -- end_of_set SHOULD be accepted (using manual_override_score to fast-forward the score)",
        "events": [
            ("manual_score", "left", 25),
            (0.0, "whistle"),
            (0.5, "team_to_serve_left"),
            (3.0, "end_of_set"),
        ],
        "expected_final_score": {"left": 26, "right": 0},
        "note": "Score set to 25-0 via manual override before the sequence starts, then one more point makes it 26-0 -- satisfies win-by-2, end_of_set should be ACCEPTED this time.",
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