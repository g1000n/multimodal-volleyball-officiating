"""
tools/export_rubric.py

Writes the grading rubric straight from the code, so the table in your paper can never disagree with the
program:

    python tools/export_rubric.py

Creates
    data/grading_rubric.md    scoring rubric + one row per check with thresholds for all three levels
    data/grading_rubric.csv   the same check table as CSV (paste into Word / Excel)

Use it as the appendix / methodology table: "signal, FIVB wording, what is measured, points, effect, thresholds".
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gesture_grader as gg  # noqa: E402


def effect(c):
    if c.critical:
        return "Critical: if failed or not visible, verdict capped at ALMOST"
    if c.strict:
        return "Strict: if failed, verdict capped at ALMOST (ignored if not visible)"
    return "Score only"


def main():
    dummy = np.zeros((10, 122))
    rows = []
    for label in gg.RULES:
        per_level = {lv: gg.RULES[label](gg.Geo(dummy), lv) for lv in gg.LEVELS}
        for i, c in enumerate(per_level["standard"]):
            rows.append([
                gg.pretty_label(label), gg.SIGNALS[label]["fivb"], c.label, c.basis, c.weight, effect(c),
                per_level["beginner"][i].need, per_level["standard"][i].need, per_level["referee"][i].need,
            ])
        ready = gg.ready_position_check(np.zeros((6, 122)), "standard", 9 / 16)
        rows.append([gg.pretty_label(label), gg.SIGNALS[label]["fivb"], ready.label, ready.basis, ready.weight,
                     "Score only",
                     gg.ready_position_check(np.zeros((6, 122)), "beginner", 9 / 16).need,
                     ready.need, gg.ready_position_check(np.zeros((6, 122)), "referee", 9 / 16).need])

    os.makedirs("data", exist_ok=True)
    head = ["Signal", "FIVB wording (paraphrased)", "Check", "Basis", "Points", "Effect",
            "Beginner", "Standard", "Referee"]
    with open(os.path.join("data", "grading_rubric.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)

    md = ["# Grading rubric", "",
          "Each attempt is scored out of 100.", "",
          "| Part | Points | Meaning |", "|---|---|---|",
          f"| Recognition | {gg.W_RECOGNITION} | Best model probability for the target signal versus the level cutoff |",
          f"| Distinctness | {gg.W_DISTINCT} | Lead of the target over the strongest other class |",
          f"| Hold | {gg.W_HOLD} | How long the signal stayed recognized |",
          f"| FIVB form | {gg.W_FORM} | Geometric checks derived from the FIVB hand signal wording (table below) |",
          f"| Ready position | {gg.W_READY} | Arms returned to the ready position afterwards (training rule) |", "",
          "| Level | Recognition cutoff | Distinctness margin | Hold (s) | CORRECT at | ALMOST at |", "|---|---|---|---|---|---|"]
    for lv in gg.LEVELS:
        c = gg.LEVEL_CONFIG[lv]
        md.append(f"| {c.name} | {c.rec_threshold:.0%} | {c.margin_required:.0%} | {c.hold_seconds} | {c.correct_cut} | {c.almost_cut} |")
    md += ["", "Verdict: CORRECT needs the score cutoff, the signal recognized by the model, a hold of at least half the "
               "required time, and no failed critical or strict check. ALMOST is the lower cutoff or a recognized signal "
               "with a critical flaw. Otherwise INCORRECT. A body not visible in enough frames gives NO READING (not counted).",
           "Trainee points: CORRECT 10, ALMOST 5, INCORRECT 0.", "",
           "## Checks by signal", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        md.append("| " + " | ".join(str(x) for x in r) + " |")
    md += ["", "Basis FIVB = derived from the wording of the FIVB referee hand signals. Basis TRAINING = a pedagogical "
               "requirement. Numeric thresholds are the team's engineering translation of the wording into what a single "
               "2D camera can measure; they are not stated by FIVB."]
    with open(os.path.join("data", "grading_rubric.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote data/grading_rubric.md and data/grading_rubric.csv ({len(rows)} check rows)")


if __name__ == "__main__":
    main()