"""
tools/export_rubric.py

Writes the grading rubric straight from the code, so the table in your paper can never disagree with the
program:

    python tools/export_rubric.py
    python tools/export_rubric.py --print service_authorization_left          # show the CURRENT thresholds on screen
    python tools/export_rubric.py --print service_authorization_left --level referee

The numbers include anything you changed in trainer_config.py (GRADER_OVERRIDES, LEVEL_OVERRIDES).

Creates
    data/grading_rubric.md    scoring rubric + one row per check with thresholds for all three levels
    data/grading_rubric.csv   the same check table as CSV (paste into Word / Excel)

Use it as the appendix / methodology table: "signal, FIVB wording, what is measured, points, effect, thresholds".
"""

import csv
import os
import sys

import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gesture_grader as gg  # noqa: E402


def effect(c):
    if c.critical:
        return "Critical: if failed or not visible, verdict capped at ALMOST"
    if c.strict:
        return "Strict: if failed, verdict capped at ALMOST (ignored if not visible)"
    return "Score only"


def print_signal(label, level):
    if label not in gg.RULES:
        print("Unknown signal. Choose one of:", ", ".join(sorted(gg.RULES)))
        return
    levels = [level] if level else list(gg.LEVELS)
    ctx = {"side": "left"} if label == "double_contact" else None
    dummy = np.zeros((10, 122))
    print(f"\n{gg.pretty_label(label)}   (config overrides: {len(gg.OVERRIDES)} check(s), "
          f"{'level overrides on' if getattr(gg._tc, 'LEVEL_OVERRIDES', None) else 'no level overrides'})")
    per = {lv: {c.id: c for c in gg.rules_for(label, gg.Geo(dummy), lv, None, ctx)} for lv in levels}
    ids = list(per[levels[0]].keys())
    def eff(c):
        return "Required" if c.critical else "Important" if c.strict else "Scored"

    head = f"{'check id':22s} {'pts':>3s}  " + "   ".join(f"{gg.LEVEL_CONFIG[lv].name:32s}" for lv in levels)
    print(head)
    for cid in ids:
        cells = [f"{per[lv][cid].need} [{eff(per[lv][cid])}]" for lv in levels]
        print(f"{cid:22s} {per[levels[0]][cid].weight:3d}  " + "   ".join(f"{c:32s}" for c in cells))
    for lv in levels:
        cfg = gg.LEVEL_CONFIG[lv]
        print(f"{cfg.name}: CORRECT from {cfg.correct_cut}, ALMOST from {cfg.almost_cut}, recognition {cfg.rec_threshold:.0%}, "
              f"hold {cfg.hold_seconds * gg.HOLD_SCALE.get(label, 1.0):.2f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", dest="show", default="", help="print the current thresholds of one signal, then exit")
    ap.add_argument("--level", choices=gg.LEVELS, default=None)
    args = ap.parse_args()
    if args.show:
        print_signal(args.show, args.level)
        return
    dummy = np.zeros((10, 122))
    head = ["Group", "Signal", "FIVB wording (paraphrased)", "Check", "Basis", "Points", "Effect",
            "Beginner", "Standard", "Referee"]
    rows, warnings = [], []
    per_signal = {}
    for label in gg.RULES:
        ctx = {"side": "left"} if label == "double_contact" else None
        per_level = {lv: gg.rules_for(label, gg.Geo(dummy), lv, None, ctx) for lv in gg.LEVELS}
        checks = []
        for i, c in enumerate(per_level["standard"]):
            checks.append([c.label, c.basis, c.weight, effect(c), per_level["beginner"][i].need,
                           per_level["standard"][i].need, per_level["referee"][i].need])
            rows.append([f"Signal form (sums to {gg.W_FORM})", gg.pretty_label(label), gg.SIGNALS[label]["fivb"]] + checks[-1])
        per_signal[label] = checks
        total = sum(c[2] for c in checks)
        if total != gg.W_FORM:
            warnings.append(f"{gg.pretty_label(label)}: the form checks add up to {total}, not {gg.W_FORM} (a GRADER_OVERRIDES weight was "
                            "changed; scoring rescales it, but the table will not add up).")

    # parts that apply to EVERY signal, listed once (not repeated under each signal)
    ready = {lv: gg.ready_position_check(np.zeros((6, 122)), lv, 9 / 16) for lv in gg.LEVELS}
    universal = [
        ["Recognition", "MODEL", gg.W_RECOGNITION, "Scored",
         *[f"model probability at least {gg.LEVEL_CONFIG[lv].rec_threshold:.0%}" for lv in gg.LEVELS]],
        ["Distinctness", "MODEL", gg.W_DISTINCT, "Scored",
         *[f"lead over the next class at least {gg.LEVEL_CONFIG[lv].margin_required:.0%}" for lv in gg.LEVELS]],
        ["Hold", "FIVB 30.1 (idea) + ENGINEERING (seconds)", gg.W_HOLD, "Scored",
         *[f"{gg.LEVEL_CONFIG[lv].hold_seconds:g} s (Service Authorization {gg.LEVEL_CONFIG[lv].hold_seconds * gg.HOLD_SCALE.get('service_authorization_left', 1.0):.2f} s)"
           for lv in gg.LEVELS]],
        [ready["standard"].label, ready["standard"].basis, gg.W_READY, "Scored (last signal only)",
         *[ready[lv].need for lv in gg.LEVELS]],
    ]
    for u in universal:
        rows.append(["Every signal, once per attempt", "All signals", "", *u])
    if gg.W_RECOGNITION + gg.W_DISTINCT + gg.W_HOLD + gg.W_FORM + gg.W_READY != 100:
        warnings.append("The five parts do not add up to 100.")

    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "grading_rubric.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)

    md = ["# Grading rubric", "",
          "Each attempt is scored out of 100. The five parts add up to 100.", "",
          "| Part | Points | Meaning |", "|---|---|---|",
          f"| Recognition | {gg.W_RECOGNITION} | Best model probability for the target signal versus the level cutoff |",
          f"| Distinctness | {gg.W_DISTINCT} | Lead of the target over the strongest other class |",
          f"| Hold | {gg.W_HOLD} | How long the signal stayed recognized |",
          f"| FIVB form | {gg.W_FORM} | Geometric checks derived from the FIVB hand signal wording; each signal has its own set that adds up to {gg.W_FORM} (tables below) |",
          f"| Ready position | {gg.W_READY} | Arms returned to the ready position afterwards. One check per attempt, not part of the {gg.W_FORM} points of any signal |",
          f"| **Total** | **{gg.W_RECOGNITION + gg.W_DISTINCT + gg.W_HOLD + gg.W_FORM + gg.W_READY}** | |", "",
          "| Level | Recognition cutoff | Distinctness margin | Hold (s) | CORRECT at | ALMOST at |", "|---|---|---|---|---|---|"]
    for lv in gg.LEVELS:
        c = gg.LEVEL_CONFIG[lv]
        md.append(f"| {c.name} | {c.rec_threshold:.0%} | {c.margin_required:.0%} | {c.hold_seconds} | {c.correct_cut} | {c.almost_cut} |")
    md += ["", "Verdict: CORRECT needs the score cutoff, the signal recognized by the model, a hold of at least "
               f"{gg.HOLD_OK_FRACTION:.0%} of the required time, and no failed Required or Important check (a Required check that "
               "could not be seen also limits the result). ALMOST is the lower cutoff, or a recognized signal with such a flaw. "
               "Otherwise INCORRECT. A body not visible in enough frames gives NO READING (not counted).",
           "Trainee points: CORRECT 10, ALMOST 5, INCORRECT 0 (added to the session total).",
           "Open hand: the FIVB text names open hands for Ball Out and End of Set, and the FIVB illustrations show an open "
           "hand for the other signals. It is applied as "
           f"{'a strict' if gg.OPEN_HAND_STRICT else 'a scored'} check (gesture_grader.OPEN_HAND_STRICT): "
           f"{'a visible fist caps the verdict at ALMOST' if gg.OPEN_HAND_STRICT else 'a closed hand only lowers the score and shows a tip, because the camera can misread fingers'}.",
           gg.DISCLAIMER, "", "## Parts that apply to every signal (once per attempt)", "",
           "| Part | Basis | Points | Effect | Beginner | Standard | Referee |", "|---|---|---|---|---|---|---|"]
    for u in universal:
        md.append("| " + " | ".join(str(x) for x in u) + " |")
    md += ["", "## FIVB form checks by signal (each table adds up to " + str(gg.W_FORM) + ")", ""]
    for label, checks in per_signal.items():
        md += [f"### {gg.pretty_label(label)}", "", f"FIVB wording (paraphrased): {gg.SIGNALS[label]['fivb']}", "",
               "| Check | Basis | Points | Effect | Beginner | Standard | Referee |", "|---|---|---|---|---|---|---|"]
        for c in checks:
            md.append("| " + " | ".join(str(x) for x in c) + " |")
        md += [f"| **Total** | | **{sum(c[2] for c in checks)}** | | | | |", ""]
    md += ["Basis FIVB = derived from the wording of the FIVB referee hand signals. Basis TRAINING = a pedagogical "
           "requirement or referee guidance. Numeric thresholds are the team's engineering translation of the wording into "
           "what a single 2D camera can measure; they are not stated by FIVB (see RULE_SOURCES.md)."]
    with open(os.path.join("data", "grading_rubric.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote data/grading_rubric.md and data/grading_rubric.csv ({len(rows)} rows)")
    for wtxt in warnings:
        print("WARNING: " + wtxt)


if __name__ == "__main__":
    main()