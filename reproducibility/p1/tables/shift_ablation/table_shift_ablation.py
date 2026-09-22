#!/usr/bin/env python3
"""Section 4.2 de Saxce shift ablation: rebuild the printed tables from the
delivered 252-cell row-level measurement.

Reads `results.csv` (21 scenes x 4 penalty rules x 3 shift modes) and
`summary.json` (the archived aggregates) and prints:

  * the 12-row aggregate table (median iterations, converged/tested, penalty
    changes);
  * the 4-row dense-random table (dense_random_contact_operator_256, index 12).

No physics is rerun; the ablation used the internal ADMM iteration and a
sparse-LU linear solve.

Run:  python table_shift_ablation.py
"""
from __future__ import annotations

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RULES = ["oracle", "structure", "lmin", "he_structure"]
SHIFTS = ["on", "off", "fixed"]


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "results.csv"))))
    summary = json.load(open(os.path.join(HERE, "summary.json")))
    print("rule | shift | median_iterations | converged/21 | rho_changes")
    for shift in SHIFTS:
        for rule in RULES:
            s = summary["by_shift"][shift][rule]
            print(f"{rule} | {shift} | {s['median_iterations']} | "
                  f"{s['converged']}/21 | {s['total_rho_changes']}")
    print()
    print("dense random (dense_random_contact_operator_256, index 12):")
    print("rule | refreshed | removed | frozen")
    r = rows[12]
    for rule in RULES:
        print(f"{rule} | {r[rule + '_on_iterations']} | {r[rule + '_off_iterations']} | "
              f"{r[rule + '_fixed_iterations']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
