#!/usr/bin/env python3
"""Appendix E spectral-penalty grid: rebuild the printed table from the
delivered row-level measurement.

Reads `spectral_grid.csv` (the archived 21-row sweep output) and prints the
Appendix E rows.  No physics is rerun; the sweep used the internal ADMM
iteration.  The `within_2_selected` column is the source of the manuscript's
"five of twenty-one scenes meet the factor-of-two criterion" statement.

Run:  python table_spectral_grid.py
"""
from __future__ import annotations

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "spectral_grid.csv"))))
    cols = ["scene", "lambda_min_positive", "lambda_max", "rho_star", "iterations",
            "selected_rule", "best_ratio", "boundary_minimum",
            "censored_grid_points", "numerical_failures"]
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(str(r[c]) for c in cols))
    n = sum(1 for r in rows if r["within_2_selected"].strip().lower() == "yes")
    print(f"within_2_selected (factor-of-two rule) = {n}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
