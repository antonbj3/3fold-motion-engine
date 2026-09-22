#!/usr/bin/env python3
"""Appendix B dense vs block elimination: rebuild the printed table from the
delivered row-level measurement.

Reads `measurements_appendix_b.json` (the archived solver-run aggregate) and
prints the Appendix B rows, including the flagged final row of unconverged
iterates.  The linear-solve agreement is separately recomputed by
`supplement/patch/table_appendix_b.py`.

Run:  python table_appendix_b_rows.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MEAS = os.path.join(HERE, "..", "..", "supplement", "patch",
                    "measurements_appendix_b.json")


def main():
    d = json.load(open(MEAS))
    order = ["eight_box_tower", "eight_box_tower_rolling",
             "mass_ratio_column", "mass_ratio_column_rolling"]
    print("scene | dense_updates | block_updates | max_abs_impulse_difference | status")
    for name in order:
        s = d["scenes"][name]
        print(f"{name} | {s['dense_updates']} | {s['block_updates']} | "
              f"{s['max_abs_impulse_difference']} | {s['status']}")
    print(f"target_Ns = {d['target_Ns']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
