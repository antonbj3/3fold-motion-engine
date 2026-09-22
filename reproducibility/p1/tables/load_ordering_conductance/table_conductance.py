#!/usr/bin/env python3
"""Section 5.4 conductance ranking: rebuild the printed table from the
delivered iterative-reference measurement.

Reads `benchmark_results.json` (the archived 8-scene load-ordering benchmark)
and prints the three conductance Spearman values for the finest foot-loaded
packing (`hertz_dw16`).

Run:  python table_conductance.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ["Resistans (H13 1/m)", "Resistans (unweighted)", "Resistans (Hertz R)"]
LABELS = ["Inverse mass", "Unweighted", "Hertz stiffness"]


def main():
    d = json.load(open(os.path.join(HERE, "benchmark_results.json")))
    r = d["results"]["hertz_dw16"]
    print(f"scene: {r['name']}  N={r['N']}  n_c={r['n_c']}")
    print("conductance | Spearman rho (iterative reference)")
    for label, key in zip(LABELS, KEYS):
        print(f"{label} | {r['metrics'][key]['spearman']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
