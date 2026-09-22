#!/usr/bin/env python3
"""Rebuild solver_relaxation.csv from the stored run JSONs in inputs/.

Regeneration only (reads stored per-solver measurements); it does NOT rerun the
physics, which would need the internal reference solver.
"""
import csv, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ["rep", "index", "scene", "n_c", "solver", "iterations", "converged",
        "residual", "seconds", "normal_gap_m", "outside_coulomb_fraction",
        "tangential_relative_deviation_from_exact", "solution_sha256"]
rows = []
for rep in (1, 2):
    for i in range(21):
        for x in json.load(open(os.path.join(HERE, "inputs", f"run{rep}_{i:02d}.json"))):
            rows.append({**{c: x.get(c) for c in COLS[1:]}, "rep": rep})
with open(os.path.join(HERE, "solver_relaxation.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
print(len(rows), "rows")
