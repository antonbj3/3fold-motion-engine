#!/usr/bin/env python3
"""Rebuild dimensionless_stop.csv from the stored dimensionless_stop.json rows.

Regeneration only; no physics is rerun.
"""
import csv, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "dimensionless_stop.json")))
cols = ["scene", "index", "unit", "a", "c", "stop", "tol_used", "cap",
        "iterations", "converged", "residual", "residual_hat", "lam_ref",
        "g_ref", "rel_phys_vs_unit0", "n_chol", "positive_spectrum_ratio"]
with open(os.path.join(HERE, "dimensionless_stop.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in d["rows"]:
        w.writerow({c: r.get(c) for c in cols})
print(len(d["rows"]), "rows")
