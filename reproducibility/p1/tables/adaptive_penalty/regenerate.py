#!/usr/bin/env python3
"""Rebuild adaptive_penalty.csv from the stored per-run JSONs beside it.

Regeneration only (reads stored measurements); no physics is rerun.
"""
import csv, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = ["oracle_fixed", "structure_fixed", "he_mean", "he_structure",
           "wohlberg_mean", "wohlberg_structure"]
rows = []
for i in range(21):
    rr = {}
    for m in METHODS:
        a = json.loads(open(os.path.join(HERE, f"run1_{i:02d}_{m}.json")).read())
        b = json.loads(open(os.path.join(HERE, f"run2_{i:02d}_{m}.json")).read())
        assert a["solution_sha256"] == b["solution_sha256"], (i, m)
        rr[m] = a
    oracle = rr["oracle_fixed"]["iterations"]
    row = {"index": i, "scene": rr["oracle_fixed"]["scene"],
           "n_c": rr["oracle_fixed"]["n_c"]}
    for m in METHODS:
        x = rr[m]
        row[f"{m}_iterations"] = x["iterations"]
        row[f"{m}_rho_changes"] = x["rho_changes"]
        row[f"{m}_ratio_to_oracle"] = x["iterations"] / oracle
        row[f"{m}_status"] = x["status"]
    rows.append(row)
with open(os.path.join(HERE, "adaptive_penalty.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(len(rows), "rows")
