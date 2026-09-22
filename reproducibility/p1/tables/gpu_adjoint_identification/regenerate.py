#!/usr/bin/env python3
"""Rebuild gpu_adjoint_identification.csv from inputs/summary.json."""
import csv, json, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
e = json.load(open(os.path.join(HERE, "inputs", "summary.json")))
cols = ["tag", "n_runs", "unique_sha", "median_iter_s_median", "wall_s_max",
        "rel_p50_m", "rel_p50_mu", "rel_p95_m", "rel_p95_mu",
        "crb_ratio_m", "crb_ratio_mu", "std_across_runs_p50_m"]
rows = []
for tag, v in e.items():
    rows.append({"tag": tag, "n_runs": v["n_runs"], "unique_sha": v["unique_sha"],
                 "median_iter_s_median": st.median(v["med_iter_s"]), "wall_s_max": max(v["wall_s"]),
                 "rel_p50_m": st.median(v["rel_p50_m"]), "rel_p50_mu": st.median(v["rel_p50_mu"]),
                 "rel_p95_m": st.median(v["rel_p95_m"]), "rel_p95_mu": st.median(v["rel_p95_mu"]),
                 "crb_ratio_m": st.median(v["crb_ratio_m"]), "crb_ratio_mu": st.median(v["crb_ratio_mu"]),
                 "std_across_runs_p50_m": v["std_across_runs_p50_m"]})
with open(os.path.join(HERE, "gpu_adjoint_identification.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
print(len(rows), "rows")
