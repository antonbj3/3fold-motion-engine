#!/usr/bin/env python3
"""Rebuild the incidence-solver CSVs from the stored inputs/ JSONs (no physics).

Writes three CSVs, all from the stored measurement JSONs:

  incidence_solvers.csv    scenes table + per-solver iterations/residuals/times
  incidence_scaling.csv    the scaling study on chains and random geometric
                           graphs (Appendix C / AC7) from inputs/scaling.json
  incidence_blockthomas.csv  the block-Thomas / preconditioner comparison
                           (Appendix D / AD1-AD3) from inputs/blockthomas.json

No numeric value is invented; the JSONs are the delivered measurements.
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
I = os.path.join(HERE, "inputs")

SOLVERS = ["amg_rs", "amg_sa", "pcg_jacobi", "direct"]


def write_scenes():
    scenes = json.load(open(os.path.join(I, "scenes.json")))
    cols = ["scene", "source", "n_c", "nnz", "bandwidth", "rho", "spd_natural",
            "lambda_min", "lambda_max", "cond"] + \
           [f"{s}_{k}" for s in SOLVERS
            for k in ("iterations", "relative_residual", "setup_s", "solve_s")]
    with open(os.path.join(HERE, "incidence_solvers.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for name, sc in scenes["scenes"].items():
            p = sc["props"]
            row = {"scene": name, "source": sc.get("source"), "n_c": p.get("n_c"),
                   "nnz": p.get("nnz"), "bandwidth": p.get("bandwidth"),
                   "rho": p.get("rho"), "spd_natural": p.get("spd_natural"),
                   "lambda_min": p.get("lambda_min"), "lambda_max": p.get("lambda_max"),
                   "cond": p.get("cond")}
            for s in SOLVERS:
                sv = sc["solvers"].get(s, {})
                for k in ("iterations", "relative_residual", "setup_s", "solve_s"):
                    row[f"{s}_{k}"] = sv.get(k)
            wr.writerow(row)
    print("incidence_solvers.csv written")


def write_scaling():
    data = json.load(open(os.path.join(I, "scaling.json")))
    solvers = ["amg_rs", "amg_sa", "pcg_jacobi", "direct", "tridiag"]
    cols = ["group", "N", "n", "nnz", "bandwidth", "rho"] + \
           [f"{s}_{k}" for s in solvers for k in ("iterations", "relative_residual")]
    rows = []
    for group in ("chains", "rgg"):
        for cell in data.get(group, []):
            row = {"group": group, "N": cell.get("N"), "n": cell.get("n"),
                   "nnz": cell.get("nnz"), "bandwidth": cell.get("bandwidth"),
                   "rho": cell.get("rho")}
            for s in solvers:
                sv = cell.get(s, {})
                for k in ("iterations", "relative_residual"):
                    row[f"{s}_{k}"] = sv.get(k)
            rows.append(row)
    with open(os.path.join(HERE, "incidence_scaling.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for row in rows:
            wr.writerow(row)
    print("incidence_scaling.csv written", len(rows), "rows")


def write_blockthomas():
    data = json.load(open(os.path.join(I, "blockthomas.json")))
    cols = ["N", "block_setup_s", "block_solve_s", "bw_offblock_nonzero",
            "chol_setup_s", "chol_solve_s", "max_abs_tri_minus_chol", "n", "nnz",
            "rel_res_tri", "rho", "sha256_tri"]
    with open(os.path.join(HERE, "incidence_blockthomas.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for cell in data["chains"]:
            wr.writerow({c: cell.get(c) for c in cols})
    print("incidence_blockthomas.csv written", len(data["chains"]), "rows")


if __name__ == "__main__":
    write_scenes()
    write_scaling()
    write_blockthomas()
