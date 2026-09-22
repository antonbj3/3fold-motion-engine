#!/usr/bin/env python3
"""residual study task 4: why accepted states break the declared residual tolerance, and a
residual acceptance gate as its own variant.

Input: a run's `out/<run>/accepted_states.csv` (the saved accepted-state log).  The
selection of cases is fixed in code BEFORE any GPU call: from the accepted rows of the
run, sorted by the logged accepted residual `res_ncp`, we take the rows at ranks
0, ceil(0.01 n), ceil(0.05 n), ceil(0.25 n), n//2, and a below-tolerance control at
ceil(0.90 n).  Each selected (iteration, env) is reconstructed from the log, and its
residual is recomputed on the SAME observations with the SAME solver by:
  (a) a forward pass at the run budget nit (reproduction check), and
  (b) a residual-driven forward recomputation at higher nit (extra work counted).
A single env is rebuilt by slicing the scene to that env; the solver is per-env, so the
env-in-batch and env-alone residuals are the same quantity.

Output: `accepted_cases.csv` with logged residual, reproduced residual, recomputed
residuals, whether the tolerance is met after recomputation, and the extra work.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

import estimator_common as C
import talos_ident_gn as GN
import estimator_state as ST
import estimator_gpu as G


def select(rows):
    """Predetermined mixed selection of accepted rows (rule fixed in code)."""
    rows = sorted(rows, key=lambda r: float(r["res_ncp"]), reverse=True)
    n = len(rows)
    ranks = [0, int(np.ceil(0.01 * n)), int(np.ceil(0.05 * n)), int(np.ceil(0.25 * n)),
             n // 2, int(np.ceil(0.90 * n))]
    out, seen = [], set()
    for rk in ranks:
        rk = min(rk, n - 1)
        row = rows[rk]
        key = (row["it"], row["env"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def slice_sp_env(sp, e):
    """Extract one env's N rows from the batch scene WITHOUT redrawing the measurement
    noise: the flattened layout is env-fast (row n*B + e).  This is required to
    reproduce the exact accepted state; rebuilding static_parts on a 1-env dict would
    draw a different rng.normal realization."""
    B, N = sp["B"], sp["N"]
    idx = np.arange(N) * B + e
    keys = ["J", "b_off", "vfy", "P", "y", "vel", "omg", "R", "hand", "pos", "L", "A"]
    out = {k: np.ascontiguousarray(np.asarray(sp[k])[idx]) for k in keys}
    out.update(N=N, B=1, F=N, dt=sp["dt"], eps=sp["eps"], sigma=sp["sigma"],
               mu_hand=sp["mu_hand"])
    out["env"] = np.zeros(N, np.int64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run", required=True)
    ap.add_argument("--sim", default=None)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--nB", type=int, default=None)
    ap.add_argument("--env-off", type=int, default=None)
    ap.add_argument("--nit", type=int, default=None)
    ap.add_argument("--recompute-nits", default="900,2700")
    ap.add_argument("--cg", type=int, default=None)
    ap.add_argument("--grad-cg", type=int, default=None)
    ap.add_argument("--res-tol", type=float, default=None)
    ap.add_argument("--out", default="accepted_cases.csv")
    a = ap.parse_args()

    run_dir = os.path.join(C.HERE, "out", a.source_run)
    rec = json.load(open(os.path.join(run_dir, "record.json")))
    # bind the scene/parameters to the run that produced the accepted states
    for k in ("sim", "sigma", "window", "seed", "nB", "env_off", "nit", "cg", "grad_cg",
              "res_tol"):
        if getattr(a, k) is None:
            setattr(a, k, rec[k])
    print(f"bound to {a.source_run}: sim={a.sim} sigma={a.sigma} window={a.window} "
          f"seed={a.seed} nB={a.nB} env_off={a.env_off} nit={a.nit}")

    with open(os.path.join(run_dir, "accepted_states.csv")) as f:
        all_rows = list(csv.DictReader(f))
    acc_rows = [r for r in all_rows if r["accepted"] == "1"]
    sel = select(acc_rows)
    print(f"{a.source_run}: {len(acc_rows)} accepted, selection {len(sel)}")

    # full per-iteration theta from the log, to keep the reconstruction explicit
    theta_by_it = {}
    for r in all_rows:
        it = int(r["it"])
        theta_by_it.setdefault(it, {})[int(r["env"])] = (
            float(r["m"]), float(r["I_zz"]), float(r["mu"]))

    sim = os.path.join(C.HERE, a.sim)
    d_all = np.load(sim)
    d_all = {k: d_all[k] for k in d_all.files}
    if a.nB and a.nB < d_all["m_true"].shape[0]:
        d_all = C.subset_env(d_all, a.nB, 0)
    sp_full = GN.static_parts(d_all, a.sigma, a.window, a.seed)
    nits = [int(x) for x in a.recompute_nits.split(",") if x]

    recs = []
    for r in sel:
        it, e = int(r["it"]), int(r["env"])
        theta_e = np.array([theta_by_it[it][e]], np.float64)  # (1,3)
        sp1 = slice_sp_env(sp_full, e)
        row = dict(run=a.source_run, it=it, env=e,
                   logged_res=float(r["res_ncp"]),
                   logged_last_cand=float(r["last_cand_res"]),
                   theta_sha256=C.theta_sha256(theta_e))
        for nit in [a.nit] + nits:
            gpu = G.make_gpu(min(1024, sp1["F"]), nit, a.cg, a.grad_cg)
            cost, lam, res_env, _ = ST.pad_forward(sp1, gpu, theta_e)
            key = f"res_nit{nit}"
            row[key] = float(res_env[0])
            row[f"cost_nit{nit}"] = float(cost[0])
        row["reproduced_at_budget"] = abs(row[f"res_nit{a.nit}"] - row["logged_res"]) \
            <= 1e-9 + 1e-6 * abs(row["logged_res"])
        row["pass_after_recompute"] = all(row[f"res_nit{n}"] <= a.res_tol for n in nits)
        row["extra_work_nit_ratio"] = max(nits) / a.nit if nits else 1.0
        recs.append(row)

    fields = list(recs[0].keys())
    with open(os.path.join(C.HERE, a.out), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in recs:
            w.writerow(row)
    summary = dict(source_run=a.source_run, n_accepted=len(acc_rows),
                   selection_ranks=[(int(r["it"]), int(r["env"]), float(r["res_ncp"]))
                                    for r in sel],
                   n_reproduced=sum(bool(r["reproduced_at_budget"]) for r in recs),
                   n_pass_after=sum(bool(r["pass_after_recompute"]) for r in recs),
                   n_selected=len(recs),
                   recompute_nits=nits, res_tol=a.res_tol)
    json.dump(summary, open(os.path.join(C.HERE, a.out.replace(".csv", ".json")), "w"),
              indent=1)
    print(json.dumps(summary, indent=1))
    print("->", a.out)


if __name__ == "__main__":
    main()
