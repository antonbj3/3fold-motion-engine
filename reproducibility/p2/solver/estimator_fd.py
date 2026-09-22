#!/usr/bin/env python3
"""run task 3b: independent finite-difference check of the adjoint gradient on a
small converged subselection, and the same check where the residual is not converged.

For a chosen accepted (iteration, env) the adjoint state-validation pass gives the
Gauss-Newton gradient `gs` with d(cost)/dtheta = 2 gs.  This script recomputes that
derivative by central differences of the forward cost on the SAME observations and the
SAME solver, for a converged env (min accepted residual in the iteration) and a
non-converged env (max accepted residual).  The two are reported separately: where the
solver has not converged (or the active set changes under the perturbation) the FD and
the adjoint disagree; that is reported, not hidden.

Reverse mode is not run in this release experiment and stays not_measured.
"""
import argparse
import csv
import json
import os

import numpy as np

import estimator_common as C
import talos_ident_gn as GN
import estimator_state as ST
import estimator_gpu as G
from estimator_residual import slice_sp_env

HREL = (1e-4, 1e-5, 1e-6)


def fd_grad(sp1, theta0, res_tol):
    """Central-difference d(cost)/dtheta and the state-validation adjoint 2*gs."""
    gpu = G.make_gpu(min(1024, sp1["F"]), 300, 4, 60)
    st = ST.state_eval(sp1, gpu, theta0)
    g_adj = 2.0 * st["gs"][0]
    out = {}
    for hrel in HREL:
        gfd = np.zeros(3)
        for j in range(3):
            h = hrel * max(abs(theta0[0, j]), 1e-6)
            tp = theta0.copy(); tp[0, j] += h
            tm = theta0.copy(); tm[0, j] -= h
            cp, _, _, _ = ST.pad_forward(sp1, gpu, tp)
            cm, _, _, _ = ST.pad_forward(sp1, gpu, tm)
            gfd[j] = (cp[0] - cm[0]) / (2.0 * h)
        denom = max(float(np.abs(g_adj).max()), 1e-300)
        out[hrel] = dict(gfd=gfd.tolist(), g_adj=g_adj.tolist(),
                         rel_err=float(np.abs(gfd - g_adj).max() / denom))
    return st, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run", required=True)
    ap.add_argument("--target-it", type=int, default=3)
    ap.add_argument("--out", default="fd_check.csv")
    a = ap.parse_args()

    run_dir = os.path.join(C.HERE, "out", a.source_run)
    rec = json.load(open(os.path.join(run_dir, "record.json")))
    sim = os.path.join(C.HERE, rec["sim"])
    d_all = np.load(sim)
    d_all = {k: d_all[k] for k in d_all.files}
    if rec["nB"] and rec["nB"] < d_all["m_true"].shape[0]:
        d_all = C.subset_env(d_all, rec["nB"], rec["env_off"])
    sp_full = GN.static_parts(d_all, rec["sigma"], rec["window"], rec["seed"])

    rows = list(csv.DictReader(open(os.path.join(run_dir, "accepted_states.csv"))))
    th = {}
    for r in rows:
        th.setdefault(int(r["it"]), {})[int(r["env"])] = (
            float(r["m"]), float(r["I_zz"]), float(r["mu"]))
    it = a.target_it
    acc = [r for r in rows if int(r["it"]) == it and r["accepted"] == "1"]
    acc.sort(key=lambda r: float(r["res_ncp"]))
    picks = [("converged_min_res", acc[0]), ("nonconverged_max_res", acc[-1])]

    out_rows = []
    for label, r in picks:
        e = int(r["env"])
        theta0 = np.array([th[it][e]], np.float64)
        sp1 = slice_sp_env(sp_full, e)
        st, fd = fd_grad(sp1, theta0, rec["res_tol"])
        row = dict(run=a.source_run, it=it, env=e, label=label,
                   logged_res=float(r["res_ncp"]), state_res=float(st["res_max"]),
                   g_adj=st["gs"][0].tolist(), cost=float(st["cost"][0]))
        for hrel in HREL:
            row[f"gfd_h{hrel}"] = fd[hrel]["gfd"]
            row[f"fd_rel_err_h{hrel}"] = fd[hrel]["rel_err"]
        out_rows.append(row)
        print(label, "env", e, "logged_res", row["logged_res"], "state_res",
              row["state_res"], "fd_rel_err", [fd[h]["rel_err"] for h in HREL])

    fields = list(out_rows[0].keys())
    with open(os.path.join(C.HERE, a.out), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)
    print("->", a.out)


if __name__ == "__main__":
    main()
