#!/usr/bin/env python3
"""run: Talos Gauss-Newton identification with a shared state-validation pass.

Differences from talos_adjoint_run.py (the corrected driver):

  1. Provenance is captured BEFORE warp is imported (`estimator_common.capture_pre`: frozen
     source sha256, serialized input file hashes, declared graph budget) and the ACTUAL
     loaded module files plus `get_module_options` are captured after import
     (`capture_post`).  The run record took the hashes after the loop.
  2. At every iteration the accepted composite `theta` goes through ONE common
     verification pass (`estimator_state.state_eval`) that returns loss, forward residual and
     the tangent/gradient Hs/gs from that same state, with timing, digest and the graph
     budget.  The line-search candidates are separate states and keep their own
     cost/residual columns; no latency or precision is paired across states.
  3. Optional residual acceptance gate (`--gate residual`): a candidate is accepted only
     if BOTH its cost decreases AND its own natural residual is <= --res-tol.  This is a
     separate variant; the default (`--gate cost`) keeps the run cost-only rule.
  4. Every run_id writes to out/<run_id>/ ; a repetition cannot overwrite another.

ACC is selected by MOTION_ACC=int64|f32 (int64 -> batch_adjoint_gpu.py, f32 -> batch_adjoint_gpu_f32.py).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import estimator_common as C
import talos_ident_gn as GN
import talos_ident_scene as S
import ncp_ref
import estimator_state as ST

if os.environ.get("MOTION_ACC", "int64") == "int64":
    import batch_adjoint_gpu as H
else:
    import batch_adjoint_gpu_f32 as H
import estimator_gpu as G

STEP_KEYS = C.STEP_KEYS
subset_env = C.subset_env


def run(sim, sigma, window, iters, chunk, nit, cg, grad_cg, trials, seed, nB, env_off,
        run_id, acc, res_tol, gate="cost", theta0=None, damp0=None, start_iter=0,
        ckpt_in=None, ckpt_out=None, verbose=True):
    modules = {"estimator_common": C, "estimator_state": ST, "talos_ident_gn": GN, "talos_ident_scene": S,
               "ncp_ref": ncp_ref, "estimator_gpu": G,
               "batch_adjoint_gpu": H if acc == "int64" else None,
               "batch_adjoint_gpu_f32": H if acc != "int64" else None}
    modules = {k: v for k, v in modules.items() if v is not None}

    # ---- phase 1: provenance BEFORE any import/compile/launch ----
    sim_path = os.path.join(C.HERE, sim)
    budget0 = dict(n_iter=nit, cg=cg, grad_cg=grad_cg, refresh_every=4, chunk=chunk,
                   trials=trials, seed=seed, gate=gate, res_tol=res_tol, window=window,
                   sigma=sigma, nB=nB, env_off=env_off)
    C.capture_pre(run_id, inputs={"sim": sim_path, "sim_sha256": C.sha256_file(sim_path)},
                  budget=budget0, extra=dict(acc=acc, start_iter=start_iter))

    # ---- phase 2: load, build scene, compile ----
    d = np.load(sim_path)
    d = {k: d[k] for k in d.files}
    if nB and nB < d["m_true"].shape[0]:
        d = subset_env(d, nB, env_off)
    tr = np.stack([d["m_true"], d["I_zz_true"], d["mu_true"]], 1)
    sp = GN.static_parts(d, sigma, window, seed)
    B, F = sp["B"], sp["F"]
    ch = min(chunk, F)
    gpu = G.make_gpu(ch, nit, cg, grad_cg)

    if ckpt_in is not None:
        ck = np.load(os.path.join(C.HERE, ckpt_in))
        theta = ck["theta"].copy()
        damp = ck["damp"].copy()
        start_iter = int(ck["next_iter"])
    elif theta0 is not None:
        theta = np.asarray(theta0, np.float64).copy()
        damp = np.full(B, 1e-6) if damp0 is None else np.asarray(damp0, np.float64).copy()
    else:
        Lb = sp["L"].reshape(sp["N"], B, 3)[0]
        theta = np.stack([np.full(B, 5.0),
                          5.0 / 12.0 * (Lb[:, 0] ** 2 + Lb[:, 1] ** 2),
                          np.full(B, 0.40)], 1)
        damp = np.full(B, 1e-6)
    theta = np.asarray(theta, np.float64)
    damp = np.asarray(damp, np.float64)

    prov_post = C.capture_post(run_id, modules)
    sp_dig = C.sp_digest(sp)
    gpu_mem_before = C.gpu_memory_mib()

    d_out = C.out_dir(run_id)
    states_path = os.path.join(d_out, "states.jsonl")
    fstates = open(states_path, "w")
    fapps = open(os.path.join(d_out, "matched_application.csv"), "w")
    fapps.write("run,it,state,cost_sum,cost_p50,res_max,res_p95,grad_norm,H_sha256,"
                "digest,n_slip,res_pass,ms_forward,ms_adjoint,ms_total,chunk,n_calls,"
                "n_iter,cg,grad_cg,theta_sha256\n")
    fcsv = open(os.path.join(d_out, "accepted_states.csv"), "w")
    fcsv.write("run,it,env,accepted,m,I_zz,mu,cost,res_ncp,res_tol,res_pass,gate,"
               "last_cand_res,m_true,I_zz_true,mu_true,rel_m,rel_I,rel_mu,theta_sha256\n")

    t_wall0 = time.perf_counter()

    def log_app(it, state, st, nslip=None):
        gnorm = float(np.linalg.norm(st["gs"], axis=1).mean())
        row = [run_id, it, state, repr(float(st["cost"].sum())),
               repr(float(np.median(st["cost"]))), repr(float(st["res_max"])),
               repr(float(st["res_p95"])), repr(gnorm),
               C.arr_sha256(st["Hs"]), st["digest"],
               -1 if nslip is None else int(nslip),
               int(st["res_max"] <= res_tol),
               repr(float(st["ms_forward"])), repr(float(st["ms_adjoint"])),
               repr(float(st["ms_total"])), int(st["budget"]["chunk"]),
               int(st["budget"]["n_calls"]), int(st["budget"]["n_iter"]),
               int(st["budget"]["cg"]), int(st["budget"]["grad_cg"]),
               C.theta_sha256(theta)]
        fapps.write(",".join(str(x) for x in row) + "\n")
        fapps.flush()

    # initial accepted state (the starting theta) through the common pass
    st0 = ST.state_eval(sp, gpu, theta)
    cost = st0["cost"]
    log_app(start_iter, "initial", st0)
    fstates.write(json.dumps(dict(run_id=run_id, it=start_iter, state="initial",
                                  cost_sum=float(cost.sum()), res_max=st0["res_max"],
                                  digest=st0["digest"], budget=st0["budget"])) + "\n")

    hist = []
    names = ("m", "I_zz", "mu")
    end_iter = start_iter + iters
    for it in range(start_iter, end_iter):
        t0 = time.perf_counter()
        # state at the PREVIOUS theta: normal gradient here (legitimate own state)
        st_prev = ST.state_eval(sp, gpu, theta)
        Hs, gs = st_prev["Hs"], st_prev["gs"]
        log_app(it, "prev_theta", st_prev)
        accmask = np.zeros(B, bool)
        best = theta.copy()
        best_cost = cost.copy()
        best_res = st_prev["res_env"].copy()
        last_cand_res = np.full(B, np.nan)
        for _tr in range(trials):
            cand = GN._step(Hs, gs, theta, damp)
            c2, lam2, res2 = G.gpu_cost_res(sp, gpu, cand)
            last_cand_res = res2.reshape(sp["N"], B).max(0) if res2.ndim == 1 else res2
            good = (~accmask) & (c2 < best_cost)
            if gate == "residual":
                good = good & (last_cand_res <= res_tol)
            best[good] = cand[good]
            best_cost[good] = c2[good]
            best_res[good] = last_cand_res[good]
            damp[good] /= 5.0
            damp[(~accmask) & (~good)] *= 10.0
            accmask |= good
            if accmask.all():
                break
        # accepted composite state through the SAME common verification pass
        st_acc = ST.state_eval(sp, gpu, best)
        cost_acc = st_acc["cost"]
        cost_diff = float(np.abs(cost_acc - best_cost).max())
        res_env = st_acc["res_env"]
        theta, cost = best, cost_acc
        it_t = time.perf_counter() - t0
        rel = np.abs(theta - tr) / np.abs(tr)
        log_app(it, "accepted", st_acc, nslip=None)
        fstates.write(json.dumps(dict(
            run_id=run_id, it=it, state="accepted", t_s=it_t, accepted=int(accmask.sum()),
            cost_sum=float(cost.sum()), cost_recompute_maxdiff=cost_diff,
            res_max=float(res_env.max()), res_p95=float(np.percentile(res_env, 95)),
            res_lastcand_max=float(np.nanmax(last_cand_res)),
            digest=st_acc["digest"], budget=st_acc["budget"],
            ms_total=st_acc["ms_total"], theta_sha256=C.theta_sha256(theta))) + "\n")
        for e in range(B):
            fcsv.write(",".join(str(x) for x in (
                run_id, it, e, int(accmask[e]),
                repr(float(theta[e, 0])), repr(float(theta[e, 1])), repr(float(theta[e, 2])),
                repr(float(cost[e])), repr(float(res_env[e])), repr(float(res_tol)),
                int(res_env[e] <= res_tol), gate, repr(float(last_cand_res[e])),
                repr(float(tr[e, 0])), repr(float(tr[e, 1])), repr(float(tr[e, 2])),
                repr(float(rel[e, 0])), repr(float(rel[e, 1])), repr(float(rel[e, 2])),
                C.theta_sha256(theta))) + "\n")
        fcsv.flush()
        hist.append(dict(it=it, t_s=it_t, accepted=int(accmask.sum()),
                         cost_sum=float(cost.sum()),
                         res_acc_max=float(res_env.max()),
                         res_acc_p95=float(np.percentile(res_env, 95)),
                         res_lastcand_max=float(np.nanmax(last_cand_res)),
                         cost_recompute_maxdiff=cost_diff,
                         gate_pass=int((res_env <= res_tol).sum()),
                         digest=st_acc["digest"],
                         ms_total=float(st_acc["ms_total"]),
                         damp_med=float(np.median(damp))))
        if verbose:
            print(f"estimator {it}: acc {accmask.sum()}/{B} cost {cost.sum():.6e} "
                  f"res_acc_max {res_env.max():.2e} res_last {np.nanmax(last_cand_res):.2e} "
                  f"gate_pass {(res_env <= res_tol).sum()}/{B} dCostRecomp {cost_diff:.1e} "
                  f"state_ms {st_acc['ms_total']:.0f} {it_t:.1f}s", flush=True)
    fstates.close()
    fapps.close()
    fcsv.close()

    wall = time.perf_counter() - t_wall0
    rel = np.abs(theta - tr) / np.abs(tr)
    gpu_mem_after = C.gpu_memory_mib()
    record = dict(run_id=run_id, acc=acc, sim=sim, sigma=sigma, window=window, iters=iters,
                  start_iter=start_iter, end_iter=end_iter, chunk=ch, B=B, F=F, nit=nit,
                  cg=cg, grad_cg=grad_cg, trials=trials, seed=seed, nB=nB, env_off=env_off,
                  res_tol=res_tol, gate=gate, wall_s=wall,
                  theta_sha256=C.theta_sha256(theta), sp_digest=sp_dig,
                  gpu_memory_before_mib=gpu_mem_before, gpu_memory_after_mib=gpu_mem_after,
                  hist=hist,
                  rel={nm: dict(p50=float(np.median(rel[:, j])),
                                p95=float(np.percentile(rel[:, j], 95)),
                                max=float(rel[:, j].max())) for j, nm in enumerate(names)},
                  provenance_pre="out/%s/provenance_pre.json" % run_id,
                  provenance_post="out/%s/provenance_post.json" % run_id,
                  states="out/%s/states.jsonl" % run_id,
                  matched_application="out/%s/matched_application.csv" % run_id,
                  accepted_states="out/%s/accepted_states.csv" % run_id,
                  prov=prov_post)
    np.savez(os.path.join(d_out, "theta.npz"), theta=theta, rel=rel, cost=cost, true=tr,
             res_env=res_env, damp=damp)
    json.dump(record, open(os.path.join(d_out, "record.json"), "w"), indent=1)
    with open(os.path.join(C.HERE, "run_records.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")
    if ckpt_out is not None:
        np.savez(os.path.join(C.HERE, ckpt_out), theta=theta, damp=damp, next_iter=end_iter,
                 acc=acc, sim=sim, sigma=sigma, window=window, seed=seed, nB=nB,
                 env_off=env_off)
    if verbose:
        for j, nm in enumerate(names):
            print(f"  {nm:5s} p50 {record['rel'][nm]['p50']:.6e} "
                  f"p95 {record['rel'][nm]['p95']:.6e}")
        print(f"  wall {wall:.1f} s (GPU shared) sha256 {record['theta_sha256']}")
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="identification_sim_B512.npz")
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--nit", type=int, default=300)
    ap.add_argument("--cg", type=int, default=4)
    ap.add_argument("--grad-cg", type=int, default=60)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--nB", type=int, default=0)
    ap.add_argument("--env-off", type=int, default=0)
    ap.add_argument("--res-tol", type=float, default=1e-5)
    ap.add_argument("--gate", default="cost", choices=["cost", "residual"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--init", default=None)
    ap.add_argument("--ckpt-in", default=None)
    ap.add_argument("--ckpt-out", default=None)
    a = ap.parse_args()
    th0 = np.load(os.path.join(C.HERE, a.init))["theta"] if a.init else None
    run(a.sim, a.sigma, a.window, a.iters, a.chunk, a.nit, a.cg, a.grad_cg, a.trials,
        a.seed, a.nB, a.env_off, a.run_id, os.environ.get("MOTION_ACC", "int64"),
        a.res_tol, gate=a.gate, theta0=th0, ckpt_in=a.ckpt_in, ckpt_out=a.ckpt_out)


if __name__ == "__main__":
    main()
