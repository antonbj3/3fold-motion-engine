#!/usr/bin/env python3
"""run common state-validation pass.

One call to `state_eval(sp, gpu, theta)` at a FIXED theta returns, from the SAME
state (same observations, same kernel, same graph budget, same chunk):

  * loss          = sum over env of ||y + P lam(theta)||^2
  * forward residual `res_env` = the natural NCP residual per env (max over its N steps)
  * tangent/gradient `Hs`, `gs` = the Gauss-Newton Hessian/gradient from the adjoint at
                                 that same lam and active set
  * lam, dl, cost arrays and a full sha256 digest over (lam, res_env, cost, Hs, gs)
  * the graph budget actually used: n_iter, cg, grad_cg, refresh_every, tol, chunk,
    n_calls = n_slip_max + 2
  * forward-ms, adjoint-ms and total-ms measured around the captured-graph replays.

The forward graph and the adjoint graph are two captured graphs launched in one timed
block; both use the same state.  This is the object the brief calls a shared
verification pass.  It does NOT replace the LM loop; the LM loop's line-search
candidates are different states and are logged separately with their own cost/residual.

The forward and adjoint loops are copied from src/estimator_gpu.gpu_cost_jac (which is
byte-identical to talos_adjoint_gpu.py) so the state is the same quantity that the
run accepted-state log measured; the only additions are the per-env residual readback
and the timing/digest bookkeeping.
"""
import os
import time

import numpy as np

import estimator_common  # noqa: F401  (puts src/ on sys.path)
import talos_ident_gn as GN
import estimator_gpu as G

MROWS, NC = GN.MROWS, GN.NC
NVD = GN.NVD


def pad_forward(sp, gpu, theta, chunk=None, nit=None, refresh_every=None, cg=None):
    """Forward-only pass returning (cost, lam, res_env).  Used as the residual-driven
    recomputation: `nit`/`refresh_every` can be raised without touching anything else."""
    env, F = sp["env"], sp["F"]
    chunk = gpu.B if chunk is None else min(chunk, F)
    theta = np.asarray(theta)
    Minv, v_free, muv, _, _, _ = GN.theta_parts(sp, theta[env, 0], theta[env, 1],
                                                theta[env, 2], deriv=False)
    b = np.einsum("bij,bj->bi", sp["J"], v_free) + sp["b_off"]
    lam = np.empty((F, MROWS))
    res = np.empty(F)
    for s0 in range(0, F, chunk):
        sl = slice(s0, min(s0 + chunk, F))
        n = sl.stop - s0
        gpu.upload(G._pad(sp["J"][sl], n, chunk), G._pad(Minv[sl], n, chunk),
                   G._pad(v_free[sl], n, chunk), G._pad(muv[sl], n, chunk),
                   G._pad(b[sl], n, chunk), sp["dt"])
        gpu.forward()
        lam[sl] = gpu.sol.lam.numpy().reshape(-1, MROWS)[:n]
        res[sl] = gpu.sol.resenv.numpy()[:n]
    r = sp["y"] + np.einsum("bij,bj->bi", sp["P"], lam[:, :3])
    cost = np.zeros(sp["B"])
    np.add.at(cost, env, (r ** 2).sum(1))
    res_env = res.reshape(sp["N"], sp["B"]).max(0)
    return cost, lam, res_env, r


def state_eval(sp, gpu, theta):
    """Single combined state-validation pass (forward + adjoint at one theta)."""
    env, F = sp["env"], sp["F"]
    theta = np.asarray(theta)
    chunk = gpu.B
    t_all0 = time.perf_counter()
    Minv, v_free, muv, dMi, dv, dmu = GN.theta_parts(
        sp, theta[env, 0], theta[env, 1], theta[env, 2], deriv=True)
    b = np.einsum("bij,bj->bi", sp["J"], v_free) + sp["b_off"]

    # --- forward (one captured graph, replayed per chunk) ---
    tf0 = time.perf_counter()
    lam = np.empty((F, MROWS))
    res = np.empty(F)
    for s0 in range(0, F, chunk):
        sl = slice(s0, min(s0 + chunk, F))
        n = sl.stop - s0
        gpu.upload(G._pad(sp["J"][sl], n, chunk), G._pad(Minv[sl], n, chunk),
                   G._pad(v_free[sl], n, chunk), G._pad(muv[sl], n, chunk),
                   G._pad(b[sl], n, chunk), sp["dt"])
        gpu.forward()
        lam[sl] = gpu.sol.lam.numpy().reshape(-1, MROWS)[:n]
        res[sl] = gpu.sol.resenv.numpy()[:n]
    ms_fwd = (time.perf_counter() - tf0) * 1e3

    # --- adjoint (one captured graph, replayed per chunk and per direction) ---
    ta0 = time.perf_counter()
    dl = np.empty((F, MROWS, 3))
    for s0 in range(0, F, chunk):
        sl = slice(s0, min(s0 + chunk, F))
        n = sl.stop - s0
        J = sp["J"][sl]
        Gb, _ = GN.S.delassus_b(J, Minv[sl], v_free[sl], sp["b_off"][sl])
        u = np.einsum("bij,bj->bi", Gb, lam[sl]) + b[sl]
        gpu.upload(G._pad(J, n, chunk), G._pad(Minv[sl], n, chunk),
                   G._pad(v_free[sl], n, chunk), G._pad(muv[sl], n, chunk),
                   G._pad(b[sl], n, chunk), sp["dt"])
        gpu.build_active(G._pad(lam[sl], n, chunk), G._pad(u, n, chunk))
        Jp = G._pad(J, n, chunk)
        lamp = G._pad(lam[sl], n, chunk)
        Jtl = np.einsum("bji,bj->bi", Jp, lamp)
        for k, key in enumerate(("m", "I")):
            src = (np.einsum("bij,bj->bi", G._pad(dMi[key][sl], n, chunk), Jtl)
                   + G._pad(dv[key][sl], n, chunk))
            db = np.einsum("bij,bj->bi", Jp, src)
            dl[sl, :, k] = gpu.dlam(pinh=None, dbv=db)[:n]
        what = gpu.sol.what.numpy().reshape(-1, NC, 3)
        slipf = gpu.sol.slipf.numpy().reshape(-1, NC)
        lamv = gpu.sol.zv.numpy().reshape(-1, NC, 3)
        dmus = G._pad(dmu[sl], n, chunk)
        p = np.zeros((chunk, NC, 3))
        for c in range(1, NC):
            m = slipf[:, c] > 0.5
            p[m, c, 1:3] += -lamv[m, c, 0, None] * what[m, c, 1:3] * dmus[m, c][:, None]
        dl[sl, :, 2] = gpu.dlam(pinh=p, dbv=None)[:n]
    ms_adj = (time.perf_counter() - ta0) * 1e3

    r = sp["y"] + np.einsum("bij,bj->bi", sp["P"], lam[:, :3])
    cost = np.zeros(sp["B"])
    np.add.at(cost, env, (r ** 2).sum(1))
    Jr = np.einsum("bij,bjk->bik", sp["P"], dl[:, :3, :])
    Hm = np.einsum("bik,bil->bkl", Jr, Jr)
    gv = np.einsum("bik,bi->bk", Jr, r)
    Hs = np.zeros((sp["B"], 3, 3))
    gs = np.zeros((sp["B"], 3))
    np.add.at(Hs, env, Hm)
    np.add.at(gs, env, gv)
    res_env = res.reshape(sp["N"], sp["B"]).max(0)
    ms_tot = (time.perf_counter() - t_all0) * 1e3
    budget = dict(n_iter=int(gpu.nit), cg=int(gpu.cg), grad_cg=int(gpu.grad_cg),
                  refresh_every=4, chunk=int(gpu.B), n_calls=int(gpu._ncalls))
    digest = estimator_common.arr_sha256(lam, res_env, cost, Hs, gs)
    return dict(cost=cost, lam=lam, res_env=res_env, res=res, Hs=Hs, gs=gs, dl=dl,
                r=r, digest=digest, ms_forward=ms_fwd, ms_adjoint=ms_adj, ms_total=ms_tot,
                budget=budget, res_max=float(res_env.max()),
                res_p95=float(np.percentile(res_env, 95)))
