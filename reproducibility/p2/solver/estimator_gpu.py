#!/usr/bin/env python3
"""run GPU-adjoint driver for Talos's Talos identification.

The gradient source is batch_adjoint_gpu.BatchProxADMMGPU.grad_affine (the corrected run copy
in this directory, k_setsig guard `j >= 0`), NOT talos_ident_gn.sens_solve / np.linalg.solve.

WHY FEEDING rhs_raw AS `dbv` IS THE SAME LINEAR MAP AS ncp_ref.sensitivity:
The reduced NCP is u(theta) = G(theta) lam + b(theta); its implicit-function
  dlam/dtheta = -A^-1 (dG/dtheta lam + db/dtheta),
where A depends only on (G, mu, labels, lam).  batch_adjoint's grad_affine(p, db) solves the SAME
system with the SAME A, and for p=0 its output is exactly the map db -> dlam (validated
against ncp_ref.sensitivity for arbitrary db in t2_gradient_allslip.py).
That map is linear in db, so feeding dbv = dG/dtheta lam + db/dtheta (the full u-direction)
reproduces ncp_ref.sensitivity for a general theta.  The mu direction is not a db: it moves
the cone boundary, which batch_adjoint expresses as p_inhom = -sum_c lam_n,c w_hat_c (contact operator study_gpu's
grad_dmu mechanism, summed over the slipping floor contacts that share mu_floor).
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import warp as wp  # noqa: E402
import talos_ident_gn as GN  # noqa: E402

if os.environ.get("MOTION_ACC", os.environ.get("TALOS_ACC", "int64")) == "int64":
    import batch_adjoint_gpu as H  # noqa: E402
else:
    import batch_adjoint_gpu_f32 as H  # noqa: E402

MROWS, NC = GN.MROWS, GN.NC
NVD = GN.NVD


class GpuAdjoint:
    def __init__(self, B, nv, nc, nit, cg=4, grad_cg=60, tol=1e-14):
        self.sol = H.BatchProxADMMGPU(B, nv, nc)
        self.B, self.nv, self.nc = B, nv, nc
        self.nit, self.cg, self.grad_cg, self.tol = nit, cg, grad_cg, tol
        self._fg = None
        self._gg = None
        self._ncalls = None

    def upload(self, J, Minv, v_free, muv, b, dt):
        s = self.sol
        s.upload(J, Minv, v_free, muv, dt)
        wp.copy(s.bb, wp.array(np.ascontiguousarray(np.asarray(b).reshape(-1, 3)),
                               dtype=H.vec3d, device=s.d))
        s.set_rho("sqrt")
        return self

    def forward(self):
        s = self.sol
        if self._fg is None:
            self._fg, _ = s.capture_fwd(self.nit, cg_iters=self.cg, refresh_every=4,
                                        tol=self.tol)
        s.replay(self._fg)
        return s.lam.numpy().reshape(self.B, MROWS).copy()

    def build_active(self, lam, u):
        s = self.sol
        wp.copy(s.zv, wp.array(np.ascontiguousarray(lam.reshape(-1, 3)),
                               dtype=H.vec3d, device=s.d))
        wp.copy(s.u, wp.array(np.ascontiguousarray(u.reshape(-1, 3)),
                              dtype=H.vec3d, device=s.d))
        s.build_active()
        wp.synchronize_device(s.d)
        n = int(s.nslip.numpy().max()) + 2
        if n != self._ncalls:
            self._gg = None
            self._ncalls = n
        return n

    def _set_vec(self, name, arr):
        s = self.sol
        dst = getattr(s, name)
        if arr is None:
            wp.copy(dst, wp.array(np.zeros((self.B * NC, 3), np.float64),
                                  dtype=H.vec3d, device=s.d))
        else:
            wp.copy(dst, wp.array(np.ascontiguousarray(np.asarray(arr).reshape(-1, 3)),
                                  dtype=H.vec3d, device=s.d))

    def dlam(self, pinh=None, dbv=None):
        s = self.sol
        self._set_vec("pinh", pinh)
        self._set_vec("dbv", dbv)
        if self._gg is None:
            s.grad_affine(self._ncalls, self.grad_cg)   # warm-up / compile
            self._gg, _ = s._capture(s.grad_affine, self._ncalls, self.grad_cg)
        wp.capture_launch(self._gg)
        wp.synchronize_device(s.d)
        return s.dlam.numpy().reshape(self.B, MROWS).copy()


def _u(theta_parts_out, J, lam, b):
    Minv, v_free, muv, _, _, _ = theta_parts_out
    return Minv, v_free, muv


def _pad(a, n, chunk):
    """Pad a leading-axis array to `chunk` rows by repeating its first row.

    run: control_run.py and derivative_check_gpu.py only ever ran with
    F divisible by the chunk (F = 25600, chunk 1024), so the incomplete-final-chunk
    case was never executed.  With F = 3200 and chunk 64 the last chunk has 32 rows
    and `upload` raises "cannot reshape array of size 11520 into shape (64,15,24)".
    Repeated real rows keep the solver well posed; their outputs are discarded, and
    because every kernel is per-env the real rows are unaffected."""
    a = np.asarray(a)
    if n == chunk:
        return a
    return np.concatenate([a, np.repeat(a[:1], chunk - n, axis=0)], axis=0)


def gpu_cost_jac(sp, gpu, theta, need_jac=True, return_dl=False):
    """Returns cost (B,), lam (F,15), resmax and, if need_jac, Hs (B,3,3), gs (B,3)."""
    env = sp["env"]
    F = sp["F"]
    chunk = gpu.B
    theta = np.asarray(theta)
    Minv, v_free, muv, dMi, dv, dmu = GN.theta_parts(sp, theta[env, 0], theta[env, 1],
                                                    theta[env, 2], deriv=need_jac)
    b = np.einsum("bij,bj->bi", sp["J"], v_free) + sp["b_off"]
    lam = np.empty((F, MROWS))
    resmax = 0.0
    for s0 in range(0, F, chunk):
        sl = slice(s0, min(s0 + chunk, F))
        n = sl.stop - s0
        gpu.upload(_pad(sp["J"][sl], n, chunk), _pad(Minv[sl], n, chunk),
                   _pad(v_free[sl], n, chunk), _pad(muv[sl], n, chunk),
                   _pad(b[sl], n, chunk), sp["dt"])
        lam[sl] = gpu.forward()[:n]
        resmax = max(resmax, float(gpu.sol.resenv.numpy()[:n].max()))
    r = sp["y"] + np.einsum("bij,bj->bi", sp["P"], lam[:, :3])
    cost = np.zeros(sp["B"])
    np.add.at(cost, env, (r ** 2).sum(1))
    if not need_jac:
        return cost, lam, resmax, None, None

    dl = np.empty((F, MROWS, 3))
    for s0 in range(0, F, chunk):
        sl = slice(s0, min(s0 + chunk, F))
        n = sl.stop - s0
        J = sp["J"][sl]
        Gb, _ = GN.S.delassus_b(J, Minv[sl], v_free[sl], sp["b_off"][sl])
        u = np.einsum("bij,bj->bi", Gb, lam[sl]) + b[sl]
        # run CANDIDATE FIX: the adjoint loop must re-upload this chunk's (J, M^-1,
        # G_diag, eta, kappa): grad_affine's operator is Z^T (J M^-1 J^T) Z + D and its
        # preconditioner Ps = k_sjac(Gd, eta).  Without this the second loop ran every
        # chunk against the LAST forward-loop chunk's geometry.
        gpu.upload(_pad(J, n, chunk), _pad(Minv[sl], n, chunk),
                   _pad(v_free[sl], n, chunk), _pad(muv[sl], n, chunk),
                   _pad(b[sl], n, chunk), sp["dt"])
        gpu.build_active(_pad(lam[sl], n, chunk), _pad(u, n, chunk))
        # --- m and I_zz: the whole u-direction as dbv ---
        Jp = _pad(J, n, chunk)
        lamp = _pad(lam[sl], n, chunk)
        Jtl = np.einsum("bji,bj->bi", Jp, lamp)
        for k, key in enumerate(("m", "I")):
            src = (np.einsum("bij,bj->bi", _pad(dMi[key][sl], n, chunk), Jtl)
                   + _pad(dv[key][sl], n, chunk))
            db = np.einsum("bij,bj->bi", Jp, src)
            dl[sl, :, k] = gpu.dlam(pinh=None, dbv=db)[:n]
        # --- mu_floor: cone-boundary particular at every slipping floor contact ---
        what = gpu.sol.what.numpy().reshape(-1, NC, 3)
        slipf = gpu.sol.slipf.numpy().reshape(-1, NC)
        lamv = gpu.sol.zv.numpy().reshape(-1, NC, 3)
        dmus = _pad(dmu[sl], n, chunk)
        p = np.zeros((chunk, NC, 3))
        for c in range(1, NC):
            m = slipf[:, c] > 0.5
            p[m, c, 1:3] += -lamv[m, c, 0, None] * what[m, c, 1:3] * dmus[m, c][:, None]
        dl[sl, :, 2] = gpu.dlam(pinh=p, dbv=None)[:n]
    Jr = np.einsum("bij,bjk->bik", sp["P"], dl[:, :3, :])
    Hm = np.einsum("bik,bil->bkl", Jr, Jr)
    gv = np.einsum("bik,bi->bk", Jr, r)
    Hs = np.zeros((sp["B"], 3, 3)); gs = np.zeros((sp["B"], 3))
    np.add.at(Hs, env, Hm); np.add.at(gs, env, gv)
    if return_dl:
        return cost, lam, resmax, Hs, gs, dl
    return cost, lam, resmax, Hs, gs


def gpu_cost_res(sp, gpu, theta):
    """Forward-only evaluation that additionally returns the PER-ENV natural NCP
    residual (F,) read back after the last refresh of the forward graph.  This is the
    quantity whose batch maximum `gpu_cost_jac` reports as `resmax`; exposing it per env
    is what run needs to log the residual of the ACCEPTED composite state on its own
    rows rather than through a scalar batch maximum of the last line-search candidate."""
    env = sp["env"]
    F = sp["F"]
    chunk = gpu.B
    theta = np.asarray(theta)
    Minv, v_free, muv, _, _, _ = GN.theta_parts(sp, theta[env, 0], theta[env, 1],
                                                theta[env, 2], deriv=False)
    b = np.einsum("bij,bj->bi", sp["J"], v_free) + sp["b_off"]
    lam = np.empty((F, MROWS))
    res = np.empty(F)
    for s0 in range(0, F, chunk):
        sl = slice(s0, min(s0 + chunk, F))
        n = sl.stop - s0
        gpu.upload(_pad(sp["J"][sl], n, chunk), _pad(Minv[sl], n, chunk),
                   _pad(v_free[sl], n, chunk), _pad(muv[sl], n, chunk),
                   _pad(b[sl], n, chunk), sp["dt"])
        gpu.forward()
        lam[sl] = gpu.sol.lam.numpy().reshape(-1, MROWS)[:n]
        res[sl] = gpu.sol.resenv.numpy()[:n]
    r = sp["y"] + np.einsum("bij,bj->bi", sp["P"], lam[:, :3])
    cost = np.zeros(sp["B"])
    np.add.at(cost, env, (r ** 2).sum(1))
    return cost, lam, res


def make_gpu(chunk, nit, cg, grad_cg):
    return GpuAdjoint(chunk, NVD, NC, nit, cg=cg, grad_cg=grad_cg)


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=os.path.join(HERE, "identification_sim_B512.npz"))
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--nit", type=int, default=200)
    ap.add_argument("--cg", type=int, default=4)
    ap.add_argument("--grad-cg", type=int, default=60)
    a = ap.parse_args()
    d = np.load(a.sim)
    d = {k: d[k] for k in d.files}
    sp = GN.static_parts(d, 0.0, a.window, 1)
    tr = np.stack([d["m_true"], d["I_zz_true"], d["mu_true"]], 1)
    theta = tr.copy()
    gpu = make_gpu(min(a.chunk, sp["F"]), a.nit, a.cg, a.grad_cg)
    t0 = time.perf_counter()
    cost, lam, rm, Hs, gs = gpu_cost_jac(sp, gpu, theta)
    t1 = time.perf_counter()
    # NumPy reference
    gn = GN.GpuNcp(min(a.chunk, sp["F"]), a.nit, cg_iters=a.cg)
    c_np, lam_np, rm_np = GN._cost(sp, gn, min(a.chunk, sp["F"]), theta)
    Hs_np, gs_np = GN._jac(sp, gn, min(a.chunk, sp["F"]), theta, lam_np)
    dH = np.abs(Hs - Hs_np).max() / max(np.abs(Hs_np).max(), 1e-300)
    dg = np.abs(gs - gs_np).max() / max(np.abs(gs_np).max(), 1e-300)
    dlam = np.abs(lam - lam_np).max() / max(np.abs(lam_np).max(), 1e-300)
    print(json.dumps(dict(F=sp["F"], B=sp["B"], cost=float(cost.sum()),
                          cost_np=float(c_np.sum()), dlam_rel=float(dlam),
                          Hs_rel=float(dH), gs_rel=float(dg),
                          ncp_res=float(rm), ncp_res_np=float(rm_np),
                          wall_gpu=float(t1 - t0)), indent=1))
