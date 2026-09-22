#!/usr/bin/env python3
"""Standalone normalized de Saxce/Cadoux ADMM solver for the `contact-scene-v1` scenes.

This is the solver dependency that produced `dimensionless_stop.json` (the
three-scene x five-unit absolute/physical/dimensionless stop comparison).  It is
a self-contained NumPy reference, written from the measurement's definition:
proximal ADMM with exact Coulomb-cone projection in `z`, the de Saxce outer
fixed point, the dimensionless inner-block target, and the natural-map residual
as the termination criterion.  It imports nothing from any project build tree.

Contract (matches the solver used for the stored rows):

    solve(G, b, mu, rho0, iters, tol, shift_mode="block", scales=None,
          norm_target=False)  ->  (z, stats)

Units: the measurement re-expresses one physical problem in five unit systems
via `G -> c*G`, `b -> b/a` (length unit a m, mass unit c kg, time unchanged),
with the impulse `lam -> lam/(c*a)` and `rho -> c*rho`.  The three stopping rules
are

    absolute       tol = 1e-10                        (written in the current unit)
    physical       tol = 1e-10/(a*c)                  (fixed 1e-10 N s, unit-consistent)
    dimensionless  tol = 1e-10*lam_ref(unit)          (a pure number threshold)

The dimensionless inner-block target is used by the physical and dimensionless
arms (`norm_target=True`, `scales` supplied); `ctrl=None` throughout means the
penalty is the fixed structural value, so the comparison isolates the outer stop.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "proj_cone", "desaxce", "natural_residual", "ref_scales",
    "solve", "positive_spectrum_ratio",
]


def _import_scipy():
    import scipy.sparse as sp
    import scipy.sparse.linalg as sl
    return sp, sl


# --------------------------------------------------------------------- cone algebra
def proj_cone(x, mu):
    """Euclidean projection onto the Coulomb cone K_mu = {||x_t|| <= mu x_n}."""
    x = np.asarray(x, dtype=np.float64)
    single = x.ndim == 1
    X = np.atleast_2d(x).copy()
    m = np.broadcast_to(np.atleast_1d(np.asarray(mu, float)), (X.shape[0],))
    xn, xt = X[:, 0], X[:, 1:3]
    nt = np.linalg.norm(xt, axis=1)
    out = np.zeros_like(X)
    zero = m <= 0.0
    if np.any(zero):
        out[zero, 0] = np.maximum(0.0, xn[zero])
    pos = ~zero
    if np.any(pos):
        inside = pos & (nt <= m * xn)
        polar = pos & (m * nt <= -xn)
        mid = pos & ~inside & ~polar
        out[inside] = X[inside]
        if np.any(mid):
            a = (m[mid] * nt[mid] + xn[mid]) / (1.0 + m[mid] ** 2)
            out[mid, 0] = a
            out[mid, 1:3] = ((m[mid] * a) / nt[mid])[:, None] * xt[mid]
    return out[0] if single else out


def desaxce(u, mu):
    """de Saxce correction Gamma(u) = (mu ||u_t||, 0, 0)."""
    u = np.asarray(u, dtype=np.float64)
    single = u.ndim == 1
    U = np.atleast_2d(u)
    m = np.broadcast_to(np.atleast_1d(np.asarray(mu, float)), (U.shape[0],))
    g = np.zeros_like(U)
    g[:, 0] = m * np.linalg.norm(U[:, 1:3], axis=1)
    return g[0] if single else g


def _rho_of(G, n_c):
    rho = np.empty(n_c)
    for c in range(n_c):
        Gcc = G[3 * c:3 * c + 3, 3 * c:3 * c + 3]
        s = np.linalg.norm(Gcc, ord=2)
        rho[c] = 1.0 / s if s > 0 else 1.0
    return rho


def natural_residual(lam, G, b, mu, rho=None):
    lam = np.asarray(lam, float)
    n_c = len(mu)
    if rho is None:
        rho = _rho_of(G, n_c)
    u = (G @ lam + b).reshape(n_c, 3)
    L = lam.reshape(n_c, 3)
    s = desaxce(u, mu)
    ref = (L - proj_cone(L - rho[:, None] * (u + s), mu)).ravel()
    return float(np.max(np.abs(ref)))


def positive_spectrum_ratio(G):
    ev = np.linalg.eigvalsh(0.5 * (G + G.T))
    pos = ev[ev > 1e-9 * max(float(ev[-1]), 1e-300)]
    return float(pos[-1] / pos[0]) if pos.size else float("inf")


def ref_scales(G, b):
    v_ref = float(np.max(np.abs(b))) or 1.0
    g_ref = float(np.max(np.abs(G).sum(axis=1))) or 1.0
    return {"v_ref": v_ref, "g_ref": g_ref, "lam_ref": v_ref / g_ref}


# --------------------------------------------------------------------- solver
def solve(G, b, mu, rho0, iters=30000, tol=1e-8, shift_mode="block", scales=None,
          norm_target=False, linear_solver="cholesky"):
    """Fixed-penalty proximal ADMM + de Saxce outer fixed point.  Returns (z, stats).

    shift_mode="block" refreshes Gamma(Gz+b) between inner blocks and holds it
    fixed inside a block (the measured schedule).  `scales` supplies the
    dimensionless reference scales; with `norm_target=True` the inner-block
    impulse threshold is the dimensionless form `tgt_hat * lam_ref`.

    `linear_solver` is the x-update factorization of G + rho I:
      "cholesky"  dense Cholesky (NumPy; exact, O(n^3) per solve)
      "sparse_lu" sparse LU (SciPy; the measured `archived ADMM solver` linear algebra).
    The two agree on 44/45 stored cells; the single rounding-sensitive cell
    (`three_body_column_normal` absolute `Mm3`) is reproduced by "sparse_lu".
    """
    G = np.asarray(G, float)
    b = np.asarray(b, float)
    mu = np.asarray(mu, float)
    n_c = len(mu)
    n = 3 * n_c
    rho_nat = _rho_of(G, n_c)
    lam_ref = None if scales is None else float(scales["lam_ref"])
    v_ref = None if scales is None else float(scales["v_ref"])
    tol_hat = (tol / lam_ref) if lam_ref else None
    use_norm = bool(norm_target)

    rho = float(rho0)
    if linear_solver == "sparse_lu":
        _scipy_sparse, _scipy_linalg = _import_scipy()
        factor = _scipy_linalg.splu(
            _scipy_sparse.csc_matrix(G) + _scipy_sparse.diags(np.full(n, rho)),
            permc_spec="COLAMD")

        def apply(rhs):
            return factor.solve(rhs)
    else:
        A = G + rho * np.eye(n)
        P = np.linalg.cholesky(A)

        def apply(rhs):
            y = np.linalg.solve(P, rhs)
            return np.linalg.solve(P.T, y)

    n_chol = 1

    z = np.zeros(n)
    x = np.zeros(n)
    gamma = np.zeros(n)

    s_shift = np.zeros(n_c)
    prev_s = None
    hist = []

    def inner_target():
        if lam_ref is None or not use_norm:
            return max(tol, 0.1 * (
                float(np.max(np.abs(b))) if prev_s is None
                else max(float(np.max(np.abs(s_shift - prev_s))), tol)))
        q = 1.0 if prev_s is None else \
            max(float(np.max(np.abs(s_shift - prev_s))) / v_ref, tol_hat)
        return max(tol_hat, 0.1 * max(1.0, q)) * lam_ref

    def admm_block(g_lin, budget, target):
        nonlocal x, z, gamma
        for _ in range(int(budget)):
            rhs = -(g_lin + gamma - rho * z)
            x = apply(rhs)
            z = proj_cone((x + gamma / rho).reshape(n_c, 3), mu).reshape(-1)
            gamma = gamma + rho * (x - z)
            r = natural_residual(z, G, b, mu, rho_nat)
            hist.append(r)
            if r < tol:
                return True
            if len(hist) >= iters:
                return False
            if target is not None and float(np.linalg.norm(x - z)) < target \
                    and len(hist) >= 5:
                return False
        return False

    n_outer = 0
    while len(hist) < iters and n_outer < 10 * iters:
        n_outer += 1
        g_lin = b.copy()
        g_lin[0::3] += mu * s_shift
        budget = 1 if shift_mode == "per_iter" else iters
        done = admm_block(g_lin, budget, None if shift_mode == "per_iter"
                          else inner_target())
        if done:
            break
        u = (G @ z + b).reshape(n_c, 3)
        prev_s = s_shift
        s_shift = np.linalg.norm(u[:, 1:3], axis=1)

    st = {"inner": len(hist), "outer": n_outer,
          "converged": bool(len(hist) and hist[-1] < tol),
          "res": float(hist[-1]) if hist else float("nan"),
          "n_chol": n_chol, "shift_mode": shift_mode, "tol_used": float(tol),
          "lam_ref": lam_ref, "v_ref": v_ref}
    return z, st


if __name__ == "__main__":
    import json

    def _demo():
        G = np.eye(3)
        b = np.array([-0.1, 0.0, 0.0])
        mu = np.array([0.5])
        z, st = solve(G, b, mu, 0.1, iters=3000, tol=1e-10)
        print("selftest", st["converged"], st["inner"])

    _demo()
