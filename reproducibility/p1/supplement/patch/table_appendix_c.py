#!/usr/bin/env python3
"""Appendix C: scalar incidence linear systems.

Recomputes, from the exported systems alone, a direct solve and a
Jacobi-preconditioned conjugate-gradient solve of

    A x = r,   A = L + rho I  (rho = 0 when L is positive definite),

to relative residual <= 1e-10.  The AMG iteration counts in the delivered table
are aggregated from measurements_appendix_c.json (saved runs of an external AMG
library, not recomputed here).

Run:  python table_appendix_c.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import loader  # noqa: E402

TOL = 1e-10


def is_spd(L):
    if L.shape[0] <= 1400:
        try:
            np.linalg.cholesky(L)
            return True
        except np.linalg.LinAlgError:
            return False
    return None


def direct_solve(A, b):
    if A.shape[0] <= 1400:
        x = np.linalg.solve(A, b)
    else:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        lu = spla.splu(sp.csc_matrix(A), permc_spec="MMD_AT_PLUS_A")
        x = lu.solve(b)
    return x


def pcg_jacobi(A, b, tol=TOL, maxit=5000):
    d = np.diag(A)
    inv = np.where(d != 0.0, 1.0 / d, 0.0)
    x = np.zeros_like(b)
    r = b.copy()
    nb = float(np.linalg.norm(b))
    z = inv * r
    p = z.copy()
    rz = float(r @ z)
    it = 0
    for it in range(1, maxit + 1):
        Ap = A @ p
        pAp = float(p @ Ap)
        if pAp <= 0:
            break
        a = rz / pAp
        x += a * p
        r -= a * Ap
        if float(np.linalg.norm(r)) <= tol * nb:
            break
        z = inv * r
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, it


def main():
    meas = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "measurements_appendix_c.json")))
    print(f"{'scene':28s} {'n_c':>5s} {'rho':>10s} {'direct res':>11s} "
          f"{'jacobi it':>9s} {'jacobi res':>11s} {'amg-sa it':>9s} {'amg-rs it':>9s}")
    for s in loader.incidence_scenes()["scenes"]:
        name = s["name"]
        sysd = loader.incidence_system(name)
        L = loader.scalar_incidence(sysd["body_pairs"], sysd["inv_mass"])
        b = sysd["rhs"]
        spd = is_spd(L)
        rho = 0.0 if spd else 1e-2 * float(np.max(np.diag(L)))
        A = L + rho * np.eye(L.shape[0])
        xd = direct_solve(A, b)
        rd = float(np.linalg.norm(A @ xd - b) / np.linalg.norm(b))
        xj, itj = pcg_jacobi(A, b)
        rj = float(np.linalg.norm(A @ xj - b) / np.linalg.norm(b))
        m = meas["scenes"][name]["solvers"]
        sa = m.get("amg_sa", {}).get("iterations", "-")
        rs = m.get("amg_rs", {}).get("iterations", "-")
        print(f"{name:28s} {L.shape[0]:5d} {rho:10.4g} {rd:11.3e} "
              f"{itj:9d} {rj:11.3e} {sa:>9} {rs:>9}")


if __name__ == "__main__":
    main()
