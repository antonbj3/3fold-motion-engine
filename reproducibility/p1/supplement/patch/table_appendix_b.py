#!/usr/bin/env python3
"""Appendix B: dense and block elimination on the patch chains.

Recomputes, from the exported geometry alone:

  * the block bandwidth of G_patch (1 = block tridiagonal on a chain),
  * the agreement between a dense Cholesky solve and a block-tridiagonal
    elimination of (G_patch + rho I) x = r on a fixed right-hand side.

The outer iteration counts and the "max abs impulse difference" of the delivered
table are aggregated from measurements_appendix_b.json (they are saved solver
runs, not recomputed here).

Run:  python table_appendix_b.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import loader  # noqa: E402


def block_bandwidth(G, blk):
    n = G.shape[0] // blk
    bw = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.any(G[blk * i:blk * i + blk, blk * j:blk * j + blk] != 0.0):
                bw = max(bw, abs(i - j))
    return bw


def block_tridiag_solve(G, rho, rhs, blk=6):
    n = G.shape[0] // blk
    A = G + rho * np.eye(G.shape[0])
    # forward elimination of the 6x6 blocks
    diag = [A[blk * i:blk * i + blk, blk * i:blk * i + blk].copy() for i in range(n)]
    off = [A[blk * i:blk * i + blk, blk * (i + 1):blk * (i + 1) + blk].copy()
           for i in range(n - 1)]
    r = [rhs[blk * i:blk * i + blk].copy() for i in range(n)]
    for i in range(1, n):
        m = np.linalg.solve(diag[i - 1], off[i - 1])
        diag[i] = diag[i] - off[i - 1].T @ m
        r[i] = r[i] - off[i - 1].T @ np.linalg.solve(diag[i - 1], r[i - 1])
    x = [None] * n
    x[-1] = np.linalg.solve(diag[-1], r[-1])
    for i in range(n - 2, -1, -1):
        x[i] = np.linalg.solve(diag[i], r[i] - off[i] @ x[i + 1])
    return np.concatenate(x)


def main():
    meas = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "measurements_appendix_b.json")))
    print(f"{'scene':28s} {'blk bw':>6s} {'dense-vs-block':>15s} "
          f"{'dense it':>9s} {'block it':>9s} {'saved max|dz|':>14s}")
    for name in ("eight_box_tower", "eight_box_tower_rolling",
                 "mass_ratio_column", "mass_ratio_column_rolling"):
        op = loader.patch_operator(name)
        G = op["G"]
        bw = block_bandwidth(G, 6)
        rho = float(np.trace(G) / G.shape[0])
        rng = np.random.default_rng(0)
        rhs = rng.standard_normal(G.shape[0])
        xd = np.linalg.solve(G + rho * np.eye(G.shape[0]), rhs)
        xb = block_tridiag_solve(G, rho, rhs, 6)
        m = meas["scenes"][name]
        print(f"{name:28s} {bw:6d} {np.max(np.abs(xd - xb)):15.3e} "
              f"{m['dense_updates']:9d} {m['block_updates']:9d} "
              f"{m['max_abs_impulse_difference']:14.3e}")


if __name__ == "__main__":
    main()
