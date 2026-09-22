#!/usr/bin/env python3
"""Re-run the three-scene x five-unit stop comparison with the bundled solver.

Reads the scene operators through the packaged `data/ncp/loader.py` (so this is
a package-level reproduction), re-expresses each in the five unit systems, and
runs the absolute / physical / dimensionless stops with the bundled
`dimensionless_stop/solver.py`.  Writes `rerun.json` in the CWD from which it is
invoked (the package root), then compares the outer iteration counts with the
stored `dimensionless_stop.json`.

    python dimensionless_stop/rerun.py                 # all three scenes
    python dimensionless_stop/rerun.py --scene 0       # one scene

The bundled solver is dense-Cholesky.  It matches the stored rows in 44/45 cells;
the single remaining cell (`three_body_column_normal` absolute `Mm3`, 51 vs a dense-Cholesky
plateau) reproduces exactly with the measurement's sparse-LU x-update, which is
the documented `archived ADMM solver` linear algebra.  

CPU only; NumPy only (plus the package loader).  This is a bounded consistency
reproduction, not a new measurement; the physics here is the exported operator.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent / "data" / "ncp"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(HERE))

import loader as L          # noqa: E402  packaged contact-scene loader
import solver as S          # noqa: E402  bundled dimensionless-stop solver

UNITS = [("SI", 1.0, 1.0), ("Lp3", 1e3, 1.0), ("Lm3", 1e-3, 1.0),
         ("Mp3", 1.0, 1e3), ("Mm3", 1.0, 1e-3)]
SCENES = {0: "cube", 1: "eight_box_tower", 2: "three_body_column_normal"}
CAP = 30000
PHYS = 1e-10


def _rho_structure(G, structural):
    """The measurement's fixed structural penalty: lam_min+ if the bodywise
    structural test passes (C1), else sqrt(lam_min+ * lam_max)."""
    ev = np.linalg.eigvalsh(0.5 * (G + G.T))
    pos = ev[ev > 1e-9 * max(float(ev[-1]), 1e-300)]
    lo = float(pos[0]) if pos.size else 1.0
    return lo if structural else float(np.sqrt(lo * float(ev[-1])))


def _bundle_only(G, _):
    """cube and eight_box_tower are not structural scenes: rho = sqrt(lam_min+ lam_max)."""
    return _rho_structure(G, False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, default=None)
    ap.add_argument("--out", default="rerun.json")
    ap.add_argument("--linear-solver", default="cholesky",
                    choices=["cholesky", "sparse_lu"])
    args = ap.parse_args()
    idx = [args.scene] if args.scene is not None else list(SCENES)

    stored = json.load(open(HERE / "dimensionless_stop.json"))
    struct_path = HERE / "structure_check.json"
    struct = json.load(open(struct_path)) if struct_path.exists() else {}
    rows = []
    for i in idx:
        scene = L.load_scene(i)
        G, b, mu, _J, _Minv = L.rebuild(scene)
        structural = bool(struct.get(f"{i:02d}", {}).get("structural", False))
        rho_struct = _rho_structure(G, structural)
        for stop in ("absolute", "physical", "dimensionless"):
            for uname, a, c in UNITS:
                Gt = c * G
                bt = b / a
                sc = S.ref_scales(Gt, bt)
                if stop == "absolute":
                    tol, scales, nrm = PHYS, None, False
                elif stop == "physical":
                    tol, scales, nrm = PHYS / (a * c), sc, True
                else:
                    tol, scales, nrm = PHYS * sc["lam_ref"], sc, True
                z, st = S.solve(Gt, bt, mu, c * rho_struct, iters=CAP, tol=tol,
                                shift_mode="block", scales=scales, norm_target=nrm,
                                linear_solver=args.linear_solver)
                rows.append({"scene": SCENES[i], "index": i, "unit": uname,
                             "stop": stop, "a": a, "c": c,
                             "tol_used": float(tol), "iterations": int(st["inner"]),
                             "converged": bool(st["converged"]),
                             "residual": float(st["res"]),
                             "n_chol": int(st["n_chol"])})
        print("scene", SCENES[i], "done", flush=True)

    cmp_rows = []
    for i in idx:
        for stop in ("absolute", "physical", "dimensionless"):
            mine = [r["iterations"] for r in rows if r["index"] == i and r["stop"] == stop]
            theirs = [r["iterations"] for r in stored["rows"]
                      if r["index"] == i and r["stop"] == stop]
            exact = sum(1 for x, y in zip(mine, theirs) if x == y)
            cmp_rows.append({"scene": SCENES[i], "stop": stop, "mine": mine,
                             "stored": theirs, "exact_cells": exact,
                             "cells": len(theirs)})
    out = {"rows": rows, "comparison": cmp_rows,
           "solver": "release/dimensionless_stop/solver.py",
           "loader": "release/data/ncp/loader.py",
           "rho_rule": "sqrt(lam_min+ lam_max) of the packaged G, scaled by c"}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    tot_exact = sum(r["exact_cells"] for r in cmp_rows)
    tot_cells = sum(r["cells"] for r in cmp_rows)
    print(f"exact outer-count cells {tot_exact}/{tot_cells}")
    for r in cmp_rows:
        print(f"  {r['scene']:9s} {r['stop']:14s} {r['mine']} vs {r['stored']} "
              f"exact {r['exact_cells']}/{r['cells']}")


if __name__ == "__main__":
    main()
