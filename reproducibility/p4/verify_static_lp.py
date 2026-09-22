#!/usr/bin/env python3
"""Self-contained check of the static method: rebuild the wrench map from the
frozen contact geometry, solve the 16-ray radial LP with an independent solver,
and compare both with subset enumeration and with the frozen reference radii.

Run:  python3 verify_static_lp.py
Prints a JSON summary and exits non-zero if any frozen value is not reproduced.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np

import scenes as scene_mod
from friction_polygon import (K, POLY_SCALE, POLY_RADIAL_ERROR_REL,
                              POLY_INRADIUS_REL, POLY_CIRCUMRADIUS_REL)
from lp_reference import solve_lp, enumerate_radius

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"


def main():
    exact = json.loads((RAW / "static_exactness.json").read_text())
    phis = [math.radians(d) for d in exact["reference_directions_deg"]]
    scenes = {s.name: s for s in scene_mod.iter_scenes()}
    out = {"scenes": {}, "polygon": {}}
    ok = True

    poly = scene_mod.polygon_info()
    out["polygon"] = {
        "sides": K,
        "straddles_circle": bool(POLY_CIRCUMRADIUS_REL > 1.0 > POLY_INRADIUS_REL),
        "circumradius_rel": POLY_CIRCUMRADIUS_REL,
        "inradius_rel": POLY_INRADIUS_REL,
        "analytic_max_radial_error_rel": POLY_RADIAL_ERROR_REL,
        "raw_scale_matches": abs(poly["scale"] - POLY_SCALE) < 1e-15,
        "raw_analytic_error_matches": abs(poly["analytic_max_relative_error"]
                                          - POLY_RADIAL_ERROR_REL) < 1e-18,
    }
    ok &= out["polygon"]["raw_scale_matches"] and out["polygon"]["raw_analytic_error_matches"]

    for row in exact["scenes"]:
        name = row["name"]
        sc = scenes[name]
        W, f0, F = scene_mod.assemble(sc)
        lp = [solve_lp(W, f0, F, p) for p in phis]
        frozen = row["lp_radii"]
        lp_diff = max(abs(a - b) / b for a, b in zip(lp, frozen))
        enum_res = []
        for p in phis:
            full = solve_lp(W, f0, F, p)
            enum, _ = enumerate_radius(W, f0, F, p)
            enum_res.append(abs(full - enum) / full)
        max_res = max(enum_res)
        rec = {
            "n_contacts": sc.n_contacts,
            "max_rel_diff_lp_vs_frozen": lp_diff,
            "max_rel_residual_lp_vs_enumeration": max_res,
            "frozen_max_rel_residual": row["max_relative_residual"],
            "residual_order_matches": (max_res <= 1e-10)
            and (row["max_relative_residual"] <= 1e-10),
            "polygon_error_soc_pct": row["polygon_error_soc_pct"],
        }
        out["scenes"][name] = rec
        ok &= lp_diff < 1e-9 and rec["residual_order_matches"]
        print(f"{name:7s} contacts={sc.n_contacts} "
              f"max|LP-frozen|/frozen={lp_diff:.3e} "
              f"enum_residual={max_res:.3e} frozen={row['max_relative_residual']:.3e}")

    out["all_checks_pass"] = bool(ok)
    (HERE / "verify_static_lp_result.json").write_text(
        json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"all_checks_pass": bool(ok)}, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
