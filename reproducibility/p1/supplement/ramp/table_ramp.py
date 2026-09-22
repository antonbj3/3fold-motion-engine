#!/usr/bin/env python3
"""Section 6.1: quasistatic slip boundary.

What this script does:

  * recomputes the static directional radius from the exported patch geometry
    (Proposition 7: min of the friction bound and the centre-of-pressure bound,
    over the interface and ground patch), and compares it with the saved
    geometry radius and the saved wrench-cone radius;
  * aggregates the saved ramp brackets and reproduces the reported mean relative
    error per ramp time over the 72 scene-direction pairs.

It does NOT rerun the contact simulation: the bracket midpoints are the exported
outputs of the delivered runs.

Run:  python table_ramp.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import loader  # noqa: E402


def main():
    geom = loader.ramp_geometry()
    meas = loader.ramp_measurements()
    phis = geom["directions_rad"]

    print("static radius (recomputed) vs saved geometry and wrench radii")
    print(f"{'scene':8s} {'direction':>9s} {'recomputed':>11s} {'geometry':>10s} "
          f"{'wrench':>10s} {'d geom':>9s} {'d wrench':>10s}")
    max_d_geom = 0.0
    max_d_wr = 0.0
    for tag, sc in geom["scenes"].items():
        wr_by_dir = {e["direction"]: e["radius"] for e in meas["wrench_radius"][tag]}
        if sorted(wr_by_dir) != list(range(len(phis))):
            raise SystemExit(f"{tag}: wrench directions are not 0..{len(phis)-1}")
        for r in meas["geometry_radius"][tag]:
            i = r["direction"]
            val, _ = loader.ramp_static_radius(sc, phis[i])
            wr = wr_by_dir[i]
            max_d_geom = max(max_d_geom, abs(val - r["L_full"]))
            max_d_wr = max(max_d_wr, abs(val - wr))
            if i in (0, 6, 12, 18):
                print(f"{tag:8s} {i:9d} {val:11.6f} {r['L_full']:10.6f} "
                      f"{wr:10.6f} {abs(val-r['L_full']):9.1e} {abs(val-wr):10.1e}")
    print(f"max |recomputed - geometry| = {max_d_geom:.3e}")
    print(f"max |recomputed - wrench|   = {max_d_wr:.3e}")

    print()
    print("ramp brackets: mean relative error vs the static radius, 72 pairs")
    print(f"{'ramp T [s]':>11s} {'mean rel wrench [%]':>20s} "
          f"{'mean rel geometry [%]':>21s} {'glide pairs':>11s}")
    by_T = {}
    for key, row in meas["bracket"].items():
        by_T.setdefault(row["ramp_T_s"], []).append(row)
    for T in sorted(by_T):
        rows = by_T[T]
        if T == 0.0:
            continue
        mw = float(np.mean([r["mean_rel_wrench_pct"] for r in rows]))
        mg = float(np.mean([r["mean_rel_geometry_pct"] for r in rows]))
        ng = sum(r["glide_count"] for r in rows)
        print(f"{T:11.2f} {mw:20.6f} {mg:21.6f} {ng:>11d}")
    lh = meas["long_horizon_box_direction_zero"]
    print(f"box direction zero, T={lh['T50_s']:.0f}s: "
          f"{lh['T50_rel_wrench_pct']:.6f} %")
    print(f"box direction zero, T={lh['T100_s']:.0f}s: "
          f"{lh['T100_rel_wrench_pct']:.6f} %")


if __name__ == "__main__":
    main()
