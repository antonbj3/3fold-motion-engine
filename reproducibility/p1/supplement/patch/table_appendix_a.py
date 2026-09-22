#!/usr/bin/env python3
"""Appendix A: assembly, rank and spectral correspondence.

Recomputes, from the exported geometry alone:

  * the number of patches and the number of bodies of degree > 2,
  * the rank of the point Jacobian and of the patch Jacobian,
  * the relative infinity error between G_patch and the graph assembly,
  * the relative spectral error of the vertex pencil (G_patch vs (Lv, Minv)).

This script recomputes the linear algebra.  It does not aggregate saved
measurements and does not solve any contact problem.

Run:  python table_appendix_a.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import loader  # noqa: E402


def main():
    print(f"{'scene':28s} {'patches':>7s} {'deg>2':>5s} {'point rk/rows':>13s} "
          f"{'patch rk/rows':>13s} {'assembly inf':>12s} {'pencil rel':>11s}")
    for s in loader.patch_scenes()["scenes"]:
        name = s["name"]
        v = loader.verify_patch(name)
        sc = loader.patch_scene(name)
        # bodies of degree > 2
        deg = {}
        for pa in sc["patches"]:
            for i in (pa["a"], pa["b"]):
                if i >= 0:
                    deg[i] = deg.get(i, 0) + 1
        nhigh = sum(1 for d in deg.values() if d > 2)
        print(f"{name:28s} {v['n_patches']:7d} {nhigh:5d} "
              f"{v['rank_J_points']:5d}/{v['rows_J_points']:<7d} "
              f"{v['rank_J_patch']:5d}/{v['rows_J_patch']:<7d} "
              f"{v['assembly_rel_inf']:12.3e} {v['pencil_rel_spectral']:11.3e}")


if __name__ == "__main__":
    main()
