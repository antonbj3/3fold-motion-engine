#!/usr/bin/env python3
"""Standalone verification of the reproduction supplement.

Loads every exported file and runs one representative round-trip per family by
default; pass --all to run every scene.  Exits non-zero if any check fails.

    python verify.py
    python verify.py --all
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loader  # noqa: E402

REPRESENTATIVE = {"patch": "eight_box_tower", "incidence": "mass_ratio_point_chain",
                  "load_ordering": "packing_50", "ramp": "box"}


def check_patch(name):
    v = loader.verify_patch(name)
    ok = (v["dG"] == 0.0 and v["db"] == 0.0 and v["dmu"] == 0.0
          and v["dJ_patch"] == 0.0 and v["dC"] == 0.0
          and v["assembly_rel_inf"] < 1e-14)
    return ok, (f"dG={v['dG']:.1e} db={v['db']:.1e} assembly={v['assembly_rel_inf']:.1e} "
                f"pencil={v['pencil_rel_spectral']:.1e}")


def check_incidence(name):
    v = loader.verify_incidence(name)
    ok = v["dL"] == 0.0 and v["symmetry"] == 0.0
    return ok, f"dL={v['dL']:.1e} symmetry={v['symmetry']:.1e}"


def check_load_ordering(name):
    scene = next(s for s in loader.load_ordering_scenes()["scenes"] if s["name"] == name)
    d = loader.load_ordering_scene(name)
    import json
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "load_ordering", "measurements_corrected.json")) as f:
        corr = json.load(f)
    tie = corr["tie_rule"]["primary_tie_rel"]
    I, diag = loader.solve_flow_diag(d["edges"], d["weights_inverse_mass"],
                                     scene["num_nodes"], scene["n_grains"],
                                     scene["node_floor"], scene["node_foot"],
                                     scene["type"])
    m = loader.metrics(I, d["lam_n"], tie)
    ref = corr["scenes"][name]["predictors"]["resistance_flow_inverse_mass"]
    ds = abs(m["spearman"] - ref["spearman"])
    ok = ds < 1e-6 and diag["n_components"] >= 1
    return ok, (f"spearman={m['spearman']:+.6f} corrected_ref={ref['spearman']:+.6f} "
                f"d={ds:.1e} ties={m['n_ties_pred']} comps={diag['n_components']}")


def check_ramp(tag):
    geom = loader.ramp_geometry()
    meas = loader.ramp_measurements()
    sc = geom["scenes"][tag]
    phis = geom["directions_rad"]
    dmax = 0.0
    for r in meas["geometry_radius"][tag]:
        val, _ = loader.ramp_static_radius(sc, phis[r["direction"]])
        dmax = max(dmax, abs(val - r["L_full"]))
    # the wrench cone must be compared at the same geometry direction, not by
    # list position, and its labels must run 0..23 with a separate source row.
    wr = meas["wrench_radius"][tag]
    dirs = [e["direction"] for e in wr]
    labels_ok = dirs == list(range(len(phis))) and all("source_row" in e for e in wr)
    dwr = 0.0
    if labels_ok:
        by_dir = {e["direction"]: e["radius"] for e in wr}
        for r in meas["geometry_radius"][tag]:
            val, _ = loader.ramp_static_radius(sc, phis[r["direction"]])
            dwr = max(dwr, abs(val - by_dir[r["direction"]]))
    ok = dmax < 1e-9 and labels_ok and dwr < 1e-8
    return ok, (f"max |radius - geometry| = {dmax:.3e}; "
                f"wrench labels 0..{len(phis)-1} = {labels_ok}; "
                f"max |radius - wrench at same direction| = {dwr:.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    failures = 0
    checks = []
    patch_names = [s["name"] for s in loader.patch_scenes()["scenes"]]
    inc_names = [s["name"] for s in loader.incidence_scenes()["scenes"]]
    lo_names = [s["name"] for s in loader.load_ordering_scenes()["scenes"]]
    ramp_names = list(loader.ramp_geometry()["scenes"].keys())

    checks += [("patch", n, check_patch) for n in
               (patch_names if args.all else [REPRESENTATIVE["patch"]])]
    checks += [("incidence", n, check_incidence) for n in
               (inc_names if args.all else [REPRESENTATIVE["incidence"]])]
    checks += [("load_ordering", n, check_load_ordering) for n in
               (lo_names if args.all else [REPRESENTATIVE["load_ordering"]])]
    checks += [("ramp", n, check_ramp) for n in
               (ramp_names if args.all else [REPRESENTATIVE["ramp"]])]

    for fam, name, fn in checks:
        ok, detail = fn(name)
        failures += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {fam:14s} {name:28s} {detail}")

    # every exported file loads
    loader.patch_scenes(); loader.incidence_scenes(); loader.load_ordering_scenes()
    loader.ramp_geometry(); loader.ramp_measurements(); loader.load_ordering_measurements()
    print("[PASS] all exported files loaded")
    if failures:
        print(f"{failures} check(s) failed")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
