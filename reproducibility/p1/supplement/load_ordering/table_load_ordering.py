#!/usr/bin/env python3
"""Section 5: load ordering from effective resistance.

What this script does:

  * recomputes the three resistance flows on an independent direct reference
    solution (sparse LU, one gauge node pinned per connected component) and the
    three geometric baselines from the exported contact graphs;
  * ranks them with the a-priori dimensionless tie radius
    ``tie_rel = 1e-9`` of each series' maximum, using leader clustering (a
    stable sort, then a value joins the current cluster while it is at most the
    radius above the cluster's first value; cluster members share the average
    rank).  The interval over the admissible tie-radius grid and
    the MINRES convergence sweep are printed as the numerical range;
  * compares the recomputed values with the delivered (aggregated) metrics and
    with the corrected reference ``measurements_corrected.json``;
  * aggregates the k=64 resistance-sketch metrics from the saved measurements
    (the sketch implementation is an external module and is not shipped).

It does NOT recompute the contact physics: the contact graph and the contact
loads are the exported outputs of the delivered DEM/MPM runs.

Run:  python table_load_ordering.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import loader  # noqa: E402

TIE_REL = 1e-9
MODELS = ["inverse_mass", "unweighted", "hertz_stiffness"]


def baselines(scene, d, n_c, seed=0):
    edges = np.asarray(d["edges"], int)
    num_nodes = int(scene["num_nodes"])
    deg = np.zeros(num_nodes)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    degree = deg[edges[:, 0]] + deg[edges[:, 1]]
    pts = d["contact_points"]
    if scene["type"] == "foot":
        dist = -np.linalg.norm(pts - d["foot_pos"], axis=1)
    else:
        dist = -pts[:, 2]
    rng = np.random.default_rng(seed)
    return {"baseline_contact_degree": degree,
            "baseline_distance_to_load": dist,
            "baseline_random": rng.random(n_c)}


def main():
    saved = loader.load_ordering_measurements()
    corr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "measurements_corrected.json")
    corr = json.load(open(corr_path)) if os.path.exists(corr_path) else None
    if corr is not None:
        tr = corr["tie_rule"]
        print(f"tie rule: leader clustering at tol = {tr['primary_tie_rel']:g} "
              f"* max|series| (members share the average rank); admissible grid "
              f"{tr['admissible_grid']}")
    print(f"{'scene':16s} {'predictor':30s} {'recomputed':>10s} {'corrected':>10s} "
          f"{'saved':>10s} {'interval':>22s} {'p10 rec':>8s} {'p10 saved':>9s}")

    for scene in loader.load_ordering_scenes()["scenes"]:
        name = scene["name"]
        d = loader.load_ordering_scene(name)
        n_c = scene["n_contacts"]
        truth = d["lam_n"]
        diag = None
        preds = {}
        for wn in MODELS:
            if wn == "inverse_mass":
                I, diag = loader.solve_flow_diag(
                    d["edges"], d[f"weights_{wn}"], scene["num_nodes"],
                    scene["n_grains"], scene["node_floor"], scene["node_foot"],
                    scene["type"])
            else:
                I, _ = loader.solve_flow(
                    d["edges"], d[f"weights_{wn}"], scene["num_nodes"],
                    scene["n_grains"], scene["node_floor"], scene["node_foot"],
                    scene["type"])
            preds[f"resistance_flow_{wn}"] = I
        preds.update(baselines(scene, d, n_c))
        sm = saved["scenes"][name]["metrics"]
        cpred = corr["scenes"][name]["predictors"] if corr else {}
        for pname, pred in preds.items():
            m = loader.metrics(pred, truth, TIE_REL)
            iv = cpred.get(pname, {}).get("tie_interval_spearman")
            ivs = f"[{iv[0]:.4f},{iv[1]:.4f}]" if iv else "--"
            print(f"{name:16s} {pname:30s} {m['spearman']:+10.5f} "
                  f"{(cpred.get(pname, {}).get('spearman', float('nan'))):+10.5f} "
                  f"{sm[pname]['spearman']:+10.5f} {ivs:>22s} "
                  f"{m['prec_10']:8.4f} {sm[pname]['prec_10']:9.4f}")
        sk = sm["resistance_sketch_k64"]
        print(f"{name:16s} {'resistance_sketch_k64 (saved)':30s} {'--':>10s} "
              f"{'--':>10s} {sk['spearman']:+10.4f} {'aggregated':>22s} "
              f"{'--':>8s} {sk['prec_10']:9.4f}")
        if diag is not None:
            print(f"    direct: rel_residual={diag['rel_residual']:.2e} "
                  f"backward_error={diag['backward_error']:.2e} "
                  f"components={diag['n_components']} "
                  f"gauge_nodes={diag['n_gauge_nodes']}")
        if corr is not None:
            sw = corr["scenes"][name]["solver_sweep_inverse_mass"]
            print("    MINRES sweep (inverse_mass): " + "; ".join(
                f"{x['method']}: rel_flow={x['rel_flow_error_vs_direct']:.2e} "
                f"spearman={x['spearman']:+.5f}" for x in sw))


if __name__ == "__main__":
    main()
