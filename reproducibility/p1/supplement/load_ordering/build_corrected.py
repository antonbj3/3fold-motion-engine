#!/usr/bin/env python3
"""Rebuild the load-ordering metrics on an independent reference solution.

Background.  The delivered load-ordering metrics were computed with
``loader.solve_flow``'s old branch: a dense solve for grounded systems up to
2000 nodes and MINRES (``rtol=1e-5``) above.  The grounded contact Laplacian is
ill-conditioned and singular on every component that does not touch the
grounded node, so the delivered flows depend on the solver and its tolerance;
exactly-equal currents arising from symmetry are then split differently by
different solvers, which moves Spearman and precision.

This script replaces that reference by an independent sparse direct solve (one
gauge node pinned per connected component) and ranks the currents with an
a-priori dimensionless tie radius.  It writes ``measurements_corrected.json``:

  * ``predictors``: Spearman and precision@5/10% at the pre-registered tie
    radius, with the interval over the tie-radius grid and over the MINRES
    convergence sweep (solver sensitivity);
  * ``diagnostics``: relative residual, componentwise backward error, number of
    connected components and gauge nodes for the direct solve of each scene.

The tie radius is fixed before the comparison: ``tie_rel = 1e-9`` of each
series' own maximum, applied as leader clustering (stable sort; a value joins
the current cluster while it is at most the radius above the cluster's FIRST
value; members share the average rank).  Rationale (dimensionless): the direct
LU backward error is ~n * eps <= 1e-11 for these systems and two independent
solves of the same grounded system (different gauge pin, different sparse
solver) agree to <=5.54e-12 relative to the maximum edge current, while the top
decile of the currents is separated by many orders of magnitude; 1e-9 is
therefore above numerical noise and below any physically meaningful difference.
The grid around it makes the sensitivity explicit instead of hiding a single
number.

Run:  python build_corrected.py [--write]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import loader  # noqa: E402

# admissible tie radius, fixed a priori: the lower end (1e-12) is the
# observed direct-solver noise; the upper end (1e-8) is still far below the
# physical resolution of the current ranking.  0 = exact equality, reported as
# the delivered convention.  1e-6 is NOT a tie radius but a coarse-graining
# scale and is kept only as a breakdown point.
TIE_GRID = [0.0, 1e-12, 1e-10, 1e-9, 1e-8]
TIE_GRID_BREAKDOWN = [1e-6]
TIE_PRIMARY = 1e-9
WEIGHTS = ["inverse_mass", "unweighted", "hertz_stiffness"]
MINRES_RTOL = [1e-5, 1e-8, 1e-10, 1e-12]


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


def predictor_row(pred, truth):
    prim = loader.metrics(pred, truth, TIE_PRIMARY)
    sp = [loader.metrics(pred, truth, tr)["spearman"] for tr in TIE_GRID]
    brk = {str(tr): loader.metrics(pred, truth, tr)["spearman"]
           for tr in TIE_GRID_BREAKDOWN}
    row = {"spearman": prim["spearman"], "prec_5": prim["prec_5"],
           "prec_10": prim["prec_10"],
           "tie_interval_spearman": [float(min(sp)), float(max(sp))],
           "spearman_by_tie_rel": {str(tr): float(v) for tr, v in zip(TIE_GRID, sp)},
           "spearman_at_breakdown_tie_rel": brk,
           "n_ties_pred": prim["n_ties_pred"],
           "n_ties_truth": prim["n_ties_truth"]}
    return row


def main():
    write = "--write" in sys.argv
    scenes = loader.load_ordering_scenes()["scenes"]
    out = {"description": "Load-ordering metrics recomputed on an independent "
                          "direct reference solution with an a-priori "
                          "dimensionless tie radius.",
           "tie_rule": {"kind": "stable sort, then leader clustering at "
                                "tol = tie_rel * max|series|: a value joins the "
                                "current cluster while (value - cluster_first_value) "
                                "<= tol; members share the average rank",
                        "primary_tie_rel": TIE_PRIMARY,
                        "admissible_grid": TIE_GRID,
                        "breakdown_grid": TIE_GRID_BREAKDOWN,
                        "justification": "direct LU backward error <= ~1e-11 for "
                                         "n <= 6.4e4; two independent solves of the "
                                         "same grounded system (different gauge pin, "
                                         "different sparse solver) agree to <=5.54e-12 "
                                         "relative to the maximum edge current; upper "
                                         "admissible end 1e-8 is still below the "
                                         "physical resolution"},
           "reference_solver": "sparse LU, one gauge node pinned per connected component",
           "minres_rtol_grid": MINRES_RTOL,
           "scenes": {}}

    for s in scenes:
        name = s["name"]
        d = loader.load_ordering_scene(name)
        truth = d["lam_n"]
        n_c = s["n_contacts"]
        preds = {}
        diag = None
        for wi, wn in enumerate(WEIGHTS):
            if wn == "inverse_mass":
                I, diag = loader.solve_flow_diag(
                    d["edges"], d[f"weights_{wn}"], s["num_nodes"], s["n_grains"],
                    s["node_floor"], s["node_foot"], s["type"])
            else:
                I, _ = loader.solve_flow(
                    d["edges"], d[f"weights_{wn}"], s["num_nodes"], s["n_grains"],
                    s["node_floor"], s["node_foot"], s["type"])
            preds[f"resistance_flow_{wn}"] = I
        preds.update(baselines(s, d, n_c))

        rows = {k: predictor_row(v, truth) for k, v in preds.items()}

        # solver sensitivity: only the large scenes actually used the iterative
        # branch; record it for every foot packing for completeness.
        sweep = []
        I_ref = preds["resistance_flow_inverse_mass"]
        ref_max = float(np.max(I_ref))
        for rt in MINRES_RTOL:
            Im, _ = loader.solve_flow(
                d["edges"], d["weights_inverse_mass"], s["num_nodes"], s["n_grains"],
                s["node_floor"], s["node_foot"], s["type"],
                method="minres", rtol=rt, maxiter=200000)
            m = loader.metrics(Im, truth, TIE_PRIMARY)
            sweep.append({"method": f"minres_rtol_{rt:g}",
                          "rel_flow_error_vs_direct":
                              float(np.linalg.norm(Im - I_ref) / max(np.linalg.norm(I_ref), 1e-300)),
                          "spearman": m["spearman"], "prec_10": m["prec_10"]})

        out["scenes"][name] = {
            "n_contacts": n_c, "n_nodes": s["num_nodes"], "type": s["type"],
            "diagnostics": diag,
            "max_abs_flow_inverse_mass": ref_max,
            "predictors": rows,
            "solver_sweep_inverse_mass": sweep}

    blob = json.dumps(out, sort_keys=True, separators=(",", ":"), allow_nan=False)
    out["sha256"] = hashlib.sha256(blob.encode()).hexdigest()
    if write:
        path = os.path.join(HERE, "measurements_corrected.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
            f.write("\n")
        print("wrote", path, "sha256", out["sha256"])
    else:
        print("dry run; sha256", out["sha256"])
        for name, sc in out["scenes"].items():
            m = sc["predictors"]["resistance_flow_inverse_mass"]
            print(f"{name:16s} spearman={m['spearman']:+.5f} "
                  f"interval={m['tie_interval_spearman']}")


if __name__ == "__main__":
    main()
