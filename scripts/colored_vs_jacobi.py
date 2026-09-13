#!/usr/bin/env python3
"""A/B/C/D/E measurement: graph-coloured Gauss-Seidel GPU solver, with the manifold reduction on the host,
on the device, and with the solve loop replayed as a CUDA graph, against the relaxed-Jacobi GPU solver.

All three columns are the same bodies, the same broad phase and the same contact model; only the solve
differs (`contact_engine_gpu_colored.GraphColoredContactEngine`, the same class with `manifold_reduce=True`,
and `contact_engine_gpu.RelaxedJacobiContactEngine`). The third and fourth columns reduce every body pair to
at most 4 E-optimal contact points before the colouring, the third with that stage in numpy on the host
(`manifold_on_host=True`), the fourth with it as GPU kernels (the default). The two select the same points. The fifth column is
the fourth with `graph_capture=True`: the whole solve loop (every colour x every iteration) is captured once
as a CUDA graph and replayed, and each colour's offset and count are read from device arrays, so the capture
survives a changed colour partition. The fifth column produces the same states as the fourth, bit for bit.
  (e) per-stage wall time of one step of the fourth and the fifth column at N = 1e3 and N = 1e4
      (generation, sort, select, colour, solve, integrate), device-synchronised per stage.
Measured here, on this machine, fresh:

  (a) iterations to the M5 tolerance (max penetration / R < 0.5 and bounded drift) on the K=8 and K=16
      stacks: the velocity-iteration sweep is run for both engines with the same position-iteration count,
      and the lowest velocity-iteration count that passes is reported. Jacobi's shipped setting is 40.
  (b) M1, M5 and M6 for all three, produced by scripts/engine_metrics.py --gpu-engine all --gpu-only
      (the pre-registered harness, unchanged metrics).
  (c) number of colours per scene, and the contacts actually solved (the two coloured columns).
  (d) wall time per step and steps per second on a lattice of boxes dropped onto the ground at N = 1e3 and
      N = 1e4 bodies, two runs per engine and size.

  python scripts/colored_vs_jacobi.py [--quick]     # -> reports/colored_vs_jacobi.json
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from engine_metrics import DIMS, DT, PITCH, m5_penetration  # noqa: E402


def _factories():
    from motion_engine.contact_engine_gpu import RelaxedJacobiContactEngine
    from motion_engine.contact_engine_gpu_colored import GraphColoredContactEngine

    def jac(**kw):
        return RelaxedJacobiContactEngine(dims=DIMS, mu=0.5, **kw)

    def col(**kw):
        return GraphColoredContactEngine(dims=DIMS, mu=0.5, **kw)

    def man(**kw):
        return GraphColoredContactEngine(dims=DIMS, mu=0.5, manifold_reduce=True, manifold_on_host=True, **kw)

    def mandev(**kw):
        return GraphColoredContactEngine(dims=DIMS, mu=0.5, manifold_reduce=True, **kw)

    def mangraph(**kw):
        return GraphColoredContactEngine(dims=DIMS, mu=0.5, manifold_reduce=True, graph_capture=True, **kw)

    def manchunk(**kw):
        return GraphColoredContactEngine(dims=DIMS, mu=0.5, manifold_reduce=True, graph_capture=True,
                                         pair_chunks=True, **kw)
    return {"jacobi": jac, "colored": col, "colored_manifold": man, "colored_manifold_device": mandev,
            "colored_manifold_graph": mangraph, "colored_manifold_chunks": manchunk}


# ───────────────────── (a) iterations to the M5 tolerance ─────────────────────
def iters_to_m5(mk, K, vits, pit, steps):
    rows = []
    found = None
    for vit in vits:
        r = m5_penetration(lambda: mk(vit=vit, pit=pit), "gpu", K, steps)
        rows.append({"vit": vit, "max_pen_over_R": r["max_pen_over_R"], "drift_class": r["drift_class"],
                     "status": r["status"], "pass": r["pass"]})
        if found is None and r["pass"]:
            found = vit
    return {"K": K, "pos_iters": pit, "steps": steps, "vel_iters_to_M5_tol": found, "sweep": rows}


# ───────────────────── (c) colours per scene ─────────────────────
def colours_for_scene(mk, centres, steps=60):
    e = mk()
    for c in centres:
        e.add_body(c)
    ncol = []
    ncon = []
    for _ in range(steps):
        e.step(DT, substeps=1)
        ncol.append(e.n_colors)
        ncon.append(e.n_contacts)
    solved = int(getattr(e, "n_contacts_solved", e.n_contacts))
    return {"bodies": len(centres), "colours_max": int(max(ncol)), "colours_final": int(ncol[-1]),
            "contacts_max": int(max(ncon)), "contacts_final": int(ncon[-1]),
            "contacts_solved_final": solved, "pairs_final": int(getattr(e, "n_pairs", 0)), "steps": steps}


def stack(K):
    return [[0, 0, 0.101 + k * PITCH] for k in range(K)]


def lattice(n_target, layers=2):
    """A lattice of boxes resting-pitch apart in z and 0.35 m apart in x and y, dropped onto the ground."""
    side = int(np.ceil(np.sqrt(n_target / layers)))
    out = []
    for lz in range(layers):
        for iy in range(side):
            for ix in range(side):
                out.append([ix * 0.35, iy * 0.35, 0.101 + lz * PITCH])
    return out[:n_target]


# ───────────────────── (d) wall time per step ─────────────────────
def timed_steps(mk, centres, warm=10, timed=30, **kw):
    e = mk(**kw)
    for c in centres:
        e.add_body(c)
    for _ in range(warm):
        e.step(DT, substeps=1)
    t0 = time.perf_counter()
    for _ in range(timed):
        e.step(DT, substeps=1)
    wall = (time.perf_counter() - t0) / timed
    out = {"bodies": len(centres), "ms_per_step": round(wall * 1e3, 3), "steps_per_s": round(1.0 / wall, 2),
           "warm_steps": warm, "timed_steps": timed}
    if hasattr(e, "n_colors"):
        out["colours_final"] = int(e.n_colors)
    out["contacts_final"] = int(getattr(e, "n_contacts", -1))
    out["contacts_solved_final"] = int(getattr(e, "n_contacts_solved", out["contacts_final"]))
    out["vel_iters"] = int(getattr(e, "vit", -1))
    out["pairs_final"] = int(getattr(e, "n_pairs", 0))
    out["graph_captures"] = int(getattr(e, "n_graph_captures", 0))
    return out


def stage_profile(mk, centres, warm=10, timed=30):
    """Per-stage wall time of one step, device-synchronised at every stage boundary (so the total is a few
    per cent above the unsynchronised ms/step of (d), which is the number to quote)."""
    e = mk(profile=True)
    for c in centres:
        e.add_body(c)
    for _ in range(warm):
        e.step(DT, substeps=1)
    e.prof_ms.clear()
    t0 = time.perf_counter()
    for _ in range(timed):
        e.step(DT, substeps=1)
    wall = (time.perf_counter() - t0) / timed
    out = {k: round(v / timed, 3) for k, v in e.prof_ms.items()}
    out["total_profiled_ms_per_step"] = round(wall * 1e3, 3)
    out["bodies"] = len(centres)
    out["contacts_final"] = int(e.n_contacts)
    out["contacts_solved_final"] = int(e.n_contacts_solved)
    out["pairs_final"] = int(getattr(e, "n_pairs", 0))
    out["graph_captures"] = int(getattr(e, "n_graph_captures", 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="short sweeps and N=1e3 only (test wrapper)")
    args = ap.parse_args()
    mks = _factories()
    res = {"schema": "colored_vs_jacobi_v5", "quick": args.quick,
           "engines": {"A": "relaxed_jacobi_gpu (contact_engine_gpu.py)",
                       "B": "graph_colored_gpu (contact_engine_gpu_colored.py)",
                       "C": "graph_colored_gpu, manifold_reduce=True, manifold_on_host=True (numpy stage)",
                       "D": "graph_colored_gpu, manifold_reduce=True (the stage as GPU kernels, default)",
                       "E": "graph_colored_gpu, manifold_reduce=True, graph_capture=True (solve loop as a "
                            "CUDA graph)",
                       "F": "graph_colored_gpu, manifold_reduce=True, graph_capture=True, pair_chunks=True "
                            "(the body-pair graph is coloured and a pair's contacts are one sequential "
                            "chunk in one thread)"}}
    t0 = time.time()

    vits = [2, 10] if args.quick else [2, 4, 6, 8, 10, 20, 40, 60, 80, 120, 160]
    steps = 100 if args.quick else 600
    Ks = [8] if args.quick else [8, 16]
    # (a) is run for the first four columns: the fifth produces the same states as the fourth, bit for bit
    res["a_iters_to_M5_tol"] = {name: [iters_to_m5(mks[name], K, vits, 10, steps) for K in Ks]
                                for name in ("jacobi", "colored", "colored_manifold",
                                             "colored_manifold_device", "colored_manifold_chunks")}

    cs = 20 if args.quick else 200
    scenes = {"M1_stack_K4": (stack(4), cs), "stack_K8": (stack(8), cs), "stack_K16": (stack(16), cs),
              "lattice_1e3": (lattice(1000), 10)}
    res["c_colours_per_scene"] = {name: {k: colours_for_scene(mks[k], cen, st)
                                         for k in ("colored", "colored_manifold", "colored_manifold_device",
                                                   "colored_manifold_graph", "colored_manifold_chunks")}
                                  for name, (cen, st) in scenes.items()}

    sizes = [1000] if args.quick else [1000, 10000]
    res["d_wall_time"] = {}
    for name, mk in mks.items():
        rows = []
        for n in sizes:
            cen = lattice(n)
            for run in range(2):
                r = timed_steps(mk, cen, 5 if args.quick else 10, 10 if args.quick else 30)
                r["run"] = run + 1
                rows.append(r)
        res["d_wall_time"][name] = rows

    res["e_stage_profile"] = {}
    for n in sizes:
        res["e_stage_profile"][str(n)] = {
            col: stage_profile(mks[col], lattice(n), 5 if args.quick else 10, 10 if args.quick else 30)
            for col in ("colored_manifold_device", "colored_manifold_graph", "colored_manifold_chunks")}

    # (b) the pre-registered harness, both GPU engines, as a subprocess so the shipped script is the source
    cmd = [sys.executable, str(ROOT / "scripts" / "engine_metrics.py"), "--gpu-engine", "all", "--gpu-only"]
    if args.quick:
        cmd.append("--quick")
    p = subprocess.run(cmd, capture_output=True, text=True)
    rel_cmd = "scripts/engine_metrics.py " + " ".join(cmd[2:])      # repo-relative, no machine paths
    res["b_engine_metrics"] = {"cmd": rel_cmd, "returncode": p.returncode,
                               "stdout_tail": p.stdout.strip().splitlines()[-8:]}
    em = ROOT / "reports" / "engine_metrics.json"
    if p.returncode == 0 and em.exists():
        d = json.loads(em.read_text())
        res["b_engine_metrics"]["engines"] = {
            k: {"M1_stack_load_K4": {kk: v["M1_stack_load_K4"][kk] for kk in
                                     ("value_worst_relerr", "status", "min_spacing_m")},
                "M5_stack8_pen": {kk: v["M5_stack8_pen"][kk] for kk in
                                  ("max_pen_over_R", "drift_class", "status")},
                "M6_determinism": {kk: v["M6_determinism"][kk] for kk in
                                   ("eps_nondet_max_dxc_m", "bit_deterministic", "pass")},
                "wall_s": v["wall_s"]}
            for k, v in d["engines"].items()}
    res["wall_s"] = round(time.time() - t0, 1)

    outp = ROOT / "reports" / "colored_vs_jacobi.json"
    outp.parent.mkdir(exist_ok=True)
    outp.write_text(json.dumps(res, indent=2))

    print("\n(a) velocity iterations to the M5 tolerance (pen/R < 0.5, bounded); pos_iters = 10")
    for name in ("jacobi", "colored", "colored_manifold", "colored_manifold_device",
                 "colored_manifold_chunks"):
        for r in res["a_iters_to_M5_tol"][name]:
            print(f"  {name:24s} K={r['K']:<3d} -> {r['vel_iters_to_M5_tol']}  "
                  + " ".join(f"{s['vit']}:{s['max_pen_over_R']:.2f}{'P' if s['pass'] else 'F'}"
                             for s in r["sweep"]))
    print("\n(c) colours per scene (the two coloured columns)")
    for k, d in res["c_colours_per_scene"].items():
        for name, v in d.items():
            print(f"  {k:14s} {name:24s} bodies={v['bodies']:<6d} contacts={v['contacts_max']:<7d} "
                  f"solved={v['contacts_solved_final']:<7d} pairs={v['pairs_final']:<6d} "
                  f"colours={v['colours_max']}")
    print("\n(d) wall time per step (lattice dropped onto the ground)")
    for name in ("jacobi", "colored", "colored_manifold", "colored_manifold_device",
                 "colored_manifold_graph", "colored_manifold_chunks"):
        for r in res["d_wall_time"][name]:
            print(f"  {name:24s} N={r['bodies']:<6d} run{r['run']}  {r['ms_per_step']:9.3f} ms/step  "
                  f"{r['steps_per_s']:8.2f} steps/s  contacts={r['contacts_final']} "
                  f"solved={r['contacts_solved_final']} pairs={r['pairs_final']} vit={r['vel_iters']} "
                  f"captures={r['graph_captures']}")
    print("\n(e) per-stage wall time of one step, manifold on device, uncaptured and graph-captured "
          "(device-synchronised per stage)")
    for n, cols in res["e_stage_profile"].items():
        for col, d in cols.items():
            stages = " ".join(f"{k}={d[k]:.2f}" for k in ("generate", "sort", "select", "pairs", "colour",
                                                          "solve", "integrate", "warm_store") if k in d)
            print(f"  N={n:<6s} {col:24s} {stages}  total(profiled)={d['total_profiled_ms_per_step']:.2f} ms")

    print("\n(b) engine_metrics rows")
    for line in res["b_engine_metrics"].get("stdout_tail", []):
        print("  " + line)
    print(f"\nwall {res['wall_s']} s -> {outp.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
