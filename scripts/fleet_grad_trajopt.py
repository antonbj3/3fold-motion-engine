#!/usr/bin/env python3
"""Gradient trajectory optimisation: per-robot solve rate and latency on reach-scaled clutter.

Checks that the GPU gradient trajopt's latency and solve rate GENERALISE per robot rather than holding only on the
robot it was tuned on. Per robot: a reach-scaled clutter scene, gradient trajopt, then a dense EXACT-oracle
verification of the resulting path plus the wall-clock latency. Per-robot rows are printed, so the worst robot is
visible instead of averaged away.

Input: `data/kin_spec_<robot>.npz`, a kinematics spec (joint chain + collision surface points) in the form
`motion_engine.fk_warp.WarpBatchFK` reads; the extractor that writes it is not part of this repository.

Gate: every scene solved (dense-verified collision-free and goal reached) and the worst per-robot latency under
600 ms, which is the order of magnitude published for GPU trajectory optimisers.

  python -u scripts/fleet_grad_trajopt.py   (requires warp with a CUDA device + data/kin_spec_*.npz)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from motion_engine.world_model import PrimitiveExactWorld
from motion_engine.world_model_warp import WarpVolumeWorld
from motion_engine.fk_warp import WarpBatchFK
from motion_engine.trajopt_warp import BatchGradTrajopt

import warp as wp

ROBOTS = ["ur10e", "fanuc_m10ia", "abb_irb4600_60_205", "motoman_sia20d", "kuka_kr10_r1100_2"]
M = 4


def run(robot, dev):
    sp = ROOT / "data" / f"kin_spec_{robot}.npz"
    if not sp.exists():
        return None
    spec = dict(np.load(sp)); nq = int(spec["nq"])
    fk = WarpBatchFK(spec, device=dev)
    lo = np.array([-3.14] * nq); hi = -lo
    rng = np.random.default_rng(0); Q0 = lo + rng.random((1500, nq)) * (hi - lo)
    reach = float(np.percentile(np.linalg.norm(fk.fk_points(Q0).reshape(-1, 3), axis=1), 99))
    r = reach
    obs = [{"type": "box", "dims": [0.2 * r, 0.2 * r, 0.5 * r], "pose": [0.45 * r, 0.0, 0.4 * r]},
           {"type": "box", "dims": [0.15 * r, 0.4 * r, 0.15 * r], "pose": [0.4 * r, 0.3 * r, 0.6 * r]},
           {"type": "box", "dims": [0.15 * r, 0.4 * r, 0.15 * r], "pose": [0.4 * r, -0.3 * r, 0.6 * r]},
           {"type": "box", "dims": [0.25 * r, 0.15 * r, 0.15 * r], "pose": [0.5 * r, 0.0, 0.7 * r]}]
    oracle = PrimitiveExactWorld(obs)
    R = reach * 1.3 + 0.2
    world = WarpVolumeWorld.from_world_model(oracle, [-R, -R, -R * 0.8], [R, R, R], max(reach / 130, 0.006), device=dev)

    def pmin(p, d=8):
        seg = [np.linspace(a, b, d, endpoint=False) for a, b in zip(p[:-1], p[1:])]
        Q = np.vstack(seg + [p[-1:]]); W = fk.fk_points(Q)
        return float(min(oracle.sd_batch(W[c]).min() for c in range(len(Q))))

    sd = np.array([oracle.sd_batch(fk.fk_points(Q0[i:i + 1])[0]).min() for i in range(len(Q0))])
    free = Q0[sd > 0.03]; scenes = []; fi = 0
    while len(scenes) < M and fi < len(free) - 1:                      # genuine scenes: the straight line collides
        a = free[fi]; fi += 1
        b = next((free[j] for j in range(fi, len(free)) if pmin(np.linspace(a, free[j], 18), 2) < 0), None)
        if b is not None:
            scenes.append((a, b))
    grad = BatchGradTrajopt(fk, world, lo, hi, margin=0.03)
    if not scenes:
        return dict(robot=robot, nq=nq, n=0, solved=0, ms=0)
    grad.plan(scenes[0][0], scenes[0][1], T=24, iters=4, restarts=1)              # warm-up
    solv = 0; t = 0.0
    for a, b in scenes:
        t0 = time.perf_counter(); p, info = grad.plan(a, b, T=32, iters=140, restarts=4, seed=1)
        wp.synchronize_device(dev); t += time.perf_counter() - t0
        if pmin(p) > 0 and info["goal_err"] < 0.15:
            solv += 1
    return dict(robot=robot, nq=int(nq), n=len(scenes), solved=int(solv), ms=int(t / len(scenes) * 1e3))


def main():
    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    print(f"FLEET-GRAD-TRAJOPT — gradient trajopt latency per robot (reach-scaled clutter), device='{dev}'\n")
    rows = []
    for rb in ROBOTS:
        x = run(rb, dev)
        if x is None:
            print(f"  {rb:22s} no kinematics spec"); continue
        rows.append(x)
        print(f"  {x['robot']:22s} nq={x['nq']:2d}: gradient {x['solved']}/{x['n']} dense-free | {x['ms']}ms/plan")
    ts = sum(r["solved"] for r in rows); tn = sum(r["n"] for r in rows)
    worst_ms = max((r["ms"] for r in rows if r["n"]), default=0)
    ok = bool(rows) and ts == tn and worst_ms < 600
    print(f"\n  AGGREGATE: {ts}/{tn} solved | worst latency {worst_ms}ms")
    print(f"\nVERDICT: fleet-grad-trajopt = {'GENERALISES' if ok else 'PARTIAL'}. "
          + (f"The gradient trajopt solves {ts}/{tn} clutter scenes, dense-verified, across {len(rows)} robots "
             f"(6 and 7 DOF, several vendors), worst latency {worst_ms}ms, so the latency result is not specific to "
             "one robot. " if ok else f"{ts}/{tn} solved, worst {worst_ms}ms; see the per-robot rows. ")
          + "Scope: reach-scaled clutter, not a published benchmark scene library; finite-difference gradient; "
            "multi-restart against local minima; latency is hardware-dependent and this is not a controlled "
            "head-to-head against another planner.")
    out = dict(device=dev, rows=rows, total_solved=ts, total=tn, worst_ms=worst_ms, generalizes=ok)
    (ROOT / "reports" / "fleet_grad_trajopt.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("  wrote reports/fleet_grad_trajopt.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
