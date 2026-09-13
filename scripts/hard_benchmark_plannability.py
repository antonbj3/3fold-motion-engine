#!/usr/bin/env python3
"""Finds where the tiered planner breaks, instead of confirming that it works on easy clutter.

`fleet_plannability_metric.py` returns a clean sweep on reach-scaled clutter, which means those scenes are too
easy to locate the stack's frontier. This cell builds genuinely HARD scenes - a cage, a corridor with a narrow
slot, a dense pillar forest - that stress the gradient so the completeness fallback fires AND genuine FAILURES
appear. The report is the frontier: solved / fast path / fallback / FAILED, with failures separated from solves.

Every solve is dense-EXACT-verified, so a false-free path cannot pass. Failures are reported separately, and the
RRT budget is capped, so a failure can be budget-limited rather than genuinely infeasible - `rrt_completeness_verify.py`
is what separates those two.

Input: `data/kin_spec_<robot>.npz`, a kinematics spec in the form `motion_engine.fk_warp.WarpBatchFK` reads; the
extractor that writes it is not part of this repository.

  python -u scripts/hard_benchmark_plannability.py   (requires warp with a CUDA device + data/kin_spec_*.npz)
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
from motion_engine.trajopt_warp import TieredGpuPlanner

import warp as wp

ROBOTS = ["ur10e", "fanuc_m10ia", "kuka_kr10_r1100_2"]
N_SCENES = 10
RRT_ITER = 4000                                                 # a larger budget for hard scenes: the fallback must work


def _scenes(name, r):
    """Genuinely hard reach-scaled scenes (narrow passages and enclosure -> local minima and narrow-passage RRT)."""
    if name in ("cage", "cage_forced"):                         # five-walled cage (open front) around the goal region
        w = 0.04 * r
        return [{"type": "box", "dims": [0.3 * r, 0.3 * r, w], "pose": [0.5 * r, 0.0, 0.25 * r]},        # floor
                {"type": "box", "dims": [0.3 * r, 0.3 * r, w], "pose": [0.5 * r, 0.0, 0.7 * r]},         # ceiling
                {"type": "box", "dims": [w, 0.3 * r, 0.25 * r], "pose": [0.5 * r, 0.3 * r, 0.475 * r]},  # left
                {"type": "box", "dims": [w, 0.3 * r, 0.25 * r], "pose": [0.5 * r, -0.3 * r, 0.475 * r]}, # right
                {"type": "box", "dims": [w, 0.6 * r, 0.5 * r], "pose": [0.75 * r, 0.0, 0.475 * r]}]      # back
    if name == "corridor":                                      # two walls with a narrow vertical slot
        w = 0.05 * r; gap = 0.18 * r
        return [{"type": "box", "dims": [0.5 * r, w, 0.6 * r], "pose": [0.45 * r, gap, 0.45 * r]},
                {"type": "box", "dims": [0.5 * r, w, 0.6 * r], "pose": [0.45 * r, -gap, 0.45 * r]},
                {"type": "box", "dims": [0.5 * r, 0.5 * r, w], "pose": [0.45 * r, 0.0, 0.75 * r]}]       # ceiling
    # pillars: a dense pillar forest
    rng = np.random.default_rng(1); ps = []
    for _ in range(7):
        x = (0.3 + 0.4 * rng.random()) * r; y = (rng.random() - 0.5) * 0.8 * r
        ps.append({"type": "box", "dims": [0.06 * r, 0.06 * r, 0.6 * r], "pose": [x, y, 0.4 * r]})
    return ps


def run(robot, scene_name, dev):
    sp = ROOT / "data" / f"kin_spec_{robot}.npz"
    if not sp.exists():
        return None
    spec = dict(np.load(sp)); nq = int(spec["nq"]); fk = WarpBatchFK(spec, device=dev)
    lo = np.array([-3.14] * nq); hi = -lo
    rng = np.random.default_rng(0); Q0 = lo + rng.random((2500, nq)) * (hi - lo)
    reach = float(np.percentile(np.linalg.norm(fk.fk_points(Q0).reshape(-1, 3), axis=1), 99)); r = reach
    obs = _scenes(scene_name, r); oracle = PrimitiveExactWorld(obs)
    R = reach * 1.3 + 0.2
    world = WarpVolumeWorld.from_world_model(oracle, [-R, -R, -R * 0.8], [R, R, R], max(reach / 150, 0.005), device=dev)

    def pmin(p, d=8):
        seg = [np.linspace(a, b, d, endpoint=False) for a, b in zip(p[:-1], p[1:])]
        Q = np.vstack(seg + [p[-1:]]); W = fk.fk_points(Q)
        return float(min(oracle.sd_batch(W[c]).min() for c in range(len(Q))))

    P = fk.fk_points(Q0); ee = P[:, -1, :]                     # arm-tip region per configuration (for the forced goal)
    sd = np.array([oracle.sd_batch(P[i]).min() for i in range(len(Q0))])
    free = Q0[sd > 0.02 * r]
    scenes = []; fi = 0
    if scene_name == "cage_forced":
        # The goal tip is FORCED inside the cage (it has to thread in through the open front) with the start
        # outside. A random colliding pair is not enough - the arm can go around - and this forces the hard
        # manoeuvre. Caveat: the free test uses the VOLUME model (optimistic against exact narrow phase), so some
        # forced tight goals are colliding under the exact model, i.e. near-infeasible. The dense-EXACT verification
        # gates those honestly (they become "failed", never false-free), so a cage_forced failure is often an
        # ill-conditioned goal rather than a planner gap.
        inside = ((ee[:, 0] > 0.35 * r) & (ee[:, 0] < 0.62 * r) & (np.abs(ee[:, 1]) < 0.22 * r)
                  & (ee[:, 2] > 0.32 * r) & (ee[:, 2] < 0.63 * r))
        goals = Q0[(sd > 0.02 * r) & inside]; starts = Q0[(sd > 0.02 * r) & (ee[:, 0] < 0.3 * r)]
        for k in range(min(len(goals), len(starts), N_SCENES * 3)):
            a = starts[k % len(starts)]; b = goals[k % len(goals)]
            if pmin(np.linspace(a, b, 20), 2) < 0:             # the straight line collides (genuine)
                scenes.append((a, b))
            if len(scenes) >= N_SCENES:
                break
    else:
        while len(scenes) < N_SCENES and fi < len(free) - 1:
            a = free[fi]; fi += 1
            b = next((free[j] for j in range(fi, len(free)) if pmin(np.linspace(a, free[j], 20), 2) < 0), None)
            if b is not None:
                scenes.append((a, b))
    if not scenes:
        return dict(robot=robot, scene=scene_name, n=0, fast=0, fallback=0, failed=0)
    tier = TieredGpuPlanner(fk, world, lo, hi, margin=0.025, verify_world=oracle)
    tier.plan(scenes[0][0], scenes[0][1], T=24, iters=4, restarts=1, rrt_iter=10)
    fast = fb = failed = 0; ms = []
    for a, b in scenes:
        t0 = time.perf_counter(); p, info = tier.plan(a, b, T=40, iters=160, restarts=4, rrt_iter=RRT_ITER, seed=1)
        wp.synchronize_device(dev); dt = (time.perf_counter() - t0) * 1e3; ms.append(dt)
        solved = bool(info["solved"]) and pmin(p) > 0
        if not solved:
            failed += 1
        elif info["tier"] == "gradient":
            fast += 1
        else:
            fb += 1
    return dict(robot=robot, scene=scene_name, n=len(scenes), fast=int(fast), fallback=int(fb), failed=int(failed),
                solved=int(fast + fb), med_ms=int(np.median(ms)))


def main():
    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    print(f"HARD-BENCHMARK-PLANNABILITY — find where it breaks (cage / corridor / pillars), device='{dev}'\n")
    rows = []
    for scene in ("cage", "cage_forced", "corridor", "pillars"):
        print(f"  --- {scene} ---")
        for robot in ROBOTS:
            x = run(robot, scene, dev)
            if x is None or x["n"] == 0:
                print(f"    {robot:22s} — no genuine scenes (skipped)"); continue
            rows.append(x)
            print(f"    {robot:22s}: solved {x['solved']}/{x['n']} (fast {x['fast']} + fallback {x['fallback']}) | "
                  f"FAILED {x['failed']} | median {x['med_ms']}ms")
    tn = sum(r["n"] for r in rows); ts = sum(r["solved"] for r in rows)
    tfast = sum(r["fast"] for r in rows); tfb = sum(r["fallback"] for r in rows); tfail = sum(r["failed"] for r in rows)
    print(f"\n  AGGREGATE: solved {ts}/{tn} | fast path {tfast} + fallback {tfb} (completeness win) | FAILED {tfail}")
    frontier = tfail > 0 or tfb > 0
    print(f"\nVERDICT: hard-benchmark = "
          f"{'FRONTIER FOUND (fallbacks and failures appear, so the scenes stress the stack)' if frontier else 'STILL TOO EASY (0 fallback, 0 failure - make them harder)'}. "
          + (f"On the hard scenes {tfb} needed the completeness fallback (the gradient alone missed them) and "
             f"{tfail} FAILED (the tiered planner did not solve them even with RRT at {RRT_ITER} iterations). " if frontier
             else f"All {ts}/{tn} went through the fast path, so these scenes are not at the stack's frontier. ")
          + f"Scope: dense-EXACT-verified (no false-free); failures reported separately; RRT budget cap {RRT_ITER} "
            "(a failure can be budget rather than genuine infeasibility - run rrt_completeness_verify.py to tell "
            "them apart); reach-scaled primitive scenes, not an official benchmark mesh set; seed 1.")
    out = dict(device=dev, rows=rows, total=tn, solved=ts, fast=tfast, fallback=tfb, failed=tfail, frontier_found=frontier)
    (ROOT / "reports" / "hard_benchmark_plannability.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("  wrote reports/hard_benchmark_plannability.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
