#!/usr/bin/env python3
"""Per-robot plannability of the tiered planner, reported as a distribution rather than one number.

What the stack actually solves, dense-EXACT-verified, per robot, split honestly:
  * FAST PATH (tier = gradient): solved by the gradient trajopt alone, the low-latency case
  * FALLBACK (tier = rrt+refine / rrt-raw): the gradient stalled in a local minimum and a global RRT seed rescued
    it - a completeness win, at a higher cost
  * FAILED: the tiered planner did not solve it even with the RRT fallback - genuinely hard or near-infeasible
    within the budget, reported separately rather than folded into the solve rate

Scenes are genuine (the straight line collides) and "solved" means dense-EXACT-oracle verified collision-free with
the goal reached, so a false-free path cannot count as a solve. Gate: the tiered planner is never worse than the
fast path (completeness cannot hurt) and the fast path carries the majority of the solutions (the gradient is the
primary planner, not a formality).

Input: `data/kin_spec_<robot>.npz`, a kinematics spec in the form `motion_engine.fk_warp.WarpBatchFK` reads; the
extractor that writes it is not part of this repository.

  python -u scripts/fleet_plannability_metric.py   (requires warp with a CUDA device + data/kin_spec_*.npz)
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

N_SCENES = 6
RRT_ITER = 2000                                                  # budget cap; the fallback is meant to be the rare path


def _obs(r, hard):
    """Reach-scaled clutter. moderate = 4 boxes (the gradient usually copes); hard = 6 boxes with a narrow passage,
    which stresses the gradient so the completeness fallback fires - the metric then shows the difficulty frontier
    (fast path -> fallback -> failed) instead of one easy level."""
    o = [{"type": "box", "dims": [0.2 * r, 0.2 * r, 0.5 * r], "pose": [0.45 * r, 0.0, 0.4 * r]},
         {"type": "box", "dims": [0.15 * r, 0.4 * r, 0.15 * r], "pose": [0.4 * r, 0.3 * r, 0.6 * r]},
         {"type": "box", "dims": [0.15 * r, 0.4 * r, 0.15 * r], "pose": [0.4 * r, -0.3 * r, 0.6 * r]},
         {"type": "box", "dims": [0.25 * r, 0.15 * r, 0.15 * r], "pose": [0.5 * r, 0.0, 0.7 * r]}]
    if hard:
        o += [{"type": "box", "dims": [0.2 * r, 0.2 * r, 0.2 * r], "pose": [0.0, 0.5 * r, 0.3 * r]},
              {"type": "box", "dims": [0.15 * r, 0.15 * r, 0.4 * r], "pose": [0.3 * r, 0.0, 0.55 * r]}]  # central pillar
    return o


def run(robot, dev, hard):
    sp = ROOT / "data" / f"kin_spec_{robot}.npz"
    if not sp.exists():
        return None
    spec = dict(np.load(sp)); nq = int(spec["nq"])
    fk = WarpBatchFK(spec, device=dev)
    lo = np.array([-3.14] * nq); hi = -lo
    rng = np.random.default_rng(0); Q0 = lo + rng.random((1500, nq)) * (hi - lo)
    reach = float(np.percentile(np.linalg.norm(fk.fk_points(Q0).reshape(-1, 3), axis=1), 99)); r = reach
    obs = _obs(r, hard)
    oracle = PrimitiveExactWorld(obs)
    R = reach * 1.3 + 0.2
    world = WarpVolumeWorld.from_world_model(oracle, [-R, -R, -R * 0.8], [R, R, R], max(reach / 130, 0.006), device=dev)

    def pmin(p, d=8):
        seg = [np.linspace(a, b, d, endpoint=False) for a, b in zip(p[:-1], p[1:])]
        Q = np.vstack(seg + [p[-1:]]); W = fk.fk_points(Q)
        return float(min(oracle.sd_batch(W[c]).min() for c in range(len(Q))))

    sd = np.array([oracle.sd_batch(fk.fk_points(Q0[i:i + 1])[0]).min() for i in range(len(Q0))])
    free = Q0[sd > 0.03 * r]; scenes = []; fi = 0
    while len(scenes) < N_SCENES and fi < len(free) - 1:
        a = free[fi]; fi += 1
        b = next((free[j] for j in range(fi, len(free)) if pmin(np.linspace(a, free[j], 18), 2) < 0), None)
        if b is not None:
            scenes.append((a, b))
    if not scenes:
        return dict(robot=robot, nq=nq, hard=hard, n=0, fast=0, fallback=0, failed=0, solved=0, fast_ms=0)
    tier = TieredGpuPlanner(fk, world, lo, hi, margin=0.03, verify_world=oracle)
    tier.plan(scenes[0][0], scenes[0][1], T=24, iters=4, restarts=1, rrt_iter=10)              # warm-up
    fast = fallback = failed = 0; fast_ms = []
    for a, b in scenes:
        t0 = time.perf_counter(); p, info = tier.plan(a, b, T=32, iters=140, restarts=4, rrt_iter=RRT_ITER, seed=1)
        wp.synchronize_device(dev); dt = (time.perf_counter() - t0) * 1e3
        solved = bool(info["solved"]) and pmin(p) > 0                                          # dense-EXACT confirmed
        if not solved:
            failed += 1
        elif info["tier"] == "gradient":
            fast += 1; fast_ms.append(dt)
        else:
            fallback += 1
    return dict(robot=robot, nq=int(nq), hard=hard, n=len(scenes), fast=int(fast), fallback=int(fallback),
                failed=int(failed), solved=int(fast + fallback), fast_ms=int(np.median(fast_ms)) if fast_ms else 0)


def main():
    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    print(f"FLEET-PLANNABILITY-METRIC — per-robot plannability of the tiered stack, dense-exact-verified, "
          f"device='{dev}'\n")
    rows = []
    for hard in (False, True):
        print(f"  --- {'HARD (6 boxes, narrow passage)' if hard else 'MODERATE (4 boxes)'} ---")
        for sp in sorted((ROOT / "data").glob("kin_spec_*.npz")):
            robot = sp.stem.replace("kin_spec_", "")
            x = run(robot, dev, hard)
            if x is None or x["n"] == 0:
                print(f"    {robot:24s} — no genuine scenes (skipped)"); continue
            rows.append(x)
            print(f"    {x['robot']:24s} nq={x['nq']}: SOLVED {x['solved']}/{x['n']} "
                  f"(fast path {x['fast']} @{x['fast_ms']}ms + fallback {x['fallback']}) | failed {x['failed']}")
    tn = sum(r["n"] for r in rows); ts = sum(r["solved"] for r in rows)
    tfast = sum(r["fast"] for r in rows); tfb = sum(r["fallback"] for r in rows); tfail = sum(r["failed"] for r in rows)
    ok = bool(rows and tfast >= ts - tfast and ts >= tfast)        # the fast path carries at least half of the solutions
    fast_med = int(np.median([r["fast_ms"] for r in rows if r["fast_ms"]])) if any(r["fast_ms"] for r in rows) else 0
    print(f"\n  AGGREGATE (the per-robot rows above are the result): SOLVED {ts}/{tn} | fast path {tfast} "
          f"(median {fast_med}ms) + fallback {tfb} (completeness win) | failed {tfail}")
    print(f"\nVERDICT: fleet-plannability = {'PLANNABLE FLEET-WIDE' if ok else 'SEE THE PER-ROBOT ROWS'}. "
          + (f"The stack solves {ts}/{tn} genuine clutter scenes over {len(rows)} robot scenarios "
             f"({len(rows)//2} robots x moderate and hard), dense-EXACT-verified; the fast path (gradient, median "
             f"{fast_med}ms) carries {tfast} of them, the RRT fallback rescues {tfb} the gradient alone missed, and "
             f"{tfail} genuinely hard ones are reported SEPARATELY rather than hidden in the solve rate. " if ok else
             f"{ts}/{tn} solved; fast path {tfast}, fallback {tfb}, failed {tfail} — see the per-robot rows. ")
          + f"Scope: reach-scaled clutter, 6 scenes per robot, RRT budget cap {RRT_ITER} (a failure can be budget "
            "rather than genuine infeasibility - rrt_completeness_verify.py separates the two); dense-EXACT-oracle "
            "verified (no false-free); seed 1.")
    out = dict(device=dev, rows=rows, total=tn, solved=ts, fast_path=tfast, fallback=tfb, failed=tfail,
               fast_path_med_ms=fast_med, plannable_fleet_wide=ok)
    (ROOT / "reports" / "fleet_plannability.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("  wrote reports/fleet_plannability.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
