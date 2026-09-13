#!/usr/bin/env python3
"""Symmetric gate on the planner's STOP claim: is "infeasible" budget-limited or genuine?

GPU MPPI is sampling-based and therefore not probabilistically complete, unlike CPU RRT-Connect. It can claim
"no path" (false-infeasible) on a scene where a path exists. Stop claims deserve the same verification as success
claims, so this audit runs MPPI on genuine clutter scenes at INCREASING budget (rollouts K, iterations): if a
low-budget MISS is solved at a higher budget, the "infeasible" was budget-limited and recoverable. A scene that no
reasonable budget solves is genuinely hard, and the honest handling there is an RRT fallback, not an infeasibility
claim.

Input: `data/kin_spec_ur10e.npz`, a kinematics spec in the form `motion_engine.fk_warp.WarpBatchFK` reads; the
extractor that writes it is not part of this repository.

Also writes `reports/mppi_completeness_scenes.npz` (start/goal configurations, obstacles, the high-budget verdict
per scene), which `rrt_completeness_verify.py` consumes on the CPU to separate false-infeasible from genuinely hard.

  python -u scripts/mppi_completeness_audit.py   (requires warp with a CUDA device + data/kin_spec_ur10e.npz)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from motion_engine.world_model import PrimitiveExactWorld
from motion_engine.world_model_warp import WarpVolumeWorld
from motion_engine.fk_warp import WarpBatchFK
from motion_engine.trajopt_warp import BatchMPPI

import warp as wp

BUDGETS = [("low", 128, 20), ("med", 512, 50), ("high", 1024, 90)]   # (name, K, iters)


def _path_min_sd(fk, oracle, path, dens=8):
    seg = [np.linspace(a, b, dens, endpoint=False) for a, b in zip(path[:-1], path[1:])]
    Q = np.vstack(seg + [path[-1:]]); W = fk.fk_points(Q)
    return float(min(oracle.sd_batch(W[c]).min() for c in range(len(Q))))


def main():
    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    spec = dict(np.load(ROOT / "data" / "kin_spec_ur10e.npz")); nq = int(spec["nq"])
    fk = WarpBatchFK(spec, device=dev)
    obs = [{"type": "box", "dims": [0.2, 0.2, 0.5], "pose": [0.45, 0.0, 0.4]},
           {"type": "box", "dims": [0.15, 0.5, 0.15], "pose": [0.4, 0.3, 0.6]},
           {"type": "box", "dims": [0.15, 0.5, 0.15], "pose": [0.4, -0.3, 0.6]},
           {"type": "box", "dims": [0.3, 0.15, 0.15], "pose": [0.55, 0.0, 0.7]},
           {"type": "box", "dims": [0.2, 0.2, 0.2], "pose": [0.0, 0.5, 0.3]}]
    oracle = PrimitiveExactWorld(obs)
    world = WarpVolumeWorld.from_world_model(oracle, [-1.3, -1.3, -1.0], [1.3, 1.3, 1.3], 0.01, device=dev)
    lo = np.array([-3.14] * nq); hi = -lo
    print("MPPI-COMPLETENESS-AUDIT — symmetric gate on 'infeasible': budget recovery of low-budget misses\n")
    rng = np.random.default_rng(5); Qs = lo + rng.random((6000, nq)) * (hi - lo)
    sd = np.array([oracle.sd_batch(fk.fk_points(Qs[i:i + 1])[0]).min() for i in range(len(Qs))])
    free = Qs[sd > 0.03]; scenes = []; fi = 0
    while len(scenes) < 8 and fi < len(free) - 1:
        a = free[fi]; fi += 1
        b = next((free[j] for j in range(fi, len(free))
                  if _path_min_sd(fk, oracle, np.linspace(a, free[j], 18), dens=2) < 0), None)
        if b is not None:
            scenes.append((a, b))
    mppi = BatchMPPI(fk, world, lo, hi, margin=0.03)
    mppi.plan(scenes[0][0], scenes[0][1], T=24, K=128, iters=4)              # warm-up
    rows = []
    for si, (qs, qg) in enumerate(scenes):
        res = {}
        for name, K, it in BUDGETS:
            path, info = mppi.plan(qs, qg, T=32, K=K, iters=it, seed=1)
            res[name] = bool(_path_min_sd(fk, oracle, path, dens=8) > 0 and info["goal_err"] < 0.15)
        rows.append(res)
        print(f"  scene {si}: " + " ".join(f"{n}({K}x{it})={'OK' if res[n] else 'MISS'}" for n, K, it in BUDGETS))

    low_miss = sum(1 for r in rows if not r["low"])
    recovered = sum(1 for r in rows if (not r["low"]) and r["high"])          # low-budget misses the high budget solved
    hard = sum(1 for r in rows if not r["high"])                             # no budget solved these
    print(f"\n  low-budget misses: {low_miss}/{len(rows)} | recovered at high budget: {recovered} | "
          f"unsolved at high budget: {hard}")
    ok = hard == 0
    print(f"\nVERDICT: mppi-completeness = {'INFEASIBLE IS BUDGET-RECOVERABLE' if ok else 'GENUINELY HARD SCENES EXIST'}. "
          + (f"All {len(rows)} clutter scenes are solved at a sufficient budget; {recovered}/{low_miss} low-budget "
             "'infeasible' verdicts were budget-limited, so the stop claim was not fundamental and the honest "
             "handling is to scale the budget before reporting infeasible. " if ok else
             f"{hard}/{len(rows)} scenes are unsolved even at the high budget (K1024 x 90), which is a "
             "false-infeasible RISK for a sampling planner; the honest handling is an RRT-Connect fallback "
             "(probabilistically complete) for those, not an infeasibility claim. ")
          + "Scope: 'solved' means dense-oracle-verified collision-free and goal-reached; this compares budget "
            "sensitivity, not MPPI against a complete planner - that comparison is rrt_completeness_verify.py.")
    out = dict(device=dev, budgets=[dict(name=n, K=K, iters=it) for n, K, it in BUDGETS], n_scenes=len(rows),
               low_miss=low_miss, recovered_by_high=recovered, unsolved_at_high=hard,
               per_scene=rows, infeasible_budget_recoverable=ok)
    (ROOT / "reports" / "mppi_completeness_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    # save the scenes (start/goal + the high-budget result + obstacles) for the CPU RRT completeness verification
    np.savez(ROOT / "reports" / "mppi_completeness_scenes.npz",
             starts=np.array([s for s, _ in scenes]), goals=np.array([g for _, g in scenes]),
             mppi_high=np.array([r["high"] for r in rows]),
             obs_dims=np.array([o["dims"] for o in obs]), obs_pose=np.array([o["pose"] for o in obs]))
    print("  wrote reports/mppi_completeness_audit.json + mppi_completeness_scenes.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
