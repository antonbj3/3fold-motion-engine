#!/usr/bin/env python3
"""Fleet-scaled motion-planning benchmark for the GPU MPPI planner.

Per robot and per scene: GPU MPPI (broad-phase coarse optimisation plus a DENSE verification) has to produce a
path that a dense EXACT oracle confirms is collision-free. Scenes are genuine (the straight line between start and
goal collides, otherwise the scene is trivial). Results are reported PER ROBOT - success rate, clearance, plan time
and configuration throughput - with failures visible rather than hidden in a mean. The same scenes and the same
dense-oracle gate are the harness another planner would be plugged into for a comparison.

Input: `data/kin_spec_<robot>.npz`, a kinematics spec (joint chain + collision surface points) in the form
`motion_engine.fk_warp.WarpBatchFK` reads; the extractor that writes it is not part of this repository.

Two scene types: a single box (baseline) and a five-box clutter scene (tight multi-obstacle), both reach-scaled per
robot. Gate: every scene in both modes solved and dense-verified.

  python -u scripts/fleet_mppi_benchmark.py   (requires warp with a CUDA device + data/kin_spec_*.npz)
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
from motion_engine.fk_warp import WarpBatchFK, subsample_spec
from motion_engine.trajopt_warp import BatchMPPI

import warp as wp

ROBOTS = ["ur10e", "fanuc_m10ia", "motoman_sia20d", "abb_irb120_3_58",
          "abb_irb4600_60_205", "kuka_kr10_r1100_2", "motoman_gp8"]
M_SCENES = 4


def _path_min_sd(fk, oracle, path, dens=8):
    seg = [np.linspace(a, b, dens, endpoint=False) for a, b in zip(path[:-1], path[1:])]
    Q = np.vstack(seg + [path[-1:]]); W = fk.fk_points(Q)
    return float(min(oracle.sd_batch(W[c]).min() for c in range(len(Q))))


def _obstacles(reach, mode):
    """Reach-scaled obstacles. single = one box (baseline); clutter = five boxes (tight multi-obstacle scene)."""
    r = reach
    if mode == "single":
        return [{"type": "box", "dims": [0.3 * r, 0.3 * r, 0.6 * r], "pose": [0.45 * r, 0.0, 0.4 * r]}]
    return [{"type": "box", "dims": [0.2 * r, 0.2 * r, 0.5 * r], "pose": [0.45 * r, 0.0, 0.4 * r]},
            {"type": "box", "dims": [0.15 * r, 0.4 * r, 0.15 * r], "pose": [0.4 * r, 0.3 * r, 0.6 * r]},
            {"type": "box", "dims": [0.15 * r, 0.4 * r, 0.15 * r], "pose": [0.4 * r, -0.3 * r, 0.6 * r]},
            {"type": "box", "dims": [0.25 * r, 0.15 * r, 0.15 * r], "pose": [0.5 * r, 0.0, 0.7 * r]},
            {"type": "box", "dims": [0.2 * r, 0.2 * r, 0.2 * r], "pose": [0.0, 0.5 * r, 0.3 * r]}]


def bench_robot(robot, dev, mode="single"):
    sp = ROOT / "data" / f"kin_spec_{robot}.npz"
    if not sp.exists():
        return None
    spec = dict(np.load(sp)); nq = int(spec["nq"])
    fk = WarpBatchFK(spec, device=dev)
    coarse = WarpBatchFK(subsample_spec(spec, 4), device=dev)                # broad phase
    lo = np.array([-3.14] * nq); hi = -lo
    rng = np.random.default_rng(0)
    Q0 = lo + rng.random((2500, nq)) * (hi - lo)
    reach = float(np.percentile(np.linalg.norm(fk.fk_points(Q0).reshape(-1, 3), axis=1), 99))
    obs = _obstacles(reach, mode)
    oracle = PrimitiveExactWorld(obs)
    R = reach * 1.3 + 0.2
    world = WarpVolumeWorld.from_world_model(oracle, [-R, -R, -R * 0.8], [R, R, R], max(reach / 130, 0.006), device=dev)
    sd = np.array([oracle.sd_batch(fk.fk_points(Q0[i:i + 1])[0]).min() for i in range(len(Q0))])
    free = Q0[sd > 0.03]
    scenes = []; fi = 0                                                      # genuine scenes (straight line collides)
    while len(scenes) < M_SCENES and fi < len(free) - 1:
        a = free[fi]; fi += 1
        b = next((free[j] for j in range(fi, len(free))
                  if _path_min_sd(fk, oracle, np.linspace(a, free[j], 18), dens=2) < 0), None)
        if b is not None:
            scenes.append((a, b))
    mppi = BatchMPPI(coarse, world, lo, hi, margin=0.03, verify_fk=fk)       # coarse optimisation, dense verification
    # clutter means tighter passages, so the planner gets a larger budget (more rollouts and iterations)
    T, K, it = (40, 768, 70) if mode == "clutter" else (32, 512, 50)
    if not scenes:
        return dict(robot=robot, nq=nq, mode=mode, n=0, succ=0, mean_clear_mm=None, plan_ms=0, eval_cfg_s=0)
    mppi.plan(scenes[0][0], scenes[0][1], T=24, K=256, iters=4)              # warm-up
    succ = 0; clr = []; t = 0.0; ev = 0
    for q_start, q_goal in scenes:
        t0 = time.perf_counter(); path, info = mppi.plan(q_start, q_goal, T=T, K=K, iters=it, seed=1)
        wp.synchronize_device(dev); t += time.perf_counter() - t0; ev += info["total_evals"]
        ps = _path_min_sd(fk, oracle, path, dens=8)                          # dense oracle verification
        if ps > 0 and info["goal_err"] < 0.15:
            succ += 1; clr.append(ps)
    return dict(robot=robot, nq=nq, mode=mode, n=len(scenes), succ=succ,
                mean_clear_mm=round(float(np.mean(clr)) * 1000, 1) if clr else None,
                plan_ms=round(t / max(len(scenes), 1) * 1000, 0), eval_cfg_s=round(ev / t) if t else 0)


def _run_mode(mode, dev):
    print(f"\n-- MODE='{mode}' ({'5-box clutter' if mode == 'clutter' else 'single box'}, reach-scaled) --")
    rows = []
    for robot in ROBOTS:
        r = bench_robot(robot, dev, mode=mode)
        if r is None:
            print(f"  {robot:22s} no kinematics spec (skipped)"); continue
        rows.append(r)
        print(f"  {robot:22s} nq={r['nq']:2d}: dense-verified {r['succ']}/{r['n']} | clearance "
              f"{r['mean_clear_mm']}mm | {r['plan_ms']:.0f}ms/plan | {r['eval_cfg_s']:,} cfg-eval/s")
    tot_s = sum(r["succ"] for r in rows); tot_n = sum(r["n"] for r in rows)
    n_full = sum(1 for r in rows if r["succ"] == r["n"] and r["n"] > 0)
    print(f"  {mode.upper()} AGGREGATE: {tot_s}/{tot_n} scenes solved, {n_full}/{len(rows)} robots at 100%")
    return dict(mode=mode, rows=rows, total_solved=tot_s, total_scenes=tot_n, robots_full=n_full, n_robots=len(rows))


def main():
    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    print(f"FLEET-MPPI-BENCHMARK — GPU MPPI per robot x {M_SCENES} genuine scenes x 2 scene types (device='{dev}')")
    res = {m: _run_mode(m, dev) for m in ("single", "clutter")}
    ts = sum(res[m]["total_solved"] for m in res); tn = sum(res[m]["total_scenes"] for m in res)
    ok = ts == tn and tn > 0
    cl = res["clutter"]
    print(f"\nVERDICT: fleet-mppi-benchmark = {'ALL SOLVED' if ok else 'PARTIAL'}. "
          + (f"GPU MPPI solves {ts}/{tn} genuine scenes across {res['single']['n_robots']} robots (6 and 7 DOF, "
             "several vendors) in BOTH the single-box and the 5-box clutter mode, every path dense-oracle-verified. "
             f"Clutter {cl['total_solved']}/{cl['total_scenes']}, so the capability carries over to tight scenes as "
             "measured, not as inferred. " if ok else
             f"{ts}/{tn} solved (single {res['single']['total_solved']}/{res['single']['total_scenes']}, clutter "
             f"{cl['total_solved']}/{cl['total_scenes']}); see the per-robot rows. ")
          + "Scope: broad-phase coarse optimisation plus dense verification; joint-space goals; reach-scaled scenes "
            "rather than a published benchmark scene library; the harness takes another planner on the same scenes "
            "and the same gate.")
    out = dict(device=dev, m_scenes=M_SCENES, modes=res, total_solved=ts, total_scenes=tn, all_solved=ok)
    (ROOT / "reports" / "fleet_mppi_benchmark.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("  wrote reports/fleet_mppi_benchmark.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
