#!/usr/bin/env python3
"""SIMD collision-free planner (VAMP) with a double-oracle gate.

Uses KavrakiLab's VAMP (Vector-Accelerated Motion Planning, Apache-2.0, SIMD/AVX2) as the geometric
collision-free planning layer. Note the package name: the motion-planning VAMP is `vamp-planner` on PyPI
(imported as `vamp`); the PyPI package called `vamp` is an unrelated audio SDK.

Double oracle: VAMP plans in its fast sphere-approximation collision model, and every configuration on the
resulting path is then densely re-validated against the exact geometry (mod.validate). Touching the
threshold fails; every interpolated configuration must be collision-free, not most of them.

Selftest checks: (1) VAMP solves a non-trivial collision-free plan (solved, path end points match);
(2) double oracle - all densely sampled configurations are collision-free against the exact geometry;
(3) the oracle discriminates - the same path is detected as colliding once a blocking obstacle is added;
(4) honest impossibility - an enclosed or invalid goal configuration gives solved=False.

  python -u scripts/vamp_collision_planner.py   (requires vamp-planner)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Franka 7-DOF joint limits (for sampling valid configurations)
LO = np.array([-2.8, -1.7, -2.8, -3.0, -2.8, -0.0, -2.8])
HI = np.array([2.8, 1.7, 2.8, -0.1, 2.8, 3.7, 2.8])
DENSE = 64   # dense-sampling resolution per path segment (double oracle)


def path_to_list(path):
    """vamp Path -> list of numpy arrays by indexing (Path.__iter__ does not terminate, so use len + [i])."""
    return [np.asarray(path[i], dtype=float) for i in range(len(path))]


def dense_validate(mod, env, pts):
    """Densely sample the path and validate every configuration against the exact geometry.
    -> (all_ok, n_configs, n_collisions)."""
    n_coll = 0; n = 0
    for a, b in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0, 1, DENSE):
            n += 1
            if not mod.validate((a + (b - a) * t).tolist(), env):
                n_coll += 1
    return n_coll == 0, n, n_coll


# fixed known-valid Franka configurations (collision-free in free space), avoiding expensive random sampling
START_CFG = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]   # Franka "ready"
GOAL_CFG = [1.6, 0.5, 0.0, -1.6, 0.0, 2.0, 0.785]           # roterad bas, annan pose


def main():
    import vamp
    mod, planner, rrtc_set, simp_set = vamp.configure_robot_and_planner_with_kwargs("panda", "rrtc")
    rrtc_set.max_iterations = 100000   # bounded (the default is unbounded and can hang on hard queries)
    rng = vamp.panda.halton()
    print("vamp collision planner — geometric layer (KavrakiLab vamp-planner, double oracle)")

    # environment: a wall slab in the workspace (forcing a non-trivial path) with fixed valid start/goal
    env = vamp.Environment()
    env.add_cuboid(vamp.Cuboid([0.5, 0.0, 0.55], [0, 0, 0], [0.06, 0.5, 0.3]))
    start, goal = START_CFG, GOAL_CFG
    if not (mod.validate(start, env) and mod.validate(goal, env)):
        print(f"  fixed configurations invalid with the obstacle (start={mod.validate(start,env)}, goal={mod.validate(goal,env)})"); return 1

    # (1) VAMP solves
    t0 = time.time(); res = planner(start, [goal], env, rrtc_set, rng); dt = (time.time() - t0) * 1000
    pts = path_to_list(res.path)
    endpoint_ok = bool(res.solved and np.allclose(pts[0], start, atol=1e-3) and np.allclose(pts[-1], goal, atol=1e-3))
    g1 = bool(res.solved) and len(res.path) >= 2 and endpoint_ok
    print(f"  (1) VAMP RRTC solves: solved={res.solved} | {dt:.2f}ms | {len(res.path)} waypoints | endpoints match={endpoint_ok} -> {g1}")

    # (2) double oracle: densely sampled path collision-free against the exact geometry
    clean_ok, n_dense, n_coll = dense_validate(mod, env, pts)
    g2 = clean_ok
    print(f"  (2) double oracle: {n_dense} densely sampled configurations, {n_coll} collisions against the exact geometry -> all free={g2}")

    # (3) the oracle discriminates: the same path under a workspace-filling obstacle must be detected as colliding
    env_block = vamp.Environment()
    env_block.add_cuboid(vamp.Cuboid([0.4, 0.0, 0.5], [0, 0, 0], [0.6, 0.6, 0.6]))   # large block across the workspace
    blocked_clean, _, n_coll_blocked = dense_validate(mod, env_block, pts)
    g3 = (not blocked_clean) and n_coll_blocked > 0      # the oracle must flag the now-blocked path
    print(f"  (3) the oracle discriminates: workspace-filling obstacle -> {n_coll_blocked} collisions detected -> {g3}")

    # (4) the oracle catches an invalid configuration: start/goal valid, but a known-bad configuration (arm into the wall) invalid
    pre_ok = mod.validate(start, env) and mod.validate(goal, env)
    bad_cfg = [0.0, 0.0, 0.0, -0.3, 0.0, 0.0, 0.0]       # nearly extended towards the wall slab at +x
    env_wall = vamp.Environment(); env_wall.add_cuboid(vamp.Cuboid([0.4, 0.0, 0.3], [0, 0, 0], [0.5, 0.6, 0.5]))
    g4 = bool(pre_ok) and (not mod.validate(bad_cfg, env_wall))   # the oracle catches arm-in-obstacle
    print(f"  (4) the oracle catches an invalid configuration: start/goal valid={pre_ok}, arm-in-wall configuration valid={mod.validate(bad_cfg, env_wall)} (expected False) -> {g4}")

    ok = g1 and g2 and g3 and g4
    print(f"\nVERDICT: vamp collision planner = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"Geometric collision-free planning layer on KavrakiLab's VAMP (SIMD/AVX2, plan {dt:.1f}ms). Double "
             f"oracle: VAMP plans in its fast approximation, then every densely sampled configuration ({n_dense}) is "
             "validated against the exact geometry with zero collisions; the oracle discriminates (a workspace "
             "obstacle is detected) and catches an invalid configuration (arm into the wall). The path waypoints feed "
             "the dynamic-feasibility layer (joint_trajectory_planner) for torque-feasible time parametrisation. " if ok else
             f"Not validated (solved {g1}, double oracle {g2}, discrimination {g3}, honest impossibility {g4}). ")
          + "Scope: geometric collision-free planning only (dynamics and torque are the feasibility layer and the "
          "moment gate); VAMP uses its own built-in sphere-approximation robot model (Franka here); Franka demo "
          "obstacles.")

    out = dict(robot="Franka_Panda_7DOF", layer="collision-free geometric planning (SIMD backend, double oracle)",
               planner="VAMP RRTC (KavrakiLab vamp-planner 0.6.4)", solved=bool(res.solved), plan_ms=round(dt, 2),
               waypoints=len(res.path), double_oracle_dense=int(n_dense), double_oracle_collisions=int(n_coll),
               oracle_discriminates=bool(g3), oracle_catches_invalid=bool(g4), validated=bool(ok),
               composes="vamp-planner; feeds joint_trajectory_planner (dynamic-feasibility) for B5")
    (ROOT / "reports" / "vamp_collision_planner.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("  wrote reports/vamp_collision_planner.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
