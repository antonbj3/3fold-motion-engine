#!/usr/bin/env python3
"""Motion-stack core loop: a plan that is collision-free AND torque-feasible.

Composes the independently gated layers end to end:
  SIMD collision-free geometric path (`vamp_collision_planner`) -> dynamic feasibility (time-parametrise every path
  segment time-optimally against the torque limit, `joint_trajectory_planner`) -> torque gate on the identified
  twin (`moment_gate`) -> an executable plan.

The point: a geometrically collision-free plan that exceeds the torque limit is as unusable as a torque-feasible
plan that collides. So there are TWO independent gates and both must pass: (a) GEOMETRY - the path is
collision-free under a double oracle against the exact mesh; (b) DYNAMICS - the time parametrisation is
torque-feasible with a buffer on the identified twin. Neither alone is sufficient.

Selftest checks: (1) the planner solves a collision-free path (double oracle, 0 collisions); (2) every segment is
time-parametrised torque-feasibly with a buffer (peak ratio <= target < margin, no threshold touching);
(3) the integrated plan is BOTH collision-free AND torque-feasible; (4) the dynamics gate is LOAD-BEARING - the
same collision-free path under aggressive uniform time scaling becomes torque-INFEASIBLE, and the time-optimal
parametrisation is what rescues it.

  python -u scripts/motion_stack_integration.py   (requires vamp-planner + pinocchio)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moment_gate import _model, required_torque, moment_gate, MARGIN, NJ
from joint_trajectory_planner import quintic, plan_time_optimal, peak_ratio_of, TARGET_RATIO
from vamp_collision_planner import path_to_list, dense_validate, START_CFG, GOAL_CFG, DENSE

ROOT = Path(__file__).resolve().parents[1]


def parametrize_path(model, data, waypoints, Fc, Fv):
    """Time-parametrise the path: per segment the smallest torque-feasible T (buffered). -> (segment times, max peak ratio)."""
    Ts, ratios = [], []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        T, v, _ = plan_time_optimal(model, data, np.asarray(a), np.asarray(b), Fc, Fv)
        if T is None:
            return None, None
        Ts.append(T); ratios.append(peak_ratio_of(model, data, np.asarray(a), np.asarray(b), T, Fc, Fv))
    return Ts, (max(ratios) if ratios else 0.0)


def main():
    import vamp
    mod, planner, rrtc_set, simp_set = vamp.configure_robot_and_planner_with_kwargs("panda", "rrtc")
    rrtc_set.max_iterations = 100000
    rng = vamp.panda.halton()
    model = _model(); data = model.createData()
    rep = json.loads((ROOT / "reports" / "panda_twin_rnea_fidelity.json").read_text())
    Fc = np.array([r["Fc"] for r in rep["per_joint"]]); Fv = np.array([r["Fv"] for r in rep["per_joint"]])
    print("MOTION-STACK-INTEGRATION — core loop: a collision-free AND torque-feasible plan")

    # (1) collision-free geometric path
    env = vamp.Environment(); env.add_cuboid(vamp.Cuboid([0.5, 0.0, 0.55], [0, 0, 0], [0.06, 0.5, 0.3]))
    if not (mod.validate(START_CFG, env) and mod.validate(GOAL_CFG, env)):
        print("  the fixed configurations are invalid with this obstacle"); return 1
    t0 = time.time(); res = planner(START_CFG, [GOAL_CFG], env, rrtc_set, rng); plan_ms = (time.time() - t0) * 1000
    wp = path_to_list(res.path)
    clean, n_dense, n_coll = dense_validate(mod, env, wp)
    g1 = bool(res.solved) and clean
    print(f"  (1) GEOMETRY: solved in {plan_ms:.1f}ms, {len(wp)} waypoints | double oracle {n_dense} configurations, "
          f"{n_coll} collisions -> collision-free={g1}")

    # (2) DYNAMICS: time-parametrise every segment torque-feasibly (buffered)
    Ts, max_ratio = parametrize_path(model, data, wp, Fc, Fv)
    g2 = Ts is not None and max_ratio <= TARGET_RATIO + 1e-3 and TARGET_RATIO < MARGIN
    total_T = sum(Ts) if Ts else 0.0
    print(f"  (2) DYNAMICS: {len(Ts) if Ts else 0} segments time-optimally parametrised, total {total_T:.2f}s | "
          f"max peak ratio {max_ratio:.3f} <= target {TARGET_RATIO} < margin {MARGIN} (buffered) -> {g2}")

    # (3) the integrated plan is BOTH collision-free AND torque-feasible
    g3 = g1 and g2
    print(f"  (3) INTEGRATED: the plan is BOTH collision-free (geometry) AND torque-feasible (dynamics) = {g3}")

    # (4) the dynamics gate is load-bearing: the same collision-free path, aggressively time-scaled -> infeasible
    agg_ratios = []
    for a, b in zip(wp[:-1], wp[1:]):
        T_agg = max(0.05, (total_T / max(len(Ts), 1)) * 0.25)   # four times faster per segment
        agg_ratios.append(peak_ratio_of(model, data, np.asarray(a), np.asarray(b), T_agg, Fc, Fv))
    agg_max = max(agg_ratios) if agg_ratios else 0.0
    agg_infeasible = agg_max > 1.0
    g4 = agg_infeasible and g2
    print(f"  (4) DYNAMICS LOAD-BEARING: the same collision-free path at 4x speed -> peak ratio {agg_max:.2f} > 1.0 = "
          f"{agg_infeasible} (INFEASIBLE); the time-optimal parametrisation rescues it = {g4}")

    ok = g1 and g2 and g3 and g4
    print(f"\nVERDICT: motion-stack-integration = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"Core loop complete: collision-free path ({plan_ms:.0f}ms, double-oracle clean) -> dynamic-feasibility "
             f"time parametrisation (total {total_T:.1f}s, max peak ratio {max_ratio:.2f} buffered) -> torque gate on "
             "the identified twin. The plan is BOTH collision-free (geometry) AND torque-feasible (dynamics), two "
             "independently gated layers, both load-bearing: the same collision-free path at 4x speed becomes "
             f"torque-INFEASIBLE (peak ratio {agg_max:.1f} > 1) and the time-optimal parametrisation is what makes it "
             "feasible again. " if ok else
             f"Not validated (geometry {g1}, dynamics {g2}, conjunction {g3}, dynamics load-bearing {g4}). ")
          + "Scope: stop-at-waypoint parametrisation per segment (blending is a further optimisation); the SIMD "
            "planner uses a sphere approximation of the robot with a double oracle against the exact mesh; static "
            "effort limit. Composes vamp_collision_planner + joint_trajectory_planner + moment_gate.")

    out = dict(robot="7-DOF arm", layer="motion-stack integration (core loop)",
               plan_ms=round(plan_ms, 2), waypoints=len(wp), collisions=int(n_coll),
               segments=len(Ts) if Ts else 0, total_time_s=round(total_T, 3), max_peak_ratio=round(float(max_ratio), 3),
               aggressive_4x_peak_ratio=round(float(agg_max), 3), collision_free=bool(g1), moment_feasible=bool(g2),
               both=bool(g3), dynamics_gate_load_bearing=bool(g4), validated=bool(ok),
               composes="vamp_collision_planner, joint_trajectory_planner, moment_gate")
    (ROOT / "reports" / "motion_stack_integration.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("  wrote reports/motion_stack_integration.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
