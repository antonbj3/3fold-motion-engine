#!/usr/bin/env python3
"""Cartesian goal -> executable plan on a fleet robot through the coal layers (UR10e).

A SIMD planner only supports its built-in robot models, so a composition built on it does not generalise to an
arbitrary fleet URDF. This cell composes the robot-agnostic coal layers end to end on the UR10e instead:
Cartesian end-effector goal -> coal IK (`movement_ik`) -> coal RRT-Connect (`fleet_agnostic_planner`, exact-mesh
collision for any URDF) -> effort feasibility (RNEA against the URDF effort limit) -> an executable plan.
Hot-swap: the SIMD planner for its built-ins (fast), coal for everything else (general).

Provenance: collision (exact coal mesh), kinematics and the effort limit are REAL, read from the URDF. Dynamic
feasibility is RNEA with URDF inertials and NOMINAL friction (Fc = Fv = 0), so effort feasibility is the real
gate; an identified friction model for this robot is not in this loop. The effort buffer is TARGET < 1, so the
gate never passes by touching the threshold.

Selftest checks: (1) the robot loads under coal and IK reaches the Cartesian goal collision-free; (2) coal
RRT-Connect finds a collision-free path to the IK goal, with dense edge validation against the exact mesh;
(3) effort feasibility - every segment is time-optimal so that the RNEA peak ratio stays <= TARGET < 1;
(4) end to end: Cartesian goal -> collision-free AND effort-feasible plan; (5) honest on unreachable - a goal
far outside the workspace does not converge.

  python -u scripts/fleet_cartesian_to_plan.py   (requires pinocchio + hppfcl/coal)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fleet_agnostic_planner import load, in_collision, edge_free, rrt_connect
from movement_ik import ik_multistart, fk
from joint_trajectory_planner import quintic, TARGET_RATIO

ROOT = Path(__file__).resolve().parents[1]


def peak_ratio_generic(model, data, q0, qf, T, nj):
    """RNEA effort ratio (URDF inertials + nominal friction) against the URDF effort limit. Robot-agnostic (nj = model.nv)."""
    q, dq, ddq = quintic(np.asarray(q0), np.asarray(qf), T)
    tau = np.array([pin.rnea(model, data, q[i], dq[i], ddq[i]) for i in range(len(q))])
    return float(np.max(np.max(np.abs(tau), axis=0) / model.effortLimit[:nj]))


def time_optimal_generic(model, data, q0, qf, nj, target=TARGET_RATIO, T_lo=0.05, T_hi=20.0, iters=40):
    """Smallest time scale T with peak ratio <= target (the peak decreases with T). -> (T, ratio), or (None, ratio) if infeasible."""
    if peak_ratio_generic(model, data, q0, qf, T_hi, nj) > target:
        return None, peak_ratio_generic(model, data, q0, qf, T_hi, nj)
    lo, hi = T_lo, T_hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if peak_ratio_generic(model, data, q0, qf, mid, nj) <= target:
            hi = mid
        else:
            lo = mid
    return hi, peak_ratio_generic(model, data, q0, qf, hi, nj)


def main():
    model, data, gm, gd, obs = load(obstacle_box=[0.2, 0.2, 0.5], box_pose=[0.55, 0.0, 0.3])
    fid = model.getFrameId("tool0"); nj = model.nv
    lo, hi = model.lowerPositionLimit[:nj], model.upperPositionLimit[:nj]
    rng = np.random.default_rng(0)
    print(f"FLEET-CARTESIAN-TO-PLAN — Cartesian goal -> plan on UR10e through the coal layers, dof {nj}")

    def free(q): return not in_collision(model, data, gm, gd, q)
    # start (collision-free) + Cartesian goal = the EE pose of a reachable free configuration
    q_start = next((q for q in (lo + rng.random(nj) * (hi - lo) for _ in range(800)) if free(q)), None)
    q_tgt = next((q for q in (lo + rng.random(nj) * (hi - lo) for _ in range(800)) if free(q)), None)
    if q_start is None or q_tgt is None:
        print("  could not find free start/goal configurations"); return 1
    target = fk(model, data, fid, q_tgt)
    print(f"  Cartesian goal (tool0): pos {np.round(target.translation,3)}")

    # (1) coal IK: Cartesian goal -> collision-free joint configuration
    t0 = time.time(); q_goal, conv, ikfree, tries = ik_multistart(model, data, fid, target, lo, hi, rng, gm=gm, gd=gd)
    ik_ms = (time.time() - t0) * 1000
    oMf = fk(model, data, fid, q_goal)
    pos_err = float(np.linalg.norm(oMf.translation - target.translation))
    rot_err = float(np.linalg.norm(pin.log3(oMf.rotation.T @ target.rotation)))
    g1 = bool(conv) and pos_err < 1e-3 and rot_err < 1e-3 and bool(ikfree)
    print(f"  (1) coal IK: Cartesian goal -> joint configuration {ik_ms:.0f}ms ({tries} restarts) "
          f"pos {pos_err*1000:.2f}mm rot {rot_err*1000:.2f}mrad collision-free {ikfree} -> {g1}")

    # (2) coal RRT-Connect -> collision-free path start -> IK goal, with dense edge validation
    g2 = False; rrt_ms = 0.0; wp = []; dense_ok = False
    if g1:
        t0 = time.time(); path = rrt_connect(model, data, gm, gd, q_start, q_goal, lo, hi, rng); rrt_ms = (time.time() - t0) * 1000
        if path is not None:
            wp = path
            dense_ok = all(edge_free(model, data, gm, gd, np.asarray(a), np.asarray(b), res=0.02) for a, b in zip(wp[:-1], wp[1:]))
            g2 = dense_ok
    print(f"  (2) coal RRT-Connect: {rrt_ms:.0f}ms, {len(wp)} waypoints, dense edge validation (res 0.02) {dense_ok} -> {g2}")

    # (3) effort feasibility: every segment time-optimal so the RNEA peak ratio stays <= TARGET < 1
    g3 = False; total_T = 0.0; max_ratio = 9.9; infeasible_seg = 0
    if g2:
        Ts, ratios = [], []
        for a, b in zip(wp[:-1], wp[1:]):
            T, r = time_optimal_generic(model, data, a, b, nj)
            if T is None:
                infeasible_seg += 1
            else:
                Ts.append(T); ratios.append(r)
        total_T = sum(Ts); max_ratio = max(ratios) if ratios else 9.9
        g3 = infeasible_seg == 0 and max_ratio <= TARGET_RATIO + 1e-3 and TARGET_RATIO < 1.0
    print(f"  (3) effort feasibility: {len(wp)-1} segments time-optimal, total {total_T:.2f}s, "
          f"max peak ratio {max_ratio:.3f} <= {TARGET_RATIO} < 1 ({infeasible_seg} infeasible) -> {g3}")

    # (4) end to end
    g4 = g1 and g2 and g3
    print(f"  (4) END-TO-END: Cartesian goal -> collision-free + effort-feasible plan = {g4}")

    # (5) honest on an unreachable Cartesian goal
    far = pin.SE3(np.eye(3), np.array([9.0, 9.0, 9.0]))
    _, conv_far, _, _ = ik_multistart(model, data, fid, far, lo, hi, rng, n_restart=10)
    g5 = not conv_far
    print(f"  (5) honest on an unreachable goal (9,9,9): conv={conv_far} (must be False) -> {g5}")

    ok = g1 and g2 and g3 and g4 and g5
    print(f"\nVERDICT: fleet-cartesian-to-plan = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"Cartesian goal -> plan on a fleet robot outside any SIMD planner's built-in set: UR10e, Cartesian EE "
             f"goal -> coal IK ({pos_err*1000:.1f}mm) -> coal RRT-Connect ({rrt_ms:.0f}ms, {len(wp)} waypoints, dense "
             f"exact-mesh edge validation) -> effort feasibility (RNEA against the URDF limit, {len(wp)-1} segments "
             f"time-optimal, peak ratio {max_ratio:.2f} <= {TARGET_RATIO} buffered) -> executable plan. Works for any "
             "fleet URDF because collision is exact coal, not a built-in model. Geometry, kinematics and effort are "
             "REAL from the URDF; unreachable goals are reported as unreachable. " if ok else
             f"Not validated (IK {g1}, coal RRT {g2}, effort {g3}, end-to-end {g4}, unreachable {g5}). ")
          + "Scope: composition, no new layer; coal RRT is slower than a SIMD planner but general. Dynamic "
            "feasibility is RNEA with URDF inertials and NOMINAL friction (Fc = Fv = 0), so the effort gate is real "
            "while friction/twin fidelity for this robot is not in this loop. Composes fleet_agnostic_planner "
            "(coal RRT) + movement_ik (coal IK) + RNEA effort.")

    out = dict(layer="cartesian goal -> coal IK -> coal RRT -> RNEA effort-feasible plan (fleet-general)",
               robot="UR10e", ee_frame="tool0", dof=nj,
               ik_pos_err_mm=round(pos_err * 1000, 3), ik_rot_err_mrad=round(rot_err * 1000, 3), ik_restarts=tries,
               coal_rrt_ms=round(rrt_ms, 1), waypoints=len(wp), dense_edge_validated=bool(dense_ok),
               segments=max(0, len(wp) - 1), total_time_s=round(total_T, 3), max_peak_ratio=round(float(max_ratio), 3),
               infeasible_segments=infeasible_seg, ik_reaches_target=bool(g1), collision_free=bool(g2),
               effort_feasible=bool(g3), end_to_end=bool(g4), honest_on_unreachable=bool(g5),
               provenance="collision (coal exact mesh) + kinematics + effort REAL from URDF; dynamics = RNEA URDF "
                          "inertials + nominal friction (Fc=Fv=0)",
               composes="fleet_agnostic_planner (coal RRT) + movement_ik (coal IK) + RNEA effort feasibility",
               validated=bool(ok))
    (ROOT / "reports" / "fleet_cartesian_to_plan.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("  wrote reports/fleet_cartesian_to_plan.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
