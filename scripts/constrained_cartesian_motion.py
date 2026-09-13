#!/usr/bin/env python3
"""Task-space constrained motion: the EE orientation is held fixed along the whole path.

The EE position moves linearly between two points while the EE orientation is held fixed; full-pose IK is
solved per interpolated pose (interpolated position + fixed orientation), giving a joint path that keeps
the orientation throughout, not only at the end points. Composes movement_ik (full-pose damped-LS) with
coal collision checking.

Selftest checks: (1) constrained IK reaches every task-space waypoint (position + fixed orientation within
tolerance); (2) the whole path keeps the orientation within tolerance (dense FK, not only the end points);
(3) the constraint binds - the same end point is also reachable at a different orientation, so holding
R_fix is a real choice; (4) the constrained path is collision-free and honest when a pose is unreachable.

  python -u scripts/constrained_cartesian_motion.py   (requires pinocchio + hppfcl/coal)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fleet_agnostic_planner import load, in_collision
from movement_ik import ik_solve, fk

ROOT = Path(__file__).resolve().parents[1]


def rot_err(R_a, R_b):
    """Orientation error (rad) between two rotations."""
    return float(np.linalg.norm(pin.log3(R_a.T @ R_b)))


def constrained_path(model, data, fid, R_fix, p0, p1, lo, hi, gm, gd, n=25, tol=1e-3, q_init=None):
    """Task-space interpolated path: position p0 -> p1 linearly, orientation fixed at R_fix. Warm-started IK per
    step -> joint path. q_init is the known start configuration (the s=0 solution), used to seed the first step."""
    qs, q_seed = [], (np.asarray(q_init, float) if q_init is not None else 0.5 * (lo + hi))
    for s in np.linspace(0, 1, n):
        target = pin.SE3(R_fix, p0 + s * (p1 - p0))
        q, conv, free = ik_solve(model, data, fid, target, q_seed, lo, hi, gm=gm, gd=gd, tol=tol)
        if not conv:
            return None, s
        qs.append(q); q_seed = q                       # warm-start the next step from this solution (continuity)
    return np.array(qs), None


def main():
    model, data, gm, gd, _ = load(obstacle_box=[0.12, 0.12, 0.3], box_pose=[0.5, 0.0, 0.2])
    fid = model.getFrameId("tool0"); nj = model.nv
    lo, hi = model.lowerPositionLimit[:nj], model.upperPositionLimit[:nj]
    rng = np.random.default_rng(0)
    print("constrained-cartesian-motion — EE orientation held fixed along the whole path (UR10e)")

    def free(q): return not in_collision(model, data, gm, gd, q)
    # pick a reachable collision-free start configuration; its EE pose gives the fixed orientation and start position
    q0 = next((q for q in (lo + rng.random(nj) * (hi - lo) for _ in range(1500)) if free(q)), None)
    if q0 is None:
        print("  no free start configuration"); return 1
    M0 = fk(model, data, fid, q0); R_fix = M0.rotation.copy(); p0 = M0.translation.copy()
    # reachable collision-free constrained motion (position delta, fixed orientation); try a few directions
    qs = None
    for dirv in ([0, 0.2, 0.1], [0, -0.2, 0.1], [0.15, 0, 0.15], [0, 0.25, -0.1], [-0.15, 0.1, 0.1]):
        p1 = p0 + np.array(dirv, float)
        cand, fail_s = constrained_path(model, data, fid, R_fix, p0, p1, lo, hi, gm, gd, q_init=q0)
        if cand is not None and all(free(q) for q in cand):
            qs = cand; break
    if qs is None:
        print("  no feasible constrained path found"); return 1
    print(f"  orientation held fixed; EE position {np.round(p0,3)} -> {np.round(p1,3)} (delta {np.linalg.norm(p1-p0):.2f}m)")

    # (2) the whole constrained path: dense FK -> orientation error against R_fix throughout
    rot_errs = [rot_err(fk(model, data, fid, q).rotation, R_fix) for q in qs]
    pos_reached = float(np.linalg.norm(fk(model, data, fid, qs[-1]).translation - p1))
    max_rot_constrained = float(np.max(rot_errs))
    coll_free = all(free(q) for q in qs)
    print(f"  constrained-bana: {len(qs)} steg, max orienterings-fel {max_rot_constrained*1000:.2f} mrad, slut-pos-fel {pos_reached*1000:.2f}mm, kollisionsfri {coll_free}")

    # (3) the constraint binds: is p1 reachable at a different orientation? If so, R_fix is one of several reachable
    #     orientations, so the constrained layer does real work (free motion could land on another orientation).
    ang = 0.5236  # 30°
    R_rot = R_fix @ pin.exp3(np.array([0.0, 0.0, ang]))
    q_alt, conv_alt, free_alt = ik_solve(model, data, fid, pin.SE3(R_rot, p1), q0, lo, hi, gm=gm, gd=gd, tol=1e-3)
    alt_reached = bool(conv_alt and free_alt and rot_err(fk(model, data, fid, q_alt).rotation, R_rot) < 5e-3)
    print(f"  constraint test: p1 reachable collision-free at a +30 deg rotated orientation = {alt_reached}")

    g1 = qs is not None and pos_reached < 5e-3
    g2 = max_rot_constrained < 5e-3                          # orientation held throughout (<5 mrad)
    g3 = alt_reached                                         # the constraint binds: p1 is reachable at another orientation
    g4 = coll_free
    print(f"  (1) constrained IK reaches the task-space waypoints (final position {pos_reached*1000:.2f}mm): {g1}")
    print(f"  (2) the whole path keeps the orientation (<5 mrad, max {max_rot_constrained*1000:.2f}): {g2}")
    print(f"  (3) the constraint binds (p1 is reachable at another orientation): {g3}")
    print(f"  (4) constrained-bana kollisionsfri: {g4}")

    ok = g1 and g2 and g3 and g4
    print(f"\nVERDICT: constrained-cartesian-motion = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"Task-space constrained motion on UR10e: the EE position moves "
             f"{np.round(np.linalg.norm(p1-p0),2)}m while the orientation is held fixed along the WHOLE path "
             f"(max {max_rot_constrained*1000:.1f} mrad over {len(qs)} steps, verified by dense FK, not only at the "
             "end points), collision-free. The constraint binds: p1 is also reachable collision-free at a +30 deg "
             "rotated orientation, so R_fix is one of several reachable orientations and the constrained layer does "
             "real work. Use cases: carrying level, welding along a line, gluing, machining. " if ok else
             f"Not validated (task waypoints {g1}, orientation held {g2}, constraint binds {g3}, collision-free {g4}). ")
          + "Scope: task-space interpolation (linear position, fixed orientation) with per-step full-pose warm-started "
          "IK; verified densely (25 FK steps), not a continuous guarantee; orientation constraint (3 DOF locked); "
          "UR10e demo, robot-agnostic through the pinocchio frame; honest when a constrained pose is unreachable.")

    out = dict(layer="constrained Cartesian motion (orientation-locked task-space motion)",
               robot="UR10e", ee_frame="tool0", steps=len(qs) if qs is not None else 0,
               max_rot_err_constrained_mrad=round(max_rot_constrained * 1000, 3),
               end_pos_err_mm=round(pos_reached * 1000, 3), collision_free=bool(coll_free),
               orientation_maintained=bool(g2), constraint_binds=bool(g3),
               binds_proof="p1 reachable at a +30deg-rotated orientation (collision-free) too -> R_fix is one of several reachable orientations -> constraint does real work",
               note="task-space interpolation (linear position, fixed orientation) + per-step full-pose IK; EE orientation held within tol throughout (dense-FK verified); p1 reachable at other orientations so the constraint binds",
               validated=bool(ok))
    (ROOT / "reports" / "constrained_cartesian_motion.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("  wrote reports/constrained_cartesian_motion.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
