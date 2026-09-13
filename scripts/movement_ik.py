#!/usr/bin/env python3
"""Inverse kinematics: Cartesian goal -> collision-free joint configuration.

pinocchio damped-least-squares IK (EE pose -> q), clipped to the joint limits and collision-checked, so the
planner contract can take a Cartesian goal (IK -> joint goal -> plan). Uses fleet_agnostic_planner for the
model and the coal collision scene.

Selftest checks: (1) IK converges on a reachable target (FK(q_ik) ~= target, position and rotation error
below tolerance) inside the joint limits; (2) the result is collision-free (coal, self + environment);
(3) an unreachable target does not converge; (4) a second independent target also solves.

  python -u scripts/movement_ik.py   (requires pinocchio + hppfcl/coal)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fleet_agnostic_planner import load, in_collision

ROOT = Path(__file__).resolve().parents[1]
URDF = str(ROOT / "assets/robots/ur_description/ur10e.urdf")


def ik_solve(model, data, fid, target, q0, lo, hi, iters=300, tol=1e-4, damp=1e-6, step=0.5,
             gm=None, gd=None, cfree=None):
    """Damped-LS IK: EE frame fid -> SE3 target. Limit-clipped; if gm/gd (or a cfree predicate) are given, the
    solution must be collision-free. -> (q, conv, free)."""
    q = np.clip(np.array(q0, float), lo, hi)
    conv = False
    for _ in range(iters):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        oMf = data.oMf[fid]
        err = pin.log6(oMf.actInv(target)).vector       # 6D twist error in the EE frame
        if np.linalg.norm(err) < tol:
            conv = True
            break
        J = pin.computeFrameJacobian(model, data, q, fid, pin.LOCAL)
        dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), err)
        q = np.clip(pin.integrate(model, q, dq * step), lo, hi)
    free = None
    if cfree is not None:                             # WorldModel path: external free predicate (self + environment)
        free = bool(cfree(q))
    elif gm is not None:
        free = not in_collision(model, data, gm, gd, q)
    return q, conv, free


def ik_multistart(model, data, fid, target, lo, hi, rng, n_restart=30, gm=None, gd=None, tol=1e-4, cfree=None):
    """Multi-restart damped-LS IK (avoids local minima; no tolerance relaxation). -> (q, conv, free, tries)."""
    nq = model.nq
    for t in range(n_restart):
        q0 = lo + rng.random(nq) * (hi - lo)
        q, conv, free = ik_solve(model, data, fid, target, q0, lo, hi, gm=gm, gd=gd, tol=tol, cfree=cfree)
        if conv and (free is None or free):
            return q, True, free, t + 1
    return q, False, free, n_restart


def fk(model, data, fid, q):
    pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
    return data.oMf[fid].copy()


def main():
    model, data, gm, gd, _ = load(obstacle_box=[0.2, 0.2, 0.5], box_pose=[0.55, 0.0, 0.3])
    fid = model.getFrameId("tool0")
    nq = model.nq
    lo, hi = model.lowerPositionLimit[:nq], model.upperPositionLimit[:nq]
    rng = np.random.default_rng(0)
    print(f"IK — cartesian goal -> collision-free joint config; EE=tool0, dof{nq}")

    # reachable, collision-free target: FK of a random free configuration
    def free(q): return not in_collision(model, data, gm, gd, q)
    q_true = next((q for q in (lo + rng.random(nq) * (hi - lo) for _ in range(500)) if free(q)), None)
    target = fk(model, data, fid, q_true)
    q0 = next((q for q in (lo + rng.random(nq) * (hi - lo) for _ in range(500)) if free(q)), None)

    q_ik, conv, ik_free, tries = ik_multistart(model, data, fid, target, lo, hi, rng, gm=gm, gd=gd)
    oMf = fk(model, data, fid, q_ik)
    pos_err = float(np.linalg.norm(oMf.translation - target.translation))
    rot_err = float(np.linalg.norm(pin.log3(oMf.rotation.T @ target.rotation)))
    within = bool(np.all(q_ik >= lo - 1e-6) and np.all(q_ik <= hi + 1e-6))

    # generality: a second independent reachable target also solves
    q_t2 = next((q for q in (lo + rng.random(nq) * (hi - lo) for _ in range(500)) if free(q)), None)
    target2 = fk(model, data, fid, q_t2)
    _, conv2, _, _ = ik_multistart(model, data, fid, target2, lo, hi, rng, gm=gm, gd=gd)

    # unreachable target (far outside the workspace) -> must not converge
    far = pin.SE3(np.eye(3), np.array([5.0, 5.0, 5.0]))
    _, conv_far, _, _ = ik_multistart(model, data, fid, far, lo, hi, rng, n_restart=10)

    print(f"  reachable target: conv={conv} ({tries} restarts) pos err {pos_err*1000:.2f}mm rot err {rot_err*1000:.2f}mrad | within limits={within} collision-free={ik_free}")
    print(f"  2nd independent target: conv={conv2} | unreachable target (5,5,5): conv={conv_far} (expected False)")

    g1 = conv and pos_err < 1e-3 and rot_err < 1e-3 and within
    g2 = bool(ik_free)
    g3 = not conv_far
    g4 = conv2   # generality: IK also solves a second independent target
    print(f"  (1) IK converges on the reachable target within limits ({tries} restarts): {g1}")
    print(f"  (2) result collision-free (coal): {g2}")
    print(f"  (3) honest on unreachable (conv=False): {g3}")
    print(f"  (4) generality (2nd independent target solves): {g4}")

    ok = g1 and g2 and g3 and g4
    print(f"\nVERDICT: movement-ik = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"Damped-LS IK (EE pose -> q) converges on the reachable target "
             f"(pos {pos_err*1000:.1f}mm, rot {rot_err*1000:.1f}mrad), within the joint limits, collision-free "
             "(coal, self + environment), and returns non-converged on an unreachable target. " if ok else
             f"Not validated (conv {g1}, free {g2}, honest-on-unreachable {g3}, generality {g4}). ")
          + "Scope: damped-LS with multi-restart (local per seed, global via 30 random restarts); the solution is "
          "collision-CHECKED but not collision-AVOIDING during IK (a colliding solution is rejected and restarted); "
          "UR10e demo, the pattern is robot-agnostic.")

    out = dict(layer="IK (cartesian goal -> collision-free joint config)",
               ee_frame="tool0", converged=bool(conv), pos_err_mm=round(pos_err * 1000, 3), rot_err_mrad=round(rot_err * 1000, 3),
               within_limits=within, collision_free=bool(ik_free), honest_on_unreachable=bool(g3), generalizes_2nd_target=bool(g4), restarts=tries,
               note="cartesian-goal layer: damped-LS IK, limit- and collision-aware; local solver with multi-restart",
               validated=bool(ok))
    (ROOT / "reports" / "movement_ik.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("  wrote reports/movement_ik.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
