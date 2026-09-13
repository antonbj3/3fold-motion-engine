#!/usr/bin/env python3
"""Full motion stack behind one hot-swappable contract surface.

One `MotionStack` contract composes the three independently gated layers (IK / Planner / Effort), each
behind a swappable backend Protocol (IKBackend, Planner, EffortBackend). A different physics engine or
solver becomes a new adapter satisfying the same Protocol, with no consumer change.

Selftest checks: (1) every backend satisfies its Protocol (runtime isinstance); (2) a full cartesian-goal
to plan run THROUGH the contract produces a genuine executable plan (IK + coal RRT + effort via the
interface); (3) hot-swap: two different effort backends run through the same plan_cartesian() call;
(4) the planner backend swaps as seamlessly via recommend().

  python -u scripts/motion_stack_contract.py   (requires pinocchio + hppfcl/coal)
"""
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, runtime_checkable

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from planner_contract import PlanRequest, Planner, _build_scene, get_planner
from movement_ik import ik_multistart, fk
from joint_trajectory_planner import quintic, TARGET_RATIO

ROOT = Path(__file__).resolve().parents[1]


# ───────────────────────── KONTRAKT-YTAN (Protocols — konsumenten ser ALDRIG impl) ─────────────────────────
@dataclass
class CartesianRequest:
    """Consumer view: cartesian EE goal + scene. Never exposes pinocchio/RNEA/vamp internals."""
    urdf: str
    q_start: np.ndarray
    target: pin.SE3
    ee_frame: str = "tool0"
    obstacle_box: Optional[list] = None
    box_pose: Optional[list] = None
    pkg: Optional[str] = None


@runtime_checkable
class IKBackend(Protocol):
    name: str
    def solve(self, model, data, fid, target, lo, hi, gm, gd): ...   # → (q, converged, collision_free)


@runtime_checkable
class EffortBackend(Protocol):
    name: str
    def feasible(self, model, data, path) -> dict: ...               # → {executable, metric, value}


# ───────────────────────── ADAPTRAR (varje lager — swappbara) ─────────────────────────
class PinocchioIK:
    name = "pinocchio_damped_ls"
    def solve(self, model, data, fid, target, lo, hi, gm, gd):
        q, conv, free, _ = ik_multistart(model, data, fid, target, lo, hi, np.random.default_rng(0), gm=gm, gd=gd)
        return q, bool(conv), bool(free)


class RneaEffort:
    """Effort backend A: RNEA torque against the URDF effort limits (time-optimal peak ratio < 1)."""
    name = "rnea_torque"
    def feasible(self, model, data, path):
        nj = model.nv
        if not (model.effortLimit[:nj] > 0).all():
            return dict(executable="N/A", metric="rnea_peak_ratio", value=None, note="no effort limit declared")
        peak = 0.0
        for a, b in zip(path[:-1], path[1:]):
            q, dq, ddq = quintic(np.asarray(a), np.asarray(b), 2.0)
            tau = np.array([pin.rnea(model, data, q[i], dq[i], ddq[i]) for i in range(len(q))])
            peak = max(peak, float(np.max(np.max(np.abs(tau), axis=0) / model.effortLimit[:nj])))
        return dict(executable=bool(peak < 1.0), metric="rnea_peak_ratio", value=round(peak, 3))


class VelocityBoundEffort:
    """Effort backend B (different physics, demonstrating hot-swap): time-optimal against the VELOCITY limits
    instead of torque. Another engine is exactly such an adapter: same Protocol, same consumer call."""
    name = "velocity_bound"
    def feasible(self, model, data, path):
        nj = model.nv
        vmax = np.asarray(model.velocityLimit[:nj])
        if not (vmax > 0).all():
            return dict(executable="N/A", metric="vel_ratio", value=None, note="no velocity limit declared")
        peak = 0.0
        for a, b in zip(path[:-1], path[1:]):
            _, dq, _ = quintic(np.asarray(a), np.asarray(b), 2.0)
            peak = max(peak, float(np.max(np.max(np.abs(dq), axis=0) / vmax)))
        return dict(executable=bool(peak < 1.0), metric="vel_ratio", value=round(peak, 3))


class TwinEffortBackend:
    """Effort backend C (learned dynamics): DynamicsTwin — torque = analytic RNEA + a learned residual that captures
    non-linear friction a three-parameter fit misses. Robot-specific (one twin per robot), so N/A when
    twin.dof != model dof. Drop-in EffortBackend with the same Protocol and call as RneaEffort."""
    def __init__(self, twin):
        self.twin = twin; self.name = f"twin_{twin.robot}"
    def feasible(self, model, data, path):
        nj = model.nv
        if self.twin.dof != nj or not (model.effortLimit[:nj] > 0).all():
            return dict(executable="N/A", metric="twin_peak_ratio", value=None, note="twin.dof != model dof, or no effort limit")
        peak = 0.0; peak_cons = 0.0; k = 3.0                          # σ-KONSERVATIV (anti-false-positive: σ var EJ kastad)
        lim = model.effortLimit[:nj]
        for a, b in zip(path[:-1], path[1:]):
            q, dq, ddq = quintic(np.asarray(a), np.asarray(b), 2.0)
            tau_ana = np.array([pin.rnea(model, data, q[i], dq[i], ddq[i]) for i in range(len(q))])
            tau_twin, sigma = self.twin.predict_torque(q, dq, ddq, tau_ana)   # learned residual + calibrated sigma
            peak = max(peak, float(np.max(np.max(np.abs(tau_twin), axis=0) / lim)))
            peak_cons = max(peak_cons, float(np.max(np.max(np.abs(tau_twin) + k * sigma, axis=0) / lim)))
        # additive: value/executable stay point estimates; *_conservative = |tau| + k*sigma flags near-limit cases
        return dict(executable=bool(peak < 1.0), metric="twin_peak_ratio", value=round(peak, 3),
                    value_conservative=round(peak_cons, 3), executable_conservative=bool(peak_cons < 1.0),
                    k_sigma=k, twin=self.twin.robot)


# ───────────────────────── MOTION-STACK (komponerar lagren bakom kontraktet) ─────────────────────────
@dataclass
class MotionStack:
    ik: IKBackend
    planner: Planner
    effort: EffortBackend

    def plan_cartesian(self, req: CartesianRequest) -> dict:
        """Consumer call: cartesian goal -> executable plan. The consumer never sees which backends are used."""
        scene_req = PlanRequest(urdf=req.urdf, q_start=req.q_start, q_goal=req.q_start,
                                obstacle_box=req.obstacle_box, box_pose=req.box_pose, pkg=req.pkg)
        model, data, gm, gd = _build_scene(scene_req)
        fid = model.getFrameId(req.ee_frame); nj = model.nv
        lo, hi = model.lowerPositionLimit[:nj], model.upperPositionLimit[:nj]
        q_goal, conv, free = self.ik.solve(model, data, fid, req.target, lo, hi, gm, gd)
        if not (conv and free):
            return dict(ok=False, stage="ik", backends=self._names())
        path = self.planner.plan(PlanRequest(urdf=req.urdf, q_start=req.q_start, q_goal=q_goal,
                                             obstacle_box=req.obstacle_box, box_pose=req.box_pose, pkg=req.pkg))
        if not path:
            return dict(ok=False, stage="planner", backends=self._names())
        eff = self.effort.feasible(model, data, path)
        return dict(ok=True, waypoints=len(path), effort=eff, backends=self._names(),
                    ik_pos_err_mm=round(float(np.linalg.norm(fk(model, data, fid, q_goal).translation - req.target.translation)) * 1000, 3))

    def _names(self):
        return dict(ik=self.ik.name, planner=self.planner.name, effort=self.effort.name)


def main():
    print("motion-stack-contract — the full stack behind one hot-swappable contract surface")
    urdf = str(ROOT / "assets/robots/ur_description/ur10e.urdf")
    pkg = str(ROOT / "assets/robots/ur_description")   # UR10e collision meshes
    # build a reachable collision-free goal from the scene
    scene = PlanRequest(urdf=urdf, q_start=np.zeros(6), q_goal=np.zeros(6), pkg=pkg, obstacle_box=[0.15, 0.15, 0.4], box_pose=[0.45, 0.0, 0.3])
    model, data, gm, gd = _build_scene(scene)
    from fleet_agnostic_planner import in_collision
    fid = model.getFrameId("tool0"); nj = model.nv
    lo, hi = model.lowerPositionLimit[:nj], model.upperPositionLimit[:nj]; rng = np.random.default_rng(1)
    free = lambda q: not in_collision(model, data, gm, gd, q)
    q_start = next((q for q in (lo + rng.random(nj) * (hi - lo) for _ in range(800)) if free(q)), None)
    q_tgt = next((q for q in (lo + rng.random(nj) * (hi - lo) for _ in range(800)) if free(q)), None)
    target = fk(model, data, fid, q_tgt)
    req = CartesianRequest(urdf=urdf, q_start=q_start, target=target, ee_frame="tool0", pkg=pkg,
                           obstacle_box=[0.15, 0.15, 0.4], box_pose=[0.45, 0.0, 0.3])

    ik, planner = PinocchioIK(), get_planner("pinocchio_coal")
    # (1) Protocol-konformans
    g1 = isinstance(ik, IKBackend) and isinstance(planner, Planner) and isinstance(RneaEffort(), EffortBackend) and isinstance(VelocityBoundEffort(), EffortBackend)
    print(f"  (1) all backends satisfy their Protocols (IK/Planner/Effort, runtime): {g1}")

    # (2) FULL kartesiskt→plan VIA kontraktet (RNEA-effort)
    stack_a = MotionStack(ik=ik, planner=planner, effort=RneaEffort())
    res_a = stack_a.plan_cartesian(req)
    g2 = res_a.get("ok") and res_a["effort"].get("executable") is True
    print(f"  (2) full cartesian->plan via the contract: ok={res_a.get('ok')} ik={res_a.get('ik_pos_err_mm')}mm wp={res_a.get('waypoints')} effort={res_a.get('effort')} -> {g2}")

    # (3) hot-swap the effort backend (RNEA torque -> velocity-limit physics) through the same consumer call
    stack_b = MotionStack(ik=ik, planner=planner, effort=VelocityBoundEffort())
    res_b = stack_b.plan_cartesian(req)
    # genuine hot-swap: both ran through the same plan_cartesian with different effort backends and metrics
    g3 = bool(res_a.get("ok") and res_b.get("ok")
              and res_a["backends"]["effort"] != res_b["backends"]["effort"]
              and res_a["effort"].get("metric") != res_b["effort"].get("metric"))
    print(f"  (3) hot-swap effort backend (same plan_cartesian call): A={res_a['backends']['effort']}({res_a['effort']['metric']}) -> B={res_b['backends']['effort']}({res_b['effort'].get('metric')}); B ok={res_b.get('ok')} -> {g3}")

    # (4) swap the planner backend the same way (recommend() returns a Planner satisfying the contract)
    from planner_contract import recommend
    pl2 = recommend(PlanRequest(urdf=urdf, q_start=q_start, q_goal=q_tgt))
    g4 = isinstance(pl2, Planner)
    print(f"  (4) swap the planner backend via recommend() -> {pl2.name} satisfies the Planner contract: {g4}")

    ok = g1 and g2 and bool(g3) and g4
    print(f"\nVERDICT: motion-stack contract = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"One `MotionStack` contract composes the three independently gated layers (IK / Planner / Effort) "
             f"behind swappable backend Protocols. A genuine cartesian-goal-to-plan run went through the contract "
             f"(UR10e, IK {res_a.get('ik_pos_err_mm')}mm + coal RRT {res_a.get('waypoints')} wp + RNEA effort peak "
             f"{res_a['effort'].get('value')}). Hot-swap shown: the effort layer was switched from RNEA torque to "
             "velocity-limit physics through the same plan_cartesian() call with the consumer code untouched, and "
             "the planner backend swaps the same way via recommend(). " if ok else
             f"Not validated (protocols {g1}, plan via contract {g2}, effort hot-swap {g3}, planner swap {g4}). ")
          + "Scope: this is the contract surface, not new physics; what is validated is Protocol conformance, a real "
          "plan through the interface, and a genuine swap of two different effort physics behind the same call. "
          "VelocityBoundEffort is a demonstration alternative (real velocity feasibility, different physics from RNEA).")

    out = dict(role="full motion-stack hot-swap contract (three gated layers behind one swappable surface)",
               protocols=["IKBackend", "Planner", "EffortBackend"],
               full_plan_via_contract=dict(ok=res_a.get("ok"), ik_pos_err_mm=res_a.get("ik_pos_err_mm"), waypoints=res_a.get("waypoints"), effort=res_a.get("effort")),
               hot_swap_effort=dict(backend_a=res_a["backends"]["effort"], metric_a=res_a["effort"].get("metric"),
                                    backend_b=res_b["backends"].get("effort"), metric_b=res_b["effort"].get("metric"), both_ran=bool(res_a.get("ok") and res_b.get("ok"))),
               planner_swappable=bool(g4),
               note="a different physics engine is a new IKBackend/EffortBackend/Planner adapter with zero consumer change; shown by 2 distinct effort backends via the same plan_cartesian()",
               validated=bool(ok))
    (ROOT / "reports" / "motion_stack_contract.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("  wrote reports/motion_stack_contract.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
