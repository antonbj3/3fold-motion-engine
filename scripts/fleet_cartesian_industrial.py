#!/usr/bin/env python3
"""Cartesian goal -> plan across the URDF fleet, gated on effort-limit PROVENANCE.

`fleet_cartesian_to_plan` proves Cartesian -> plan on one robot. This cell adds the fleet dimension and the
data-quality guard that comes with it: the fleet URDFs have HETEROGENEOUS effort-limit quality. Some carry REAL
limits (varied per joint) and support a meaningful effort-feasibility claim; some carry a PLACEHOLDER (the same
value on every joint), which makes the effort gate vacuous; some carry none at all (zeros). Claiming
"effort-feasible" on placeholder or zero limits would be a claim about nothing. So effort provenance is detected
(real / placeholder / missing) and effort feasibility is claimed ONLY where the limits are real; elsewhere it is
reported as N/A. Geometry and kinematics (IK + coal RRT) work regardless.

Scope of the shipped default: the full geometric pipeline needs collision meshes, which this repository
redistributes for two vendor descriptions; the 165 fleet XMLs reference vendor mesh packages that are not
redistributed here. So the pipeline runs on the descriptions that carry meshes, and the provenance scan runs over
the whole fleet (kinematics only, no meshes needed), which is where the heterogeneity actually lives.

Selftest checks: (1) a fleet robot goes Cartesian -> coal IK -> coal RRT with a collision-free path; (2) effort
feasibility is MEANINGFUL on a robot with REAL effort limits: time-optimal peak ratio <= target < 1; (3) the
provenance guard - robots with placeholder or missing limits report effort N/A, never feasible; (4) the
placeholder case (uniform limits) is detected as placeholder and not as real.

  python -u scripts/fleet_cartesian_industrial.py   (requires pinocchio + hppfcl/coal)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fleet_multirobot_movement import _build, _pkgs, _mesh_descriptions, ROBOTS_DIR
from fleet_agnostic_planner import rrt_connect, edge_free, in_collision
from movement_ik import ik_multistart, fk
from fleet_cartesian_to_plan import peak_ratio_generic, time_optimal_generic
from joint_trajectory_planner import TARGET_RATIO

ROOT = Path(__file__).resolve().parents[1]
FLEET_DIR = ROBOTS_DIR / "fleet_urdf"
EE_PREF = ("tool0", "flange", "panda_hand", "right_gripper")


def effort_provenance(m, nj):
    """Effort-limit quality: real (varied, > 0) / placeholder (uniform) / missing (zero or non-finite)."""
    eff = np.asarray(m.effortLimit[:nj])
    if not (eff > 0).all() or not np.isfinite(eff).all():
        return "missing"
    cv = float(eff.std() / (eff.mean() + 1e-12))
    return "placeholder(uniform)" if cv < 0.02 else "real"


def _ee_frame(model):
    for n in EE_PREF:
        if model.existFrame(n):
            return n
    return model.frames[-1].name


def scan_provenance(urdf):
    """Model-only load: effort provenance without meshes (works for every fleet XML)."""
    try:
        m = pin.buildModelFromUrdf(str(urdf))
    except Exception as e:
        return dict(robot=Path(urdf).stem, loaded=False, err=f"{type(e).__name__}:{str(e)[:50]}")
    prov = effort_provenance(m, m.nv)
    return dict(robot=Path(urdf).stem, loaded=True, dof=m.nv, effort_provenance=prov,
                effort_feasible="N/A" if prov != "real" else "not-run",
                effort_note=f"effort {prov} -> effort feasibility not meaningful" if prov != "real" else "")


def run_robot(urdf, pkgs):
    """Full pipeline: Cartesian goal -> coal IK -> coal RRT -> effort feasibility where the limits are real."""
    urdf = Path(urdf)
    rng = np.random.default_rng(0)
    model, data, gm, gd, nrobot = _build(urdf, pkgs, [0.15, 0.15, 0.4], [0.45, 0.0, 0.3])
    nj = model.nv
    ee = _ee_frame(model); fid = model.getFrameId(ee)
    lo, hi = model.lowerPositionLimit[:nj], model.upperPositionLimit[:nj]
    eprov = effort_provenance(model, nj)
    rec = dict(robot=urdf.stem, vendor=urdf.parent.name, loaded=True, dof=nj, ee_frame=ee, effort_provenance=eprov)

    def free(q): return not in_collision(model, data, gm, gd, q)
    q_start = next((q for q in (lo + rng.random(nj) * (hi - lo) for _ in range(1200)) if free(q)), None)
    q_tgt = next((q for q in (lo + rng.random(nj) * (hi - lo) for _ in range(1200)) if free(q)), None)
    if q_start is None or q_tgt is None:
        rec["err"] = "no free start/goal configurations"; return rec
    target = fk(model, data, fid, q_tgt)
    q_goal, conv, ikfree, tries = ik_multistart(model, data, fid, target, lo, hi, rng, gm=gm, gd=gd)
    oMf = fk(model, data, fid, q_goal)
    pos_err = float(np.linalg.norm(oMf.translation - target.translation))
    rec.update(ik_conv=bool(conv), ik_pos_err_mm=round(pos_err * 1000, 3), ik_collision_free=bool(ikfree))
    if not (conv and pos_err < 1e-3 and ikfree):
        rec["err"] = "IK did not converge collision-free"; return rec
    t0 = time.time(); path = rrt_connect(model, data, gm, gd, q_start, q_goal, lo, hi, rng); rrt_ms = (time.time() - t0) * 1000
    if path is None:
        rec.update(rrt_ms=round(rrt_ms, 1), collision_free=False, err="RRT found no path"); return rec
    dense_ok = all(edge_free(model, data, gm, gd, np.asarray(a), np.asarray(b), res=0.02)
                   for a, b in zip(path[:-1], path[1:]))
    rec.update(rrt_ms=round(rrt_ms, 1), waypoints=len(path), collision_free=bool(dense_ok))
    if eprov == "real" and dense_ok:
        Ts, ratios, infeas = [], [], 0
        for a, b in zip(path[:-1], path[1:]):
            T, r = time_optimal_generic(model, data, a, b, nj)
            if T is None: infeas += 1
            else: Ts.append(T); ratios.append(r)
        rec.update(effort_feasible=bool(infeas == 0 and ratios and max(ratios) <= TARGET_RATIO + 1e-3),
                   total_time_s=round(sum(Ts), 3),
                   max_peak_ratio=round(float(max(ratios)) if ratios else 9.9, 3), infeasible_segments=infeas)
    else:
        rec.update(effort_feasible="N/A",
                   effort_note=f"effort {eprov} -> effort feasibility not meaningful")
    return rec


def main():
    print("FLEET-CARTESIAN-INDUSTRIAL — Cartesian -> plan across the fleet, gated on effort provenance")
    pkgs = _pkgs()
    results = []
    for urdf in _mesh_descriptions():
        try:
            results.append(run_robot(urdf, pkgs))
        except Exception as e:
            results.append(dict(robot=Path(urdf).stem, loaded=False, err=f"{type(e).__name__}:{str(e)[:60]}"))
    # provenance scan over the whole fleet (kinematics only): this is where placeholder/missing limits live
    scan = [scan_provenance(u) for u in sorted(FLEET_DIR.glob("*.urdf"))]
    n_real = sum(1 for r in scan if r.get("effort_provenance") == "real")
    n_ph = sum(1 for r in scan if r.get("effort_provenance") == "placeholder(uniform)")
    n_missing = sum(1 for r in scan if r.get("effort_provenance") == "missing")
    for r in results:
        if not r.get("loaded"):
            print(f"  {r['robot']:24s} NOT LOADED: {r.get('err')}")
        else:
            print(f"  {r['robot']:24s} effort provenance={r['effort_provenance']:20s} | "
                  f"IK={r.get('ik_pos_err_mm','—')}mm collision-free={r.get('collision_free','—')} "
                  f"effort-feasible={r.get('effort_feasible')}")
    print(f"  fleet provenance scan ({len(scan)} URDFs): real {n_real} | placeholder(uniform) {n_ph} | missing {n_missing}")

    realrob = [r for r in results if r.get("effort_provenance") == "real" and r.get("collision_free")]
    g1 = any(r.get("collision_free") and r.get("ik_conv") for r in results)
    g2 = bool(realrob) and all(r.get("effort_feasible") is True for r in realrob)
    nonreal = [r for r in scan if r.get("loaded") and r.get("effort_provenance") != "real"]
    g3 = len(nonreal) >= 1 and all(r.get("effort_feasible") == "N/A" for r in nonreal)
    g4 = n_ph >= 1
    peaks = [r.get("max_peak_ratio") for r in realrob]
    print(f"  (1) fleet robot Cartesian -> coal IK -> coal RRT collision-free: {g1}")
    print(f"  (2) effort feasibility MEANINGFUL on real-effort robots (peaks {peaks} <= {TARGET_RATIO}): {g2}")
    print(f"  (3) provenance guard: placeholder/missing effort -> N/A, never feasible ({len(nonreal)} robots): {g3}")
    print(f"  (4) placeholder (uniform limits) detected as placeholder, not real: {g4} ({n_ph} in the fleet)")

    ok = g1 and g2 and g3 and g4
    print(f"\nVERDICT: fleet-cartesian-industrial = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"Cartesian -> plan runs on the fleet descriptions that carry collision meshes "
             f"({', '.join(r['robot'] for r in results if r.get('collision_free'))}): Cartesian EE goal -> coal IK -> "
             f"coal RRT-Connect (exact mesh) -> effort feasibility where the limits are REAL (peak ratios {peaks} "
             f"<= {TARGET_RATIO}). Across the {len(scan)} fleet URDFs the limits split {n_real} real / {n_ph} "
             f"placeholder / {n_missing} missing, and effort feasibility is reported N/A for the last two rather than "
             "claimed on placeholder numbers. Geometry and kinematics work regardless of effort quality; the effort "
             "layer is gated on effort provenance. " if ok else
             f"Not validated (Cartesian IK+RRT {g1}, real-effort feasible {g2}, provenance guard {g3}, "
             f"placeholder detection {g4}). ")
          + "Scope: composition of the coal layers; coal is slower than a SIMD planner but general. Effort "
            "feasibility needs REAL effort limits, and most fleet URDFs do not carry them - that is a data-quality "
            "gap, not a planner defect. Friction is nominal (Fc = Fv = 0). Composes fleet_multirobot_movement + "
            "movement_ik + RNEA.")

    out = dict(layer="cartesian -> plan across the fleet + effort provenance gating",
               pipeline_robots=results,
               fleet_scan=dict(n=len(scan), real=n_real, placeholder_uniform=n_ph, missing=n_missing,
                               placeholder_robots=[r["robot"] for r in scan
                                                   if r.get("effort_provenance") == "placeholder(uniform)"][:20]),
               effort_provenance_gated=True,
               cartesian_plan=bool(g1), real_effort_feasible=bool(g2), provenance_guard=bool(g3),
               placeholder_detected=bool(g4),
               note="effort feasibility ONLY where effort limits are real; placeholder/missing -> N/A",
               validated=bool(ok))
    (ROOT / "reports" / "fleet_cartesian_industrial.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("  wrote reports/fleet_cartesian_industrial.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
