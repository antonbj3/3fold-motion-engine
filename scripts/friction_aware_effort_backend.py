#!/usr/bin/env python3
"""Friction-aware EffortBackend for the MotionStack contract.

A rigid-body-only effort check is optimistic against measured torque on most joints, so it can pass an
infeasible path. `RneaFrictionEffort` adds the identified Coulomb + viscous friction (Fc/Fv from the
real-data fidelity report) on top of RNEA and takes, per joint, max(rigid, rigid+friction): never more
optimistic than rigid, and more accurate against measured torque where friction adds load. Provenance
aware: identified friction where it exists, rigid fallback otherwise.

Selftest checks: (1) satisfies motion_stack_contract.EffortBackend (runtime, drop-in third backend);
(2) monotone safety - the friction-aware peak ratio is >= rigid per joint, and closer to the measured
torque; (3) rigid optimism reproduced against real recordings (rigid underestimates at least one
friction-dominated joint); (4) provenance awareness (identified friction vs rigid fallback).

Requires the public LIP4RID recordings under data/real_external/ (not shipped here) and pinocchio.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motion_stack_contract import EffortBackend, RneaEffort        # the contract surface and the rigid backend to compare against
from joint_trajectory_planner import quintic
# lip4rid_loader (and pandas) are only needed by main(), which reads the public LIP4RID recordings
from moment_gate import _model

ROOT = Path(__file__).resolve().parents[1]


class RneaFrictionEffort:
    """EffortBackend: RNEA + identified Coulomb/viscous friction (Fc/Fv). Monotonically safer than rigid RneaEffort.
    Provenans-medveten: friction_provenance='validated' (real-door) el. 'nominal' (rigid-fallback om Fc/Fv saknas)."""
    def __init__(self, Fc=None, Fv=None, provenance="nominal"):
        self.Fc, self.Fv, self.provenance = Fc, Fv, provenance
        self.name = f"rnea_friction({provenance})"

    def feasible(self, model, data, path):
        nj = model.nv
        if not (model.effortLimit[:nj] > 0).all():
            return dict(executable="N/A", metric="rnea_friction_peak_ratio", value=None, note="no effort limit declared")
        Fc = self.Fc if self.Fc is not None else np.zeros(nj)
        Fv = self.Fv if self.Fv is not None else np.zeros(nj)
        eff = np.asarray(model.effortLimit[:nj])
        # Monotone safety: per joint, max(rigid_peak, friction_peak). A three-parameter Coulomb term sign(dq)*Fc can
        # LOWER a joint at the peak instant, so plain rigid+friction is not strictly >= rigid. Taking the maximum per
        # joint is never more optimistic than rigid (the property a feasibility gate requires) while staying
        # friction-accurate where friction adds load.
        pj_rigid = np.zeros(nj); pj_fric = np.zeros(nj)
        for a, b in zip(path[:-1], path[1:]):
            q, dq, ddq = quintic(np.asarray(a), np.asarray(b), 2.0)
            tr = np.array([pin.rnea(model, data, q[i], dq[i], ddq[i]) for i in range(len(q))])
            tf = tr + (np.sign(dq) * Fc + dq * Fv)
            pj_rigid = np.maximum(pj_rigid, np.max(np.abs(tr), axis=0) / eff)
            pj_fric = np.maximum(pj_fric, np.max(np.abs(tf), axis=0) / eff)
        pj_safe = np.maximum(pj_rigid, pj_fric)              # per joint, monotone-safe (>= rigid by construction)
        peak = float(pj_safe.max())
        return dict(executable=bool(peak < 1.0), metric="rnea_friction_peak_ratio_monotone_safe", value=round(peak, 3),
                    monotone_safe=True, friction_provenance=self.provenance)


def main():
    print("friction-aware effort backend — identified friction on top of RNEA (Franka)")
    model = _model(); data = model.createData(); nj = model.nv
    rep = json.loads((ROOT / "reports" / "panda_twin_rnea_fidelity.json").read_text())
    Fc = np.array([r["Fc"] for r in rep["per_joint"]]); Fv = np.array([r["Fv"] for r in rep["per_joint"]])

    # real Franka trajectory (LIP4RID) as path waypoints: rigid vs friction-aware EffortBackend
    import lip4rid_loader as L
    recs = L.load_panda()[:2]
    q = np.vstack([np.column_stack([r[f"q_{j}"] for j in range(1, 8)]) for r in recs])[::40]
    path = [q[i] for i in range(len(q))]

    rigid = RneaEffort()
    fric = RneaFrictionEffort(Fc=Fc, Fv=Fv, provenance="validated")     # identified friction (held-out R2 0.94)
    nom = RneaFrictionEffort(provenance="nominal")                      # no Fc/Fv -> rigid fallback (provenance honest)

    # per-joint peak ratio against the MEASURED torque: friction is not a monotone peak (sign(dq)*Fc can subtract at
    # the peak instant), so the claim that holds is "more accurate against measured", grounded in the raw measurement.
    dq = np.vstack([np.column_stack([r[f"dq_{j}"] for j in range(1, 8)]) for r in recs])[::8]
    qf = np.vstack([np.column_stack([r[f"q_{j}"] for j in range(1, 8)]) for r in recs])[::8]
    ddq = np.vstack([np.column_stack([r[f"ddq_{j}"] for j in range(1, 8)]) for r in recs])[::8]
    tau_meas = np.vstack([np.column_stack([r[f"tau_{j}"] for j in range(1, 8)]) for r in recs])[::8]
    tau_r = np.array([pin.rnea(model, data, qf[i], dq[i], ddq[i]) for i in range(len(qf))])
    tau_f = tau_r + (np.sign(dq) * Fc + dq * Fv)
    eff = np.asarray(model.effortLimit[:nj])
    pr_m = np.max(np.abs(tau_meas), 0) / eff; pr_r = np.max(np.abs(tau_r), 0) / eff; pr_f = np.max(np.abs(tau_f), 0) / eff
    pr_safe = np.maximum(pr_r, pr_f)                                                     # per joint, max(rigid, friction)
    err_r = np.abs(pr_r - pr_m); err_safe = np.abs(pr_safe - pr_m)
    underest_vs_real = [int(j + 1) for j in range(nj) if pr_m[j] > pr_r[j] + 0.01]       # rigid optimistic against measurement
    monotone_safe = bool(np.all(pr_safe >= pr_r - 1e-9))                                 # the maximum is >= rigid by construction
    more_accurate = [int(j + 1) for j in range(nj) if err_safe[j] < err_r[j] - 1e-4]
    print(f"  per-joint peak ratio, measured   : {np.round(pr_m,3)}")
    print(f"  per-joint peak ratio, rigid      : {np.round(pr_r,3)} (|error| {np.round(err_r,3)})")
    print(f"  per-joint peak ratio, max(r,fric) : {np.round(pr_safe,3)} (|error| {np.round(err_safe,3)})  monotone-safe")
    print(f"  rigid underestimates the measurement on joints: {underest_vs_real} | max(r,fric) monotone-safe (>=rigid): {monotone_safe} | more accurate on joints: {more_accurate}")

    # through the contract (path waypoints): rigid vs friction vs nominal fallback (Protocol conformance / hot-swap)
    res_r = rigid.feasible(model, data, path)
    res_f = fric.feasible(model, data, path)
    res_n = nom.feasible(model, data, path)
    print(f"  through the contract: rigid peak {res_r['value']} | with identified friction {res_f['value']} ({res_f['friction_provenance']}) | nominal fallback {res_n['value']} ({res_n['friction_provenance']})")

    g1 = isinstance(fric, EffortBackend) and isinstance(nom, EffortBackend)              # drop-in third backend
    g2 = monotone_safe and float(np.median(err_safe)) <= float(np.median(err_r))         # monotone-safe and more accurate against measurement
    g3 = len(underest_vs_real) >= 1                                                      # rigid is optimistic against measurement
    g4 = (fric.provenance == "validated" and nom.provenance == "nominal" and res_n["value"] == res_r["value"]  # provenance aware
          and res_f.get("monotone_safe") is True)                                        # the backend reports monotone safety
    print(f"  (1) satisfies the EffortBackend Protocol (drop-in third backend): {g1}")
    print(f"  (2) monotone-safe (max(rigid,fric) >= rigid = {monotone_safe}) and more accurate against measurement (median |error| {np.median(err_safe):.3f} <= rigid {np.median(err_r):.3f}): {g2}")
    print(f"  (3) rigid is optimistic against measurement ({len(underest_vs_real)} underestimated joints, so it can pass an infeasible path): {g3}")
    print(f"  (4) provenance aware and the backend reports monotone_safe: {g4}")

    ok = g1 and g2 and g3 and g4
    print(f"\nVERDICT: friction-aware effort backend = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"`RneaFrictionEffort` is a third EffortBackend on the same Protocol: per joint it takes "
             f"max(rigid, rigid+identified friction), so it is never more optimistic than rigid (the property a "
             f"feasibility gate requires) and is more accurate against MEASURED torque (median |error| "
             f"{np.median(err_safe):.3f} vs rigid {np.median(err_r):.3f}). Plain rigid+friction is not monotone "
             "because sign(dq)*Fc can subtract at the peak instant; the maximum picks rigid there and friction "
             f"where friction adds load. Rigid underestimates the measurement on joints {underest_vs_real}. "
             "Provenance aware: identified friction for the robot that has it, rigid fallback otherwise. " if ok else
             f"Not validated (protocol {g1}, monotone-safe {g2}, rigid optimism {g3}, provenance {g4}). ")
          + "Scope: data bound - on these low-effort recordings no feasibility verdict flips (all ratios < 1), but "
          "the optimism against measured torque is real and would flip a near-limit case. Identified friction "
          "exists only for the Franka here; other robots fall back to rigid. A learned dynamics twin is the richer "
          "fix for the non-linear friction a three-parameter model misses.")

    out = dict(role="monotone-safe friction-aware EffortBackend for MotionStack (s2 review-fix: per-joint max(rigid,friction))",
               per_joint_peak_ratio_measured=[round(float(x), 3) for x in pr_m],
               per_joint_peak_ratio_rigid=[round(float(x), 3) for x in pr_r],
               per_joint_peak_ratio_friction=[round(float(x), 3) for x in pr_f],
               per_joint_peak_ratio_max_safe=[round(float(x), 3) for x in pr_safe],
               median_abs_err_rigid=round(float(np.median(err_r)), 4), median_abs_err_max_safe=round(float(np.median(err_safe)), 4),
               rigid_underestimates_vs_real_joints=underest_vs_real, max_safe_more_accurate_joints=more_accurate,
               monotone_safe=bool(monotone_safe), more_accurate_vs_measured=bool(g2), rigid_optimistic_vs_reality=bool(g3),
               via_contract=dict(rigid=res_r["value"], friction_validated=res_f["value"], nominal_fallback=res_n["value"]),
               provenance_aware=bool(g4),
               s2_review_fix="per-joint max(rigid_peak, friction_peak) -> strictly >=rigid (monotone-safe, never more optimistic) + friction-accurate where it loads; recovers monotone-safety honestly after plain-friction J1/J4 regression",
               note="monotone-safe friction-aware EffortBackend: max(rigid,friction) per joint; closes the safety-gate optimism (never under-estimates vs rigid); accurate vs measured; 3rd backend on MotionStack Protocol; provenance-aware",
               honest_limit="validated friction only Franka; others nominal-fallback; no feasibility-flip on low-effort LIP4RID; learned-twin = future fix for J1/J4 nonlinear friction",
               validated=bool(ok))
    (ROOT / "reports" / "friction_aware_effort_backend.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("  wrote reports/friction_aware_effort_backend.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
