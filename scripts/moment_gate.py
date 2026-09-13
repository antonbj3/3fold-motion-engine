#!/usr/bin/env python3
"""Torque feasibility gate for planned trajectories.

Given a planned trajectory (q, dq, ddq), compute the required joint torque with the real-data-validated
Panda twin (RNEA + identified Coulomb/viscous friction) and compare it against the URDF effort limits
with a margin; touching the threshold is never reported as feasible.

Pre-registered: margin = 0.9 (at or above 90% of the limit is MARGINAL, not feasible); FEASIBLE only when
every joint stays below margin * limit.

Selftest checks (require the public LIP4RID recordings under data/real_external/, not shipped here):
(1) real recorded trajectories, which the robot did execute, are reported FEASIBLE (no false infeasibility);
(2) a 3x more aggressive version (scaled dq/ddq) is reported INFEASIBLE; (3) the gate's torque estimate
matches the measured torque per joint; (4) a trajectory scaled to exactly the limit is flagged
MARGINAL/INFEASIBLE, never feasible.

  python scripts/moment_gate.py   (requires pinocchio)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parent))
# lip4rid_loader (and pandas) are only needed by main(), which reads the public LIP4RID recordings

ROOT = Path(__file__).resolve().parents[1]
URDF = str(ROOT / "assets/robots/franka_description/franka_panda.urdf")
NJ = 7
MARGIN = 0.9


def _model():
    full = pin.buildModelFromUrdf(URDF)
    locked = [full.getJointId(f"panda_finger_joint{i}") for i in (1, 2)]
    return pin.buildReducedModel(full, locked, pin.neutral(full))


def required_torque(model, data, q, dq, ddq, Fc, Fv):
    """Required torque per time step = rigid-body RNEA + Coulomb/viscous friction (validated twin)."""
    tau = np.array([pin.rnea(model, data, q[i], dq[i], ddq[i]) for i in range(len(q))])
    tau += Fc * np.sign(dq) + Fv * dq
    return tau


def moment_gate(model, tau_req, margin=MARGIN):
    """Feasibility verdict against the effort limits with a margin. Returns a per-joint and aggregate dict."""
    lim = model.effortLimit[:NJ]
    peak = np.max(np.abs(tau_req), axis=0)
    ratio = peak / lim
    rmax = float(np.max(ratio))
    if rmax > 1.0:
        verdict = "INFEASIBLE"
    elif rmax >= margin:
        verdict = "MARGINAL"
    else:
        verdict = "FEASIBLE"
    return dict(verdict=verdict, max_ratio=round(rmax, 3), per_joint_ratio=[round(float(x), 3) for x in ratio],
                worst_joint=int(np.argmax(ratio)) + 1)


def main():
    model = _model(); data = model.createData()
    print(f"moment-gate — torque feasibility via the validated Panda twin (RNEA + friction); margin={MARGIN}")
    print(f"  effort limits (Nm): {np.round(model.effortLimit[:NJ],0)}")
    rep = json.loads((ROOT / "reports" / "panda_twin_rnea_fidelity.json").read_text())
    Fc = np.array([r["Fc"] for r in rep["per_joint"]]); Fv = np.array([r["Fv"] for r in rep["per_joint"]])
    import lip4rid_loader as L
    recs = L.load_panda()

    # (1) real trajectories -> FEASIBLE; (3) the gate torque vs the measured torque
    feas_ok = 0; ratio_err = []
    for r in recs[:8]:
        q = np.column_stack([r[f"q_{j}"] for j in range(1, 8)])[::3]
        dq = np.column_stack([r[f"dq_{j}"] for j in range(1, 8)])[::3]
        ddq = np.column_stack([r[f"ddq_{j}"] for j in range(1, 8)])[::3]
        tau_meas = np.column_stack([r[f"tau_{j}"] for j in range(1, 8)])[::3]
        tau_req = required_torque(model, data, q, dq, ddq, Fc, Fv)
        gv = moment_gate(model, tau_req)
        feas_ok += gv["verdict"] == "FEASIBLE"
        # gate peak ratio vs measured peak ratio (validates the torque estimate)
        meas_ratio = np.max(np.abs(tau_meas), axis=0) / model.effortLimit[:NJ]
        ratio_err.append(np.abs(np.array(gv["per_joint_ratio"]) - meas_ratio))
    ratio_err = np.concatenate(ratio_err)
    g1 = feas_ok == 8
    # per-unit, not only the median: a median over 8 trajectories x 7 joints hides one joint/trajectory with a large
    # torque-estimate error, so a tail floor (90th percentile <= 0.12) catches a tail of badly estimated units.
    p90_err = float(np.percentile(ratio_err, 90))
    g3 = float(np.median(ratio_err)) < 0.05 and p90_err < 0.12
    print(f"  (1) {feas_ok}/8 real trajectories -> FEASIBLE (no false infeasibility)")
    print(f"  (3) gate torque ratio vs measured: median error {np.median(ratio_err):.3f}, p90 {p90_err:.3f}")

    # (2) 3×-aggressiv → INFEASIBLE
    r = recs[0]
    q = np.column_stack([r[f"q_{j}"] for j in range(1, 8)])[::3]
    dq = np.column_stack([r[f"dq_{j}"] for j in range(1, 8)])[::3]
    ddq = np.column_stack([r[f"ddq_{j}"] for j in range(1, 8)])[::3]
    g_agg = moment_gate(model, required_torque(model, data, q, 3 * dq, 9 * ddq, Fc, Fv))
    g2 = g_agg["verdict"] == "INFEASIBLE"
    print(f"  (2) 3x aggressive trajectory -> {g_agg['verdict']} (max ratio {g_agg['max_ratio']}, worst joint J{g_agg['worst_joint']})")

    # (4) margin discipline: test the verdict logic at the margin boundary by scaling tau_req to exact peak ratios
    # (a direct unit test of the classification)
    base = required_torque(model, data, q, dq, ddq, Fc, Fv)
    cur = np.max(np.abs(base), axis=0) / model.effortLimit[:NJ]; cur_max = float(np.max(cur))
    def at_ratio(target):
        return moment_gate(model, base * (target / cur_max))   # skala HELA tau_req → peak-ratio = target
    v_feas = at_ratio(0.85)["verdict"]; v_marg = at_ratio(0.95)["verdict"]; v_inf = at_ratio(1.05)["verdict"]
    g4 = v_feas == "FEASIBLE" and v_marg == "MARGINAL" and v_inf == "INFEASIBLE"
    print(f"  (4) margin discipline (verdict logic at the boundary): peak ratio 0.85 -> {v_feas}, 0.95 -> {v_marg}, 1.05 -> {v_inf}; touching the threshold is MARGINAL, not feasible")

    ok = g1 and g2 and g3 and g4
    print(f"\nVERDICT: moment-gate = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"Required joint torque from the real-data-validated Panda twin (RNEA + friction) compared against the "
             f"URDF effort limits with a margin. Real executed trajectories pass as FEASIBLE ({feas_ok}/8, no false "
             f"infeasibility); the gate's torque estimate matches the measured torque (median ratio error "
             f"{np.median(ratio_err):.3f}); a 3x aggressive trajectory is caught as INFEASIBLE; and the margin "
             "discipline holds, so a trajectory at the limit is not reported feasible. " if ok else
             f"Not validated (real-feasible {g1}, aggressive caught {g2}, torque match {g3}, margin {g4}). ")
          + "Scope: Panda-specific (its URDF and validated twin); static effort limits (no speed-dependent torque "
          "curve); friction is a simple Coulomb + viscous model.")

    out = dict(robot="Franka_Panda_7DOF", margin=MARGIN, effort_limits=list(np.round(model.effortLimit[:NJ], 1)),
               real_traj_feasible=f"{feas_ok}/8", moment_estimate_median_ratio_err=round(float(np.median(ratio_err)), 4),
               aggressive_3x=g_agg, margin_discipline={"r0.85": v_feas, "r0.95": v_marg, "r1.05": v_inf}, validated=bool(ok),
               composes="panda_twin_rnea_fidelity (R²0.94 vs real), lip4rid_loader")
    (ROOT / "reports" / "moment_gate.json").write_text(json.dumps(out, indent=1))
    print("  wrote reports/moment_gate.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
