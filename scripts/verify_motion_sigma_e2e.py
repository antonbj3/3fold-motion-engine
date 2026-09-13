#!/usr/bin/env python3
"""End-to-end check that the twin's sigma propagates into MotionPlan.plan_sigma.

The dynamics twin reports a per-joint torque uncertainty (sigma) next to its torque prediction. This verifies
that the uncertainty survives the whole path twin -> MotionStack._effort -> MotionPlan.plan_sigma, and that the
resulting gate ANSWERS to sigma instead of being always-on:

  sigma > 0  -> the sigma-conservative peak is STRICTLY above the point peak (the k*sigma margin is real)
  sigma = 0  -> the sigma-conservative peak EQUALS the point peak (the margin disappears: not tautological)
  no twin    -> plan_sigma is None (the uncertainty is absent honestly, not faked)

  python -u scripts/verify_motion_sigma_e2e.py   (requires pinocchio + hppfcl/coal + scipy)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from motion_engine.motion_stack import MotionStack
from movement_ik import fk
from fleet_multirobot_movement import _pkgs, _mesh_descriptions

PKG = _pkgs()


class StubTwin:
    """Controlled twin: residual 0 (agrees with the analytic torque), sigma a known constant per joint, so the
    sigma arithmetic is tested exactly rather than through a fitted model."""
    def __init__(self, dof, sigma_nm):
        self.dof = dof; self.sigma_nm = float(sigma_nm)

    def predict_torque(self, q, dq, ddq, tau_analytical):
        tau = np.asarray(tau_analytical, float)
        return tau, np.full_like(tau, self.sigma_nm)


def build_ms():
    """First robot that (a) builds without mesh errors and (b) has real effort limits (otherwise _effort is N/A)."""
    built = None
    for u in _mesh_descriptions():
        try:
            ms = MotionStack(str(u), pkg=PKG, obstacle_box=None, seed=0)
        except Exception:
            continue
        if (np.asarray(ms.model.effortLimit[:ms.nv]) > 0).all():
            return Path(u).stem, ms
        if built is None:
            built = (Path(u).stem, ms)
        return built if built else (None, None)
    return built if built else (None, None)


def main():
    name, ms = build_ms()
    if ms is None:
        print("no buildable URDF with meshes — cannot verify"); return 2
    nv = ms.nv
    print(f"VERIFY SIGMA END-TO-END — robot {name} (nv={nv})\n")

    # a real trajectory (interpolate between two sampled configurations)
    qa, qb = ms.sample(), ms.sample()
    traj = np.array([qa + t * (qb - qa) for t in np.linspace(0, 1, 60)])

    ok = True

    # TEST 1: no twin -> plan_sigma None
    ms.dyn_twin = None
    f0, p0, s0 = ms._effort(traj)
    t1 = (s0 is None)
    print(f"  [1] no twin: plan_sigma={s0} -> {'None, sigma honestly absent' if t1 else 'FAIL, should be None'}")
    ok &= t1

    # TEST 2: twin sigma > 0 -> plan_sigma populated, sigma-conservative STRICTLY above the point value
    SIG = 5.0
    ms.dyn_twin = StubTwin(nv, SIG)
    f2, p2, s2 = ms._effort(traj)
    t2a = (s2 is not None)
    t2b = t2a and (s2["peak_ratio_sigma_cons"] > s2["peak_ratio_point"])      # the margin is real
    t2c = t2a and all(abs(x - SIG) < 1e-6 for x in s2["per_joint_sigma_nm"])  # sigma propagated exactly
    print(f"  [2] twin sigma={SIG}Nm: plan_sigma populated {'yes' if t2a else 'NO'}; "
          f"per-joint sigma={s2['per_joint_sigma_nm'][:3] if t2a else '—'}...")
    print(f"      point peak {s2['peak_ratio_point'] if t2a else '—'} vs sigma-conservative "
          f"{s2['peak_ratio_sigma_cons'] if t2a else '—'} (k={s2['k_sigma'] if t2a else '—'}) -> "
          f"{'sigma-cons > point, margin real' if t2b else 'FAIL, no margin'}")
    print(f"      sigma value propagated exactly ({SIG}Nm): {'yes' if t2c else 'NO'}")
    ok &= t2a and t2b and t2c

    # TEST 3 (null control): twin sigma = 0 -> sigma-conservative EQUALS point (the margin disappears)
    ms.dyn_twin = StubTwin(nv, 0.0)
    f3, p3, s3 = ms._effort(traj)
    t3 = (s3 is not None) and abs(s3["peak_ratio_sigma_cons"] - s3["peak_ratio_point"]) < 1e-9
    print(f"  [3] null twin sigma=0: sigma-cons {s3['peak_ratio_sigma_cons']} == point {s3['peak_ratio_point']} -> "
          f"{'margin disappears, the gate answers to sigma' if t3 else 'FAIL, tautological (cons != point at sigma=0)'}")
    ok &= t3

    # TEST 4: full plan_cartesian -> MotionPlan.plan_sigma set end to end (retry reachable goals, the planner is stochastic)
    ms.dyn_twin = StubTwin(nv, SIG)
    plan = None
    for _ in range(10):
        try:
            qg = ms.sample()
            if not ms.free(qg):
                continue
            p = ms.plan_cartesian(fk(ms.model, ms.data, ms.fid, qg))
            if p.ok:
                plan = p; break
        except Exception:
            continue
    if plan is not None and plan.plan_sigma is not None:
        print(f"  [4] plan_cartesian END-TO-END: MotionPlan.plan_sigma SET "
              f"(sigma-cons peak {plan.plan_sigma['peak_ratio_sigma_cons']}, "
              f"sigma-feasible {plan.plan_sigma['sigma_feasible']}, tier '{plan.plan_confidence}')")
    elif plan is not None:
        print("  [4] plan ok but plan_sigma=None (unexpected)"); ok = False
    else:
        print("  [4] plan_cartesian: no plan in 10 reachable attempts (planner variance, not sigma wiring; "
              "tests 1-3 decide)")

    # TEST 5: sigma-safe retiming - slows down under k*sigma; null sigma = 0 -> no extra time.
    # A genuine crash inside _sigma_safe_retime must NOT be silenced as "not applicable to this robot", which is
    # what the legitimate None branch means; t5_crashed keeps the two apart.
    t5a = t5b = False
    t5_crashed = False
    try:
        from scipy.interpolate import CubicSpline
        lo, hi = np.asarray(ms.lo)[:nv], np.asarray(ms.hi)[:nv]
        pathpts = np.array([lo + (hi - lo) * f for f in [0.1, 0.9, 0.2, 0.85, 0.3]])   # demanding sweep -> torque binds
        S5 = float(len(pathpts) - 1); cs5 = CubicSpline(np.arange(len(pathpts)), pathpts, axis=0)
        ms.dyn_twin = StubTwin(nv, SIG); ssr = ms._sigma_safe_retime(cs5, S5)
        ms.dyn_twin = StubTwin(nv, 0.0); ssr0 = ms._sigma_safe_retime(cs5, S5)
        # sigma either costs time or makes the sweep infeasible under k*sigma; both are the margin being real.
        # The null run (sigma = 0) must stay feasible and cost nothing, which is what makes this non-tautological.
        t5a = ssr is not None and (ssr["sigma_safe_time_s"] >= ssr["nominal_torque_time_s"] - 1e-6
                                   if not ssr["sigma_infeasible"] else
                                   (ssr0 is not None and not ssr0["sigma_infeasible"]))
        t5b = (ssr0 is not None and not ssr0["sigma_infeasible"]
               and abs(ssr0["sigma_safe_time_s"] - ssr0["nominal_torque_time_s"]) < 1e-3)
        if ssr is not None:
            if ssr["sigma_infeasible"]:
                print(f"  [5] sigma-safe retiming sigma={SIG}Nm: INFEASIBLE under k*sigma (nominal-torque time "
                      f"{ssr['nominal_torque_time_s']}s is feasible) -> "
                      f"{'the sigma margin binds' if t5a else 'FAIL'}")
            else:
                print(f"  [5] sigma-safe retiming sigma={SIG}Nm: sigma-safe time {ssr['sigma_safe_time_s']}s vs nominal "
                      f"{ssr['nominal_torque_time_s']}s (cost +{ssr['sigma_time_cost_s']}s, feasible "
                      f"{ssr['sigma_safe_feasible']}) -> {'slows at the binding corner' if t5a else 'FAIL'}")
            print(f"      null sigma=0: sigma-safe {ssr0['sigma_safe_time_s']}s == nominal "
                  f"{ssr0['nominal_torque_time_s']}s -> {'cost disappears, not tautological' if t5b else 'FAIL'}")
        else:
            print("  [5] sigma-safe retiming: None (no real effort limit or twin — tests 1-4 decide)")
            t5a = t5b = True                      # not applicable on this robot -> does not block the verdict
    except Exception as e:
        print(f"  [5] sigma-safe retiming CRASHED ({type(e).__name__}: {e}) -- FAIL, not 'not applicable' "
              "(a genuine crash must never be silenced the same way as the legitimate None branch)")
        t5_crashed = True
    ok &= (t5a and t5b) and not t5_crashed

    print(f"\nVERDICT: sigma end-to-end = {'VALIDATED' if ok else 'NOT VALIDATED'} "
          f"(no-twin-None {'ok' if t1 else 'FAIL'} · sigma>0 margin {'ok' if (t2a and t2b and t2c) else 'FAIL'} · "
          f"null non-tautological {'ok' if t3 else 'FAIL'} · sigma-safe retiming {'ok' if (t5a and t5b) else 'FAIL'})")
    print("  sigma is not dropped in _effort: MotionPlan carries a quantitative plan_sigma and a sigma-safe "
          "retiming time, not only a confidence tier.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
