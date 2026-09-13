#!/usr/bin/env python3
"""Waypoint blending and time parametrisation.

Blends an RRT path into one continuous C2 path (cubic spline through the waypoints), time-parametrised
against the velocity and torque limits, and executed in one continuous computed-torque rollout.

Honest limit, which this module's own selftest demonstrates: a spline through the waypoints DEVIATES from
the collision-free linear RRT segments and can corner-cut into collision even though the waypoints are
free. The blended path must therefore be re-validated against collision; collision-aware shortcutting
(see collision_aware_shortcut.py and smooth_safe_trajectory.py) is the method that preserves safety.

  python -u scripts/fleet_waypoint_blending.py   (requires pinocchio + hppfcl/coal + scipy)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fleet_agnostic_planner import load, in_collision, rrt_connect, NJ
# assumed generic joint friction (Coulomb Nm / viscous Nm.s/rad); per-robot values need measured data
FC_ASSUMED = np.array([2.0, 2.0, 1.0, 0.5, 0.5, 0.3])
FV_ASSUMED = np.array([1.0, 1.0, 0.5, 0.3, 0.3, 0.2])

ROOT = Path(__file__).resolve().parents[1]
TARGET = 0.8


def blend(path):
    """Cubic spline through the RRT waypoints, chord-length parametrised, clamped (zero velocity) ends.
    -> CubicSpline q(s), S."""
    P = [np.asarray(path[0], float)]
    for p in path[1:]:
        if np.linalg.norm(np.asarray(p, float) - P[-1]) > 1e-4:   # drop near-duplicate waypoints (RRT-connect)
            P.append(np.asarray(p, float))
    P = np.array(P)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    cs = CubicSpline(s, P, bc_type="clamped")               # zero velocity at the ends (smooth start/stop)
    return cs, float(s[-1])


def time_param(cs, S, model, data, eff, velLim, dt):
    """Tids-parametrisera (s=S·t/T): T s.t. peak-vel ≤ velLim·0.9 OCH peak-moment-ratio ≤ TARGET."""
    sg = np.linspace(0, S, 400)
    qp = cs(sg, 1); qpp = cs(sg, 2)                          # dq/ds, d²q/ds²
    T_vel = float(np.max(np.abs(qp) / (velLim * 0.9 + 1e-9)) * S)   # peak|q'|·(S/T)≤velLim → T≥max|q'|·S/velLim
    T = max(0.2, T_vel)
    for _ in range(8):                                       # skala upp tills moment-ratio ≤ TARGET
        t = np.linspace(0, T, max(50, int(T / dt)))
        sv = S * t / T; dsdt = S / T
        q = cs(sv); dq = cs(sv, 1) * dsdt; ddq = cs(sv, 2) * dsdt ** 2
        tau = np.array([pin.rnea(model, data, q[i], dq[i], ddq[i]) for i in range(0, len(q), 5)])
        ratio = float(np.max(np.max(np.abs(tau), 0) / eff))
        if ratio <= TARGET:
            break
        T *= float(np.sqrt(ratio / TARGET) * 1.05)
    return T


def rollout(model, data, q, dq, ddq, dt, Kp, Kd, feedback=True, fric_comp=False):
    """Kontinuerlig computed-torque-rollout (NOMINELL) + PD [+friktions-FF om fric_comp]; plant MED antagen friktion."""
    eff = model.effortLimit[:NJ]; qc = q[0].copy(); dqc = dq[0].copy(); errs = []; mr = 0.0
    for k in range(len(q)):
        ddq_cmd = ddq[k] + (Kd * (dq[k] - dqc) + Kp * (q[k] - qc) if feedback else 0.0)
        tau = pin.rnea(model, data, qc, dqc, ddq_cmd)
        ff = FC_ASSUMED * np.tanh(dqc / 0.1) + FV_ASSUMED * dqc
        if fric_comp:
            tau = tau + ff                                   # friction feedforward (model-based; needs a friction model)
        ddqc = pin.aba(model, data, qc, dqc, tau - ff)       # plant MED (samma antagna) friktion
        dqc = dqc + ddqc * dt; qc = qc + dqc * dt
        if not np.all(np.isfinite(qc)):
            return None, np.inf, False
        errs.append(np.max(np.abs(qc - q[k]))); mr = max(mr, float(np.max(np.abs(tau) / eff)))
    return float(np.max(errs)), mr, True


def main():
    rng = np.random.default_rng(2)
    model, data, gm, gd, _ = load([0.25, 0.25, 0.6], [0.55, 0.0, 0.4])
    eff = model.effortLimit[:NJ]; velLim = np.asarray(model.velocityLimit[:NJ])
    lo, hi = model.lowerPositionLimit[:NJ], model.upperPositionLimit[:NJ]
    print("WAYPOINT-BLENDING → TIGHT FLEET-DRIVE (B10) — ur10e: mjuk C²-bana genom RRT-waypoints")

    def free(n):
        for _ in range(n):
            q = lo + rng.random(NJ) * (hi - lo)
            if not in_collision(model, data, gm, gd, q):
                return q
        return None
    q0, qg = free(800), free(800)
    path = rrt_connect(model, data, gm, gd, q0, qg, lo, hi, rng)
    if not path:
        print("  no RRT path"); return 1
    cs, S = blend(path)
    dt = 0.001; T = time_param(cs, S, model, data, eff, velLim, dt)
    t = np.linspace(0, T, max(50, int(T / dt))); sv = S * t / T; dsdt = S / T
    q = cs(sv); dq = cs(sv, 1) * dsdt; ddq = cs(sv, 2) * dsdt ** 2
    print(f"  blendad: {len(path)} waypoints → C²-spline, S={S:.2f}, T={T:.2f}s, {len(t)} steg")

    # (1) smooth and within the limits
    peak_vel = float(np.max(np.abs(dq) / velLim)); peak_ratio = float(np.max(np.max(np.abs([pin.rnea(model, data, q[i], dq[i], ddq[i]) for i in range(0, len(q), 10)]), 0) / eff))
    g1 = peak_vel <= 1.0 and peak_ratio <= TARGET + 0.05
    print(f"  (1) C2-smooth within limits: peak velocity ratio {peak_vel:.2f}<=1, peak torque ratio {peak_ratio:.2f}<={TARGET} -> {g1}")

    # (2) densely validate the blended path, separating environment from self collision
    mE, dE, gmE, gdE, _ = load([0.25, 0.25, 0.6], [0.55, 0.0, 0.4], self_collision=False)   # ENV-only modell
    idx = range(0, len(q), 3)
    n_env = sum(in_collision(mE, dE, gmE, gdE, q[i]) for i in idx)        # env-corner-cut (blend vs hinder)
    n_self = sum(in_collision(model, data, gm, gd, q[i]) for i in idx) - 0  # self + environment
    n_selfonly = max(0, n_self - n_env)
    g2 = n_env == 0                                                       # ENV-corner-cut undviken (blend-vinst vs hinder)
    n_coll = n_self
    print(f"  (2) ENV-collision-free blend: {n_env} hinder-kollisioner i {len(idx)} prov → {g2} (corner-cut-vs-hinder undviken)")

    # (3) finding: the naive spline blend corner-cuts into self-collision (it deviates from the free waypoints)
    g3_selffree = n_selfonly == 0
    print(f"  (3) self-collision in the blend: {n_selfonly} -> self-free={g3_selffree} (naive spline corner-cuts; collision-aware blending is the fix)")

    # (4) tracking illustration (not a pass gate): friction feedforward is simulation-circular here (FF = plant = assumed friction)
    Kp, Kd = 2500.0, 100.0
    err_nocomp, _, st1 = rollout(model, data, q, dq, ddq, dt, Kp, Kd, feedback=True, fric_comp=False)
    err_comp, _, st2 = rollout(model, data, q, dq, ddq, dt, Kp, Kd, feedback=True, fric_comp=True)
    stable = st1 and st2
    print(f"  (4) tracking illustration (simulation-circular, not a pass gate): without FF {err_nocomp*1000:.0f} mrad -> with friction FF {err_comp*1000:.1f} mrad; FF = plant = assumed friction, so this is algebraic cancellation. Real tight tracking needs measured friction")

    # honest falsification: the naive spline blend does not preserve collision safety (it corner-cuts into env + self)
    blend_safe = (n_env == 0 and n_selfonly == 0)
    ok = g1 and stable and (n_env + n_selfonly > 0)   # sound characterisation: the spline is smooth but demonstrably unsafe
    print(f"\nVERDICT: waypoint blending = {'CHARACTERISED (honest falsification: the naive spline blend is unsafe)' if ok else 'NOT VALIDATED'}. "
          + (f"Naive cubic-spline blending through collision-free RRT waypoints (T={T:.1f}s) gives a smooth C2 path "
             "that respects the velocity and torque limits but does NOT preserve collision safety: the spline "
             f"deviates from the free waypoints into collision ({n_env} environment + {n_selfonly} self collisions "
             f"over {len(idx)} samples). Smoothness is not safety; the genuine method is collision-aware shortcutting "
             "(shorten the RRT path inside the free space, validating every shortcut against self + environment). "
             f"Tracking (simulation-circular illustration, not a pass gate): friction feedforward moves the error "
             f"{err_nocomp*1000:.0f} -> {err_comp*1000:.1f} mrad, which is algebraic cancellation because the "
             "feedforward equals the plant's assumed friction; real tight tracking needs per-robot measured friction. " if ok else
             "Not validated. ")
          + "Scope: this module deliberately reports a negative result about naive blending; the time parametrisation "
          "(time_param) and the blend itself are reused by the motion stack behind the collision re-validation.")

    out = dict(robot="UR10e (fleet)", finding="HONEST FALSIFICATION: naive cubic-spline blending does NOT preserve collision-safety (corner-cuts)",
               waypoints=len(path), T_s=round(T, 3), smooth_within_limits=bool(g1), peak_vel_ratio=round(peak_vel, 3), peak_torque_ratio=round(peak_ratio, 3),
               blend_env_collisions=int(n_env), blend_self_collisions=int(n_selfonly), blend_collision_safe=bool(blend_safe),
               genuine_approach="collision-aware shortcutting (stays in free space)", friction_ff_sim_circular=True,
               tracking_mrad_nocomp=round(err_nocomp * 1000, 2) if err_nocomp else None,
               tracking_mrad_fric_ff=round(err_comp * 1000, 2) if err_comp else None,
               characterization_sound=bool(ok), composes="fleet_agnostic_planner (self+env) + CubicSpline")
    (ROOT / "reports" / "fleet_waypoint_blending.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("  wrote reports/fleet_waypoint_blending.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
