"""DYNAMIC FEASIBILITY — the twin↔planner fusion product as deployable code (2026-06-20, s2).

Composes a calibrated dynamics twin with a planner's geometric path → (1) σ-conservative dynamic-feasibility
check, (2) torque-aware retiming = the FASTEST feasible execution (slow only at binding corners). The vertical
loop as a TOOL: foundation (twin dynamics) drives application (the path's time-profile).

    from motion_engine.dynamic_feasibility import feasibility, torque_aware_retime
    pt = lambda q, dq, ddq: cal.predict(base_fn, q, dq, ddq)        # InstanceCalibration → (tau, sigma)
    feasibility(pt, q, dq, ddq, effort_limit, k=3)                  # point + σ-conservative peak ratios
    res = torque_aware_retime(pt, q_u, qp_u, qpp_u, effort_limit, vmax)   # → RetimeResult(total_time, feasible, ...)

GEOMETRY IS NEVER CHANGED (retiming only re-times → collision-freeness of the input path is preserved). Verify
the input path's collision-freeness separately (e.g. on a PCHIP densification — NOT a savgol smoothing, which
moves geometry into obstacles). The retimer re-verifies the FULL ü-coupled torque constraint before returning.
Honest scope: a heuristic (per-point bisection + iterative full-dynamics correction), not a provably
time-optimal forward-backward TOPP.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def feasibility(predict_torque, q, dq, ddq, effort_limit, *, k: float = 3.0) -> dict:
    """Point + σ-conservative peak effort-ratio along a (q,dq,ddq) trajectory.
    `predict_torque(q,dq,ddq) -> (tau (N,J), sigma (N,J))`. σ-conservative = (|τ|+k·σ)/limit (anti-false-positive)."""
    tau, sigma = predict_torque(q, dq, ddq)
    lim = np.asarray(effort_limit, float)[: tau.shape[1]]
    point = float(np.max(np.max(np.abs(tau), 0) / lim))
    cons = float(np.max(np.max(np.abs(tau) + k * sigma, 0) / lim))
    return dict(executable=bool(point < 1.0), peak_point=round(point, 3),
                executable_conservative=bool(cons < 1.0), peak_conservative=round(cons, 3), k=k)


@dataclass
class RetimeResult:
    udot: np.ndarray        # path-velocity profile u̇(u) (per path point)
    total_time: float       # execution time (s)
    peak_ratio: float       # full-dynamics peak torque ratio (≤ tol ⟹ feasible)
    feasible: bool
    n_iters: int            # iterations of the full-dynamics correction


def torque_aware_retime(predict_torque, q_u, qp_u, qpp_u, effort_limit, vmax, *,
                        k: float = 0.0, max_iter: int = 60, tol: float = 1.0) -> RetimeResult:
    """Fastest feasible retiming of a geometric path q(u), u∈[0,1] (q_u/qp_u/qpp_u = value/1st/2nd deriv wrt u,
    e.g. PCHIP). Per-point bisect path-velocity u̇ s.t. (|τ|+k·σ) ≤ limit + velocity limit, then ITERATE with
    the full ü-coupled dynamics until feasible (torque is NOT ∝ u̇² — gravity/friction don't scale — so one
    correction under-shoots; the savgol/tautology lesson → re-verify the real constraint, data-driven verdict)."""
    q_u, qp_u, qpp_u = np.asarray(q_u, float), np.asarray(qp_u, float), np.asarray(qpp_u, float)
    N = len(q_u); du = 1.0 / (N - 1)
    lim = np.asarray(effort_limit, float); vmax = np.asarray(vmax, float)

    def peak(udot, full):
        dq = qp_u * udot[:, None]
        ddq = qpp_u * (udot ** 2)[:, None]
        if full:
            ddq = ddq + qp_u * (np.gradient(udot, du) * udot)[:, None]   # + q'·ü
        tau, sig = predict_torque(q_u, dq, ddq)
        return np.max((np.abs(tau) + k * sig) / lim, axis=1)

    udot_vel = np.min(vmax[None, :] / (np.abs(qp_u) + 1e-9), axis=1)      # velocity limit on u̇
    lo, hi = np.zeros(N), udot_vel.copy()                                 # torque limit (ignore ü): bisect
    for _ in range(30):
        mid = 0.5 * (lo + hi); ok = peak(mid, False) <= 1.0
        lo = np.where(ok, mid, lo); hi = np.where(ok, hi, mid)
    target = 0.99 * tol            # ★converge 1% UNDER the hard limit (honest margin — never graze the threshold)
    udot = lo
    it, r = 0, peak(lo, True)
    for it in range(max_iter):                                           # iterate full ü-coupled dynamics
        r = peak(udot, True)
        if r.max() <= target:
            break
        udot = udot / np.maximum(np.sqrt(r / target), 1.0)               # per-point: slow where r>target (drives below)
    r = peak(udot, True)
    T = float(np.sum(du / (0.5 * (udot[:-1] + udot[1:]) + 1e-9)))
    return RetimeResult(udot=udot, total_time=T, peak_ratio=float(r.max()),
                        feasible=bool(r.max() <= tol), n_iters=it)         # feasible = strictly within hard limit


__all__ = ["feasibility", "torque_aware_retime", "RetimeResult"]
