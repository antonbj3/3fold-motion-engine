"""Deployable dynamics twin: a loadable, runnable twin, not just a validation report.

A reality-grounded dynamics twin that can be loaded and run on arbitrary (q, dq, ddq) to predict joint
torque plus a calibrated uncertainty. The twin is an analytic base (RNEA/system identification) plus a
learned ensemble residual (friction and unmodelled effects) with a heteroscedastic sigma head over a
K-member ensemble (epistemic spread). Inference is pure numpy (MLP = matmul + tanh), so no JAX or torch
is needed on the consumer side.

  twin = DynamicsTwin.load("franka", root=REPO_ROOT)
  tau, sigma = twin.predict_torque(q, dq, ddq, tau_analytical)   # (N,dof) mean + per-joint sigma
  resid, sigma = twin.predict_residual(q, dq, ddq)               # the learned residual only
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DEFAULT_ROOT = Path(__file__).resolve().parents[2]   # repository root


@dataclass
class DynamicsTwin:
    """Loadable reality-grounded dynamics twin. Pure-numpy inference (no JAX/torch needed)."""

    W1: np.ndarray; b1: np.ndarray; W2: np.ndarray; b2: np.ndarray   # (members, ...) staplat
    Wm: np.ndarray; bm: np.ndarray; Ws: np.ndarray; bs: np.ndarray
    mu: np.ndarray; sd: np.ndarray; rstd: np.ndarray                 # feature- + target-normalisering
    dof: int
    robot: str
    load_bearing: np.ndarray
    epi_ref: np.ndarray = None    # per-joint training-calibrated epistemic-sigma reference (sidecar)

    @property
    def n_members(self) -> int:
        return int(self.W1.shape[0])

    @classmethod
    def load(cls, robot: str, root: Path | str | None = None) -> "DynamicsTwin":
        root = Path(root) if root is not None else _DEFAULT_ROOT
        p = root / f"data/twin_corpus/twin_model_{robot}.npz"
        if not p.exists():
            raise FileNotFoundError(
                f"no saved twin for '{robot}' ({p})")
        d = np.load(p, allow_pickle=False)
        epi_ref = None                                                # optional sidecar: training-calibrated epistemic reference
        ep = p.parent / f"twin_epiref_{robot}.npz"
        if ep.exists():
            try: epi_ref = np.load(ep)["epi_ref"]
            except Exception: epi_ref = None
        return cls(W1=d["W1"], b1=d["b1"], W2=d["W2"], b2=d["b2"],
                   Wm=d["Wm"], bm=d["bm"], Ws=d["Ws"], bs=d["bs"],
                   mu=d["mu"], sd=d["sd"], rstd=d["rstd"], dof=int(d["dof"]),
                   robot=str(d["robot"]), load_bearing=d["load_bearing"], epi_ref=epi_ref)

    def _features(self, q, dq, ddq) -> np.ndarray:
        q, dq, ddq = (np.atleast_2d(np.asarray(a, float)) for a in (q, dq, ddq))
        X = np.concatenate([q, dq, ddq], axis=1)
        if X.shape[1] != 3 * self.dof:
            raise ValueError(f"expected (N,{self.dof}) for each of q/dq/ddq; got X width {X.shape[1]}")
        return ((X - self.mu) / self.sd).astype(np.float64)

    def _member(self, i: int, Xs: np.ndarray):
        h = np.tanh(Xs @ self.W1[i] + self.b1[i])
        h = np.tanh(h @ self.W2[i] + self.b2[i])
        return h @ self.Wm[i] + self.bm[i], h @ self.Ws[i] + self.bs[i]   # mu, log sigma^2 (normalised target space)

    def predict_residual(self, q, dq, ddq):
        """Learned residual (friction + unmodelled effects) plus total uncertainty sigma (aleatoric + epistemic), de-normalised.
        Returnerar (residual (N,dof), sigma (N,dof))."""
        Xs = self._features(q, dq, ddq)
        mus, vars_ = [], []
        for i in range(self.n_members):
            m, ls = self._member(i, Xs)
            mus.append(m); vars_.append(np.exp(np.clip(ls, -8, 8)))
        mus = np.stack(mus); ale = np.stack(vars_).mean(0)
        mu_ens = mus.mean(0); epi = mus.var(0)                  # epistemisk = ensemble-spridning
        residual = mu_ens * self.rstd
        sigma = np.sqrt(ale + epi) * self.rstd
        return residual, sigma

    def predict_torque(self, q, dq, ddq, tau_analytical):
        """Full twin torque = analytic base + learned residual, with calibrated sigma. tau_analytical = (N,dof)."""
        residual, sigma = self.predict_residual(q, dq, ddq)
        return np.asarray(tau_analytical, float) + residual, sigma

    def predict_residual_decomposed(self, q, dq, ddq):
        """As predict_residual but decomposes sigma -> (residual, sigma_total, sigma_epistemic, sigma_aleatoric), de-normalised.
        sigma_epistemic (ensemble spread) is the out-of-distribution signal: it grows where the twin extrapolates (members
        disagree). sigma_aleatoric (heteroscedastic head) is the irreducible noise. On real Franka data sigma_epistemic
        grows monotonically with feature distance and tracks the growth of the error, which makes it a usable validity
        signal for the deployed twin."""
        Xs = self._features(q, dq, ddq)
        mus, vars_ = [], []
        for i in range(self.n_members):
            m, ls = self._member(i, Xs)
            mus.append(m); vars_.append(np.exp(np.clip(ls, -8, 8)))
        mus = np.stack(mus); ale = np.stack(vars_).mean(0); epi = mus.var(0)
        residual = mus.mean(0) * self.rstd
        return (residual, np.sqrt(ale + epi) * self.rstd, np.sqrt(epi) * self.rstd, np.sqrt(ale) * self.rstd)

    def certificate(self, q, dq, ddq, *, epi_ref=None) -> dict:
        """Validity certificate for the deployed twin: is (q,dq,ddq) inside the trained distribution?
        The signal is the ensemble epistemic sigma (it grows out of distribution, where the members disagree).
        Covered where sigma_epi <= epi_ref (per joint, training-calibrated, default self.epi_ref from the sidecar);
        above it the twin sigma is less trustworthy. This is a separate axis from the sigma magnitude.
        Returns dict(covered, epi_ratio (sigma_epi/epi_ref), frac_extrap) per batch (max over joints)."""
        ref = epi_ref if epi_ref is not None else self.epi_ref
        _, _, epi, _ = self.predict_residual_decomposed(q, dq, ddq)
        epi = np.atleast_2d(np.abs(epi))
        if ref is None:
            return dict(covered=True, epi_ratio=0.0, frac_extrap=0.0, note="no epi_ref calibrated -> unbounded")
        ref = np.asarray(ref, float)
        ratio_pj = epi / max(float(ref), 1e-12) if ref.ndim == 0 else epi / np.maximum(ref, 1e-12)[None, :]
        per_sample = ratio_pj.max(1)                                  # worst joint per sample (epi/epi_ref)
        frac = float(np.mean(per_sample > 1.0))                       # fraction of samples/waypoints above the reference
        # covered = the trajectory is mostly in-distribution (<=5% above the 99th-percentile reference). For a single
        # query frac is 0 or 1, so covered iff below the reference.
        return dict(covered=bool(frac <= 0.05), epi_ratio=round(float(per_sample.max()), 3), frac_extrap=round(frac, 3))

    def forward_qddot(self, M, bias, q, dq, tau, *, shrink=False, iters=2):
        """Forward dynamics: qddot = M^-1 (tau - bias - residual(q, dq, qddot)), solved as a fixed point over `iters`.
        Base-agnostic: the caller supplies M (dof x dof) and bias (dof) from any physics engine. One sample.

        shrink=False (default): apply the learned residual in full. Best when the analytic base is weak and the
        residual is large and substantive.
        shrink=True: Wiener shrinkage r^2/(r^2+sigma^2) via the sigma head, pulling the residual towards the analytic
        base. Only helps when the base is good and the residual is small and noise-dominated; it does not generalise,
        because sigma cannot distinguish prediction noise from genuine signal variability.
        Returns qddot (dof,)."""
        M = np.asarray(M, float); bias = np.asarray(bias, float)
        q1 = np.atleast_2d(np.asarray(q, float)); dq1 = np.atleast_2d(np.asarray(dq, float))
        qddot = np.linalg.solve(M, np.asarray(tau, float) - bias)
        for _ in range(int(iters)):
            r, s = self.predict_residual(q1, dq1, qddot[None])
            r, s = r[0], s[0]
            if shrink:
                r = r * (r ** 2 / (r ** 2 + s ** 2 + 1e-12))
            qddot = np.linalg.solve(M, np.asarray(tau, float) - bias - r)
        return qddot


__all__ = ["DynamicsTwin"]
