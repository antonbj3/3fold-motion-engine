"""Shared harness for the friction-model tournament: every model is scored the same way on the same split.

Each friction model supplies a `feature_builder(data, joint, split) -> (N, k) ndarray` (friction feature
columns; stateful models integrate their state inside). The harness does everything else identically per
model:

  * worst_dir metric (certificate): fit [I, *features, 1] -> tau_dyn on TRAIN, then take the multivariate
    worst direction on the HELD-OUT load-bearing joints.
  * discriminator AUC (gap): fit features -> measured friction (= a*I + b - tau_dyn) on TRAIN, sample with
    sigma, and run a classifier two-sample test on the held-out (dq, friction) pairs.

For non-linear parameters (e.g. LuGre vs/sigma0), call `train_sse(robot, fb)` inside a scipy optimiser and
then `evaluate(robot, fb)`.

The real UR recordings this harness scores are not shipped with this repository.
Requires numpy and scipy; the distribution discriminator is optional (set FRICTION_DISCRIMINATOR_PATH to a
directory providing lib.world_model.reality_gap_discriminator, otherwise the AUC column is reported as None).
"""
import os
import sys
from pathlib import Path

import numpy as np

_DISC = os.environ.get("FRICTION_DISCRIMINATOR_PATH")
if _DISC:
    sys.path.insert(0, _DISC)
try:
    from lib.world_model.reality_gap_discriminator import reality_gap  # noqa: E402
except Exception:  # discriminator not available -> AUC is reported as None
    reality_gap = None

_CACHE = Path(__file__).resolve().parents[1] / "data"
CERT = 0.58


def load(robot):
    d = dict(np.load(_CACHE / f"friction_cache_{robot}.npz"))
    d["load_bearing"] = [int(j) for j in d["load_bearing"]]
    return d


def _r2(y, p):
    return 1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30)


def _worst_dir(yr, yp):
    mu, sd = yr.mean(0), yr.std(0) + 1e-9
    A, Bb = (yr - mu) / sd, (yp - mu) / sd
    M = np.linalg.solve(np.cov(A.T), np.cov((A - Bb).T))
    return float(1 - np.linalg.eigvalsh((M + M.T) / 2).max())


def _design(data, j, split, fb):
    I = data[f"I_{split}"][:, j]
    F = np.atleast_2d(fb(data, j, split))
    if F.shape[0] != len(I):
        F = F.T
    return np.column_stack([I, F, np.ones(len(I))]), F


def train_sse(robot, fb, data=None):
    """Summed squared TRAIN residual over the load-bearing joints (for non-linear parameter optimisation)."""
    data = data or load(robot)
    sse = 0.0
    for j in data["load_bearing"]:
        A, _ = _design(data, j, "tr", fb)
        c, *_ = np.linalg.lstsq(A, data["tau_dyn_tr"][:, j], rcond=None)
        sse += float(((A @ c - data["tau_dyn_tr"][:, j]) ** 2).sum())
    return sse


def evaluate(robot, fb, seed=0, data=None):
    """Held-out metrics: worst_dir (certificate), per-joint R2, discriminator AUC (median over load-bearing joints)."""
    data = data or load(robot)
    lb = data["load_bearing"]
    Ttr, Tte = data["tau_dyn_tr"], data["tau_dyn_te"]
    preds = np.zeros((len(Tte), 6)); r2 = {}
    aucs = []
    rng = np.random.default_rng(seed)
    for j in lb:
        Atr, Ftr = _design(data, j, "tr", fb)
        c, *_ = np.linalg.lstsq(Atr, Ttr[:, j], rcond=None)
        Ate, Fte = _design(data, j, "te", fb)
        preds[:, j] = Ate @ c
        r2[int(j + 1)] = round(float(_r2(Tte[:, j], preds[:, j])), 4)
        # discriminator: fit features -> measured friction on TRAIN, sample with sigma, two-sample test on held-out
        kb, *_ = np.linalg.lstsq(np.column_stack([data["I_tr"][:, j], np.ones(len(Ttr))]), Ttr[:, j], rcond=None)
        realf_tr = kb[0] * data["I_tr"][:, j] + kb[1] - Ttr[:, j]
        realf_te = kb[0] * data["I_te"][:, j] + kb[1] - Tte[:, j]
        cf, *_ = np.linalg.lstsq(np.column_stack([Ftr, np.ones(len(Ftr))]), realf_tr, rcond=None)
        modelf_te = np.column_stack([Fte, np.ones(len(Fte))]) @ cf
        sigma = float(np.std(realf_tr - np.column_stack([Ftr, np.ones(len(Ftr))]) @ cf))
        flav = modelf_te + rng.standard_normal(len(modelf_te)) * sigma
        if reality_gap is None:
            continue
        g = reality_gap(np.column_stack([data["dq_te"][:, j], realf_te]),
                        np.column_stack([data["dq_te"][:, j], flav]), n_perm=100, seed=1)
        aucs.append(g.auc)
    wd = _worst_dir(Tte[:, lb], preds[:, lb])
    auc = round(float(np.median(aucs)), 4) if aucs else None
    return dict(worst_dir=round(wd, 4), per_joint_r2=r2, discriminator_auc=auc,
                crossed_cert=bool(wd > CERT), load_bearing=[int(j + 1) for j in lb])


# ── reference models (the baselines every model is compared against) ──
def fb_static_stribeck(data, j, split, vs=0.05):
    dq = data[f"dq_{split}"][:, j]
    return np.column_stack([np.sign(dq), dq, np.exp(-(np.abs(dq) / vs) ** 2) * np.sign(dq)])


def fb_lugre_factory(s0, s1, Fc, Fs, Fv, vs):
    def fb(data, j, split):
        dq, dt, gaps = data[f"dq_{split}"][:, j], data[f"dt_{split}"], data[f"gaps_{split}"]
        z = 0.0; F = np.empty_like(dq)
        for k in range(len(dq)):
            if gaps[k]:
                z = 0.0
            g = Fc + (Fs - Fc) * np.exp(-(dq[k] / vs) ** 2); g = g if g > 1e-6 else 1e-6
            zn = (z + dt[k] * dq[k]) / (1.0 + dt[k] * s0 * abs(dq[k]) / g)
            F[k] = s0 * zn + s1 * (zn - z) / dt[k] + Fv * dq[k]; z = zn
        return F.reshape(-1, 1)
    return fb


if __name__ == "__main__":
    # self-verification: reproduces the known UR10e static 0.559 and LuGre ~0.659
    print("harness self-verification, UR10e:")
    print("  static Stribeck:", evaluate("ur10e", fb_static_stribeck))
    from scipy.optimize import least_squares
    data = load("ur10e")
    # fit LuGre non-linearly (joint-shared parameters for simplicity in the verification)
    p0 = [100., 1., 1., 2., .5, .05]; lo = [1, 0, 0, 0, 0, .005]; hi = [1e5, 1e3, 50, 100, 50, 1]
    sol = least_squares(lambda p: [np.sqrt(train_sse("ur10e", fb_lugre_factory(*p), data))], p0,
                        bounds=(lo, hi), max_nfev=40)
    print("  dynamic LuGre :", evaluate("ur10e", fb_lugre_factory(*sol.x), data=data))
