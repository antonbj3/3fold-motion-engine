#!/usr/bin/env python
"""Selftest for dynamics_twin: loadable, runnable and round-trip correct.

Checks that twin_model_{robot}.npz reloads into identical per-joint predictions (DynamicsTwin.load() +
predict_torque() against the training-time twin_pred), plus arbitrary single-sample input and the
forward-dynamics API. Requires the training corpus artefacts (twin_corpus_*.npz, twin_pred_*.npz), which
are recorded robot data and are not shipped with this repository; without them the checks are skipped.

  python src/motion_engine/selftest_dynamics_twin.py
"""
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
from motion_engine.dynamics_twin import DynamicsTwin  # noqa: E402

C = []


def chk(n, c):
    C.append((n, bool(c)))


for robot in ("franka", "melfa"):
    mp = _ROOT / f"data/twin_corpus/twin_model_{robot}.npz"
    cp = _ROOT / f"data/twin_corpus/twin_corpus_{robot}.npz"
    pp = _ROOT / f"data/twin_corpus/twin_pred_{robot}.npz"
    if not (mp.exists() and cp.exists() and pp.exists()):
        print(f"  - {robot}: artefacts missing — skipped")
        continue
    corp = np.load(cp, allow_pickle=True); pred = np.load(pp, allow_pickle=True)
    Xte = corp["X_te"]; base = corp["tau_dyn_arm_te"]; dof = int(corp["dof"])
    q, dq, ddq = Xte[:, :dof], Xte[:, dof:2 * dof], Xte[:, 2 * dof:3 * dof]

    twin = DynamicsTwin.load(robot, root=_ROOT)
    chk(f"{robot}: laddad (dof={twin.dof}, members={twin.n_members})", twin.dof == dof and twin.n_members >= 1)

    tau_load, sig_load = twin.predict_torque(q, dq, ddq, base)
    tau_train, sig_train = pred["tau_twin"], pred["sigma"]
    rel_tau = float(np.max(np.abs(tau_load - tau_train)) / max(np.abs(tau_train).max(), 1e-9))
    rel_sig = float(np.max(np.abs(sig_load - sig_train)) / max(np.abs(sig_train).max(), 1e-9))
    chk(f"{robot}: round-trip tau (loaded == trained, rel {rel_tau:.1e})", rel_tau < 1e-3)
    chk(f"{robot}: round-trip sigma (rel {rel_sig:.1e})", rel_sig < 1e-3)

    # arbitrary single-sample input (runnable on a single query, not only on batches)
    t1, s1 = twin.predict_torque(q[0], dq[0], ddq[0], base[0])
    chk(f"{robot}: single-sample inference finite and correctly shaped", t1.shape == (1, dof) and np.all(np.isfinite(t1)) and np.all(s1 > 0))

    # forward-dynamics API: base-agnostic qddot = M^-1 (tau - bias - residual); synthetic M=I, bias=0
    Mid = np.eye(dof); bz = np.zeros(dof); tau_test = base[0] + np.asarray(twin.predict_residual(q[0], dq[0], ddq[0])[0][0])
    qdd_shrink = twin.forward_qddot(Mid, bz, q[0], dq[0], tau_test, shrink=True)
    qdd_raw = twin.forward_qddot(Mid, bz, q[0], dq[0], tau_test, shrink=False)
    chk(f"{robot}: forward_qddot finite and correctly shaped", qdd_shrink.shape == (dof,) and np.all(np.isfinite(qdd_shrink)))
    chk(f"{robot}: sigma shrinkage changes qddot (the sigma head is active)", not np.allclose(qdd_shrink, qdd_raw))

ok = all(c for _, c in C) and len(C) > 0
for n, c in C:
    print(f"  {'✓' if c else '✗'} {n}")
print(f"\ndynamics_twin selftest ({len(C)} checks): {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
