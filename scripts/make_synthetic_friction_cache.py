#!/usr/bin/env python3
"""Generate a synthetic friction cache with the schema the friction-band cells consume.

The cells were developed against recordings of two real robots. Those recordings are not
redistributable, so this generator writes a cache with the same keys and the same qualitative physics:
rigid-body torque, motor current with Coulomb + viscous friction, and a PRESLIDING term that the
Coulomb+viscous model cannot represent (displacement-proportional near zero velocity, with
loading/unloading hysteresis through an acceleration-dependent term). The result is that the three cells
reproduce their structure -- identifiable in sliding, a systematic near-static residual, flat response to a
richer static velocity law, rising response to a dynamic term, and a regime-gated error band -- on data
whose ground truth is known. The numbers are NOT the original measured numbers.

Keys written: q_tr, dq_tr, ddq_tr, I_tr, dt_tr, gaps_tr, tau_dyn_tr, the same six with the _te suffix, and
load_bearing.

I/O:
  --robot NAME      cache name (default writes both `ur10e` and `ur3e` style caches)
  --n-train N       training samples (default 40000)
  --n-test N        test samples (default 6000)
  --out-dir PATH    default data/friction/
"""
import argparse
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def synth(n, rng, scale, dt=0.002):
    """One split: joint trajectories, rigid-body torque, and current with presliding friction."""
    t = np.arange(n) * dt
    q = np.zeros((n, 6))
    dq = np.zeros((n, 6))
    ddq = np.zeros((n, 6))
    # half the record is fast sliding motion, half is creep at 60x lower frequency, so the presliding
    # regime (|dq| < 0.02 rad/s) is sampled densely instead of only at the turning points
    half = n // 2
    slow = np.ones(n)
    slow[half:] = 1.0 / 60.0
    for j in range(6):
        f1, f2 = 0.11 + 0.037 * j, 1.7 + 0.31 * j
        a1 = 0.9 * scale[j]
        a2 = 0.04 * scale[j]
        w1 = 2 * np.pi * f1 * slow
        w2 = 2 * np.pi * f2 * slow
        ph1 = np.cumsum(w1) * dt + j
        ph2 = np.cumsum(w2) * dt
        q[:, j] = a1 * np.sin(ph1) + a2 * np.sin(ph2)
        dq[:, j] = a1 * w1 * np.cos(ph1) + a2 * w2 * np.cos(ph2)
        ddq[:, j] = -a1 * w1 ** 2 * np.sin(ph1) - a2 * w2 ** 2 * np.sin(ph2)
    inertia = np.array([2.4, 3.1, 1.6, 0.45, 0.38, 0.22])
    grav = np.array([0.0, 42.0, 18.0, 1.4, 1.1, 0.0])
    tau_dyn = ddq * inertia + grav * np.sin(q)
    Kt = np.array([0.42, 0.44, 0.37, 0.21, 0.20, 0.19])
    fc = np.array([0.9, 3.4, 3.6, 0.42, 0.40, 0.31])   # largest on the load-bearing joints
    fv = np.array([0.85, 1.10, 0.62, 0.19, 0.18, 0.14])
    # presliding: near zero velocity friction is displacement-proportional, with a dynamic (accel)
    # term that makes loading and unloading differ -- exactly what Coulomb+viscous cannot express
    # LuGre bristle state: dz/dt = dq - |dq| * z / g(dq), F = sigma0*z + sigma1*dz + fv*dq.
    # This is a STATE, not a function of velocity, so no static velocity law (Stribeck included) can
    # absorb it, while an acceleration term partly can -- and it produces loading/unloading hysteresis.
    sigma0 = np.array([210.0, 620.0, 700.0, 96.0, 92.0, 74.0])
    sigma1 = np.array([1.6, 2.0, 1.3, 0.36, 0.34, 0.27])
    v_s = 0.05
    z = np.zeros((n, 6))
    zi = np.zeros(6)
    g = fc + (1.45 * fc - fc) * np.exp(-(np.abs(dq) / v_s) ** 2)      # Stribeck curve of the bristles
    for k in range(n):
        dz = dq[k] - np.abs(dq[k]) * zi / np.maximum(g[k] / sigma0, 1e-9)
        zi = zi + dz * dt
        z[k] = zi
    dz_arr = np.vstack([np.zeros((1, 6)), np.diff(z, axis=0) / dt])
    # plus a directional presliding deficit: the fixed Coulomb level is not yet reached while the
    # bristles are still deflecting, so the near-static residual has a systematic sign as well
    gate = np.exp(-(np.abs(dq) / v_s) ** 2)
    tau_fr = sigma0 * z + sigma1 * dz_arr + fv * dq - 0.45 * fc * np.sign(dq) * gate
    I = (tau_dyn + tau_fr) / Kt + rng.normal(0.0, 0.005, (n, 6))
    gaps = np.zeros(n, dtype=bool)
    gaps[rng.random(n) < 0.002] = True
    return q, dq, ddq, I, np.full(n, dt), gaps, tau_dyn.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default=None)
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--n-test", type=int, default=6000)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data", "friction"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    robots = [args.robot] if args.robot else ["ur10e", "ur3e"]
    for ri, robot in enumerate(robots):
        rng = np.random.default_rng(11 + ri)
        scale = np.array([1.0, 0.9, 1.1, 1.0, 0.95, 1.05]) * (1.0 if ri == 0 else 0.7)
        tr = synth(args.n_train, rng, scale)
        # the second robot's test split is drawn from a shifted regime mix, so a band built on its own
        # train split does not transfer -- the reason the certificate has to be validated per robot
        te = synth(args.n_test, rng, scale * (1.03 if ri == 0 else 1.9))
        d = {}
        for name, split in (("tr", tr), ("te", te)):
            for key, v in zip(("q", "dq", "ddq", "I", "dt", "gaps", "tau_dyn"), split):
                d[f"{key}_{name}"] = v
        d["load_bearing"] = np.array([2, 3], dtype=np.int64)
        p = os.path.join(args.out_dir, f"friction_cache_{robot}.npz")
        np.savez_compressed(p, **d)
        print(f"synthetic friction cache: {robot}, {args.n_train} train / {args.n_test} test samples, "
              f"load-bearing joints {list(d['load_bearing'])} -> {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
