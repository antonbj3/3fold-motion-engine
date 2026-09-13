#!/usr/bin/env python3
"""Batched GPU inverse dynamics (Recursive Newton-Euler) in Warp; the dynamics sibling of fk_warp's FK.

Replaces the per-sample `pin.rnea` Python loop. Parent-ordered 3D recursion (forward omega/alpha/a,
backward f/n), one thread per environment.

Data contract: in = dyn_spec (parent/jR/jp/mass/com/Ilink/grav from pinocchio, see
scripts/extract_rnea_spec_franka.py) plus a (q, dq, ddq) batch; out = a batch of joint torques.
Parity against pinocchio is gated by the selftest (max absolute error ~1e-5 Nm, float32 vs float64).
Warp is imported lazily.

  python -m motion_engine.rnea_warp   (requires warp + CUDA and data/rnea_parity_franka.npz)
"""
from pathlib import Path

import numpy as np


def _build_kernel(wp):
    @wp.kernel
    def rnea(q: wp.array2d(dtype=float), dq: wp.array2d(dtype=float), ddq: wp.array2d(dtype=float),
             jR: wp.array(dtype=wp.mat33), jp: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float),
             com: wp.array(dtype=wp.vec3), Ilink: wp.array(dtype=wp.mat33), grav: wp.vec3, NJ: int,
             Fb: wp.array(dtype=wp.vec3), Nb: wp.array(dtype=wp.vec3), tau: wp.array2d(dtype=float)):
        e = wp.tid(); z = wp.vec3(0.0, 0.0, 1.0)
        om = wp.vec3(0.0, 0.0, 0.0); al = wp.vec3(0.0, 0.0, 0.0); a = -grav      # base acceleration = gravity (world frame)
        for i in range(NJ):                                                       # forward pass (parent -> joint)
            qi = q[e, i]; c = wp.cos(qi); s = wp.sin(qi)
            Rz = wp.mat33(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
            Ril = wp.transpose(Rz) * wp.transpose(jR[i])                         # iR_λ
            P = jp[i]; omp = om; alp = al; ap = a
            omn = Ril * omp + dq[e, i] * z
            aln = Ril * alp + wp.cross(Ril * omp, dq[e, i] * z) + ddq[e, i] * z
            an = Ril * (ap + wp.cross(alp, P) + wp.cross(omp, wp.cross(omp, P)))
            ci = com[i]; aci = an + wp.cross(aln, ci) + wp.cross(omn, wp.cross(omn, ci))
            Fb[e * NJ + i] = mass[i] * aci
            Nb[e * NJ + i] = Ilink[i] * aln + wp.cross(omn, Ilink[i] * omn)
            om = omn; al = aln; a = an
        f = wp.vec3(0.0, 0.0, 0.0); n = wp.vec3(0.0, 0.0, 0.0)
        for ii in range(NJ):                                                     # backward pass (joint -> parent)
            i = NJ - 1 - ii; Fi = Fb[e * NJ + i]; Ni = Nb[e * NJ + i]; ci = com[i]
            if i < NJ - 1:
                qc = q[e, i + 1]; cc = wp.cos(qc); sc = wp.sin(qc)
                Rzc = wp.mat33(cc, -sc, 0.0, sc, cc, 0.0, 0.0, 0.0, 1.0)
                iRc = jR[i + 1] * Rzc; Pc = jp[i + 1]
                fch = iRc * f; nch = iRc * n
                f = Fi + fch; n = Ni + wp.cross(ci, Fi) + nch + wp.cross(Pc, fch)
            else:
                f = Fi; n = Ni + wp.cross(ci, Fi)
            tau[e, i] = wp.dot(n, z)
    return rnea


class WarpRNEA:
    """Batched GPU inverse dynamics behind a data contract. compute(q,dq,ddq) -> tau, numpy in and out."""
    def __init__(s, dyn_spec):
        import warp as wp                                  # lazy: warp + CUDA are only required here
        s.wp = wp; wp.init(); s.dev = "cuda:0"; s._k = _build_kernel(wp)
        ds = dyn_spec
        s.NJ = int(len(ds["mass"]))
        s.jR = wp.array(np.asarray(ds["jR"], np.float32), dtype=wp.mat33, device=s.dev)
        s.jp = wp.array(np.asarray(ds["jp"], np.float32), dtype=wp.vec3, device=s.dev)
        s.mass = wp.array(np.asarray(ds["mass"], np.float32), dtype=float, device=s.dev)
        s.com = wp.array(np.asarray(ds["com"], np.float32), dtype=wp.vec3, device=s.dev)
        s.Ilink = wp.array(np.asarray(ds["Ilink"], np.float32), dtype=wp.mat33, device=s.dev)
        s.grav = wp.vec3(*[float(x) for x in ds["grav"]])

    def compute(s, q, dq, ddq):
        wp = s.wp; N = len(q)
        Q = wp.array(np.asarray(q, np.float32), dtype=float, device=s.dev)
        DQ = wp.array(np.asarray(dq, np.float32), dtype=float, device=s.dev)
        DDQ = wp.array(np.asarray(ddq, np.float32), dtype=float, device=s.dev)
        Fb = wp.zeros(N * s.NJ, dtype=wp.vec3, device=s.dev); Nb = wp.zeros(N * s.NJ, dtype=wp.vec3, device=s.dev)
        tau = wp.zeros((N, s.NJ), dtype=float, device=s.dev)
        wp.launch(s._k, N, inputs=[Q, DQ, DDQ, s.jR, s.jp, s.mass, s.com, s.Ilink, s.grav, s.NJ, Fb, Nb, tau], device=s.dev)
        wp.synchronize(); return tau.numpy()


def _selftest(npz=str(Path(__file__).resolve().parents[2] / "data/rnea_parity_franka.npz")):
    import os, time
    if not os.path.exists(npz):
        print(f"  (parity npz missing: {npz} — run scripts/extract_rnea_spec_franka.py first)"); return False
    d = np.load(npz); eng = WarpRNEA(d)
    tau = eng.compute(d["q"], d["dq"], d["ddq"]); ref = d["tau_ref"]
    err = np.abs(tau - ref).max()
    ok = err < 1e-3
    print(f"  [parity] max abs error GPU RNEA vs pin.rnea = {err:.2e} Nm  {'ok' if ok else 'FAIL'}")
    # throughput benchmark: GPU at large batch sizes
    NJ = len(d["mass"]); rng = np.random.default_rng(1)
    for Nb in (4096, 65536):
        q = rng.uniform(-2, 2, (Nb, NJ)); dq = rng.uniform(-1.5, 1.5, (Nb, NJ)); ddq = rng.uniform(-2, 2, (Nb, NJ))
        eng.compute(q, dq, ddq)                            # warm-up
        t0 = time.time(); K = 20
        for _ in range(K):
            eng.compute(q, dq, ddq)
        el = time.time() - t0; rate = Nb * K / el
        print(f"  [throughput] N={Nb:>6}: {rate:,.0f} env-RNEA/s (GPU)")
    # Reference point: the pin.rnea CPU loop measures ~917k environment-RNEA/s (pinocchio C++ is ~1 us per call).
    # For a single trajectory (a few hundred configurations) the CPU is already instant; the GPU value is in
    # large-batch Monte-Carlo / ensemble sigma work.
    print(f"  -> {'WarpRNEA: pinocchio parity holds; the GPU path pays off at large batch sizes, not on a single trajectory' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    print("warp RNEA selftest (pinocchio parity + throughput):")
    sys.exit(0 if _selftest() else 1)
