"""GPU RNEA (Warp) parity check against pin.rnea. Parent-ordered forward (omega/alpha/a) and backward (f/n)
3D recursion, one thread per environment. Reads data/rnea_parity_franka.npz (see
scripts/extract_rnea_spec_franka.py). Requires a CUDA device.
"""
from pathlib import Path

import warp as wp, numpy as np

wp.init(); DEV = "cuda:0"
NPZ = Path(__file__).resolve().parents[1] / "data/rnea_parity_franka.npz"
d = np.load(NPZ); NJ = len(d["mass"]); N = len(d["q"])
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
    for ii in range(NJ):                                                      # backward pass (joint -> parent)
        i = NJ - 1 - ii; Fi = Fb[e * NJ + i]; Ni = Nb[e * NJ + i]; ci = com[i]
        if i < NJ - 1:
            qc = q[e, i + 1]; cc = wp.cos(qc); sc = wp.sin(qc)
            Rzc = wp.mat33(cc, -sc, 0.0, sc, cc, 0.0, 0.0, 0.0, 1.0)
            iRc = jR[i + 1] * Rzc; Pc = jp[i + 1]                            # iR_{i+1}, P_{i+1} i ram i
            fch = iRc * f; nch = iRc * n
            f = Fi + fch; n = Ni + wp.cross(ci, Fi) + nch + wp.cross(Pc, fch)
        else:
            f = Fi; n = Ni + wp.cross(ci, Fi)
        tau[e, i] = wp.dot(n, z)
a = lambda x, t: wp.array(x, dtype=t, device=DEV)
q = a(d["q"].astype(np.float32), float); dq = a(d["dq"].astype(np.float32), float); ddq = a(d["ddq"].astype(np.float32), float)
jR = a(d["jR"], wp.mat33); jp = a(d["jp"], wp.vec3); mass = a(d["mass"], float); com = a(d["com"], wp.vec3)
Ilink = a(d["Ilink"], wp.mat33); grav = wp.vec3(*[float(x) for x in d["grav"]])
Fb = wp.zeros(N * NJ, dtype=wp.vec3, device=DEV); Nb = wp.zeros(N * NJ, dtype=wp.vec3, device=DEV)
tau = wp.zeros((N, NJ), dtype=float, device=DEV)
wp.launch(rnea, N, inputs=[q, dq, ddq, jR, jp, mass, com, Ilink, grav, NJ, Fb, Nb, tau], device=DEV)
wp.synchronize()
tg = tau.numpy(); ref = d["tau_ref"]
err = np.abs(tg - ref); rel = err / (np.abs(ref) + 1e-6)
print(f"GPU RNEA vs pin.rnea parity (franka, N={N} samples, {NJ} joints):")
print(f"  max abs error {err.max():.2e} Nm | mean {err.mean():.2e} | max rel error {rel.max():.2e}")
print(f"  tau_GPU[0] = {np.round(tg[0],3)}")
print(f"  tau_ref[0] = {np.round(ref[0],3)}")
ok = err.max() < 1e-3
print(f"  -> {'PARITY (GPU RNEA == pin.rnea)' if ok else 'MISMATCH'}")
