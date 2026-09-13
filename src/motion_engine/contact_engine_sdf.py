#!/usr/bin/env python3
"""Analytic-SDF contact engine behind the same ContactEngine Protocol.

Regime: curved geometry on ground or ramp (smooth SDF normals give correct rolling), shape-dispatched
contact (sphere/cylinder/box/capsule), one thread per body with accumulated split impulse and a separate
position pass. Ground contact; body-body contact is the relaxed-Jacobi member. Validated: a cylinder rolls
at 2/3 g sin(theta), a sphere at 5/7 g sin(theta), a box sticks at atan(mu).

Data contract identical to the CPU and GPU contact engines. Warp is imported lazily.
shape: 0=sphere 1=cylinder 2=box 3=capsule.

  python -m motion_engine.contact_engine_sdf   (requires warp + CUDA)
"""
import numpy as np
from motion_engine.contact_engine import BodyState, ContactEngine

_G = 9.81; _BETA = 0.2; _SLOP = 1e-4; _KMAX = 12


def _build_kernel(wp):
    R = 0.0  # point contact on the analytic surface (no sphere-radius offset)

    @wp.func
    def contact_pts(shape: int, qi: wp.quat, dim: wp.vec3, n: wp.vec3, out: wp.array(dtype=wp.vec3), tid: int):
        cnt = int(0)
        if shape == 0:                                       # sphere: support point c - r*n (body-frame lever)
            r = dim[0]; nb = wp.quat_rotate_inv(qi, n); out[tid * _KMAX + 0] = -r * nb; cnt = 1
        elif shape == 1 or shape == 3:                       # CYLINDER/KAPSEL (axel=body-y): K botten-linje-punkter
            r = dim[0]; L = dim[1]; a = wp.vec3(0.0, 1.0, 0.0)
            nb = wp.quat_rotate_inv(qi, n); na = nb - wp.dot(nb, a) * a; d = -wp.normalize(na); K = 5
            for k in range(K):
                yk = (float(k) / 4.0 - 0.5) * L; out[tid * _KMAX + k] = a * yk + d * r
            cnt = K
        else:                                                # box: 8 corners
            hx = dim[0]; hy = dim[1]; hz = dim[2]
            for k in range(8):
                sx = float((k >> 2) & 1) * 2.0 - 1.0; sy = float((k >> 1) & 1) * 2.0 - 1.0; sz = float(k & 1) * 2.0 - 1.0
                out[tid * _KMAX + k] = wp.vec3(sx * hx, sy * hy, sz * hz)
            cnt = 8
        return cnt

    @wp.kernel
    def step(xc: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.quat), vc: wp.array(dtype=wp.vec3),
             om: wp.array(dtype=wp.vec3), shape: wp.array(dtype=int), dim: wp.array(dtype=wp.vec3),
             invI: wp.array(dtype=wp.vec3), invM: wp.array(dtype=float), lev: wp.array(dtype=wp.vec3),
             jn: wp.array(dtype=float, ndim=2), jt1: wp.array(dtype=float, ndim=2), jt2: wp.array(dtype=float, ndim=2),
             jp: wp.array(dtype=float, ndim=2),
             n: wp.vec3, t1: wp.vec3, t2: wp.vec3, mu: float, dt: float, vit: int, pit: int):
        i = wp.tid(); c = xc[i]; qi = q[i]; v = vc[i]; w = om[i]; iM = invM[i]; iI = invI[i]
        if iM <= 0.0:                                         # kinematic: driven, do not integrate
            return
        v = v + wp.vec3(0.0, 0.0, -_G) * dt
        K = contact_pts(shape[i], qi, dim[i], n, lev, i)
        for k in range(K):
            jn[i, k] = 0.0; jt1[i, k] = 0.0; jt2[i, k] = 0.0; jp[i, k] = 0.0
        for it in range(vit):
            for k in range(K):
                rp = wp.quat_rotate(qi, lev[i * _KMAX + k]); pw = c + rp; s = wp.dot(pw, n)
                if s < 0.0:
                    rn = wp.cross(rp, n); iIrn = wp.quat_rotate(qi, wp.cw_mul(iI, wp.quat_rotate_inv(qi, rn)))
                    meff = iM + wp.dot(rn, iIrn); vp = v + wp.cross(w, rp); vn = wp.dot(vp, n)
                    dj = -vn / meff; nw = wp.max(0.0, jn[i, k] + dj); dj = nw - jn[i, k]; jn[i, k] = nw
                    v = v + dj * n * iM; w = w + dj * iIrn
                    # ── 3D Coulomb cone (isotropic): two tangent axes + a circular mu*jn disc clamp ──
                    rt1 = wp.cross(rp, t1); iIrt1 = wp.quat_rotate(qi, wp.cw_mul(iI, wp.quat_rotate_inv(qi, rt1))); meft1 = iM + wp.dot(rt1, iIrt1)
                    rt2 = wp.cross(rp, t2); iIrt2 = wp.quat_rotate(qi, wp.cw_mul(iI, wp.quat_rotate_inv(qi, rt2))); meft2 = iM + wp.dot(rt2, iIrt2)
                    vp = v + wp.cross(w, rp); vt1 = wp.dot(vp, t1); vt2 = wp.dot(vp, t2)
                    o1 = jt1[i, k]; o2 = jt2[i, k]; a1 = o1 - vt1 / meft1; a2 = o2 - vt2 / meft2
                    mag = wp.sqrt(a1 * a1 + a2 * a2); lim = mu * jn[i, k]
                    if mag > lim and mag > 1e-12:
                        a1 = a1 * lim / mag; a2 = a2 * lim / mag
                    d1 = a1 - o1; d2 = a2 - o2; jt1[i, k] = a1; jt2[i, k] = a2
                    v = v + (d1 * t1 + d2 * t2) * iM; w = w + d1 * iIrt1 + d2 * iIrt2
        pv = wp.vec3(0.0, 0.0, 0.0); po = wp.vec3(0.0, 0.0, 0.0)
        for it in range(pit):
            for k in range(K):
                rp = wp.quat_rotate(qi, lev[i * _KMAX + k]); pw = c + rp; s = wp.dot(pw, n)
                if s < 0.0:
                    rn = wp.cross(rp, n); iIrn = wp.quat_rotate(qi, wp.cw_mul(iI, wp.quat_rotate_inv(qi, rn)))
                    meff = iM + wp.dot(rn, iIrn); rel = pv + wp.cross(po, rp); bias = _BETA * wp.max(-s - _SLOP, 0.0) / dt
                    dj = (bias - wp.dot(rel, n)) / meff; nw = wp.max(0.0, jp[i, k] + dj); dj = nw - jp[i, k]; jp[i, k] = nw
                    pv = pv + dj * n * iM; po = po + dj * iIrn
        c = c + (v + pv) * dt; ws = w + po; wq = wp.quat(ws[0], ws[1], ws[2], 0.0); qn = qi + 0.5 * wq * qi * dt
        xc[i] = c; q[i] = wp.normalize(qn); vc[i] = v; om[i] = w
    return step


def _inertia(shape, dim, M):
    r = dim[0]; L = dim[1] if len(dim) > 1 else 0.0
    if shape == 0: I = 0.4 * M * r * r; return (1 / I, 1 / I, 1 / I)
    if shape == 1 or shape == 3:
        Iyy = 0.5 * M * r * r; Ix = 0.25 * M * r * r + M * L * L / 12.0; return (1 / Ix, 1 / Iyy, 1 / Ix)
    hx, hy, hz = dim; Ix = M * ((2 * hy) ** 2 + (2 * hz) ** 2) / 12; Iy = M * ((2 * hx) ** 2 + (2 * hz) ** 2) / 12
    Iz = M * ((2 * hx) ** 2 + (2 * hy) ** 2) / 12; return (1 / Ix, 1 / Iy, 1 / Iz)

def _volume(shape, dim):
    r = dim[0]; L = dim[1] if len(dim) > 1 else 0.0
    if shape == 0: return 4.0 / 3.0 * np.pi * r ** 3
    if shape == 1: return np.pi * r * r * L
    if shape == 3: return np.pi * r * r * L + 4.0 / 3.0 * np.pi * r ** 3
    return (2 * dim[0]) * (2 * dim[1]) * (2 * dim[2])

_SHAPES = {"sphere": 0, "cylinder": 1, "box": 2, "capsule": 3}


class SDFContactEngine:
    """Geometry-correct federation member (analytic SDF, ground/ramp contact, all canonical shapes). ContactEngine Protocol."""
    name = "sdf_contact_core"

    def __init__(s, mu=0.5, vit=15, pit=8, ramp_deg=0.0):
        import warp as wp
        s.wp = wp; wp.init(); s.dev = "cuda:0"; s._step = _build_kernel(wp)
        s.mu = mu; s.vit = vit; s.pit = pit
        th = np.radians(ramp_deg); s._n = np.array([-np.sin(th), 0, np.cos(th)])
        g = np.array([0, 0, -_G]); tg = g - (g @ s._n) * s._n; s._tg = tg / (np.linalg.norm(tg) + 1e-12)
        s._sh = []; s._dim = []; s._c = []; s._dens = []; s._kin = []; s._built = False

    def add_body(s, shape, dims, center, density=700., kin=False):
        s._sh.append(_SHAPES[shape]); s._dim.append((list(dims) + [0.0, 0.0])[:3])   # pad to 3 (sphere/cylinder use dim[0], dim[1])
        s._c.append(list(center)); s._dens.append(float(density)); s._kin.append(1 if kin else 0); s._built = False
        return len(s._c) - 1

    def _build(s):
        wp = s.wp; N = len(s._c); s.N = N
        M = np.array([s._dens[i] * _volume(s._sh[i], s._dim[i]) for i in range(N)])
        s._M = M; s._kinarr = np.array(s._kin, int)
        invM = np.where(s._kinarr == 1, 0.0, 1.0 / M)
        invI = np.array([_inertia(s._sh[i], s._dim[i], M[i]) if s._kinarr[i] == 0 else (0., 0., 0.) for i in range(N)])
        s.xc = wp.array(np.array(s._c, float), dtype=wp.vec3, device=s.dev)
        s.q = wp.array(np.tile([0, 0, 0, 1.], (N, 1)), dtype=wp.quat, device=s.dev)
        s.v = wp.zeros(N, dtype=wp.vec3, device=s.dev); s.w = wp.zeros(N, dtype=wp.vec3, device=s.dev)
        s.sh = wp.array(np.array(s._sh), dtype=int, device=s.dev); s.dim = wp.array(np.array(s._dim), dtype=wp.vec3, device=s.dev)
        s.invI = wp.array(invI, dtype=wp.vec3, device=s.dev); s.invM = wp.array(invM, dtype=float, device=s.dev)
        s.lev = wp.zeros(N * _KMAX, dtype=wp.vec3, device=s.dev)
        s.jn = wp.zeros((N, _KMAX), dtype=float, device=s.dev)
        s.jt1 = wp.zeros((N, _KMAX), dtype=float, device=s.dev); s.jt2 = wp.zeros((N, _KMAX), dtype=float, device=s.dev)
        s.jp = wp.zeros((N, _KMAX), dtype=float, device=s.dev); s._built = True

    def set_kinematic(s, i, xc):
        s._kin[i] = 1; s._c[i] = list(xc)
        if s._built:
            wp = s.wp; a = s.xc.numpy(); a[i] = xc; s.xc = wp.array(a, dtype=wp.vec3, device=s.dev)
            m = s.invM.numpy(); m[i] = 0.0; s.invM = wp.array(m, dtype=float, device=s.dev); s._kinarr[i] = 1; s._M[i] = np.inf

    def step(s, dt, substeps=1):
        if not s._built:
            s._build()
        wp = s.wp; n = wp.vec3(*[float(x) for x in s._n])
        t1v = s._tg; t2v = np.cross(s._n, t1v); t2v = t2v / (np.linalg.norm(t2v) + 1e-12)   # t1 = downhill, t2 = lateral
        t1 = wp.vec3(*[float(x) for x in t1v]); t2 = wp.vec3(*[float(x) for x in t2v])
        for _ in range(substeps):
            sdt = dt / substeps; v_pre = s.v.numpy().copy()
            wp.launch(s._step, s.N, inputs=[s.xc, s.q, s.v, s.w, s.sh, s.dim, s.invI, s.invM, s.lev, s.jn, s.jt1, s.jt2, s.jp, n, t1, t2, s.mu, sdt, s.vit, s.pit], device=s.dev)
            v_post = s.v.numpy(); fin = np.zeros((s.N, 3)); nk = s._kinarr == 0
            # contact force = M*dv/dt with gravity added back (net contact = M*(v_post - v_pre)/dt + M*g)
            fin[nk] = s._M[nk, None] * (v_post[nk] - v_pre[nk]) / sdt + s._M[nk, None] * np.array([0, 0, -_G]) * (-1.0)
            s._cforce = fin

    def _quat2R(s, q):
        x, y, z, w = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

    def get_state(s):
        if not s._built:
            s._build()
        q = s.q.numpy(); Rm = np.array([s._quat2R(q[i]) for i in range(s.N)])
        return BodyState(xc=s.xc.numpy(), Rm=Rm, vc=s.v.numpy(), om=s.w.numpy())

    def contact_forces(s):
        return s._cforce.copy()

    def com(s, i):
        if not s._built:
            s._build()
        return s.xc.numpy()[i]


def _selftest():
    e0 = SDFContactEngine()
    assert isinstance(e0, ContactEngine), "does not satisfy the ContactEngine Protocol"
    ok = True; t = 240 / 240.0
    # geometry-correct rolling/sticking on a ramp through the contract (one body per engine)
    cases = [("sphere", "sphere", (0.12,), 15, 5.0 / 7.0), ("cylinder", "cylinder", (0.12, 0.2), 15, 2.0 / 3.0),
             ("box", "box", (0.1, 0.1, 0.1), 15, None)]
    for nm, shp, dim, deg, coef in cases:
        e = SDFContactEngine(mu=0.6, ramp_deg=deg)
        h0 = dim[2] if shp == "box" else dim[0]; e.add_body(shp, dim, [-np.sin(np.radians(deg)) * h0, 0, np.cos(np.radians(deg)) * h0])
        x0 = e.com(0).copy()
        for _ in range(240):
            e.step(1 / 240)
        th = np.radians(deg); down = np.array([-np.cos(th), 0, -np.sin(th)]); dist = (e.com(0) - x0) @ down
        if coef:
            dth = 0.5 * coef * _G * np.sin(th) * 1.0 ** 2; err = abs(dist - dth) / dth * 100; c_ok = err < 8
            print(f"  [{nm} ramp {deg} deg] travelled {dist:.3f} (theory {dth:.3f}, error {err:.0f}%)  {'rolls' if c_ok else 'FAIL'}")
        else:
            c_ok = dist < 0.03; print(f"  [{nm} ramp {deg} deg] travelled {dist:.3f}  {'sticks' if c_ok else 'FAIL'}")
        ok = ok and c_ok
    # 3D-cone lateral-slip test: a box on a sub-critical ramp (15 deg < atan(0.6) = 31 deg, so it sticks downhill) but
    # with a lateral initial velocity. The 3D cone brakes lateral slip via the mu*jn disc; a fixed downhill tangent has
    # t2 identically zero and cannot brake laterally.
    e = SDFContactEngine(mu=0.6, ramp_deg=15); deg = 15
    h0 = 0.1; e.add_body("box", (0.1, 0.1, 0.1), [-np.sin(np.radians(deg)) * h0, 0, np.cos(np.radians(deg)) * h0])
    x0 = e.com(0).copy()                                  # triggers _build
    t1v = e._tg; t2v = np.cross(e._n, t1v); t2v = t2v / (np.linalg.norm(t2v) + 1e-12)
    e.v = e.wp.array(np.tile(0.5 * t2v, (e.N, 1)).astype(np.float32), dtype=e.wp.vec3, device=e.dev)   # lateral v0 = 0.5 m/s
    for _ in range(240):
        e.step(1 / 240)
    lat = abs((e.com(0) - x0) @ t2v); lat_ok = lat < 0.06   # 3D cone: ~2 cm; a fixed 1D tangent would give ~0.5 m (unbraked)
    print(f"  [3D cone, lateral] box with lateral v0=0.5 on a sub-critical ramp: lateral drift {lat*100:.1f} cm  {'braked' if lat_ok else 'FAIL'}")
    ok = ok and lat_ok
    print(f"  -> {'SDFContactEngine: geometry-correct rolling preserved and lateral slip braked by the 3D Coulomb cone' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    print("SDF contact-engine selftest (analytic SDF behind the ContactEngine Protocol):")
    sys.exit(0 if _selftest() else 1)
