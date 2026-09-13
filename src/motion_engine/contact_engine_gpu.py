#!/usr/bin/env python3
"""GPU contact engine (federation member) behind the same ContactEngine Protocol as the CPU engine.

Relaxed Jacobi with atomic double buffering: read v_old, accumulate deltas via atomics, then
v += relax * delta. Naive Jacobi diverges; the relaxed, double-buffered form is stable for deep coupling
(a K=8 tower). Per-body invM/invIw, so kinematic bodies (invM=0) are position-driven. GPU broad phase
(hash grid). Contact force = M*(v_after - v_before_contact)/dt, which is robust to the relaxation (a box
at rest gives Mg). Measured: K=8 tower stable, contact force = Mg at 0% error, 2.42x real time at N=125,
1% Coulomb drift.

Per-body deltas are accumulated in int64 fixed point (scale 1e10, float64 rounding) instead of float32 atomics:
integer addition is associative, so the sum is the same for every contact ordering and two runs are bit-identical
on the GPU. Adopted from the int64 fixed-point accumulation probe; the box3d bitset-order mechanism is the
atomics-free alternative and is not used here.

Warp is imported lazily: the module imports without warp and only raises at instantiation. Runs on the CPU
backend when no CUDA device is present (slow, for tests).

  python -u -m motion_engine.contact_engine_gpu        # contract selftest (requires warp + CUDA)
"""
from dataclasses import dataclass
import numpy as np
from motion_engine.contact_engine import BodyState, ContactEngine  # same data contract as the CPU engine

_G = 9.81; _R = 0.05; _BETA = 0.2; _SLOP = 1e-4; _MAXC = 400000
_S = 1.0e10  # fixed-point scale for int64 delta accumulation (|delta| < 9e8 fits int64)


def _build_kernels(wp):
    """Build the warp kernels (called at instantiation so importing the module does not require warp)."""
    @wp.func
    def _acc(ax: wp.array(dtype=wp.int64), ay: wp.array(dtype=wp.int64), az: wp.array(dtype=wp.int64),
             b: int, d: wp.vec3, sign: float, scale: float):
        # order-invariant accumulation: round(delta * scale) in float64, then int64 atomics
        sc = wp.float64(sign * scale)
        wp.atomic_add(ax, b, wp.int64(wp.round(wp.float64(d[0]) * sc)))
        wp.atomic_add(ay, b, wp.int64(wp.round(wp.float64(d[1]) * sc)))
        wp.atomic_add(az, b, wp.int64(wp.round(wp.float64(d[2]) * sc)))

    @wp.kernel
    def k_grav(v: wp.array(dtype=wp.vec3), invM: wp.array(dtype=float), dt: float):
        i = wp.tid()
        if invM[i] > 0.0:
            v[i] = v[i] + wp.vec3(0.0, 0.0, -_G) * dt

    @wp.kernel
    def k_invIw(q: wp.array(dtype=wp.quat), IbInv: wp.array(dtype=wp.mat33), kin: wp.array(dtype=int),
                out: wp.array(dtype=wp.mat33)):
        i = wp.tid()
        if kin[i] == 1:
            out[i] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        else:
            Rm = wp.quat_to_matrix(q[i]); out[i] = Rm * IbInv[i] * wp.transpose(Rm)

    @wp.kernel
    def k_worldp(xc: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.quat), rest: wp.array(dtype=wp.vec3), P: int,
                 allp: wp.array(dtype=wp.vec3), owner: wp.array(dtype=int)):
        gid = wp.tid(); bi = gid // P
        allp[gid] = xc[bi] + wp.quat_rotate(q[bi], rest[gid % P]); owner[gid] = bi

    @wp.kernel
    def k_gen(allp: wp.array(dtype=wp.vec3), owner: wp.array(dtype=int), grid: wp.uint64, cnt: wp.array(dtype=int),
              cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3),
              cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3), cpen: wp.array(dtype=float)):
        gid = wp.tid(); xi = allp[gid]; bi = owner[gid]; s = xi[2] - _R
        if s < 0.0:
            idx = wp.atomic_add(cnt, 0, 1)
            if idx < _MAXC:
                cbi[idx] = bi; cbj[idx] = -1; cpA[idx] = xi; cpB[idx] = wp.vec3(0.0, 0.0, 0.0)
                cn[idx] = wp.vec3(0.0, 0.0, 1.0); cpen[idx] = -s
        qy = wp.hash_grid_query(grid, xi, 2.0 * _R); j = int(0)
        while wp.hash_grid_query_next(qy, j):
            if j > gid and owner[j] != bi:
                dvec = xi - allp[j]; dist = wp.length(dvec)
                if dist < 2.0 * _R and dist > 1e-9:
                    idx = wp.atomic_add(cnt, 0, 1)
                    if idx < _MAXC:
                        cbi[idx] = bi; cbj[idx] = owner[j]; cpA[idx] = xi; cpB[idx] = allp[j]
                        cn[idx] = dvec / dist; cpen[idx] = 2.0 * _R - dist

    @wp.kernel
    def k_jac_vel(C: int, cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3),
                  cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3), jn: wp.array(dtype=float),
                  jt1: wp.array(dtype=float), jt2: wp.array(dtype=float), xc: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                  w: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33), invM: wp.array(dtype=float),
                  mu: float, scale: float,
                  dvx: wp.array(dtype=wp.int64), dvy: wp.array(dtype=wp.int64), dvz: wp.array(dtype=wp.int64),
                  dwx: wp.array(dtype=wp.int64), dwy: wp.array(dtype=wp.int64), dwz: wp.array(dtype=wp.int64)):
        c = wp.tid()
        if c >= C:
            return
        bi = cbi[c]; bj = cbj[c]; nrm = cn[c]; rA = cpA[c] - xc[bi]
        iMA = invM[bi]; IA = invIw[bi]; va = v[bi] + wp.cross(w[bi], rA)
        vb = wp.vec3(0.0, 0.0, 0.0); rB = wp.vec3(0.0, 0.0, 0.0); iMB = float(0.0)
        if bj >= 0:
            rB = cpB[c] - xc[bj]; vb = v[bj] + wp.cross(w[bj], rB); iMB = invM[bj]
        # 3D cone on a slip-direction basis: t1 = slip (dissipative, hence relaxed-Jacobi stable), t2 perpendicular,
        # giving a circular mu*jn disc, i.e. an isotropic cone
        vrel = va - vb; vn = wp.dot(vrel, nrm); vt = vrel - vn * nrm; mt = wp.length(vt)
        t1 = vt / mt
        if mt <= 5e-3:
            aa = wp.vec3(1.0, 0.0, 0.0)
            if wp.abs(nrm[0]) >= 0.9:
                aa = wp.vec3(0.0, 1.0, 0.0)
            t1 = wp.normalize(aa - wp.dot(aa, nrm) * nrm)
        t2 = wp.cross(nrm, t1)
        # NORMAL (ackumulerad, clamp ≥0)
        rnA = wp.cross(rA, nrm); meff = iMA + wp.dot(rnA, IA * rnA)
        if bj >= 0:
            rnB = wp.cross(rB, nrm); meff = meff + iMB + wp.dot(rnB, invIw[bj] * rnB)
        if meff < 1e-12:
            return
        dj = -vn / meff; nw = wp.max(0.0, jn[c] + dj); dj = nw - jn[c]; jn[c] = nw; Jn = dj * nrm
        _acc(dvx, dvy, dvz, bi, Jn * iMA, 1.0, scale); _acc(dwx, dwy, dwz, bi, IA * wp.cross(rA, Jn), 1.0, scale)
        if bj >= 0:
            _acc(dvx, dvy, dvz, bj, Jn * iMB, -1.0, scale); _acc(dwx, dwy, dwz, bj, invIw[bj] * wp.cross(rB, Jn), -1.0, scale)
        # friction 3D cone: solve t1 and t2 (own effective masses) and clamp the VECTOR to the circular mu*jn disc
        rtA1 = wp.cross(rA, t1); meft1 = iMA + wp.dot(rtA1, IA * rtA1)
        rtA2 = wp.cross(rA, t2); meft2 = iMA + wp.dot(rtA2, IA * rtA2)
        if bj >= 0:
            rtB1 = wp.cross(rB, t1); meft1 = meft1 + iMB + wp.dot(rtB1, invIw[bj] * rtB1)
            rtB2 = wp.cross(rB, t2); meft2 = meft2 + iMB + wp.dot(rtB2, invIw[bj] * rtB2)
        o1 = jt1[c]; o2 = jt2[c]; a1 = o1 - wp.dot(vrel, t1) / meft1; a2 = o2 - wp.dot(vrel, t2) / meft2
        mag = wp.sqrt(a1 * a1 + a2 * a2); lim = mu * jn[c]
        if mag > lim and mag > 1e-12:
            a1 = a1 * lim / mag; a2 = a2 * lim / mag
        # the relaxation is tuned for one friction DOF, so the cross-slip component (t2) is damped 0.5x for energy stability
        jt1[c] = a1; jt2[c] = a2; Jt = (a1 - o1) * t1 + 0.5 * (a2 - o2) * t2
        _acc(dvx, dvy, dvz, bi, Jt * iMA, 1.0, scale); _acc(dwx, dwy, dwz, bi, IA * wp.cross(rA, Jt), 1.0, scale)
        if bj >= 0:
            _acc(dvx, dvy, dvz, bj, Jt * iMB, -1.0, scale); _acc(dwx, dwy, dwz, bj, invIw[bj] * wp.cross(rB, Jt), -1.0, scale)

    @wp.kernel
    def k_apply(v: wp.array(dtype=wp.vec3), w: wp.array(dtype=wp.vec3), relax: float, scale: float,
                dvx: wp.array(dtype=wp.int64), dvy: wp.array(dtype=wp.int64), dvz: wp.array(dtype=wp.int64),
                dwx: wp.array(dtype=wp.int64), dwy: wp.array(dtype=wp.int64), dwz: wp.array(dtype=wp.int64)):
        i = wp.tid(); inv = 1.0 / scale
        dv = wp.vec3(float(wp.float64(dvx[i]) * wp.float64(inv)), float(wp.float64(dvy[i]) * wp.float64(inv)),
                     float(wp.float64(dvz[i]) * wp.float64(inv)))
        dw = wp.vec3(float(wp.float64(dwx[i]) * wp.float64(inv)), float(wp.float64(dwy[i]) * wp.float64(inv)),
                     float(wp.float64(dwz[i]) * wp.float64(inv)))
        v[i] = v[i] + relax * dv; w[i] = w[i] + relax * dw
        z = wp.int64(0); dvx[i] = z; dvy[i] = z; dvz[i] = z; dwx[i] = z; dwy[i] = z; dwz[i] = z

    @wp.kernel
    def k_jac_pos(C: int, cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3),
                  cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3), cpen: wp.array(dtype=float),
                  jp: wp.array(dtype=float), xc: wp.array(dtype=wp.vec3), pv: wp.array(dtype=wp.vec3),
                  po: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33), invM: wp.array(dtype=float), dt: float,
                  scale: float,
                  dvx: wp.array(dtype=wp.int64), dvy: wp.array(dtype=wp.int64), dvz: wp.array(dtype=wp.int64),
                  dwx: wp.array(dtype=wp.int64), dwy: wp.array(dtype=wp.int64), dwz: wp.array(dtype=wp.int64)):
        c = wp.tid()
        if c >= C:
            return
        bi = cbi[c]; bj = cbj[c]; nrm = cn[c]; rA = cpA[c] - xc[bi]; iMA = invM[bi]; IA = invIw[bi]
        rel = pv[bi] + wp.cross(po[bi], rA); rB = wp.vec3(0.0, 0.0, 0.0); iMB = float(0.0)
        if bj >= 0:
            rB = cpB[c] - xc[bj]; rel = rel - (pv[bj] + wp.cross(po[bj], rB)); iMB = invM[bj]
        rnA = wp.cross(rA, nrm); meff = iMA + wp.dot(rnA, IA * rnA)
        if bj >= 0:
            rnB = wp.cross(rB, nrm); meff = meff + iMB + wp.dot(rnB, invIw[bj] * rnB)
        if meff < 1e-12:
            return
        bias = _BETA * wp.max(cpen[c] - _SLOP, 0.0) / dt; dj = (bias - wp.dot(rel, nrm)) / meff
        nw = wp.max(0.0, jp[c] + dj); dj = nw - jp[c]; jp[c] = nw; Jn = dj * nrm
        _acc(dvx, dvy, dvz, bi, Jn * iMA, 1.0, scale); _acc(dwx, dwy, dwz, bi, IA * wp.cross(rA, Jn), 1.0, scale)
        if bj >= 0:
            _acc(dvx, dvy, dvz, bj, Jn * iMB, -1.0, scale); _acc(dwx, dwy, dwz, bj, invIw[bj] * wp.cross(rB, Jn), -1.0, scale)

    @wp.kernel
    def k_integ(xc: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.quat), v: wp.array(dtype=wp.vec3),
                w: wp.array(dtype=wp.vec3), pv: wp.array(dtype=wp.vec3), po: wp.array(dtype=wp.vec3),
                invM: wp.array(dtype=float), dt: float):
        i = wp.tid()
        if invM[i] <= 0.0:        # kinematic: position driven externally, do not integrate
            return
        xc[i] = xc[i] + (v[i] + pv[i]) * dt
        ww = w[i] + po[i]; wq = wp.quat(ww[0], ww[1], ww[2], 0.0); qn = q[i] + 0.5 * wq * q[i] * dt
        q[i] = wp.normalize(qn)

    return dict(grav=k_grav, invIw=k_invIw, worldp=k_worldp, gen=k_gen, jac_vel=k_jac_vel,
                apply=k_apply, jac_pos=k_jac_pos, integ=k_integ)


def _voxbox(w, h, d, res=_R * 2):
    a = lambda L: np.arange(-L / 2 + res / 2, L / 2, res)
    return np.array([[x, y, z] for z in a(d) for y in a(h) for x in a(w)], float)


class RelaxedJacobiContactEngine:
    """GPU federation member (relaxed Jacobi, fixed-point accumulation). Satisfies the ContactEngine Protocol. Boxes, GPU broad phase, deep coupling."""
    name = "relaxed_jacobi_gpu"

    def __init__(s, dims=(0.3, 0.3, 0.2), relax=0.2, vit=40, pit=20, mu=0.5):
        import warp as wp                                  # lazy: warp + CUDA are only needed at instantiation
        s.wp = wp; wp.init(); s.dev = "cuda:0" if wp.is_cuda_available() else "cpu"
        s.K = _build_kernels(wp)
        s.dims = dims; s.rest_np = _voxbox(*dims); s.P = len(s.rest_np)
        s.relax = relax; s.vit = vit; s.pit = pit; s.mu = mu
        s._centers = []; s._dens = []; s._kin = []
        s._built = False

    # ── construction (data in) ──
    def add_body(s, center, density=700., kin=False):
        s._centers.append(list(center)); s._dens.append(float(density)); s._kin.append(1 if kin else 0)
        s._built = False; return len(s._centers) - 1

    def _build(s):
        wp = s.wp; N = len(s._centers); w, h, d = s.dims; vol = w * h * d
        s.N = N; mp = np.array(s._dens) * vol / s.P; M = mp * s.P
        s._M = M; s._kinarr = np.array(s._kin, int)
        invM = np.where(s._kinarr == 1, 0.0, 1.0 / M)
        Ib = sum(mp[0] * ((r @ r) * np.eye(3) - np.outer(r, r)) for r in s.rest_np)  # shape inertia (per unit point mass)
        IbInv = np.array([np.linalg.inv((mp[i] / mp[0]) * Ib) if s._kinarr[i] == 0 else np.eye(3) for i in range(N)])
        # Same class of issue as _Body.Iwi() in contact_engine.py: np.linalg.inv does not sanitise NaN, it returns a
        # partially-NaN inverse for a degenerate or corrupt (mass, inertia) pair instead of raising. This is computed
        # once in _build() and persists for the whole GPU-batched simulation, so a corrupt body would silently poison
        # every later step.
        if not np.all(np.isfinite(IbInv)):
            raise ValueError("contact_engine_gpu._build: IbInv contains non-finite (NaN/inf) entries -- "
                             "degenerate body mass/inertia; refusing to build a corrupted rigid-body sim")
        s.xc = wp.array(np.array(s._centers, float), dtype=wp.vec3, device=s.dev)
        s.q = wp.array(np.tile([0, 0, 0, 1.], (N, 1)), dtype=wp.quat, device=s.dev)
        s.v = wp.zeros(N, dtype=wp.vec3, device=s.dev); s.w = wp.zeros(N, dtype=wp.vec3, device=s.dev)
        s.invM = wp.array(invM, dtype=float, device=s.dev)
        s.IbInv = wp.array(IbInv, dtype=wp.mat33, device=s.dev); s.invIw = wp.zeros(N, dtype=wp.mat33, device=s.dev)
        s.rest = wp.array(s.rest_np, dtype=wp.vec3, device=s.dev)
        s.allp = wp.zeros(N * s.P, dtype=wp.vec3, device=s.dev); s.owner = wp.zeros(N * s.P, dtype=int, device=s.dev)
        s.grid = wp.HashGrid(64, 64, 64, device=s.dev); s.cnt = wp.zeros(1, dtype=int, device=s.dev)
        s.cbi = wp.zeros(_MAXC, dtype=int, device=s.dev); s.cbj = wp.zeros(_MAXC, dtype=int, device=s.dev)
        s.cpA = wp.zeros(_MAXC, dtype=wp.vec3, device=s.dev); s.cpB = wp.zeros(_MAXC, dtype=wp.vec3, device=s.dev)
        s.cn = wp.zeros(_MAXC, dtype=wp.vec3, device=s.dev); s.cpen = wp.zeros(_MAXC, dtype=float, device=s.dev)
        s.jn = wp.zeros(_MAXC, dtype=float, device=s.dev)
        s.jt1 = wp.zeros(_MAXC, dtype=float, device=s.dev); s.jt2 = wp.zeros(_MAXC, dtype=float, device=s.dev)
        s.jp = wp.zeros(_MAXC, dtype=float, device=s.dev)
        s.acc = [wp.zeros(N, dtype=wp.int64, device=s.dev) for _ in range(6)]   # dvx dvy dvz dwx dwy dwz, fixed point
        s.pv = wp.zeros(N, dtype=wp.vec3, device=s.dev); s.po = wp.zeros(N, dtype=wp.vec3, device=s.dev)
        s._cforce = np.zeros((N, 3)); s._built = True

    def set_kinematic(s, i, xc):
        s._kin[i] = 1; s._centers[i] = list(xc)
        if s._built:                                       # live-update the GPU state (kinematic body: driven position, zero velocity, invM=0)
            wp = s.wp
            a = s.xc.numpy(); a[i] = xc; s.xc = wp.array(a, dtype=wp.vec3, device=s.dev)
            m = s.invM.numpy(); m[i] = 0.0; s.invM = wp.array(m, dtype=float, device=s.dev)
            vv = s.v.numpy(); vv[i] = [0.0, 0.0, 0.0]; s.v = wp.array(vv, dtype=wp.vec3, device=s.dev)
            ww = s.w.numpy(); ww[i] = [0.0, 0.0, 0.0]; s.w = wp.array(ww, dtype=wp.vec3, device=s.dev)
            s._kinarr[i] = 1; s._M[i] = np.inf

    # ── avancera (DATA) ──
    def step(s, dt, substeps=1):
        if not s._built:
            s._build()
        wp = s.wp; K = s.K
        for _ in range(substeps):
            sdt = dt / substeps
            v_pre = s.v.numpy().copy()                       # for contact force = M * delta v_contact / dt
            wp.launch(K["grav"], s.N, inputs=[s.v, s.invM, sdt], device=s.dev)
            wp.launch(K["invIw"], s.N, inputs=[s.q, s.IbInv, wp.array(s._kinarr, dtype=int, device=s.dev), s.invIw], device=s.dev)
            wp.launch(K["worldp"], s.N * s.P, inputs=[s.xc, s.q, s.rest, s.P, s.allp, s.owner], device=s.dev)
            s.grid.build(s.allp, 2.0 * _R); s.cnt.zero_()
            wp.launch(K["gen"], s.N * s.P, inputs=[s.allp, s.owner, s.grid.id, s.cnt, s.cbi, s.cbj, s.cpA, s.cpB, s.cn, s.cpen], device=s.dev)
            C = int(s.cnt.numpy()[0]); C = min(C, _MAXC)
            v_grav = s.v.numpy().copy()
            if C > 0:
                s.jn.zero_(); s.jt1.zero_(); s.jt2.zero_(); s.jp.zero_(); s.pv.zero_(); s.po.zero_()
                for a in s.acc:
                    a.zero_()
                for _ in range(s.vit):
                    wp.launch(K["jac_vel"], C, inputs=[C, s.cbi, s.cbj, s.cpA, s.cpB, s.cn, s.jn, s.jt1, s.jt2, s.xc, s.v, s.w, s.invIw, s.invM, s.mu, _S, *s.acc], device=s.dev)
                    wp.launch(K["apply"], s.N, inputs=[s.v, s.w, s.relax, _S, *s.acc], device=s.dev)
                for _ in range(s.pit):
                    wp.launch(K["jac_pos"], C, inputs=[C, s.cbi, s.cbj, s.cpA, s.cpB, s.cn, s.cpen, s.jp, s.xc, s.pv, s.po, s.invIw, s.invM, sdt, _S, *s.acc], device=s.dev)
                    wp.launch(K["apply"], s.N, inputs=[s.pv, s.po, s.relax, _S, *s.acc], device=s.dev)
            wp.launch(K["integ"], s.N, inputs=[s.xc, s.q, s.v, s.w, s.pv, s.po, s.invM, sdt], device=s.dev)
            v_post = s.v.numpy()
            fin = np.zeros((s.N, 3)); nk = s._kinarr == 0          # net contact force per body (robust to relax); kinematic = 0 (driven)
            fin[nk] = s._M[nk, None] * (v_post[nk] - v_grav[nk]) / sdt
            s._cforce = fin

    # ── data out ──
    def _quat2R(s, q):
        x, y, z, w = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

    def get_state(s):
        if not s._built:
            s._build()
        xc = s.xc.numpy(); q = s.q.numpy(); Rm = np.array([s._quat2R(q[i]) for i in range(s.N)])
        return BodyState(xc=xc, Rm=Rm, vc=s.v.numpy(), om=s.w.numpy())

    def contact_forces(s):
        return s._cforce.copy()

    def com(s, i):
        return s.xc.numpy()[i]


# ───────────────────────── contract selftest ─────────────────────────
def _selftest():
    e = RelaxedJacobiContactEngine()
    assert isinstance(e, ContactEngine), "does not satisfy the ContactEngine Protocol"
    ok = True
    # 1) deep stack of K=8 stable through the contract
    for k in range(8):
        e.add_body([0, 0, 0.101 + k * 0.205])
    for _ in range(400):
        e.step(1 / 240)
    st = e.get_state(); ke = float(np.sum(0.5 * e._M[:, None] * st.vc ** 2))
    zs = sorted(st.xc[:, 2]); seps = [zs[i + 1] - zs[i] for i in range(7)]
    s_ok = ke < 0.5 and all(0.15 < sp < 0.25 for sp in seps)
    print(f"  [stack K=8] KE={ke:.3f} sep {min(seps):.3f}-{max(seps):.3f}  {'stable' if s_ok else 'FAIL'}"); ok = ok and s_ok
    # 2) contact force on a box at rest ~= Mg (through the contract)
    e2 = RelaxedJacobiContactEngine(); e2.add_body([0, 0, 0.101])
    for _ in range(200):
        e2.step(1 / 240)
    Fz = e2.contact_forces()[0, 2]; Mg = e2._M[0] * _G; err = abs(Fz - Mg) / Mg * 100
    f_ok = err < 8
    print(f"  [contact force, box at rest] Fz={Fz:.2f} Mg={Mg:.2f} error={err:.1f}%  {'ok' if f_ok else 'FAIL'}"); ok = ok and f_ok
    # 3) set_kinematic: the body stays at the commanded position (does not fall) and can be moved live
    e3 = RelaxedJacobiContactEngine(); e3.add_body([0, 0, 0.5]); e3.set_kinematic(0, [0, 0, 0.5])
    for _ in range(120):
        e3.step(1 / 240)
    z_stay = e3.get_state().xc[0, 2]
    e3.set_kinematic(0, [0.1, 0, 0.5]); e3.step(1 / 240)             # live move (after build)
    x_moved = e3.get_state().xc[0, 0]
    k_ok = abs(z_stay - 0.5) < 1e-6 and abs(x_moved - 0.1) < 1e-6
    print(f"  [set_kinematic] stays z={z_stay:.4f} + live move x={x_moved:.4f}  {'ok' if k_ok else 'FAIL'}"); ok = ok and k_ok
    # 4) determinism: int64 fixed-point accumulation makes two runs bit-identical; on CUDA the gate requires exactly 0
    def run_stack4():
        e = RelaxedJacobiContactEngine()
        for k in range(4):
            e.add_body([0, 0, 0.101 + k * 0.205])
        for _ in range(200):
            e.step(1 / 240)
        return e.get_state().xc
    dmax = float(np.max(np.abs(run_stack4() - run_stack4())))
    det_ok = dmax == 0.0
    print(f"  [determinism] max |delta xc| over two runs = {dmax:.2e} m  device={e.dev}  "
          f"{'bit-identical' if det_ok else 'FAIL: runs differ'}"); ok = ok and det_ok
    print(f"  -> {'GPU ContactEngine: all four contract checks pass' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    print("GPU contact engine selftest (relaxed Jacobi, int64 fixed-point accumulation, behind the ContactEngine Protocol):")
    sys.exit(0 if _selftest() else 1)
