"""Separate solver candidates with explicit friction-dot rounding boundaries."""

import warp as wp

@wp.func_native('return __fadd_rn(__fadd_rn(__fmul_rn(a[0], b[0]), __fmul_rn(a[1], b[1])), __fmul_rn(a[2], b[2]));')
def friction_dot(a:wp.vec3,b:wp.vec3)->float: ...

@wp.kernel(module='unique', module_options={'enable_backward': False})
def ordered_pair(porder: wp.array(dtype=int), pidx: wp.array(dtype=int), pstart: wp.array(dtype=int), pcount: wp.array(dtype=int), pbi: wp.array(dtype=int), pbj: wp.array(dtype=int), ci: int, cstart: wp.array(dtype=int), ccount: wp.array(dtype=int), cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3), cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3), jn: wp.array(dtype=float), jt1: wp.array(dtype=float), jt2: wp.array(dtype=float), xc: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), w: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33), invM: wp.array(dtype=float), mu: float, t2d: float):
    """The same chunk, chunk_mode='registers': the two bodies of the pair are loaded once, every contact
        of the chunk updates the local velocities, and they are stored once at the end. Expression for
        expression and in the same order as `f_gs_vel`, so this is the exact sequential Gauss-Seidel result
        of the chunk (the WY compression of the updates does not apply here: each of the <= 4 updates is
        state dependent through max(0, .) on the normal impulse and the projection onto the friction cone,
        so they are not a fixed product of rank-1 projectors)."""
    t = wp.tid()
    if t >= ccount[ci]:
        return
    p = porder[cstart[ci] + t]
    st = pstart[p]
    n = pcount[p]
    bi = pbi[p]
    bj = pbj[p]
    iMA = invM[bi]
    IA = invIw[bi]
    vA = v[bi]
    wA = w[bi]
    iMB = float(0.0)
    IB = invIw[bi]
    vB = wp.vec3(0.0, 0.0, 0.0)
    wB = wp.vec3(0.0, 0.0, 0.0)
    if bj >= 0:
        iMB = invM[bj]
        IB = invIw[bj]
        vB = v[bj]
        wB = w[bj]
    for k in range(n):
        c = pidx[st + k]
        nrm = cn[c]
        rA = cpA[c] - xc[bi]
        va = vA + wp.cross(wA, rA)
        vb = wp.vec3(0.0, 0.0, 0.0)
        rB = wp.vec3(0.0, 0.0, 0.0)
        if bj >= 0:
            rB = cpB[c] - xc[bj]
            vb = vB + wp.cross(wB, rB)
        vrel = va - vb
        vn = wp.dot(vrel, nrm)
        vt = vrel - vn * nrm
        mt = wp.length(vt)
        t1 = vt / mt
        if mt <= 0.005:
            aa = wp.vec3(1.0, 0.0, 0.0)
            if wp.abs(nrm[0]) >= 0.9:
                aa = wp.vec3(0.0, 1.0, 0.0)
            t1 = wp.normalize(aa - wp.dot(aa, nrm) * nrm)
        t2 = wp.cross(nrm, t1)
        rnA = wp.cross(rA, nrm)
        meff = iMA + wp.dot(rnA, IA * rnA)
        if bj >= 0:
            rnB = wp.cross(rB, nrm)
            meff = meff + iMB + wp.dot(rnB, IB * rnB)
        if meff >= 1e-12:
            dj = -vn / meff
            nw = wp.max(0.0, jn[c] + dj)
            dj = nw - jn[c]
            jn[c] = nw
            Jn = dj * nrm
            if iMA > 0.0:
                vA = vA + Jn * iMA
                wA = wA + IA * wp.cross(rA, Jn)
            if bj >= 0:
                if iMB > 0.0:
                    vB = vB - Jn * iMB
                    wB = wB - IB * wp.cross(rB, Jn)
            va = vA + wp.cross(wA, rA)
            vb = wp.vec3(0.0, 0.0, 0.0)
            if bj >= 0:
                vb = vB + wp.cross(wB, rB)
            vrel = va - vb
            rtA1 = wp.cross(rA, t1)
            meft1 = iMA + wp.dot(rtA1, IA * rtA1)
            rtA2 = wp.cross(rA, t2)
            meft2 = iMA + wp.dot(rtA2, IA * rtA2)
            if bj >= 0:
                rtB1 = wp.cross(rB, t1)
                meft1 = meft1 + iMB + wp.dot(rtB1, IB * rtB1)
                rtB2 = wp.cross(rB, t2)
                meft2 = meft2 + iMB + wp.dot(rtB2, IB * rtB2)
            if meft1 >= 1e-12:
                if meft2 >= 1e-12:
                    o1 = jt1[c]
                    o2 = jt2[c]
                    a1 = o1 - friction_dot(vrel, t1) / meft1
                    a2 = o2 - friction_dot(vrel, t2) / meft2
                    mag = wp.sqrt(a1 * a1 + a2 * a2)
                    lim = mu * jn[c]
                    if mag > lim and mag > 1e-12:
                        a1 = a1 * lim / mag
                        a2 = a2 * lim / mag
                    jt1[c] = a1
                    jt2[c] = a2
                    Jt = (a1 - o1) * t1 + t2d * (a2 - o2) * t2
                    if iMA > 0.0:
                        vA = vA + Jt * iMA
                        wA = wA + IA * wp.cross(rA, Jt)
                    if bj >= 0:
                        if iMB > 0.0:
                            vB = vB - Jt * iMB
                            wB = wB - IB * wp.cross(rB, Jt)
    if iMA > 0.0:
        v[bi] = vA
        w[bi] = wA
    if bj >= 0:
        if iMB > 0.0:
            v[bj] = vB
            w[bj] = wB

@wp.kernel(module='unique', module_options={'enable_backward': False})
def ordered_fixed(porder: wp.array(dtype=int), pidx: wp.array(dtype=int), pstart: wp.array(dtype=int), pcount: wp.array(dtype=int), pbi: wp.array(dtype=int), pbj: wp.array(dtype=int), ci: int, cstart: wp.array(dtype=int), ccount: wp.array(dtype=int), cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3), cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3), jn: wp.array(dtype=float), jt1: wp.array(dtype=float), jt2: wp.array(dtype=float), xc: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), w: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33), invM: wp.array(dtype=float), mu: float, t2d: float):
    """Sequential pair chunk with normal-derived tangent coordinates."""
    t = wp.tid()
    if t >= ccount[ci]:
        return
    p = porder[cstart[ci] + t]
    st = pstart[p]
    n = pcount[p]
    bi = pbi[p]
    bj = pbj[p]
    iMA = invM[bi]
    IA = invIw[bi]
    vA = v[bi]
    wA = w[bi]
    iMB = float(0.0)
    IB = invIw[bi]
    vB = wp.vec3(0.0, 0.0, 0.0)
    wB = wp.vec3(0.0, 0.0, 0.0)
    if bj >= 0:
        iMB = invM[bj]
        IB = invIw[bj]
        vB = v[bj]
        wB = w[bj]
    for k in range(n):
        c = pidx[st + k]
        nrm = cn[c]
        rA = cpA[c] - xc[bi]
        va = vA + wp.cross(wA, rA)
        vb = wp.vec3(0.0, 0.0, 0.0)
        rB = wp.vec3(0.0, 0.0, 0.0)
        if bj >= 0:
            rB = cpB[c] - xc[bj]
            vb = vB + wp.cross(wB, rB)
        vrel = va - vb
        vn = wp.dot(vrel, nrm)
        vt = vrel - vn * nrm
        mt = wp.length(vt)
        aa = wp.vec3(1.0, 0.0, 0.0)
        if wp.abs(nrm[0]) >= 0.9:
            aa = wp.vec3(0.0, 1.0, 0.0)
        t1 = wp.normalize(aa - wp.dot(aa, nrm) * nrm)
        t2 = wp.cross(nrm, t1)
        rnA = wp.cross(rA, nrm)
        meff = iMA + wp.dot(rnA, IA * rnA)
        if bj >= 0:
            rnB = wp.cross(rB, nrm)
            meff = meff + iMB + wp.dot(rnB, IB * rnB)
        if meff >= 1e-12:
            dj = -vn / meff
            nw = wp.max(0.0, jn[c] + dj)
            dj = nw - jn[c]
            jn[c] = nw
            Jn = dj * nrm
            if iMA > 0.0:
                vA = vA + Jn * iMA
                wA = wA + IA * wp.cross(rA, Jn)
            if bj >= 0:
                if iMB > 0.0:
                    vB = vB - Jn * iMB
                    wB = wB - IB * wp.cross(rB, Jn)
            va = vA + wp.cross(wA, rA)
            vb = wp.vec3(0.0, 0.0, 0.0)
            if bj >= 0:
                vb = vB + wp.cross(wB, rB)
            vrel = va - vb
            rtA1 = wp.cross(rA, t1)
            meft1 = iMA + wp.dot(rtA1, IA * rtA1)
            rtA2 = wp.cross(rA, t2)
            meft2 = iMA + wp.dot(rtA2, IA * rtA2)
            if bj >= 0:
                rtB1 = wp.cross(rB, t1)
                meft1 = meft1 + iMB + wp.dot(rtB1, IB * rtB1)
                rtB2 = wp.cross(rB, t2)
                meft2 = meft2 + iMB + wp.dot(rtB2, IB * rtB2)
            if meft1 >= 1e-12:
                if meft2 >= 1e-12:
                    o1 = jt1[c]
                    o2 = jt2[c]
                    a1 = o1 - friction_dot(vrel, t1) / meft1
                    a2 = o2 - friction_dot(vrel, t2) / meft2
                    mag = wp.sqrt(a1 * a1 + a2 * a2)
                    lim = mu * jn[c]
                    if mag > lim and mag > 1e-12:
                        a1 = a1 * lim / mag
                        a2 = a2 * lim / mag
                    jt1[c] = a1
                    jt2[c] = a2
                    Jt = (a1 - o1) * t1 + t2d * (a2 - o2) * t2
                    if iMA > 0.0:
                        vA = vA + Jt * iMA
                        wA = wA + IA * wp.cross(rA, Jt)
                    if bj >= 0:
                        if iMB > 0.0:
                            vB = vB - Jt * iMB
                            wB = wB - IB * wp.cross(rB, Jt)
    if iMA > 0.0:
        v[bi] = vA
        w[bi] = wA
    if bj >= 0:
        if iMB > 0.0:
            v[bj] = vB
            w[bj] = wB

from motion_engine.contact_engine_gpu_colored import GraphColoredContactEngine
from motion_engine.contact_engine_gpu_color_cache import ColorCacheContactEngine

class OrderedPairContactEngine(GraphColoredContactEngine):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.KC=dict(self.KC);self.KC['chunk_vel_r']=ordered_pair

class OrderedColorCacheContactEngine(ColorCacheContactEngine):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.KC=dict(self.KC);self.KC['chunk_vel_r']=ordered_fixed

