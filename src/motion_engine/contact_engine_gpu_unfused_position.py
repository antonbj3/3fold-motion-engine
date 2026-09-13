"""Noncontracting position sweep beside retained velocity-only candidates."""
import warp as wp
from motion_engine.contact_engine_gpu import _BETA, _SLOP
from motion_engine.contact_engine_gpu_unfused_solve import UnfusedPairContactEngine, UnfusedColorCacheContactEngine

@wp.kernel(module='unique', module_options={'fuse_fp': False})
def unfused_position(porder: wp.array(dtype=int), pidx: wp.array(dtype=int), pstart: wp.array(dtype=int), pcount: wp.array(dtype=int), pbi: wp.array(dtype=int), pbj: wp.array(dtype=int), ci: int, cstart: wp.array(dtype=int), ccount: wp.array(dtype=int), cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3), cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3), cpen: wp.array(dtype=float), jp: wp.array(dtype=float), xc: wp.array(dtype=wp.vec3), pv: wp.array(dtype=wp.vec3), po: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33), invM: wp.array(dtype=float), dt: float):
    """Chunk position sweep, chunk_mode='registers': same expressions as `f_gs_pos`, pseudo-velocities of
        the two bodies loaded once and stored once."""
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
    pvA = pv[bi]
    poA = po[bi]
    iMB = float(0.0)
    IB = invIw[bi]
    pvB = wp.vec3(0.0, 0.0, 0.0)
    poB = wp.vec3(0.0, 0.0, 0.0)
    if bj >= 0:
        iMB = invM[bj]
        IB = invIw[bj]
        pvB = pv[bj]
        poB = po[bj]
    for k in range(n):
        c = pidx[st + k]
        nrm = cn[c]
        rA = cpA[c] - xc[bi]
        rel = pvA + wp.cross(poA, rA)
        rB = wp.vec3(0.0, 0.0, 0.0)
        if bj >= 0:
            rB = cpB[c] - xc[bj]
            rel = rel - (pvB + wp.cross(poB, rB))
        rnA = wp.cross(rA, nrm)
        meff = iMA + wp.dot(rnA, IA * rnA)
        if bj >= 0:
            rnB = wp.cross(rB, nrm)
            meff = meff + iMB + wp.dot(rnB, IB * rnB)
        if meff >= 1e-12:
            bias = _BETA * wp.max(cpen[c] - _SLOP, 0.0) / dt
            dj = (bias - wp.dot(rel, nrm)) / meff
            nw = wp.max(0.0, jp[c] + dj)
            dj = nw - jp[c]
            jp[c] = nw
            Jn = dj * nrm
            if iMA > 0.0:
                pvA = pvA + Jn * iMA
                poA = poA + IA * wp.cross(rA, Jn)
            if bj >= 0:
                if iMB > 0.0:
                    pvB = pvB - Jn * iMB
                    poB = poB - IB * wp.cross(rB, Jn)
    if iMA > 0.0:
        pv[bi] = pvA
        po[bi] = poA
    if bj >= 0:
        if iMB > 0.0:
            pv[bj] = pvB
            po[bj] = poB

class UnfusedPositionPairContactEngine(UnfusedPairContactEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.KC = dict(self.KC)
        self.KC['chunk_pos_r'] = unfused_position

class UnfusedPositionColorCacheContactEngine(UnfusedColorCacheContactEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.KC = dict(self.KC)
        self.KC['chunk_pos_r'] = unfused_position
