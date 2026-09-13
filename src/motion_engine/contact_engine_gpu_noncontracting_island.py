"""Bounded noncontracting component solve after the original warm application."""
from motion_engine.contact_engine_gpu import _BETA, _SLOP
from motion_engine.contact_engine_gpu_prepared_noncontracting import PreparedNoncontractingContactEngine


def build_noncontracting_island_solve(wp):

    @wp.func
    def velocity_pair(p: int, pidx: wp.array(dtype=int), pstart: wp.array(dtype=int), pcount: wp.array(dtype=int), pbi: wp.array(dtype=int), pbj: wp.array(dtype=int), cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3), cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3), jn: wp.array(dtype=float), jt1: wp.array(dtype=float), jt2: wp.array(dtype=float), xc: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), w: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33), invM: wp.array(dtype=float), mu: float, t2d: float):
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
                        a1 = o1 - wp.dot(vrel, t1) / meft1
                        a2 = o2 - wp.dot(vrel, t2) / meft2
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

    @wp.func
    def position_pair(p: int, pidx: wp.array(dtype=int), pstart: wp.array(dtype=int), pcount: wp.array(dtype=int), pbi: wp.array(dtype=int), pbj: wp.array(dtype=int), cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3), cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3), cpen: wp.array(dtype=float), jp: wp.array(dtype=float), xc: wp.array(dtype=wp.vec3), pv: wp.array(dtype=wp.vec3), po: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33), invM: wp.array(dtype=float), dt: float):
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

    @wp.kernel(module='unique', module_options={'fuse_fp': False})
    def solve(packed: wp.array(dtype=int), starts: wp.array(dtype=int), counts: wp.array(dtype=int), pidx: wp.array(dtype=int), pstart: wp.array(dtype=int), pcount: wp.array(dtype=int), pbi: wp.array(dtype=int), pbj: wp.array(dtype=int), cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3), cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3), cpen: wp.array(dtype=float), jn: wp.array(dtype=float), jt1: wp.array(dtype=float), jt2: wp.array(dtype=float), jp: wp.array(dtype=float), xc: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), w: wp.array(dtype=wp.vec3), pv: wp.array(dtype=wp.vec3), po: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33), invM: wp.array(dtype=float), mu: float, t2d: float, dt: float, vit: int, pit: int):
        island = wp.tid()
        first = starts[island]
        count = counts[island]
        for iteration in range(vit):
            for j in range(count):
                p = packed[first + j]
                velocity_pair(p, pidx, pstart, pcount, pbi, pbj, cbi, cbj, cpA, cpB, cn, jn, jt1, jt2, xc, v, w, invIw, invM, mu, t2d)
        for iteration in range(pit):
            for j in range(count):
                p = packed[first + j]
                position_pair(p, pidx, pstart, pcount, pbi, pbj, cbi, cbj, cpA, cpB, cn, cpen, jp, xc, pv, po, invIw, invM, dt)
    return solve


class NoncontractingIslandContactEngine(PreparedNoncontractingContactEngine):
    name = "noncontracting_island_contact_gpu"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._noncontracting_island_solve = build_noncontracting_island_solve(self.wp)

    def _chunk_launch(self, ncol, dim, sdt):
        if self.max_island_pairs > 64:
            return super()._chunk_launch(ncol, dim, sdt)
        for ci in range(ncol):
            self.wp.launch(self._apply_warm, dim,
                inputs=[self.porder, self.pvalP, self.pstart, self.pcount,
                        self.pbi, self.pbj, ci, self.cstart, self.ccount,
                        self.spA, self.spB, self.sn, self.jn, self.jt1, self.jt2,
                        self.xc, self.v, self.w, self.invIw, self.invM], device=self.dev)
        self.wp.launch(self._noncontracting_island_solve, self.island_count,
            inputs=[self._island_pairs, self._island_starts, self._island_counts,
                    self.pvalP, self.pstart, self.pcount, self.pbi, self.pbj,
                    self.sbi, self.sbj, self.spA, self.spB, self.sn, self.spen,
                    self.jn, self.jt1, self.jt2, self.jp, self.xc, self.v, self.w,
                    self.pv, self.po, self.invIw, self.invM, self.mu, self.t2_damp,
                    sdt, self.vit, self.pit], device=self.dev)
