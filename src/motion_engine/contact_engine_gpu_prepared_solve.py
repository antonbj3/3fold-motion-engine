"""Prepare immutable contact geometry once per step before component sweeps."""
from motion_engine.contact_engine_gpu import _BETA, _SLOP
from motion_engine.contact_engine_gpu_colored import _MAXC
from motion_engine.contact_engine_gpu_prepared_planar import PreparedPlanarContactEngine


def build_prepared_solve(wp):
    @wp.func
    def warm_pair(p: int, pidx: wp.array(dtype=int), pstart: wp.array(dtype=int), pcount: wp.array(dtype=int), pbi: wp.array(dtype=int), pbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3), cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3), jn: wp.array(dtype=float), jt1: wp.array(dtype=float), jt2: wp.array(dtype=float), xc: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), w: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33), invM: wp.array(dtype=float)):
        a = pbi[p]
        b = pbj[p]
        va = v[a]
        wa = w[a]
        vb = wp.vec3(0.0)
        wb = wp.vec3(0.0)
        if b >= 0:
            vb = v[b]
            wb = w[b]
        for k in range(pcount[p]):
            c = pidx[pstart[p] + k]
            n = cn[c]
            axis = wp.vec3(1.0, 0.0, 0.0)
            if wp.abs(n[0]) >= 0.9:
                axis = wp.vec3(0.0, 1.0, 0.0)
            t1 = wp.normalize(axis - wp.dot(axis, n) * n)
            t2 = wp.cross(n, t1)
            impulse = jn[c] * n + jt1[c] * t1 + jt2[c] * t2
            if invM[a] > 0.0:
                va = va + invM[a] * impulse
                wa = wa + invIw[a] * wp.cross(cpA[c] - xc[a], impulse)
            if b >= 0:
                if invM[b] > 0.0:
                    vb = vb - invM[b] * impulse
                    wb = wb - invIw[b] * wp.cross(cpB[c] - xc[b], impulse)
        if invM[a] > 0.0:
            v[a] = va
            w[a] = wa
        if b >= 0:
            if invM[b] > 0.0:
                v[b] = vb
                w[b] = wb

    @wp.kernel
    def prepare(pidx: wp.array(dtype=int), pstart: wp.array(dtype=int), pcount: wp.array(dtype=int),
                pbi: wp.array(dtype=int), pbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3),
                cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3), cpen: wp.array(dtype=float),
                xc: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33), invM: wp.array(dtype=float),
                dt: float, arm_a: wp.array(dtype=wp.vec3), arm_b: wp.array(dtype=wp.vec3), tangent1: wp.array(dtype=wp.vec3), tangent2: wp.array(dtype=wp.vec3), effective: wp.array(dtype=wp.vec4)):
        p = wp.tid()
        bi = pbi[p]; bj = pbj[p]
        iMA = invM[bi]; IA = invIw[bi]
        iMB = float(0.0); IB = invIw[bi]
        if bj >= 0:
            iMB = invM[bj]; IB = invIw[bj]
        for k in range(pcount[p]):
            c = pidx[pstart[p] + k]
            nrm = cn[c]
            rA = cpA[c] - xc[bi]
            rB = wp.vec3(0.0, 0.0, 0.0)
            if bj >= 0:
                rB = cpB[c] - xc[bj]
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
            rtA1 = wp.cross(rA, t1)
            meft1 = iMA + wp.dot(rtA1, IA * rtA1)
            rtA2 = wp.cross(rA, t2)
            meft2 = iMA + wp.dot(rtA2, IA * rtA2)
            if bj >= 0:
                rtB1 = wp.cross(rB, t1)
                meft1 = meft1 + iMB + wp.dot(rtB1, IB * rtB1)
                rtB2 = wp.cross(rB, t2)
                meft2 = meft2 + iMB + wp.dot(rtB2, IB * rtB2)
            bias = _BETA * wp.max(cpen[c] - _SLOP, 0.0) / dt
            arm_a[c] = rA; arm_b[c] = rB
            tangent1[c] = t1; tangent2[c] = t2
            effective[c] = wp.vec4(meff, meft1, meft2, bias)

    @wp.func
    def velocity_pair(p: int, pidx: wp.array(dtype=int), pstart: wp.array(dtype=int), pcount: wp.array(dtype=int), pbi: wp.array(dtype=int), pbj: wp.array(dtype=int), cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3), cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3), jn: wp.array(dtype=float), jt1: wp.array(dtype=float), jt2: wp.array(dtype=float), xc: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), w: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33), invM: wp.array(dtype=float), mu: float, t2d: float, arm_a: wp.array(dtype=wp.vec3), arm_b: wp.array(dtype=wp.vec3), tangent1: wp.array(dtype=wp.vec3), tangent2: wp.array(dtype=wp.vec3), effective: wp.array(dtype=wp.vec4)):
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
            rA = arm_a[c]
            va = vA + wp.cross(wA, rA)
            vb = wp.vec3(0.0, 0.0, 0.0)
            rB = wp.vec3(0.0, 0.0, 0.0)
            if bj >= 0:
                rB = arm_b[c]
                vb = vB + wp.cross(wB, rB)
            vrel = va - vb
            vn = wp.dot(vrel, nrm)
            vt = vrel - vn * nrm
            mt = wp.length(vt)
            t1 = tangent1[c]
            t2 = tangent2[c]
            meff = effective[c][0]
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
                meft1 = effective[c][1]
                meft2 = effective[c][2]
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
    def position_pair(p: int, pidx: wp.array(dtype=int), pstart: wp.array(dtype=int), pcount: wp.array(dtype=int), pbi: wp.array(dtype=int), pbj: wp.array(dtype=int), cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3), cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3), cpen: wp.array(dtype=float), jp: wp.array(dtype=float), xc: wp.array(dtype=wp.vec3), pv: wp.array(dtype=wp.vec3), po: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33), invM: wp.array(dtype=float), dt: float, arm_a: wp.array(dtype=wp.vec3), arm_b: wp.array(dtype=wp.vec3), tangent1: wp.array(dtype=wp.vec3), tangent2: wp.array(dtype=wp.vec3), effective: wp.array(dtype=wp.vec4)):
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
            rA = arm_a[c]
            rel = pvA + wp.cross(poA, rA)
            rB = wp.vec3(0.0, 0.0, 0.0)
            if bj >= 0:
                rB = arm_b[c]
                rel = rel - (pvB + wp.cross(poB, rB))
            meff = effective[c][0]
            if meff >= 1e-12:
                bias = effective[c][3]
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

    @wp.kernel
    def solve(packed: wp.array(dtype=int), starts: wp.array(dtype=int), counts: wp.array(dtype=int),
              pidx: wp.array(dtype=int), pstart: wp.array(dtype=int), pcount: wp.array(dtype=int),
              pbi: wp.array(dtype=int), pbj: wp.array(dtype=int), cbi: wp.array(dtype=int), cbj: wp.array(dtype=int),
              cpA: wp.array(dtype=wp.vec3), cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3),
              cpen: wp.array(dtype=float), jn: wp.array(dtype=float), jt1: wp.array(dtype=float),
              jt2: wp.array(dtype=float), jp: wp.array(dtype=float), xc: wp.array(dtype=wp.vec3),
              v: wp.array(dtype=wp.vec3), w: wp.array(dtype=wp.vec3), pv: wp.array(dtype=wp.vec3),
              po: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33), invM: wp.array(dtype=float),
              mu: float, t2d: float, dt: float, vit: int, pit: int, arm_a: wp.array(dtype=wp.vec3), arm_b: wp.array(dtype=wp.vec3), tangent1: wp.array(dtype=wp.vec3), tangent2: wp.array(dtype=wp.vec3), effective: wp.array(dtype=wp.vec4)):
        island=wp.tid()
        first=starts[island]
        count=counts[island]
        for j in range(count):
            p=packed[first+j]
            warm_pair(p,pidx,pstart,pcount,pbi,pbj,cpA,cpB,cn,jn,jt1,jt2,xc,v,w,invIw,invM)
        for iteration in range(vit):
            for j in range(count):
                p=packed[first+j]
                velocity_pair(p,pidx,pstart,pcount,pbi,pbj,cbi,cbj,cpA,cpB,cn,jn,jt1,jt2,xc,v,w,invIw,invM,mu,t2d,arm_a,arm_b,tangent1,tangent2,effective)
        for iteration in range(pit):
            for j in range(count):
                p=packed[first+j]
                position_pair(p,pidx,pstart,pcount,pbi,pbj,cbi,cbj,cpA,cpB,cn,cpen,jp,xc,pv,po,invIw,invM,dt,arm_a,arm_b,tangent1,tangent2,effective)
    return prepare, solve


class PreparedSolveContactEngine(PreparedPlanarContactEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prepare_geometry,self._prepared_solve=build_prepared_solve(self.wp)

    def _build(self):
        super()._build()
        self._arm_a=self.wp.empty(_MAXC,dtype=self.wp.vec3,device=self.dev)
        self._arm_b=self.wp.empty_like(self._arm_a)
        self._tangent1=self.wp.empty_like(self._arm_a)
        self._tangent2=self.wp.empty_like(self._arm_a)
        self._effective=self.wp.empty(_MAXC,dtype=self.wp.vec4,device=self.dev)

    def _chunk_launch(self,ncol,dim,sdt):
        if self.max_island_pairs>64:
            return super()._chunk_launch(ncol,dim,sdt)
        s=self
        geometry=[s._arm_a,s._arm_b,s._tangent1,s._tangent2,s._effective]
        s.wp.launch(s._prepare_geometry,s.n_pairs,
            inputs=[s.pvalP,s.pstart,s.pcount,s.pbi,s.pbj,s.spA,s.spB,s.sn,s.spen,
                    s.xc,s.invIw,s.invM,sdt]+geometry,device=s.dev)
        s.wp.launch(s._prepared_solve,s.island_count,
            inputs=[s._island_pairs,s._island_starts,s._island_counts,s.pvalP,s.pstart,s.pcount,s.pbi,s.pbj,
                    s.sbi,s.sbj,s.spA,s.spB,s.sn,s.spen,s.jn,s.jt1,s.jt2,s.jp,s.xc,s.v,s.w,s.pv,s.po,
                    s.invIw,s.invM,s.mu,s.t2_damp,sdt,s.vit,s.pit]+geometry,device=s.dev)


if __name__=='__main__':
    import runpy
    from pathlib import Path
    runpy.run_path(str(Path(__file__).resolve().parents[2]/'probes/innovation_prepared_solve.py'),run_name='__main__')
