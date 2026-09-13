"""Advance independent dynamic contact components in one captured solve kernel."""
import numpy as np
from motion_engine.contact_engine_gpu import _BETA, _SLOP
from motion_engine.contact_engine_gpu_packed_select import PackedSelectContactEngine


def partition_pairs(a, b, inverse_mass, order):
    a=np.asarray(a);b=np.asarray(b);mass=np.asarray(inverse_mass);order=np.asarray(order)
    if a.ndim!=1 or b.shape!=a.shape or order.shape!=a.shape or mass.ndim!=1:
        raise ValueError("Invalid partition shapes")
    if not all(np.issubdtype(x.dtype,np.integer) for x in (a,b,order)):
        raise ValueError("Integer pair indices required")
    if not np.isfinite(mass).all() or np.any(mass<0):raise ValueError("Invalid inverse masses")
    if np.any(a<0) or np.any(a>=len(mass)) or np.any(b < -1) or np.any(b>=len(mass)):
        raise ValueError("Invalid pair endpoints")
    if not np.array_equal(np.sort(order),np.arange(len(a))):raise ValueError("Order must be a permutation")
    parent=np.arange(len(mass),dtype=np.int32)
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=int(parent[i])
        return i
    for x,y in zip(a,b):
        if y>=0 and mass[x]>0 and mass[y]>0:
            u,v=root(int(x)),root(int(y));parent[max(u,v)]=min(u,v)
    groups={}
    for p in order:
        x,y=int(a[p]),int(b[p])
        component=root(x) if mass[x]>0 else root(y) if y>=0 and mass[y]>0 else len(mass)+int(p)
        groups.setdefault(component,[]).append(int(p))
    groups=[groups[k] for k in sorted(groups)]
    counts=np.array([len(g) for g in groups],np.int32)
    starts=np.cumsum(np.r_[0,counts[:-1]],dtype=np.int32) if len(counts) else np.zeros(0,np.int32)
    packed=np.array([p for group in groups for p in group],np.int32)
    return packed,starts,counts


def build_island_solve(wp):
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

    @wp.kernel
    def solve(packed: wp.array(dtype=int), starts: wp.array(dtype=int), counts: wp.array(dtype=int),
              pidx: wp.array(dtype=int), pstart: wp.array(dtype=int), pcount: wp.array(dtype=int),
              pbi: wp.array(dtype=int), pbj: wp.array(dtype=int), cbi: wp.array(dtype=int), cbj: wp.array(dtype=int),
              cpA: wp.array(dtype=wp.vec3), cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3),
              cpen: wp.array(dtype=float), jn: wp.array(dtype=float), jt1: wp.array(dtype=float),
              jt2: wp.array(dtype=float), jp: wp.array(dtype=float), xc: wp.array(dtype=wp.vec3),
              v: wp.array(dtype=wp.vec3), w: wp.array(dtype=wp.vec3), pv: wp.array(dtype=wp.vec3),
              po: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33), invM: wp.array(dtype=float),
              mu: float, t2d: float, dt: float, vit: int, pit: int):
        island=wp.tid()
        first=starts[island]
        count=counts[island]
        for j in range(count):
            p=packed[first+j]
            warm_pair(p,pidx,pstart,pcount,pbi,pbj,cpA,cpB,cn,jn,jt1,jt2,xc,v,w,invIw,invM)
        for iteration in range(vit):
            for j in range(count):
                p=packed[first+j]
                velocity_pair(p,pidx,pstart,pcount,pbi,pbj,cbi,cbj,cpA,cpB,cn,jn,jt1,jt2,xc,v,w,invIw,invM,mu,t2d)
        for iteration in range(pit):
            for j in range(count):
                p=packed[first+j]
                position_pair(p,pidx,pstart,pcount,pbi,pbj,cbi,cbj,cpA,cpB,cn,cpen,jp,xc,pv,po,invIw,invM,dt)
    return solve


class IslandSolveContactEngine(PackedSelectContactEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._island_solve=build_island_solve(self.wp)

    def _build(self):
        super()._build()
        self._island_pairs=self.wp.empty_like(self.pbi)
        self._island_starts=self.wp.empty_like(self.pbi)
        self._island_counts=self.wp.empty_like(self.pbi)
        self.island_count=0
        self.max_island_pairs=0
        self.island_rebuilds=0
        self._graphs.clear()

    def _color_pairs(self,pairs):
        before=self.color_cache_misses
        colors=super()._color_pairs(pairs)
        if self.color_cache_misses!=before:
            packed,starts,counts=partition_pairs(self.pbi.numpy()[:pairs],self.pbj.numpy()[:pairs],
                                                self.invM.numpy(),self.porder.numpy()[:pairs])
            for dest,data in ((self._island_pairs,packed),(self._island_starts,starts),(self._island_counts,counts)):
                if len(data):self.wp.copy(dest,self.wp.array(data,dtype=int,device=self.dev),count=len(data))
            self.island_count=len(counts)
            self.max_island_pairs=int(max(counts,default=0))
            self.island_rebuilds+=1
            self._graphs.clear()
        return colors

    def _chunk_launch(self,ncol,dim,sdt):
        if self.max_island_pairs>64:
            return super()._chunk_launch(ncol,dim,sdt)
        s=self
        s.wp.launch(s._island_solve,s.island_count,
            inputs=[s._island_pairs,s._island_starts,s._island_counts,s.pvalP,s.pstart,s.pcount,s.pbi,s.pbj,
                    s.sbi,s.sbj,s.spA,s.spB,s.sn,s.spen,s.jn,s.jt1,s.jt2,s.jp,s.xc,s.v,s.w,s.pv,s.po,
                    s.invIw,s.invM,s.mu,s.t2_damp,sdt,s.vit,s.pit],device=s.dev)


if __name__=='__main__':
    import runpy
    from pathlib import Path
    runpy.run_path(str(Path(__file__).resolve().parents[2]/'probes/innovation_island_solve.py'),run_name='__main__')
