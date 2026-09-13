"""Apply cached impulses consistently before fixed-frame pair iterations."""
from motion_engine.contact_engine_gpu_fixed_frame import FixedFrameContactEngine


def build_warm(wp):
    @wp.kernel
    def apply_warm(porder: wp.array(dtype=int), pidx: wp.array(dtype=int),
                   pstart: wp.array(dtype=int), pcount: wp.array(dtype=int),
                   pbi: wp.array(dtype=int), pbj: wp.array(dtype=int), ci: int,
                   cstart: wp.array(dtype=int), ccount: wp.array(dtype=int),
                   cpA: wp.array(dtype=wp.vec3), cpB: wp.array(dtype=wp.vec3),
                   cn: wp.array(dtype=wp.vec3), jn: wp.array(dtype=float),
                   jt1: wp.array(dtype=float), jt2: wp.array(dtype=float),
                   xc: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                   w: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33),
                   invM: wp.array(dtype=float)):
        t = wp.tid()
        if t >= ccount[ci]:
            return
        p = porder[cstart[ci] + t]
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

    return apply_warm


class AppliedWarmContactEngine(FixedFrameContactEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.t2_damp != 1.0:
            raise ValueError("Applied warm start requires full tangent updates")
        self._apply_warm = build_warm(self.wp)

    def _build(self):
        super()._build()
        self.grid = self.wp.HashGrid(256, 256, 8, device=self.dev)

    def _chunk_launch(self, ncol, dim, sdt):
        for ci in range(ncol):
            self.wp.launch(self._apply_warm, dim,
                           inputs=[self.porder, self.pvalP, self.pstart, self.pcount,
                                   self.pbi, self.pbj, ci, self.cstart, self.ccount,
                                   self.spA, self.spB, self.sn, self.jn, self.jt1, self.jt2,
                                   self.xc, self.v, self.w, self.invIw, self.invM], device=self.dev)
        super()._chunk_launch(ncol, dim, sdt)


if __name__ == "__main__":
    import runpy
    from pathlib import Path
    runpy.run_path(str(Path(__file__).resolve().parents[2] /
                      "probes/innovation_stack_applied_warm.py"), run_name="__main__")
