"""Pack independent pair-selection threads while preserving frozen arithmetic."""
from motion_engine.contact_engine_gpu_color_cache import ColorCacheContactEngine
from motion_engine.contact_engine_gpu_colored import _MAXSEG


def build_packed_select(wp, max_keep=8):
    veck = wp.types.vector(length=max_keep, dtype=wp.int32)

    @wp.func
    def _planar(p: wp.vec3, ref: wp.vec3, u: wp.vec3d, w: wp.vec3d):
        """Planar coordinates of a contact point about the reference, in the (u, w) tangent basis."""
        dx = p[0] - ref[0]; dy = p[1] - ref[1]; dz = p[2] - ref[2]            # float32, as on the host
        ax = wp.float64(dx); ay = wp.float64(dy); az = wp.float64(dz)
        return wp.vec2d(ax * u[0] + ay * u[1] + az * u[2], ax * w[0] + ay * w[1] + az * w[2])

    @wp.kernel(module="unique", module_options={"fuse_fp": False})
    def k_select(C: int, mx: int, leaders: wp.array(dtype=int), leader_scan: wp.array(dtype=int), gorder: wp.array(dtype=int), flag: wp.array(dtype=int),
                 cbi: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3),
                 cpen: wp.array(dtype=float), xc: wp.array(dtype=wp.vec3), jn_in: wp.array(dtype=wp.float64),
                 keep: wp.array(dtype=int), jn_out: wp.array(dtype=wp.float64), flags: wp.array(dtype=int)):
        """One thread per body pair: keep at most `mx` points, E-optimal, deepest point always kept, and move
        the normal impulse of every dropped point onto the kept point nearest to it in the contact plane."""
        thread = wp.tid()
        if thread >= leader_scan[C - 1]:
            return
        i = leaders[thread]
        L = int(1)
        while i + L < C and flag[i + L] == 0:
            L += 1
        if L <= mx:
            for t in range(L):
                g = gorder[i + t]
                keep[g] = 1; jn_out[g] = jn_in[g]
            return
        if L > _MAXSEG or mx > max_keep:
            wp.atomic_add(flags, 1, 1)               # reported to the host, which raises
            for t in range(L):
                g = gorder[i + t]
                keep[g] = 1; jn_out[g] = jn_in[g]
            return
        deep = int(0); dbest = cpen[gorder[i]]
        for t in range(1, L):
            p = cpen[gorder[i + t]]
            if p > dbest:
                dbest = p; deep = t
        gd = gorder[i + deep]; nr = cn[gd]
        ax = wp.float64(1.0); ay = wp.float64(0.0); az = wp.float64(0.0)
        if wp.abs(nr[0]) >= 0.9:
            ax = wp.float64(0.0); ay = wp.float64(1.0); az = wp.float64(0.0)
        n0 = wp.float64(nr[0]); n1 = wp.float64(nr[1]); n2 = wp.float64(nr[2])
        dn = ax * n0 + ay * n1 + az * n2
        u0 = ax - dn * n0; u1 = ay - dn * n1; u2 = az - dn * n2
        ln = wp.sqrt(u0 * u0 + u1 * u1 + u2 * u2)
        u = wp.vec3d(u0 / ln, u1 / ln, u2 / ln)
        w = wp.vec3d(n1 * u[2] - n2 * u[1], n2 * u[0] - n0 * u[2], n0 * u[1] - n1 * u[0])
        ref = xc[cbi[gd]]
        rd = _planar(cpA[gd], ref, u, w)
        S00 = rd[0] * rd[0]; S01 = rd[0] * rd[1]; S10 = rd[1] * rd[0]; S11 = rd[1] * rd[1]
        pick = veck(); pick[0] = deep
        for k in range(1, mx):
            sbest = wp.float64(-1.0e300); tbest = int(-1)
            c00 = wp.float64(0.0); c01 = wp.float64(0.0); c10 = wp.float64(0.0); c11 = wp.float64(0.0)
            for t in range(L):
                taken = int(0)
                for q in range(k):
                    if pick[q] == t:
                        taken = 1
                if taken == 0:
                    rt = _planar(cpA[gorder[i + t]], ref, u, w)
                    a00 = S00 + rt[0] * rt[0]; a01 = S01 + rt[0] * rt[1]
                    a10 = S10 + rt[1] * rt[0]; a11 = S11 + rt[1] * rt[1]
                    tr = a00 + a11
                    det = a00 * a11 - a01 * a10
                    disc = wp.max(tr * tr / wp.float64(4.0) - det, wp.float64(0.0))
                    lam = tr / wp.float64(2.0) - wp.sqrt(disc)          # sigma_min(G)^2 = min(n, lam_min)
                    sc = wp.min(wp.float64(k + 1), lam)
                    if sc > sbest:
                        sbest = sc; tbest = t
                        c00 = a00; c01 = a01; c10 = a10; c11 = a11
            pick[k] = tbest
            S00 = c00; S01 = c01; S10 = c10; S11 = c11
        for k in range(mx):
            g = gorder[i + pick[k]]
            keep[g] = 1; jn_out[g] = wp.float64(0.0)
        for t in range(L):
            rt = _planar(cpA[gorder[i + t]], ref, u, w)
            dbest2 = wp.float64(1.0e300); kbest = int(0)
            for k in range(mx):
                rk = _planar(cpA[gorder[i + pick[k]]], ref, u, w)
                e0 = rt[0] - rk[0]; e1 = rt[1] - rk[1]; d2 = e0 * e0 + e1 * e1
                if d2 < dbest2:
                    dbest2 = d2; kbest = k
            gk = gorder[i + pick[kbest]]
            jn_out[gk] = jn_out[gk] + jn_in[gorder[i + t]]

    return k_select


class PackedSelectContactEngine(ColorCacheContactEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._packed_select = build_packed_select(self.wp)

    def _device_manifold(s, C):
        """The grouping, the E-optimal selection and the warm start on the GPU: feature-key radix sort,
        pair-key radix sort, run-length segment flags, one thread per body pair for the selection, a scan
        and a compaction. Returns the number of kept contacts; the only host readback is the two-integer
        flag array (kept count, oversize-segment counter). Same selection as the host stage."""
        wp = s.wp; KD = s.KD; KC = s.KC; mx = s.manifold_max_points
        with s._stage("sort"):
            wp.launch(KD["fkey"], C, inputs=[C, s.cfa, s.cfb, s.N * s.P + 1, s.skey, s.sval], device=s.dev)
            wp.utils.radix_sort_pairs(s.skey, s.sval, C)          # stable: the order is a pure function of the keys
            wp.copy(s.order, s.sval, count=C)
            wp.launch(KC["gather"], C, inputs=[s.order, C, s.cbi, s.cbj, s.cpA, s.cpB, s.cn, s.cpen,
                                               s.tbi, s.tbj, s.tpA, s.tpB, s.tn, s.tpen], device=s.dev)
            wp.launch(KD["pkey_pair"], C, inputs=[C, s.tbi, s.tbj, s.N + 1, s.pkey_s, s.pval_s], device=s.dev)
            wp.utils.radix_sort_pairs(s.pkey_s, s.pval_s, C)
            wp.launch(KD["runflag"], C, inputs=[C, s.pkey_s, s.segflag], device=s.dev)
        with s._stage("select"):
            wn = s._ws_n if s.warm_start else 0
            wp.launch(KD["warm"], C, inputs=[C, s.skey, wn, s.ws_keys, s.ws_jn, s.ws_jt1, s.ws_jt2,
                                             s.jn_in, s.jt1_in, s.jt2_in], device=s.dev)
            s.keep.zero_(); s.dflags.zero_()
            wp.utils.array_scan(s.segflag[:C], s.segscan[:C], True)
            wp.launch(KD["runstart"], C, inputs=[C, s.segflag, s.segscan, s.runstart], device=s.dev)
            wp.launch(s._packed_select, C, inputs=[C, mx, s.runstart, s.segscan, s.pval_s, s.segflag, s.tbi, s.tpA, s.tn, s.tpen,
                                               s.xc, s.jn_in, s.keep, s.jn_red, s.dflags], device=s.dev)
            wp.utils.array_scan(s.keep[:C], s.segscan[:C], True)
            wp.launch(KD["compact"], C, inputs=[C, s.keep, s.segscan, s.skey, s.tbi, s.tbj, s.tpA, s.tpB,
                                                s.tn, s.tpen, s.jn_red, s.jt1_in, s.jt2_in, s.ckeys,
                                                s.sbi, s.sbj, s.spA, s.spB, s.sn, s.spen,
                                                s.jn, s.jt1, s.jt2, s.dflags], device=s.dev)
            fl = s.dflags.numpy(); Ck = int(fl[0])
            if int(fl[1]) > 0:
                raise RuntimeError(f"contact_engine_gpu_colored._device_manifold: {int(fl[1])} body pairs carry "
                                   f"more than {_MAXSEG} contacts or more than the kept-point capacity; the "
                                   f"device selection kernel is fixed size -- use manifold_on_host=True")
            s.jp.zero_()
            wp.launch(KD["kept_runflag"], Ck, inputs=[Ck, s.sbi, s.sbj, s.N + 1, s.segflag], device=s.dev)
            wp.utils.array_scan(s.segflag[:Ck], s.segscan[:Ck], True)
            wp.launch(KD["runstart"], Ck, inputs=[Ck, s.segflag, s.segscan, s.runstart], device=s.dev)
            wp.launch(KD["prio"], Ck, inputs=[Ck, s.segscan, s.runstart, s.pkey], device=s.dev)
        return Ck


if __name__ == "__main__":
    import runpy
    from pathlib import Path
    runpy.run_path(str(Path(__file__).resolve().parents[2] /
                      "probes/innovation_packed_select.py"), run_name="__main__")
