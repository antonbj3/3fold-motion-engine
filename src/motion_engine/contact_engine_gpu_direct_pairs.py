"""GPU body-pair generation and direct manifold selection beside the frozen path."""
import numpy as np
from motion_engine.contact_engine_gpu_prepared_planar import PreparedPlanarContactEngine
from motion_engine.contact_body_pairs_gpu import BodyPairSearch
from motion_engine.contact_generation_pair_bitmask import build_kernels
from motion_engine.contact_engine_gpu_colored import _MAXC


def integration_kernels(wp):
    @wp.kernel
    def total(counts:wp.array(dtype=int),offsets:wp.array(dtype=int),number:int,result:wp.array(dtype=int)):
        result[0]=counts[number-1]+offsets[number-1]

    @wp.kernel
    def identity(values:wp.array(dtype=int)):
        i=wp.tid();values[i]=i

    @wp.kernel
    def gather(order:wp.array(dtype=int),keys:wp.array(dtype=wp.int64),
               bi:wp.array(dtype=int),bj:wp.array(dtype=int),pa:wp.array(dtype=wp.vec3),pb:wp.array(dtype=wp.vec3),
               normal:wp.array(dtype=wp.vec3),pen:wp.array(dtype=float),jn:wp.array(dtype=float),jt1:wp.array(dtype=float),jt2:wp.array(dtype=float),
               okeys:wp.array(dtype=wp.int64),obi:wp.array(dtype=int),obj:wp.array(dtype=int),opa:wp.array(dtype=wp.vec3),opb:wp.array(dtype=wp.vec3),
               onormal:wp.array(dtype=wp.vec3),open_:wp.array(dtype=float),ojn:wp.array(dtype=float),ojt1:wp.array(dtype=float),ojt2:wp.array(dtype=float)):
        i=wp.tid();j=order[i]
        okeys[i]=keys[i];obi[i]=bi[j];obj[i]=bj[j];opa[i]=pa[j];opb[i]=pb[j]
        onormal[i]=normal[j];open_[i]=pen[j];ojn[i]=jn[j];ojt1[i]=jt1[j];ojt2[i]=jt2[j]
    return total,identity,gather


class DirectPairsContactEngine(PreparedPlanarContactEngine):
    def _build(self):
        super()._build();w=self.wp
        self._body_search=BodyPairSearch(w,self.N,self.P)
        capacity=self._body_search.capacity;self._words=(self.P*self.P+31)//32
        self._masks=w.zeros((capacity,self._words),dtype=w.uint32,device=self.dev)
        self._raw_counts=w.zeros(capacity,dtype=int,device=self.dev)
        self._raw_offsets=w.zeros_like(self._raw_counts);self._raw_overflow=w.zeros(1,dtype=int,device=self.dev)
        self._raw_count,self._raw_emit=build_kernels(w)
        self._raw_total,self._identity,self._selected_gather=integration_kernels(w)
        self._selected=[w.empty(_MAXC,dtype=t,device=self.dev) for t in (w.int64,int,int,w.vec3,w.vec3,w.vec3,float,float,float,float)]

    def _generate_pairs(self):
        w=self.wp;number=self._body_search.rebuild(self.allp)
        if not number:return 0
        w.launch(self._raw_count,number*32,inputs=[self.allp,self._body_search.pairs,self.P,self._words,self._masks,self._raw_counts],device=self.dev,block_dim=128)
        w.utils.array_scan(self._raw_counts[:number],self._raw_offsets[:number],inclusive=False)
        w.launch(self._raw_total,1,inputs=[self._raw_counts,self._raw_offsets,number,self.cnt],device=self.dev)
        count=int(self.cnt.numpy()[0])
        if count>_MAXC:raise RuntimeError('Raw contact capacity exceeded')
        self._raw_overflow.zero_()
        w.launch(self._raw_emit,number*32,inputs=[self.allp,self._body_search.pairs,self.P,self._words,self._masks,self._raw_offsets,_MAXC,self._raw_overflow,
            self.cbi,self.cbj,self.cpA,self.cpB,self.cn,self.cpen,self.cfa,self.cfb],device=self.dev,block_dim=128)
        return count

    def _device_manifold(s,C):
        w=s.wp;d=s.KD
        with s._stage('sort'):
            w.launch(d['fkey'],C,inputs=[C,s.cfa,s.cfb,s.N*s.P+1,s.skey,s.sval],device=s.dev)
            w.launch(d['pkey_pair'],C,inputs=[C,s.cbi,s.cbj,s.N+1,s.pkey_s,s.pval_s],device=s.dev)
            w.launch(d['runflag'],C,inputs=[C,s.pkey_s,s.segflag],device=s.dev)
        with s._stage('select'):
            wn=s._ws_n if s.warm_start else 0
            w.launch(d['warm'],C,inputs=[C,s.skey,wn,s.ws_keys,s.ws_jn,s.ws_jt1,s.ws_jt2,s.jn_in,s.jt1_in,s.jt2_in],device=s.dev)
            s.keep.zero_();s.dflags.zero_()
            w.utils.array_scan(s.segflag[:C],s.segscan[:C],True)
            w.launch(d['runstart'],C,inputs=[C,s.segflag,s.segscan,s.runstart],device=s.dev)
            w.launch(s._packed_select,C,inputs=[C,s.manifold_max_points,s.runstart,s.segscan,s._planar_coords,s.pval_s,s.segflag,
                s.cbi,s.cpA,s.cn,s.cpen,s.xc,s.jn_in,s.keep,s.jn_red,s.dflags],device=s.dev)
            w.utils.array_scan(s.keep[:C],s.segscan[:C],True)
            w.launch(d['compact'],C,inputs=[C,s.keep,s.segscan,s.skey,s.cbi,s.cbj,s.cpA,s.cpB,s.cn,s.cpen,s.jn_red,s.jt1_in,s.jt2_in,
                *s._selected,s.dflags],device=s.dev)
            flags=s.dflags.numpy();kept=int(flags[0])
            if flags[1]:raise RuntimeError('Direct manifold selection capacity exceeded')
            if kept:
                w.copy(s.skey,s._selected[0],count=kept)
                w.launch(s._identity,kept,inputs=[s.sval],device=s.dev)
                w.utils.radix_sort_pairs(s.skey,s.sval,kept)
                w.launch(s._selected_gather,kept,inputs=[s.sval,s.skey,*s._selected[1:],s.ckeys,s.sbi,s.sbj,s.spA,s.spB,s.sn,s.spen,s.jn,s.jt1,s.jt2],device=s.dev)
            s.jp.zero_()
        return kept

    def step(s, dt, substeps=1):
        if not s._built:
            s._build()
        wp = s.wp
        K = s.K
        KC = s.KC
        for _ in range(substeps):
            sdt = dt / substeps
            with s._stage("generate"):
                wp.launch(K["grav"], s.N, inputs=[s.v, s.invM, sdt], device=s.dev)
                wp.launch(K["invIw"], s.N, inputs=[s.q, s.IbInv, s._kin_wp, s.invIw], device=s.dev)
                wp.launch(K["worldp"], s.N * s.P,
                          inputs=[s.xc, s.q, s.rest, s.P, s.allp, s.owner], device=s.dev)
                C = s._generate_pairs()
            s.n_contacts = C
            wp.copy(s._force_before, s.v)
            if C > 0:
                C = s._device_manifold(C)
                s.n_contacts_solved = C
                s.pv.zero_()
                s.po.zero_()
                with s._stage("pairs"):
                    P = s._build_pairs(C)
                s.n_pairs = P
                with s._stage("colour"):
                    ncol = s._color_pairs(P)
                s.n_colors = ncol
                with s._stage("solve"):
                    wp.capture_launch(s._solve_chunk_graph(P, ncol, sdt))
                if s.warm_start:
                    with s._stage("warm_store"):
                        s._store_warm_device(C)
            else:
                s.n_colors = 0
                s.n_contacts_solved = 0
                s.n_pairs = 0
                s.pv.zero_()
                s.po.zero_()
                s._ws_keys = np.zeros(0, np.int64)
                s._ws_imp = np.zeros((0, 3))
                s._ws_n = 0
            with s._stage("integrate"):
                wp.launch(K["integ"], s.N,
                          inputs=[s.xc, s.q, s.v, s.w, s.pv, s.po, s.invM, sdt], device=s.dev)
            s._force_dt = sdt
            s._prof_steps += 1
        wp.synchronize_device(s.dev)


if __name__=='__main__':
    import runpy
    from pathlib import Path
    runpy.run_path(str(Path(__file__).resolve().parents[2]/'probes/innovation_direct_pairs.py'),run_name='__main__')
