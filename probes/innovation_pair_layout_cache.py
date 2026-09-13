"""Certify exact pair-layout reuse with invalidation and full-engine gates."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from cross_hardware_state_probe import ROOT
from innovation_stack_probe import idle


def digest(e):
    state=e.get_state();h=hashlib.sha256();finite=True
    arrays=[np.ascontiguousarray(getattr(state,k)) for k in ('xc','Rm','vc','om')]
    arrays.append(np.ascontiguousarray(e.contact_forces()))
    c=e.n_contacts_solved
    for field in ('ckeys','sbi','sbj','spA','spB','sn','spen','jn','jt1','jt2','jp'):
        arrays.append(getattr(e,field).numpy()[:c].copy())
    for a in arrays:h.update(a.tobytes());finite &= bool(np.isfinite(a).all())
    return h.hexdigest(),finite


class Trace:
    def __init__(self,engine):self.engine=engine;self.sha=hashlib.sha256();self.finite=True
    def __getattr__(self,key):return getattr(self.engine,key)
    def step(self,*args,**kw):
        self.engine.step(*args,**kw)
        h,finite=digest(self.engine);self.sha.update(bytes.fromhex(h));self.finite &= finite


def controls(engine_class=None, shared=False):
    from motion_engine.contact_engine_gpu_pair_layout_cache import PairLayoutCacheContactEngine
    from motion_engine.contact_engine_gpu_colored import GraphColoredContactEngine
    from motion_engine.contact_engine_gpu_island_solve import partition_pairs
    from engine_metrics import DIMS
    constructor=PairLayoutCacheContactEngine if engine_class is None else engine_class
    e=constructor(dims=DIMS,mu=.5,vit=40,pit=10)
    for i in range(4):e.add_body([0,0,.101+i*.205])
    for _ in range(100):e.step(1/240,substeps=1)
    C=e.n_contacts_solved;rows=[]
    def pair_hash(P,C):
        h=hashlib.sha256()
        for field,count in (('pvalP',C),('pstart',P),('pcount',P),('pbi',P),('pbj',P)):
            h.update(getattr(e,field).numpy()[:count].tobytes())
        return h.hexdigest()
    for label in ('unchanged','permutation','count','mass','unchanged_again'):
        if label=='permutation':
            a=e.sbi.numpy()[:C].copy();b=e.sbj.numpy()[:C].copy()
            other=next(i for i in range(1,C) if (a[i],b[i])!=(a[0],b[0]))
            a[[0,other]]=a[[other,0]];b[[0,other]]=b[[other,0]]
            e.wp.copy(e.sbi,e.wp.array(a,dtype=int,device=e.dev),count=C)
            e.wp.copy(e.sbj,e.wp.array(b,dtype=int,device=e.dev),count=C)
        if label=='count':C-=1
        if label=='mass':
            mass=e.invM.numpy().copy();mass[1]=0
            e.wp.copy(e.invM,e.wp.array(mass,dtype=float,device=e.dev))
        if shared:
            e.dflags.zero_()
            e.wp.copy(e.dflags,e.wp.array(np.array([C,0,0,0],dtype=np.int32),dtype=int,device=e.dev))
            e._queue_layout_check(C)
            flags=e.dflags.numpy()
            e._layout_hint=(C,bool(flags[2]),bool(flags[3]))
        misses=e.pair_layout_misses;colour_misses=e.color_cache_misses
        P=e._build_pairs(C);colors=e._color_pairs(P)
        observed=pair_hash(P,C)
        referenceP=GraphColoredContactEngine._build_pairs(e,C)
        reference=pair_hash(referenceP,C)
        packed,starts,counts=partition_pairs(e.pbi.numpy()[:P],e.pbj.numpy()[:P],e.invM.numpy(),e.porder.numpy()[:P])
        packing_exact=all(x.tobytes()==y for x,y in ((packed,e._island_pairs.numpy()[:P].tobytes()),(starts,e._island_starts.numpy()[:len(starts)].tobytes()),(counts,e._island_counts.numpy()[:len(counts)].tobytes())))
        invalidated=e.pair_layout_misses>misses
        color_invalidated=e.color_cache_misses>colour_misses
        rows.append(dict(label=label,pair_sha256=observed,reference_sha256=reference,
                         invalidated=invalidated,color_invalidated=color_invalidated,components=len(counts),
                         pass_=P==referenceP and observed==reference and packing_exact and invalidated==(label in ('permutation','count')) and (label!='mass' or color_invalidated)))
        rows[-1]['pass']=rows[-1].pop('pass_')
    return rows


def worker(reverse=False):
    from motion_engine.contact_engine_gpu_island_solve import IslandSolveContactEngine as ColorCacheContactEngine
    from motion_engine.contact_engine_gpu_pair_layout_cache import PairLayoutCacheContactEngine as PackedSelectContactEngine
    from engine_metrics import DIMS,m5_penetration,m1_stack_load
    from colored_vs_jacobi import lattice,timed_steps
    constructors=[('baseline',ColorCacheContactEngine),('layout',PackedSelectContactEngine)]
    rows={}
    for name,cls in (constructors[::-1] if reverse else constructors):
        traces=[];last=[]
        def make(**kw):
            e=cls(dims=DIMS,mu=.5,**kw);last[:]=[e];return e
        def traced(**kw):
            e=Trace(make(**(dict(vit=40,pit=10)|kw)));traces.append(e);return e
        m5=m5_penetration(traced,'gpu',16,600)
        m1=m1_stack_load(traced,'gpu',4,400)
        rest=traced();rest.add_body([0,0,.101])
        for _ in range(200):rest.step(1/240,substeps=1)
        force_error=abs(float(rest.contact_forces()[0,2])-rest._M[0]*9.81)/(rest._M[0]*9.81)
        timing=timed_steps(make,lattice(10000),warm=10,timed=30,vit=40,pit=20)
        h,finite=digest(last[0])
        rows[name]=dict(m5=m5,m1=m1,force_error=force_error,timing=timing,
                       traces=[t.sha.hexdigest() for t in traces],large_sha=h,
                       finite=finite and all(t.finite for t in traces))
        print('ENGINE '+name+' complete',flush=True)
    rows['controls']=controls()
    print('RESULT '+json.dumps(rows),flush=True)


def main():
    if len(sys.argv)==3 and sys.argv[1]=='--worker':worker(sys.argv[2]=='1');return 0
    legs=[];background=[]
    for leg in range(2):
        before=idle()
        p=subprocess.run([sys.executable,__file__,'--worker',str(leg)],cwd=ROOT,capture_output=True,text=True)
        if p.returncode:print(p.stdout);print(p.stderr);raise RuntimeError('Worker failed')
        lines=[s[7:] for s in p.stdout.splitlines() if s.startswith('RESULT ')]
        if len(lines)!=1:raise RuntimeError('Missing worker receipt')
        legs.append(json.loads(lines[0]));background.append(dict(before=before,after=idle()))
        print(lines[0],flush=True)
    def numeric(row):return {k:v for k,v in row.items() if k!='timing'}
    gates=dict(exact_reference=all(numeric(r['baseline'])==numeric(r['layout']) for r in legs),
        exact_repeats=all(numeric(legs[0][name])==numeric(legs[1][name]) for name in ('baseline','layout')),
        k16=all(r['layout']['m5']['pass'] for r in legs),k4=all(r['layout']['m1']['pass'] for r in legs),
        rest=all(r['layout']['force_error']<.01 for r in legs),finite=all(r['layout']['finite'] for r in legs),
        no_timing_regression=all(r['layout']['timing']['ms_per_step']<=r['baseline']['timing']['ms_per_step'] for r in legs),
        original_speed=all(r['layout']['timing']['ms_per_step']<=4 for r in legs))
    gates['invalidation_controls']=all(all(c['pass'] for c in r['controls']) for r in legs)
    gates['control_repeats']=legs[0]['controls']==legs[1]['controls']
    report=dict(legs=legs,background=background,gates=gates,
                submillisecond_goal=all(r['layout']['timing']['ms_per_step']<1 for r in legs))
    (ROOT/'reports/innovation_pair_layout_cache.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(gates),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
