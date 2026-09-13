"""Certify prepared solve geometry against frozen full-engine physical and byte gates."""

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


def worker(reverse=False):
    from motion_engine.contact_engine_gpu_prepared_planar import PreparedPlanarContactEngine
    from motion_engine.contact_engine_gpu_prepared_solve import PreparedSolveContactEngine
    from engine_metrics import DIMS,m5_penetration,m1_stack_load
    from colored_vs_jacobi import lattice,timed_steps
    constructors=[('baseline',PreparedPlanarContactEngine),('prepared',PreparedSolveContactEngine)]
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
    gates=dict(exact_reference=all(numeric(r['baseline'])==numeric(r['prepared']) for r in legs),
        exact_repeats=all(numeric(legs[0][name])==numeric(legs[1][name]) for name in ('baseline','prepared')),
        k16=all(r['prepared']['m5']['pass'] for r in legs),k4=all(r['prepared']['m1']['pass'] for r in legs),
        rest=all(r['prepared']['force_error']<.01 for r in legs),finite=all(r['prepared']['finite'] for r in legs),
        no_timing_regression=all(r['prepared']['timing']['ms_per_step']<=r['baseline']['timing']['ms_per_step'] for r in legs),
        original_speed=all(r['prepared']['timing']['ms_per_step']<=4 for r in legs))
    report=dict(legs=legs,background=background,gates=gates,
                submillisecond_goal=all(r['prepared']['timing']['ms_per_step']<1 for r in legs))
    (ROOT/'reports/innovation_prepared_solve.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(gates),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
