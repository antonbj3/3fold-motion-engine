"""Current contact-engine evidence for a scoped comparison table."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from engine_metrics import DIMS,DT,m1_stack_load,m4_iters_to_tol,m5_penetration
from colored_vs_jacobi import lattice
from innovation_stack_probe import idle


def factories():
    from motion_engine.contact_engine_gpu import RelaxedJacobiContactEngine
    from motion_engine.contact_engine_gpu_colored import GraphColoredContactEngine
    from motion_engine.contact_engine_gpu_color_cache import ColorCacheContactEngine
    return dict(jacobi=lambda **kw:RelaxedJacobiContactEngine(dims=DIMS,mu=.5,**kw),
                manifold_graph=lambda **kw:GraphColoredContactEngine(dims=DIMS,mu=.5,manifold_reduce=True,graph_capture=True,**kw),
                pair_chunks=lambda **kw:GraphColoredContactEngine(dims=DIMS,mu=.5,manifold_reduce=True,graph_capture=True,pair_chunks=True,**kw),
                color_cache=lambda **kw:ColorCacheContactEngine(dims=DIMS,mu=.5,**kw))


def state_hash(engine):
    state=engine.get_state()
    h=hashlib.sha256()
    finite=True
    for name in ('xc','Rm','vc','om'):
        a=np.ascontiguousarray(getattr(state,name));h.update(a.tobytes());finite &= bool(np.isfinite(a).all())
    return h.hexdigest(),finite


class Observed:
    def __init__(self,engine):
        self.engine=engine;self.digest=hashlib.sha256();self.finite=True;self.steps=0;self.contacts_max=0
    def __getattr__(self,name):return getattr(self.engine,name)
    def step(self,*args,**kw):
        self.engine.step(*args,**kw)
        h,finite=state_hash(self.engine);self.digest.update(bytes.fromhex(h));self.finite &= finite;self.steps+=1
        self.contacts_max=max(self.contacts_max,int(self.engine.cnt.numpy()[0]))
    def receipt(self):return dict(sha256=self.digest.hexdigest(),finite=self.finite,steps=self.steps,contacts_max=self.contacts_max)


def worker():
    rows={}
    for name,factory in factories().items():
        observations=[]
        def observed(**kw):
            e=Observed(factory(**(dict(vit=40,pit=10)|kw)));observations.append(e);return e
        m1=m1_stack_load(observed,'gpu',4,400)
        m5={str(k):m5_penetration(observed,'gpu',k,600) for k in (8,16)}
        m4={str(ratio):[m4_iters_to_tol(observed,'gpu',ratio,[vit],200) for vit in (2,4,6,8,10,20,40)] for ratio in (1,1000)}
        fall=[]
        for steps in (60,120):
            e=observed();e.add_body([0,0,10.0])
            for _ in range(steps):e.step(.25/steps,substeps=1)
            state=e.get_state();z=float(state.xc[0,2]);v=float(state.vc[0,2])
            fall.append(dict(steps=steps,position_error=abs(z-(10-.5*9.81*.25**2)),energy_relative=(9.81*z+.5*v*v-9.81*10)/(9.81*10)))
        order=float(np.log2(fall[0]['position_error']/fall[1]['position_error']))
        timings=[]
        for count in (1000,10000):
            e=factory(vit=40,pit=20)
            for pos in lattice(count):e.add_body(pos)
            for _ in range(10):e.step(DT,substeps=1)
            e.wp.synchronize();start=time.perf_counter()
            for _ in range(30):e.step(DT,substeps=1)
            e.wp.synchronize();ms=(time.perf_counter()-start)*1000/30
            h,finite=state_hash(e)
            timings.append(dict(bodies=count,ms_per_step=ms,sha256=h,finite=finite,contacts=int(e.cnt.numpy()[0])))
        rows[name]=dict(m1=m1,m5=m5,m4=m4,fall=fall,observed_order=order,observations=[o.receipt() for o in observations],timings=timings)
        print('ENGINE '+name+' complete',flush=True)
    print('RESULT '+json.dumps(rows),flush=True)


def main():
    if sys.argv[1:]==['--worker']:worker();return 0
    if sys.argv[1:]:raise ValueError('Expected no arguments or --worker')
    legs=[]
    for _ in range(2):
        idle()
        p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=ROOT,capture_output=True,text=True)
        if p.returncode:print(p.stdout);print(p.stderr,file=sys.stderr);return p.returncode
        idle()
        lines=[s[7:] for s in p.stdout.splitlines() if s.startswith('RESULT ')]
        if len(lines)!=1:raise RuntimeError('Expected one worker result')
        legs.append(json.loads(lines[0]));print('LEG '+str(len(legs))+' complete',flush=True)
    def deterministic(leg):
        data=json.loads(json.dumps(leg))
        for r in data.values():
            for timing in r['timings']:timing.pop('ms_per_step')
        return data
    gates=dict(coverage=all(len(leg)==4 and all(len(r['observations'])==19 for r in leg.values()) for leg in legs),
               exact_repeats=deterministic(legs[0])==deterministic(legs[1]),
               finite=all(o['finite'] for leg in legs for r in leg.values() for o in r['observations']+r['timings']),
               timing=all(np.isfinite(t['ms_per_step']) and t['ms_per_step']>0 for leg in legs for r in leg.values() for t in r['timings']),
               contact_constraint_active=all(r['observations'][0]['contacts_max']>0 for leg in legs for r in leg.values()))
    (ROOT/'reports/outclass_rescore_probe.json').write_text(json.dumps(dict(legs=legs,gates=gates),indent=2)+'\n')
    print(json.dumps(gates),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
