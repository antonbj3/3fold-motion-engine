"""Measure unchanged substepping quality and separate fixture costs."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'probes'),str(ROOT/'scripts'),str(ROOT/'src')]
import hashlib
import json
import subprocess
import tempfile
import time
import numpy as np
from innovation_position_physics import digest
from cross_hardware_state_probe import fingerprint
from engine_metrics import DIMS,DT,PITCH,m5_penetration
from colored_vs_jacobi import lattice
from motion_engine.contact_engine_gpu_unfused_position import UnfusedPositionColorCacheContactEngine as Engine
CONFIGS=[(s,v) for s in (1,2,4,8) for v in (4,8,16,32,40)]


def public(e):
    state=e.get_state()
    return {k:np.ascontiguousarray(getattr(state,k)) for k in ('xc','Rm','vc','om')} | {'force':np.ascontiguousarray(e.contact_forces())}


class Observed:
    def __init__(self,s,v):
        self.engine=Engine(dims=DIMS,mu=.5,vit=v,pit=10)
        self.substeps=s;self.trace=hashlib.sha256();self.arrays=[];self.finite=True
    def __getattr__(self,k):return getattr(self.engine,k)
    def step(self,dt,substeps=1):
        self.engine.step(dt,substeps=self.substeps)
        h,finite=digest(self.engine);self.trace.update(bytes.fromhex(h));self.finite &= finite
        self.arrays.append(public(self.engine))


def timed(s,v,centers,steps,warm):
    e=Engine(dims=DIMS,mu=.5,vit=v,pit=10)
    for center in centers:e.add_body(center)
    e._build()
    for _ in range(warm):e.step(DT,substeps=s)
    e.wp.synchronize();begin=time.perf_counter()
    for _ in range(steps):e.step(DT,substeps=s)
    e.wp.synchronize();elapsed=(time.perf_counter()-begin)*1000/steps
    return public(e),elapsed


def worker(reverse,output):
    rows={};saved={}
    for s,v in (CONFIGS[::-1] if reverse else CONFIGS):
        name=f's{s}_v{v}';observed=Observed(s,v)
        metric=m5_penetration(lambda:observed,'gpu',16,600)
        # Preserve all numeric gate fields; omit legacy display-only unit strings.
        metric.pop('units');metric['invalid_reasons']=['spacing_below_floor'] if not metric['valid'] else []
        history={k:np.stack([a[k] for a in observed.arrays]) for k in observed.arrays[0]}
        final,small_ms=timed(s,v,[[0,0,.101+k*PITCH] for k in range(16)],600,0)
        large,large_ms=timed(s,v,lattice(10000),30,10)
        exact=all(history[k][-1].shape==final[k].shape and history[k][-1].tobytes()==final[k].tobytes() for k in final)
        arrays={**{'history_'+k:a for k,a in history.items()},**{'timed_'+k:a for k,a in final.items()},**{'large_'+k:a for k,a in large.items()}}
        saved.update({name+'__'+k:a for k,a in arrays.items()})
        rows[name]=dict(substeps=s,velocity_per_substep=v,velocity_per_step=s*v,position_per_step=s*10,
            metric=metric,trace_sha256=observed.trace.hexdigest(),timed_exact=exact,
            finite=observed.finite and all(bool(np.isfinite(a).all()) for a in arrays.values()),
            hashes={k:fingerprint(a) for k,a in arrays.items()},timing_ms=dict(stack=small_ms,lattice=large_ms))
        print(name+' complete '+json.dumps(dict(m5=metric['max_pen_over_R'],valid=metric['pass'],stack_ms=small_ms,lattice_ms=large_ms)),flush=True)
    np.savez_compressed(output,**saved)
    print('RESULT '+json.dumps(rows),flush=True)


def selection(results,fixture):
    values=[]
    for name,a in results[0].items():
        b=results[1][name]
        if a['metric']['pass'] and b['metric']['pass']:
            values.append(dict(name=name,substeps=a['substeps'],velocity=a['velocity_per_substep'],
                quality=max(a['metric']['max_pen_over_R'],b['metric']['max_pen_over_R']),
                cost=max(a['timing_ms'][fixture],b['timing_ms'][fixture])))
    frontier=[a for a in values if not any(b['quality']<=a['quality'] and b['cost']<=a['cost'] and (b['quality']<a['quality'] or b['cost']<a['cost']) for b in values)]
    budgets={}
    for budget in (.5,1.,2.,4.):
        eligible=[a for a in values if a['cost']<=budget]
        budgets[str(budget)]=min(eligible,key=lambda a:(a['quality'],a['cost'],a['substeps'],a['velocity'])) if eligible else None
    return dict(frontier=sorted(frontier,key=lambda a:(a['cost'],a['quality'])),budgets=budgets)


def main():
    if len(sys.argv)==4 and sys.argv[1]=='--worker':worker(sys.argv[2]=='1',sys.argv[3]);return 0
    results=[];arrays=[]
    with tempfile.TemporaryDirectory() as tmp:
        for leg in range(2):
            output=Path(tmp)/f'{leg}.npz'
            p=subprocess.run([sys.executable,__file__,'--worker',str(leg),str(output)],cwd=ROOT,capture_output=True,text=True,timeout=800)
            if p.returncode:raise RuntimeError(p.stdout+p.stderr)
            lines=p.stdout.splitlines()
            for line in lines:
                if ' complete ' in line:print(line,flush=True)
            results.append(json.loads(next(line[7:] for line in lines if line.startswith('RESULT '))))
            with np.load(output) as data:arrays.append({k:data[k].copy() for k in data.files})
            print('leg '+str(leg)+' complete',flush=True)
    def fixed(result):return {k:{a:b for a,b in row.items() if a!='timing_ms'} for k,row in result.items()}
    archive=json.loads((ROOT/'reports/innovation_position_physics.json').read_text())['legs'][0]['unfused']
    anchor=dict(archive['m5']);anchor.pop('units');anchor['invalid_reasons']=[]
    choices={k:selection(results,k) for k in ('stack','lattice')}
    rows=[v for r in results for v in r.values()]
    gates=dict(coverage=all(len(r)==20 for r in results),
        exact_repeats=fixed(results[0])==fixed(results[1]) and arrays[0].keys()==arrays[1].keys() and all(arrays[0][k].tobytes()==arrays[1][k].tobytes() for k in arrays[0]),
        archive_anchor=all(r['s1_v40']['metric']==anchor and r['s1_v40']['trace_sha256']==archive['traces'][0] for r in results),
        timed_parity=all(r['timed_exact'] for r in rows),
        finite=all(r['finite'] and all(np.isfinite(v) and v>0 for v in r['timing_ms'].values()) for r in rows),
        valid_frontier=all(bool(v['frontier']) for v in choices.values()))
    report=dict(gates=gates,results=results,selection=choices)
    (ROOT/'reports/substep_pareto_probe.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(ROOT/'reports/substep_pareto_probe.npz',**{str(i)+'__'+k:a for i,r in enumerate(arrays) for k,a in r.items()})
    print(json.dumps(dict(gates=gates,selection=choices)),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
