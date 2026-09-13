"""Measure exact fixed-point sweep counts with unchanged full-engine outputs."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from cross_hardware_state_probe import ROOT,fingerprint
from innovation_packed_select import digest
from innovation_stack_probe import idle


def worker(output):
    from motion_engine.contact_engine_gpu_pair_layout_cache import PairLayoutCacheContactEngine
    from motion_engine.contact_engine_gpu_fixed_point_observer import FixedPointObserverEngine
    from colored_vs_jacobi import lattice,stack
    from engine_metrics import DIMS
    arrays={};rows=[]
    for name,centers,steps,pit in (('stack16',stack(16),600,10),('lattice10000',lattice(10000),40,20)):
        baseline=PairLayoutCacheContactEngine(dims=DIMS,mu=.5,vit=40,pit=pit)
        observed=FixedPointObserverEngine(dims=DIMS,mu=.5,vit=40,pit=pit)
        for e in (baseline,observed):
            for c in centers:e.add_body(c)
        parity=True;finite=True;hist={'velocity':{},'position':{}};recurrence=True;traces=[]
        for step in range(steps):
            baseline.step(1/240,substeps=1);observed.step(1/240,substeps=1)
            a,fa=digest(baseline);b,fb=digest(observed)
            parity &= a==b;finite &= fa and fb;traces.append(dict(reference=a,observed=b))
            if observed.n_contacts_solved:
                activity=observed._activity.numpy()[:observed.island_count].copy()
                arrays[f'{name}_{step}']=activity
                for phase,part in (('velocity',activity[:,:40]),('position',activity[:,40:])):
                    zero=part==0
                    needed=np.where(zero.any(axis=1),zero.argmax(axis=1)+1,part.shape[1])
                    recurrence &= not bool(np.any((np.arange(part.shape[1])[None,:]>=needed[:,None]) & (part!=0)))
                    counts=np.bincount(needed,minlength=part.shape[1]+1)
                    for i,n in enumerate(counts):
                        if n:hist[phase][str(i)]=hist[phase].get(str(i),0)+int(n)
        rows.append(dict(scene=name,parity=parity,finite=finite,no_restart=recurrence,histograms=hist,traces=traces))
    np.savez_compressed(output,**arrays)
    print('RESULT '+json.dumps(dict(rows=rows,hashes={k:fingerprint(a) for k,a in arrays.items()})),flush=True)


def main():
    if len(sys.argv)==3 and sys.argv[1]=='--worker':worker(sys.argv[2]);return 0
    legs=[];saved=[];background=[]
    with tempfile.TemporaryDirectory() as tmp:
        for leg in range(2):
            path=Path(tmp)/f'{leg}.npz';before=idle()
            p=subprocess.run([sys.executable,__file__,'--worker',str(path)],cwd=ROOT,capture_output=True,text=True)
            if p.returncode:print(p.stdout);print(p.stderr);raise RuntimeError('Worker failed')
            lines=[s[7:] for s in p.stdout.splitlines() if s.startswith('RESULT ')]
            if len(lines)!=1:raise RuntimeError('Missing receipt')
            legs.append(json.loads(lines[0]));background.append(dict(before=before,after=idle()))
            with np.load(path) as data:saved.append({k:data[k].copy() for k in data.files})
            print('LEG '+str(leg)+' '+json.dumps([{k:r[k] for k in ('scene','parity','histograms')} for r in legs[-1]['rows']]),flush=True)
    exact=legs[0]==legs[1] and all(a.tobytes()==saved[1][k].tobytes() for k,a in saved[0].items())
    gates=dict(exact_reference=all(r['parity'] for leg in legs for r in leg['rows']),exact_repeats=exact,
               finite=all(r['finite'] for leg in legs for r in leg['rows']),coverage=len(saved[0])>600,
               no_change_after_fixed_point=all(r['no_restart'] for leg in legs for r in leg['rows']))
    np.savez_compressed(ROOT/'reports/innovation_fixed_point_observer.npz',**{str(i)+'__'+k:a for i,leg in enumerate(saved) for k,a in leg.items()})
    (ROOT/'reports/innovation_fixed_point_observer.json').write_text(json.dumps(dict(legs=legs,gates=gates,background=background),indent=2)+'\n')
    print(json.dumps(gates),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
