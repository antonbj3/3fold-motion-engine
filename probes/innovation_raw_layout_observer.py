"""Measure raw contact ordering reuse without modifying the contact path."""

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
from cross_hardware_state_probe import ROOT, fingerprint
from innovation_stack_probe import idle


def worker(observe, output):
    from motion_engine.contact_engine_gpu_prepared_planar import PreparedPlanarContactEngine
    from colored_vs_jacobi import lattice
    from engine_metrics import DIMS
    e=PreparedPlanarContactEngine(dims=DIMS,mu=.5,vit=40,pit=20)
    for center in lattice(10000):e.add_body(center)
    arrays={};rows=[];step=[0];active=[]
    original=e._device_manifold
    def observed(count):
        selected=original(count)
        keys=e.skey.numpy()[:count].copy()
        order=e.pval_s.numpy()[:count].copy()
        arrays[f"raw_keys_{step[0]}"]=keys
        arrays[f"pair_order_{step[0]}"]=order
        rows.append(dict(step=step[0],count=count,keys=fingerprint(keys),pair_order=fingerprint(order),
                         duplicate_keys=int(count-len(np.unique(keys)))))
        return selected
    if observe:e._device_manifold=observed
    for i in range(40):
        step[0]=i;e.step(1/240,substeps=1)
        if e.n_contacts>0:active.append(i)
    state=e.get_state()
    for k in ('xc','Rm','vc','om'):arrays[k]=np.ascontiguousarray(getattr(state,k))
    arrays['force']=np.ascontiguousarray(e.contact_forces())
    for k in ('ckeys','sbi','sbj','spA','spB','sn','spen','jn','jt1','jt2','jp'):
        arrays[k]=getattr(e,k).numpy()[:e.n_contacts_solved].copy()
    np.savez_compressed(output,**arrays)
    equal=lambda a,b: a['count']==b['count'] and a['keys']==b['keys'] and a['pair_order']==b['pair_order']
    print('RESULT '+json.dumps(dict(observe=observe,rows=rows,active=active,
        equal_transitions=sum(equal(a,b) for a,b in zip(rows,rows[1:])),transitions=max(0,len(rows)-1),
        finite=all(bool(np.isfinite(a).all()) for a in arrays.values()),
        hashes={k:fingerprint(a) for k,a in arrays.items()})),flush=True)


def main():
    if len(sys.argv)==4 and sys.argv[1]=='--worker':
        worker(sys.argv[2]!='reference',sys.argv[3]);return 0
    records={};arrays={};background=[]
    with tempfile.TemporaryDirectory() as tmp:
        for mode in ('reference','first','second'):
            output=Path(tmp)/f'{mode}.npz';before=idle()
            p=subprocess.run([sys.executable,__file__,'--worker',mode,str(output)],cwd=ROOT,capture_output=True,text=True)
            if p.returncode:print(p.stdout);print(p.stderr);raise RuntimeError('Worker failed')
            lines=[s[7:] for s in p.stdout.splitlines() if s.startswith('RESULT ')]
            if len(lines)!=1:raise RuntimeError('Missing receipt')
            records[mode]=json.loads(lines[0]);background.append(dict(before=before,after=idle()))
            with np.load(output) as data:arrays[mode]={k:data[k].copy() for k in data.files}
            print(mode+' complete',flush=True)
    reference=arrays['reference']
    gates=dict(exact_reference=all(all(a[k].tobytes()==reference[k].tobytes() for k in reference) for a in arrays.values()),
        exact_repeats=records['first']==records['second'] and all(arrays['first'][k].tobytes()==arrays['second'][k].tobytes() for k in arrays['first']),
        finite=all(r['finite'] for r in records.values()),
        complete_observations=all([row['step'] for row in records[mode]['rows']]==records[mode]['active'] for mode in ('first','second')))
    np.savez_compressed(ROOT/'reports/innovation_raw_layout_observer.npz',**{mode+'__'+k:a for mode,data in arrays.items() for k,a in data.items()})
    (ROOT/'reports/innovation_raw_layout_observer.json').write_text(json.dumps(dict(records=records,gates=gates,background=background),indent=2)+'\n')
    print(json.dumps(dict(gates=gates,counts={k:(r['equal_transitions'],r['transitions']) for k,r in records.items()})),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
