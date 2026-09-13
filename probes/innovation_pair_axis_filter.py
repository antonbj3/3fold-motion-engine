"""Audit a coordinate filter on complete retained world-point snapshots, on CPU."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def worker(output):
    report=json.loads((ROOT/'reports/innovation_body_pair_observer.json').read_text())
    if not all(report['gates'].values()):raise ValueError('Source observer gates failed')
    bound=np.nextafter(np.float32(.1),np.float32(np.inf))
    rows={};masks={}
    with np.load(ROOT/'reports/innovation_body_pair_observer.npz') as data:
        for leg in ('first','second'):
            rows[leg]=[]
            for scene,snapshots in report['records'][leg]['records'].items():
                for receipt in snapshots:
                    name=f"{leg}__{scene}__{receipt['step']}"
                    p=data[name+'__points'];pairs=data[name+'__candidate_pairs'];keys=data[name+'__keys']
                    n=receipt['bodies'];width=len(p)//n;body=p.reshape(n,width,3)
                    mask=np.empty((len(pairs),width,width),dtype=bool)
                    for start in range(0,len(pairs),512):
                        ab=pairs[start:start+512]
                        delta=body[ab[:,0],:,None,:]-body[ab[:,1],None,:,:]
                        mask[start:start+len(ab)]=np.all(np.abs(delta)<=bound,axis=-1)
                    stride=len(p)+1;fa=keys//stride;fb=keys%stride-1
                    actual=[(int(a),int(b)) for a,b in zip(fa,fb) if b>=0]
                    index={tuple(map(int,pair)):i for i,pair in enumerate(pairs)}
                    missed=sum(not mask[index[(a//width,b//width)],a%width,b%width] for a,b in actual)
                    packed=np.packbits(mask,axis=None,bitorder='little');masks[name]=packed
                    rows[leg].append(dict(scene=scene,step=receipt['step'],exhaustive_tests=int(mask.size),
                        axis_survivors=int(mask.sum()),current_grid_distance_tests=receipt['point_distance_tests'],
                        actual_contacts=len(actual),missed_contacts=int(missed),mask_sha256=hashlib.sha256(packed.tobytes()).hexdigest()))
    np.savez_compressed(output,**masks)
    print('RESULT '+json.dumps(dict(bound=float(bound),rows=rows)),flush=True)


def main():
    if len(sys.argv)==3 and sys.argv[1]=='--worker':worker(sys.argv[2]);return 0
    results=[];arrays=[]
    with tempfile.TemporaryDirectory() as tmp:
        for leg in range(2):
            target=Path(tmp)/f'{leg}.npz'
            proc=subprocess.run([sys.executable,__file__,'--worker',str(target)],capture_output=True,text=True,check=True)
            lines=[s[7:] for s in proc.stdout.splitlines() if s.startswith('RESULT ')]
            if len(lines)!=1:raise RuntimeError('Missing receipt')
            results.append(json.loads(lines[0]))
            with np.load(target) as data:arrays.append({k:data[k].copy() for k in data.files})
    rows=[r for result in results for leg in result['rows'].values() for r in leg]
    gates=dict(exact_repeats=results[0]==results[1] and all(arrays[0][k].tobytes()==arrays[1][k].tobytes() for k in arrays[0]),
        captured_legs_exact=all(r['rows']['first']==r['rows']['second'] for r in results),
        no_missed_contacts=all(r['missed_contacts']==0 for r in rows),
        finite_complete=all(len(r['rows']['first'])==6 and len(r['rows']['second'])==6 for r in results) and all(0<=r['actual_contacts']<=r['axis_survivors']<=r['exhaustive_tests'] for r in rows))
    np.savez_compressed(ROOT/'reports/innovation_pair_axis_filter.npz',**{str(i)+'__'+k:a for i,data in enumerate(arrays) for k,a in data.items()})
    (ROOT/'reports/innovation_pair_axis_filter.json').write_text(json.dumps(dict(results=results,gates=gates),indent=2)+'\n')
    print(json.dumps(dict(gates=gates,table=results[0]['rows']['first'])),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
