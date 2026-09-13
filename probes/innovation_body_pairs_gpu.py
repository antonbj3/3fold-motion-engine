"""Audit complete GPU broad-phase rebuild against retained conservative bounds."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
from cross_hardware_state_probe import ROOT, fingerprint
from innovation_pair_bitmask_probe import fixtures
from innovation_stack_probe import idle
from motion_engine.contact_body_pairs_gpu import BodyPairSearch


def cpu_bounds(points, width):
    cloud=points.astype(np.float64).reshape(-1,width,3)
    padding=8*np.finfo(np.float32).eps*max(1.,float(np.max(np.abs(points))))
    radius=float(np.float32(.05))+padding
    return cloud.min(axis=1)-radius,cloud.max(axis=1)+radius


def worker(reverse, output):
    import warp as wp
    if wp.config.version!='1.13.0' or np.__version__!='2.5.3':
        raise RuntimeError('Pinned environment required')
    wp.init(); rows={}; saved={}
    cases=[c for c in fixtures() if c[0]!='cutoff_control']
    for name,points,width,pairs,_ in (cases[::-1] if reverse else cases):
        gpu=wp.array(points,dtype=wp.vec3,device='cuda:0')
        search=BodyPairSearch(wp,len(points)//width,width)
        for _ in range(3):search.rebuild(gpu)
        wp.synchronize_device('cuda:0');start=time.perf_counter()
        for _ in range(20):search.rebuild(gpu)
        wp.synchronize_device('cuda:0');elapsed=(time.perf_counter()-start)*1000/20
        start=time.perf_counter();lo,hi=cpu_bounds(points,width)
        bounds_ms=(time.perf_counter()-start)*1000
        actual=[search.pairs.numpy()[:search.number].copy(),search.lower.numpy(),search.upper.numpy()]
        expected=[pairs.astype(np.int32),lo,hi]
        exact=[a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes() for a,b in zip(actual,expected)]
        rows[name]=dict(count=search.number,exact=exact,finite=all(bool(np.isfinite(a).all()) for a in actual),
                        hashes=[fingerprint(a) for a in actual],rebuild_ms=elapsed,cpu_bounds_only_ms=bounds_ms)
        for i,a in enumerate(actual):saved[name+'__'+str(i)]=a
        print(name+' complete',flush=True)
    valid=np.array([[0,0,.01],[.06,0,.01]],np.float32)
    gpu=wp.array(valid,dtype=wp.vec3,device='cuda:0'); refused={}
    for name,options in [('capacity',dict(capacity=0)),('radius',dict(radius_limit=.01)),('coordinate',dict(coordinate_limit=.5))]:
        search=BodyPairSearch(wp,2,1,**options)
        try:search.rebuild(gpu); raised=False
        except RuntimeError:raised=True
        refused[name]=raised and search.number==0
    for name,values in [('nonfinite',np.array([[np.nan,0,.01],[.06,0,.01]],np.float32)),('layout',valid[:1])]:
        search=BodyPairSearch(wp,2,1);search.rebuild(gpu)
        if search.number<=0:raise RuntimeError('Missing prior valid state')
        try:search.rebuild(wp.array(values,dtype=wp.vec3,device='cuda:0'));raised=False
        except (RuntimeError,ValueError):raised=True
        refused[name]=raised and search.number==0
    search=BodyPairSearch(wp,2,1);search.rebuild(gpu)
    empty=search.rebuild(wp.array([[0,0,1],[2,0,1]],dtype=wp.vec3,device='cuda:0'))==0 and search.number==0
    np.savez_compressed(output,**saved)
    print('RESULT '+json.dumps(dict(rows=rows,refused=refused,empty=empty)),flush=True)


def main():
    if len(sys.argv)==4 and sys.argv[1]=='--worker':worker(sys.argv[2]=='1',sys.argv[3]);return 0
    results=[];arrays=[];background=[]
    with tempfile.TemporaryDirectory() as tmp:
        for leg in range(2):
            output=Path(tmp)/f'{leg}.npz';before=idle()
            p=subprocess.run([sys.executable,__file__,'--worker',str(leg),str(output)],cwd=ROOT,capture_output=True,text=True)
            if p.returncode:print(p.stdout);print(p.stderr);raise RuntimeError('Worker failed')
            lines=[s[7:] for s in p.stdout.splitlines() if s.startswith('RESULT ')]
            if len(lines)!=1:raise RuntimeError('Missing receipt')
            results.append(json.loads(lines[0]));background.append(dict(before=before,after=idle()))
            with np.load(output) as data:arrays.append({k:data[k].copy() for k in data.files})
            print('leg '+str(leg)+' complete',flush=True)
    def fixed(result):
        r=json.loads(json.dumps(result))
        for row in r['rows'].values():row.pop('rebuild_ms');row.pop('cpu_bounds_only_ms')
        return r
    rows=[r for result in results for r in result['rows'].values()]
    gates=dict(exact_reference=all(all(r['exact']) for r in rows),
        exact_repeats=fixed(results[0])==fixed(results[1]) and arrays[0].keys()==arrays[1].keys() and all(arrays[0][k].tobytes()==arrays[1][k].tobytes() for k in arrays[0]),
        finite_complete=all(len(r['rows'])==6 for r in results) and all(r['finite'] and np.isfinite(r['rebuild_ms']) and r['rebuild_ms']>0 for r in rows),
        refusals=all(len(r['refused'])==5 and all(r['refused'].values()) for r in results),empty=all(r['empty'] for r in results))
    np.savez_compressed(ROOT/'reports/innovation_body_pairs_gpu.npz',**{str(i)+'__'+k:a for i,data in enumerate(arrays) for k,a in data.items()})
    (ROOT/'reports/innovation_body_pairs_gpu.json').write_text(json.dumps(dict(results=results,gates=gates,background=background),indent=2)+'\n')
    print(json.dumps(dict(gates=gates,results=results)),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
