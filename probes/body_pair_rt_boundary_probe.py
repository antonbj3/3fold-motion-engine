"""Check exact padded-boundary acceptance for the unchanged RT candidate."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'probes'),str(ROOT/'scripts'),str(ROOT/'src')]
import json
import subprocess
import tempfile
import numpy as np
from motion_engine.contact_body_pairs_rt import RTBodyPairSearch
from cross_hardware_state_probe import fingerprint


def fixtures():
    r=np.float32(np.float32(.05)+np.float32(8*np.finfo(np.float32).eps))
    cutoff=np.float32(2*r)
    for axis in range(3):
        for tag,distance,accept in [('below',np.nextafter(cutoff,np.float32(0)),True),('equal',cutoff,True),('above',np.nextafter(cutoff,np.float32(np.inf)),False)]:
            points=np.zeros((2,3),np.float32);points[1,axis]=distance
            expected=[(0,-1)]
            if accept:expected.append((0,1))
            if axis!=2:expected.append((1,-1))
            yield str(axis)+'_'+tag,points,np.array(expected,np.int32),r
    for tag,height,accept in [('below',np.nextafter(r,np.float32(0)),True),('equal',r,False),('above',np.nextafter(r,np.float32(np.inf)),False)]:
        points=np.array([[0,0,height]],np.float32)
        yield 'ground_'+tag,points,np.array([(0,-1)] if accept else [],np.int32).reshape(-1,2),r


def worker(reverse,output):
    import warp as wp
    wp.init();rows={};arrays={};cases=list(fixtures())
    for name,points,expected,radius in (cases[::-1] if reverse else cases):
        with RTBodyPairSearch(wp,len(points),1) as search:
            search.rebuild(wp.array(points,dtype=wp.vec3,device='cuda:0'))
            actual=search.pairs.numpy()[:search.number].copy()
            values=dict(points=points,pairs=actual,expected=expected,lower=search.lower.numpy(),upper=search.upper.numpy())
            arrays.update({name+'__'+k:v for k,v in values.items()})
            rows[name]=dict(exact=actual.shape==expected.shape and actual.tobytes()==expected.tobytes(),
                radius_exact=np.array_equal(values['lower'],points.astype(np.float64)-float(radius)) and np.array_equal(values['upper'],points.astype(np.float64)+float(radius)),
                hashes={k:fingerprint(v) for k,v in values.items()})
    np.savez_compressed(output,**arrays)
    print('RESULT '+json.dumps(rows),flush=True)


def main():
    if len(sys.argv)==4 and sys.argv[1]=='--worker':worker(sys.argv[2]=='1',sys.argv[3]);return 0
    records=[];arrays=[]
    with tempfile.TemporaryDirectory() as tmp:
        for leg in range(2):
            output=Path(tmp)/f'{leg}.npz'
            p=subprocess.run([sys.executable,__file__,'--worker',str(leg),str(output)],capture_output=True,text=True,timeout=240)
            if p.returncode:raise RuntimeError(p.stdout+p.stderr)
            # Preserve sanitizer diagnostics from child processes.
            print(p.stderr,flush=True)
            records.append(json.loads(next(s[7:] for s in p.stdout.splitlines() if s.startswith('RESULT '))))
            with np.load(output) as data:arrays.append({k:data[k].copy() for k in data.files})
    gates=dict(exact_expected=all(v['exact'] for r in records for v in r.values()),
        actual_boundary=all(v['radius_exact'] for r in records for v in r.values()),
        complete_repeat=all(len(r)==12 for r in records) and records[0]==records[1] and arrays[0].keys()==arrays[1].keys() and all(arrays[0][k].tobytes()==arrays[1][k].tobytes() for k in arrays[0]))
    report=dict(gates=gates,records=records)
    suffix=sys.argv[1] if len(sys.argv)==2 else 'plain'
    if suffix not in ('plain','memcheck'):raise ValueError('Unknown report suffix')
    path=ROOT/'reports'/('body_pair_rt_boundary_'+suffix)
    path.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(path.with_suffix('.npz'),**{str(i)+'__'+k:v for i,a in enumerate(arrays) for k,v in a.items()})
    print(json.dumps(gates),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
