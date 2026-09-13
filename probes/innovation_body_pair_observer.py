"""Measure conservative body-pair work against the unchanged bead contact path."""

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
from scipy.spatial import cKDTree
from cross_hardware_state_probe import ROOT,fingerprint
from innovation_stack_probe import idle

CASES={'stack16':(16,600,(19,199,599)),'lattice10000':(10000,40,(9,19,39))}


def visit_kernel(wp):
    @wp.kernel
    def count(points:wp.array(dtype=wp.vec3),owner:wp.array(dtype=int),grid:wp.uint64,
              output:wp.array(dtype=wp.vec2i)):
        i=wp.tid();visits=int(0);tests=int(0)
        query=wp.hash_grid_query(grid,points[i],0.1);j=int(0)
        while wp.hash_grid_query_next(query,j):
            visits+=1
            if j>i and owner[j]!=owner[i]:tests+=1
        output[i]=wp.vec2i(visits,tests)
    return count


def pairs_table(e,count,prefix,arrays):
    points=e.allp.numpy().copy();owners=e.owner.numpy().copy()
    if not np.array_equal(owners,np.repeat(np.arange(e.N),e.P)):raise ValueError('Unexpected point ownership')
    keys=e.skey.numpy()[:count].copy();stride=e.N*e.P+1
    fa=keys//stride;fb=keys%stride-1
    a=fa//e.P;b=np.where(fb>=0,fb//e.P,-1)
    actual=np.unique(np.column_stack((a[fb>=0],b[fb>=0])),axis=0)
    ground=np.unique(a[fb<0])
    cloud=points.astype(np.float64).reshape(e.N,e.P,3)
    padding=8*np.finfo(np.float32).eps*max(1.,float(np.max(np.abs(points))))
    radius=float(np.float32(.05))+padding
    lower=cloud.min(axis=1)-radius;upper=cloud.max(axis=1)+radius
    center=(lower+upper)*.5;half=(upper-lower)*.5
    proposed=cKDTree(center).query_pairs(2*float(np.max(np.linalg.norm(half,axis=1))),output_type='ndarray')
    if len(proposed):
        keep=np.all(lower[proposed[:,0]]<=upper[proposed[:,1]],axis=1)&np.all(lower[proposed[:,1]]<=upper[proposed[:,0]],axis=1)
        proposed=proposed[keep];proposed=proposed[np.lexsort((proposed[:,1],proposed[:,0]))]
    proposed=proposed.astype(np.int64).reshape(-1,2)
    missing=set(map(tuple,actual))-set(map(tuple,proposed))
    ground_proposed=np.flatnonzero(lower[:,2]<0)
    missing_ground=set(ground)-set(ground_proposed)
    out=e.wp.empty(e.N*e.P,dtype=e.wp.vec2i,device=e.dev)
    e.wp.launch(visit_kernel(e.wp),e.N*e.P,inputs=[e.allp,e.owner,e.grid.id,out],device=e.dev)
    visits=out.numpy().astype(np.int64)
    for name,value in dict(points=points,keys=keys,actual_pairs=actual,candidate_pairs=proposed,visits=visits).items():
        arrays[prefix+'__'+name]=value
    return dict(bodies=e.N,points=e.N*e.P,raw_contacts=count,raw_ground_contacts=int(np.sum(fb<0)),
        active_body_pairs=len(actual),candidate_body_pairs=len(proposed),ground_bodies=len(ground),
        candidate_ground_bodies=len(ground_proposed),missed_pairs=len(missing),missed_ground=len(missing_ground),
        hash_grid_visits=int(visits[:,0].sum()),point_distance_tests=int(visits[:,1].sum()),
        exhaustive_pair_distance_tests=len(proposed)*e.P*e.P,ground_point_tests=len(ground_proposed)*e.P,
        radius_padding=float(padding))


def worker(observe,output):
    from motion_engine.contact_engine_gpu_prepared_planar import PreparedPlanarContactEngine
    from colored_vs_jacobi import lattice,stack
    from engine_metrics import DIMS
    arrays={};records={}
    for scene,(n,steps,samples) in CASES.items():
        e=PreparedPlanarContactEngine(dims=DIMS,mu=.5,vit=40,pit=20 if n>16 else 10)
        for center in (stack(n) if scene=='stack16' else lattice(n)):e.add_body(center)
        step=[0];rows=[];original=e._device_manifold
        def observer(count):
            kept=original(count)
            if step[0] in samples:
                rows.append(dict(step=step[0],**pairs_table(e,count,scene+'__'+str(step[0]),arrays)))
            return kept
        if observe:e._device_manifold=observer
        for i in range(steps):step[0]=i;e.step(1/240,substeps=1)
        state=e.get_state()
        for name in ('xc','Rm','vc','om'):arrays[scene+'__'+name]=np.ascontiguousarray(getattr(state,name))
        arrays[scene+'__force']=np.ascontiguousarray(e.contact_forces())
        for name in ('ckeys','sbi','sbj','spA','spB','sn','spen','jn','jt1','jt2','jp'):
            arrays[scene+'__'+name]=getattr(e,name).numpy()[:e.n_contacts_solved].copy()
        records[scene]=rows
    np.savez_compressed(output,**arrays)
    print('RESULT '+json.dumps(dict(records=records,finite=all(bool(np.isfinite(a).all()) for a in arrays.values()),
        hashes={k:fingerprint(v) for k,v in arrays.items()})),flush=True)


def main():
    if len(sys.argv)==4 and sys.argv[1]=='--worker':worker(sys.argv[2]!='reference',sys.argv[3]);return 0
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
    ref=arrays['reference'];rows=[r for mode in ('first','second') for values in records[mode]['records'].values() for r in values]
    gates=dict(exact_reference=all(all(a[k].tobytes()==ref[k].tobytes() for k in ref) for a in arrays.values()),
        exact_repeats=records['first']==records['second'] and all(arrays['first'][k].tobytes()==arrays['second'][k].tobytes() for k in arrays['first']),
        full_coverage=all([r['step'] for r in records[mode]['records'][scene]]==list(spec[2]) for mode in ('first','second') for scene,spec in CASES.items()),
        conservative_pairs=all(r['missed_pairs']==0 and r['missed_ground']==0 for r in rows),
        finite_valid=all(r['finite'] for r in records.values()) and all(r['point_distance_tests']>=r['raw_contacts']-r['raw_ground_contacts'] and r['hash_grid_visits']>=r['point_distance_tests'] for r in rows))
    np.savez_compressed(ROOT/'reports/innovation_body_pair_observer.npz',**{mode+'__'+k:a for mode,data in arrays.items() for k,a in data.items()})
    (ROOT/'reports/innovation_body_pair_observer.json').write_text(json.dumps(dict(records=records,gates=gates,background=background),indent=2)+'\n')
    print(json.dumps(dict(gates=gates,table=records['first']['records'])),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
