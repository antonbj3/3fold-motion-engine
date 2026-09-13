"""Measure hash-query candidate inflation without changing an engine or kernel."""

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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))


def worker():
    import warp as wp
    from colored_vs_jacobi import _factories,lattice
    from motion_engine.contact_engine_gpu import _R
    wp.init()
    @wp.kernel
    def observe(p:wp.array(dtype=wp.vec3),owner:wp.array(dtype=int),grid:wp.uint64,
                visits:wp.array(dtype=int),valid:wp.array(dtype=int)):
        i=wp.tid();j=int(0);seen=int(0);accepted=int(0)
        query=wp.hash_grid_query(grid,p[i],2.0*_R)
        while wp.hash_grid_query_next(query,j):
            seen+=1
            if j>i and owner[j]!=owner[i]:
                distance=wp.length(p[i]-p[j])
                if distance<2.0*_R and distance>1e-9:
                    accepted+=1
        visits[i]=seen;valid[i]=accepted
    rows=[]
    for bodies in (1000,10000):
        engine=_factories()['colored_manifold_chunks'](vit=40,pit=20)
        for point in lattice(bodies):engine.add_body(point)
        for _ in range(40):engine.step(1/240,substeps=1)
        p=engine.allp.numpy();o=engine.owner.numpy();n=len(p)
        visits=wp.empty(n,dtype=int,device=engine.dev);valid=wp.empty(n,dtype=int,device=engine.dev)
        groups=[];original=None
        for shape in ((64,64,64),(128,128,128),(256,256,8)):
            grid=wp.HashGrid(*shape,device=engine.dev);grid.build(engine.allp,2*_R)
            wp.launch(observe,n,inputs=[engine.allp,engine.owner,grid.id,visits,valid],device=engine.dev)
            v=visits.numpy();a=valid.numpy()
            if original is None:original=a.copy()
            ground=int(np.count_nonzero(p[:,2]-np.float32(_R)<0))
            groups.append({'shape':list(shape),'buckets':int(np.prod(shape)),
                           'visits_total':int(v.astype(np.int64).sum()),'visits_mean':float(v.mean()),
                           'visits_max':int(v.max()),'valid_total':int(a.sum()),
                           'contacts_with_ground':int(a.sum())+ground,
                           'same_valid_counts':a.tobytes()==original.tobytes(),
                           'visits_cover_valid':bool(np.all(v>=a)),
                           'counts_sha256':hashlib.sha256(v.tobytes()+a.tobytes()).hexdigest()})
        rows.append({'bodies':bodies,'points':n,'fixture_sha256':hashlib.sha256(p.tobytes()+o.tobytes()).hexdigest(),
                     'grids':groups})
    print('RESULT '+json.dumps(rows),flush=True)


def main():
    if sys.argv[1:]==['--worker']:worker();return 0
    if sys.argv[1:]:raise ValueError('Expected no arguments or --worker')
    from innovation_stack_probe import idle
    legs=[]
    for _ in range(2):
        idle()
        child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=ROOT,capture_output=True,text=True)
        if child.returncode:
            print(child.stdout);print(child.stderr,file=sys.stderr);raise RuntimeError('Worker failed')
        idle()
        records=[line[7:] for line in child.stdout.splitlines() if line.startswith('RESULT ')]
        if len(records)!=1:raise RuntimeError('Expected one result')
        legs.append(json.loads(records[0]));print(records[0],flush=True)
    prior=json.loads((ROOT/'reports/innovation_contact_gpu_stage_l4.json').read_text())['legs'][0]
    gates={'two_full_results_identical':legs[0]==legs[1],
           'valid_counts_grid_invariant':all(g['same_valid_counts'] for leg in legs for row in leg for g in row['grids']),
           'exact_canonical_count':all(row['fixture_sha256']==ref['fixture_sha256'] and all(g['contacts_with_ground']==ref['candidate_count'] for g in row['grids']) for leg in legs for row,ref in zip(leg,prior)),
           'visits_cover_valid':all(g['visits_cover_valid'] for leg in legs for row in leg for g in row['grids'])}
    report={'legs':legs,'gates':gates,'performance_measured':False}
    (ROOT/'reports/innovation_contact_grid_visits.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
