"""Measure grid allocation alone against the frozen full contact engine."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))


def state_hash(engine):
    digest=hashlib.sha256()
    for name in ('xc','q','v','w'):
        digest.update(name.encode());digest.update(getattr(engine,name).numpy().tobytes())
    return digest.hexdigest()


def worker():
    from innovation_stack_probe import trajectory
    from colored_vs_jacobi import _factories,lattice,timed_steps,stage_profile
    from engine_metrics import DIMS,m5_penetration
    from motion_engine.contact_engine_gpu_wide_grid import WideGridContactEngine
    import warp as wp
    wp.init()
    def wide(**kw):
        return WideGridContactEngine(dims=DIMS,mu=.5,manifold_reduce=True,graph_capture=True,pair_chunks=True,**kw)
    rows={}
    for name,mk in (('baseline',_factories()['colored_manifold_chunks']),('wide',wide)):
        m5={str(v):m5_penetration(lambda:mk(vit=v,pit=10),'gpu',16,600) for v in (40,160)}
        small=trajectory(mk)
        engines=[]
        def retain(**kw):
            e=mk(**kw);engines.append(e);return e
        timing=timed_steps(retain,lattice(10000),warm=10,timed=30,vit=40,pit=20)
        timed_hash=state_hash(engines.pop())
        stages=stage_profile(retain,lattice(10000),warm=10,timed=30)
        stage_hash=state_hash(engines.pop())
        rows[name]={'m5':m5,'trajectory_sha256':small,'timing':timing,'stages':stages,
                    'large_timed_state_sha256':timed_hash,'large_profiled_state_sha256':stage_hash}
    print('RESULT '+json.dumps(rows),flush=True)


def main():
    if sys.argv[1:]==['--worker']:worker();return 0
    if sys.argv[1:]:raise ValueError('Expected no arguments or --worker')
    from innovation_stack_probe import idle
    legs=[];backgrounds=[]
    for _ in range(2):
        before=idle()
        child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=ROOT,capture_output=True,text=True)
        if child.returncode:
            print(child.stdout);print(child.stderr,file=sys.stderr);raise RuntimeError('Worker failed')
        after=idle()
        lines=[line[7:] for line in child.stdout.splitlines() if line.startswith('RESULT ')]
        if len(lines)!=1:raise RuntimeError('Expected one result')
        legs.append(json.loads(lines[0]));backgrounds.append({'before':before,'after':after});print(lines[0],flush=True)
    keys=('m5','trajectory_sha256','large_timed_state_sha256','large_profiled_state_sha256')
    gates={'same_physics_as_baseline':all(all(leg['wide'][k]==leg['baseline'][k] for k in keys) for leg in legs),
           'two_runs_identical':all(all(legs[0][name][k]==legs[1][name][k] for k in keys) for name in ('baseline','wide')),
           'wide_no_slower':all(leg['wide']['timing']['ms_per_step']<=leg['baseline']['timing']['ms_per_step'] for leg in legs),
           'original_k16_40':all(leg['wide']['m5']['40']['pass'] for leg in legs),
           'original_wall_le_4ms':all(leg['wide']['timing']['ms_per_step']<=4 for leg in legs)}
    result={'legs':legs,'backgrounds':backgrounds,'gates':gates,'grid':[256,256,8],
            'scope':'grid-allocation-only subclass, unchanged frozen contact and solver kernels'}
    (ROOT/'reports/innovation_wide_grid_engine.json').write_text(json.dumps(result,indent=2)+'\n')
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
