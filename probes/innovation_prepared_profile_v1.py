"""Account for current phase costs before designing a faster contact path."""

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
from cross_hardware_state_probe import ROOT,fingerprint
from innovation_stack_probe import idle


def worker(profile, output):
    from motion_engine.contact_engine_gpu_prepared_planar import PreparedPlanarContactEngine as ProfiledEngine
    from colored_vs_jacobi import lattice
    from engine_metrics import DIMS
    e=ProfiledEngine(dims=DIMS,mu=.5,vit=40,pit=20,profile=profile)
    for center in lattice(10000):e.add_body(center)
    for _ in range(10):e.step(1/240,substeps=1)
    e.prof_ms.clear();e.wp.synchronize()
    start=time.perf_counter()
    for _ in range(30):e.step(1/240,substeps=1)
    e.wp.synchronize();wall=(time.perf_counter()-start)*1000/30
    state=e.get_state()
    arrays={k:np.ascontiguousarray(getattr(state,k)) for k in ('xc','Rm','vc','om')}
    arrays['force']=np.ascontiguousarray(e.contact_forces())
    for field in ('ckeys','sbi','sbj','spA','spB','sn','spen','jn','jt1','jt2','jp'):
        arrays[field]=getattr(e,field).numpy()[:e.n_contacts_solved].copy()
    np.savez_compressed(output,**arrays)
    phases={k:v/30 for k,v in e.prof_ms.items()}
    observed=ProfiledEngine(dims=DIMS,mu=.5,vit=40,pit=20)
    for center in lattice(10000):observed.add_body(center)
    layout=[]
    original=observed._build_pairs
    def inspect_layout(count):
        layout.append(dict(count=count,a=fingerprint(observed.sbi.numpy()[:count]),b=fingerprint(observed.sbj.numpy()[:count])))
        return original(count)
    observed._build_pairs=inspect_layout
    for _ in range(40):observed.step(1/240,substeps=1)
    final=observed.get_state()
    observer_parity=all(np.ascontiguousarray(getattr(final,k)).tobytes()==arrays[k].tobytes() for k in ('xc','Rm','vc','om'))
    print('RESULT '+json.dumps(dict(profile=profile,wall_ms=wall,phases=phases,
        layout_observer_parity=observer_parity,layout=layout,layout_equal_transitions=sum(a==b for a,b in zip(layout,layout[1:])),phase_sum_ms=sum(phases.values()),outside_phase_ms=wall-sum(phases.values()),
        contacts=e.n_contacts,selected=e.n_contacts_solved,pairs=e.n_pairs,colors=e.n_colors,
        solve_nodes=1 if e.max_island_pairs<=64 else e.n_colors*(1+e.vit+e.pit),cache_hits=e.color_cache_hits,cache_misses=e.color_cache_misses,
        hashes={k:fingerprint(a) for k,a in arrays.items()},finite=all(bool(np.isfinite(a).all()) for a in arrays.values()))),flush=True)


def main():
    if len(sys.argv)==4 and sys.argv[1]=='--worker':
        worker(sys.argv[2]=='profiled',sys.argv[3]);return 0
    rows=[];arrays={};background=[]
    with tempfile.TemporaryDirectory() as tmp:
        for mode in ('plain','profiled'):
            for leg in range(2):
                output=Path(tmp)/f'{mode}_{leg}.npz'
                before=idle()
                p=subprocess.run([sys.executable,__file__,'--worker',mode,str(output)],cwd=ROOT,capture_output=True,text=True)
                if p.returncode:print(p.stdout);print(p.stderr);raise RuntimeError('Worker failed')
                background.append(dict(before=before,after=idle()))
                records=[line[7:] for line in p.stdout.splitlines() if line.startswith('RESULT ')]
                if len(records)!=1:raise RuntimeError('Missing receipt')
                rows.append(json.loads(records[0]));print(records[0],flush=True)
                with np.load(output) as data:
                    arrays[f'{mode}_{leg}']={k:data[k].copy() for k in data.files}
    reference=arrays['plain_0']
    exact=all(all(a[k].tobytes()==reference[k].tobytes() for k in reference) for a in arrays.values())
    gates=dict(layout_observer_parity=all(r["layout_observer_parity"] for r in rows),exact_full_states=exact,finite=all(r['finite'] and np.isfinite(r['wall_ms']) and r['wall_ms']>0 for r in rows),
        phase_accounting=all(0<=r['phase_sum_ms']<=r['wall_ms'] and all(np.isfinite(t) and t>=0 for t in r['phases'].values()) for r in rows),
        active_contacts=all(r['contacts']>0 and r['selected']>0 and r['pairs']>0 for r in rows))
    np.savez_compressed(ROOT/'reports/innovation_prepared_profile_v1.npz',**{mode+'__'+k:a for mode,values in arrays.items() for k,a in values.items()})
    (ROOT/'reports/innovation_prepared_profile_v1.json').write_text(json.dumps(dict(rows=rows,gates=gates,background=background),indent=2)+'\n')
    print(json.dumps(gates),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
