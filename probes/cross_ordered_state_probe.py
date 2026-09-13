"""Capture complete contract outputs for explicit cross-hardware comparison."""

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
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from colored_vs_jacobi import lattice,stack
from outclass_rescore_probe import factories as original_factories

def factories():
    from engine_metrics import DIMS
    from motion_engine.contact_engine_gpu_ordered_friction import OrderedPairContactEngine,OrderedColorCacheContactEngine
    out=original_factories()
    out['pair_chunks']=lambda **kw:OrderedPairContactEngine(dims=DIMS,mu=.5,manifold_reduce=True,graph_capture=True,pair_chunks=True,**kw)
    out['color_cache']=lambda **kw:OrderedColorCacheContactEngine(dims=DIMS,mu=.5,**kw)
    return out


ENGINES=('jacobi','pair_chunks','color_cache')
SCENES={'stack8':(8,120),'lattice1000':(1000,40)}


def fingerprint(a):
    a=np.ascontiguousarray(a)
    return dict(shape=list(a.shape),dtype=a.dtype.str,sha256=hashlib.sha256(a.tobytes()).hexdigest())


def worker(engine,scene,output):
    import warp as wp
    if wp.config.version!='1.13.0' or np.__version__!='2.5.3':raise RuntimeError('Pinned numeric environment required')
    e=factories()[engine](vit=40,pit=10)
    if e.dev!='cuda:0':raise RuntimeError('CUDA required')
    count,steps=SCENES[scene]
    for center in (stack(count) if scene=='stack8' else lattice(count)):e.add_body(center)
    e._build()
    initial={name:fingerprint(getattr(e,name).numpy()) for name in ('xc','q','v','w','invM','IbInv','rest')}
    arrays={name:[] for name in ('xc','Rm','vc','om','force')}
    contacts=[]
    for _ in range(steps):
        e.step(1/240,substeps=1)
        state=e.get_state()
        for name in ('xc','Rm','vc','om'):arrays[name].append(np.array(getattr(state,name),copy=True))
        arrays['force'].append(np.array(e.contact_forces(),copy=True))
        contacts.append(int(e.cnt.numpy()[0]))
    arrays={k:np.ascontiguousarray(v) for k,v in arrays.items()}
    np.savez_compressed(output,**arrays)
    result=dict(variant="ordered_friction_dot",engine=engine,scene=scene,initial=initial,arrays={k:fingerprint(v) for k,v in arrays.items()},
                contacts=contacts,finite=bool(all(np.isfinite(a).all() for a in arrays.values())),
                warp=wp.config.version,numpy=np.__version__)
    print('RESULT '+json.dumps(result),flush=True)


def capture(label):
    if not label.replace('_','').isalnum():raise ValueError('Simple artifact label required')
    records=[];saved={}
    for engine in ENGINES:
        for scene in SCENES:
            legs=[]
            with tempfile.TemporaryDirectory(prefix='state-capture-') as tmp:
                for leg in range(2):
                    path=Path(tmp)/f'leg{leg}.npz'
                    p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',engine,scene,str(path)],cwd=ROOT,capture_output=True,text=True)
                    if p.returncode:print(p.stdout);print(p.stderr,file=sys.stderr);raise RuntimeError('Capture worker failed')
                    lines=[line[7:] for line in p.stdout.splitlines() if line.startswith('RESULT ')]
                    if len(lines)!=1:raise RuntimeError('Expected one worker receipt')
                    legs.append(json.loads(lines[0]))
                    with np.load(path) as data:
                        for key in data.files:saved[f'{engine}__{scene}__{leg}__{key}']=data[key].copy()
            exact=legs[0]==legs[1] and all(saved[f'{engine}__{scene}__0__{k}'].tobytes()==saved[f'{engine}__{scene}__1__{k}'].tobytes() for k in legs[0]['arrays'])
            records.append(dict(engine=engine,scene=scene,legs=legs,exact=exact))
            print(engine+' '+scene+' repeat='+str(exact),flush=True)
    gates=dict(coverage=len(records)==6,exact_repeats=all(r['exact'] for r in records),
               finite=all(leg['finite'] for r in records for leg in r['legs']),
               active_contacts=all(max(leg['contacts'])>0 for r in records for leg in r['legs']))
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip()
    archive=ROOT/f'reports/cross_ordered_{label}.npz'
    np.savez_compressed(archive,**saved)
    report=dict(label=label,hardware=hardware,records=records,gates=gates,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (ROOT/f'reports/cross_ordered_{label}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(gates),flush=True)
    return 0 if all(gates.values()) else 1


def compare(labels):
    reports={label:json.loads((ROOT/f'reports/cross_ordered_{label}.json').read_text()) for label in labels}
    comparisons=[]
    base=labels[0]
    with np.load(ROOT/f'reports/cross_ordered_{base}.npz') as ref:
        for label in labels[1:]:
            diffs=[]
            with np.load(ROOT/f'reports/cross_ordered_{label}.npz') as other:
                for key in ref.files:
                    a,b=ref[key],other[key]
                    if a.shape!=b.shape or a.dtype!=b.dtype:raise RuntimeError('Array contract mismatch')
                    changed=a.view(np.uint8).reshape(a.size,a.itemsize)!=b.view(np.uint8).reshape(b.size,b.itemsize)
                    mask=changed.any(axis=1).reshape(a.shape)
                    if mask.any():
                        first=tuple(int(x) for x in np.argwhere(mask)[0])
                        bits=np.uint32 if a.dtype.itemsize==4 else np.uint64
                        # Bit-pattern distance is not signed-float ULP ordering.
                        av=a.view(bits).reshape(-1);bv=b.view(bits).reshape(-1)
                        distance=max(abs(int(x)-int(y)) for x,y in zip(av[mask.reshape(-1)],bv[mask.reshape(-1)]))
                        diffs.append(dict(array=key,changed_values=int(mask.sum()),first_index=first,
                                          max_absolute=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))),
                                          max_bit_pattern_distance=distance))
            init=all(x['legs'][0]['initial']==y['legs'][0]['initial'] for x,y in zip(reports[base]['records'],reports[label]['records']))
            comparisons.append(dict(reference=base,other=label,initial_exact=init,differences=diffs))
    gates=dict(three_backends=len(labels)==3,local_gates=all(all(r['gates'].values()) for r in reports.values()),
               initial_exact=all(c['initial_exact'] for c in comparisons),
               all_output_bytes_exact=all(not c['differences'] for c in comparisons))
    (ROOT/'reports/cross_hardware_comparison.json').write_text(json.dumps(dict(labels=labels,gates=gates,comparisons=comparisons),indent=2)+'\n')
    print(json.dumps(dict(gates=gates,comparisons=comparisons)),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__':
    if len(sys.argv)==5 and sys.argv[1]=='--worker':worker(*sys.argv[2:]);raise SystemExit(0)
    if len(sys.argv)==3 and sys.argv[1]=='--capture':raise SystemExit(capture(sys.argv[2]))
    if len(sys.argv)==5 and sys.argv[1]=='--compare':raise SystemExit(compare(sys.argv[2:]))
    raise SystemExit('Use --capture LABEL or --compare LABEL LABEL LABEL')
