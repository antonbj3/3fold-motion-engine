"""Capture complete contract outputs for explicit cross-hardware comparison."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from motion_engine.contact_engine_gpu_complement_normal import ComplementNormalContactEngine

def factories():
    return {'complement_normal':lambda **kw:ComplementNormalContactEngine(dims=(.3,.3,.2),mu=.5,**kw)}

ENGINES=('complement_normal',)
SCENES={'ratio1':(1,200),'ratio1000':(1000,200)}


def fingerprint(a):
    a=np.ascontiguousarray(a)
    return dict(shape=list(a.shape),dtype=a.dtype.str,sha256=hashlib.sha256(a.tobytes()).hexdigest())


def worker(engine,scene,output):
    import warp as wp
    if wp.config.version!='1.13.0' or np.__version__!='2.5.3':raise RuntimeError('Pinned numeric environment required')
    e=factories()[engine](vit=40,pit=10)
    if e.dev!='cuda:0':raise RuntimeError('CUDA required')
    count,steps=SCENES[scene]
    e.add_body([0,0,.101],density=700.)
    e.add_body([0,0,.306],density=700.*count)
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
    result=dict(variant="complement_normal_hybrid",engine=engine,scene=scene,initial=initial,arrays={k:fingerprint(v) for k,v in arrays.items()},
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
    gates=dict(coverage=len(records)==2,exact_repeats=all(r['exact'] for r in records),
               finite=all(leg['finite'] for r in records for leg in r['legs']),
               active_contacts=all(max(leg['contacts'])>0 for r in records for leg in r['legs']))
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip()
    archive=ROOT/f'reports/cross_complement_{label}.npz'
    np.savez_compressed(archive,**saved)
    report=dict(label=label,hardware=hardware,records=records,gates=gates,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (ROOT/f'reports/cross_complement_{label}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(gates),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__':
    if len(sys.argv)==5 and sys.argv[1]=='--worker':worker(*sys.argv[2:]);raise SystemExit(0)
    if len(sys.argv)==3 and sys.argv[1]=='--capture':raise SystemExit(capture(sys.argv[2]))
    raise SystemExit('Use --capture LABEL')
