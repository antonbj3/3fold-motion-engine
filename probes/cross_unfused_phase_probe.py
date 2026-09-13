"""Locate cross-hardware divergence while checking observer parity."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from cross_unfused_state_probe import ROOT,factories,fingerprint,stack


def worker(label,engine,output):
    e=factories()[engine](vit=40,pit=10)
    for pos in stack(8):e.add_body(pos)
    e._build()
    records=[];arrays={};step=0
    original=e._stage
    def record(name):
        c=int(e.cnt.numpy()[0]);data={n:getattr(e,n).numpy().copy() for n in ('xc','q','v','w','invIw')}
        if name=='generate':
            fields={n:getattr(e,n).numpy()[:c].copy() for n in ('cfa','cfb','cbi','cbj','cpA','cpB','cn','cpen')}
            order=np.lexsort((fields['cfb'],fields['cfa']))
            data.update({n:a[order] for n,a in fields.items()})
        if name=='sort':
            data.update({n:getattr(e,n).numpy()[:c].copy() for n in ('skey','tbi','tbj','tpA','tpB','tn','tpen','pkey_s','pval_s')})
        if name in ('select','pairs','colour','solve','warm_store','integrate') and c>0:
            kept=int(e.dflags.numpy()[0])
            data.update({n:getattr(e,n).numpy()[:kept].copy() for n in ('ckeys','sbi','sbj','spA','spB','sn','spen','jn','jt1','jt2','jp')})
        if name in ('colour','solve','warm_store'):
            pairs=e.n_pairs
            data.update({n:getattr(e,n).numpy()[:pairs].copy() for n in ('pbi','pbj','pstart','pcount','colP','porder')})
            data.update({n:getattr(e,n).numpy().copy() for n in ('pv','po')})
        keys={}
        for field,a in data.items():
            key=f'{step}__{name}__{field}'
            arrays[key]=np.ascontiguousarray(a);keys[field]=fingerprint(a)
        records.append(dict(step=step,phase=name,fields=keys))
    @contextmanager
    def observed(name):
        with original(name):yield
        record(name)
    e._stage=observed
    contract={name:[] for name in ('xc','Rm','vc','om','force')}
    for step in range(30):
        e.step(1/240,substeps=1);s=e.get_state()
        for name in ('xc','Rm','vc','om'):contract[name].append(np.array(getattr(s,name),copy=True))
        contract['force'].append(np.array(e.contact_forces(),copy=True))
    contract={k:np.ascontiguousarray(v) for k,v in contract.items()}
    with np.load(ROOT/f'reports/cross_unfused_{label}.npz') as reference:
        parity=all(a.tobytes()==reference[f'{engine}__stack8__0__{k}'][:30].tobytes() for k,a in contract.items())
    for k,a in contract.items():arrays['contract__'+k]=a
    np.savez_compressed(output,**arrays)
    print('RESULT '+json.dumps(dict(records=records,contract={k:fingerprint(a) for k,a in contract.items()},parity=parity)),flush=True)


def main(label):
    rows=[];all_arrays={}
    for engine in ('pair_chunks','color_cache'):
        legs=[];saved=[]
        with tempfile.TemporaryDirectory(prefix='phase-probe-') as tmp:
            for leg in range(2):
                path=Path(tmp)/f'{leg}.npz'
                p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',label,engine,str(path)],cwd=ROOT,capture_output=True,text=True)
                if p.returncode:print(p.stdout);print(p.stderr,file=sys.stderr);raise RuntimeError('Phase worker failed')
                lines=[line[7:] for line in p.stdout.splitlines() if line.startswith('RESULT ')]
                if len(lines)!=1:raise RuntimeError('Expected one phase receipt')
                legs.append(json.loads(lines[0]))
                with np.load(path) as data:saved.append({k:data[k].copy() for k in data.files})
        exact=legs[0]==legs[1] and all(saved[0][k].tobytes()==saved[1][k].tobytes() for k in saved[0])
        rows.append(dict(engine=engine,legs=legs,exact=exact))
        for leg,arrays in enumerate(saved):
            for k,a in arrays.items():all_arrays[f'{engine}__{leg}__{k}']=a
    gates=dict(exact_repeats=all(r['exact'] for r in rows),unmodified_contract=all(leg['parity'] for r in rows for leg in r['legs']),
               finite=bool(all(np.isfinite(a).all() for a in all_arrays.values())))
    np.savez_compressed(ROOT/f'reports/cross_unfused_phase_{label}.npz',**all_arrays)
    (ROOT/f'reports/cross_unfused_phase_{label}.json').write_text(json.dumps(dict(rows=rows,gates=gates),indent=2)+'\n')
    print(json.dumps(gates),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__':
    if len(sys.argv)==5 and sys.argv[1]=='--worker':worker(*sys.argv[2:]);raise SystemExit(0)
    if len(sys.argv)==2:raise SystemExit(main(sys.argv[1]))
    raise SystemExit('Expected backend label')
