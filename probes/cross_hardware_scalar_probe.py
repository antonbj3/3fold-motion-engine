"""Record solve assignments with explicit frozen-contract parity."""

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
from cross_hardware_state_probe import ROOT,factories,fingerprint,stack


def worker(label,engine,output):
    e=factories()[engine](vit=40,pit=10)
    for pos in stack(8):e.add_body(pos)
    e._build()
    e.graph_capture=False
    def uncaptured_graph(pairs,ncol,sdt):
        dim=1
        while dim<max(pairs,1):dim*=2
        return lambda:e._chunk_launch(ncol,dim,sdt)
    e._solve_chunk_graph=uncaptured_graph
    original_capture_launch=e.wp.capture_launch
    def observed_capture_launch(graph,*args,**kwargs):
        if callable(graph):return graph()
        return original_capture_launch(graph,*args,**kwargs)
    e.wp.capture_launch=observed_capture_launch
    from contact_solver_scalar_observer import pair_trace,fixed_trace
    schema=json.loads((ROOT/'reports/cross_scalar_schema.json').read_text())
    trace_kernel=pair_trace if engine=='pair_chunks' else fixed_trace
    columns=len(schema['pair_trace' if engine=='pair_chunks' else 'fixed_trace'])
    trace=e.wp.zeros((8,4,columns),dtype=e.wp.float64,device=e.dev)
    seen=e.wp.zeros((8,4,columns),dtype=int,device=e.dev)
    records=[];arrays={};step=0;launch_index=0
    original=e._stage
    def record(name):
        c=int(e.cnt.numpy()[0]);data={n:getattr(e,n).numpy().copy() for n in ('xc','q','v','w','invIw')}
        if name=='generate':
            fields={n:getattr(e,n).numpy()[:c].copy() for n in ('cfa','cfb','cbi','cbj','cpA','cpB','cn','cpen')}
            order=np.lexsort((fields['cfb'],fields['cfa']))
            data.update({n:a[order] for n,a in fields.items()})
        if name=='sort':
            data.update({n:getattr(e,n).numpy()[:c].copy() for n in ('skey','tbi','tbj','tpA','tpB','tn','tpen','pkey_s','pval_s')})
        if (name.startswith('kernel:') or name in ('select','pairs','colour','solve','warm_store','integrate')) and c>0:
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
    original_launch=e.wp.launch
    watched={e.KC['chunk_vel_r'],e.KC['chunk_pos_r']}
    if hasattr(e,'_apply_warm'):watched.add(e._apply_warm)
    def observed_launch(kernel,*args,**kwargs):
        nonlocal launch_index
        if kernel in watched and step=={'pair_chunks':3,'color_cache':8}[engine] and launch_index=={'pair_chunks':1,'color_cache':3}[engine]:
            kwargs=dict(kwargs);kwargs['inputs']=[*kwargs['inputs'],trace,seen]
            result=original_launch(trace_kernel,*args,**kwargs)
        else:
            result=original_launch(kernel,*args,**kwargs)
        if kernel in watched and step=={'pair_chunks':3,'color_cache':8}[engine]:
            record(f'kernel:{launch_index}:{kernel.key}')
            launch_index+=1
        return result
    e.wp.launch=observed_launch
    contract={name:[] for name in ('xc','Rm','vc','om','force')}
    for step in range(10):
        e.step(1/240,substeps=1);s=e.get_state()
        for name in ('xc','Rm','vc','om'):contract[name].append(np.array(getattr(s,name),copy=True))
        contract['force'].append(np.array(e.contact_forces(),copy=True))
    contract={k:np.ascontiguousarray(v) for k,v in contract.items()}
    with np.load(ROOT/f'reports/cross_hardware_{label}.npz') as reference:
        parity=all(a.tobytes()==reference[f'{engine}__stack8__0__{k}'][:10].tobytes() for k,a in contract.items())
    arrays['scalar_trace']=trace.numpy();arrays['scalar_seen']=seen.numpy()
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
               coverage=len(rows)==2 and all(np.any(a) for k,a in all_arrays.items() if k.endswith('scalar_seen')),
               finite=bool(all(np.isfinite(a).all() for k,a in all_arrays.items() if not k.endswith('scalar_trace'))))
    np.savez_compressed(ROOT/f'reports/cross_scalar_{label}.npz',**all_arrays)
    (ROOT/f'reports/cross_scalar_{label}.json').write_text(json.dumps(dict(rows=rows,gates=gates),indent=2)+'\n')
    print(json.dumps(gates),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__':
    if len(sys.argv)==5 and sys.argv[1]=='--worker':worker(*sys.argv[2:]);raise SystemExit(0)
    if len(sys.argv)==2:raise SystemExit(main(sys.argv[1]))
    raise SystemExit('Expected backend label')
