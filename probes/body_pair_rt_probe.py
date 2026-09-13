"""Compare complete bounded RT rebuilds with unchanged hash-grid pair lists."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'probes'),str(ROOT/'scripts'),str(ROOT/'src')]
import json
import subprocess
import tempfile
import time
import numpy as np
from body_pair_scale_probe import fixtures
from cross_hardware_state_probe import fingerprint
from motion_engine.contact_body_pairs_gpu import BodyPairSearch
from motion_engine.contact_body_pairs_rt import RTBodyPairSearch


def controls():
    points=np.array([[0,0,.05],[.1,0,.05],
        [np.nextafter(np.float32(.1),np.float32(0)),0,.05],
        [np.nextafter(np.float32(.1),np.float32(np.inf)),0,.05],
        [0,.1,.0499],[0,-.1,.0501],[0,0,.15],[-.1,0,-.01]],np.float32)
    yield 'boundary',points,1,0
    yield 'translated',points+np.array([9999.,-9999.,2.],np.float32),1,0
    yield 'empty',np.array([[0,0,1],[2,0,1]],np.float32),1,0


def worker(reverse,output):
    import warp as wp
    wp.init()
    if wp.config.version!='1.13.0' or np.__version__!='2.5.3':raise RuntimeError('Pinned environment required')
    rows={};saved={};cases=list(fixtures())+list(controls())
    for name,points,width,_ in (cases[::-1] if reverse else cases):
        gpu=wp.array(points,dtype=wp.vec3,device='cuda:0')
        baseline=BodyPairSearch(wp,len(points)//width,width)
        with RTBodyPairSearch(wp,len(points)//width,width) as candidate:
            searches=dict(baseline=baseline,candidate=candidate);times={};arrays={}
            for kind in (('candidate','baseline') if reverse else ('baseline','candidate')):
                search=searches[kind]
                for _ in range(3):search.rebuild(gpu)
                wp.synchronize();begin=time.perf_counter()
                for _ in range(20):search.rebuild(gpu)
                wp.synchronize();times[kind]=(time.perf_counter()-begin)*1000/20
                arrays[kind]=dict(pairs=search.pairs.numpy()[:search.number].copy(),lower=search.lower.numpy(),upper=search.upper.numpy())
                saved.update({name+'__'+kind+'__'+k:v for k,v in arrays[kind].items()})
            exact=all(arrays['baseline'][k].dtype==arrays['candidate'][k].dtype and arrays['baseline'][k].shape==arrays['candidate'][k].shape and arrays['baseline'][k].tobytes()==arrays['candidate'][k].tobytes() for k in arrays['baseline'])
            rows[name]=dict(count=baseline.number,exact=exact,timing_ms=times,
                hashes={kind:{k:fingerprint(v) for k,v in a.items()} for kind,a in arrays.items()})
            print(name+' complete',flush=True)
    valid=np.array([[0,0,.01],[.06,0,.01]],np.float32)
    gpu=wp.array(valid,dtype=wp.vec3,device='cuda:0');refused={}
    options=[('capacity',dict(capacity=0)),('slot_capacity',dict(slot_capacity=1)),
             ('radius',dict(radius_limit=.01)),('coordinate',dict(coordinate_limit=.5))]
    for name,kw in options:
        with RTBodyPairSearch(wp,2,1,**kw) as search:
            try:search.rebuild(gpu);raised=False
            except RuntimeError:raised=True
            refused[name]=raised and search.number==0
    for name,points in [('nonfinite',np.array([[np.nan,0,.01],[.06,0,.01]],np.float32)),('layout',valid[:1])]:
        with RTBodyPairSearch(wp,2,1) as search:
            search.rebuild(gpu)
            if not search.number:raise RuntimeError('Prior valid state required')
            try:search.rebuild(wp.array(points,dtype=wp.vec3,device='cuda:0'));raised=False
            except (RuntimeError,ValueError):raised=True
            refused[name]=raised and search.number==0
    search=RTBodyPairSearch(wp,2,1);search.rebuild(gpu);search.close();search.close()
    try:search.rebuild(gpu);raised=False
    except RuntimeError:raised=True
    refused['closed']=raised and search.number==0
    for name,kw in [('body_limit',dict(bodies=100001)),('slot_limit',dict(slot_capacity=257)),
                    ('spatial_contract',dict(radius_limit=.36)),('coordinate_contract',dict(coordinate_limit=10001))]:
        args=dict(bodies=2,width=1);args.update(kw)
        try:search=RTBodyPairSearch(wp,**args);search.close();raised=False
        except ValueError:raised=True
        refused[name]=raised
    np.savez_compressed(output,**saved)
    print('RESULT '+json.dumps(dict(rows=rows,refused=refused)),flush=True)


def main():
    if len(sys.argv)==4 and sys.argv[1]=='--worker':worker(sys.argv[2]=='1',sys.argv[3]);return 0
    results=[];arrays=[]
    with tempfile.TemporaryDirectory() as temp:
        for leg in range(2):
            output=Path(temp)/f'{leg}.npz'
            p=subprocess.run([sys.executable,__file__,'--worker',str(leg),str(output)],cwd=ROOT,capture_output=True,text=True,timeout=240)
            if p.returncode:raise RuntimeError(p.stdout+p.stderr)
            results.append(json.loads(next(s[7:] for s in p.stdout.splitlines() if s.startswith('RESULT '))))
            with np.load(output) as data:arrays.append({k:data[k].copy() for k in data.files})
            print('leg '+str(leg)+' complete',flush=True)
    def fixed(result):
        result=json.loads(json.dumps(result))
        for row in result['rows'].values():row.pop('timing_ms')
        return result
    rows=[r for result in results for r in result['rows'].values()]
    gates=dict(exact_baseline=all(r['exact'] for r in rows) and all(r['rows']['empty']['count']==0 for r in results),
        exact_repeats=fixed(results[0])==fixed(results[1]) and arrays[0].keys()==arrays[1].keys() and all(arrays[0][k].tobytes()==arrays[1][k].tobytes() for k in arrays[0]),
        refusals=all(len(r['refused'])==11 and all(r['refused'].values()) for r in results),
        finite_complete=len(rows)==10 and all(np.isfinite(v) and v>0 for r in rows for v in r['timing_ms'].values()) and all(np.isfinite(v).all() for a in arrays for v in a.values()),
        faster_full_rebuild=all(r['rows'][n]['timing_ms']['candidate']<r['rows'][n]['timing_ms']['baseline'] for r in results for n in ('10000','100000')))
    np.savez_compressed(ROOT/'reports/body_pair_rt_probe.npz',**{str(i)+'__'+k:v for i,a in enumerate(arrays) for k,v in a.items()})
    report=dict(gates=gates,results=results,scope='Full BVH rebuild and canonical pairs; bounded synthetic fixtures; no full-step or cross-device claim.')
    (ROOT/'reports/body_pair_rt_probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
