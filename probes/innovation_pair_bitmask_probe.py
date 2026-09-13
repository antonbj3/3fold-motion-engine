"""Certify raw pair-bitmask contact geometry and measure isolated generation."""

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
import time
import numpy as np
from cross_hardware_state_probe import ROOT,fingerprint
from innovation_stack_probe import idle
from motion_engine.contact_engine_gpu import _MAXC
from motion_engine.contact_generation_pair_bitmask import PairBitmaskGenerator


def canonical(arrays,points):
    keys=arrays[6].astype(np.int64)*(points+1)+arrays[7].astype(np.int64)+1
    order=np.argsort(keys,kind='stable')
    return [a[order] for a in arrays],keys[order]


def fixtures():
    report=json.loads((ROOT/'reports/innovation_body_pair_observer.json').read_text())
    with np.load(ROOT/'reports/innovation_body_pair_observer.npz') as data:
        for scene,rows in report['records']['first']['records'].items():
            for row in rows:
                name=scene+'__'+str(row['step']);prefix='first__'+name
                points=data[prefix+'__points'].copy();pairs=data[prefix+'__candidate_pairs'].copy();width=len(points)//row['bodies']
                radius=float(np.float32(.05))+np.float32(row['radius_padding'])
                ground=np.flatnonzero(points.astype(np.float64).reshape(-1,width,3).min(axis=1)[:,2]-radius<0)
                pairs=np.concatenate((pairs,np.column_stack((ground,np.full(len(ground),-1)))),axis=0)
                pairs=pairs[np.lexsort((pairs[:,1],pairs[:,0]))]
                yield name,points,width,pairs,data[prefix+'__keys'].copy()
    points=np.array([[0,0,.049],[0,0,.049],[.1,0,.049],
        [np.nextafter(np.float32(.1),np.float32(0)),0,.049],
        [np.nextafter(np.float32(.1),np.float32(np.inf)),0,.049],[-.1,0,.051],[1e-10,0,.05],[1,0,-.01]],np.float32)
    pairs=np.array([(a,b) for a in range(len(points)) for b in [-1,*range(a+1,len(points))]],np.int64)
    yield 'cutoff_control',points,1,pairs,None


class Frozen:
    def __init__(self,wp,points,width):
        from motion_engine.contact_engine_gpu_colored import _build_colored_kernels
        self.wp=wp;self.n=len(points);self.points=wp.array(points,dtype=wp.vec3,device='cuda:0')
        self.owner=wp.array(np.repeat(np.arange(len(points)//width,dtype=np.int32),width),dtype=int,device='cuda:0')
        self.grid=wp.HashGrid(256,256,8,device='cuda:0');self.grid.build(self.points,.1)
        self.count=wp.zeros(1,dtype=int,device='cuda:0')
        self.outputs=[wp.zeros(_MAXC,dtype=t,device='cuda:0') for t in (int,int,wp.vec3,wp.vec3,wp.vec3,float,int,int)]
        self.kernel=_build_colored_kernels(wp)['gen_fid']
    def launch(self):
        self.count.zero_()
        self.wp.launch(self.kernel,self.n,inputs=[self.points,self.owner,self.grid.id,self.count,*self.outputs],device='cuda:0')
    def result(self):
        count=int(self.count.numpy()[0])
        if count>_MAXC:raise RuntimeError('Frozen overflow')
        return [a.numpy()[:count].copy() for a in self.outputs]


def timed(wp,stage):
    stage.launch();wp.synchronize_device('cuda:0')
    with wp.ScopedCapture(device='cuda:0') as cap:stage.launch()
    for _ in range(5):wp.capture_launch(cap.graph)
    wp.synchronize_device('cuda:0');start=time.perf_counter()
    for _ in range(80):wp.capture_launch(cap.graph)
    wp.synchronize_device('cuda:0')
    return (time.perf_counter()-start)*1000/80


def worker(reverse,output):
    import warp as wp
    if wp.config.version!='1.13.0' or np.__version__!='2.5.3':raise RuntimeError('Pinned environment required')
    wp.init();rows={};saved={};cases=list(fixtures())
    for name,points,width,pairs,expected in (cases[::-1] if reverse else cases):
        stages={'frozen':Frozen(wp,points,width),'candidate':PairBitmaskGenerator(wp,points,width,pairs)}
        times={};outputs={};keys={}
        for kind in (('candidate','frozen') if reverse else ('frozen','candidate')):
            stage=stages[kind];times[kind]=timed(wp,stage)
            outputs[kind],keys[kind]=canonical(stage.result(),len(points))
            for i,a in enumerate(outputs[kind]):saved[name+'__'+kind+'__'+str(i)]=a
        rows[name]=dict(count=len(keys['frozen']),exact_reference=all(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes() for a,b in zip(outputs['frozen'],outputs['candidate'])),
            saved_ids_exact=expected is None or keys['frozen'].tobytes()==expected.tobytes(),
            finite=all(bool(np.isfinite(a).all()) for out in outputs.values() for a in out),
            hashes={kind:[fingerprint(a) for a in out] for kind,out in outputs.items()},timing_ms=times)
        print(name+' complete',flush=True)
    _,points,width,pairs,_=cases[-1]
    zero=PairBitmaskGenerator(wp,points,width,pairs,capacity=0);zero.launch()
    raised=False
    try:zero.result()
    except RuntimeError:raised=True
    capacity=dict(refused=raised,overflow=int(zero.overflow.numpy()[0]),
        count=int(zero.counts.numpy().sum()),untouched=all(not np.any(a.numpy()) for a in zero.outputs))
    bad=[]
    invalid=[dict(width=0),dict(width=33),dict(points=points.astype(np.float64)),dict(pairs=np.array([[0,-1],[0,-1]])),
             dict(pairs=np.array([[1,0]])),dict(pairs=np.array([[0,len(points)]])),dict(pairs=np.array([[0,-2]])),dict(capacity=_MAXC+1)]
    for change in invalid:
        try:PairBitmaskGenerator(wp,**(dict(points=points,width=width,pairs=pairs)|change));bad.append(False)
        except ValueError:bad.append(True)
    np.savez_compressed(output,**saved)
    print('RESULT '+json.dumps(dict(rows=rows,capacity=capacity,invalid=bad)),flush=True)


def main():
    if len(sys.argv)==4 and sys.argv[1]=='--worker':worker(sys.argv[2]=='1',sys.argv[3]);return 0
    results=[];arrays=[];background=[]
    with tempfile.TemporaryDirectory() as tmp:
        for leg in range(2):
            output=Path(tmp)/f'{leg}.npz';before=idle()
            p=subprocess.run([sys.executable,__file__,'--worker',str(leg),str(output)],cwd=ROOT,capture_output=True,text=True)
            if p.returncode:print(p.stdout);print(p.stderr);raise RuntimeError('Worker failed')
            lines=[s[7:] for s in p.stdout.splitlines() if s.startswith('RESULT ')]
            if len(lines)!=1:raise RuntimeError('Missing receipt')
            results.append(json.loads(lines[0]));background.append(dict(before=before,after=idle()))
            with np.load(output) as data:arrays.append({k:data[k].copy() for k in data.files})
            print('leg '+str(leg)+' complete',flush=True)
    def fixed(result):
        r=json.loads(json.dumps(result))
        for row in r['rows'].values():row.pop('timing_ms')
        return r
    rows=[r for result in results for r in result['rows'].values()]
    gates=dict(exact_reference=all(r['exact_reference'] for r in rows),saved_ids_exact=all(r['saved_ids_exact'] for r in rows),
        exact_repeats=fixed(results[0])==fixed(results[1]) and all(arrays[0][k].tobytes()==arrays[1][k].tobytes() for k in arrays[0]),
        finite_complete=all(len(r['rows'])==7 for r in results) and all(r['finite'] and all(np.isfinite(t) and t>0 for t in r['timing_ms'].values()) for r in rows),
        capacity_refusal=all(r['capacity']['refused'] and r['capacity']['untouched'] and r['capacity']['count']==r['capacity']['overflow']>0 for r in results),
        invalid_inputs=all(len(r['invalid'])==8 and all(r['invalid']) for r in results))
    np.savez_compressed(ROOT/'reports/innovation_pair_bitmask_probe.npz',**{str(i)+'__'+k:a for i,data in enumerate(arrays) for k,a in data.items()})
    (ROOT/'reports/innovation_pair_bitmask_probe.json').write_text(json.dumps(dict(results=results,gates=gates,background=background),indent=2)+'\n')
    print(json.dumps(dict(gates=gates,timing=[{k:v['timing_ms'] for k,v in r['rows'].items()} for r in results])),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
