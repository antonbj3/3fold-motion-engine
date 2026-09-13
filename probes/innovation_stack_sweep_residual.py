"""Observe frozen post-solve contact update residuals without applying updates."""

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
from innovation_stack_normal_space import SnapshotEngine,SAMPLES
from engine_metrics import m5_penetration


class PostSolveSnapshots(SnapshotEngine):
    def __init__(self,vit):
        super().__init__();self.vit=vit;self.post=[]

    def _store_warm_device(self,c):
        super()._store_warm_device(c)
        if self.calls in SAMPLES:
            s={'call':self.calls,'contacts':c}
            for name in ('sbi','sbj','spA','spB','sn','jn','jt1','jt2'):
                s[name]=getattr(self,name).numpy()[:c].copy()
            for name in ('xc','v','w','invM','invIw'):
                s[name]=getattr(self,name).numpy().copy()
            self.post.append(s)


def measure(s):
    normal_updates=[];tangent_updates=[];closing=[]
    for c in range(s['contacts']):
        a,b=int(s['sbi'][c]),int(s['sbj'][c]);n=s['sn'][c].astype(float)
        ra=s['spA'][c].astype(float)-s['xc'][a]
        rb=s['spB'][c].astype(float)-s['xc'][b] if b>=0 else np.zeros(3)
        ma=float(s['invM'][a]);mb=float(s['invM'][b]) if b>=0 else 0.
        ia=s['invIw'][a].astype(float);ib=s['invIw'][b].astype(float) if b>=0 else np.zeros((3,3))
        va=s['v'][a].astype(float).copy();wa=s['w'][a].astype(float).copy()
        vb=s['v'][b].astype(float).copy() if b>=0 else np.zeros(3)
        wb=s['w'][b].astype(float).copy() if b>=0 else np.zeros(3)
        rel=va+np.cross(wa,ra)-vb-np.cross(wb,rb)
        vn=float(rel@n);vt=rel-vn*n;mt=np.linalg.norm(vt)
        if mt>5e-3:t1=vt/mt
        else:
            axis=np.array([1.,0.,0.]) if abs(n[0])<.9 else np.array([0.,1.,0.])
            t1=axis-(axis@n)*n;t1/=np.linalg.norm(t1)
        t2=np.cross(n,t1)
        def effective(d):
            ca=np.cross(ra,d);cb=np.cross(rb,d)
            return ma+mb+ca@ia@ca+cb@ib@cb
        me=effective(n)
        if me<1e-12:continue
        old=float(s['jn'][c]);new=max(0.,old-vn/me);delta=new-old;j=delta*n
        va+=ma*j;wa+=ia@np.cross(ra,j);vb-=mb*j;wb-=ib@np.cross(rb,j)
        rel=va+np.cross(wa,ra)-vb-np.cross(wb,rb)
        oldt=np.array([s['jt1'][c],s['jt2'][c]],dtype=float)
        eff=np.array([effective(t1),effective(t2)])
        if np.min(eff)<1e-12:continue
        newt=oldt-np.array([rel@t1,rel@t2])/eff
        radius=np.linalg.norm(newt);limit=.5*new
        if radius>limit and radius>1e-12:newt*=limit/radius
        normal_updates.append(abs(delta));tangent_updates.append(float(np.linalg.norm(newt-oldt)))
        closing.append(max(0.,-vn))
    normal=np.array(normal_updates);tangent=np.array(tangent_updates)
    cone=np.hypot(s['jt1'].astype(float),s['jt2'].astype(float))-.5*s['jn']
    digest=hashlib.sha256()
    for key in sorted(s):digest.update(key.encode());digest.update(np.ascontiguousarray(s[key]).tobytes())
    return {'call':s['call'],'contacts':s['contacts'],'normal_update_max':float(normal.max()),
            'normal_update_l2':float(np.linalg.norm(normal)),'tangent_update_max':float(tangent.max()),
            'tangent_update_l2':float(np.linalg.norm(tangent)),'closing_velocity_max':float(max(closing)),
            'minimum_normal_impulse':float(s['jn'].min()),'cone_violation':float(max(0.,cone.max())),
            'finite':bool(np.isfinite(normal).all() and np.isfinite(tangent).all() and np.isfinite(cone).all()),
            'snapshot_sha256':digest.hexdigest()}


def main():
    if sys.argv[1:]==['--worker']:
        import warp as wp
        wp.init();result={}
        for vit in (40,160):
            e=PostSolveSnapshots(vit);m5=m5_penetration(lambda:e,'gpu',16,600)
            result[str(vit)]={'m5':m5,'rows':[measure(s) for s in e.post]}
        print('RESULT '+json.dumps(result),flush=True);return 0
    if sys.argv[1:]:raise ValueError('Expected no arguments or --worker')
    from innovation_stack_probe import idle
    legs=[]
    for _ in range(2):
        idle();child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=ROOT,capture_output=True,text=True)
        if child.returncode:
            print(child.stdout);print(child.stderr,file=sys.stderr);raise RuntimeError('Worker failed')
        idle();lines=[line[7:] for line in child.stdout.splitlines() if line.startswith('RESULT ')]
        if len(lines)!=1:raise RuntimeError('Expected one result')
        legs.append(json.loads(lines[0]));print(lines[0],flush=True)
    frozen=json.loads((ROOT/'reports/innovation_cloud_stack_l4.json').read_text())['legs'][0]['m5']
    rows=[r for leg in legs for v in leg.values() for r in v['rows']]
    gates={'two_results_identical':legs[0]==legs[1],
           'unchanged_m5':all(leg[k]['m5']==frozen[k] for leg in legs for k in ('40','160')),
           'finite':all(r['finite'] for r in rows),
           'impulse_feasible':all(r['minimum_normal_impulse']>=-1e-12 and r['cone_violation']<=1e-6 for r in rows),
           'all_snapshots':all([r['call'] for r in v['rows']]==list(SAMPLES) for leg in legs for v in leg.values())}
    report={'legs':legs,'gates':gates,'scope':'post-solve isolated float64 update diagnostic, no engine correction'}
    (ROOT/'reports/innovation_stack_sweep_residual.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
