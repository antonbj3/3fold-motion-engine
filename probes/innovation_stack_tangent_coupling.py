"""Observe normal-only correction coupling without changing the frozen engine."""

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
from scipy.optimize import nnls

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from innovation_stack_normal_space import SnapshotEngine,basis,SAMPLES
from engine_metrics import m5_penetration


class CouplingSnapshots(SnapshotEngine):
    def _color_pairs(self,pairs):
        colors=super()._color_pairs(pairs)
        if self.calls in SAMPLES:
            self.snapshots[-1]['jn']=self.jn.numpy()[:self.n_contacts_solved].copy()
        return colors


def measure(s):
    c,n=s['contacts'],len(s['xc'])
    normals=s['sn'].astype(float)
    t1,t2=SnapshotEngine._plane_basis(normals)
    directions=np.stack([normals,t1,t2],axis=1)
    orth_error=float(np.max(np.abs(np.einsum('cki,cli->ckl',directions,directions)-np.eye(3))))
    jac=np.zeros((c,3,n,6))
    for k in range(c):
        a,b=s['sbi'][k],s['sbj'][k]
        for q in range(3):
            d=directions[k,q]
            jac[k,q,a,:3]=d;jac[k,q,a,3:]=np.cross(s['spA'][k].astype(float)-s['xc'][a],d)
            if b>=0:
                jac[k,q,b,:3]=-d;jac[k,q,b,3:]=-np.cross(s['spB'][k].astype(float)-s['xc'][b],d)
    eig,vec=np.linalg.eigh(s['invIw'].astype(float))
    root=(vec*np.sqrt(eig)[:,None,:])@vec.transpose(0,2,1)
    mass=s['invM'].astype(float)
    weighted=jac.copy()
    weighted[:,:,:,:3]*=np.sqrt(mass)[None,None,:,None]
    weighted[:,:,:,3:]=np.einsum('cqni,nij->cqnj',jac[:,:,:,3:],root)
    normal=weighted[:,0].reshape(c,-1)
    tangent=weighted[:,1:].reshape(2*c,-1)
    bn,_,rn=basis(normal);bt,_,rt=basis(tangent)
    _,_,rjoint=basis(np.vstack([normal,tangent]))
    norms_n=np.linalg.norm(normal,axis=1);norms_t=np.linalg.norm(tangent,axis=1)
    cross=normal@tangent.T/norms_n[:,None]/norms_t[None,:]
    u=np.concatenate((s['v'],s['w']),axis=1).astype(float)
    u[:,:3]/=np.sqrt(mass)[:,None]
    u[:,3:]=np.linalg.solve(root,u[:,3:,None])[:,:,0]
    rhs=normal.T@s['jn'].astype(float)-u.ravel()
    solution,_=nnls(normal.T,rhs,maxiter=3*c)
    grad=normal@(normal.T@solution-rhs)
    kkt=max(float(np.maximum(-grad,0).max()),float(np.abs(solution*grad).max()))
    delta=normal.T@(solution-s['jn'].astype(float))
    digest=hashlib.sha256()
    for key in sorted(s):
        digest.update(key.encode());digest.update(np.ascontiguousarray(s[key]).tobytes())
    return {'call':s['call'],'contacts':c,'normal_rank':rn,'tangent_rank':rt,
            'intersection_rank':rn+rt-rjoint,'max_normalized_cross_coupling':float(np.abs(cross).max()),
            'normal_velocity_before_norm':float(np.linalg.norm(normal@u.ravel())),
            'normal_velocity_after_norm':float(np.linalg.norm(normal@(u.ravel()+delta))),
            'induced_tangent_velocity_norm':float(np.linalg.norm(tangent@delta)),
            'tangent_velocity_before_norm':float(np.linalg.norm(tangent@u.ravel())),
            'basis_error':orth_error,'normal_kkt':kkt,'snapshot_sha256':digest.hexdigest()}


def main():
    if sys.argv[1:]==['--worker']:
        import warp as wp
        wp.init()
        e=CouplingSnapshots()
        m5=m5_penetration(lambda:e,'gpu',16,600)
        print('RESULT '+json.dumps({'m5':m5,'rows':[measure(s) for s in e.snapshots]}),flush=True)
        return 0
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
    frozen=json.loads((ROOT/'reports/innovation_cloud_stack_l4.json').read_text())['legs'][0]['m5']['40']
    rows=[r for leg in legs for r in leg['rows']]
    gates={'two_full_results_identical':legs[0]==legs[1],
           'unchanged_frozen_m5':all(leg['m5']==frozen for leg in legs),
           'all_samples':all(tuple(r['call'] for r in leg['rows'])==SAMPLES for leg in legs),
           'finite_and_basis':all(np.isfinite([v for k,v in r.items() if isinstance(v,(int,float))]).all()
                                  and r['basis_error']<=1e-6 for r in rows),
           'normal_kkt':all(r['normal_kkt']<=1e-8 for r in rows)}
    report={'legs':legs,'gates':gates,'physical_k16_pass':all(leg['m5']['pass'] for leg in legs),
            'scope':'nonperturbing synthetic snapshot coupling; no new solver or timing claim'}
    (ROOT/'reports/innovation_stack_tangent_coupling.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
