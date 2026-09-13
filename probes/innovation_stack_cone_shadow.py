"""Convex cone shadow calculation on frozen snapshots; never apply its impulses."""

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
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from innovation_stack_tangent_coupling import CouplingSnapshots
from engine_metrics import m5_penetration


def project(x,mu=.5):
    p=x.reshape(-1,3);radius=np.linalg.norm(p[:,1:],axis=1)
    inside=(p[:,0]>=0)&(radius<=mu*p[:,0])
    height=np.maximum(0.,(p[:,0]+mu*radius)/(1+mu*mu))
    result=np.zeros_like(p);result[:,0]=height
    result[:,1:]=p[:,1:]*(mu*height/np.maximum(radius,1e-300))[:,None]
    result[inside]=p[inside]
    return result.ravel()


def solve(s):
    c,n=s['contacts'],len(s['xc'])
    normals=s['sn'].astype(float);t1,t2=CouplingSnapshots._plane_basis(normals)
    directions=np.stack([normals,t1,t2],axis=1)
    jac=np.zeros((c,3,n,6))
    for i in range(c):
        a,b=s['sbi'][i],s['sbj'][i]
        for k,d in enumerate(directions[i]):
            jac[i,k,a,:3]=d;jac[i,k,a,3:]=np.cross(s['spA'][i].astype(float)-s['xc'][a],d)
            if b>=0:
                jac[i,k,b,:3]=-d;jac[i,k,b,3:]=-np.cross(s['spB'][i].astype(float)-s['xc'][b],d)
    eig,vec=np.linalg.eigh(s['invIw'].astype(float));mass=s['invM'].astype(float)
    root=(vec*np.sqrt(eig)[:,None,:])@vec.transpose(0,2,1)
    weighted=jac.copy();weighted[:,:,:,:3]*=np.sqrt(mass)[None,None,:,None]
    weighted[:,:,:,3:]=np.einsum('cqni,nij->cqnj',jac[:,:,:,3:],root)
    operator=weighted.reshape(3*c,-1).T
    normal=weighted[:,0].reshape(c,-1).T
    u=np.concatenate([s['v'],s['w']],axis=1).astype(float)
    u[:,:3]/=np.sqrt(mass)[:,None];u[:,3:]=np.linalg.solve(root,u[:,3:,None])[:,:,0]
    free=u.ravel()-normal@s['jn'].astype(float)
    lam,_=nnls(normal,-free,maxiter=3*c)
    x=np.zeros((c,3));x[:,0]=lam;x=x.ravel()
    initial=x.copy();normal_velocity=free+operator@x
    gram=operator.T@operator;linear=operator.T@free
    lipschitz=float(np.linalg.eigvalsh(gram)[-1])*1.0000001
    objective=lambda v:float(np.dot(free+operator@v,free+operator@v)*.5)
    energy0=energy=objective(x);y=x.copy();momentum=1.
    residual=float('inf')
    for iteration in range(1,40001):
        proposed=project(y-(gram@y+linear)/lipschitz)
        cost=objective(proposed)
        if cost>energy:
            y=x;momentum=1.
            proposed=project(y-(gram@y+linear)/lipschitz);cost=objective(proposed)
        if cost<=energy:
            next_momentum=(1+np.sqrt(1+4*momentum*momentum))*.5
            y=proposed+(momentum-1)/next_momentum*(proposed-x)
            x=proposed;energy=cost;momentum=next_momentum
        else:
            y=x;momentum=1.
        residual=float(np.max(np.abs(x-project(x-(gram@x+linear)))))
        if residual<=1e-8:break
    p=x.reshape(c,3);final=free+operator@x
    cone=max(float(np.maximum(-p[:,0],0).max()),float(np.maximum(np.linalg.norm(p[:,1:],axis=1)-.5*p[:,0],0).max()))
    velocities=(operator.T@final).reshape(c,3)
    initial_velocities=(operator.T@normal_velocity).reshape(c,3)
    return {'call':s['call'],'contacts':c,'iterations':iteration,'projected_residual':residual,
            'cone_violation':cone,'finite':bool(np.isfinite(x).all()),'normal_only_energy':energy0,'coupled_energy':energy,
            'normal_only_normal_velocity_norm':float(np.linalg.norm(initial_velocities[:,0])),
            'normal_only_tangent_velocity_norm':float(np.linalg.norm(initial_velocities[:,1:])),
            'coupled_normal_velocity_norm':float(np.linalg.norm(velocities[:,0])),
            'coupled_tangent_velocity_norm':float(np.linalg.norm(velocities[:,1:])),
            'solution_sha256':hashlib.sha256(x.tobytes()+final.tobytes()).hexdigest()}


def main():
    if sys.argv[1:]==['--worker']:
        import warp as wp
        wp.init();e=CouplingSnapshots();m5=m5_penetration(lambda:e,'gpu',16,600)
        print('RESULT '+json.dumps({'m5':m5,'rows':[solve(s) for s in e.snapshots]}),flush=True);return 0
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
    frozen=json.loads((ROOT/'reports/innovation_cloud_stack_l4.json').read_text())['legs'][0]['m5']['40']
    rows=[r for leg in legs for r in leg['rows']]
    gates={'two_full_results_identical':legs[0]==legs[1],
           'unchanged_m5':all(leg['m5']==frozen for leg in legs),
           'finite_cone_feasible':all(r['finite'] and r['cone_violation']<=1e-12 for r in rows),
           'projected_residual':all(r['projected_residual']<=1e-8 for r in rows),
           'energy_not_worse':all(r['coupled_energy']<=r['normal_only_energy'] for r in rows)}
    report={'legs':legs,'gates':gates,'mu':.5,'max_iterations':40000,
            'scope':'convex energy surrogate on frozen synthetic snapshots; no physical engine correction'}
    (ROOT/'reports/innovation_stack_cone_shadow.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
