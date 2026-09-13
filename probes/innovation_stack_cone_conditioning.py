"""Measure frozen contact operator conditioning without a solver change."""

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
from innovation_stack_tangent_coupling import CouplingSnapshots
from engine_metrics import m5_penetration


def measure(s):
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
    singular=np.linalg.svd(operator,compute_uv=False)
    cutoff=max(operator.shape)*np.finfo(float).eps*singular[0]
    positive=singular[singular>cutoff]
    gram=operator.T@operator;values=np.linalg.eigvalsh(gram)
    diagonal=np.diag(gram).reshape(c,3)
    return {'call':s['call'],'contacts':c,'columns':3*c,'rank':len(positive),
            'nullity':3*c-len(positive),'rank_cutoff':float(cutoff),
            'largest_singular':float(singular[0]),'smallest_positive_singular':float(positive[-1]),
            'positive_gram_condition':float((positive[0]/positive[-1])**2),
            'normal_diagonal_min':float(diagonal[:,0].min()),'normal_diagonal_max':float(diagonal[:,0].max()),
            'tangent_diagonal_min':float(diagonal[:,1:].min()),'tangent_diagonal_max':float(diagonal[:,1:].max()),
            'symmetry_error':float(np.max(np.abs(gram-gram.T))),
            'min_eigenvalue':float(values[0]),'max_eigenvalue':float(values[-1]),
            'finite':bool(np.isfinite(singular).all() and np.isfinite(gram).all()),
            'operator_sha256':hashlib.sha256(operator.tobytes()+free.tobytes()).hexdigest()}

def main():
    if sys.argv[1:]==['--worker']:
        import warp as wp
        wp.init();e=CouplingSnapshots();m5=m5_penetration(lambda:e,'gpu',16,600)
        print('RESULT '+json.dumps({'m5':m5,'rows':[measure(s) for s in e.snapshots]}),flush=True);return 0
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
           'finite':all(r['finite'] for r in rows),
           'gram_identity':all(r['symmetry_error']<=1e-12 and r['min_eigenvalue']>=-1e-10*r['max_eigenvalue'] for r in rows)}
    report={'legs':legs,'gates':gates,'scope':'frozen synthetic contact operator conditioning; no solver change'}
    (ROOT/'reports/innovation_stack_cone_conditioning.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
