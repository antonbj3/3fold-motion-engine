"""Measure sampled bracket values and gradients before contact integration."""

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
from scipy.ndimage import distance_transform_edt
ROOT=Path(__file__).resolve().parents[1]


def query(grid,origin,pitch,points):
    u=(np.asarray(points,np.float64)-origin)/pitch
    index=np.floor(u).astype(np.int64);f=u-index
    if np.any(index<0) or np.any(index+1>=grid.shape):raise ValueError('Outside interpolation domain')
    value=np.zeros(len(u));gradient=np.zeros((len(u),3))
    for x in (0,1):
        for y in (0,1):
            for z in (0,1):
                bits=np.array([x,y,z]);weights=np.where(bits,f,1-f)
                sample=grid[tuple((index+bits).T)].astype(np.float64)
                value+=sample*np.prod(weights,axis=1)
                for axis in range(3):
                    other=[k for k in range(3) if k!=axis]
                    gradient[:,axis]+=sample*(1 if bits[axis] else -1)*np.prod(weights[:,other],axis=1)/pitch
    return value,gradient


def edt(levelset,pitch):
    solid=levelset<0
    a=distance_transform_edt(~solid,sampling=pitch).astype(np.float32)
    b=distance_transform_edt(solid,sampling=pitch).astype(np.float32)
    distance=a-b
    return (distance-np.sign(distance)*np.float32(.5*pitch)).astype(np.float32)


def probes():
    out={}
    offsets=np.array([-.2,-.05,0,.05,.2,.6])
    out['support']=np.column_stack((np.full(6,10.125),np.full(6,9.125),2+offsets))
    out['hole']=np.column_stack((np.full(6,6.5),np.full(6,4.),2+offsets))
    out['wall']=np.column_stack((4+offsets,np.full(6,2.125),np.full(6,10.125)))
    out['corner']=np.column_stack((4+offsets,np.full(6,6.125),np.full(6,16.125)))
    return out


def worker(output):
    saved={};rows=[];coverage=True
    metadata=json.loads((ROOT/'reports/fixtures/bracket_contact_fields.json').read_text())
    with np.load(ROOT/'reports/fixtures/bracket_contact_fields.npz') as d:
        origin=d['origin']
        for pitch in (.5,.25):
            levelset=d[f'levelset_{pitch}'];key=f'bracket_plate_rectangular_holes_{pitch}_0_field'
            coverage &= hashlib.sha256(levelset.tobytes()).hexdigest()==metadata['source_array_sha256'][key]
            for mode,grid in [('levelset',levelset),('edt',edt(levelset,pitch))]:
                saved[f'{pitch}_{mode}_grid']=grid
                for name,points in probes().items():
                    values,grad=query(grid,origin,pitch,points);norm=np.linalg.norm(grad,axis=1)
                    prefix=f'{pitch}_{mode}_{name}'
                    for label,a in [('points',points),('values',values),('gradient',grad)]:saved[prefix+'_'+label]=a
                    row=dict(pitch=pitch,mode=mode,probe=name,values=values.tolist(),normal_lengths=norm.tolist(),zero_gradients=int(np.count_nonzero(norm<1e-12)))
                    if name in ('support','wall'):
                        axis=2 if name=='support' else 0;face=2 if axis==2 else 4
                        normal=np.eye(3)[axis]
                        row.update(max_face_distance_error=float(np.max(abs(values-(points[:,axis]-face)))),max_gradient_error=float(np.max(abs(grad-normal))))
                    rows.append(row)
        shape=(8,9,10);coords=np.indices(shape).transpose(1,2,3,0).astype(float)
        affine=coords@np.array([2.,-3.,.5])+7
        p=np.array([[1.123,2.234,3.345],[5.789,6.125,7.321]])
        value,grad=query(affine,np.zeros(3),1.,p)
        affine_error=max(float(np.max(abs(value-(p@np.array([2.,-3.,.5])+7)))),float(np.max(abs(grad-[2,-3,.5]))))
    holes=all(np.all(saved[f'{pitch}_{mode}_hole_values']>0) and saved[f'{pitch}_{mode}_support_values'][0]<0 for pitch in (.5,.25) for mode in ('levelset','edt'))
    np.savez_compressed(output,**saved)
    print('RESULT '+json.dumps(dict(rows=rows,coverage=bool(coverage),affine_error=affine_error,holes=bool(holes))))


def main():
    if len(sys.argv)==3 and sys.argv[1]=='--worker':worker(sys.argv[2]);return 0
    results=[];arrays=[]
    with tempfile.TemporaryDirectory() as tmp:
        for leg in range(2):
            path=Path(tmp)/f'{leg}.npz'
            p=subprocess.run([sys.executable,__file__,'--worker',str(path)],cwd=ROOT,capture_output=True,text=True)
            if p.returncode:raise RuntimeError(p.stdout+p.stderr)
            results.append(json.loads(next(line[7:] for line in p.stdout.splitlines() if line.startswith('RESULT '))))
            with np.load(path) as d:arrays.append({k:d[k].copy() for k in d.files})
    gates=dict(coverage=all(r['coverage'] and len(r['rows'])==16 for r in results),repeat=results[0]==results[1] and all(arrays[0][k].tobytes()==arrays[1][k].tobytes() for k in arrays[0]),affine=all(r['affine_error']<=1e-12 for r in results),holes=all(r['holes'] for r in results))
    report=dict(results=results,gates=gates,status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL')
    (ROOT/'reports/innovation_sampled_field_mechanism.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(ROOT/'reports/innovation_sampled_field_mechanism.npz',**{str(i)+'__'+k:v for i,a in enumerate(arrays) for k,v in a.items()})
    print(json.dumps(report));return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
