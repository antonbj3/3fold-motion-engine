"""Static sampled bracket: original physical gates and explicit field controls."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
from innovation_sampled_field_mechanism import ROOT,edt,query,probes
from innovation_prepared_planar import Trace,digest
from innovation_stack_probe import idle
from motion_engine.contact_engine_gpu_sampled_field import SampledFieldContactEngine,validate_field


def fixture():
    with np.load(ROOT/'reports/fixtures/bracket_contact_fields.npz') as d:
        return (edt(d['levelset_0.25'],.25)*np.float32(.05)).astype(np.float32),d['origin']*.05+[-.5,-.45,-.1],.0125


def worker(reverse,output):
    import warp as wp
    from motion_engine.contact_engine_gpu_prepared_planar import PreparedPlanarContactEngine
    from engine_metrics import DIMS,m1_stack_load,m5_penetration,m6_determinism
    if wp.config.version!='1.13.0' or np.__version__!='2.5.3':raise RuntimeError('Pinned runtime required')
    field,origin,pitch=fixture();rows={};saved={}
    classes=[('baseline',PreparedPlanarContactEngine),('sampled',lambda **kw:SampledFieldContactEngine(field,origin,pitch,**kw))]
    for name,cls in (classes[::-1] if reverse else classes):
        traces=[]
        def make(**kw):return cls(**(dict(dims=DIMS,mu=.5,vit=40,pit=10)|kw))
        def traced(**kw):
            e=Trace(make(**kw));traces.append(e);return e
        m5=m5_penetration(traced,'gpu',16,600);m1=m1_stack_load(traced,'gpu',4,400)
        rest=traced();rest.add_body([0,0,.101])
        for _ in range(200):rest.step(1/240)
        force_error=abs(float(rest.contact_forces()[0,2])-rest._M[0]*9.81)/(rest._M[0]*9.81)
        m6=m6_determinism(make,'gpu',4,4,100)
        engine=traces[0].engine
        for _ in range(10):engine.step(1/240)
        wp.synchronize_device('cuda:0');start=time.perf_counter()
        for _ in range(30):engine.step(1/240)
        wp.synchronize_device('cuda:0');ms=(time.perf_counter()-start)*1000/30
        state=engine.get_state()
        for key in ('xc','Rm','vc','om'):saved[name+'__'+key]=getattr(state,key)
        final,finite=digest(engine)
        rows[name]=dict(m1=m1,m5=m5,m6=m6,force_error=force_error,traces=[t.sha.hexdigest() for t in traces],final=final,finite=finite and all(t.finite for t in traces),ms_per_step=ms)
    e=SampledFieldContactEngine(field,origin,pitch,dims=DIMS);e.add_body([0,0,.101]);e._build()
    points=(np.concatenate(list(probes().values()))*.05+[-.5,-.45,-.1]).astype(np.float32)
    gpu=wp.array(points,dtype=wp.vec3,device='cuda:0');out=wp.empty(len(points),dtype=wp.vec4d,device='cuda:0')
    wp.launch(e._query,len(points),inputs=[e._field,e._origin,pitch,gpu,out],device='cuda:0')
    actual=out.numpy();v,g=query(field,origin,pitch,points);expected=np.column_stack((v,g));error=float(np.max(abs(actual-expected)))
    saved['query_actual']=actual;saved['query_reference']=expected;saved['query_points']=points
    controls=np.array([[0,0,.02],[-.175,-.25,0],[-.299,-.35,.4],[0,0,2]],np.float32)
    gpu=wp.array(controls,dtype=wp.vec3,device='cuda:0');owners=wp.zeros(4,dtype=int,device='cuda:0');e.cnt.zero_();e._invalid.zero_()
    outputs=[e.cbi,e.cbj,e.cpA,e.cpB,e.cn,e.cpen,e.cfa,e.cfb]
    wp.launch(e._ground,4,inputs=[e._field,e._origin,pitch,gpu,owners,e.cnt,e._invalid,*outputs],device='cuda:0')
    count=int(e.cnt.numpy()[0]);ids=e.cfa.numpy()[:count];normals=e.cn.numpy()[:count];order=np.argsort(ids)
    ids=ids[order];normals=normals[order]
    geometry=ids.tolist()==[0,2] and np.max(abs(normals-np.array([[0,0,1],[1,0,0]])))<1e-6 and int(e._invalid.numpy()[0])==0
    saved['control_ids']=ids;saved['control_normals']=normals;saved['control_points']=controls
    invalid=[]
    bad_field=field.copy();bad_field[0,0,0]=np.nan
    no_padding=field.copy();no_padding[0,:,:]=0
    changes=[dict(field=field.astype(np.float64)),dict(field=field[:1]),dict(field=bad_field),dict(origin=[0,0]),dict(origin=[0,np.inf,0]),dict(pitch=0),dict(field=np.ones_like(field)),dict(field=no_padding)]
    for change in changes:
        try:validate_field(**(dict(field=field,origin=origin,pitch=pitch)|change));invalid.append(False)
        except ValueError:invalid.append(True)
    np.savez_compressed(output,**saved)
    print('RESULT '+json.dumps(dict(rows=rows,query_error=error,geometry=bool(geometry),invalid=invalid)),flush=True)


def main():
    if len(sys.argv)==4 and sys.argv[1]=='--worker':worker(sys.argv[2]=='1',sys.argv[3]);return 0
    results=[];arrays=[];background=[]
    with tempfile.TemporaryDirectory() as tmp:
        for leg in range(2):
            out=Path(tmp)/f'{leg}.npz';before=idle()
            p=subprocess.run([sys.executable,__file__,'--worker',str(leg),str(out)],cwd=ROOT,capture_output=True,text=True)
            if p.returncode:print(p.stdout);print(p.stderr);raise RuntimeError('Worker failed')
            results.append(json.loads(next(line[7:] for line in p.stdout.splitlines() if line.startswith('RESULT '))))
            background.append(dict(before=before,after=idle()))
            with np.load(out) as d:arrays.append({k:d[k].copy() for k in d.files})
            print('leg '+str(leg)+' complete',flush=True)
    def fixed(result):
        r=json.loads(json.dumps(result))
        for row in r['rows'].values():row.pop('ms_per_step')
        return r
    rows=[r['rows']['sampled'] for r in results]
    gates=dict(query=all(r['query_error']<=1e-12 for r in results),
        repeat=fixed(results[0])==fixed(results[1]) and all(arrays[0][k].tobytes()==arrays[1][k].tobytes() for k in arrays[0]) and all(r['m6']['eps_nondet_max_dxc_m']==0 for r in rows),
        m1=all(r['m1']['pass'] for r in rows),m5=all(r['m5']['pass'] for r in rows),rest=all(r['force_error']<.01 for r in rows),
        finite_timing=all(r['finite'] and np.isfinite(r['ms_per_step']) and r['ms_per_step']>0 for r in rows),
        refusals=all(len(r['invalid'])==8 and all(r['invalid']) for r in results),geometry=all(r['geometry'] for r in results))
    report=dict(results=results,gates=gates,status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',background=background)
    (ROOT/'reports/innovation_sampled_field_contact.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(ROOT/'reports/innovation_sampled_field_contact.npz',**{str(i)+'__'+k:v for i,a in enumerate(arrays) for k,v in a.items()})
    print(json.dumps(report),flush=True);return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
