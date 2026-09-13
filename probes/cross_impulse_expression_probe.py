"""Replay retained friction operands with explicit arithmetic boundary controls."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from cross_hardware_state_probe import ROOT


def fixtures():
    schema=json.loads((ROOT/'reports/cross_scalar_schema.json').read_text());rows=[];expect=[]
    with np.load(ROOT/'reports/cross_scalar_l4.npz') as a,np.load(ROOT/'reports/cross_scalar_local.npz') as b:
        for engine,key,contact,axis in [('pair_chunks','pair_trace',1,2),('color_cache','fixed_trace',2,1)]:
            labels=schema[key];stop=next(i for i,x in enumerate(labels) if x['target']==f'a{axis}')
            def read(data,name,component=0):
                index=max(i for i,x in enumerate(labels[:stop]) if x['target']==name and x['component']==component and data[f'{engine}__0__scalar_seen'][0,contact,i]!=0)
                return data[f'{engine}__0__scalar_trace'][0,contact,index]
            row=[*[read(a,'vrel',i) for i in range(3)],*[read(a,f't{axis}',i) for i in range(3)],read(a,f'o{axis}'),read(a,f'meft{axis}')]
            other=[*[read(b,'vrel',i) for i in range(3)],*[read(b,f't{axis}',i) for i in range(3)],read(b,f'o{axis}'),read(b,f'meft{axis}')]
            if np.array(row).tobytes()!=np.array(other).tobytes():raise RuntimeError('Operand mismatch precedes selected expression')
            rows.append(row);expect.append([a[f'{engine}__0__scalar_trace'][0,contact,stop],b[f'{engine}__0__scalar_trace'][0,contact,stop]])
    return np.array(rows,np.float32),np.array(expect,np.float32)


def kernels(wp,fuse):
    @wp.func_native('return __fadd_rn(__fadd_rn(__fmul_rn(a[0], b[0]), __fmul_rn(a[1], b[1])), __fmul_rn(a[2], b[2]));')
    def dot_rn(a:wp.vec3,b:wp.vec3)->float: ...
    @wp.func_native('return __fdiv_rn(a, b);')
    def div_rn(a:float,b:float)->float: ...
    @wp.func_native('return __fsub_rn(a, b);')
    def sub_rn(a:float,b:float)->float: ...
    @wp.kernel(module='unique',module_options={'fuse_fp':fuse,'enable_backward':False})
    def evaluate(data:wp.array2d(dtype=float),out:wp.array2d(dtype=float)):
        i=wp.tid();a=wp.vec3(data[i,0],data[i,1],data[i,2]);b=wp.vec3(data[i,3],data[i,4],data[i,5]);old=data[i,6];mass=data[i,7]
        dot=wp.dot(a,b);quotient=dot/mass
        out[i,0]=dot;out[i,1]=quotient;out[i,2]=old-quotient
        out[i,3]=old-wp.dot(a,b)/mass
        fixed=dot_rn(a,b);out[i,4]=fixed;out[i,5]=div_rn(fixed,mass);out[i,6]=sub_rn(old,div_rn(fixed,mass))
    return evaluate


def worker(output):
    import warp as wp
    wp.init();data,expected=fixtures();gpu=wp.array(data,dtype=float,device='cuda:0');saved=dict(inputs=data,expected=expected)
    for fuse in (True,False):
        out=wp.zeros((2,7),dtype=float,device='cuda:0');kernel=kernels(wp,fuse)
        wp.launch(kernel,2,inputs=[gpu,out],device='cuda:0');saved[str(fuse)]=out.numpy()
    np.savez(output,**saved)


def main(label):
    saved=[]
    with tempfile.TemporaryDirectory() as tmp:
        for leg in range(2):
            path=Path(tmp)/f'{leg}.npz'
            p=subprocess.run([sys.executable,__file__,'--worker',str(path)],cwd=ROOT,capture_output=True,text=True)
            if p.returncode:raise RuntimeError(p.stdout+p.stderr)
            with np.load(path) as d:saved.append({k:d[k].copy() for k in d.files})
    backend=0 if label=='l4' else 1
    gates=dict(repeat=all(saved[0][k].tobytes()==saved[1][k].tobytes() for k in saved[0]),finite=all(np.isfinite(a).all() for a in saved[0].values()),
        source_expression_reproduced=saved[0]['True'][:,3].tobytes()==saved[0]['expected'][:,backend].tobytes(),
        explicit_boundaries_stable=saved[0]['True'][:,4:].tobytes()==saved[0]['False'][:,4:].tobytes())
    report=dict(gates=gates,rows={k:a.tolist() for k,a in saved[0].items()},columns=['dot','quotient','split_subtraction','whole_expression','rounded_dot','rounded_quotient','rounded_subtraction'])
    (ROOT/f'reports/cross_impulse_{label}.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez(ROOT/f'reports/cross_impulse_{label}.npz',**{str(i)+'__'+k:a for i,d in enumerate(saved) for k,a in d.items()})
    print(json.dumps(report));return 0 if all(gates.values()) else 1


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--worker':worker(sys.argv[2]);raise SystemExit(0)
    raise SystemExit(main(sys.argv[1]))
