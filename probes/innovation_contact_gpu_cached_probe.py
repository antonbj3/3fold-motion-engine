"""Full-array gates for the separate GPU bitset generator, with frozen reference."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"))
from innovation_contact_reference import fixture, keys
from innovation_stack_probe import idle
from motion_engine.contact_engine_gpu import _R
from motion_engine.contact_engine_gpu_colored import _build_colored_kernels
from motion_engine.contact_generation_gpu_cached import generate


def frozen(wp, p, o):
    n = len(p)
    allp = wp.array(p, dtype=wp.vec3, device="cuda:0")
    owner = wp.array(o, dtype=int, device="cuda:0")
    grid = wp.HashGrid(64, 64, 64, device="cuda:0")
    grid.build(allp, 2*_R)
    count = wp.zeros(1, dtype=int, device="cuda:0")
    arrays = [wp.empty(n*(n-1)//2+n, dtype=t, device="cuda:0") for t in
              (int,int,wp.vec3,wp.vec3,wp.vec3,float,int,int)]
    wp.launch(_build_colored_kernels(wp)["gen_fid"], n,
              inputs=[allp,owner,grid.id,count,*arrays], device="cuda:0")
    c = int(count.numpy()[0])
    a = [v.numpy()[:c] for v in arrays]
    endpoints = np.stack(a[-2:],axis=1).astype(np.int64)
    fk,pk = keys(endpoints,o.astype(np.int64),n)
    perm = np.argsort(fk,kind="stable")
    return [v[perm] for v in a+[fk,pk]]


def same(a,b):
    return all(x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes()
               for x,y in zip(a,b))


def main():
    idle()
    import warp as wp
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("CUDA required")
    scenes = [(name,*fixture(k,lat)) for name,k,lat in
              (("stack4",4,False),("stack16",16,False),("lattice50",50,True))]
    scenes.append(("cutoff_coincidence",np.array([
        [0,0,.05],[0,0,.05],[.1,0,.05],
        [np.nextafter(np.float32(.1),np.float32(0)),0,.05],
        [np.nextafter(np.float32(.1),np.float32(1)),0,.05],
        [-.1,0,.049],[-.2,0,.051],[1e-10,0,.05]],dtype=np.float32),np.arange(8)))
    dense=np.zeros((33,3),np.float32)
    dense[:,0]=np.arange(33)*1e-4
    dense[:,2]=.05
    scenes.append(("exact_capacity32",dense,np.arange(33,dtype=np.int32)))
    rows=[]
    for name,p,o in scenes:
        p,o=np.asarray(p,np.float32),np.asarray(o,np.int32)
        ref=frozen(wp,p,o)
        a,ma=generate(p,o)
        b,mb=generate(p,o)
        row={"scene":name,"contacts":len(a[0]),"max_neighbours":ma,
             "exact_frozen_reference":same(a,ref),"bit_identical":same(a,b) and ma==mb,
             "reference_field_identity":[same([x],[y]) for x,y in zip(a,ref)],
             "canonical_features":bool(np.all(np.diff(a[-2])>0)),
             "finite":all(bool(np.isfinite(v).all()) for v in a),
             "sha256":hashlib.sha256(b"".join(v.tobytes() for v in a)).hexdigest()}
        rows.append(row)
        print(json.dumps(row),flush=True)
    p=np.zeros((34,3),np.float32)
    p[:,0]=np.arange(34)*1e-4
    p[:,2]=.05
    overflow=[]
    for _ in range(2):
        try:
            generate(p,np.arange(34,dtype=np.int32))
        except ValueError as exc:
            overflow.append(str(exc))
        else:
            overflow.append("NOT REJECTED")
    gates={"exact_frozen_reference":all(r["exact_frozen_reference"] for r in rows),
           "bit_identical":all(r["bit_identical"] for r in rows),
           "canonical_features":all(r["canonical_features"] for r in rows),
           "finite":all(r["finite"] for r in rows),
           "overflow_rejected":overflow==["Neighbour capacity exceeded: 33 > 32"]*2}
    result={"rows":rows,"overflow":overflow,"gates":gates,"performance_measured":False}
    (ROOT/"reports"/"innovation_contact_gpu_cached.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=="__main__":
    raise SystemExit(main())
