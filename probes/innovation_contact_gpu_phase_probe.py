"""Measure unchanged canonical phases beside the frozen complete-stage probe."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"scripts"))


def worker():
    import warp as wp
    from colored_vs_jacobi import _factories,lattice
    from motion_engine.contact_engine_gpu import _R,_MAXC
    from motion_engine.contact_generation_gpu_bitset import build_kernels
    from innovation_contact_reference import keys
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("CUDA required")
    count,emit=build_kernels(wp)

    @wp.kernel
    def summary(counts:wp.array(dtype=int),offsets:wp.array(dtype=int),
                maximum:wp.array(dtype=int),out:wp.array(dtype=int),last:int):
        out[0]=maximum[0]
        out[1]=offsets[last]+counts[last]

    results=[]
    for bodies in (1000,10000):
        e=_factories()["colored_manifold_chunks"](vit=40,pit=20)
        for c in lattice(bodies):
            e.add_body(c)
        for _ in range(40):
            e.step(1/240,substeps=1)
        p=e.allp.numpy()
        o=e.owner.numpy()
        n=len(p)
        counts=wp.zeros(n,dtype=int,device=e.dev)
        offsets=wp.empty(n,dtype=int,device=e.dev)
        maximum=wp.zeros(1,dtype=int,device=e.dev)
        result=wp.empty(2,dtype=int,device=e.dev)
        fields=[wp.empty(_MAXC,dtype=t,device=e.dev) for t in
                (int,int,wp.vec3,wp.vec3,wp.vec3,float,int,int,wp.int64,wp.int64)]
        ref_fields=[e.cbi,e.cbj,e.cpA,e.cpB,e.cn,e.cpen,e.cfa,e.cfb]
        latest={}
        def baseline():
            e.grid.build(e.allp,2*_R)
            e.cnt.zero_()
            wp.launch(e.KC["gen_fid"],n,inputs=[e.allp,e.owner,e.grid.id,e.cnt,*ref_fields],device=e.dev)
            c=int(e.cnt.numpy()[0])
            if c>_MAXC:
                raise RuntimeError("Baseline capacity exceeded")
            latest["baseline_count"]=c
            wp.synchronize()
        def candidate():
            e.grid.build(e.allp,2*_R)
            maximum.zero_()
            wp.launch(count,n,inputs=[e.allp,e.owner,e.grid.id,counts,maximum],device=e.dev)
            wp.utils.array_scan(counts,offsets,inclusive=False)
            wp.launch(summary,1,inputs=[counts,offsets,maximum,result,n-1],device=e.dev)
            mx,c=map(int,result.numpy())
            if mx>32 or c>_MAXC:
                raise RuntimeError(f"Candidate capacity exceeded: {mx}, {c}")
            wp.launch(emit,n,inputs=[e.allp,e.owner,e.grid.id,offsets,wp.int64(n+1),
                      wp.int64(bodies+1),*fields],device=e.dev)
            latest["candidate_count"],latest["max_neighbours"]=c,mx
            wp.synchronize()
        timings={}
        for name,fn in (("baseline_raw",baseline),("candidate_canonical",candidate)):
            for _ in range(10):
                fn()
            start=time.perf_counter()
            for _ in range(30):
                fn()
            timings[name]=(time.perf_counter()-start)*1000/30
        c=latest["baseline_count"]
        ref=[a.numpy()[:c] for a in ref_fields]
        endpoints=np.stack(ref[-2:],axis=1).astype(np.int64)
        fk,pk=keys(endpoints,o.astype(np.int64),n)
        perm=np.argsort(fk,kind="stable")
        ref=[a[perm] for a in ref+[fk,pk]]
        got=[a.numpy()[:latest["candidate_count"]] for a in fields]
        equal=all(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes() for a,b in zip(ref,got))
        phases={}
        actions=(
            ("grid_build",lambda: e.grid.build(e.allp,2*_R)),
            ("maximum_reset",lambda: maximum.zero_()),
            ("neighbour_count",lambda: wp.launch(count,n,inputs=[e.allp,e.owner,e.grid.id,counts,maximum],device=e.dev)),
            ("exclusive_scan",lambda: wp.utils.array_scan(counts,offsets,inclusive=False)),
            ("summary",lambda: wp.launch(summary,1,inputs=[counts,offsets,maximum,result,n-1],device=e.dev)),
            ("host_readback",lambda: result.numpy()),
            ("canonical_emit",lambda: wp.launch(emit,n,inputs=[e.allp,e.owner,e.grid.id,offsets,wp.int64(n+1),
                      wp.int64(bodies+1),*fields],device=e.dev)))
        for label,action in actions:
            for _ in range(10):
                action()
            wp.synchronize()
            begin=wp.Event(device=e.dev,enable_timing=True)
            end=wp.Event(device=e.dev,enable_timing=True)
            start=time.perf_counter()
            wp.record_event(begin)
            for _ in range(30):
                action()
            wp.record_event(end)
            wp.synchronize()
            wall=(time.perf_counter()-start)*1000/30
            phases[label]={"wall_ms":wall,"event_ms":wp.get_event_elapsed_time(begin,end)/30}
        after=[a.numpy()[:latest["candidate_count"]] for a in fields]
        phase_identity=all(a.tobytes()==b.tobytes() for a,b in zip(got,after))
        results.append({"bodies":bodies,"points":n,**latest,"timings_ms":timings,
                        "phase_timings":phases,"instrumentation_preserves_output":phase_identity,
                        "exact_reference":equal,"candidate_sha256":hashlib.sha256(b"".join(a.tobytes() for a in got)).hexdigest(),
                        "fixture_sha256":hashlib.sha256(p.tobytes()+o.tobytes()).hexdigest(),
                        "canonical_features":bool(np.all(np.diff(got[-2])>0))})
    print("RESULT "+json.dumps(results),flush=True)


def main():
    if sys.argv[1:]==["--worker"]:
        worker()
        return 0
    if sys.argv[1:]:
        raise ValueError("Expected no arguments or --worker")
    from innovation_stack_probe import idle
    legs=[]
    backgrounds=[]
    for _ in range(2):
        before=idle()
        child=subprocess.run([sys.executable,str(Path(__file__).resolve()),"--worker"],cwd=ROOT,capture_output=True,text=True)
        if child.returncode:
            print(child.stdout)
            print(child.stderr,file=sys.stderr)
            raise RuntimeError(f"Worker failed: {child.returncode}")
        after=idle()
        lines=[line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT ")]
        if len(lines)!=1:
            raise RuntimeError("Expected exactly one worker result")
        rows=json.loads(lines[0]);legs.append(rows)
        backgrounds.append({"before_worker":before,"after_worker":after})
        print(json.dumps(rows),flush=True)
    gates={"exact_reference":all(r["exact_reference"] for leg in legs for r in leg),
           "instrumentation_preserves_output":all(r["instrumentation_preserves_output"] for leg in legs for r in leg),
           "canonical_features":all(r["canonical_features"] for leg in legs for r in leg),
           "two_run_identity":all(all(a[k]==b[k] for k in ("fixture_sha256","candidate_sha256","candidate_count","max_neighbours")) for a,b in zip(*legs)),
           "no_slower_than_raw":all(r["timings_ms"]["candidate_canonical"]<=r["timings_ms"]["baseline_raw"] for leg in legs for r in leg)}
    report={"legs":legs,"backgrounds":backgrounds,"gates":gates,"warm":10,"timed":30,
            "scope":"individually warmed phases plus frozen full stages; not additive asynchronous-pipeline costs"}
    (ROOT/"reports"/"innovation_contact_gpu_phase.json").write_text(json.dumps(report,indent=2)+"\n")
    return 0 if all(gates.values()) else 1


if __name__=="__main__":
    raise SystemExit(main())
