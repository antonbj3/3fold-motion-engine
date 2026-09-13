"""Observe frozen GPU generation order and geometry before a new generator."""

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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from innovation_contact_reference import fixture, exhaustive
from innovation_stack_probe import idle
from motion_engine.contact_engine_gpu import _R
from motion_engine.contact_engine_gpu_colored import _build_colored_kernels


def digest(arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def main():
    idle()
    import warp as wp
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("CUDA required")
    kernels = _build_colored_kernels(wp)
    rows = []
    for name, k, lattice in (("stack4", 4, False), ("stack16", 16, False),
                             ("lattice50", 50, True)):
        points, owner = fixture(k, lattice)
        points = points.astype(np.float32)
        n = len(points)
        capacity = n*(n-1)//2+n
        allp = wp.array(points, dtype=wp.vec3, device="cuda:0")
        owners = wp.array(owner.astype(np.int32), dtype=int, device="cuda:0")
        grid = wp.HashGrid(64, 64, 64, device="cuda:0")
        count = wp.zeros(1, dtype=int, device="cuda:0")
        fields = [wp.empty(capacity, dtype=t, device="cuda:0") for t in
                  (int, int, wp.vec3, wp.vec3, wp.vec3, float, int, int)]
        legs = []
        canonical_arrays = []
        for _ in range(2):
            grid.build(allp, 2*_R)
            count.zero_()
            wp.launch(kernels["gen_fid"], n,
                      inputs=[allp, owners, grid.id, count, *fields], device="cuda:0")
            c = int(count.numpy()[0])
            if c > capacity:
                raise RuntimeError("Contact capacity exceeded")
            arrays = [a.numpy()[:c] for a in fields]
            feature = arrays[6].astype(np.int64)*(n+1)+arrays[7]+1
            perm = np.argsort(feature, kind="stable")
            canonical = [a[perm] for a in arrays]
            canonical_arrays.append(canonical)
            legs.append({"contacts": c, "raw_sha256": digest(arrays),
                         "canonical_sha256": digest(canonical),
                         "unique_features": len(np.unique(feature)) == c,
                         "raw_feature_order": bool(np.all(np.diff(feature) > 0)),
                         "finite_geometry": all(bool(np.isfinite(a).all()) for a in arrays[2:6])})
        cpu = exhaustive(points.astype(float), owner)
        gpu = np.stack(canonical_arrays[0][-2:], axis=1).astype(np.int64)
        cpu_set, gpu_set = set(map(tuple, cpu)), set(map(tuple, gpu))
        row = {"scene": name, "points": n, "all_point_pairs": n*(n-1)//2,
               "legs": legs, "canonical_bit_identical": all(np.array_equal(a,b) for a,b in zip(*canonical_arrays)),
               "raw_bit_identical": legs[0]["raw_sha256"] == legs[1]["raw_sha256"],
               "cpu_only_endpoints": len(cpu_set-gpu_set), "gpu_only_endpoints": len(gpu_set-cpu_set)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    gates = {"canonical_bit_identical": all(r["canonical_bit_identical"] for r in rows),
             "unique_features": all(q["unique_features"] for r in rows for q in r["legs"]),
             "finite_geometry": all(q["finite_geometry"] for r in rows for q in r["legs"])}
    result = {"rows": rows, "gates": gates, "performance_measured": False}
    (ROOT/"reports"/"innovation_contact_gpu_reference.json").write_text(json.dumps(result, indent=2)+"\n")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
