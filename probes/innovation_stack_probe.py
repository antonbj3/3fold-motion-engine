"""Frozen-baseline mechanism probe; gates in docs/RUNNING.md precede execution."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from colored_vs_jacobi import _factories, lattice, stage_profile, timed_steps
from engine_metrics import m5_penetration


def idle():
    # Reject other pure compute contexts; desktop (graphics) processes are allowed.
    # Uses the light field queries only: the full `nvidia-smi -q -x` dump was the one thing common to
    # three GPU firmware halts (Xid 62) on this box, each time it ran next to CUDA context creation.
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                       capture_output=True, text=True, check=True)
    for line in q.stdout.strip().splitlines():
        pid, _, name = line.partition(",")
        if pid.strip() == str(os.getpid()):
            continue
        if not any(x in name for x in ("chrome", "brave", "firefox", "Xorg", "gnome-shell", "gnome-control-center")):
            raise RuntimeError("GPU has another compute process; measurement aborted")
    g = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True).stdout.strip().split(",")
    return {"gpu_utilization": g[0].strip() + " %", "memory_used": g[1].strip() + " MiB"}


def trajectory(mk):
    e = mk(vit=40, pit=10)
    for k in range(16):
        e.add_body([0, 0, 0.101 + k * 0.205])
    h = hashlib.sha256()
    for _ in range(600):
        e.step(1 / 240, substeps=1)
        st = e.get_state()
        for name in ("xc", "Rm", "vc", "om"):
            h.update(np.ascontiguousarray(getattr(st, name)).tobytes())
    return h.hexdigest()


def main():
    initial_utilization = idle()
    if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] not in ("baseline", "smoke")):
        raise ValueError("Expected baseline or smoke")
    mode = sys.argv[1] if len(sys.argv) == 2 else "baseline"
    import warp as wp
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("CUDA is required; CPU fallback cannot certify this measurement")
    mk = _factories()["colored_manifold_chunks"]
    if mode == "smoke":
        snapshots = []
        finite = True
        for _ in range(2):
            e = mk(vit=40, pit=10)
            for k in range(4):
                e.add_body([0, 0, .101+k*.205])
            digest = hashlib.sha256()
            for _ in range(20):
                e.step(1/240, substeps=1)
                state = e.get_state()
                for name in ("xc", "Rm", "vc", "om"):
                    a = np.ascontiguousarray(getattr(state, name))
                    finite = finite and bool(np.isfinite(a).all())
                    digest.update(a.tobytes())
            snapshots.append(digest.hexdigest())
        report = {"steps_per_run": 20, "bodies": 4, "initial_background": initial_utilization,
                  "hashes": snapshots, "finite": finite, "bit_identical": snapshots[0] == snapshots[1]}
        report["pass"] = finite and report["bit_identical"]
        (ROOT / "reports" / "innovation_stack_smoke.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report))
        return 0 if report["pass"] else 1
    out = {"mode": mode, "initial_utilization": initial_utilization, "gates": {"K": 16, "steps": 600, "pit": 10,
           "vit_max": 40, "pen_over_R_lt": 0.5, "drift": "bounded",
           "geometry_valid": True, "each_wall_ms_le": 4.0, "trajectory_byte_identity": True}, "legs": []}
    for leg in range(2):
        idle()
        row = {"leg": leg + 1, "m5": {str(v): m5_penetration(lambda: mk(vit=v, pit=10), "gpu", 16, 600)
                                      for v in [40, 160]},
               "trajectory_sha256": trajectory(mk)}
        row["background_before_timing"] = idle()
        row["timing"] = timed_steps(mk, lattice(10000), warm=10, timed=30, vit=40, pit=20)
        row["background_before_profile"] = idle()
        row["stages"] = stage_profile(mk, lattice(10000), warm=10, timed=30)
        out["legs"].append(row)
        print(json.dumps(row), flush=True)
    out["bit_identical"] = out["legs"][0]["trajectory_sha256"] == out["legs"][1]["trajectory_sha256"]
    out["target_pass"] = out["bit_identical"] and all(r["m5"]["40"]["pass"] and
                              r["timing"]["ms_per_step"] <= 4.0 for r in out["legs"])
    (ROOT / "reports" / f"innovation_stack_{mode}.json").write_text(json.dumps(out, indent=2) + "\n")
    print("TARGET", "PASS" if out["target_pass"] else "FAIL")
    return 0 if out["target_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
