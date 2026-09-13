"""Two-run full-engine coarse-correction probe; gates in docs/RUNNING.md."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"))
from innovation_stack_probe import idle, trajectory
from colored_vs_jacobi import lattice, timed_steps
from engine_metrics import DIMS, m5_penetration, m1_stack_load
from motion_engine.contact_engine_gpu_coarse import CoarsePairContactEngine


def main():
    background = idle()
    import warp as wp
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("CUDA required")
    last = [None]
    def mk(**kw):
        e = CoarsePairContactEngine(dims=DIMS, mu=.5, **kw)
        last[0] = e
        return e
    report = {"initial_background": background, "legs": []}
    for leg in range(2):
        m5 = m5_penetration(lambda: mk(vit=40, pit=10), "gpu", 16, 600)
        counters = {"applied": last[0].coarse_applied, "singular": last[0].coarse_singular}
        digest = trajectory(mk)
        m1 = m1_stack_load(lambda: mk(vit=40, pit=10), "gpu", 4, 400)
        box = mk(vit=40, pit=10)
        box.add_body([0,0,.101])
        for _ in range(200):
            box.step(1/240, substeps=1)
        expected = box._M[0]*9.81
        force_error = abs(float(box.contact_forces()[0,2])-expected)/expected
        timing = timed_steps(mk, lattice(10000), warm=10, timed=30, vit=40, pit=20)
        row = {"leg": leg+1, "m5": m5, "coarse_counters_K16": counters, "trajectory_sha256": digest,
               "m1": m1, "box_force_relative_error": force_error, "timing": timing,
               "large_scene_skips": last[0].coarse_large_skipped}
        report["legs"].append(row)
        print(json.dumps(row), flush=True)
    report["bit_identical"] = report["legs"][0]["trajectory_sha256"] == report["legs"][1]["trajectory_sha256"]
    report["pass"] = report["bit_identical"] and all(r["m5"]["pass"] and r["m1"]["pass"] and
                      r["box_force_relative_error"] < .01 and r["timing"]["ms_per_step"] <=4 for r in report["legs"])
    (ROOT/"reports"/"innovation_stack_coarse.json").write_text(json.dumps(report, indent=2)+"\n")
    print("TARGET", "PASS" if report["pass"] else "FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
