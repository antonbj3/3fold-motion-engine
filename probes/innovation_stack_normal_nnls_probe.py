"""Two-run normal-NNLS experiment with unchanged physical gates; no timings."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from innovation_stack_probe import idle, trajectory
from engine_metrics import DIMS, m5_penetration, m1_stack_load
from motion_engine.contact_engine_gpu_normal_nnls import NormalNNLSContactEngine


def main():
    idle()
    import warp as wp
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("CUDA required")
    last = [None]
    def mk(**kw):
        e = NormalNNLSContactEngine(dims=DIMS, mu=.5, **kw)
        last[0] = e
        return e
    legs = []
    for _ in range(2):
        m5 = m5_penetration(lambda: mk(vit=40, pit=10), "gpu", 16, 600)
        diagnostics = {"normal_applied": last[0].normal_applied,
                       "normal_max_kkt": last[0].normal_max_kkt}
        digest = trajectory(mk)
        m1 = m1_stack_load(lambda: mk(vit=40, pit=10), "gpu", 4, 400)
        box = mk(vit=40, pit=10)
        box.add_body([0, 0, .101])
        for _ in range(200):
            box.step(1/240, substeps=1)
        expected = box._M[0]*9.81
        force_error = abs(float(box.contact_forces()[0, 2])-expected)/expected
        row = {"m5": m5, "diagnostics": diagnostics, "trajectory_sha256": digest,
               "m1": m1, "box_force_relative_error": force_error}
        legs.append(row)
        print(json.dumps(row), flush=True)
    gates = {"bit_identical": legs[0] == legs[1],
             "m5": all(r["m5"]["pass"] for r in legs),
             "m1": all(r["m1"]["pass"] for r in legs),
             "box_force": all(r["box_force_relative_error"] < .01 for r in legs)}
    report = {"legs": legs, "gates": gates, "performance_measured": False}
    (ROOT/"reports"/"innovation_stack_normal_nnls.json").write_text(json.dumps(report, indent=2)+"\n")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
