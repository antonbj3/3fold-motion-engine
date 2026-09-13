"""Observe frozen solvers under labelled friction/position-pass ablations."""

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
from innovation_stack_probe import idle
from engine_metrics import DIMS, m5_penetration
from motion_engine.contact_engine_gpu_colored import GraphColoredContactEngine
from motion_engine.contact_engine_gpu_coarse import CoarsePairContactEngine


class Observed:
    def __init__(self, engine):
        self.engine = engine
        self.digest = hashlib.sha256()
        self.peak_angular_speed = 0.
        self.peak_angle = 0.
        self.finite = True

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def get_state(self):
        state = self.engine.get_state()
        for name in ("xc", "Rm", "vc", "om"):
            a = np.ascontiguousarray(getattr(state, name))
            self.digest.update(a.tobytes())
            self.finite &= bool(np.isfinite(a).all())
        self.peak_angular_speed = max(self.peak_angular_speed, float(np.linalg.norm(state.om, axis=1).max()))
        cos_angle = (np.trace(state.Rm, axis1=1, axis2=2)-1)/2
        self.peak_angle = max(self.peak_angle, float(np.arccos(np.clip(cos_angle,-1,1)).max()))
        return state


def main():
    initial = idle()
    import warp as wp
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("CUDA required")
    rows = []
    for name in ("baseline", "coarse"):
        for mu in (.5, 0.):
            for pit in (10, 40):
                legs = []
                for _ in range(2):
                    kw = dict(dims=DIMS, mu=mu, vit=40, pit=pit)
                    e = (GraphColoredContactEngine(manifold_reduce=True, graph_capture=True, pair_chunks=True, **kw)
                         if name == "baseline" else CoarsePairContactEngine(**kw))
                    observed = Observed(e)
                    m5 = m5_penetration(lambda: observed, "gpu", 16, 600)
                    legs.append({"m5": m5, "peak_angular_speed": observed.peak_angular_speed,
                                 "peak_angle_radians": observed.peak_angle, "finite": observed.finite,
                                 "sha256": observed.digest.hexdigest()})
                row = {"engine": name, "friction": mu, "position_iterations": pit,
                       "primary_case": mu == .5 and pit == 10, "legs": legs,
                       "bit_identical": legs[0] == legs[1]}
                rows.append(row)
                print(json.dumps(row), flush=True)
    out = {"initial_background": initial, "rows": rows,
           "instrument_pass": all(r["bit_identical"] and all(q["finite"] for q in r["legs"]) for r in rows)}
    (ROOT/"reports"/"innovation_stack_ablation.json").write_text(json.dumps(out, indent=2)+"\n")
    return 0 if out["instrument_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
