"""Profile the frozen applied-warm engine before changing execution."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def main():
    if sys.argv[1:] == ["--worker"]:
        from motion_engine.contact_engine_gpu_applied_warm import AppliedWarmContactEngine
        from colored_vs_jacobi import stage_profile, lattice
        from engine_metrics import DIMS
        last = []

        def factory(**kw):
            e = AppliedWarmContactEngine(dims=DIMS, vit=40, pit=20, **kw)
            last[:] = [e]
            return e

        stages = stage_profile(factory, lattice(10000), warm=10, timed=30)
        state = last[0].get_state()
        digest = hashlib.sha256()
        finite = True
        for name in ("xc", "Rm", "vc", "om"):
            a = np.ascontiguousarray(getattr(state, name))
            digest.update(a.tobytes())
            finite = finite and bool(np.isfinite(a).all())
        print("RESULT " + json.dumps(dict(stages=stages, sha256=digest.hexdigest(), finite=finite)), flush=True)
        return 0
    if sys.argv[1:]:
        raise ValueError("Expected no arguments or --worker")
    from innovation_stack_probe import idle
    legs = []
    for _ in range(2):
        idle()
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
                               cwd=ROOT, capture_output=True, text=True)
        if child.returncode:
            print(child.stdout)
            print(child.stderr, file=sys.stderr)
            raise RuntimeError("Worker failed")
        idle()
        lines = [line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT ")]
        if len(lines) != 1:
            raise RuntimeError("Expected one result")
        legs.append(json.loads(lines[0]))
        print(lines[0], flush=True)
    gates = dict(exact_state=legs[0]["sha256"] == legs[1]["sha256"],
                 finite=all(r["finite"] and all(np.isfinite(v) and v >= 0 for v in r["stages"].values()) for r in legs))
    (ROOT / "reports/innovation_applied_warm_profile.json").write_text(json.dumps(dict(legs=legs, gates=gates), indent=2) + "\n")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
