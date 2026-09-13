"""Compare deferred force against frozen physical, state and force references."""

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


def worker():
    from motion_engine.contact_engine_gpu_applied_warm import AppliedWarmContactEngine
    from motion_engine.contact_engine_gpu_deferred_force import DeferredForceContactEngine
    from engine_metrics import DIMS, m5_penetration, m1_stack_load
    from innovation_stack_ablation import Observed
    from colored_vs_jacobi import lattice, timed_steps
    rows = {}
    for budget in (40,):
        last = []

        def factory(**kw):
            e = DeferredForceContactEngine(dims=DIMS, mu=0.5, **kw)
            last[:] = [e]
            return e

        observer = Observed(factory(vit=budget, pit=10))
        m5 = m5_penetration(lambda: observer, "gpu", 16, 600)
        m1 = m1_stack_load(lambda: factory(vit=budget, pit=10), "gpu", 4, 400)
        box = factory(vit=budget, pit=10)
        box.add_body([0, 0, 0.101])
        for _ in range(200):
            box.step(1 / 240, substeps=1)
        expected = box._M[0] * 9.81
        force = abs(float(box.contact_forces()[0, 2]) - expected) / expected
        timing = timed_steps(factory, lattice(10000), warm=10, timed=30, vit=budget, pit=20)
        large = last[0].get_state()
        digest = hashlib.sha256()
        finite = observer.finite
        for name in ("xc", "Rm", "vc", "om"):
            a = np.ascontiguousarray(getattr(large, name))
            digest.update(a.tobytes())
            finite = finite and bool(np.isfinite(a).all())
        force_digest = hashlib.sha256(np.ascontiguousarray(last[0].contact_forces()).tobytes()).hexdigest()
        baseline_last = []
        def baseline_factory(**kw):
            e = AppliedWarmContactEngine(dims=DIMS, mu=0.5, **kw)
            baseline_last[:] = [e]
            return e
        baseline_timing = timed_steps(baseline_factory, lattice(10000), warm=10, timed=30, vit=budget, pit=20)
        baseline_state = baseline_last[0].get_state()
        baseline_digest = hashlib.sha256()
        for name in ("xc", "Rm", "vc", "om"):
            baseline_digest.update(np.ascontiguousarray(getattr(baseline_state, name)).tobytes())
        baseline_force = hashlib.sha256(np.ascontiguousarray(baseline_last[0].contact_forces()).tobytes()).hexdigest()
        rows[str(budget)] = dict(force_sha256=force_digest, baseline_force_sha256=baseline_force,
                                 baseline_state_sha256=baseline_digest.hexdigest(), baseline_timing=baseline_timing,m5=m5, m1=m1, trajectory_sha256=observer.digest.hexdigest(),
                                 force_error=force, timing=timing, finite=finite,
                                 large_state_sha256=digest.hexdigest())
    print("RESULT " + json.dumps(rows), flush=True)


def main():
    if sys.argv[1:] == ["--worker"]:
        worker()
        return 0
    if sys.argv[1:]:
        raise ValueError("Expected no arguments or --worker")
    from innovation_stack_probe import idle
    legs = []
    backgrounds = []
    for _ in range(2):
        before = idle()
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
                               cwd=ROOT, capture_output=True, text=True)
        if child.returncode:
            print(child.stdout)
            print(child.stderr, file=sys.stderr)
            raise RuntimeError("Worker failed")
        backgrounds.append(dict(before=before, after=idle()))
        lines = [line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT ")]
        if len(lines) != 1:
            raise RuntimeError("Expected exactly one worker result")
        legs.append(json.loads(lines[0]))
        print(lines[0], flush=True)
    frozen = json.loads((ROOT / "reports/innovation_stack_applied_warm_l4.json").read_text())["legs"][0]["40"]
    gates = {}
    for budget in ("40",):
        rows = [leg[budget] for leg in legs]
        keys = ("m5", "m1", "trajectory_sha256", "large_state_sha256", "force_error", "finite", "force_sha256", "baseline_force_sha256", "baseline_state_sha256")
        gates[budget] = dict(unchanged_physics=all(all(r[k] == frozen[k] for k in ("m5", "m1", "trajectory_sha256", "force_error")) for r in rows),
                             exact_force=all(r["force_sha256"] == r["baseline_force_sha256"] for r in rows),
                             exact_reference_state=all(r["large_state_sha256"] == r["baseline_state_sha256"] for r in rows),exact_repeats=all(rows[0][k] == rows[1][k] for k in keys),
                             k16_m5=all(r["m5"]["pass"] for r in rows),
                             k4_m1=all(r["m1"]["pass"] for r in rows),
                             resting_force=all(r["force_error"] < 0.01 for r in rows),
                             finite=all(r["finite"] for r in rows),
                             original_speed=all(r["timing"]["ms_per_step"] <= 4.0 for r in rows))
    report = dict(legs=legs, backgrounds=backgrounds, gates=gates,
                  day_plan_speed={k: all(r[k]["timing"]["ms_per_step"] <= 4.9 for r in legs)
                                  for k in gates})
    (ROOT / "reports/innovation_stack_deferred_force.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(gates), flush=True)
    return 0 if any(all(g.values()) for g in gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
