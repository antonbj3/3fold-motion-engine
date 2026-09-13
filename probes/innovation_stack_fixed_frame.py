"""Full-engine fixed-frame gates; thresholds registered before execution."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def worker():
    from motion_engine.contact_engine_gpu_fixed_frame import FixedFrameContactEngine
    from engine_metrics import DIMS, m5_penetration, m1_stack_load
    from innovation_stack_probe import trajectory
    from colored_vs_jacobi import lattice, timed_steps

    def factory(**kw):
        return FixedFrameContactEngine(dims=DIMS, mu=0.5, **kw)

    m5 = m5_penetration(lambda: factory(vit=40, pit=10), "gpu", 16, 600)
    digest = trajectory(factory)
    m1 = m1_stack_load(lambda: factory(vit=40, pit=10), "gpu", 4, 400)
    box = factory(vit=40, pit=10)
    box.add_body([0, 0, 0.101])
    for _ in range(200):
        box.step(1 / 240, substeps=1)
    expected = box._M[0] * 9.81
    force = abs(float(box.contact_forces()[0, 2]) - expected) / expected
    timing = timed_steps(factory, lattice(10000), warm=10, timed=30, vit=40, pit=20)
    print("RESULT " + json.dumps(dict(m5=m5, m1=m1, trajectory_sha256=digest,
                                      force_error=force, timing=timing)), flush=True)


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
    keys = ("m5", "m1", "trajectory_sha256", "force_error")
    gates = dict(exact_repeats=all(legs[0][k] == legs[1][k] for k in keys),
                 k16_m5=all(r["m5"]["pass"] for r in legs),
                 k4_m1=all(r["m1"]["pass"] for r in legs),
                 resting_force=all(r["force_error"] < 0.01 for r in legs),
                 original_speed=all(r["timing"]["ms_per_step"] <= 4.0 for r in legs))
    report = dict(legs=legs, backgrounds=backgrounds, gates=gates,
                  day_plan_speed=all(r["timing"]["ms_per_step"] <= 4.9 for r in legs))
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/innovation_stack_fixed_frame.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(gates), flush=True)
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
