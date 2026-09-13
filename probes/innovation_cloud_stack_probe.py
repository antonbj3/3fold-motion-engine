"""Two isolated baseline legs with the frozen idle guard outside CUDA workers."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def leg():
    from innovation_stack_probe import trajectory
    from colored_vs_jacobi import _factories, lattice, timed_steps, stage_profile
    from engine_metrics import m5_penetration
    import warp as wp
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("CUDA required")
    mk = _factories()["colored_manifold_chunks"]
    m5 = {str(v): m5_penetration(lambda: mk(vit=v, pit=10), "gpu", 16, 600)
          for v in (40, 160)}
    trajectory_hash = trajectory(mk)
    timing = timed_steps(mk, lattice(10000), warm=10, timed=30, vit=40, pit=20)
    stages = stage_profile(mk, lattice(10000), warm=10, timed=30)
    print("RESULT " + json.dumps({"m5": m5, "trajectory_sha256": trajectory_hash,
                                "timing": timing, "stages": stages}), flush=True)


def main():
    if sys.argv[1:] == ["--leg"]:
        leg()
        return 0
    if sys.argv[1:]:
        raise ValueError("Expected no arguments or --leg")
    # This supervisor never initializes Warp. The same guard executes before a
    # worker starts and after it exits, so its own context cannot be misidentified.
    from innovation_stack_probe import idle
    backgrounds, legs = [], []
    for _ in range(2):
        before = idle()
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--leg"],
                               cwd=ROOT, capture_output=True, text=True)
        if child.returncode:
            print(child.stdout)
            print(child.stderr, file=sys.stderr)
            raise RuntimeError(f"Worker failed with exit {child.returncode}")
        after = idle()
        results = [line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT ")]
        if len(results) != 1:
            raise RuntimeError("Expected exactly one worker result")
        row = json.loads(results[0])
        legs.append(row)
        backgrounds.append({"before_worker": before, "after_worker": after})
        print(json.dumps(row), flush=True)
    gates = {"trajectory_bit_identical": legs[0]["trajectory_sha256"] == legs[1]["trajectory_sha256"],
             "m5_bit_identical": legs[0]["m5"] == legs[1]["m5"],
             "k16_40": all(r["m5"]["40"]["pass"] for r in legs),
             "each_wall_ms_le_4": all(r["timing"]["ms_per_step"] <= 4.0 for r in legs)}
    report = {"legs": legs, "backgrounds": backgrounds, "gates": gates,
              "guard": "frozen guard in CPU supervisor before and after each isolated worker"}
    (ROOT/"reports"/"innovation_cloud_stack.json").write_text(json.dumps(report, indent=2)+"\n")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
