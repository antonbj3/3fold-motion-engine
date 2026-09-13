"""Measure cached impulses and velocity changes across the frozen warm seam."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from innovation_stack_normal_space import SnapshotEngine, SAMPLES
from engine_metrics import m5_penetration


class WarmObserver(SnapshotEngine):
    def __init__(self):
        super().__init__()
        self.rows = []

    def _device_manifold(self, count):
        sample = self.calls + 1 in SAMPLES
        if sample:
            before = np.concatenate([self.v.numpy(), self.w.numpy()])
        count = super()._device_manifold(count)
        if sample:
            after = np.concatenate([self.v.numpy(), self.w.numpy()])
            jn = self.jn.numpy()[:count]
            jt = np.stack([self.jt1.numpy()[:count], self.jt2.numpy()[:count]])
            self.rows.append(dict(call=self.calls + 1, contacts=count,
                                  cached_normal_max=float(jn.max()),
                                  cached_normal_sum=float(jn.astype(float).sum()),
                                  cached_tangent_norm=float(np.linalg.norm(jt.astype(float))),
                                  velocity_change_max=float(np.max(np.abs(after-before))),
                                  finite=bool(np.isfinite(jn).all() and np.isfinite(jt).all())))
        return count


def main():
    if sys.argv[1:] == ["--worker"]:
        observer = WarmObserver()
        m5 = m5_penetration(lambda: observer, "gpu", 16, 600)
        print("RESULT " + json.dumps(dict(m5=m5, rows=observer.rows)), flush=True)
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
    frozen = json.loads((ROOT / "reports/innovation_cloud_stack_l4.json").read_text())["legs"][0]["m5"]["40"]
    gates = dict(exact_repeats=legs[0] == legs[1], unchanged_m5=all(r["m5"] == frozen for r in legs),
                 samples=all([s["call"] for s in r["rows"]] == list(SAMPLES) for r in legs),
                 finite=all(s["finite"] for r in legs for s in r["rows"]))
    (ROOT / "reports/innovation_stack_warm_mechanism.json").write_text(
        json.dumps(dict(legs=legs, gates=gates), indent=2) + "\n")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
