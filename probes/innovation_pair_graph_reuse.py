"""Measure exact pair-graph persistence without changing the colour stage."""

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
from motion_engine.contact_engine_gpu_deferred_force import DeferredForceContactEngine


class Observer(DeferredForceContactEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graphs = []

    def step(self, dt, substeps=1):
        if substeps != 1:
            raise ValueError("Observer records one substep per call")
        count = len(self.graphs)
        super().step(dt, substeps=1)
        if len(self.graphs) == count:
            self.graphs.append("no_contacts")

    def _color_pairs(self, pairs):
        digest = hashlib.sha256()
        for a in (self.pbi.numpy()[:pairs], self.pbj.numpy()[:pairs], self.invM.numpy()):
            digest.update(np.ascontiguousarray(a).tobytes())
        self.graphs.append(digest.hexdigest())
        return super()._color_pairs(pairs)


def summary(e):
    state = e.get_state()
    h = hashlib.sha256()
    for name in ("xc", "Rm", "vc", "om"):
        h.update(np.ascontiguousarray(getattr(state, name)).tobytes())
    return dict(graphs=e.graphs, equal_successive=sum(a == b for a, b in zip(e.graphs, e.graphs[1:])),
                unique=len(set(e.graphs)), state_sha256=h.hexdigest())


def main():
    if sys.argv[1:] == ["--worker"]:
        from engine_metrics import DIMS, m5_penetration
        from colored_vs_jacobi import lattice
        stack = Observer(dims=DIMS, vit=40, pit=10)
        m5 = m5_penetration(lambda: stack, "gpu", 16, 600)
        large = Observer(dims=DIMS, vit=40, pit=20)
        for center in lattice(10000):
            large.add_body(center)
        for _ in range(40):
            large.step(1 / 240, substeps=1)
        print("RESULT " + json.dumps(dict(stack=summary(stack), large=summary(large), m5=m5)), flush=True)
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
        print(json.dumps({k: {q: v for q, v in legs[-1][k].items() if q != "graphs"}
                          for k in ("stack", "large")}), flush=True)
    frozen = json.loads((ROOT / "reports/innovation_stack_applied_warm_l4.json").read_text())["legs"][0]["40"]
    gates = dict(exact_repeats=legs[0] == legs[1],
                 unchanged_m5=all(r["m5"] == frozen["m5"] for r in legs),
                 unchanged_large=all(r["large"]["state_sha256"] == frozen["large_state_sha256"] for r in legs),
                 all_steps=all(len(r["stack"]["graphs"]) == 600 and len(r["large"]["graphs"]) == 40 for r in legs))
    (ROOT / "reports/innovation_pair_graph_reuse.json").write_text(json.dumps(dict(legs=legs, gates=gates), indent=2) + "\n")
    print(json.dumps(gates), flush=True)
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
