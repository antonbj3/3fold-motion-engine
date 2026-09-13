"""Normal-only stack mechanism; synthetic evidence, never a GPU certification.

Unit masses, gravity increment one, no friction, rotations or warm start.
The contact Jacobian is ground e0, then adjacent-body differences. The exact
impulses are K, K-1, ..., 1. Three cyclic independent pair colours are an
explicit synthetic schedule, not a reconstruction of the engine colouring.
Gates and interpretation are registered in docs/RUNNING.md before execution.
"""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
import hashlib
import json
from pathlib import Path
import numpy as np


def measure(k, iterations):
    j = np.eye(k)
    j[np.arange(1, k), np.arange(k - 1)] = -1
    a = j @ j.T
    b = j @ np.ones(k)
    exact = np.arange(k, 0, -1, dtype=np.float64)
    impulse = np.zeros(k)
    trajectory = []
    order = [i for color in range(3) for i in range(color, k, 3)]
    for _ in range(iterations):
        for i in order:
            impulse[i] = max(0., impulse[i] + (b[i] - a[i] @ impulse) / a[i, i])
        trajectory.append(impulse.copy())
    forces = j.T @ impulse
    return {"k": k, "iterations": iterations,
            "order": order, "first_sweep_nonzero": int(np.count_nonzero(trajectory[0])),
            "worst_body_force_error": float(np.max(np.abs(forces - 1))),
            "impulse_relative_error": float(np.max(np.abs(impulse - exact)) / k),
            "exact_force_error": float(np.max(np.abs(j.T @ exact - 1))),
            "dense_reference_error": float(np.max(np.abs(np.linalg.solve(a, b) - exact))),
            "independent_colors": all(abs(i - q) != 1 for c in range(3)
                                      for i in range(c, k, 3) for q in range(c, k, 3)),
            "sha256": hashlib.sha256(np.asarray(trajectory).tobytes()).hexdigest()}


def main():
    rows = [measure(k, n) for k in (4, 8, 16) for n in (1, 40, 160)]
    gates = {"exact_force": all(r["exact_force_error"] == 0 for r in rows),
             "dense_reference": all(r["dense_reference_error"] < 1e-12 for r in rows),
             "independent_colors": all(r["independent_colors"] for r in rows),
             "finite": all(np.isfinite(r["worst_body_force_error"]) for r in rows)}
    out = {"status": "SYNTHETIC-ONLY", "rows": rows, "gates": gates,
           "scope": "normal-only fixed linear system; no engine or timing claim"}
    path = Path(__file__).resolve().parents[1] / "reports" / "innovation_stack_linear_probe.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, sort_keys=True))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
