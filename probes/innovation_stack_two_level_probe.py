"""Synthetic schedule experiment after the baseline mechanism table.

A coarse correction solves only adjacent-pair constant impulse modes. It adds
work, and this cell makes no runtime, contact-engine or GPU claim.
"""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
import hashlib
import json
from pathlib import Path
import numpy as np
from innovation_stack_linear_probe import measure


def run(k, iterations, variant):
    j = np.eye(k)
    j[np.arange(1, k), np.arange(k - 1)] = -1
    a, b = j @ j.T, j @ np.ones(k)
    p = np.zeros((k, (k + 1) // 2))
    p[np.arange(k), np.arange(k) // 2] = 1
    coarse = p.T @ a @ p
    impulse, states = np.zeros(k), []
    cyclic = [i for c in range(3) for i in range(c, k, 3)]
    for n in range(iterations):
        if variant == "two_level":
            correction = np.linalg.solve(coarse, p.T @ (b - a @ impulse))
            impulse = np.maximum(0, impulse + p @ correction)
        order = (list(range(k)) if n % 2 == 0 else list(reversed(range(k)))) if variant == "alternating" else cyclic
        for i in order:
            impulse[i] = max(0., impulse[i] + (b[i] - a[i] @ impulse) / a[i, i])
        states.append(impulse.copy())
    error = float(np.max(np.abs(j.T @ impulse - 1)))
    return {"variant": variant, "k": k, "sweeps": iterations,
            "contact_updates": k * iterations,
            "coarse_solves": iterations if variant == "two_level" else 0,
            "worst_body_force_error": error,
            "nonnegative_finite": bool(np.isfinite(states).all() and np.min(states) >= 0),
            "sha256": hashlib.sha256(np.asarray(states).tobytes()).hexdigest()}


def main():
    rows = [run(k, n, v) for v in ("alternating", "two_level")
            for k in (4, 8, 16) for n in (1, 40, 160)]
    references = [measure(k, 40) for k in (4, 8, 16)]
    gates = {"references": all(r["dense_reference_error"] < 1e-12 and r["exact_force_error"] == 0
                              and r["independent_colors"] for r in references),
             "nonnegative_finite": all(r["nonnegative_finite"] for r in rows)}
    targets = {r["variant"]: r["worst_body_force_error"] < 0.10
               for r in rows if r["k"] == 16 and r["sweeps"] == 40}
    out = {"status": "SYNTHETIC-ONLY", "rows": rows, "instrument_gates": gates,
           "diagnostic_targets": targets,
           "scope": "fixed normal-only system; coarse solves add work; no engine or speed claim"}
    path = Path(__file__).resolve().parents[1] / "reports" / "innovation_stack_two_level_probe.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, sort_keys=True))
    # Any rejected schedule is a failed experiment and remains visible in the exit code.
    return 0 if all(gates.values()) and all(targets.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
