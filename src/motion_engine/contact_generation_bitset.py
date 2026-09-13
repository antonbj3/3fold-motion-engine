"""Canonical point-contact generation with per-cell bitsets and fused keys.

CPU prototype: stable point ids, exact radius predicate, no final sort. This does
not replace the GPU generator or claim equivalence of CPU and GPU arithmetic.
"""
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np
from motion_engine.contact_engine_gpu import _R


def generate(points, owner, reverse=False):
    points = np.asarray(points, dtype=np.float64)
    owner = np.asarray(owner, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or owner.shape != (len(points),):
        raise ValueError("Expected points[N,3] and owner[N]")
    if not np.isfinite(points).all() or np.any(owner < 0):
        raise ValueError("Finite points and nonnegative owners are required")
    if not len(points):
        return np.empty((0, 2), np.int64), np.empty(0, np.int64), np.empty(0, np.int64), 0
    cells = np.floor(points / (2 * _R)).astype(np.int64)
    grid = {}
    ids = reversed(range(len(points))) if reverse else range(len(points))
    for i in ids:
        key = tuple(cells[i])
        grid[key] = grid.get(key, 0) | (1 << i)
    offsets = list(itertools.product((-1, 0, 1), repeat=3))
    if reverse:
        offsets.reverse()
    rows, feature, pair = [], [], []
    stride, nbody = len(points) + 1, int(owner.max()) + 1
    distance_tests = 0

    def emit(i, q):
        rows.append((i, q))
        feature.append(i * stride + q + 1)
        pair.append(int(owner[i]) * (nbody + 1) + (int(owner[q]) if q >= 0 else -1))

    for i, point in enumerate(points):
        if point[2] < _R:
            emit(i, -1)
        candidates = 0
        for offset in offsets:
            candidates |= grid.get(tuple(cells[i] + offset), 0)
        candidates &= ~((1 << (i + 1)) - 1)
        while candidates:
            bit = candidates & -candidates
            q = bit.bit_length() - 1
            candidates ^= bit
            if owner[q] == owner[i]:
                continue
            distance_tests += 1
            distance = np.linalg.norm(point - points[q])
            if 1e-9 < distance < 2 * _R:
                emit(i, q)
    return (np.asarray(rows, dtype=np.int64).reshape(-1, 2), np.asarray(feature, np.int64),
            np.asarray(pair, np.int64), distance_tests)


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "probes"))
    from innovation_contact_reference import fixture, exhaustive, keys
    scenes = [(name, *fixture(k, lat)) for name, k, lat in
              (("stack4", 4, False), ("stack16", 16, False), ("lattice50", 50, True))]
    scenes.append(("cutoff_and_coincidence", np.array([
        [-.1, 0, .049], [-.1, 0, .049], [-.0000000001, 0, .049],
        [.0000000001, 0, .049], [-.2, 0, .051], [-.3, 0, .05]]), np.arange(6)))
    evidence = []
    for name, points, owner in scenes:
        reference = exhaustive(points, owner)
        rf, rp = keys(reference, owner, len(points))
        rows, fk, pk, tests = generate(points, owner)
        reverse = generate(points, owner, reverse=True)
        parity = np.array_equal(rows, reference) and np.array_equal(fk, rf) and np.array_equal(pk, rp)
        identity = all(a.tobytes() == b.tobytes() for a, b in zip((rows, fk, pk), reverse[:3]))
        evidence.append({"scene": name, "contacts": len(rows), "distance_tests": tests,
                         "exact_reference": bool(parity), "reverse_identity": bool(identity),
                         "canonical_order": bool(np.all(np.diff(fk) > 0)),
                         "sha256": hashlib.sha256(rows.tobytes()+fk.tobytes()+pk.tobytes()).hexdigest()})
    out = {"status": "SYNTHETIC-ONLY", "rows": evidence,
           "pass": all(r["exact_reference"] and r["reverse_identity"] and r["canonical_order"] for r in evidence),
           "scope": "CPU prototype only; no GPU arithmetic or speed claim"}
    (root / "reports" / "contact_generation_bitset.json").write_text(json.dumps(out, indent=2, sort_keys=True)+"\n")
    print(json.dumps(out, sort_keys=True))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
