"""Exhaustive CPU contact reference for synthetic box fixtures.

The imported voxel geometry is unchanged. This reference models only the point
contact predicates and feature/pair keys, not Warp arithmetic or GPU execution.
"""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from motion_engine.contact_engine_gpu import _voxbox, _R


def fixture(k, lattice=False):
    rest = _voxbox(.3, .3, .2)
    centers = [[(i % 5) * .35, ((i // 5) % 5) * .35, .099 + (i // 25) * .195]
               for i in range(k)] if lattice else [[0, 0, .099 + i * .195] for i in range(k)]
    points = (np.array(centers)[:, None, :] + rest[None, :, :]).reshape(-1, 3)
    return points, np.repeat(np.arange(k), len(rest))


def exhaustive(points, owner):
    rows = []
    for i, x in enumerate(points):
        if x[2] < _R:
            rows.append((i, -1))
        for q in range(i + 1, len(points)):
            if owner[q] != owner[i]:
                d = np.linalg.norm(x - points[q])
                if 1e-9 < d < 2 * _R:
                    rows.append((i, q))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 2)


def keys(rows, owner, npoints):
    n = int(owner.max()) + 1
    feature = rows[:, 0] * (npoints + 1) + rows[:, 1] + 1
    other = np.where(rows[:, 1] < 0, -1, owner[np.maximum(rows[:, 1], 0)])
    pair = owner[rows[:, 0]] * (n + 1) + other
    return feature, pair


def main():
    out = {"status": "SYNTHETIC-ONLY", "rows": []}
    for name, k, lat in (("stack4", 4, False), ("stack16", 16, False), ("lattice50", 50, True)):
        p, o = fixture(k, lat)
        rows = exhaustive(p, o)
        fk, pk = keys(rows, o, len(p))
        reversed_rows = rows[::-1]
        rf, _ = keys(reversed_rows, o, len(p))
        out["rows"].append({"scene": name, "points": len(p), "all_point_pairs": len(p)*(len(p)-1)//2,
            "contacts": len(rows), "body_pairs": len(np.unique(pk)),
            "raw_order_changes_when_emission_reversed": not np.array_equal(rows, reversed_rows),
            "sorted_identity": bool(np.array_equal(rows, reversed_rows[np.argsort(rf)])),
            "unique_features": len(np.unique(fk)) == len(rows),
            "strict_feature_order": bool(np.all(np.diff(fk) > 0)),
            "sha256": hashlib.sha256(rows.tobytes()).hexdigest()})
    out["pass"] = all(r["sorted_identity"] and r["unique_features"] and r["strict_feature_order"]
                      and r["raw_order_changes_when_emission_reversed"] for r in out["rows"])
    (ROOT / "reports" / "innovation_contact_reference.json").write_text(json.dumps(out, indent=2, sort_keys=True)+"\n")
    print(json.dumps(out, sort_keys=True))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
