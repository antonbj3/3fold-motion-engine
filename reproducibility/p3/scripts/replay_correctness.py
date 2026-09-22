#!/usr/bin/env python3
"""Exact-count agreement matrix on the frozen states (GPU replay).

On the frozen states (120/140/150/160/160-shifted) and the geometry families used by the timing
matrix, checks that all exact paths agree:

  independent NumPy D1 (bench_common.d1_all)  ==  range_reference D1  ==  owned/gated path  ==
  direct strong kernel  ==  cell-list baseline  ==  sorted-key baseline

Counts are the only decision output, so zero mismatch is the precondition for any timing claim.
Usage: python replay_correctness.py [--out PATH]
"""
import argparse
import json
import os
import sys
import zlib
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import bench_common as U  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "replay_out" / "correctness.json"))
    args = ap.parse_args()

    import warp as wp
    wp.init()
    dev = "cuda:0"
    import producer_grid_keepout as K
    import owned_keepout_contract as O
    import baseline_cell_list as B1
    import baseline_sorted_key as B2

    fam = U.box_config()
    lat = K.Lattice(res=U.RES, origin=tuple(U.ORIGIN), dx=U.DX, rho=U.RHO)

    rows = []
    for fkey in ["120", "140", "150", "160", "160geom"]:
        fr = U.load_frame(fkey)
        xn, xa, m, ids, grid, n = (fr["x_now"], fr["x_at"], fr["mass"], fr["ids"], fr["grid"],
                                   fr["n"])
        xa_w = wp.array(xa, dtype=wp.vec3, device=dev)
        xn_w = wp.array(xn, dtype=wp.vec3, device=dev)
        m_w = wp.array(m, dtype=wp.float32, device=dev)
        ids_w = wp.array(ids, dtype=wp.int64, device=dev)
        grid_w = wp.array(grid, dtype=wp.float32, device=dev)
        base_frame = int(fkey[:3])
        cap = O.capture_from_solver_grid(grid_w, xa_w, m_w, ids_w, lat, frame=base_frame,
                                         version=1, verify=True, device=dev)
        st = O.state_from_particles(xn_w, m_w, ids_w, frame=base_frame + 1, version=1, device=dev)
        batch = O.bind_batch(cap, st, device=dev)
        b1 = B1.CellListBaseline(U.RES, U.ORIGIN, U.DX, dev).build(xn_w, n)
        b2 = B2.SortedKeyBaseline(U.RES, U.ORIGIN, U.DX, dev).build(xn_w, n)
        for dist in ["sparse", "allnear", "slab", "corner", "empty"]:
            cfg = fam["box_dists"][dist]
            for nb in [32, 256, 1024]:
                if dist in ("allnear", "corner", "empty") and nb == 32:
                    continue
                if fkey != "160" and not (dist == "sparse" and nb == 32):
                    continue
                seed = 90000 + nb + (zlib.crc32(("%s|%s" % (fkey, dist)).encode()) % 1000)
                blo, bhi = U.boxes_for_dist(cfg, nb, seed)
                c_np = U.d1_all(xn, blo, bhi)
                c_ref = U.C.d1_counts_all(xn, blo, bhi)
                c_gate, gate, info = batch.query(blo, bhi)
                c_dir = K.exact_box_counts(xn_w, blo, bhi, n=n, device=dev)
                c_b1, w1 = b1.query(blo, bhi)
                c_b2, w2 = b2.query(blo, bhi)
                rec = {
                    "frame": fkey, "dist": dist, "nb": nb, "n": n, "route": info["route"],
                    "n_flagged": int((gate > 0).sum()),
                    "np_eq_ref": bool(np.array_equal(c_np, c_ref)),
                    "gate_eq_np": bool(np.array_equal(c_gate.astype(np.int64), c_np)),
                    "direct_eq_np": bool(np.array_equal(c_dir.astype(np.int64), c_np)),
                    "base1_eq_np": bool(np.array_equal(c_b1.astype(np.int64), c_np)),
                    "base2_eq_np": bool(np.array_equal(c_b2.astype(np.int64), c_np)),
                    "max_abs_diff_any": int(max(
                        np.abs(c_gate.astype(np.int64) - c_np).max(),
                        np.abs(c_dir.astype(np.int64) - c_np).max(),
                        np.abs(c_b1.astype(np.int64) - c_np).max(),
                        np.abs(c_b2.astype(np.int64) - c_np).max())),
                    "sum_counts": int(c_np.sum()),
                    "base1_candidate_tests": w1, "base2_candidate_tests": w2,
                }
                rows.append(rec)
                ok = all(rec[k] for k in ("np_eq_ref", "gate_eq_np", "direct_eq_np",
                                          "base1_eq_np", "base2_eq_np"))
                print("%-7s %-7s %4d route=%-6s flagged=%3d counts=%8d ok=%s" % (
                    fkey, dist, nb, info["route"], rec["n_flagged"], rec["sum_counts"], ok),
                    flush=True)

    all_ok = all(all(r[k] for k in ("np_eq_ref", "gate_eq_np", "direct_eq_np", "base1_eq_np",
                                    "base2_eq_np")) for r in rows)
    zero_mismatch = all(r["max_abs_diff_any"] == 0 for r in rows)
    out = {"kind": "correctness", "d1_definition": fam["d1_definition"], "n_cases": len(rows),
           "all_paths_agree": bool(all_ok), "zero_count_mismatch": bool(zero_mismatch),
           "checks": rows}
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("ALL_PATHS_AGREE", all_ok, "ZERO_COUNT_MISMATCH", zero_mismatch, flush=True)


if __name__ == "__main__":
    main()
