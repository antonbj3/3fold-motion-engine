#!/usr/bin/env python3
"""Matched complete-cost replay on the frozen states (GPU).

Bounded question: on the SAME captured producer state and the SAME strict-D1 particle-in-AABB
count, does reuse of the producer's mass grid beat a competent exact spatial accelerator including
its build/update cost?

Paths compared (per cell: state, box-distance family, nb, Q):
  * gated_amort : owned capture + owned state + bind_batch + Q gated queries (grid reused)
  * gated_full  : capture + state + Q x (bind + query)
  * cell_list   : state copy + cell-list build + Q exact range queries
  * sorted_key  : state copy + radix-sorted-key build + Q exact range queries
  * direct      : state copy + Q direct exact queries (uncharged reference semantics)

All paths return the identical exact strict-D1 integer count (validated by replay_correctness.py).
Times are paired 5-repeat medians; the device was shared during measurement, so the ratios and
paired medians are the carried quantities. Usage: python replay_benchmark.py --tag a
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import bench_common as U  # noqa: E402

REPEATS = 5

CELLS = [
    ("160", "sparse", 1024, "win_new_frame_sparse1024"),
    ("150", "sparse", 1024, "win_known_frame_sparse1024"),
    ("160", "slab", 1024, "negative_new_geometry_slab1024"),
    ("160", "allnear", 256, "negative_dense_allnear256"),
    ("160", "corner", 256, "negative_corner256"),
    ("160", "sparse", 32, "small_sparse32"),
    ("160", "sparse", 1, "small_sparse1"),
]


def gpu_mem():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True).strip().splitlines()[0]
        return int(out)
    except Exception:  # noqa: BLE001
        return -1


def timed(fn, repeats, device):
    import warp as wp
    ts = []
    for _ in range(repeats):
        wp.synchronize_device(device)
        t0 = time.perf_counter()
        fn()
        wp.synchronize_device(device)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), [float(t) for t in ts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="a")
    ap.add_argument("--out", default=None)
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
    mem0 = gpu_mem()

    rec = {"tag": args.tag, "gpu": "shared", "python": sys.executable, "repeats": REPEATS,
           "unit_note": "every *_s key is perf_counter seconds; *_ms = *_s*1e3; ratios dimensionless",
           "lattice": {"res": U.RES, "origin": list(U.ORIGIN), "dx": U.DX, "rho": U.RHO},
           "d1_definition": fam["d1_definition"],
           "paths": {
               "gated_amort": "owned capture + owned state + bind_batch (extra P2G + validation "
                              "+ occupancy) + Q gated queries; reuses the producer grid",
               "gated_full": "owned capture + owned state + Q x (bind_batch + query)",
               "cell_list": "state copy + histogram/prefix-scan/scatter cell-list build + Q exact "
                            "range queries on the cell list",
               "sorted_key": "state copy + radix-sort cell-key build (no prefix scan) + Q exact "
                             "range queries by binary search",
               "direct": "state copy + Q direct exact box counts (uncharged reference)",
           },
           "build_regimes": {
               "per_frame": "all complete paths rebuild capture/state/index for the frame before Q "
                            "queries (end-to-end per-frame cost)",
               "reusable": "the accelerator (or grid) is built once and Q query batches reuse it; "
                           "query-only cost reported separately",
           }}

    frame_cache = {}

    def frame_data(fkey):
        if fkey not in frame_cache:
            frame_cache[fkey] = U.load_frame(fkey)
        return frame_cache[fkey]

    cells = CELLS
    sel = os.environ.get("BENCH_CELLS", "")
    if sel:
        keep = set(sel.split(","))
        cells = [c for c in cells if c[3] in keep]
    Q = int(os.environ.get("BENCH_Q", "16"))
    scen = []
    for ci, (frame, dist, nb, label) in enumerate(cells):
        fr = frame_data(frame)
        xn, xa, m, ids, grid, n = (fr["x_now"], fr["x_at"], fr["mass"], fr["ids"], fr["grid"],
                                   fr["n"])
        xa_w = wp.array(xa, dtype=wp.vec3, device=dev)
        xn_w = wp.array(xn, dtype=wp.vec3, device=dev)
        m_w = wp.array(m, dtype=wp.float32, device=dev)
        ids_w = wp.array(ids, dtype=wp.int64, device=dev)
        grid_w = wp.array(grid, dtype=wp.float32, device=dev)
        cfg = fam["box_dists"][dist]
        base_frame = int(frame[:3])
        boxes = [U.boxes_for_dist(cfg, nb, 81000 + (7000 * ci) + (j * 7919)) for j in range(Q)]
        mem_before = gpu_mem()

        blo, bhi = boxes[0]
        c_np = U.d1_all(xn, blo, bhi)

        def direct_complete():
            s = O.state_from_particles(xn, m, ids, frame=base_frame + 1, version=7, device=dev)
            for bl, bh in boxes:
                K.exact_box_counts(s._x, bl, bh, n=n, device=dev)

        def gated_amort_complete():
            c = O.capture_from_solver_grid(grid_w, xa_w, m_w, ids_w, lat, frame=base_frame,
                                           version=7, verify=True, device=dev)
            s = O.state_from_particles(xn, m, ids, frame=base_frame + 1, version=7, device=dev)
            bb = O.bind_batch(c, s, device=dev)
            for bl, bh in boxes:
                bb.query(bl, bh)

        def gated_full_complete():
            c = O.capture_from_solver_grid(grid_w, xa_w, m_w, ids_w, lat, frame=base_frame,
                                           version=7, verify=True, device=dev)
            s = O.state_from_particles(xn, m, ids, frame=base_frame + 1, version=7, device=dev)
            for bl, bh in boxes:
                bb = O.bind_batch(c, s, device=dev)
                bb.query(bl, bh)

        def celllist_complete():
            s = O.state_from_particles(xn, m, ids, frame=base_frame + 1, version=7, device=dev)
            base = B1.CellListBaseline(U.RES, U.ORIGIN, U.DX, dev)
            base.build(s._x, n)
            for bl, bh in boxes:
                base.query(bl, bh)

        def sortedkey_complete():
            s = O.state_from_particles(xn, m, ids, frame=base_frame + 1, version=7, device=dev)
            base = B2.SortedKeyBaseline(U.RES, U.ORIGIN, U.DX, dev)
            base.build(s._x, n)
            for bl, bh in boxes:
                base.query(bl, bh)

        cap_r = O.capture_from_solver_grid(grid_w, xa_w, m_w, ids_w, lat, frame=base_frame,
                                           version=7, verify=True, device=dev)
        st_r = O.state_from_particles(xn, m, ids, frame=base_frame + 1, version=7, device=dev)
        sess_r = O.bind_batch(cap_r, st_r, device=dev)
        base1_r = B1.CellListBaseline(U.RES, U.ORIGIN, U.DX, dev).build(xn_w, n)
        base2_r = B2.SortedKeyBaseline(U.RES, U.ORIGIN, U.DX, dev).build(xn_w, n)

        def gated_query_only():
            for bl, bh in boxes:
                sess_r.query(bl, bh)

        def celllist_query_only():
            for bl, bh in boxes:
                base1_r.query(bl, bh)

        def sortedkey_query_only():
            for bl, bh in boxes:
                base2_r.query(bl, bh)

        c_dir0 = K.exact_box_counts(xn_w, blo, bhi, n=n, device=dev)
        c_ga, _, i_ga = sess_r.query(blo, bhi)
        c_gf, _, _ = O.bind_batch(cap_r, st_r, device=dev).query(blo, bhi)
        c_b1, _ = base1_r.query(blo, bhi)
        c_b2, _ = base2_r.query(blo, bhi)
        agree = {
            "direct_eq_np": bool(np.array_equal(c_dir0.astype(np.int64), c_np)),
            "gated_amort_eq_np": bool(np.array_equal(c_ga.astype(np.int64), c_np)),
            "gated_full_eq_np": bool(np.array_equal(c_gf.astype(np.int64), c_np)),
            "celllist_eq_np": bool(np.array_equal(c_b1.astype(np.int64), c_np)),
            "sortedkey_eq_np": bool(np.array_equal(c_b2.astype(np.int64), c_np)),
        }

        t_dir, r_dir = timed(direct_complete, REPEATS, dev)
        t_ga, r_ga = timed(gated_amort_complete, REPEATS, dev)
        t_gf, r_gf = timed(gated_full_complete, REPEATS, dev)
        t_b1, r_b1 = timed(celllist_complete, REPEATS, dev)
        t_b2, r_b2 = timed(sortedkey_complete, REPEATS, dev)
        t_qga, r_qga = timed(gated_query_only, REPEATS, dev)
        t_qb1, r_qb1 = timed(celllist_query_only, REPEATS, dev)
        t_qb2, r_qb2 = timed(sortedkey_query_only, REPEATS, dev)
        mem_after = gpu_mem()

        def ratio(a, b):
            return (a / b) if b else None

        row = {
            "label": label, "frame": frame, "dist": dist, "nb": nb, "Q": Q, "n": n,
            "route": i_ga["route"], "n_exact_boxes": int(i_ga["n_exact_boxes"]),
            "sum_counts_box0": int(c_np.sum()), "gpu": "shared",
            "direct_complete_s": t_dir, "direct_complete_repeats_s": r_dir,
            "gated_amort_complete_s": t_ga, "gated_amort_repeats_s": r_ga,
            "gated_full_complete_s": t_gf, "gated_full_repeats_s": r_gf,
            "celllist_complete_s": t_b1, "celllist_repeats_s": r_b1,
            "sortedkey_complete_s": t_b2, "sortedkey_repeats_s": r_b2,
            "gated_query_only_s": t_qga, "gated_query_only_repeats_s": r_qga,
            "celllist_query_only_s": t_qb1, "celllist_query_only_repeats_s": r_qb1,
            "sortedkey_query_only_s": t_qb2, "sortedkey_query_only_repeats_s": r_qb2,
            "gated_amort_over_direct": ratio(t_ga, t_dir),
            "gated_amort_over_celllist": ratio(t_ga, t_b1),
            "gated_amort_over_sortedkey": ratio(t_ga, t_b2),
            "celllist_over_direct": ratio(t_b1, t_dir),
            "sortedkey_over_direct": ratio(t_b2, t_dir),
            "gated_query_only_over_celllist_query_only": ratio(t_qga, t_qb1),
            "gated_query_only_over_sortedkey_query_only": ratio(t_qga, t_qb2),
            "gain_gated_amort_vs_direct": (1.0 - t_ga / t_dir) if t_dir else None,
            "gain_gated_amort_vs_celllist": (1.0 - t_ga / t_b1) if t_b1 else None,
            "gain_gated_amort_vs_sortedkey": (1.0 - t_ga / t_b2) if t_b2 else None,
            "gpu_mem_used_mib_before": mem_before, "gpu_mem_used_mib_after": mem_after,
            "agree": agree,
        }
        row["all_paths_agree_box0"] = all(agree.values())
        scen.append(row)
        print("%-34s %-7s %4d route=%-6s flag=%3d  gated/direct=%.3f gated/celllist=%.3f "
              "agree=%s" % (label, dist, nb, i_ga["route"], row["n_exact_boxes"],
                            row["gated_amort_over_direct"], row["gated_amort_over_celllist"],
                            row["all_paths_agree_box0"]), flush=True)

    rec["scenarios"] = scen
    dec = [{"label": s["label"], "sum_counts_box0": s["sum_counts_box0"], "route": s["route"],
            "n_exact_boxes": s["n_exact_boxes"]} for s in scen]
    rec["decision_canonical_sha256"] = hashlib.sha256(
        json.dumps(dec, sort_keys=True).encode()).hexdigest()
    op = Path(args.out) if args.out else (HERE.parent / "replay_out" / ("benchmark_%s.json"
                                                                        % args.tag))
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print("decision sha256:", rec["decision_canonical_sha256"])
    print("gpu mem mib:", mem0, gpu_mem())


if __name__ == "__main__":
    main()
