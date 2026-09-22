#!/usr/bin/env python3
"""run table builder: run_records.jsonl -> matched.csv.

One row per (scene, B, acc_mode).  For every row the cost, the physical residual and
the derivative error come from the SAME timed execution (per direction); the row reports
the spread over the three declared directions and, separately, whether the three
repetition processes produced byte-identical output.

Gates (declared, not tuned after seeing the data):
    forward   : res_device_max < 1e-8              (stop tolerance of the timed graph)
    derivative: ref_err_rel_l2  < 1e-8             (independent CPU implicit reference)
    joint     : forward AND derivative on one execution.
A row that fails a gate keeps the failure; nothing is re-run to change a criterion.

  python matched_graph_table.py --records run_records.jsonl --out matched.csv
"""
import argparse
import csv
import json
import os
import statistics as st

RES_TOL = 1e-8
REF_TOL = 1e-8


def gpu_util(s):
    try:
        return int(s.rsplit(",", 1)[-1].strip().rstrip("%"))
    except Exception:                                          # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=os.path.join(os.path.dirname(__file__),
                                                      "run_records.jsonl"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "matched.csv"))
    a = ap.parse_args()

    rs = [json.loads(l) for l in open(a.records) if l.strip()]
    cells = {}
    for r in rs:
        cells.setdefault((r["scene"], r["B"], r["acc_mode"]), []).append(r)

    order = {"cube": 0, "eight_box_tower": 1, "eight_box_tower_mass_imbalance": 2, "lattice": 3, "sphere_packing_200": 4}
    rows = []
    for (scene, B, acc), group in sorted(
            cells.items(), key=lambda kv: (order.get(kv[0][0], 9), kv[0][1], kv[0][2])):
        base = group[0]
        g = base["graph"]
        dirs = sorted({r["graph"]["jdir"] for r in group})
        ms = [r["timing"]["graph_launch_ms"] for r in group]
        util = [gpu_util(r.get("gpu_after", "")) for r in group]
        util = [u for u in util if u is not None]
        res = [r["outputs"]["res_device_max"] for r in group]
        errs_by_dir = {}
        for d in dirs:
            e = [r["outputs"]["ref_err_rel_l2"] for r in group
                 if r["graph"]["jdir"] == d]
            errs_by_dir[d] = max(e) if e else None
        err_all = [r["outputs"]["ref_err_rel_l2"] for r in group]
        slip_ok = all(r["outputs"]["n_slip_device_max"] == r["outputs"]["n_slip_cpu"]
                      for r in group)
        fwd_ok = all(x < RES_TOL for x in res)
        deriv_ok = all(x < REF_TOL for x in err_all)
        deriv_ok6 = all(x < 1e-6 for x in err_all)
        reps = [r for r in group if str(r.get("run_id", "")).startswith("rep")]
        if reps:
            digs = [r["outputs"]["digest"] for r in reps]
            repeat_equal = len(set(digs)) == 1
            n_proc = len(reps)
        else:
            repeat_equal = ""
            n_proc = 1
        rows.append({
            "scene": scene, "B": B, "acc_mode": acc,
            "n_dir": len(dirs), "dirs": "|".join(str(d) for d in dirs),
            "n_proc_repeat": n_proc, "repeat_digest_equal": repeat_equal,
            "n_iter": g["n_iter"], "cg_iters": g["cg_iters"],
            "grad_cg": g["grad_cg"], "refresh_every": g["refresh_every"],
            "tol": g["tol"], "n_calls": g["n_calls"],
            "rho": base["rho"]["value"], "rho_mode": base["rho"]["mode"],
            "res_device_max": max(res), "res_converged": fwd_ok,
            "res_ref_cpu": base["outputs"]["res_ref_cpu"],
            "ref_err_rel_l2_max": max(err_all),
            "ref_err_rel_l2_by_dir": ";".join(
                f"{d}:{errs_by_dir[d]:.3e}" for d in dirs),
            "deriv_accepted_1e8": deriv_ok,
            "deriv_accepted_1e6": deriv_ok6,
            "active_set_slip_match": slip_ok,
            "joint_gate_1e8": bool(fwd_ok and deriv_ok),
            "joint_gate_1e6": bool(fwd_ok and deriv_ok6),
            "cost_graph_ms_median": st.median(ms),
            "cost_graph_ms_max": max(ms),
            "cost_joint_ok_1e6": st.median(ms) if (fwd_ok and deriv_ok6) else "",
            "gpu_util_after_max_pct": max(util) if util else "",
            "digest_first": group[0]["outputs"]["digest"],
            "module_sha256": base["provenance"]["module_sha256"],
            "refs_sha256": base["provenance"]["refs_sha256"],
            "status": base["status"],
        })

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ok = sum(1 for r in rows if r["joint_gate_1e8"])
    ok6 = sum(1 for r in rows if r["joint_gate_1e6"])
    print(f"[matched.csv] {len(rows)} cells, {ok}/{len(rows)} pass the joint 1e-8 gate, "
          f"{ok6}/{len(rows)} pass the joint 1e-6 gate; -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
