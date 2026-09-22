#!/usr/bin/env python3
"""Regenerate every displayed table from the frozen raw measurement artefacts (CPU only).

The tables are recomputed from `data/measurements/*.json`; no printed number is copied into a table
source. The decisive quantity is the explicit per-run ratio

    t_gate_over_best = t_gated_amort_complete_s / min(t_celllist_complete_s, t_sortedkey_complete_s)

where a value above 1 means the producer-grid reuse is SLOWER than the best exact accelerator. Every
ratio is computed within a single run (the same-run inputs); medians from different runs are never
combined into one measured path.

Usage:
  python regenerate_tables.py                 # write all tables into expected/
  python regenerate_tables.py --table matched # write one table
  python regenerate_tables.py --out DIR       # write into DIR
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
MEAS = PKG / "data" / "measurements"
EXP = PKG / "expected"


def load(name):
    return json.loads((MEAS / name).read_text())


def set_meas(path):
    """Point the generator at an alternative measurement directory (used by the corruption control)."""
    global MEAS
    MEAS = Path(path)


def g(v, nd=9):
    if v is None:
        return ""
    return ("%." + str(nd) + "g") % float(v)


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return path


def best_of(scen):
    """Return (t_best, kind) from the same scenario's cell-list and sorted-key medians."""
    cl = scen.get("celllist_complete_s")
    sk = scen.get("sortedkey_complete_s")
    if cl is None:
        return sk, "sorted_key"
    if sk is None:
        return cl, "cell_list"
    return (cl, "cell_list") if cl <= sk else (sk, "sorted_key")


def table_matched(out):
    """Primary matched comparison, per run (same-run inputs only)."""
    header = ["run", "label", "frame", "dist", "nb", "Q", "n", "route", "n_flagged",
              "t_gate_s", "t_celllist_s", "t_sortedkey_s", "t_best_s", "best_kind",
              "t_gate_over_best", "t_gate_over_direct", "best_over_direct",
              "t_gate_query_only_s", "t_gate_query_only_over_best_query_only", "device",
              "time_boundary"]
    rows = []
    for run in ["a", "b", "repeat_a", "repeat_b", "query1", "query4"]:
        fname = {"a": "matched_cost_run_a.json", "b": "matched_cost_run_b.json",
                 "repeat_a": "matched_cost_repeat_a.json",
                 "repeat_b": "matched_cost_repeat_b.json",
                 "query1": "matched_cost_query1.json", "query4": "matched_cost_query4.json"}[run]
        rec = load(fname)
        for s in rec["scenarios"]:
            tb, kind = best_of(s)
            tg = s["gated_amort_complete_s"]
            td = s["direct_complete_s"]
            qcl = s.get("celllist_query_only_s")
            qsk = s.get("sortedkey_query_only_s")
            qbest = min([x for x in (qcl, qsk) if x is not None]) if (qcl or qsk) else None
            qgate = s.get("gated_query_only_s")
            rows.append([
                run, s["label"], s["frame"], s["dist"], s["nb"], s["Q"], s["n"], s["route"],
                s["n_exact_boxes"], g(tg), g(s["celllist_complete_s"]),
                g(s["sortedkey_complete_s"]), g(tb), kind,
                g(tg / tb) if tb else "", g(tg / td) if td else "",
                g(tb / td) if td else "", g(qgate),
                g(qgate / qbest) if (qbest and qgate) else "", "shared GPU", "paired 5-repeat median",
            ])
    rows.sort(key=lambda r: (r[0], r[1]))
    return write_csv(out / "table_matched_comparison.csv", header, rows)


def table_query_sweep(out):
    """Q sweep on the two cells present in all runs, same-run ratios."""
    header = ["cell", "Q", "run", "t_gate_s", "t_celllist_s", "t_sortedkey_s", "t_best_s",
              "t_gate_over_best", "t_gate_over_direct", "t_gate_query_only_over_best_query_only",
              "device", "time_boundary"]
    files = {"query1": "matched_cost_query1.json", "query4": "matched_cost_query4.json",
             "a": "matched_cost_run_a.json"}
    rows = []
    for cell in ["win_new_frame_sparse1024", "negative_dense_allnear256"]:
        for run, fname in files.items():
            rec = load(fname)
            for s in rec["scenarios"]:
                if s["label"] != cell:
                    continue
                tb, _ = best_of(s)
                tg = s["gated_amort_complete_s"]
                td = s["direct_complete_s"]
                qcl = s.get("celllist_query_only_s")
                qsk = s.get("sortedkey_query_only_s")
                qbest = min([x for x in (qcl, qsk) if x is not None]) if (qcl or qsk) else None
                qgate = s.get("gated_query_only_s")
                rows.append([cell, s["Q"], run, g(tg), g(s["celllist_complete_s"]),
                             g(s["sortedkey_complete_s"]), g(tb), g(tg / tb) if tb else "",
                             g(tg / td) if td else "",
                             g(qgate / qbest) if (qbest and qgate) else "",
                             "shared GPU", "paired 5-repeat median"])
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    return write_csv(out / "table_query_sweep.csv", header, rows)


def table_producer_cost(out):
    """Producer-side complete cost vs the direct kernel only (not vs a spatial index)."""
    header = ["run", "label", "frame", "dist", "nb", "Q", "route", "n", "n_flagged_summed",
              "amort_over_direct", "complete_gain_vs_direct", "full_over_direct",
              "query_only_over_direct", "tests_reported_eq_actual", "device", "time_boundary"]
    rows = []
    for fname in ["producer_complete_cost_run_a.json", "producer_complete_cost_run_b.json"]:
        rec = load(fname)
        run = {"p1": "a", "p2": "b"}.get(rec.get("tag"), rec.get("tag", "?"))
        for s in rec["scenarios"]:
            rows.append([
                run, s["label"], s["frame"], s["dist"], s["nb"], s["Q"],
                s["route"], s["n"],
                s["n_flagged_summed"], g(s["amort_over_direct"]), g(s["complete_gain_amort"]),
                g(s["full_over_direct"]), g(s["query_only_over_direct"]),
                bool(s["n_box_particle_tests_actual_summed"]
                     == s["n_box_particle_tests_amort_summed"]), "shared GPU",
                "paired 5-repeat median"])
    rows.sort(key=lambda r: (r[0], r[1]))
    return write_csv(out / "table_producer_complete_cost.csv", header, rows)


def table_audit_reproduction(out):
    """Independent audit reproduction alongside the producer run (labelled by source, not merged)."""
    header = ["cell", "source", "run", "t_gate_s", "t_celllist_s", "t_sortedkey_s", "t_best_s",
              "t_gate_over_best", "t_gate_over_direct", "device", "time_boundary"]
    rows = []
    # producer matched run a
    rec = load("matched_cost_run_a.json")
    for s in rec["scenarios"]:
        tb, _ = best_of(s)
        rows.append([s["label"], "producer", "a", g(s["gated_amort_complete_s"]),
                     g(s["celllist_complete_s"]), g(s["sortedkey_complete_s"]), g(tb),
                     g(s["gated_amort_complete_s"] / tb) if tb else "",
                     g(s["gated_amort_complete_s"] / s["direct_complete_s"])
                     if s["direct_complete_s"] else "", "shared GPU", "paired 5-repeat median"])
    # independent audit runs a/b (different labels)
    for run, fname in [("a", "audit_independent_cost_run_a.json"),
                       ("b", "audit_independent_cost_run_b.json")]:
        rec = load(fname)
        for s in rec["scenarios"]:
            cl = s.get("celllist_s")
            sk = s.get("sortedkey_s")
            tb = min([x for x in (cl, sk) if x is not None]) if (cl or sk) else None
            tg = s.get("gated_amort_s")
            td = s.get("direct_s")
            rows.append([s["label"], "independent audit", run, g(tg), g(cl), g(sk), g(tb),
                         g(tg / tb) if (tb and tg) else "",
                         g(tg / td) if (td and tg) else "", "shared GPU",
                         "paired 5-repeat median"])
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    return write_csv(out / "table_audit_reproduction.csv", header, rows)


def table_gate_stress(out):
    header = ["tag", "scale_dx", "state_frame", "D", "reason", "route", "n_flagged",
              "sum_counts", "counts_eq_oracle", "gate_false_negatives"]
    rec = load("audit_gate_stress.json")
    rows = [[r["tag"], g(r["scale_dx"]), r["state_frame"], g(r["D"]), r["reason"], r["route"],
             r["n_flagged"], r["sum_counts"], r["counts_eq_oracle"], r["gate_false_negatives"]]
            for r in rec["results"]]
    rows.sort(key=lambda r: r[0])
    return write_csv(out / "table_gate_stress.csv", header, rows)


def table_correctness_summary(out):
    header = ["check", "value", "source"]
    corr = load("correctness_matrix.json")
    recon = load("audit_reconcile.json")
    nsc = load("audit_new_state_cases.json")
    orc = load("audit_oracle_crosscheck.json")
    rows = [
        ["matched_correctness_n_cases", corr["n_cases"], "correctness_matrix.json"],
        ["matched_all_paths_agree", corr["all_paths_agree"], "correctness_matrix.json"],
        ["matched_zero_count_mismatch", corr["zero_count_mismatch"], "correctness_matrix.json"],
        ["audit_csv_matches_raw", recon["csv_matches_raw"], "audit_reconcile.json"],
        ["audit_unit_errors", len(recon["unit_errors"]), "audit_reconcile.json"],
        ["audit_route_consistency_ok", recon["route_consistency_ok"], "audit_reconcile.json"],
        ["audit_gain_vs_best_label_consistent", recon["gain_vs_best_label_consistent"],
         "audit_reconcile.json"],
        ["audit_new_state_cases_pass", nsc["all_cases_pass"], "audit_new_state_cases.json"],
        ["audit_oracle_crosscheck_ok", orc["all_ok"], "audit_oracle_crosscheck.json"],
    ]
    for st, v in sorted(orc["states"].items()):
        rows.append(["audit_oracle_" + st + "_n_boxes", v["n_boxes"], "audit_oracle_crosscheck.json"])
    return write_csv(out / "table_correctness_summary.csv", header, rows)


TABLES = {
    "matched": table_matched,
    "query_sweep": table_query_sweep,
    "producer_cost": table_producer_cost,
    "audit_reproduction": table_audit_reproduction,
    "gate_stress": table_gate_stress,
    "correctness_summary": table_correctness_summary,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", choices=sorted(TABLES), default=None)
    ap.add_argument("--out", default=str(EXP))
    ap.add_argument("--meas", default=None, help="alternative measurement directory")
    args = ap.parse_args()
    if args.meas:
        set_meas(args.meas)
    out = Path(args.out)
    names = [args.table] if args.table else sorted(TABLES)
    for name in names:
        p = TABLES[name](out)
        print("wrote", p)


if __name__ == "__main__":
    main()
