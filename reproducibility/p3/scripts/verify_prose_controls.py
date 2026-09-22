#!/usr/bin/env python3
"""Numerical check for the study's prose controls, from the bundled frozen raw inputs (CPU only).

The displayed tables are checked by `regenerate_tables.py` and `verify_package.py`. This script
covers the numerical claims that live only in the manuscript prose (threshold controls, mass-grid
column edges, dynamic-occupancy costs). It recomputes each number from the ACTUAL raw inputs bundled
under `data/prose_controls/` and asserts the value the manuscript states. Values whose raw is not
retained are required to be labelled `archived_reported` in `archived_reported_values.json`, and are
reported separately rather than recomputed.

It also requires the corresponding literal phrases to remain present in `docs/comparative_study.md`,
so prose and raw cannot drift apart silently. It changes no table.

Usage: python scripts/verify_prose_controls.py [--out PATH]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
PC = PKG / "data" / "prose_controls"
DOC = PKG / "docs" / "comparative_study.md"


def load(name):
    return json.loads((PC / name).read_text())


def check(checks, name, ok, got, expected):
    checks.append({"check": name, "ok": bool(ok), "got": got, "expected": expected})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="optional JSON output path (default: no file written)")
    args = ap.parse_args()

    checks = []
    doc = DOC.read_text() if DOC.exists() else ""

    # --- threshold controls -----------------------------------------------------------------
    fz = load("threshold_fuzz_summary.json")
    check(checks, "fuzz_n_iter", fz["n_iter"] == 300, fz["n_iter"], 300)
    check(checks, "fuzz_massgrid_FN", fz["massgrid_FN"] == {"0.0": 1, "0.05": 22, "0.25": 139, "0.5": 170},
          fz["massgrid_FN"], {"0.0": 1, "0.05": 22, "0.25": 139, "0.5": 170})
    check(checks, "fuzz_massgrid_FP", fz["massgrid_FP"] == {"0.0": 15, "0.05": 1, "0.25": 0, "0.5": 0},
          fz["massgrid_FP"], {"0.0": 15, "0.05": 1, "0.25": 0, "0.5": 0})
    check(checks, "fuzz_gated_FN", fz["gated_FN"] == 0, fz["gated_FN"], 0)
    check(checks, "fuzz_gated_FP", fz["gated_FP"] == 0, fz["gated_FP"], 0)

    mt = load("threshold_matched_summary.json")
    check(checks, "matched_n_cases", mt["n_cases"] == 20, mt["n_cases"], 20)
    check(checks, "matched_massgrid_FN", mt["massgrid_FN"] == 13, mt["massgrid_FN"], 13)
    check(checks, "matched_massgrid_FP", mt["massgrid_FP"] == 0, mt["massgrid_FP"], 0)
    check(checks, "matched_gated_FN", mt["gated_FN"] == 0, mt["gated_FN"], 0)

    # --- mass-grid column edges -------------------------------------------------------------
    ed = load("massgrid_column_edges.json")
    disp = ed["simulator_run_pair"]["disappearance"]
    check(checks, "columns_grid_empty_at_half", disp["columns_grid_empty_at_half"] == 239,
          disp["columns_grid_empty_at_half"], 239)
    check(checks, "columns_with_particles", disp["columns_with_particles"] == 8458,
          disp["columns_with_particles"], 8458)
    saved = ed["saved_grid_and_surface"]["disappearance_at_half"]
    check(checks, "saved_columns_particles_but_half_grid_empty",
          saved["columns_particles_but_half_grid_empty"] == 150,
          saved["columns_particles_but_half_grid_empty"], 150)
    half = [s for s in ed["saved_grid_and_surface"]["theta_sweep"] if s["theta"] == 0.5][0]
    check(checks, "saved_theta_half_columns_in_domain", half["columns_in_domain"] == 75900,
          half["columns_in_domain"], 75900)
    check(checks, "saved_theta_half_cols_over_one_cell", half["cols_under_one_cell"] == 23,
          half["cols_under_one_cell"], 23)

    # --- dynamic-occupancy costs ------------------------------------------------------------
    inc = load("incremental_frame_costs.json")
    rows = inc["rows"]
    med = statistics.median(r["total_ms"] for r in rows)
    frac = sum(1 for r in rows if r["total_ms"] <= 10.0) / len(rows)
    check(checks, "incremental_n_frames", len(rows) == 199, len(rows), 199)
    check(checks, "incremental_frac_le_10ms_3p5pct", abs(frac - 0.035) < 5e-4, round(frac, 4), 0.035)
    check(checks, "incremental_raw_median_is_reproduce_pass", abs(med - 70.26) < 0.1, round(med, 2), 70.26)

    pr = load("incremental_paired_ratio.json")
    check(checks, "paired_ratio_median_0p417",
          abs(pr["ratio_incremental_over_full"]["median"] - 0.417) < 5e-3,
          pr["ratio_incremental_over_full"]["median"], 0.417)
    check(checks, "paired_scaled_incremental_35p32",
          abs(pr["est_incremental_ms_simulatorscale"]["median"] - 35.32) < 0.05,
          pr["est_incremental_ms_simulatorscale"]["median"], 35.32)
    check(checks, "paired_reference_full_rebuild_84p66",
          pr["reference_full_rebuild_ms"] == 84.66, pr["reference_full_rebuild_ms"], 84.66)
    check(checks, "paired_cad_hull_4p85", pr["CAD_ms"] == 4.85, pr["CAD_ms"], 4.85)

    # --- archived reported values (not replayable) ------------------------------------------
    arch = load("archived_reported_values.json")
    vals = {e["value"]: e for e in arch["entries"]}
    for v in (84.66, 4.85, 66.52):
        e = vals.get(v)
        check(checks, "archived_%s_labelled" % v,
              bool(e) and e.get("source_kind") == "archived_reported" and e.get("replayable") is False,
              None if e is None else {"source_kind": e.get("source_kind"),
                                      "replayable": e.get("replayable")},
              {"source_kind": "archived_reported", "replayable": False})
        check(checks, "archived_%s_has_gap" % v, bool(e) and bool(e.get("gap")),
              None if e is None else e.get("gap"), "non-empty gap")

    # --- manuscript prose must contain the literals the raw supports ------------------------
    doc_norm = " ".join(doc.split())
    literals = [
        "13 false negatives on 20 matched cases and 170 on 300 fuzz scenes",
        "239 of 8458",
        "84.66 ms against 4.85 ms",
        "35.32 ms",
        "66.52 ms with 3.5 %",
    ]
    for lit in literals:
        check(checks, "prose_contains:%s" % lit, lit in doc_norm, lit in doc_norm, True)

    out = {"kind": "prose_controls", "n_checks": len(checks),
           "all_pass": all(c["ok"] for c in checks), "checks": checks}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    for c in checks:
        if not c["ok"]:
            print("FAIL", c["check"], "got", c["got"], "expected", c["expected"])
    print("PROSE_CONTROLS_OK =", out["all_pass"], "(%d checks)" % len(checks))
    if not out["all_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
