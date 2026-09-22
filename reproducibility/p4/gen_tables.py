#!/usr/bin/env python3
"""Regenerate the machine-readable manuscript tables from raw/*.json.

Every table is a pure function of the frozen raw payloads in `raw/`; no value
is transcribed from the printed manuscript.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
TAB = HERE / "tables"
TAB.mkdir(exist_ok=True)

CODE = {"box": "S1", "stack": "S2", "pallet": "S3"}


def jload(n):
    return json.loads((RAW / n).read_text())


def write(name, rows):
    with (TAB / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ex = jload("static_exactness.json")
    sc = {s["name"]: s["n_contacts"] for s in jload("static_scenes.json")["scenes"]}
    write("T1_static_exactness.csv", [
        {"scene": CODE[s["name"]], "n_contacts": sc[s["name"]],
         "max_relative_residual": s["max_relative_residual"],
         "polygon_error_soc_pct": s["polygon_error_soc_pct"]}
        for s in ex["scenes"]])

    rp = jload("reached_ramp.json")
    rows = []
    for k, t in enumerate(rp["durations_s"]):
        for j in range(3):
            rows.append({"scene_group": f"planar-scene-{j+1}", "T_s": t,
                         "mean_rel_poly": rp["per_scene_mean_rel_poly"][k][j]})
    write("T2_reached_ramp.csv", rows)

    an = jload("anisotropic.json")
    ic = jload("isotropic_controls.json")
    rows = []
    for s in an["scenes"]:
        rows.append({"scene": s["name"], "max_rel_error_pct": s["max_rel_error_pct"],
                     "median_rel_error_pct": s["median_rel_error_pct"],
                     "radius_ratio": s["radius_ratio"], "control_kind": "anisotropic"})
    box = ic["box_s1_sampled_24dir"]
    rows.append({"scene": "box (S1), 24 sampled LP radii",
                 "max_rel_error_pct": "",
                 "median_rel_error_pct": "",
                 "radius_ratio": 1.0,
                 "control_kind": "four-contact box, friction-limited; sampled "
                                 f"radii constant to {box['spread_rel']:.2e}"})
    n12 = ic["n12_platform_12dir"]
    rows.append({"scene": "N12 platform (12 dirs, reached T=50 s)",
                 "max_rel_error_pct": 100.0 * n12["max_relative_error"],
                 "median_rel_error_pct": 100.0 * n12["median_relative_error"],
                 "radius_ratio": n12["lp_radius_ratio"],
                 "control_kind": "genuine 12-contact elliptical 3-fold-perturbed ring; separate control"})
    rows.append({"scene": "N20 (exact 3D scalar LP)",
                 "max_rel_error_pct": 100.0 * ic["n20_exact_scalar"]["rel_error"],
                 "median_rel_error_pct": "",
                 "radius_ratio": "",
                 "control_kind": "not a 12-direction sample"})
    write("T3_anisotropic.csv", rows)

    hu = jload("humanoid.json")
    write("T4_humanoid.csv", [
        {"direction_deg": a, "static_lp_m_s2": lp, "tipping_m_s2": tip,
         "zmp_m_s2": zmp} for a, lp, tip, zmp in zip(
            hu["directions_deg"], hu["lp_radii"], hu["tipping"], hu["zmp"])])

    cr = jload("certified_regions.json")
    rows = []
    for desc in cr["reduced"]:
        for method, key in (("reduced scalar", "reduced"),
                            ("full LP, real-arithmetic witness", "full"),
                            ("outward interval (not verified)", "outward")):
            a = cr[key][desc]
            rows.append({"geometry": desc, "method": method, "safe": a["safe"],
                         "unsafe": a["unsafe"], "unresolved": a["unresolved"]})
    write("T5_certified_regions.csv", rows)

    le = jload("local_events.json")
    write("T6_local_events.csv", [
        {"case": e["case"], "kind": e["kind"], "predicted_load": e["predicted_load"],
         "branch_low": e["branch_interval"][0] if e["branch_interval"] else "",
         "branch_high": e["branch_interval"][1] if e["branch_interval"] else "",
         "status": e["status"]} for e in le["events"]])

    me = jload("macro_events.json")
    write("T7_macro_events.csv", [me])

    # ── additional source-backed tables; control measurements kept separate ───
    ic = jload("isotropic_controls.json")
    box = ic["box_s1_sampled_24dir"]
    n12 = ic["n12_platform_12dir"]
    n20 = ic["n20_exact_scalar"]
    write("T8_isotropic_controls.csv", [
        {"control": "box S1 (24 sampled LP radii)", "quantity": "spread_rel",
         "value": box["spread_rel"], "note": box["kind"]},
        {"control": "box S1 (24 sampled LP radii)", "quantity": "poly_min",
         "value": box["poly_min"], "note": ""},
        {"control": "box S1 (24 sampled LP radii)", "quantity": "poly_max",
         "value": box["poly_max"], "note": ""},
        {"control": "N12 platform (12 dirs)", "quantity": "max_relative_error",
         "value": n12["max_relative_error"], "note": n12["kind"]},
        {"control": "N12 platform (12 dirs)", "quantity": "median_relative_error",
         "value": n12["median_relative_error"], "note": ""},
        {"control": "N12 platform (12 dirs)", "quantity": "lp_radius_ratio",
         "value": n12["lp_radius_ratio"], "note": ""},
        {"control": "N20 exact scalar", "quantity": "lp_exact_3d",
         "value": n20["lp_exact_3d"], "note": n20["kind"]},
        {"control": "N20 exact scalar", "quantity": "measured",
         "value": n20["measured"], "note": ""},
        {"control": "N20 exact scalar", "quantity": "rel_error",
         "value": n20["rel_error"], "note": ""},
    ])

    mn = jload("min_norm_falsification.json")["multistart"]
    write("T9_min_norm_falsification.csv", [
        {"quantity": "cold_norm_min", "value": mn["cold_norm_min"]},
        {"quantity": "cold_norm_max", "value": mn["cold_norm_max"]},
        {"quantity": "rand_min_min", "value": mn["rand_min_min"]},
        {"quantity": "rand_max_max", "value": mn["rand_max_max"]},
        {"quantity": "lp_norm_min", "value": mn["lp_norm_min"]},
        {"quantity": "lp_norm_max", "value": mn["lp_norm_max"]},
        {"quantity": "n_cases", "value": mn["n_cases"]},
    ])

    af = jload("active_set_falsification.json")
    write("T10_active_set_falsification.csv", [
        {"quantity": k, "value": v}
        for k, v in sorted(af["pooled_2scenes"].items())])

    mb = jload("macro_bound_falsification.json")
    rows = []
    for h, blk in mb["heights"].items():
        for p in blk["pred"]:
            rows.append({"height_m": blk["h"], "angle_deg": p["angle"],
                         "P0": p["P0"], "P1": p["P1"], "P2": p["P2"]})
    write("T11_macro_bound_falsification.csv", rows)

    ce = jload("continuation_events.json")["summary"]
    write("T12_continuation_events.csv", [
        {"quantity": k, "value": v} for k, v in sorted(ce.items())])

    print("wrote", sorted(p.name for p in TAB.glob("*.csv")))


if __name__ == "__main__":
    main()
