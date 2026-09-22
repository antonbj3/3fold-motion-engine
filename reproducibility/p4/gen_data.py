#!/usr/bin/env python3
"""Regenerate the public figure CSV payloads from the frozen raw inputs.

Reads only raw/*.json; writes data/*.csv.  The transformations match the
frozen public data byte-for-byte.  This script contains no external imports and
no private identifiers.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

GEOM_CODE = {"right foot +20 deg": "g2", "left foot +60 deg": "g8",
             "right foot -70 deg": "g9"}


def jload(name):
    return json.loads((RAW / name).read_text())


def write(name, rows):
    with (DATA / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def f1():
    d = jload("static_exactness.json")
    code = {"box": "S1", "stack": "S2", "pallet": "S3"}
    write("f1.csv", [{"scene": code[s["name"]],
                      "max_relative_residual": s["max_relative_residual"],
                      "polygon_error_soc_pct": s["polygon_error_soc_pct"],
                      "directions": s["directions"]} for s in d["scenes"]])


def f2():
    d = jload("anisotropic.json")
    rows = []
    for scene, tag in (("anisotropic-A", "A"), ("anisotropic-B", "B")):
        s = next(x for x in d["scenes"] if x["name"] == scene)
        for i, (lp, lo, hi) in enumerate(zip(s["lp_radii"], s["reached_low"],
                                             s["reached_high"])):
            rows.append({"scene": tag, "direction_deg": i * 30,
                         "static_lp_m_s2": lp, "reached_low_m_s2": lo,
                         "reached_high_m_s2": hi})
    write("f2.csv", rows)


def f3():
    d = jload("reached_ramp.json")
    rows = []
    for k, t in enumerate(d["durations_s"]):
        vals = [d["per_scene_mean_rel_poly"][k][j] for j in range(3)]
        rows.append({"duration_s": t,
                     "mean_abs_relative_error_pct": 100 * sum(vals) / 3,
                     "sample": "72 directions, three scenes"})
    for t, pct in zip(d["long_box_T_s"], d["long_box_rel_pct"]):
        rows.append({"duration_s": int(t) if float(t).is_integer() else t,
                     "mean_abs_relative_error_pct": pct,
                     "sample": "one box direction"})
    write("f3.csv", rows)


def f4():
    d = jload("humanoid.json")
    write("f4.csv", [{"direction_deg": a, "static_lp_m_s2": lp,
                      "tipping_m_s2": tip, "zmp_m_s2": zmp}
                     for a, lp, tip, zmp in zip(d["directions_deg"],
                                                d["lp_radii"], d["tipping"],
                                                d["zmp"])])


def f5():
    d = jload("placement_field.json")
    n = d["n"]
    lo = d["offset_min_m"]
    step = d["step_m"]
    rows = []
    for i in range(n):
        for j, v in enumerate(d["r_min"][i]):
            rows.append({"x_m": round(lo + j * step, 10),
                         "y_m": round(lo + i * step, 10),
                         "static_radius_m_s2": v,
                         "status": "no_static_double_support" if v <= 0 else "feasible"})
    write("f5.csv", rows)


def f6():
    d = jload("certified_regions.json")
    rows = []
    for desc in ("right foot +20 deg", "left foot +60 deg", "right foot -70 deg"):
        geo = GEOM_CODE[desc]
        for kind, areas in (("reduced scalar", d["reduced"][desc]),
                            ("full LP, real-arithmetic witness", d["full"][desc])):
            rows.append({"geometry": geo, "method": kind,
                         "safe_cell_area": areas["safe"],
                         "unsafe_cell_area": areas["unsafe"],
                         "unresolved_cell_area": areas["unresolved"]})
    write("f6.csv", rows)


def f7():
    d = jload("local_events.json")
    rows = []
    for e in d["events"]:
        b = e["branch_interval"]
        rows.append({"case": e["case"], "candidate_kind": e["kind"],
                     "predicted_load": e["predicted_load"],
                     "branch_low": b[0] if b else "",
                     "branch_high": b[1] if b else "",
                     "status": e["status"]})
    write("f7_events.csv", rows)
    m = jload("macro_events.json")
    write("f7_macro.csv", [{"case": m["case"],
                            "first_local_load": m["first_local_load"],
                            "collapse_load": m["collapse_load"],
                            "macro_threshold_load": m["macro_threshold_load"]}])


if __name__ == "__main__":
    for fn in (f1, f2, f3, f4, f5, f6, f7):
        fn()
    print("wrote", sorted(p.name for p in DATA.glob("*.csv")))
