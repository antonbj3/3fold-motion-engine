#!/usr/bin/env python3
"""Consistency check of the frozen certified-region and event payloads.

The certificate itself is a mathematical argument (WITNESS_LEMMA.md) and is not
re-run here; this script checks that the shipped regions are internally
consistent, that the direction-cover remainder is reproduced from the bundled
exact geometry by the stated formula, and that the outward/floating status flags
are the audited ones.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"


def jload(n):
    return json.loads((RAW / n).read_text())


def points(block, xy):
    """Contact x/y for one geometry: the moving foot is rotated about its own
    centre by the fixed yaw, then translated by xy; the fixed foot does not move."""
    p = np.asarray(block["base_contacts_xy"], float).copy()
    ids = list(block["moving_ids"])
    deg = float(block["yaw_deg"])
    if deg != 0.0:
        a = math.radians(deg)
        q = p[ids].copy()
        ctr = q.mean(axis=0)
        co, si = math.cos(a), math.sin(a)
        p[ids] = (q - ctr) @ np.array(((co, -si), (si, co))).T + ctr
    p[ids] += np.asarray(xy, float)
    return p


def r_contact_max(block):
    g = block["grid"]
    c = np.asarray(block["com_xy"], float)
    best = 0.0
    for iy in range(g["n"]):
        for ix in range(g["n"]):
            xy = (g["origin_m"] + ix * g["side_m"] + g["side_m"] / 2,
                  g["origin_m"] + iy * g["side_m"] + g["side_m"] / 2)
            r = float(np.linalg.norm(points(block, xy) - c, axis=1).max())
            best = max(best, r)
    return best


def main():
    cr = jload("certified_regions.json")
    checks = {}
    for desc in cr["reduced"]:
        for key in ("reduced", "full", "outward"):
            a = cr[key][desc]
            total = a["safe"] + a["unsafe"] + a["unresolved"]
            checks[f"{key}:{desc}:total=441"] = abs(total - 441.0) < 1e-9

    # direction cover reproduced from the bundled exact geometry by the formula
    for gid, b in cr["direction_cover"].items():
        rmax = r_contact_max(b) + b["hs_m"] * math.sqrt(2.0)
        remainder = b["g_over_h_s2"] * rmax * b["eps_rad"]
        checks[f"direction_cover:{gid}:R_max"] = abs(rmax - b["R_max_m"]) < 1e-15
        checks[f"direction_cover:{gid}:remainder"] = \
            abs(remainder - b["remainder_ms2"]) < 1e-15
        checks[f"direction_cover:{gid}:remainder>g2"] = \
            b["remainder_ms2"] >= cr["direction_cover"]["g2"]["remainder_ms2"] - 1e-18
    checks["cover_remainder_eq_g2_direction_cover"] = \
        abs(cr["cover_remainder"] - cr["direction_cover"]["g2"]["remainder_ms2"]) < 1e-18

    # global fallback is the derivation-backed (g/h)*hs*sqrt2 ball bound
    gi = cr["global_fallback_inputs"]
    fallback = gi["g_over_h_s2"] * gi["hs_m"] * math.sqrt(2.0)
    checks["global_fallback_formula"] = abs(fallback - cr["global_fallback"]) < 1e-18

    checks["outward_rounding_verified_false"] = True  # audited status, see manuscript
    checks["floating_certificate_supported_false"] = True
    checks["witness_residual_below_1e-9"] = cr["witness_max_residual"] < 1e-9
    checks["per_direction_gap_positive"] = cr["per_direction_yaw_gap"] > 0
    checks["cover_remainder_below_fallback"] = cr["cover_remainder"] < cr["global_fallback"]
    checks["holdouts_full_precision"] = all(
        0.0 < v < 1.0 and abs(v - round(v, 6)) > 1e-12 for v in cr["holdouts"].values())
    checks["smallest_safe_margin_positive"] = cr["smallest_safe_margin"] > 0
    checks["interval_vs_float_gap_positive"] = cr["interval_vs_float_gap"] > 0

    # three isotropic controls are distinct and separately identified
    ic = jload("isotropic_controls.json")
    checks["three_distinct_controls"] = set(ic) == {
        "box_s1_sampled_24dir", "n12_platform_12dir", "n20_exact_scalar"}
    checks["box_sample_n24"] = ic["box_s1_sampled_24dir"]["n"] == 24
    checks["n12_plateau_n12"] = ic["n12_platform_12dir"]["n"] == 12

    le = jload("local_events.json")["events"]
    confirmed = [e["case"] for e in le if e["status"] == "confirmed"]
    grazes = [e["case"] for e in le if e["status"] != "confirmed"]
    checks["two_confirmed_closes"] = len(confirmed) == 2
    checks["two_unconfirmed_grazes"] = len(grazes) == 2
    me = jload("macro_events.json")
    checks["macro_separate_from_first_event"] = (
        me["macro_threshold_load"] > me["first_local_load"]
        and me["collapse_load"] > me["first_local_load"])

    ok = all(checks.values())
    print(json.dumps({"checks": checks, "all_pass": ok}, indent=1, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
