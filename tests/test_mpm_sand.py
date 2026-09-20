"""Material-point sand: the anchors that hold, and the bearing capacity that does not.

    OMP_NUM_THREADS=4 python -m pytest -s -q tests/test_mpm_sand.py

Green here means an anchor the model reproduces. The rows that fail are xfail(strict)
carrying the measured limit, so a change that moves them shows up as an unexpected pass
rather than disappearing. Nothing in this file re-runs a grid: a single triaxial sweep
row costs seconds to a minute of shared GPU and the whole res x dt table costs minutes,
so the table is committed with its per-row sha256 and the closed-form parts are
recomputed.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SWEEP = json.loads((ROOT / "data/mpm/sand_res_dt_sweep.json").read_text())
BEARING = json.loads((ROOT / "data/mpm/sand_bearing_reference.json").read_text())
ROWS = SWEEP["rows"]


def _rows_at(dt_us):
    return [r for r in ROWS if r["dt_us"] == dt_us]


# ───────────────────────── green: the constitutive anchors ─────────────────────────

def test_drucker_prager_cone_equals_mohr_coulomb_in_triaxial_compression():
    """The identity the triaxial anchor rests on: M = 6 sin phi / (3 - sin phi), exactly."""
    from motion_engine.mpm import geotech
    for phi_deg in (20.0, 30.0, 40.0):
        s = np.sin(np.radians(phi_deg))
        M = 6.0 * s / (3.0 - s)
        phi_back = np.degrees(np.arcsin(3.0 * M / (6.0 + M)))
        assert phi_back == pytest.approx(phi_deg, rel=1e-12)
    assert 6.0 * np.sin(np.radians(40.0)) / (3.0 - np.sin(np.radians(40.0))) == pytest.approx(
        1.6361383785219161, rel=1e-12), "the M target the sweep measures against"


@pytest.mark.parametrize("row", _rows_at(10.0), ids=lambda r: f"res{r['res']}")
def test_triaxial_friction_angle_is_recovered_at_a_converged_time_step(row):
    """phi = 40 deg back out of the grid within half a degree at dt = 10 us, at every
    resolution from 32 to 128. The documented gate is 2 deg; the measured spread is 0.49."""
    err = row["phi_out_deg"] - SWEEP["phi_in_deg"]
    assert abs(err) <= SWEEP["acceptance_gate_deg"]
    assert abs(err) <= 0.5, f"res {row['res']}: {row['phi_out_deg']} deg"
    s = np.sin(np.radians(row["phi_out_deg"]))
    assert row["M_mean"] == pytest.approx(6.0 * s / (3.0 - s), rel=2e-3)
    assert len(row["sha256"]) == 64


def test_the_angle_is_resolution_independent_once_the_time_step_converges():
    phi = [r["phi_out_deg"] for r in _rows_at(10.0)]
    assert len(phi) == 4
    assert max(phi) - min(phi) < 0.3, phi
    # halving dt again changes it by at most 0.02 deg
    by_res = {r["res"]: r["phi_out_deg"] for r in _rows_at(5.0)}
    for r in _rows_at(10.0):
        if r["res"] in by_res:
            assert abs(by_res[r["res"]] - r["phi_out_deg"]) <= 0.02, r["res"]


def test_the_dt_rule_is_what_the_coarse_rows_measured():
    """res 96 at 20 us and res 128 at 15-20 us are the CFL failures the rule exists for."""
    bad = {(r["res"], r["dt_us"]): r["phi_out_deg"] for r in ROWS
           if r["res"] >= 96 and r["dt_us"] > 10.0}
    assert bad[(96, 20.0)] == 38.53
    assert bad[(128, 20.0)] is None, "res 128 at 20 us blew up"
    assert bad[(128, 15.0)] == 35.89
    assert SWEEP["dt_rule"]["rule"] == "dt <= 10 us at res >= 96"
    assert SWEEP["what_the_audit_overturned"]["verdict"] == "falls"


def test_the_band_width_spread_is_not_threshold_independent():
    """The 4.5 % spread was reported as evidence of no grid dependence; at twice the
    threshold it is 336 %, so it carries nothing on its own."""
    o = SWEEP["what_the_audit_overturned"]
    assert o["band_width_spread_percent"] == pytest.approx(4.49)
    assert "336" in o["band_spread_reservation"]


def test_geostatic_initialisation_closes_the_bed():
    g = BEARING["geostatic_initialisation"]
    assert g["without_initialisation"]["surface_settlement_mm"] > 80.0
    assert abs(g["with_K0_phi33"]["surface_settlement_mm"]) < 1e-3
    assert g["with_K0_phi33"]["ke_over_mgh"] < 1e-8
    assert abs(g["with_K0_phi33"]["bottom_over_weight"] - 1.0) < 0.01
    # Jaky: K0 = 1 - sin phi, and the mean-pressure ratio follows it
    for phi, ratio in ((21.0, g["jaky_mean_pressure_ratio"]["phi21"]),
                       (33.0, g["jaky_mean_pressure_ratio"]["phi33"])):
        k0 = 1.0 - np.sin(np.radians(phi))
        assert ratio == pytest.approx((1.0 + 2.0 * k0) / 3.0, rel=1e-6)


def test_bearing_capacity_formulas_and_the_local_failure_reduction():
    """The module's own closed forms, against the ratios the comparison quotes. The 2x
    a brief claimed for Vesic over Hansen is wrong; the interval is 1.4 to 1.5."""
    from motion_engine.mpm.geotech import calc_bearing_factors
    d = BEARING["discrete_element_comparison"]
    lo, hi = d["local_failure_reduces_ngamma_by"]
    for phi_deg, expected in ((30.0, d["nq_ratio_vesic_over_hansen"]["phi30"]),
                              (35.0, d["nq_ratio_vesic_over_hansen"]["phi35"]),
                              (40.0, d["nq_ratio_vesic_over_hansen"]["phi40"])):
        f = calc_bearing_factors(phi_deg)
        assert f["Ny_Vesic"] / f["Ny_Hansen"] == pytest.approx(expected, rel=0.02), phi_deg
        assert 1.3 < f["Ny_Vesic"] / f["Ny_Hansen"] < 1.6, "not the 2x a brief claimed"
        # local failure: phi* = atan(2/3 tan phi)
        phi_star = np.degrees(np.arctan(2.0 / 3.0 * np.tan(np.radians(phi_deg))))
        drop = f["Ny_Vesic"] / calc_bearing_factors(phi_star)["Ny_Vesic"]
        assert lo <= drop <= hi, f"phi {phi_deg}: local failure lowers N_gamma by {drop:.2f}x"


def test_refining_the_grains_moves_the_discrete_model_away_from_terzaghi():
    d = BEARING["discrete_element_comparison"]
    ratios = [r["ratio"] for r in sorted(d["grain_size_sweep"], key=lambda r: r["w_over_d"])]
    assert ratios == sorted(ratios, reverse=True), ratios
    assert ratios[0] > 1.0 > ratios[-1]
    assert d["richardson_order"] < 1.0, "sublinear: the extrapolated limit carries no weight"


# ───────────────────────── xfail: the limits, with their numbers ─────────────────────

@pytest.mark.xfail(strict=True, reason="measured: crater prefactor 0.8664 against the "
                                       "published 0.3298 m^0.1667 = 2.63x, gate 30 %")
def test_crater_prefactor_matches_the_published_law():
    c = BEARING["crater_prefactor"]
    assert abs(c["prefactor_ratio"] - 1.0) <= c["prefactor_gate_rel"]


@pytest.mark.xfail(strict=True, reason="measured: drop-height exponent 0.0714 against "
                                       "1/3, error 0.262, gate 0.10")
def test_crater_drop_height_exponent_matches_the_published_law():
    c = BEARING["crater_prefactor"]
    assert abs(c["exponents_measured_in_metres"]["H"]
               - c["exponents_published"]["H"]) <= c["exponent_gate_abs"]


@pytest.mark.parametrize("name", ["D_b", "rho"])
def test_the_two_crater_exponents_that_do_pass(name):
    """Ball diameter and density exponents are inside the gate; only drop height is not."""
    c = BEARING["crater_prefactor"]
    assert abs(c["exponents_measured_in_metres"][name]
               - c["exponents_published"][name]) <= c["exponent_gate_abs"]


@pytest.mark.xfail(strict=True, reason="measured: the stiff bed gives 0.197 and 0.309 of "
                                       "the published crater depth, geometric mean 0.247, "
                                       "gate 0.7 to 1.3")
def test_stiff_bed_crater_depth_matches_the_published_law():
    c = BEARING["crater_prefactor"]
    lo, hi = c["stiff_bed_ratio_to_uehara"]["gate"]
    assert lo <= c["stiff_bed_ratio_to_uehara"]["geometric_mean"] <= hi


@pytest.mark.parametrize("case", ["K0 phi21", "K0 phi33"])
def test_strip_footing_matches_vesic_local_failure(case, request):
    """Measured 4.78x and 9.33x of Vesic local; the gate is 0.8 to 1.2."""
    request.node.add_marker(pytest.mark.xfail(
        strict=True, reason="measured: footing 4.783809x (phi 21) and 9.326082x (phi 33) "
                            "of Vesic local failure, gate 0.8 to 1.2"))
    f = BEARING["strip_footing"]
    row = next(r for r in f["rows"] if r["case"] == case)
    lo, hi = f["gate"]
    assert lo <= row["ratio"] <= hi


@pytest.mark.xfail(strict=True, reason="measured: the soft bed carries 0.72x Terzaghi and "
                                       "0.60x as the plain Drucker-Prager baseline")
def test_soft_bed_footing_matches_terzaghi():
    assert BEARING["strip_footing"]["soft_bed_ratio_to_terzaghi"] >= 0.8


@pytest.mark.xfail(strict=True, reason="measured: discrete 1.974x and 2.412x against "
                                       "material-point 0.60x of Terzaghi; the two do not "
                                       "agree within 20 %")
def test_the_two_carriers_agree_within_twenty_percent():
    d = BEARING["discrete_element_comparison"]
    a, b = d["dem_rigid_ratio"], d["mpm_drucker_prager_ratio"]
    assert abs(a - b) / max(a, b) <= 0.20
