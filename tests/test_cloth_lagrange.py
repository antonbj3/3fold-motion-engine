"""Lagrangian cloth: the bending law and the anchors it reproduces, and the drape
coefficient it does not.

    OMP_NUM_THREADS=4 python -m pytest -s -q tests/test_cloth_lagrange.py

The bending law is two parallel lines of slope B separated by 2HB, implemented as a
Jenkin element (a spring in series with a slider of yield moment) in parallel with the
elastic spring. Both B and 2HB come from one published KES-FB2 table; nothing is fitted.
The law and its residual curvature are closed form and are recomputed here. The drape
runs are minutes of shared GPU each, so their numbers are committed and asserted, with
the two limits as xfail(strict) rows.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REF = json.loads((ROOT / "data/cloth/bending_and_drape_reference.json").read_text())
FABRICS = REF["fabrics"]
LOOP = REF["moment_law_verification"]
DRAPE = REF["drape_coefficient"]
GREEN = REF["green_anchors"]
NAMES = sorted(FABRICS)


def moment(theta, theta_rest, theta_slider, K_e, K_p, M_y):
    """M(theta) = K_e (theta - theta_rest) + clamp(K_p (theta - theta_slider), -M_y, M_y)."""
    return K_e * (theta - theta_rest) + np.clip(K_p * (theta - theta_slider), -M_y, M_y)


# ───────────────────────── green: the measured parameters and the law ────────────────

@pytest.mark.parametrize("name", NAMES)
def test_kes_units_convert_exactly(name):
    f = FABRICS[name]
    c = REF["unit_conversion"]
    assert f["B_Nm"] == pytest.approx(f["B_gf_cm2_per_cm"] * c["gf_cm2_per_cm_to_Nm"], rel=1e-9)
    assert f["twoHB_Npm"] == pytest.approx(
        f["twoHB_gf_cm_per_cm"] * c["gf_cm_per_cm_to_Nm_per_m"], rel=1e-9)


@pytest.mark.parametrize("name", NAMES)
def test_residual_curvature_is_2hb_over_2b(name):
    """The unloaded strip stops at kappa_res = 2HB / (2B). Closed form, no simulation."""
    f = FABRICS[name]
    kappa = f["twoHB_Npm"] / (2.0 * f["B_Nm"])
    assert kappa == pytest.approx(f["kappa_res_per_m"], rel=1e-12)
    assert 1.0 / kappa * 1e3 == pytest.approx(f["R_res_mm"], rel=1e-4)


@pytest.mark.parametrize("name", NAMES)
def test_the_implemented_loop_returns_b_and_2hb(name):
    """Running the law over a load-unload loop must give back the two numbers it was
    built from. Measured: B to 8.9e-14 relative and 2HB exactly."""
    v = LOOP[name]
    assert abs(v["B_err_pct"]) < 1e-12
    assert abs(v["twoHB_err_pct"]) < 1e-12
    assert abs(v["kappa_res_err_pct"]) < 1e-12
    assert v["kappa_res_from_unloading_1pm"] == pytest.approx(
        -v["kappa_res_from_reloading_1pm"], rel=1e-12)
    assert v["slope_branch"] == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("name", NAMES)
def test_the_jenkin_law_reproduces_the_loop_width_and_the_slider_band(name):
    """Independent re-derivation. With the slider at either extreme the two branches are
    2 M_y apart, which is the loop width; the slider travels 2 M_y / K_p before it moves,
    which is the stick band. Both in curvature units at K_e = 1."""
    v, f = LOOP[name], FABRICS[name]
    K_e = 1.0
    K_p = v["ratio"] * K_e
    M_y = f["kappa_res_per_m"] * K_e           # yields at the residual curvature
    up = moment(60.0, 0.0, 60.0 - M_y / K_p, K_e, K_p, M_y)
    down = moment(60.0, 0.0, 60.0 + M_y / K_p, K_e, K_p, M_y)
    assert (up - down) / K_e == pytest.approx(v["loop_width_at_plus_halfcm_1pm"], rel=1e-9)
    assert v["loop_width_at_plus_halfcm_1pm"] == pytest.approx(
        v["loop_width_at_minus_halfcm_1pm"], rel=1e-12), "the loop must be symmetric"
    assert 2.0 * M_y / K_p == pytest.approx(v["stick_band_1pm"], rel=1e-9)
    assert v["loop_width_at_plus_halfcm_1pm"] == pytest.approx(
        2.0 * f["kappa_res_per_m"], rel=1e-12)


def test_the_moment_law_is_psd_in_both_branches():
    """Sticking gives K_e + K_p, sliding gives K_e; both non-negative, so the
    Gauss-Newton tangent stays positive semi-definite."""
    K_e, K_p, M_y = 2.0, 40.0, 3.0
    th = np.linspace(-1.0, 1.0, 2001)
    m = moment(th, 0.0, 0.0, K_e, K_p, M_y)
    slope = np.diff(m) / np.diff(th)
    assert slope.min() >= K_e - 1e-9
    assert slope.max() <= K_e + K_p + 1e-9
    assert np.all(np.diff(m) >= 0.0), "the law must be monotone, hence integrable to a convex potential"


def test_the_static_anchors_are_inside_their_gate():
    g = GREEN
    for v in g["cantilever_tip_error_percent"].values():
        if isinstance(v, (int, float)):
            assert v <= g["cantilever_tip_error_percent"]["gate"]
    for v in g["tension_strip_error_percent"].values():
        if isinstance(v, (int, float)):
            assert v <= g["tension_strip_error_percent"]["gate"]
    lo, hi = g["residual_curvature_error_percent_range"]
    assert hi < 1.0, "the released strip stops within a percent of the analytical curvature"
    for name, (a, b) in g["residual_curvature_measured_per_m"].items():
        target = FABRICS[name]["kappa_res_per_m"]
        assert min(abs(a - target), abs(b - target)) / target * 100 <= hi


def test_self_contact_is_certified_on_the_run_that_reaches_the_target():
    c = GREEN["contact_certificate_fabric_A"]
    assert c["contacts_below_minus_half_thickness"] == 0
    assert c["max_self_penetration_m"] < c["half_thickness_m"]
    assert c["ncp_residual"] < 1e-10
    assert c["hinges_dropped"] == 0


# ───────────────────────── xfail: the two measured limits ────────────────────────────

@pytest.mark.xfail(strict=True, reason="measured: without the hysteresis term fabric A "
                                       "stops at 24.770 % against 82.9, -58.1 points")
def test_drape_without_hysteresis_reaches_the_published_value():
    assert DRAPE["fabric_A_without_hysteresis_percent"] >= 0.9 * DRAPE["fabric_A_target_percent"]


@pytest.mark.xfail(strict=True, reason="measured: with the hysteresis term fabric A "
                                       "passes 82.9 % at t = 0.96 s and keeps falling to "
                                       "57.317 % at 3.00 s, still -9.89 points per second")
def test_drape_with_hysteresis_settles_at_the_published_value():
    h = DRAPE["fabric_A_with_hysteresis"]
    assert abs(h["value_at_3_s_percent"] - DRAPE["fabric_A_target_percent"]) <= 5.0


def test_the_hysteresis_term_brakes_the_drape_it_does_not_stop_it():
    """Both halves of the limit, as one statement: it does reach the target on the way
    past, and it does not stay there."""
    h = DRAPE["fabric_A_with_hysteresis"]
    assert h["value_at_0.97_s_percent"] < DRAPE["fabric_A_target_percent"] < h["value_at_0.96_s_percent"]
    assert h["still_falling_pp_per_s"] > 0.0
    assert DRAPE["fabric_A_without_hysteresis_percent"] < h["value_at_3_s_percent"]


def test_fabric_f_is_reported_as_the_solvers_number_not_the_models():
    """An area floor changes no force and no energy, only the Newton direction, and still
    moves the drape coefficient by 8.5 points. Those runs decide nothing about the model."""
    f = DRAPE["fabric_F_is_the_solvers_number_not_the_models"]
    assert abs(f["round5_percent"] - f["round6_percent"]) == pytest.approx(f["difference_pp"], abs=0.02)
    assert "unconverged" in f["cause"]
    assert DRAPE["verdict"].startswith("fails")
    assert "too elastic" in DRAPE["two_measured_limits"] and "too soft" in DRAPE["two_measured_limits"]


def test_the_modules_are_importable_and_carry_the_law():
    from motion_engine.cloth import r6_kernels, r6_solver, r6_setup
    assert hasattr(r6_solver, "Cloth6")
    assert hasattr(r6_setup, "hinge_bend_length")
