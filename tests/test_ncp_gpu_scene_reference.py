"""The multi-step GPU scene numbers, locked as data.

    OMP_NUM_THREADS=4 python -m pytest -q tests/test_ncp_gpu_scene_reference.py

These rows cost minutes each on a shared GPU, so they are not recomputed here. The
file `data/ncp/gpu_scene_reference.json` holds the measured values and this test
holds them to the bounds the measurement was made against, including the two
measured negatives: the 1000:1 column at 200 iterations and the K8 tower that never
reaches v_sleep = 1e-3.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = json.loads((ROOT / "data/ncp/gpu_scene_reference.json").read_text())["rows"]


def test_gliding_gap_separates_the_two_cone_models_by_eight_decades():
    r = REF["sliding_box_gliding_gap_m"]
    assert r["exact_cone_max_gap"] < r["bound_exact_cone_max_gap"]
    assert r["convex_relaxation_max_gap"] > r["bound_convex_relaxation_min_gap"]
    assert r["convex_relaxation_max_gap"] / r["exact_cone_max_gap"] > 1e7


def test_sliding_distance_error_is_first_order_in_dt_and_not_the_cone_model():
    r = REF["sliding_distance_vs_closed_form"]
    ratio = r["exact_cone_dt_240_rel_err"] / r["exact_cone_dt_960_rel_err"]
    assert 3.9 < ratio < 4.1, f"4x refinement gave {ratio:.3f}x, not first order"
    gap = abs(r["exact_cone_dt_240_rel_err"] - r["convex_dt_240_rel_err"])
    assert gap < 0.01 * abs(r["exact_cone_dt_240_rel_err"]), (
        "the two cone models must differ far below the discretisation error")


def test_massratio_column_negative_is_recorded_not_smoothed():
    r = REF["massratio_column"]
    assert r["ratio_1_iters_800_max_pen_mm"] < r["ratio_1_iters_200_max_pen_mm"]
    # the negative: 200 iterations do not hold the 1000:1 column
    assert r["ratio_1000_iters_200_max_excursion_mm"] > 100.0
    assert r["ratio_1000_iters_800_max_excursion_mm"] < 5.0
    assert r["ratio_1000_iters_800_residual"] > 1e-3, (
        "800 iterations hold the column geometrically but do not solve it")
    assert r["substeps_S4_same_budget"]["penetration_factor"] > 2.0


def test_tower_penetration_against_the_independent_oracle():
    r = REF["tower_k8"]
    assert r["penetration_over_radius"] < r["independent_oracle_penetration_over_radius"]
    assert r["sleeping_at_1e-3"] == "never"
    assert r["jitter_floor_m_per_s"] > 1e-3, (
        "the jitter floor sits above the sleep threshold; that is why the tower never sleeps")


def test_sleeping_win_is_small_and_host_bound():
    r = REF["sleeping_lattice_n10000"]
    assert r["contacts_after"] < 0.15 * r["contacts_before"]
    assert 1.1 < r["speedup"] < 1.3, (
        "a 10x contact reduction buys 1.18x: the colour loop is launch-bound")


def test_int64_fixed_point_drift_fix():
    r = REF["int64_fixed_point_drift"]
    assert r["after_fix"] < 1e-9
    assert r["before_fix"] / r["after_fix"] > 1e4


def test_colour_cache_and_capture():
    r = REF["colour_cache_and_graph_capture"]
    assert r["cache_warm_ms"] < min(r["cache_cold_ms_range"])
    assert r["cache_speedup_at_4096"] > 50.0
    assert r["graph_capture_speedup"] > 2.0
