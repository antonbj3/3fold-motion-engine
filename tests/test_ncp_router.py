"""The router: the spectral rho rules against a real sweep, the certificate gate, and
the exact device factorisation the x-step uses.

    OMP_NUM_THREADS=4 python -m pytest -s -q tests/test_ncp_router.py

`data/ncp/rho_oracle_sweep.csv` is the oracle: 21 scenes x 25 values of rho, ADMM from a
zero start to residual 1e-8 with a 5000-iteration cap, bit-identical over two runs
(525/525). The hit rates recomputed here from that file are the rule's score, and the
score is 5/21 for the structure choice. The earlier 18/21 was the prediction pasted in as
the oracle; this file replaces it.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from motion_engine.ncp import ncp_router as Router  # noqa: E402

SWEEP = list(csv.DictReader((ROOT / "data/ncp/rho_oracle_sweep.csv").open()))
WITHIN = 2.0


# ───────────────────────── the rho rules, scored against the sweep ─────────────────────────

def test_the_sweep_is_the_21_by_25_oracle():
    assert len(SWEEP) == 21
    assert sum(int(r["converged_grid_points"]) + int(r["censored_grid_points"])
               + int(r["numerical_failures"]) for r in SWEEP) == 21 * 25
    assert sum(int(r["censored_grid_points"]) for r in SWEEP) == 223, (
        "223 of 525 grid points hit the 5000-iteration cap; a censored scene is a miss")
    assert sum(1 for r in SWEEP if r["boundary_minimum"] == "True") == 6, (
        "6 scenes have their minimum on the edge of the grid and identify no optimum")


def test_rule_hit_rates_are_the_measured_ones():
    hits = {
        "A": sum(1 for r in SWEEP if float(r["ratio_A"]) <= WITHIN),
        "B": sum(1 for r in SWEEP if float(r["ratio_B"]) <= WITHIN),
        "structure": sum(1 for r in SWEEP if float(r["selected_ratio"]) <= WITHIN),
        "hindsight_best": sum(1 for r in SWEEP if float(r["best_ratio"]) <= WITHIN),
    }
    assert hits == {k: v[0] for k, v in Router.RULE_HIT_RATES.items()}, hits
    assert hits == {"A": 2, "B": 7, "structure": 5, "hindsight_best": 8}
    assert hits["hindsight_best"] > hits["structure"], (
        "picking the better of A and B after the fact beats the rule; that is not a rule")


def test_the_structure_choice_is_a_if_the_graph_is_a_chain():
    for r in SWEEP:
        expected = "A" if r["C1"] == "1" else "B"
        assert r["selected_rule"] == expected, r["scene"]
        # degree <= 2 is necessary but not sufficient: `humanoid` has degree 2 and C1 = 0
        if r["C1"] == "1":
            assert int(r["degree_max"]) <= 2, r["scene"]
    degree2_but_not_c1 = [r["scene"] for r in SWEEP
                          if int(r["degree_max"]) <= 2 and r["C1"] == "0"]
    assert degree2_but_not_c1 == ["humanoid"], degree2_but_not_c1


def test_structure_class_reproduces_the_sweeps_c1_on_a_chain_and_on_a_shared_body():
    """Two contacts on one body with opposite blocks is a chain; the same two contacts
    with blocks that are not negatives of each other is not."""
    dof = 6
    J = np.zeros((6, 2 * dof))
    J[0:3, 0:3] = np.eye(3); J[0:3, dof:dof + 3] = -np.eye(3)
    J[3:6, 0:3] = -np.eye(3); J[3:6, dof:dof + 3] = np.eye(3)
    s = Router.structure_class(J, 3, dof)
    assert s["degree_max"] == 2 and s["max_shared"] == 2 and s["C1"] == 0, s
    J2 = np.zeros((6, 2 * dof))
    J2[0:3, 0:3] = np.eye(3)
    J2[3:6, dof:dof + 3] = np.eye(3)
    s2 = Router.structure_class(J2, 3, dof)
    assert s2["degree_max"] == 1 and s2["max_shared"] == 0 and s2["C1"] == 1, s2


@pytest.mark.parametrize("row", SWEEP, ids=[r["scene"] for r in SWEEP])
def test_spectral_rho_formulas_match_the_sweeps_columns(row):
    lmin, lmax = float(row["lambda_min_positive"]), float(row["lambda_max"])
    assert float(row["rho_A"]) == pytest.approx(lmin, rel=1e-12)
    assert float(row["rho_B"]) == pytest.approx(np.sqrt(lmin * lmax), rel=1e-12)
    ratio_a = max(float(row["rho_star"]) / lmin, lmin / float(row["rho_star"]))
    assert float(row["ratio_A"]) == pytest.approx(ratio_a, rel=1e-9)


def test_spectral_rho_reproduces_lambda_min_plus_on_a_rank_deficient_operator():
    """lambda_min^+ must skip the numerical null space, not return it."""
    Q = np.linalg.qr(np.random.default_rng(0).normal(size=(9, 9)))[0]
    ev = np.array([0.0, 1e-17, 3e-3, 0.02, 0.1, 0.4, 1.0, 2.0, 5.0])
    G = Q @ np.diag(ev) @ Q.T
    s = Router.spectral_rho(G)
    assert s["lambda_max"] == pytest.approx(5.0, rel=1e-9)
    assert s["lambda_min_positive"] == pytest.approx(3e-3, rel=1e-6)
    assert s["rho_B"] == pytest.approx(np.sqrt(3e-3 * 5.0), rel=1e-6)


# ───────────────────────── the certificate gate ─────────────────────────

def _two_body_J():
    J = np.zeros((6, 12))
    J[0:3, 0:3] = np.eye(3)
    J[3:6, 6:9] = np.eye(3)
    return J


def _interior_scene():
    """Two contacts pressed straight down: the bilateral solution is strictly inside."""
    G = np.diag([2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    b = np.array([-1.0, 0.0, 0.0, -1.5, 0.0, 0.0])
    return G, b, np.array([0.6, 0.6]), _two_body_J()


def _cone_binding_scene():
    G = np.diag([2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    b = np.array([-1.0, 4.0, 0.0, -1.5, 0.0, 5.0])
    return G, b, np.array([0.1, 0.1]), _two_body_J()


def test_certificate_takes_the_interior_scene_without_any_iteration():
    G, b, mu, J = _interior_scene()
    cert = Router.bilateral_certificate(G, b, mu)
    assert cert["is_interior"] and cert["n_out"] == 0
    lam = cert["lam"]
    assert np.allclose(G @ lam + b, 0.0, atol=1e-12), "the closed set must be solved exactly"
    assert Router.route(G, b, mu, J)["branch"] == "exact_bilateral"


def test_certificate_refuses_when_the_cone_binds():
    G, b, mu, J = _cone_binding_scene()
    cert = Router.bilateral_certificate(G, b, mu)
    assert not cert["is_interior"] and cert["n_out"] > 0
    assert Router.route(G, b, mu, J)["branch"] != "exact_bilateral"


def test_articulated_is_an_argument_not_a_guess():
    """The audit reservation, as a test: routing to the PGS branch must require the caller
    to say so. Detected from a scene name it mis-routed a real quadruped by 28-39x."""
    G, b, mu, J = _cone_binding_scene()
    assert Router.route(G, b, mu, J, articulated=False)["branch"] != "pgs_certificate_stop"
    assert Router.route(G, b, mu, J, articulated=True)["branch"] == "pgs_certificate_stop"


# ───────────────────────── the exact device factorisation ─────────────────────────

def _spd(n, seed):
    M = np.random.default_rng(seed).normal(size=(n, n))
    return M @ M.T + n * np.eye(n)


@pytest.fixture(scope="module")
def cuda():
    wp = pytest.importorskip("warp")
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("the device factorisation needs a CUDA device")
    return wp


@pytest.mark.parametrize("n_c", [3, 4, 32, 64])
def test_cusolver_factorisation_is_bit_identical_run_to_run(cuda, n_c):
    from motion_engine.ncp.device_cholesky import TorchCholesky
    A = _spd(3 * n_c, seed=n_c)
    L1 = TorchCholesky().factor(A).L_np()
    L2 = TorchCholesky().factor(A).L_np()
    assert L1.tobytes() == L2.tobytes(), f"n_c={n_c}: two factorisations differ"
    b = np.random.default_rng(7).normal(size=3 * n_c)
    x = TorchCholesky().factor(A).solve_np(b)
    assert np.abs(A @ x - b).max() < 1e-10


@pytest.mark.parametrize("n_c", [3, 8])
def test_single_threaded_warp_reference_agrees_with_lapack(cuda, n_c):
    """The deterministic reference, at the sizes where a one-thread Cholesky is cheap."""
    from motion_engine.ncp.device_cholesky import WarpCholesky
    A = _spd(3 * n_c, seed=100 + n_c)
    c = WarpCholesky(3 * n_c).factor(A)
    assert np.abs(c.L() - np.linalg.cholesky(A)).max() < 1e-11
    assert c.L().tobytes() == WarpCholesky(3 * n_c).factor(A).L().tobytes()
    b = np.random.default_rng(11).normal(size=3 * n_c)
    assert np.abs(A @ c.solve_np(b) - b).max() < 1e-10


def test_the_512_contact_ceiling_is_stated():
    from motion_engine.ncp.device_cholesky import TorchCholesky, CUSOLVER_ULP_SCENES
    assert TorchCholesky.MAX_CONTACTS == 512
    assert max(CUSOLVER_ULP_SCENES) == 512
