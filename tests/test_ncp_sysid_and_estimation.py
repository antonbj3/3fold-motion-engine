"""Per-env friction estimation and object system identification: the numbers, and the
two gates that fail.

    OMP_NUM_THREADS=4 python -m pytest -s -q tests/test_ncp_sysid_and_estimation.py

The identification runs 512 envs over 100 steps of full floating-base dynamics with the
robot's mass matrix, bias terms and contact Jacobian recomputed at the integrated state
every step; the input trajectory is 164 MB. So the per-env Fisher matrices, the per-env
relative errors and the true parameters are committed under `data/ncp/sysid/` and the
Cramer-Rao arithmetic is redone here from them. That is the small-batch CRB test: the
bound is recomputed from the committed information matrices, not copied from a summary.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data/ncp"
REF = json.loads((DATA / "sysid_gn_reference.json").read_text())
CRB_SUMMARY = json.loads((DATA / "sysid_crb.json").read_text())
SCALE = json.loads((DATA / "sysid_scale.json").read_text())
BITS = json.loads((DATA / "sysid_batch_bits.json").read_text())
FRICTION_BITS = json.loads((DATA / "online_friction_bits.json").read_text())


def _crb():
    from motion_engine.ncp.sysid_gn import crb_from_fisher, HALF_NORMAL_MEDIAN
    return crb_from_fisher, HALF_NORMAL_MEDIAN


# ───────────────────────── Cramer-Rao, recomputed from the Fisher matrices ────────────

@pytest.mark.parametrize("window", [20, 50, 100])
def test_crb_recomputed_from_the_committed_fisher_matrices(window):
    crb_from_fisher, _ = _crb()
    d = np.load(DATA / f"sysid/crb_w{window}.npz")
    crb, lmin = crb_from_fisher(d["H"], d["true"])
    assert np.allclose(crb, d["crb"], rtol=1e-12, atol=0)
    assert np.allclose(lmin, d["lmin"], rtol=1e-12, atol=0)
    summary = CRB_SUMMARY[str(window)]
    for j, name in enumerate(("m", "I_zz", "mu")):
        assert float(np.median(crb[:, j])) == pytest.approx(
            summary[f"crb_rel_per_sigma_{name}_p50"], rel=1e-10)
    assert float(np.median(lmin)) == pytest.approx(summary["H_lmin_p50"], rel=1e-10)


def test_the_bound_tightens_as_the_window_grows():
    p50 = [CRB_SUMMARY[str(w)]["crb_rel_per_sigma_m_p50"] for w in (20, 50, 100)]
    assert p50[0] > p50[1] > p50[2]
    lmin = [CRB_SUMMARY[str(w)]["H_lmin_p50"] for w in (20, 50, 100)]
    assert lmin[0] < lmin[1] < lmin[2]


def test_the_estimator_sits_on_the_bound_at_sigma_one():
    crb_from_fisher, half_normal = _crb()
    gn = np.load(DATA / "sysid/gn_sigma1.0_w50.npz")
    c = np.load(DATA / "sysid/crb_w50.npz")
    crb, _ = crb_from_fisher(c["H"], c["true"])
    sigma = float(gn["sigma"])
    assert sigma == 1.0 and int(gn["window"]) == 50
    measured = float(np.median(gn["rel"][:, 0]))
    predicted = half_normal * sigma * float(np.median(crb[:, 0]))
    assert measured == pytest.approx(REF["on_the_cramer_rao_bound"]["measured_median_m"], rel=1e-6)
    assert predicted == pytest.approx(
        REF["on_the_cramer_rao_bound"]["half_normal_median_of_crb"], rel=1e-4)
    assert abs(measured - predicted) / predicted < 0.02, (
        f"measured {measured:.4e} against the bound {predicted:.4e}: the error is "
        f"information-bound, not a local minimum")


def test_the_error_grows_monotonically_with_the_noise():
    med = []
    for s in ("0.0", "0.1", "1.0", "5.0"):
        d = np.load(DATA / f"sysid/gn_sigma{s}_w50.npz")
        med.append(float(np.median(d["rel"][:, 0])))
    assert med == sorted(med), med
    assert med[0] < 1e-5, "at zero noise the estimator recovers the truth"


def test_the_acceptance_gate_fails_and_fails_on_the_bound():
    """p95 < 0.05 for m and mu at sigma_tau = 1 N m over 50 steps."""
    d = np.load(DATA / "sysid/gn_sigma1.0_w50.npz")
    p95_m = float(np.percentile(d["rel"][:, 0], 95))
    p95_mu = float(np.percentile(d["rel"][:, 2], 95))
    assert p95_m > 0.05 and p95_mu > 0.05, "the gate is recorded as failing; it must fail"
    assert p95_m == pytest.approx(REF["error_vs_noise_50_steps_p95"]["sigma_tau_1.0"]["m"], rel=1e-4)
    assert p95_mu == pytest.approx(REF["error_vs_noise_50_steps_p95"]["sigma_tau_1.0"]["mu"], rel=1e-4)
    # it fails because the bound itself is above the gate
    assert CRB_SUMMARY["50"]["crb_rel_per_sigma_m_p50"] > 0.05
    # and it passes one decade of noise lower
    e = np.load(DATA / "sysid/gn_sigma0.1_w50.npz")
    assert float(np.percentile(e["rel"][:, 0], 95)) < 0.05
    assert float(np.percentile(e["rel"][:, 2], 95)) < 0.05


def test_the_fisher_screen_is_below_its_own_oracle():
    r = REF["fisher_roc"]
    assert r["lambda_min_score_auc_50_steps"] < r["crb_score_auc_50_steps"]
    assert r["oracle_crb_at_true_parameters_auc_50_steps"] < 0.8, (
        "an oracle below 0.8 means 0.9 is out of reach for the observable, not the estimator")


def test_the_baseline_that_beats_the_estimator_is_the_one_handed_the_truth():
    b = REF["baselines_median_m_error"]
    assert b["gauss_newton"] < b["quasi_static_F_equals_ma"] / 5
    assert b["recursive_mu_with_true_mass"] < b["gauss_newton"], (
        "handed the true mass, the recursive estimator wins on mu alone; that is the "
        "comparison to keep, not the one where it is not")


# ───────────────────────── batched determinism and the estimator's cost ───────────────

def test_batch_is_deterministic_and_env_in_batch_equals_env_alone():
    for bits, key, a, z in ((BITS, "envs", "batch", "alone"),
                            (FRICTION_BITS, "env_checks", "hash_in_batch", "hash_alone")):
        assert bits["h_batch1"] == bits["h_batch2"], "two batch runs are not bit-identical"
        assert bits["batch_deterministic"] is True
        envs = bits[key]
        assert len(envs) == 8
        assert all(e[a] == e[z] for e in envs), "env in batch differs from env alone"
        assert all(e["match"] for e in envs)
        assert all(len(e[a]) == 64 for e in envs)
    # the friction run adds mu_hat, sigma and mu_eff to the hashed set at no cost
    assert FRICTION_BITS["env_identical_count"] == FRICTION_BITS["env_check_count"] == 8


def test_throughput_scales_and_the_memory_per_env_is_flat():
    by_b = {r["B"]: r for r in SCALE}
    assert by_b[4096]["env_steps_s"] / by_b[1]["env_steps_s"] > 500
    mem = [by_b[b]["mem_B_per_env"] for b in (64, 1024, 4096)]
    assert max(mem) - min(mem) < 1.0, mem
    assert by_b[4096]["res_max"] < 1e-6


def test_estimator_cost_is_below_a_tenth_of_a_percent():
    from motion_engine.ncp.online_friction import (ESTIMATOR_COST_FRACTION,
                                                   MU_ERROR_P95_AFTER_50_STEPS)
    assert ESTIMATOR_COST_FRACTION < 1e-3
    assert MU_ERROR_P95_AFTER_50_STEPS < 0.08


def test_the_recursive_estimator_converges_on_a_synthetic_slip_sequence():
    """The prior is a deliberate over-estimate (0.8); feeding it the true ratio must pull
    it down and shrink sigma. This is the recursion the device kernel implements."""
    from motion_engine.ncp.online_friction import RecursiveStudentT
    est = RecursiveStudentT()
    assert est.mu == pytest.approx(0.8)
    rng = np.random.default_rng(0)
    truth = 0.35
    for _ in range(200):
        est.update(truth + 0.01 * rng.standard_normal())
    assert abs(est.mu - truth) < 0.02, est.mu
    assert est.sigma < 0.05
    assert est.effective_mu(1.0) < est.mu, "the margin must be subtracted, not added"


def test_the_withdrawn_control_claim_stays_withdrawn():
    """The lane's headline was that the estimator beats known mu. Known mu with its own
    sigma margin gives 0 falls against 784 without, so the margin carried it."""
    from motion_engine.ncp import online_friction
    doc = online_friction.__doc__
    assert "FALLS" in doc and "784" in doc
    assert "margin" in doc
