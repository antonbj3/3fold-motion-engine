"""Body-space Newton solvers (ncp_newton) against the CPU reference (ncp_ref), same residual.

    OMP_NUM_THREADS=4 python -m pytest -s -q tests/test_ncp_newton.py

Locked here: (1) on the SPEC A.6 scenes plus tripod and random scenes the Newton pipeline reaches
natural residual < 1e-10 and the same body impulse J^T lam as ncp_ref ADMM (relative 1e-6; lam
itself is not unique on hyperstatic scenes); (2) the uniqueness report: K8 (four stuck corners per
interface) is force-ambiguous with a 48-dimensional fibre but its motion linearization is regular; the exact four-corner
slip counterexample (box 24 x 11 x 2, mu 1/2, implicit-function matrix of rank 11/12) is reported
MOTION_LINEARIZATION_SINGULAR; a singular nonlinear derivative alone is not a proof;
(3) SuperLU and PARDISO back ends agree; (4) the multiplicity probe: on random four-box stacks with
slipping four-corner interfaces a second start converges to a DIFFERENT verified solution (seeds 1-4,
|dv|/|v| 2e-4 .. 3e-2) and the local verdict is checked separately from the global numerical witness.
The comparison with Siconos 4.3.1 FC3D NSGS is a separate measurement and is not re-run here.
"""

import sys
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from motion_engine.ncp import ncp_newton as KN  # noqa: E402
from motion_engine.ncp import ncp_ref as R  # noqa: E402


@pytest.fixture(autouse=True)
def require_body_newton_work(monkeypatch):
    """An ADMM substitution must not satisfy tests intended for this new solver.

    Count actual body-space factors inside each entry-point call, independently
    of the returned metadata. Empty scenes legitimately need no factors.
    """
    count = [0]
    factor = KN.BodySolver.factor
    entry = KN.solve_ncp_newton

    def observed_factor(self, matrix):
        assert matrix.shape == (self.n, self.n)
        count[0] += 1
        return factor(self, matrix)

    def checked_entry(scene, *args, **kwargs):
        before = count[0]
        answer = entry(scene, *args, **kwargs)
        nc = scene.nc if isinstance(scene, KN.Problem) else scene.n_c
        if nc:
            assert count[0] > before, "entry point performed no body-space Newton factorization"
        return answer

    monkeypatch.setattr(KN.BodySolver, "factor", observed_factor)
    monkeypatch.setattr(KN, "solve_ncp_newton", checked_entry)


def _first_step(bodies, cfg, force_fn=None, t=0.0):
    vf = np.zeros(6 * len(bodies))
    for k, bd in enumerate(bodies):
        f = bd.mass * R.GRAVITY.copy()
        if force_fn is not None:
            f = f + force_fn(t, k, bd)[0]
        vf[6 * k:6 * k + 3] = bd.vel + cfg["dt"] * f / bd.mass
        vf[6 * k + 3:6 * k + 6] = bd.omega
    return R.build_scene(bodies, R.collect_contacts(bodies), cfg["mu"], cfg["dt"], v_free=vf)


def make_scene(name):
    if name == "i_pushed_cube":
        b, ff, cfg = R.scene_pushed_cube()
        return _first_step(b, cfg, ff, t=0.15)
    if name == "ii_sliding_box":
        b, _, cfg = R.scene_sliding_box()
        return _first_step(b, cfg)
    if name.startswith("iii_stack_"):
        b, _, cfg = R.scene_massratio_stack(ratio=float(name.split("_")[-1]))
        return _first_step(b, cfg)
    if name == "iv_K8":
        b, _, cfg = R.scene_tower(n=8)
        return _first_step(b, cfg)
    if name == "tripod":
        return R.scene_tripod()
    raise KeyError(name)


SCENES = ["i_pushed_cube", "ii_sliding_box", "iii_stack_1", "iii_stack_100", "iii_stack_1000", "iv_K8", "tripod"]


@pytest.mark.parametrize("name", SCENES)
def test_newton_matches_admm_body_impulse(name):
    sc = make_scene(name)
    G, b, mu = sc.delassus(), sc.b(), sc.mu
    lam_n, info = KN.solve_ncp_newton(sc, tol=1e-10)
    lam_a, hist, _ = R.solve_ncp_admm(G, b, mu, iters=20000, tol=1e-12)
    res_n = R.natural_residual(lam_n, G, b, mu)
    res_a = R.natural_residual(lam_a, G, b, mu)
    imp_n, imp_a = sc.J.T @ lam_n, sc.J.T @ lam_a
    rel = float(np.linalg.norm(imp_n - imp_a) / max(np.linalg.norm(imp_a), 1e-300))
    print(f"\n[{name}] newton res={res_n:.2e} iters={info['iters']} n_fact={info['n_fact']} "
          f"stage={info['solved_by']} verdict={info['uniqueness']['verdict']} | admm res={res_a:.2e} "
          f"iters={len(hist)} | |dJ^T lam|/|J^T lam|={rel:.2e}")
    assert info["converged"] and res_n < 1e-10
    assert rel < 1e-6


def test_random_stacks_honest_about_motion():
    """Four random boxes stacked with random velocities: slipping four-corner interfaces.
    Measured (seeds 0-4): Newton reaches < 1e-10 on all five; ADMM (20 000 iterations) only on
    seeds 0 and 3.  Seed 3: both converge yet the motions differ (|dv|/|v| ~ 7e-6): two valid
    answers.  The rule locked here: whenever two converged solvers disagree on the motion,
    the report must not call the answer unique; when it says motion-unique, they must agree."""
    seen_disagree = 0
    for seed in range(5):
        sc = R.build_scene(*_random_bodies(seed))
        G, b, mu = sc.delassus(), sc.b(), sc.mu
        lam_n, info = KN.solve_ncp_newton(sc, tol=1e-10, probe_multiplicity=True)
        lam_a, hist, _ = R.solve_ncp_admm(G, b, mu, iters=20000, tol=1e-12)
        rn, ra = R.natural_residual(lam_n, G, b, mu), R.natural_residual(lam_a, G, b, mu)
        mp = info["multiplicity"]
        if mp["verdict"] == "NUMERICAL_MULTIPLICITY_WITNESS":          # a numerical witness: both residuals checked
            assert R.natural_residual(mp["lam_second"], G, b, mu) < 1e-10
            assert info["uniqueness"]["verdict"] not in ("LOCALLY_REGULAR",)
        vn, va = sc.apply(lam_n), sc.apply(lam_a)
        dv = float(np.linalg.norm(vn - va) / np.linalg.norm(va))
        verdict = info["uniqueness"]["verdict"]
        print(f"\n[random {seed}] newton {rn:.1e} admm {ra:.1e} ({len(hist)} it) |dv|/|v| {dv:.1e} {verdict} "
              f"probe={mp['verdict']} {mp.get('dv_rel', float('nan')):.1e}")
        assert rn < 1e-10
        if ra < 1e-10 and dv > 1e-8:
            seen_disagree += 1
            assert verdict not in ("LOCALLY_REGULAR", "FORCE_AMBIGUOUS_MOTION_REGULAR")
        if verdict in ("LOCALLY_REGULAR", "FORCE_AMBIGUOUS_MOTION_REGULAR") and ra < 1e-10:
            assert dv < 1e-6
    assert seen_disagree >= 1          # the witness of seed 3 is part of the lock


def _random_bodies(seed):
    rng = np.random.default_rng(seed)
    bodies = []
    for k in range(4):
        h = rng.uniform(0.05, 0.15, 3)
        bodies.append(R.Box(half=h, mass=float(rng.uniform(0.5, 3.0)),
                            pos=np.array([0.0, 0.0, h[2] + sum(2 * bb.half[2] for bb in bodies)])))
    vf = np.zeros(6 * len(bodies))
    for k, bd in enumerate(bodies):
        vf[6 * k:6 * k + 3] = rng.normal(0, 0.2, 3) + 0.004 * R.GRAVITY
    return bodies, R.collect_contacts(bodies), 0.4, 0.004, vf


def test_report_K8_force_ambiguous_motion_regular():
    sc = make_scene("iv_K8")
    lam, info = KN.solve_ncp_newton(sc)
    rep = info["uniqueness"]
    print("\n[K8 report]", {k: rep[k] for k in ("verdict", "force_null_dim", "A_null_dim", "motion_frac_null")})
    assert rep["verdict"] == "FORCE_AMBIGUOUS_MOTION_REGULAR"
    assert rep["force_null_dim"] == 3 * 32 - 6 * 8
    assert rep["motion_frac_null"] < 1e-9


def _four_corner_slip_box():
    """Four slipping corners with yaw: exact rational data of the rank-11/12 counterexample."""
    corners = np.array([[12, 5.5, -1], [12, -5.5, -1], [-12, 5.5, -1], [-12, -5.5, -1]], float)
    Iinv = np.diag([float(Fr(12, 125)), float(Fr(3, 145)), float(Fr(12, 697))])
    vfree = np.array([float(Fr(1519, 130)), 0.0, -4.0, 0.0, float(Fr(-231, 9425)), float(Fr(61343, 45305))])
    J = np.zeros((12, 6))
    for c, r in enumerate(corners):
        n = np.array([0.0, 0.0, 1.0])
        t1, t2 = R._tangents(n)
        Bm = np.vstack([n, t1, t2])
        J[3 * c:3 * c + 3, :3] = Bm
        J[3 * c:3 * c + 3, 3:] = -Bm @ R._skew(r)
    Minv = np.zeros((6, 6))
    Minv[:3, :3] = np.eye(3)
    Minv[3:, 3:] = Iinv
    return R.Scene(J=J, Minv=Minv, v_free=vfree, mu=np.full(4, 0.5), body_pairs=np.array([[0, -1]] * 4), dt=0.01)


def test_report_four_corner_slip_singular_linearization():
    sc = _four_corner_slip_box()
    lam, info = KN.solve_ncp_newton(sc)
    rep = info["uniqueness"]
    v = sc.apply(lam)
    print("\n[four-corner slip box]", "v=", np.round(v, 6), {k: rep[k] for k in ("verdict", "A_null_dim", "motion_frac_null")})
    assert info["converged"]
    assert rep["verdict"] == "MOTION_LINEARIZATION_SINGULAR"
    assert rep["verdict"] not in ("LOCALLY_REGULAR",)


def test_superlu_and_pardiso_agree():
    pytest.importorskip("pypardiso")
    sc = make_scene("iii_stack_100")
    lam_api, info_api = KN.solve_ncp_newton(sc, report=False)
    assert info_api["converged"] and R.natural_residual(lam_api, sc.delassus(), sc.b(), sc.mu) < 1e-10
    pb = KN.Problem.from_engine_scene(sc)
    lam_p = KN.ssn(pb, np.zeros(3 * pb.nc), linsolver="pardiso", eps0=1e-2, dir_from_X="", retries=2)["lam"]
    lam_s = KN.ssn(pb, np.zeros(3 * pb.nc), linsolver="superlu", eps0=1e-2, dir_from_X="", retries=2)["lam"]
    assert pb.natres(lam_p) < 1e-10 and pb.natres(lam_s) < 1e-10
    assert np.linalg.norm(sc.J.T @ (lam_p - lam_s)) <= 1e-8 * np.linalg.norm(sc.J.T @ lam_p)


def test_pardiso_factor_state_is_owned():
    pytest.importorskip("pypardiso")
    import scipy.sparse as sp
    _, info = KN.solve_ncp_newton(make_scene("tripod"), report=False)
    assert info["converged"]
    a = KN.BodySolver(6, "pardiso", symmetric=True)
    b = KN.BodySolver(6, "pardiso", symmetric=True)
    try:
        a.factor(2 * sp.eye(6, format="csr"))
        b.factor(3 * sp.eye(6, format="csr"))
        assert a._ps is not b._ps
        np.testing.assert_allclose(a.solve(np.ones(6)), .5, atol=1e-14, rtol=0)
        a.free()
        np.testing.assert_allclose(b.solve(np.ones(6)), 1 / 3, atol=1e-14, rtol=0)
    finally:
        a.free()
        b.free()


def _single_contact_scene():
    # Physical contact basis (ex, ey, ez), lever arm ez, mass 100.
    J = np.array([[1, 0, 0, 0, 1, 0], [0, 1, 0, -1, 0, 0],
                  [0, 0, 1, 0, 0, 0]], dtype=float)
    Mi = np.diag([.01, .01, .01, 9.99, .99, 4 / 49])
    Mi[3, 4] = Mi[4, 3] = -3
    return R.Scene(J, Mi, np.array([1., 5., 0., 0., 0., 0.]),
                   np.array([.5]), np.array([[0, -1]]), .01)


def test_single_contact_two_rational_answers():
    sc = _single_contact_scene()
    G = sc.delassus()
    # Exact rationals give q=(1,5,0), W=[[1,3,0],[3,10,0],[0,0,1/100]].
    # Both r=0 (open) and r=(2,-1,0) (positive slip) obey Coulomb.
    a, b = np.zeros(3), np.array([2., -1., 0.])
    assert R.natural_residual(a, G, sc.b(), sc.mu) < 1e-12
    assert R.natural_residual(b, G, sc.b(), sc.mu) < 1e-12
    assert np.linalg.norm(sc.apply(a) - sc.apply(b)) > 1
    lam, info = KN.solve_ncp_newton(sc)
    assert info["converged"] and R.natural_residual(lam, G, sc.b(), sc.mu) < 1e-10
    assert info["uniqueness"]["certificate"] is False


def test_report_and_probe_validate_first_answer():
    sc = _single_contact_scene()
    sc.v_free = np.array([-1., 0., 0., 0., 0., 0.])
    lam, info = KN.solve_ncp_newton(sc)
    assert info["converged"]
    pb = KN.Problem.from_engine_scene(sc)
    assert KN.uniqueness_report(pb, lam, dense_max=0)["verdict"] == "RANK_UNCHECKED_INCONCLUSIVE"
    invalid = np.zeros(3)
    assert pb.natres(invalid) > .01
    assert KN.uniqueness_report(pb, invalid)["verdict"] == "NOT_CONVERGED_INCONCLUSIVE"
    probe = KN.multiplicity_probe(pb, invalid, "ssn_cold")
    assert probe["verdict"] == "FIRST_START_NOT_CONVERGED"
    assert probe["certificate"] is False
    assert KN.uniqueness_report(pb, np.full(3, np.nan))["verdict"] == "INVALID_SOLUTION"
    for tol in (0., -1., np.inf, np.nan):
        with pytest.raises(ValueError):
            KN.solve_ncp_newton(sc, tol=tol)


def test_scene_contract_and_empty_contact_step():
    sc = make_scene("tripod")
    _, info = KN.solve_ncp_newton(sc)
    assert info["converged"]
    empty = R.Scene(np.zeros((0, 6)), np.eye(6), np.ones(6), np.empty(0),
                    np.empty((0, 2), dtype=int), .01)
    lam, info = KN.solve_ncp_newton(empty, probe_multiplicity=True)
    assert lam.size == 0 and info["converged"] and info["n_fact"] == 0
    assert info["uniqueness"]["verdict"] == "EMPTY_CONTACT_SET"
    assert info["multiplicity"]["verdict"] == "SECOND_START_AGREES"
    coupled = R.Scene(np.hstack([np.eye(3), np.zeros((3, 9))]),
                      np.eye(12) + .1 * np.ones((12, 12)), np.zeros(12),
                      np.array([.5]), np.array([[0, 1]]), .01)
    with pytest.raises(ValueError, match="independent"):
        KN.Problem.from_engine_scene(coupled)
    sc.mu[:] = 0.
    with pytest.raises(ValueError, match="positive friction"):
        KN.Problem.from_engine_scene(sc)


def test_stationary_probe_rejects_relative_roundoff():
    _, info = KN.solve_ncp_newton(make_scene("iv_K8"), report=False, probe_multiplicity=True)
    assert info["converged"]
    probe = info["multiplicity"]
    assert probe["converged"]
    assert probe["dv_max_translation"] < 1e-6
    assert probe["dv_max_angular"] < 1e-6
    assert probe["verdict"] == "SECOND_START_AGREES"
    assert probe["agreement_scope"] == "declared_motion_budget"
    assert probe["certificate"] is False
