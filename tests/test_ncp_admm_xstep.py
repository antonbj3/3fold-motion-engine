"""Regression: the opt-in ADMM x-step (xstep="triangular") uses BLAS triangular substitution
on a Fortran-ordered Cholesky factor and drives the same iteration as the numpy-only
Cholesky/LU x-step (same iteration count, lam to 1e-12). The default x-step keeps the
legacy arithmetic byte for byte."""
import numpy as np
import pytest

from motion_engine.ncp import ncp_ref as R


def _numpy_only_admm(G, b, mu, iters, tol):
    # the pre-change x-step, verbatim: np.linalg.solve on the two Cholesky factors
    import builtins
    real_import = builtins.__import__
    blocked = []

    def no_scipy(name, *a, **k):
        if name.startswith("scipy"):
            blocked.append(name)
            raise ImportError(name)
        return real_import(name, *a, **k)
    builtins.__import__ = no_scipy
    try:
        out = R.solve_ncp_admm(G, b, mu, iters, tol, xstep="triangular")
    finally:
        builtins.__import__ = real_import
    assert blocked, "the x-step never tried scipy, so the numpy fallback was not exercised"
    return out


@pytest.mark.parametrize("n", [3, 8])
def test_admm_xstep_matches_numpy_path(n, monkeypatch):
    bodies, _, cfg = R.scene_tower(n=n)
    ct = R.collect_contacts(bodies)
    v_free = np.zeros(6 * n)
    v_free[2::6] = cfg["dt"] * -9.81
    sc = R.build_scene(bodies, ct, cfg["mu"], cfg["dt"], v_free=v_free)
    G, b = sc.delassus(), sc.b()
    import scipy.linalg.blas as blas
    fortran = []
    actual = blas.dtrsv
    def observed(a, x, *args, **kwargs):
        fortran.append(bool(a.flags.f_contiguous))
        return actual(a, x, *args, **kwargs)
    monkeypatch.setattr(blas, "dtrsv", observed)
    def no_full_lu(*args, **kwargs):
        raise AssertionError("accelerated x-step called a full LU solve")
    with monkeypatch.context() as ctx:
        ctx.setattr(np.linalg, "solve", no_full_lu)
        lam_a, hist_a, lab_a = R.solve_ncp_admm(G, b, sc.mu, 5000, 1e-10, xstep="triangular")
    assert fortran, "the accelerated path must execute triangular substitution"
    assert all(fortran), "the factor must be Fortran-ordered so BLAS does not copy it per solve"
    lam_b, hist_b, lab_b = _numpy_only_admm(G, b, sc.mu, 5000, 1e-10)
    assert hist_a[-1] < 1e-10 and hist_b[-1] < 1e-10
    assert len(hist_a) == len(hist_b)
    assert np.max(np.abs(lam_a - lam_b)) <= 1e-12 * max(1.0, np.max(np.abs(lam_b)))
    assert np.all(lab_a == lab_b)


@pytest.mark.parametrize("seed", [3, 7])
def test_default_admm_preserves_legacy_bytes(seed):
    sc = R.scene_random(n_c=4, n_b=2, seed=seed)
    G, b = sc.delassus(), sc.b()
    default = R.solve_ncp_admm(G, b, sc.mu, 5000, 1e-14)
    legacy = _numpy_only_admm(G, b, sc.mu, 5000, 1e-14)
    for x, y in zip(default, legacy):
        assert x.dtype == y.dtype and x.shape == y.shape
        assert x.tobytes() == y.tobytes()
