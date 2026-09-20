"""The batched GPU adjoint `grad_affine`, and the k_setsig guard that makes it exact.

    OMP_NUM_THREADS=4 python -m pytest -s -q tests/test_ncp_batch_adjoint.py

The all-slip row is the test that found the defect: every contact slipping, several envs
in one batch, n_c = 1, 2 and 4. It FAILS on the pre-fix kernel (max abs error 0.14 / 0.79)
and passes with `if j >= 0 and j < nslip[e]`. The nine-scene table it belongs to costs a
GPU run per scene, so it is committed as `data/ncp/adjoint_fix_table.csv` and checked here
against the bounds it was measured against, unfixed column included.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ALL_SLIP_TOL = 1e-9


@pytest.fixture(scope="module")
def gpu():
    wp = pytest.importorskip("warp")
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("the batched adjoint needs a CUDA device")
    wp.set_module_options({"fuse_fp": False, "fast_math": False})
    from motion_engine.ncp import ncp_batch_gpu
    return wp, ncp_batch_gpu


def _all_slip_scene(n_c):
    """SPD Delassus and a rhs that puts EVERY contact in slip. Contact c couples body c to
    body c+1 (the last to ground), so G is block-tridiagonal and non-singular: the contacts
    are coupled, not independent."""
    n = 3 * n_c
    J = np.zeros((n, n))
    for c in range(n_c):
        J[3 * c:3 * c + 3, 3 * c:3 * c + 3] = np.eye(3)
        if c + 1 < n_c:
            J[3 * c:3 * c + 3, 3 * (c + 1):3 * (c + 1) + 3] = -np.eye(3)
    m = 1.0 + 0.5 * np.arange(n_c)
    Minv = np.diag(np.repeat(1.0 / m, 3))
    G = J @ Minv @ J.T
    mu = np.full(n_c, 0.3)
    b = np.zeros(n)
    for c in range(n_c):
        b[3 * c + 0] = -1.0 - 0.1 * c
        b[3 * c + 1] = 0.9 + 0.2 * c
        b[3 * c + 2] = -0.7 + 0.1 * c
    return G, b, mu


def _batched_jacobian(wp, H25, G, b, mu, cg_iters=60):
    """dlam/db out of the BATCHED grad_affine: one env per b-direction, replayed twice."""
    from motion_engine.ncp import ncp_ref as R
    n = len(b); n_c = len(mu); B = n
    s = H25.BatchProxADMMGPU(B, n, n_c)
    s.upload(np.tile(np.eye(n), (B, 1, 1)), np.tile(G, (B, 1, 1)),
             np.tile(b, (B, 1)), np.tile(mu, B), 0.01)
    # the residual floor for these scenes is 3.6e-15 and PGS is there by sweep 133;
    # the lane ran a 200000 cap that only spins after that, so the budget is 2000
    # and the residual it reaches is the lane's: 2.498e-16 / 4.441e-16 / 3.553e-15.
    lam = R.solve_ncp_pgs(G, b, mu, 2000, 1e-15)[0]
    labs = R.classify(lam, G, b, mu)
    wp.copy(s.zv, wp.array(np.tile(lam, (B, 1)).reshape(-1, 3), dtype=H25.vec3d, device=s.d))
    wp.copy(s.u, wp.array(np.tile(G @ lam + b, (B, 1)).reshape(-1, 3),
                          dtype=H25.vec3d, device=s.d))
    s.build_active(); wp.synchronize()
    wp.copy(s.dbv, wp.array(np.eye(n).reshape(-1, 3), dtype=H25.vec3d, device=s.d))
    n_calls = int(s.nslip.numpy().max()) + 2
    wp.load_module(device=s.d)
    with wp.ScopedCapture(device=s.d) as cap:
        s.grad_affine(n_calls, cg_iters)
    outs = []
    for _ in range(2):
        wp.capture_launch(cap.graph); wp.synchronize()
        outs.append(s.dlam.numpy().reshape(B, n).T.copy())
    return outs, lam, labs, s.nslip.numpy().copy(), n_calls


@pytest.mark.parametrize("n_c", [1, 2, 4])
def test_all_slip_batched_adjoint_matches_the_cpu_ift(gpu, n_c):
    wp, H25 = gpu
    from motion_engine.ncp import ncp_ref as R
    G, b, mu = _all_slip_scene(n_c)
    outs, lam, labs, nslip, n_calls = _batched_jacobian(wp, H25, G, b, mu)
    res = float(R.natural_residual(lam, G, b, mu))
    ref = R.sensitivity(lam, G, b, mu, labs)[1]
    err = float(np.abs(outs[0] - ref).max())
    print(f"\n  all_slip n_c={n_c}: residual={res:.3e} n_slip={nslip.tolist()} "
          f"n_calls={n_calls} max abs err={err:.6e} bit-identical replays="
          f"{outs[0].tobytes() == outs[1].tobytes()}")
    assert all(l == "slip" for l in labs), "the scene must put every contact in slip"
    assert (nslip == n_c).all(), "every env must report n_slip == n_c"
    assert outs[0].tobytes() == outs[1].tobytes(), "two graph replays differ"
    assert res < 1e-14, f"forward did not reach its residual floor: {res:.3e}"
    assert err <= ALL_SLIP_TOL, (
        f"batched grad_affine vs CPU IFT {err:.3e} > {ALL_SLIP_TOL:g}; the pre-fix kernel "
        f"gives 0.14 to 0.79 here")


# ---------------------------------------------------------------------------
# the nine-scene table, locked as data
# ---------------------------------------------------------------------------

def _table():
    with (ROOT / "data/ncp/adjoint_fix_table.csv").open() as fh:
        return list(csv.DictReader(fh))


TABLE = _table()
SCENES = [r["scene"] for r in TABLE]


@pytest.mark.parametrize("row", TABLE, ids=SCENES)
def test_fixed_gradient_is_exact_on_every_scene(row):
    db = float(row["fixed_max_rel_dlam_db"])
    dmu = float(row["fixed_max_rel_dlam_dmu"])
    assert db <= 1.8e-9, f"{row['scene']}: dlam/db {db:.3e}"
    # the measured maximum over the nine scenes is 2.4006e-12 (all_slip_8)
    assert dmu <= 2.5e-12, f"{row['scene']}: dlam/dmu {dmu:.3e}"
    assert row["bit_identical_two_runs_fixed"] == "True"


@pytest.mark.parametrize("row", TABLE, ids=SCENES)
def test_the_defect_is_visible_in_the_unfixed_column(row):
    """Six of nine scenes are wrong without the guard; three are not. Both halves are the
    measurement, so both are asserted."""
    unfixed = max(float(row["unfixed_max_rel_dlam_db"]), float(row["unfixed_max_rel_dlam_dmu"]))
    fixed = max(float(row["fixed_max_rel_dlam_db"]), float(row["fixed_max_rel_dlam_dmu"]))
    slips_last_contact = row["scene"] not in ("column", "tower", "partial_4_last")
    if slips_last_contact:
        assert unfixed > 1e-2, f"{row['scene']} should be broken without the guard"
        assert unfixed / max(fixed, 1e-300) > 1e6
    else:
        assert unfixed == pytest.approx(fixed, rel=0, abs=0), (
            f"{row['scene']} must be bit-identical with and without the guard")


def test_scope_is_contact_c_minus_one_slipping_not_n_slip_equals_n_c():
    """`partial_4` and `partial_4_last` have the same labels (slip=3, stick=1) and differ
    only in WHERE the stuck contact sits. The first breaks, the second does not."""
    by = {r["scene"]: r for r in TABLE}
    a, z = by["partial_4"], by["partial_4_last"]
    assert a["labels"] == z["labels"] == "slip=3;stick=1"
    assert float(a["unfixed_max_rel_dlam_db"]) > 1e-2
    assert float(z["unfixed_max_rel_dlam_db"]) < 1e-13
