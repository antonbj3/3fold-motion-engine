"""GPU exact-cone NCP: the fast gates. Requires CUDA + Warp.

    OMP_NUM_THREADS=4 python -m pytest -s -q tests/test_ncp_gpu_cone.py

The GPU is shared, so this file keeps only the gates that finish in seconds. The
multi-step scene numbers that need minutes -- the 1000:1 mass-ratio column, the K8
tower, the sliding-distance dt sweep, sleeping at N = 10000, the substep ladder and
the soft-step penetration law -- are committed as measured values in
`data/ncp/gpu_scene_reference.json` and checked there against their stated bounds,
including the two measured negatives they contain (the 1000:1 column at 200
iterations and the K8 tower that never reaches v_sleep = 1e-3).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from motion_engine.ncp import ncp_gpu as G  # noqa: E402


def _ref():
    from motion_engine.ncp import ncp_ref
    return ncp_ref


@pytest.fixture(scope="module", autouse=True)
def _cuda():
    import warp as wp
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("test_ncp_gpu requires CUDA")


# ───────────────────────── contract ─────────────────────────


def test_dense_scene_matches_block_scene():
    """The dense-J `Scene` of the A/B contract slices into the same blocks the GPU solver consumes."""
    bs = G.random_wellposed_scene(n_c=6, n_b=6, seed=3)
    n_c, n_b = bs.n_c, bs.n_b
    J = np.zeros((3 * n_c, 6 * n_b))
    for c in range(n_c):
        a, b = bs.body_pairs[c]
        J[3 * c:3 * c + 3, 6 * a:6 * a + 6] += bs.Ja[c]
        J[3 * c:3 * c + 3, 6 * b:6 * b + 6] += bs.Jb[c]
    Mi = np.zeros((6 * n_b, 6 * n_b))
    for i in range(n_b):
        Mi[6 * i:6 * i + 6, 6 * i:6 * i + 6] = bs.Minv[i]
    sc = G.Scene(J=J, Minv=Mi, v_free=bs.v_free.reshape(-1), mu=bs.mu,
                 body_pairs=bs.body_pairs, dt=bs.dt, bias=bs.bias.reshape(-1), feature=bs.feature)
    bs2 = sc.blocks()
    assert np.allclose(bs2.Ja, bs.Ja) and np.allclose(bs2.Jb, bs.Jb)
    assert np.allclose(bs2.Minv, bs.Minv) and np.allclose(bs2.v_free, bs.v_free)
    r1 = G.solve_scene_gpu(bs, iters=1500)
    r2 = G.solve_scene_gpu(bs2, iters=1500)
    assert r1.lam.tobytes() == r2.lam.tobytes()


# ───────────────────────── (a) bit identity ─────────────────────────


@pytest.mark.parametrize("scene,dtype", [("wellposed", "float32"), ("synthetic256", "float32"),
                                         ("wellposed", "float64")])
def test_bit_identical_two_runs(scene, dtype):
    """SPEC B test (a): two runs on the same input -> identical BYTES of lambda and of v."""
    bs = (G.random_wellposed_scene() if scene == "wellposed"
          else G.synthetic_scene(256, n_b=64, seed=7))
    out = []
    for _ in range(2):
        s = G.NCPSolverGPU(bs.n_b, bs.n_c, dtype=dtype)
        s.solve(bs, iters=300)
        out.append(s.raw_bytes())
    assert out[0][0] == out[1][0], "lambda bytes differ between two runs"
    assert out[0][1] == out[1][1], "velocity bytes differ between two runs"


def test_bit_identical_under_input_permutation():
    """The colouring is a pure function of the SORTED (body_a, body_b, feature) keys, so shuffling
    the order in which contacts are handed in must not change a single bit of the answer."""
    bs = G.synthetic_scene(256, n_b=64, seed=11)
    rng = np.random.default_rng(5)
    p = rng.permutation(bs.n_c)
    bp = G.BlockScene(Ja=bs.Ja[p], Jb=bs.Jb[p], Minv=bs.Minv, v_free=bs.v_free, mu=bs.mu[p],
                      body_pairs=bs.body_pairs[p], dt=bs.dt, bias=bs.bias[p], feature=bs.feature[p])
    s1 = G.NCPSolverGPU(bs.n_b, bs.n_c); r1 = s1.solve(bs, iters=300)
    s2 = G.NCPSolverGPU(bs.n_b, bs.n_c); r2 = s2.solve(bp, iters=300)
    assert r1.lam[p].tobytes() == r2.lam.tobytes()
    assert r1.v.tobytes() == r2.v.tobytes()


def test_coloring_is_a_proper_coloring():
    """No two contacts of a colour share a dynamic body -> Jacobi inside a colour is exact GS."""
    bs = G.synthetic_scene(1024, n_b=128, seed=2)
    s = G.NCPSolverGPU(bs.n_b, bs.n_c)
    s.solve(bs, iters=10, measure_residual=False)
    col, order, cstart, ccount = s.coloring()
    for ci in range(s.n_colors):
        idx = order[cstart[ci]:cstart[ci] + ccount[ci]]
        bodies = np.concatenate([bs.body_pairs[idx, 0], bs.body_pairs[idx, 1]])
        bodies = bodies[bodies >= 0]
        assert len(np.unique(bodies)) == len(bodies), f"colour {ci} reuses a body"
    assert sum(ccount) == bs.n_c


# ───────────────────────── (b) CPU reference ─────────────────────────


def test_matches_cpu_reference_lambda():
    """SPEC B test (b): well-posed scene (3 n_c <= n_v, J full rank -> G SPD -> unique lambda);
    both solvers driven to convergence, compared in lambda."""
    ref = _ref()
    bs = G.random_wellposed_scene(n_c=8, n_b=8, seed=1)
    Gm, b = bs.delassus(), bs.rhs()
    lam_ref = ref.solve_ncp_pgs(Gm, b, bs.mu, iters=5000, tol=1e-14)[0].reshape(-1, 3)
    r = G.solve_scene_gpu(bs, iters=5000, dtype="float32")
    r64 = G.solve_scene_gpu(bs, iters=5000, dtype="float64")
    d32 = float(np.max(np.abs(r.lam - lam_ref)))
    d64 = float(np.max(np.abs(r64.lam - lam_ref)))
    print(f"\n  |lambda_gpu - lambda_ref|_inf: float32 {d32:.3e}   float64 {d64:.3e}"
          f"   (|lambda|_inf = {np.max(np.abs(lam_ref)):.3e})")
    assert d64 < 1e-6, f"float64 GPU differs from the CPU reference by {d64:.3e}"
    assert d32 < 1e-5, f"float32 GPU differs from the CPU reference by {d32:.3e}"


def test_matches_cpu_reference_wrench_on_pushed_cube():
    """Scene (i) is statically indeterminate (3 n_c = 12 > n_v = 6), so lambda is not unique but the
    net wrench J^T lambda and the contact velocity u = G lambda + b are. Compare those."""
    ref = _ref()
    w = G.BoxWorld(dt=0.004, mu=0.5, iters=1)
    w.add([0.1, 0.1, 0.1], 15.0, [0.0, 0.0, 0.1])
    bs = w.build_scene(f_ext=lambda i: np.array([100.0, 0.0, 0.0]))
    Gm, b = bs.delassus(), bs.rhs()
    lam_ref = ref.solve_ncp_pgs(Gm, b, bs.mu, iters=5000, tol=1e-14)[0]
    r = G.solve_scene_gpu(bs, iters=5000, dtype="float64")
    lam_gpu = r.lam.reshape(-1)
    u_ref, u_gpu = Gm @ lam_ref + b, Gm @ lam_gpu + b
    Jt = np.zeros((6, 3 * bs.n_c))
    for c in range(bs.n_c):
        Jt[:, 3 * c:3 * c + 3] = bs.Ja[c].T
    du = float(np.max(np.abs(u_ref - u_gpu)))
    dw = float(np.max(np.abs(Jt @ (lam_ref - lam_gpu))))
    dl = float(np.max(np.abs(lam_ref - lam_gpu)))
    wmag = float(np.max(np.abs(Jt @ lam_ref)))
    print(f"\n  pushed cube (indeterminate): |du|_inf {du:.3e}  |d wrench|_inf {dw:.3e} "
          f"(|wrench|_inf {wmag:.3e})  |d lambda|_inf {dl:.3e}")
    # lambda itself is NOT unique here (4 coplanar points, 12 unknowns, 6 DOF): the two solvers land
    # on different members of the solution set, 1e-2 apart in lambda. What is compared is u and the
    # net wrench. Their residual floor is the int64 fixed-point drift, see test_fixed_point_drift.
    assert du < 1e-5 and dw < 1e-4


def test_fixed_point_drift_does_not_compound():
    """The int64 accumulator carries the per-body TOTAL M^-1 J^T lambda and each contact update
    REPLACES its own quantized contribution (`f_swap`), so the accumulator is a pure function of the
    current lambda and the fixed-point rounding does not compound with the iteration count. Gate: the
    drift is bounded by (contacts on a body) x 1 quantum and is FLAT in the iteration count."""
    w = G.BoxWorld(dt=0.004, mu=0.5, iters=1)
    w.add([0.1, 0.1, 0.1], 15.0, [0.0, 0.0, 0.1])
    bs = w.build_scene(f_ext=lambda i: np.array([100.0, 0.0, 0.0]))
    out = []
    for it in (500, 5000, 20000, 80000):
        r = G.solve_scene_gpu(bs, iters=it, dtype="float64")
        v_exact = bs.v_free.copy()
        for c in range(bs.n_c):
            a = bs.body_pairs[c, 0]
            v_exact[a] += bs.Minv[a] @ (bs.Ja[c].T @ r.lam[c])
        out.append((it, float(np.max(np.abs(r.v - v_exact))), r.residual))
    print("\n  int64 fixed-point drift |v_gpu - (v_free + Minv J^T lambda)|_inf:")
    for it, d, res in out:
        print(f"    iters={it:6d}  drift={d:.3e}  solver residual={res:.2e}")
    d = np.array([o[1] for o in out])
    quantum = 1.0 / 1.0e10
    assert d.max() < bs.n_c * quantum, f"drift {d.max():.3e} above {bs.n_c} quanta"
    assert d[-1] <= 4.0 * max(d[0], quantum), (
        f"drift grows with the iteration count: {d[0]:.3e} at 500 -> {d[-1]:.3e} at 80000")


def test_velocity_is_a_pure_function_of_lambda():
    """Consequence of the accumulation scheme, and the sharpest test of it: because the accumulator
    is rebuilt exactly from the absolute lambda, stopping the solver and restarting it warm is
    BIT-identical to running straight through. With the old delta accumulation the restart threw the
    accumulated rounding away and the two runs diverged."""
    bs = G.synthetic_scene(256, n_b=64, seed=7)
    s1 = G.NCPSolverGPU(bs.n_b, bs.n_c, dtype="float64")
    ra = s1.solve(bs, iters=1200)
    s2 = G.NCPSolverGPU(bs.n_b, bs.n_c, dtype="float64")
    rb = s2.solve(bs, iters=400)
    for _ in range(2):
        rb = s2.solve(bs, iters=400, warm=rb.lam)
    assert ra.lam.tobytes() == rb.lam.tobytes(), "1x1200 != 3x400 warm-restarted in lambda"
    assert ra.v.tobytes() == rb.v.tobytes(), "1x1200 != 3x400 warm-restarted in v"


def test_float32_vs_float64_gap():
    """SPEC B test (b) rider: report the float32-vs-float64 gap; a float64 Warp build is available."""
    bs = G.random_wellposed_scene(n_c=8, n_b=8, seed=1)
    a = G.solve_scene_gpu(bs, iters=5000, dtype="float32")
    b = G.solve_scene_gpu(bs, iters=5000, dtype="float64")
    gap = float(np.max(np.abs(a.lam - b.lam)))
    print(f"\n  |lambda_f32 - lambda_f64|_inf = {gap:.3e}   residual f32 {a.residual:.3e} "
          f"f64 {b.residual:.3e}")
    assert gap < 1e-5


# ───────────────────────── (c) scenes ─────────────────────────


def test_scene_pushed_cube():
    """(i) ODYNSim pushed cube."""
    w, traj, pen, res = G.scene_pushed_cube(steps=250, iters=400)
    print(f"\n  (i) end xy=({traj[-1,0]:.6f},{traj[-1,1]:.6f}) max_pen={pen.max()*1e3:.4e} mm "
          f"max_res={res.max():.3e} contacts={w.n_contacts} colours={w.n_colors}")
    assert w.n_contacts == 4 and w.n_colors == 4
    assert pen.max() < 1e-4, f"penetration {pen.max()*1e3:.3e} mm"
    assert res.max() < 1e-4
    assert abs(traj[-1, 2] - 0.1) < 1e-4


def test_scene_sliding_box_gliding_number():
    """(ii) NCP gap = 0; the convex relaxation (no de Saxce) hovers -- record the number."""
    _, gap_ncp, _ = G.scene_sliding_box(steps=200, iters=400, desaxce=True)
    _, gap_cvx, _ = G.scene_sliding_box(steps=200, iters=400, desaxce=False)
    print(f"\n  (ii) max normal gap: NCP {gap_ncp.max():.6e} m   convex relaxation "
          f"{gap_cvx.max():.6e} m   (first step: {gap_ncp[0]:.3e} vs {gap_cvx[0]:.3e})")
    assert abs(gap_ncp.max()) < 1e-8, "NCP hovers"
    assert gap_cvx.max() > 1e-4, "convex relaxation did not show the gliding artifact"






# ───────────────────────── (d) timing + colour cache ─────────────────────────


def test_color_cache_hit_and_invalidation():
    bs = G.synthetic_scene(256, n_b=64, seed=4)
    s = G.NCPSolverGPU(bs.n_b, bs.n_c)
    r0 = s.solve(bs, iters=50)
    r1 = s.solve(bs, iters=50)
    assert r0.cache_hit is False and r1.cache_hit is True
    assert r0.lam.tobytes() == r1.lam.tobytes(), "cache hit changed the answer"
    bs2 = G.BlockScene(Ja=bs.Ja, Jb=bs.Jb, Minv=bs.Minv, v_free=bs.v_free, mu=bs.mu,
                       body_pairs=np.roll(bs.body_pairs, 1, axis=0), dt=bs.dt, bias=bs.bias,
                       feature=bs.feature)
    r2 = s.solve(bs2, iters=50)
    assert r2.cache_hit is False, "changed body pairs did not invalidate the colour cache"




# ───────────────────────── islands + sleeping ─────────────────────────


def test_islands_static_ground_is_not_a_union():
    """Two boxes resting on the same ground plane are TWO islands: a static endpoint (-1) never
    unions. A box-box contact merges them into one."""
    pairs = np.array([[0, -1], [1, -1], [2, -1]], np.int64)
    lab = G.connected_islands(3, pairs)
    assert len(np.unique(lab)) == 3
    lab2 = G.connected_islands(3, np.vstack([pairs, [[1, 2]]]))
    assert len(np.unique(lab2)) == 2 and lab2[1] == lab2[2] and lab2[0] != lab2[1]
    # a chain of 8 (a tower) is one island whatever order the pairs arrive in
    chain = np.array([[i, i + 1] for i in range(7)], np.int64)
    rng = np.random.default_rng(3)
    a = G.connected_islands(8, chain)
    b = G.connected_islands(8, chain[rng.permutation(7)])
    assert len(np.unique(a)) == 1 and np.array_equal(a, b)




def test_sleeping_wakes_on_a_new_contact_pair():
    """A sleeping island wakes when a contact pair appears that was not in the previous step's pair
    set -- here a second box dropped on top of a box that has already gone to sleep."""
    cfg = G.SleepConfig(enabled=True, v_sleep=1e-2, w_sleep=2e-2, k_steps=4)
    w = G.BoxWorld(dt=1.0 / 240.0, mu=0.5, iters=200, sleep=cfg)
    w.add([0.1, 0.1, 0.1], 1.0, [0.0, 0.0, 0.1])
    w.add([0.1, 0.1, 0.1], 1.0, [0.0, 0.0, 0.9])
    slept, woke = -1, -1
    for k in range(140):
        w.step()
        a = w.sleeper.asleep
        if slept < 0 and a[0]:
            slept = k
        if slept >= 0 and woke < 0 and not a[0]:
            woke = k
    print(f"\n  box 0 asleep from step {slept}, woken by the landing box at step {woke}")
    assert slept >= 0, "the resting box never slept"
    assert woke > slept, "the sleeping box was not woken by the new contact"
    assert abs(w.bodies[0].x[2] - 0.1) < 5e-3 and abs(w.bodies[1].x[2] - 0.3) < 5e-3


def test_sleeping_is_deterministic():
    """Sleep decisions are a function of the (bit-identical) GPU velocities, the pair set and the
    order-independent island labels: two runs agree bit for bit, sleep mask included."""
    out = []
    for _ in range(2):
        w = G.LatticeWorld(n=2000, iters=120,
                           sleep=G.SleepConfig(enabled=True, v_sleep=1e-3, w_sleep=1e-3, k_steps=4))
        f = G.lattice_pushed_forces(w)
        for _ in range(14):
            w.step(f_ext=f)
        out.append((w.x.tobytes(), w.sleeper.asleep.tobytes(), int(w.n_asleep)))
    assert out[0][0] == out[1][0], "positions differ between two sleeping runs"
    assert out[0][1] == out[1][1], "sleep masks differ between two sleeping runs"
    assert out[0][2] > 0


def test_sleeping_lattice_n10000():
    """N = 10000 two-layer lattice (motion-engine N10000 fixture geometry), 90 % of the bodies at
    rest and 10 % pushed. Sleeping must not move the sleeping bodies."""
    res = {}
    for on in (False, True):
        cfg = (G.SleepConfig(enabled=True, v_sleep=1e-3, w_sleep=1e-3, k_steps=8) if on else None)
        w = G.LatticeWorld(n=10000, iters=120, sleep=cfg)
        f = G.lattice_pushed_forces(w)
        for _ in range(12):
            w.step(f_ext=f)
        ts = []
        for _ in range(20):
            t0 = time.perf_counter()
            w.step(f_ext=f)
            ts.append((time.perf_counter() - t0) * 1e3)
        res[on] = (w, float(np.median(ts)))
    won, woff = res[True][0], res[False][0]
    asleep = won.sleeper.asleep
    dx = np.abs(won.x - woff.x)
    tol = 1e-3 * won.dt * 32
    print(f"\n  N10000: {res[False][1]:.2f} ms/step off vs {res[True][1]:.2f} ms/step on "
          f"({res[False][1] / res[True][1]:.2f}x), asleep {int(asleep.sum())}/10000, "
          f"n_c {woff.n_contacts} -> {won.n_contacts}, max|dx| sleeping bodies {dx[asleep].max():.3e} m "
          f"(tolerance {tol:.3e} m)")
    assert 0.85 <= asleep.mean() <= 0.95, f"asleep fraction {asleep.mean():.3f}"
    assert won.n_contacts < 0.2 * woff.n_contacts
    assert dx[asleep].max() < tol


# ───────────────────────── mass ratio diagnostics ─────────────────────────






# ───────────────────────── (ii) rider: sliding distance ─────────────────────────





# ───────────────────────── TGS soft-step (B3) ─────────────────────────


def _soft_module():
    """`motion_engine.soft_step`, the CPU primitive `ncp_gpu` re-derives. Imported read-only for the
    value-for-value comparison; skipped (not failed) when the package is not on the path."""
    import importlib
    try:
        return importlib.import_module("motion_engine.soft_step")
    except Exception as e:                                   # pragma: no cover - env dependent
        pytest.skip(f"motion_engine.soft_step not importable: {e!r}")


def test_soft_matches_motion_engine_soft_step():
    """The re-derivation in `ncp_gpu` reproduces `motion_engine/soft_step.py` value for value:
    make_soft over a (hertz, zeta, h) grid, contact_hertz, and all three branches of
    bias_velocity (speculative / biased+clamped / relax)."""
    M = _soft_module()
    worst_s, worst_b = 0.0, 0.0
    for hz in (0.0, 5.0, 15.0, 30.0, 60.0, 120.0):
        for zeta in (1.0, 5.0, 10.0):
            for h in (1 / 60, 1 / 240, 1 / 960):
                a, b = G.make_soft(hz, zeta, h), M.make_soft(hz, zeta, h)
                worst_s = max(worst_s, abs(a.bias_rate - b.bias_rate),
                              abs(a.mass_scale - b.mass_scale),
                              abs(a.impulse_scale - b.impulse_scale))
                assert abs(a.mass_scale + a.impulse_scale - (1.0 if hz > 0 else 0.0)) < 1e-12
    for inv_h in (60.0, 240.0, 960.0):
        for wh in (15.0, 30.0, 240.0):
            assert G.contact_hertz(inv_h, wh) == M.contact_hertz(inv_h, wh)
        s_g = G.make_soft(G.contact_hertz(inv_h), G.CONTACT_DAMPING_RATIO, 1 / inv_h)
        s_m = M.make_soft(M.contact_hertz(inv_h), M.CONTACT_DAMPING_RATIO, 1 / inv_h)
        for sep in (1e-2, 1e-4, 0.0, -1e-5, -1e-3, -0.35, -1.0, -10.0):
            for ub in (True, False):
                g = G.bias_velocity(sep, s_g, ub, inv_h)
                m = M.bias_velocity(sep, s_m, ub, inv_h)
                worst_b = max(worst_b, max(abs(x - y) for x, y in zip(g, m)))
    print(f"\n  make_soft max |ncp_gpu - motion_engine| over the grid = {worst_s:.3e}; "
          f"bias_velocity (3 branches x 8 separations x 3 inv_h) = {worst_b:.3e}")
    assert worst_s == 0.0 and worst_b == 0.0
    assert G.CONTACT_HERTZ == M.CONTACT_HERTZ and G.CONTACT_DAMPING_RATIO == M.CONTACT_DAMPING_RATIO
    assert G.CONTACT_SPEED == M.CONTACT_SPEED


def test_soft_invariants():
    """soft_step.py's own invariants, on the ncp_gpu re-derivation: I-2 (mass_scale + impulse_scale
    == 1), hertz <= 0 short-circuit, I-3 (speculative uses inv_h with unit scales), the push-out
    clamp at -contact_speed and the relax pass."""
    for hz in (5, 15, 30, 60, 120):
        for z in (1.0, 5.0, 10.0):
            for h in (1 / 60, 1 / 240, 1 / 960):
                s = G.make_soft(hz, z, h)
                assert abs(s.mass_scale + s.impulse_scale - 1.0) < 1e-12
    assert G.make_soft(0.0, 10.0, 1 / 240) == G.Softness(0.0, 0.0, 0.0)
    inv_h = 240.0
    soft = G.make_soft(G.contact_hertz(inv_h), G.CONTACT_DAMPING_RATIO, 1 / inv_h)
    vb, ms, isc = G.bias_velocity(0.01, soft, True, inv_h)
    assert abs(vb - 0.01 * inv_h) < 1e-12 and ms == 1.0 and isc == 0.0
    vb_deep, _, _ = G.bias_velocity(-10.0, soft, True, inv_h)
    assert abs(vb_deep + G.CONTACT_SPEED) < 1e-12
    assert G.bias_velocity(-0.05, soft, False, inv_h) == (0.0, 1.0, 0.0)
    # the substep Nyquist cap: 4 sub-steps of 1/240 make a 120 Hz contact spring reachable
    assert G.contact_hertz(240.0, 240.0) == 30.0 and G.contact_hertz(960.0, 240.0) == 120.0


def _independent_soft_scene(n_c=6, seed=11, all_touching=False):
    """n_c contacts, each on its OWN dynamic body against static ground: no two contacts share a
    body, so the colouring is a single colour and one solver iteration is a plain Jacobi step that
    the host can reproduce exactly."""
    rng = np.random.default_rng(seed)
    Ja = rng.normal(size=(n_c, 3, 6))
    Jb = np.zeros((n_c, 3, 6))
    Minv = np.zeros((n_c, 6, 6))
    for i in range(n_c):
        A = rng.normal(size=(6, 6))
        Minv[i] = A @ A.T + 3.0 * np.eye(6)
    pairs = np.stack([np.arange(n_c), -np.ones(n_c, int)], axis=1).astype(np.int32)
    sep = np.array([0.01, 0.0, -1e-5, -1e-3, -1.0, -5.0])[:n_c]
    if all_touching:                       # no speculative contact: s <= 0 everywhere
        sep = np.array([0.0, -1e-6, -1e-5, -1e-3, -1.0, -5.0])[:n_c]
    return G.BlockScene(Ja=Ja, Jb=Jb, Minv=Minv, v_free=rng.normal(size=(n_c, 6)),
                        mu=np.full(n_c, 0.4), body_pairs=pairs, dt=1 / 240,
                        bias=np.zeros((n_c, 3)), feature=np.arange(n_c, dtype=np.int32), sep=sep)


def _host_soft_iteration(bs, soft_p, use_bias, lam, desaxce=True):
    """Host mirror of the kernel's soft-step update, from `ncp_gpu.bias_velocity`."""
    br, ms0, is0, inv_h, cspeed = soft_p
    soft = G.Softness(br, ms0, is0)
    v = bs.v_free.copy()
    for c in range(bs.n_c):
        v[bs.body_pairs[c, 0]] += bs.Minv[bs.body_pairs[c, 0]] @ (bs.Ja[c].T @ lam[c])
    out = lam.copy()
    for c in range(bs.n_c):
        a = bs.body_pairs[c, 0]
        W = bs.Ja[c] @ bs.Minv[a] @ bs.Ja[c].T
        rho = 1.0 / np.max(np.linalg.eigvalsh(W))
        u = bs.bias[c] + bs.Ja[c] @ v[a]
        vb, ms, isc = G.bias_velocity(bs.sep[c], soft, use_bias, inv_h, cspeed)
        rn = u[0] + (bs.mu[c] * np.hypot(u[1], u[2]) if desaxce else 0.0)
        z = np.array([(1.0 - isc) * lam[c, 0] - rho * (ms * rn + vb),
                      lam[c, 1] - rho * u[1], lam[c, 2] - rho * u[2]])
        out[c] = _proj_cone(z, bs.mu[c])
    return out


def _proj_cone(z, mu):
    zt = np.hypot(z[1], z[2])
    if zt <= mu * z[0]:
        return z.copy()
    if mu * zt <= -z[0]:
        return np.zeros(3)
    s = (z[0] + mu * zt) / (1.0 + mu * mu)
    return np.array([s, mu * s / zt * z[1], mu * s / zt * z[2]])


@pytest.mark.parametrize("use_bias", [True, False])
def test_soft_kernel_matches_host_bias_velocity(use_bias):
    """One soft-step iteration on the GPU against the host mirror of soft_step.bias_velocity, with
    separations that exercise all three branches (speculative s > 0, biased s < 0, clamped
    s = -5 m at -contact_speed) and the relax pass."""
    bs = _independent_soft_scene()
    sp = G.SoftStep().params(1 / 240)
    s = G.NCPSolverGPU(bs.n_b, bs.n_c, dtype="float64", graph_capture=False)
    if use_bias:
        r = s.solve(bs, iters=1, soft=sp, measure_residual=False)
    else:
        r = s.solve(bs, iters=0, relax_iters=1, soft=sp, measure_residual=False)
    ref = _host_soft_iteration(bs, sp, use_bias, np.zeros((bs.n_c, 3)))
    err = np.max(np.abs(r.lam - ref))
    print(f"\n  soft kernel vs host bias_velocity, use_bias={use_bias}: "
          f"max|lambda_gpu - lambda_host| = {err:.3e} (|lambda| = {np.max(np.abs(ref)):.3e})")
    assert err < 1e-12
    assert s.n_colors == 1


def test_soft_relax_pass_is_the_unbiased_sweep():
    """use_bias = False collapses the soft branch to mass_scale 1 / impulse_scale 0 / bias 0, i.e.
    to the Baumgarte kernel path with a zero bias. Every separation is <= 0 here: the speculative
    branch (soft_step.py:59-60) is taken BEFORE the use_bias test, so a contact with a positive gap
    keeps its s*inv_h bias in the relax pass too.

    Agreement is to float64 rounding, not byte for byte: the relax branch evaluates
    `(1-is)*lam - rho*(ms*rn + vb)` with is = 0, ms = 1, vb = 0, which is the same real number as
    the Baumgarte `lam - rho*rn` but not the same instruction sequence (the multiply-add fuses
    differently). The measured gap is at the last bit; it is printed, not assumed."""
    bs = _independent_soft_scene(all_touching=True)
    sp = G.SoftStep().params(1 / 240)
    a = G.NCPSolverGPU(bs.n_b, bs.n_c, dtype="float64")
    ra = a.solve(bs, iters=0, relax_iters=64, soft=sp, measure_residual=False)
    b = G.NCPSolverGPU(bs.n_b, bs.n_c, dtype="float64")
    rb = b.solve(bs, iters=64, soft=None, measure_residual=False)
    dl = float(np.max(np.abs(ra.lam - rb.lam)))
    dv = float(np.max(np.abs(ra.v - rb.v)))
    scale = float(np.max(np.abs(rb.lam)))
    print(f"\n  relax-only soft sweep vs Baumgarte sweep with zero bias: "
          f"bytes {'identical' if a.raw_bytes() == b.raw_bytes() else 'differ'}, "
          f"max|dlambda|={dl:.3e} (|lambda|={scale:.3e}, rel {dl / scale:.3e}), max|dv|={dv:.3e}")
    assert dl < 1e-14 * scale and dv < 1e-14 * max(float(np.max(np.abs(rb.v))), 1e-30)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_soft_bit_identical_two_runs(dtype):
    """SPEC B test (a) on the soft-step path: the int64 fixed point makes two runs byte-identical
    with the soft bias and the two-pass (biased + relax) schedule too."""
    bs = _independent_soft_scene(n_c=6, seed=4)
    sp = G.SoftStep().params(1 / 240)
    out = []
    for _ in range(2):
        s = G.NCPSolverGPU(bs.n_b, bs.n_c, dtype=dtype)
        s.solve(bs, iters=40, relax_iters=13, soft=sp)
        out.append(s.raw_bytes())
    assert out[0][0] == out[1][0] and out[0][1] == out[1][1]
    # and through the whole world loop, 4 sub-steps x 25 macro steps
    b = []
    for _ in range(2):
        w, _pen = G.scene_tower(4, dtype=dtype, steps=25, iters=8, substeps=4,
                                soft=G.SoftStep(hertz=240.0))
        b.append(np.array([bd.x for bd in w.bodies]).tobytes())
    print(f"\n  soft-step {dtype}: solver bytes identical, world-loop position bytes "
          f"{'identical' if b[0] == b[1] else 'DIFFER'}")
    assert b[0] == b[1]








# ───────────────────────── bench ─────────────────────────




# ───────────────────────── extra benches (drift / sleeping / mass ratio) ─────────────────────────













def _first_sleep(v, w, thr=1e-3, k=8):
    """First step at which the island sleep rule fires: k consecutive steps with every body below
    `thr` in both speed and spin. `vmax_history` is already the max over the bodies, so the max
    being under the threshold IS every body being under it (ncp_gpu.Sleeper.begin)."""
    ok = (np.asarray(v) < thr) & (np.asarray(w) < thr)
    run = 0
    for i, o in enumerate(ok):
        run = run + 1 if o else 0
        if run >= k:
            return i
    return -1


_TOWER_CFG = {
    "baum-200":   ("baumgarte S=1 it=200        ", dict(iters=200)),
    "baum-32":    ("baumgarte S=1 it= 32        ", dict(iters=32)),
    "soft30-S1":  ("soft hz=30  S=1 it=200      ", dict(iters=200, substeps=1, soft=G.SoftStep())),
    "soft30-S2":  ("soft hz=30  S=2 it=100      ", dict(iters=100, substeps=2, soft=G.SoftStep())),
    "soft30-S4":  ("soft hz=30  S=4 it= 50      ", dict(iters=50, substeps=4, soft=G.SoftStep())),
    "softN-S1":   ("soft hz=240 S=1 it=200 (30) ",
                   dict(iters=200, substeps=1, soft=G.SoftStep(hertz=240.0))),
    "softN-S2":   ("soft hz=240 S=2 it=100 (60) ",
                   dict(iters=100, substeps=2, soft=G.SoftStep(hertz=240.0))),
    "softN-S4":   ("soft hz=240 S=4 it= 50 (120)",
                   dict(iters=50, substeps=4, soft=G.SoftStep(hertz=240.0))),
    "softN-S1b":  ("soft hz=240 S=1 it= 32 (30) ",
                   dict(iters=32, substeps=1, soft=G.SoftStep(hertz=240.0))),
    "softN-S2b":  ("soft hz=240 S=2 it= 16 (60) ",
                   dict(iters=16, substeps=2, soft=G.SoftStep(hertz=240.0))),
    "softN-S4b":  ("soft hz=240 S=4 it=  8 (120)",
                   dict(iters=8, substeps=4, soft=G.SoftStep(hertz=240.0))),
    "softN-S4n":  ("soft hz=240 S=4 it=  2 (120)",
                   dict(iters=2, substeps=4, soft=G.SoftStep(hertz=240.0))),
    "soft30-S4n": ("soft hz=30  S=4 it=  2      ", dict(iters=2, substeps=4, soft=G.SoftStep())),
}








