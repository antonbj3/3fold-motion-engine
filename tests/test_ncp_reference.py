"""CPU float64 exact-cone NCP reference: the numbers the module claims, locked.

    OMP_NUM_THREADS=4 python -m pytest -s -q tests/test_ncp_reference.py

Run with `-s` to print the residual, iteration and label tables; those printed values
are the ones quoted in `motion_engine.ncp.ncp_ref`'s docstring.

Residual requirement throughout: 1e-8, measured as
`||lam - proj_K(lam - rho (u + Gamma(u)))||_inf` with `rho_c = 1/||G_cc||_2`, in N s.

MEASURED NEGATIVE, kept as four xfail rows instead of being deleted: PGS does not
reach 1e-8 at a 2000-sweep budget on the four hyperstatic scenes. The measured
residuals at that budget are pushed cube 5.572e-07, stack-100 3.108e-05,
stack-1000 1.458e-03, K8 5.380e-06. At 20 000 sweeps three of the four are still
descending (K8 7.72e-11, pushed cube 1.69e-08, stack-100 8.04e-08); only
stack-1000 is flat (1.46e-03 -> 3.11e-05 -> 3.11e-05), so "slow" is the right word
for three of them and "stuck" only for stack-1000. ADMM reaches 1e-12 on all six in
64-299 iterations with one Cholesky of G + rho I.
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from motion_engine.ncp import ncp_ref as R  # noqa: E402

TOL_SPEC = 1e-8           # residual requirement
PGS_BUDGET = 2000
ADMM_BUDGET = 20000


# ---------------------------------------------------------------------------
# scene fixtures: the SPEC A.6 scenes, first step (t = 0 unless stated)
# ---------------------------------------------------------------------------


def _first_step(bodies, cfg, force_fn=None, t=0.0):
    vf = np.zeros(6 * len(bodies))
    for k, bd in enumerate(bodies):
        f = bd.mass * R.GRAVITY.copy()
        if force_fn is not None:
            f = f + force_fn(t, k, bd)[0]
        vf[6 * k:6 * k + 3] = bd.vel + cfg["dt"] * f / bd.mass
        vf[6 * k + 3:6 * k + 6] = bd.omega
    ct = R.collect_contacts(bodies)
    sc = R.build_scene(bodies, ct, cfg["mu"], cfg["dt"], v_free=vf)
    return sc


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
    raise KeyError(name)


SCENES = ["i_pushed_cube", "ii_sliding_box", "iii_stack_1", "iii_stack_100",
          "iii_stack_1000", "iv_K8"]

# total mass above the ground interface, for the analytic impulse check
SCENE_MASS = {"i_pushed_cube": 15.0, "ii_sliding_box": 1.0, "iii_stack_1": 3.0,
              "iii_stack_100": 102.0, "iii_stack_1000": 1002.0, "iv_K8": 8.0}


# ---------------------------------------------------------------------------
# 1. cone algebra
# ---------------------------------------------------------------------------


def test_cone_projection_properties():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5000, 3)) * 2.0
    mu = rng.uniform(0.02, 2.0, 5000)
    P = R.proj_cone(X, mu)
    Po = X - P
    feas = float(np.max(np.linalg.norm(P[:, 1:3], axis=1) - mu * P[:, 0]))
    idem = float(np.max(np.abs(R.proj_cone(P, mu) - P)))
    moreau = float(np.max(np.abs(np.sum(P * Po, axis=1))))
    polar = float(np.max(np.linalg.norm(Po[:, 1:3], axis=1) + Po[:, 0] / mu))
    print(f"\n[cone] feasibility={feas:.3e} idempotence={idem:.3e} "
          f"Moreau<P,x-P>={moreau:.3e} polar_violation={polar:.3e}")
    assert feas <= 1e-12
    assert idem <= 1e-12
    assert moreau <= 1e-12
    assert polar <= 1e-12

    # degenerate mu = 0:  K_0 = {lam_t = 0, lam_n >= 0}
    z = R.proj_cone(np.array([[-1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]), np.zeros(2))
    assert np.allclose(z, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


def test_dual_cone_is_K_one_over_mu():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2000, 3)) * 2.0
    mu = rng.uniform(0.05, 1.5, 2000)
    D = R.proj_dual_cone(X, mu)
    assert np.max(np.linalg.norm(D[:, 1:3], axis=1) - D[:, 0] / mu) <= 1e-12
    assert np.allclose(D, R.proj_cone(X, 1.0 / mu))
    # Gamma(u, mu) = (mu ||u_t||, 0, 0)          [2304.06372 eq. 14]
    u = np.array([[-1.0, 3.0, 4.0]])
    assert np.allclose(R.desaxce(u, np.array([0.5])), [[2.5, 0.0, 0.0]])


# ---------------------------------------------------------------------------
# 2. A<->B contract
# ---------------------------------------------------------------------------


def test_scene_contract():
    import dataclasses
    names = [f.name for f in dataclasses.fields(R.Scene)]
    print(f"\n[contract] Scene fields = {names}")
    assert names[:6] == ["J", "Minv", "v_free", "mu", "body_pairs", "dt"]
    sc = make_scene("iv_K8")
    assert sc.J.shape == (3 * sc.n_c, sc.n_v) and sc.J.dtype == np.float64
    assert sc.mu.shape == (sc.n_c,)
    assert sc.body_pairs.shape == (sc.n_c, 2)
    assert sc.body_pairs.min() >= -1
    G = sc.delassus()
    assert np.max(np.abs(G - G.T)) <= 1e-14 * max(1.0, np.abs(G).max())
    assert np.min(np.linalg.eigvalsh(0.5 * (G + G.T))) >= -1e-10 * np.abs(G).max()
    # positional construction (what the GPU side will do)
    s2 = R.Scene(sc.J, sc.Minv, sc.v_free, sc.mu, sc.body_pairs, sc.dt)
    assert np.allclose(s2.b(), sc.J @ sc.v_free)


def test_contact_generation():
    b, _, _ = R.scene_tower(n=8)
    ct = R.collect_contacts(b)
    n_ground = sum(1 for c in ct if c["b"] == -1)
    print(f"\n[geom] K8 contacts={len(ct)} ground={n_ground} "
          f"max|gap|={max(abs(c['gap']) for c in ct):.2e}")
    assert len(ct) == 32 and n_ground == 4
    assert all(np.allclose(c["n"], [0, 0, 1]) for c in ct)
    b, _, _ = R.scene_massratio_stack(ratio=1000.0)
    assert len(R.collect_contacts(b)) == 12
    b, _, _ = R.scene_pushed_cube()
    assert len(R.collect_contacts(b)) == 4


# ---------------------------------------------------------------------------
# 3. solver convergence on the SPEC scenes  (A.7: residual < 1e-8)
# ---------------------------------------------------------------------------

_CONVERGE = {}


# PGS misses 1e-8 on exactly these four at a 2000-sweep budget; the measured residual
# at that budget is the reason string, so a silent improvement fails the strict xfail.
PGS_MEASURED_MISS = {"i_pushed_cube": "5.572e-07", "iii_stack_100": "3.108e-05",
                     "iii_stack_1000": "1.458e-03", "iv_K8": "5.380e-06"}
PGS_SCENES = [
    pytest.param(n, marks=pytest.mark.xfail(
        strict=True,
        reason=f"measured: PGS residual {PGS_MEASURED_MISS[n]} at {PGS_BUDGET} sweeps, "
               f"requirement 1e-8")) if n in PGS_MEASURED_MISS else n
    for n in SCENES
]


@pytest.mark.parametrize("name", PGS_SCENES)
def test_ncp_pgs_residual(name):
    sc = make_scene(name)
    G, b = sc.delassus(), sc.b()
    t0 = time.time()
    lam, hist, labels = R.solve_ncp_pgs(G, b, sc.mu, PGS_BUDGET, 1e-12)
    dt = time.time() - t0
    rank = int(np.linalg.matrix_rank(G, tol=1e-10 * max(1.0, np.abs(G).max())))
    _CONVERGE[("pgs", name)] = (len(hist), float(hist[-1]))
    print(f"\n[PGS  ] {name:15} n_c={sc.n_c:3d} rank(G)={rank:3d}/{3*sc.n_c:3d} "
          f"iters={len(hist):5d}/{PGS_BUDGET} res={hist[-1]:.3e} t={dt:.2f}s "
          f"labels={dict(zip(*np.unique(labels, return_counts=True)))}")
    assert hist[-1] < TOL_SPEC, (
        f"{name}: PGS stalled at {hist[-1]:.3e} after {len(hist)} sweeps "
        f"(SPEC requires < {TOL_SPEC})")


@pytest.mark.parametrize("name", SCENES)
def test_ncp_admm_residual(name):
    sc = make_scene(name)
    G, b = sc.delassus(), sc.b()
    t0 = time.time()
    lam, hist, labels = R.solve_ncp_admm(G, b, sc.mu, ADMM_BUDGET, 1e-12)
    dt = time.time() - t0
    _CONVERGE[("admm", name)] = (len(hist), float(hist[-1]))
    print(f"\n[ADMM ] {name:15} n_c={sc.n_c:3d} iters={len(hist):5d} "
          f"res={hist[-1]:.3e} t={dt:.2f}s "
          f"labels={dict(zip(*np.unique(labels, return_counts=True)))}")
    assert hist[-1] < TOL_SPEC


@pytest.mark.parametrize("name", SCENES)
def test_complementarity_per_contact(name):
    """eps_p = dist_K(lam), eps_d = dist_K*(u+Gamma(u)), eps_c = |<lam, u+Gamma(u)>|
    checked per contact on the ADMM solution   [2304.06372 §III-A]."""
    sc = make_scene(name)
    G, b = sc.delassus(), sc.b()
    lam, hist, labels = R.solve_ncp_admm(G, b, sc.mu, ADMM_BUDGET, 1e-13)
    ep, ed, ec = R.complementarity(lam, G, b, sc.mu)
    print(f"\n[compl] {name:15} max eps_p={ep.max():.3e} eps_d={ed.max():.3e} "
          f"eps_c={ec.max():.3e}  (per-contact, n_c={sc.n_c})")
    assert ep.max() <= 1e-12           # lam in K, exactly (projection is exact)
    assert ed.max() <= 1e-9
    assert ec.max() <= 1e-9
    # per-contact label consistency
    u = (G @ lam + b).reshape(-1, 3)
    L = lam.reshape(-1, 3)
    for c in range(sc.n_c):
        if labels[c] == "open":
            assert np.linalg.norm(L[c]) <= 1e-10
        elif labels[c] == "stick":
            assert np.linalg.norm(u[c]) <= 1e-7
        else:                                     # slip: u_n = 0, no hover
            assert abs(u[c, 0]) <= 1e-7


@pytest.mark.parametrize("name", SCENES)
def test_normal_impulse_equals_stacked_weight(name):
    """External anchor: sum of ground-interface normal impulses = M_total g dt."""
    sc = make_scene(name)
    G, b = sc.delassus(), sc.b()
    lam, hist, _ = R.solve_ncp_admm(G, b, sc.mu, ADMM_BUDGET, 1e-13)
    gi = [c for c in range(sc.n_c) if sc.body_pairs[c, 1] == -1]
    meas = float(lam.reshape(-1, 3)[gi, 0].sum())
    ana = SCENE_MASS[name] * 9.81 * sc.dt
    rel = abs(meas - ana) / ana
    print(f"\n[weight] {name:15} sum lam_n = {meas:.9f} N.s   analytic = {ana:.9f} N.s"
          f"   rel = {rel:.3e}")
    assert rel < 1e-8


# ---------------------------------------------------------------------------
# 4. the gliding number: NCP vs Anitescu convex relaxation
# ---------------------------------------------------------------------------


def test_gliding_ncp_vs_anitescu():
    """SPEC A.6(ii): sliding box, v0 = 1 m/s, mu = 0.3, dt = 0.004.
    NCP must give gap 0; the convex relaxation hovers at Anitescu's dt*mu*||u_t||
    [2304.06372 §III-B, eq. 19]."""
    dt, mu, v0, steps = 0.004, 0.3, 1.0, 200
    out = {}
    for tag, solver in (("NCP", R.solve_ncp_pgs), ("CCP", R.solve_convex_pgs)):
        bodies, _, cfg = R.scene_sliding_box(dt=dt, mu=mu, v0=v0)
        h = R.simulate(bodies, steps, dt, mu, solver=solver, iters=500, tol=1e-12)
        out[tag] = h
    gn, gc = out["NCP"]["min_gap"], out["CCP"]["min_gap"]
    vtc = np.abs(out["CCP"]["vel"][:, 0, 0])
    win = slice(20, 80)                                 # quasi-steady slide window
    pred = dt * mu * vtc[win]
    ratio = gc[win] / pred
    print(f"\n[glide] dt={dt} mu={mu} v0={v0} steps={steps}")
    print(f"[glide]   NCP   max gap over slide = {gn.max():+.3e} m   "
          f"|gap|_max = {np.abs(gn).max():.3e} m")
    print(f"[glide]   CCP   max gap over slide = {gc.max():+.3e} m   "
          f"mean gap (steps 20-80) = {gc[win].mean():.3e} m")
    print(f"[glide]   Anitescu dt*mu*|u_t| over the same window: "
          f"mean = {pred.mean():.3e} m")
    print(f"[glide]   measured/analytic ratio: mean = {ratio.mean():.4f} "
          f"min = {ratio.min():.4f} max = {ratio.max():.4f}")
    print(f"[glide]   first-step tangential velocity: NCP {out['NCP']['vel'][0,0,0]:.6f} "
          f"m/s  CCP {out['CCP']['vel'][0,0,0]:.6f} m/s  "
          f"(analytic NCP: {v0 - dt*mu*9.81:.6f})")
    assert np.abs(gn).max() < 1e-12                     # NCP: no hover
    assert gc.max() > 1e-3                              # CCP: hovers
    assert abs(ratio.mean() - 1.0) < 0.05               # matches Anitescu's dt mu |u_t|
    assert abs(out["NCP"]["vel"][0, 0, 0] - (v0 - dt * mu * 9.81)) < 1e-12


def test_convex_relaxation_inflates_first_step_friction():
    """Second consequence of the missing de Saxce term: the hover push inflates
    the normal impulse, hence the friction impulse, on the impacting step."""
    dt, mu, v0 = 0.004, 0.3, 1.0
    res = {}
    for tag, solver in (("NCP", R.solve_ncp_pgs), ("CCP", R.solve_convex_pgs)):
        bodies, _, cfg = R.scene_sliding_box(dt=dt, mu=mu, v0=v0)
        h = R.simulate(bodies, 1, dt, mu, solver=solver, iters=500, tol=1e-12)
        res[tag] = (float(h["vel"][0, 0, 0]), float(h["lam_sum"][0]))
    dv_ncp = v0 - res["NCP"][0]
    dv_ccp = v0 - res["CCP"][0]
    print(f"\n[glide2] step-0 tangential velocity loss: NCP {dv_ncp:.6e} m/s  "
          f"CCP {dv_ccp:.6e} m/s  ratio {dv_ccp/dv_ncp:.3f}")
    print(f"[glide2] step-0 sum lam_n: NCP {res['NCP'][1]:.6e}  CCP {res['CCP'][1]:.6e} N.s")
    assert dv_ccp > 5.0 * dv_ncp


# ---------------------------------------------------------------------------
# 5. analytical sensitivity vs central finite differences
# ---------------------------------------------------------------------------


def _relerr(a, f):
    return float(np.max(np.abs(a - f)) / max(float(np.max(np.abs(f))), 1e-30))


def test_sensitivity_vs_fd_tripod():
    """Statically determinate 3-point contact -> non-singular active-set Jacobian."""
    sc = R.scene_tripod(mu=0.4)
    G, b = sc.delassus(), sc.b()
    lam, hist, labels = R.solve_ncp_pgs(G, b, sc.mu, 5000, 1e-14)
    dmu, db, info = R.sensitivity(lam, G, b, sc.mu, labels)
    fmu, fdb = R.sensitivity_fd(G, b, sc.mu, lam, h_mu=1e-6, h_b=1e-6)
    e_mu, e_b = _relerr(dmu, fmu), _relerr(db, fdb)
    print(f"\n[sens] tripod labels={list(labels)} res={hist[-1]:.2e} "
          f"cond(dF/dlam)={info['cond']:.3e} singular={info['singular']}")
    print(f"[sens]   rel err dlam/dmu = {e_mu:.3e}   dlam/db = {e_b:.3e}   "
          f"(FD h=1e-6, central)   boundary cases: {info['boundary']}")
    assert not info["singular"]
    assert e_mu < 1e-5
    assert e_b < 1e-5


@pytest.mark.parametrize("seed", [3, 7, 11, 13, 17])
def test_sensitivity_vs_fd_random(seed):
    sc = R.scene_random(n_c=4, n_b=2, mu=0.4, seed=seed)
    G, b = sc.delassus(), sc.b()
    lam, hist, labels = R.solve_ncp_pgs(G, b, sc.mu, 5000, 1e-14)
    dmu, db, info = R.sensitivity(lam, G, b, sc.mu, labels)
    fmu, fdb = R.sensitivity_fd(G, b, sc.mu, lam, h_mu=1e-6, h_b=1e-6)
    e_mu, e_b = _relerr(dmu, fmu), _relerr(db, fdb)
    print(f"\n[sens] seed={seed} labels={list(labels)} cond={info['cond']:.3e} "
          f"singular={info['singular']} rel dlam/dmu={e_mu:.3e} dlam/db={e_b:.3e}")
    assert e_mu < 1e-5
    assert e_b < 1e-5


def test_sensitivity_hyperstatic_is_flagged_not_hidden():
    """4-point box on a plane: rank(G) = 6 < 12 -> lam is NOT unique, the
    active-set Jacobian is singular.  sensitivity() must report it, and the FD
    comparison is meaningless there; recorded as a negative result."""
    sc = make_scene("i_pushed_cube")
    G, b = sc.delassus(), sc.b()
    lam, hist, labels = R.solve_ncp_admm(G, b, sc.mu, ADMM_BUDGET, 1e-13)
    dmu, db, info = R.sensitivity(lam, G, b, sc.mu, labels)
    rank = int(np.linalg.matrix_rank(G, tol=1e-10 * np.abs(G).max()))
    print(f"\n[sens-hyp] pushed cube: n_c={sc.n_c} rank(G)={rank}/{3*sc.n_c} "
          f"labels={list(labels)} cond(dF/dlam)={info['cond']:.3e} "
          f"singular={info['singular']} -> pseudo-inverse used")
    assert rank < 3 * sc.n_c
    assert info["singular"]


def test_sensitivity_slip_direction_formula():
    """Directional check: d lam_t / d mu along the slip direction equals
    -lam_n * u_t/||u_t|| to first order (pure Coulomb scaling) when G is diagonal."""
    G = np.diag([1.0, 1.0, 1.0])
    b = np.array([-1.0, 2.0, 0.0])
    mu = np.array([0.3])
    lam, hist, labels = R.solve_ncp_pgs(G, b, mu, 5000, 1e-14)
    dmu, db, info = R.sensitivity(lam, G, b, mu, labels)
    fmu, _ = R.sensitivity_fd(G, b, mu, lam, h_mu=1e-6, h_b=1e-6)
    print(f"\n[sens-1c] label={labels[0]} lam={lam} dlam/dmu={dmu[:,0]} "
          f"fd={fmu[:,0]} rel={_relerr(dmu, fmu):.3e}")
    assert labels[0] == "slip"
    assert _relerr(dmu, fmu) < 1e-5


# ---------------------------------------------------------------------------
# 6. baselines: convex relaxation and pyramid
# ---------------------------------------------------------------------------


def test_convex_pgs_is_a_relaxation():
    """CCP allows u_n > 0 while lam_n > 0 (eq. 19); the NCP does not."""
    sc = make_scene("ii_sliding_box")
    G, b = sc.delassus(), sc.b()
    ln, _, _ = R.solve_ncp_pgs(G, b, sc.mu, 2000, 1e-13)
    lc, _, _ = R.solve_convex_pgs(G, b, sc.mu, 2000, 1e-13)
    un = (G @ ln + b).reshape(-1, 3)[:, 0]
    uc = (G @ lc + b).reshape(-1, 3)[:, 0]
    ut = np.linalg.norm((G @ lc + b).reshape(-1, 3)[:, 1:3], axis=1)
    print(f"\n[ccp] NCP max u_n = {np.abs(un).max():.3e} m/s   "
          f"CCP max u_n = {uc.max():.6e} m/s   mu*||u_t|| = {(sc.mu*ut).max():.6e} m/s")
    assert np.abs(un).max() < 1e-12
    assert uc.max() > 1e-3
    assert np.max(np.abs(uc - sc.mu * ut)) < 1e-9        # exactly eq. (19)


def test_pyramid_inscribed_is_inside_the_exact_cone():
    s = 1.0 / math.sqrt(2.0)
    sc = R.scene_random(n_c=6, n_b=3, mu=0.5, seed=5)
    G, b = sc.delassus(), sc.b()
    lp, hp, _ = R.solve_pyramid_qp(G, b, sc.mu, 500, 1e-14, scale=s, method="scipy")
    L = lp.reshape(-1, 3)
    viol_pyr = float(np.max(np.abs(L[:, 1:3]) - (s * sc.mu * L[:, 0])[:, None]))
    viol_cone = float(np.max(np.linalg.norm(L[:, 1:3], axis=1) - sc.mu * L[:, 0]))
    print(f"\n[pyr] scipy/SLSQP: pyramid violation={viol_pyr:.3e} "
          f"exact-cone violation={viol_cone:.3e} (inscribed -> must be <= 0)")
    assert viol_pyr <= 1e-9
    assert viol_cone <= 1e-9


@pytest.mark.parametrize("seed", [3, 7, 11, 13])
def test_pyramid_pgs_is_not_the_qp_optimum(seed):
    """Negative result: the classic ODE/Bullet boxed-LCP PGS sweep (friction bound
    frozen at the current lam_n) has a fixed point that is NOT the pyramid QP
    minimiser.  scipy/SLSQP reaches a strictly lower objective."""
    s = 1.0 / math.sqrt(2.0)
    sc = R.scene_random(n_c=6, n_b=3, mu=0.5, seed=seed)
    G, b = sc.delassus(), sc.b()
    obj = lambda x: 0.5 * float(x @ (G @ x)) + float(b @ x)
    lp, hp, _ = R.solve_pyramid_qp(G, b, sc.mu, 50000, 1e-15, scale=s, method="pgs")
    ls, _, _ = R.solve_pyramid_qp(G, b, sc.mu, 500, 1e-14, scale=s, method="scipy")
    print(f"\n[pyr] seed={seed} pgs obj={obj(lp):.10f} ({len(hp)} sweeps)  "
          f"scipy obj={obj(ls):.10f}  gap={obj(lp)-obj(ls):.3e}")
    assert obj(ls) <= obj(lp) + 1e-12


def test_pyramid_vs_second_order_cone_same_relaxation():
    """Cone SHAPE effect, isolated: the pyramid QP and solve_convex_pgs are the same
    convex relaxation (no de Saxce), so comparing them isolates the linearisation
    from the relaxation.  Comparing the pyramid against the exact NCP would conflate
    the two, so that number is printed separately and labelled."""
    sc = make_scene("ii_sliding_box")
    G, b = sc.delassus(), sc.b()
    lncp, _, _ = R.solve_ncp_pgs(G, b, sc.mu, 2000, 1e-13)
    lccp, _, _ = R.solve_convex_pgs(G, b, sc.mu, 2000, 1e-13)
    ft = lambda x: float(np.linalg.norm(x.reshape(-1, 3)[:, 1:3].sum(axis=0)))
    print()
    for s, tag in ((1.0 / math.sqrt(2.0), "inscribed"), (1.0, "circumscribed")):
        lp, _, _ = R.solve_pyramid_qp(G, b, sc.mu, 500, 1e-14, scale=s, method="scipy")
        print(f"[pyr] {tag:13} scale={s:.6f}: |sum lam_t| pyramid={ft(lp):.6e}  "
              f"SOC-convex={ft(lccp):.6e}  ratio(pyr/SOC)={ft(lp)/ft(lccp):.6f}  "
              f"| exact-NCP={ft(lncp):.6e} (different model, not a cone-shape number)")


# ---------------------------------------------------------------------------
# 8. multi-step trajectories: analytic anchors and penetration
# ---------------------------------------------------------------------------


def test_pushed_cube_trajectory_matches_coulomb_acceleration():
    """SPEC A.6(i).  F = 147.15 N, mu = 0.5, m = 15 kg  ->  during full slip
    a = (F - mu m g)/m = (147.15 - 73.575)/15 = 4.905 m/s^2 exactly."""
    bodies, ff, cfg = R.scene_pushed_cube()
    h = R.simulate(bodies, 125, cfg["dt"], cfg["mu"], solver=R.solve_ncp_admm,
                   iters=ADMM_BUDGET, tol=1e-12, force_fn=ff)
    vx = h["vel"][:, 0, 0]
    dtc = cfg["dt"]
    # steady slip window: after the 0.3 s ramp (step 75) and before the 0.5 s switch
    w = slice(80, 124)
    a_meas = float(np.mean(np.diff(vx[w]) / dtc))
    a_ana = (147.15 - 0.5 * 15.0 * 9.81) / 15.0
    print(f"\n[cube] steps=125 dt={dtc} max_res={h['res'].max():.3e} "
          f"mean_iters={h['iters'].mean():.1f} n_c={h['n_c'].max()}")
    print(f"[cube]   measured a_x (steps 80-124) = {a_meas:.9f} m/s^2   "
          f"analytic = {a_ana:.9f} m/s^2   rel = {abs(a_meas-a_ana)/a_ana:.3e}")
    print(f"[cube]   max penetration over the run = {h['max_pen'].max():.3e} m   "
          f"final pos = {bodies[0].pos}")
    assert abs(a_meas - a_ana) / a_ana < 1e-9
    assert h["max_pen"].max() < 1e-12


@pytest.mark.parametrize("ratio", [1.0, 100.0, 1000.0])
def test_massratio_stack_does_not_penetrate(ratio):
    """SPEC A.6(iii), geometry of primal_vs_dual_massratio.py (H=0.20, dt=1/240)."""
    bodies, _, cfg = R.scene_massratio_stack(ratio=ratio)
    h = R.simulate(bodies, 300, cfg["dt"], cfg["mu"], solver=R.solve_ncp_admm,
                   iters=ADMM_BUDGET, tol=1e-12)
    z = np.array([b.pos[2] for b in bodies])
    z_ref = np.array([(k + 0.5) * 0.20 for k in range(len(bodies))])
    print(f"\n[stack] ratio={ratio:7.0f} steps=300 max_pen={h['max_pen'].max()*1000:.6f} mm "
          f"max_res={h['res'].max():.3e} mean_iters={h['iters'].mean():.1f} "
          f"max|z - z_rest|={np.abs(z-z_ref).max()*1000:.6f} mm")
    assert h["max_pen"].max() * 1000 < 1e-6
    assert np.abs(z - z_ref).max() * 1000 < 1e-6


def test_K8_tower_penetration_vs_M5_gate():
    """SPEC A.6(iv).  motion-engine M5 gate: max penetration / R < 0.5 with R = 0.05 m,
    K = 8, dt = 1/240, 600 steps (scripts/engine_metrics.py m5_penetration).
    CONVENTION DIFFERENCE, stated rather than hidden: M5 measures
    (rest pitch 0.205 - spacing) for 0.3x0.3x0.2 corner-particle bodies whose rest
    pitch exceeds the box height by the corner radius; here bodies are rigid 0.20 m
    cubes whose rest pitch IS 0.20, so the same subtraction is done against 0.20.
    The two numbers share the normaliser R but not the rest-pitch convention."""
    R_CORNER, PITCH, GATE = 0.05, 0.20, 0.5
    bodies, _, cfg = R.scene_tower(n=8)
    h = R.simulate(bodies, 600, 1.0 / 240.0, cfg["mu"], solver=R.solve_ncp_admm,
                   iters=ADMM_BUDGET, tol=1e-12, record=None)
    z = np.sort(h["pos"][:, :, 2], axis=1)
    pen = np.max(PITCH - np.diff(z, axis=1), axis=1)
    half = len(pen) // 2
    slope = float(np.polyfit(np.arange(half), pen[half:], 1)[0])
    drift = "secular" if abs(slope) * half > 0.1 * R_CORNER else "bounded"
    print(f"\n[K8] steps=600 dt=1/240 n_c={h['n_c'].max()} max_res={h['res'].max():.3e} "
          f"mean_iters={h['iters'].mean():.1f}")
    print(f"[K8]   max_pen = {pen.max():.6e} m   pen/R = {pen.max()/R_CORNER:.9f}   "
          f"gate = {GATE}   drift = {drift} (slope {slope:.3e} m/step)")
    assert pen.max() / R_CORNER < GATE
    assert drift == "bounded"


# ---------------------------------------------------------------------------
# 7. warm start + solver agreement
# ---------------------------------------------------------------------------


def test_warm_start_reduces_iterations():
    sc = make_scene("ii_sliding_box")
    G, b = sc.delassus(), sc.b()
    lam, h_cold, _ = R.solve_ncp_pgs(G, b, sc.mu, 2000, 1e-13)
    _, h_warm, _ = R.solve_ncp_pgs(G, b, sc.mu, 2000, 1e-13, warm=lam)
    print(f"\n[warm] cold={len(h_cold)} sweeps  warm={len(h_warm)} sweeps")
    assert len(h_warm) <= len(h_cold)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_pgs_and_admm_agree_on_wellposed_scenes(seed):
    """Where G is full rank the NCP solution is unique -> the two solvers must
    return the same lam."""
    sc = R.scene_random(n_c=5, n_b=4, mu=0.4, seed=seed)
    G, b = sc.delassus(), sc.b()
    rank = int(np.linalg.matrix_rank(G, tol=1e-10 * np.abs(G).max()))
    l1, h1, _ = R.solve_ncp_pgs(G, b, sc.mu, 5000, 1e-14)
    l2, h2, _ = R.solve_ncp_admm(G, b, sc.mu, ADMM_BUDGET, 1e-14)
    d = float(np.max(np.abs(l1 - l2)))
    print(f"\n[agree] seed={seed} rank(G)={rank}/{3*sc.n_c} "
          f"PGS {len(h1)}it/{h1[-1]:.1e}  ADMM {len(h2)}it/{h2[-1]:.1e}  |dlam|={d:.3e}")
    if rank == 3 * sc.n_c:
        assert d < 1e-8


def test_nonuniqueness_is_reported_on_hyperstatic_scenes():
    """Negative result: on the box scenes rank(G) < 3 n_c, so PGS and ADMM land on
    different (equally valid) impulses; only the resultant is unique."""
    rows = []
    for name in SCENES:
        sc = make_scene(name)
        G, b = sc.delassus(), sc.b()
        rank = int(np.linalg.matrix_rank(G, tol=1e-10 * np.abs(G).max()))
        l1, _, _ = R.solve_ncp_pgs(G, b, sc.mu, PGS_BUDGET, 1e-12)
        l2, _, _ = R.solve_ncp_admm(G, b, sc.mu, ADMM_BUDGET, 1e-13)
        J = sc.J
        r1, r2 = J.T @ l1, J.T @ l2
        rows.append((name, sc.n_c, rank, float(np.max(np.abs(l1 - l2))),
                     float(np.max(np.abs(r1 - r2)))))
    print()
    for nm, nc, rk, dl, dr in rows:
        print(f"[nonuniq] {nm:15} n_c={nc:3d} rank(G)={rk:3d}/{3*nc:3d} "
              f"max|lam_PGS - lam_ADMM|={dl:.3e}  max|J^T dlam| (generalised force)={dr:.3e}")
    assert all(rk < 3 * nc for _, nc, rk, _, _ in rows)
