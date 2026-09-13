"""Checks on the individual layers: IK, planner, smoothing, time parametrisation, world model."""
import numpy as np
import pytest

pytest.importorskip("pinocchio")


@pytest.fixture(scope="module")
def scene():
    import fleet_agnostic_planner as fap
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    fap.URDF = str(root / "assets/robots/ur_description/ur10e.urdf")
    fap.PKG = str(root / "assets/robots/ur_description")
    return fap.load(obstacle_box=[0.2, 0.2, 0.5], box_pose=[0.55, 0.0, 0.3])


def test_ik_converges_and_is_honest_on_unreachable(scene):
    import pinocchio as pin
    from movement_ik import ik_multistart, fk
    from fleet_agnostic_planner import in_collision
    model, data, gm, gd, _ = scene
    fid = model.getFrameId("tool0")
    lo, hi = model.lowerPositionLimit[:model.nq], model.upperPositionLimit[:model.nq]
    rng = np.random.default_rng(0)
    q_true = next(q for q in (lo + rng.random(model.nq) * (hi - lo) for _ in range(500))
                  if not in_collision(model, data, gm, gd, q))
    target = fk(model, data, fid, q_true)
    q_ik, conv, free, _ = ik_multistart(model, data, fid, target, lo, hi, rng, gm=gm, gd=gd)
    oMf = fk(model, data, fid, q_ik)
    assert conv and free
    assert np.linalg.norm(oMf.translation - target.translation) < 1e-3
    far = pin.SE3(np.eye(3), np.array([5.0, 5.0, 5.0]))
    _, conv_far, _, _ = ik_multistart(model, data, fid, far, lo, hi, rng, n_restart=10)
    assert not conv_far


def _free_cfg(model, data, gm, gd, lo, hi, rng):
    from fleet_agnostic_planner import in_collision
    return next(q for q in (lo + rng.random(model.nq) * (hi - lo) for _ in range(500))
                if not in_collision(model, data, gm, gd, q))


def test_rrt_path_is_dense_edge_free(scene):
    from fleet_agnostic_planner import in_collision, rrt_connect
    model, data, gm, gd, _ = scene
    lo, hi = model.lowerPositionLimit[:model.nq], model.upperPositionLimit[:model.nq]
    rng = np.random.default_rng(1)
    q0 = _free_cfg(model, data, gm, gd, lo, hi, rng)
    q1 = _free_cfg(model, data, gm, gd, lo, hi, rng)
    path = rrt_connect(model, data, gm, gd, q0, q1, lo, hi, rng, n_iter=4000)
    assert path is not None and len(path) >= 2
    for a, b in zip(path[:-1], path[1:]):
        for s in np.linspace(0, 1, 16):
            assert not in_collision(model, data, gm, gd, a + s * (b - a))


def test_smoothing_keeps_the_smoothed_path_collision_free(scene):
    from fleet_agnostic_planner import rrt_connect
    from smooth_safe_trajectory import smooth_safe, spline_collides
    model, data, gm, gd, _ = scene
    lo, hi = model.lowerPositionLimit[:model.nq], model.upperPositionLimit[:model.nq]
    rng = np.random.default_rng(2)
    q0 = _free_cfg(model, data, gm, gd, lo, hi, rng)
    q1 = _free_cfg(model, data, gm, gd, lo, hi, rng)
    path = rrt_connect(model, data, gm, gd, q0, q1, lo, hi, rng, n_iter=4000)
    assert path is not None
    wp, cs, S, inserts, safe = smooth_safe(path, model, data, gm, gd, rng)
    assert safe
    collides, _, ncoll = spline_collides(cs, S, model, data, gm, gd)
    assert not collides and ncoll == 0
    assert np.isfinite(cs(np.linspace(0, S, 200))).all()


def test_time_parametrisation_respects_the_velocity_limits(scene):
    from fleet_agnostic_planner import rrt_connect
    from fleet_waypoint_blending import blend, time_param
    model, data, gm, gd, _ = scene
    lo, hi = model.lowerPositionLimit[:model.nq], model.upperPositionLimit[:model.nq]
    rng = np.random.default_rng(3)
    q0 = _free_cfg(model, data, gm, gd, lo, hi, rng)
    q1 = _free_cfg(model, data, gm, gd, lo, hi, rng)
    path = rrt_connect(model, data, gm, gd, q0, q1, lo, hi, rng, n_iter=4000)
    assert path is not None
    cs, S = blend([np.asarray(p, float) for p in path])
    eff = np.asarray(model.effortLimit[:model.nv])
    vel = np.asarray(model.velocityLimit[:model.nv])
    T = time_param(cs, S, model, data, eff, vel, dt=0.02)
    assert np.isfinite(T) and T > 0
    t = np.linspace(0, T, 400)
    dq = cs(S * t / T, 1) * (S / T)
    assert (np.abs(dq).max(0) <= vel).all()


def test_world_model_backends_agree():
    from motion_engine.world_model import PrimitiveExactWorld, EdtVoxelWorld
    obstacles = [dict(type="box", dims=[0.2, 0.2, 0.4], pose=[0.4, 0.0, 0.3])]
    exact = PrimitiveExactWorld(obstacles)
    voxel = EdtVoxelWorld([0.0, -0.4, 0.0], [0.8, 0.4, 0.8], 0.02,
                          lambda P: exact.sd_batch(P) < 0.0)
    rng = np.random.default_rng(0)
    P = np.column_stack([rng.uniform(0.05, 0.75, 200), rng.uniform(-0.35, 0.35, 200),
                         rng.uniform(0.05, 0.75, 200)])
    d_exact = exact.sd_batch(P)
    d_voxel = voxel.sd_batch(P)
    assert np.abs(d_exact - d_voxel).max() < 0.05
    g = exact.grad_batch(P)
    assert np.allclose(np.linalg.norm(g, axis=1), 1.0, atol=0.05)
    assert bool(exact.free(P[d_exact > 0.01]))
