"""The manipulation library (`motion_engine.manip`) exercised as one pick cycle on the vendored UR10e.

scene -> servo -> sensors -> grip in a single CUDA run: build the replicated pick scene, drive the
end-effector to the cube with the J^T servo, read the contact force and latch the grasp weld on one world.
"""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _cuda_available():
    try:
        import warp as wp
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _cuda_available(), reason="needs a CUDA device")


def test_pick_cycle_scene_servo_sensors_grip():
    pytest.importorskip("newton")
    pytest.importorskip("torch")
    import newton
    import warp as wp
    from newton.selection import ArticulationView

    from motion_engine.manip import grip as mgrip, scene as mscene, sensors as msens, servo as mservo
    from motion_engine.manip.profiles import load_profile
    from motion_engine.torch_dynamics import TorchDynamics

    wp.init()
    W = 2
    urdf = ROOT / "assets/robots/ur_description/ur10e.urdf"
    prof = load_profile("newton_contact_pick", "ur10e")
    cube = np.array([0.6, 0.0, 0.375])
    sc = mscene.build_pick_scene(urdf, W=W, plant_seed=777, cube_pos=cube, cube_h=0.025, ee_r=0.02)
    model = sc.builder.finalize()
    solver = newton.solvers.SolverMuJoCo(model, disable_contacts=False, nconmax=4096, njmax=300,
                                         cone="elliptic", impratio=10.0, use_mujoco_contacts=False)
    view = ArticulationView(model, prof.arm_pattern, exclude_joint_types=[newton.JointType.FREE])
    nj = view.joint_dof_count
    s0, s1 = model.state(), model.state()
    control = model.control()
    qa = view.get_attribute("joint_q", s0)
    arr = qa.numpy(); arr[:, 0, :nj] = np.asarray(prof.home[:nj])
    qa.assign(arr); view.set_attribute("joint_q", s0, qa)
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    state = {"s0": s0, "s1": s1}

    dyn = TorchDynamics(ROOT / "data/ur10e_gravparams.npz", device="cuda")
    srv = mservo.JTServo(model=model, solver=solver, view=view, control=control, state=state, dyn=dyn,
                         W=W, sub=4, dt=1.0 / 30.0 / 4, profile=prof, nj=nj, device="cuda")
    sensor = msens.ContactSensor(model, solver, W=W)
    latch = mgrip.WeldLatch(model, solver)

    goal = np.tile(cube + np.array([0.0, 0.0, 0.02]), (W, 1))
    err0 = float(np.linalg.norm(srv.read_state()[0] - goal, axis=1).mean())
    for _ in range(60):
        srv.servo_step(goal)
    err1 = float(np.linalg.norm(srv.read_state()[0] - goal, axis=1).mean())
    assert np.isfinite(err1) and err1 < err0 / 5.0          # the servo closes the task-space error

    force = sensor.force(state["s0"])
    assert force.shape == (W, 3) and np.all(np.isfinite(force))

    bq = state["s0"].body_q.numpy()
    x_ee = bq[np.arange(W) * sc.nb_t + (sc.nb_rob - 1)]
    x_cube = bq[np.arange(W) * sc.nb_t + (sc.nb_t - 1)]
    relpose = latch.engage([0], x_ee, x_cube)
    assert np.all(np.isfinite(relpose))
    enabled = model.equality_constraint_enabled.numpy()
    assert bool(enabled[0]) and not bool(enabled[1])        # the latch binds per world, not globally
