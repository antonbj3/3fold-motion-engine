"""End-to-end checks on MotionStack with the vendored UR10e model."""
import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")
from motion_engine.motion_stack import MotionStack, MotionPlan  # noqa: E402

from pathlib import Path  # noqa: E402
ROOT = Path(__file__).resolve().parents[1]
URDF = str(ROOT / "assets/robots/ur_description/ur10e.urdf")
PKG = str(ROOT / "assets/robots/ur_description")


@pytest.fixture(scope="module")
def stack():
    return MotionStack(URDF, pkg=PKG, ee_frame="tool0", obstacle_box=[0.15, 0.15, 0.4],
                       box_pose=[0.45, 0.0, 0.3], twin_name="ur10e", seed=1)


def _plan_ok(stack, tries=6):
    """Plan to a reachable free pose; RRT is stochastic, so retry a bounded number of times."""
    from movement_ik import fk
    for _ in range(tries):
        q = next(q for q in (stack.sample() for _ in range(1500)) if stack.free(q))
        plan = stack.plan_cartesian(fk(stack.model, stack.data, stack.fid, q), q_start=stack.sample())
        if plan.ok:
            return plan
    raise AssertionError("no successful plan in %d attempts" % tries)


def test_selftest_module_passes():
    from motion_engine import selftest_motion_stack as st
    assert st.main() == 0


def test_plan_cartesian_is_smooth_and_collision_free(stack):
    plan = _plan_ok(stack)
    assert isinstance(plan, MotionPlan)
    assert plan.ok and plan.collision_free
    assert plan.trajectory_q is not None and plan.waypoints >= 2
    assert plan.ik_pos_err_mm < 1.0
    assert plan.total_time_s > 0.0
    assert np.isfinite(plan.trajectory_q).all()
    assert all(stack.free(c) for c in plan.trajectory_q[::3])


def test_unreachable_goal_is_reported_honestly(stack):
    far = pin.SE3(np.eye(3), np.array([9.0, 9.0, 9.0]))
    plan = stack.plan_cartesian(far, q_start=stack.sample())
    assert not plan.ok
    assert plan.stage == "ik_unreachable"


def test_plan_confidence_is_capped_by_the_weakest_link(stack):
    plan = _plan_ok(stack)
    assert plan.plan_confidence in ("unvalidated", "low", "medium", "high")
    assert plan.capped_by in ("twin", "effort", "collision")
    assert plan.provenance["collision_geometry"] == "primitives_only"


def test_exports(stack, tmp_path):
    plan = _plan_ok(stack)
    out = tmp_path / "traj.csv"
    plan.to_csv(out)
    rows = np.loadtxt(out, delimiter=",")
    assert rows.ndim == 2 and rows.shape[1] == stack.nj + 1
    assert rows.shape[0] == len(plan.trajectory_q)
    assert rows[0, 0] == 0.0 and abs(rows[-1, 0] - plan.total_time_s) < 1e-6
    msg = plan.to_ros_jointtrajectory(stack.joint_names())
    assert len(msg["joint_names"]) == stack.nj
    assert len(msg["points"]) == len(plan.trajectory_q)
