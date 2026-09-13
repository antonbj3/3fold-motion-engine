"""Gripper primitives for the manipulation cells: open/close targets and the weld/release
transitions between a free body and the end effector.
"""
from __future__ import annotations

import numpy as np

from newton.solvers import SolverNotifyFlags


def qinv(q):  # xyzw
    out = q.copy(); out[..., :3] *= -1
    return out / (q * q).sum(-1, keepdims=True)


def qmul(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz], -1)


def qrot(q, v):
    uv = np.cross(q[..., :3], v) * 2.0
    return v + q[..., 3:4] * uv + np.cross(q[..., :3], uv)


def rel_transform(x1, x2):
    """relpose av body2 i body1-ram; x = [p(3), q_xyzw(4)]."""
    p1, q1 = x1[..., :3], x1[..., 3:7]
    p2, q2 = x2[..., :3], x2[..., 3:7]
    qi = qinv(q1)
    return np.concatenate([qrot(qi, p2 - p1), qmul(qi, q2)], -1)


class WeldLatch:
    """Touch-triggered grasp weld per world.

    The constructor reads the equality arrays BEFORE the control loop (the same point as in the
    `eq_en = model.equality_constraint_enabled.numpy()` osv). engage() tar
    world indices that just crossed the force threshold, plus X_ee/X_cube for ALL worlds
    (rel_transform is computed for the whole batch, then indexed).
    The measurement bookkeeping (grasped/grasp_step/pre_drift/snap) belongs to the experiment and
    ligger kvar i konsumenten.
    """

    def __init__(self, model, solver):
        self.model = model
        self.solver = solver
        self.eq_en = model.equality_constraint_enabled.numpy()   # (W,)
        self.eq_rp = model.equality_constraint_relpose.numpy()   # (W,7)

    def engage(self, new_idx, x_ee, x_cu):
        """Lock the weld for the worlds in new_idx: relpose = inv(X_ee) . X_cube;
        the grasp takes the cube WHERE IT STANDS. Returns the rel_transform batch."""
        rp = rel_transform(x_ee, x_cu)
        for wdx in new_idx:
            self.eq_rp[wdx] = rp[wdx]
            self.eq_en[wdx] = True
        self.model.equality_constraint_relpose.assign(self.eq_rp)
        self.model.equality_constraint_enabled.assign(self.eq_en)
        self.solver.notify_model_changed(SolverNotifyFlags.CONSTRAINT_PROPERTIES)
        return rp
