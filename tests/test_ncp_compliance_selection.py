"""Normal compliance can select different hard NCP velocities at identical rigid inputs.

Four positive-sliding contacts, exact geometry and Pythagorean initial speeds.
The limit equations retain the checkerboard compliance selector. These are
regressions, not a global convergence or material-validation assertion.
"""
import numpy as np
import pytest
from scipy.optimize import root
from motion_engine.ncp.ncp_ref import natural_residual


def fixture():
    J = np.vstack([
        [[0, 0, 1, y, -x, 0], [1, 0, 0, 0, -1, -y], [0, 1, 0, 1, 0, x]]
        for x, y in [(12, 5.5), (12, -5.5), (-12, 5.5), (-12, -5.5)]
    ])
    Mi = np.diag([1, 1, 1, 12/125, 3/145, 12/697])
    v = np.array([10.5, 0, 0, 0, 0, 1])
    u = (J @ v).reshape(4, 3)
    l = np.ones((4, 3))
    l[:, 1:] = -.5 * u[:, 1:] / np.linalg.norm(u[:, 1:], axis=1)[:, None]
    l = l.ravel()
    vf = v - Mi @ J.T @ l
    return J, Mi, vf, l


def selected(J, Mi, vf, seed, rho, eps):
    G = J @ Mi @ J.T
    b = J @ vf
    D = np.array([1 + rho, 1, 1, 1])
    E = np.zeros(12)
    E[::3] = eps * D
    H = G + np.diag(E)
    rows = [i for i in range(12) if i != 9]

    def residual(l):
        u = (G @ l + b).reshape(4, 3)
        f = np.empty((4, 3))
        f[:, 0] = u[:, 0] + eps * D * l[::3]
        f[:, 1:] = l.reshape(4, 3)[:, 1:] + .5 * l[::3, None] * (
            u[:, 1:] / np.linalg.norm(u[:, 1:], axis=1)[:, None]
        )
        return np.r_[f.ravel()[rows], np.array([1, -1, -1, 1]) @ (D * l[::3])]

    result = root(residual, seed, tol=1e-11)
    l = result.x
    assert np.max(np.abs(residual(l))) < 1e-10
    assert np.min(l[::3]) > .9
    assert natural_residual(l, H, b, np.full(4, .5)) < 1e-10
    perturbed = l.copy()
    perturbed[0] += .2
    assert natural_residual(perturbed, H, b, np.full(4, .5)) > 1e-3
    v = vf + Mi @ J.T @ l
    return l, v, float(v @ np.linalg.solve(Mi, v) / 2)


def test_rigid_velocity_and_energy_depend_on_compliance_ratio():
    J, Mi, vf, seed = fixture()
    l1, v1, e1 = selected(J, Mi, vf, seed, -.1, 0)
    l2, v2, e2 = selected(J, Mi, vf, seed, .1, 0)
    assert np.linalg.norm(v1-v2) > .015
    assert abs(e1-e2) > .004
    G, b = J @ Mi @ J.T, J @ vf
    for l in [l1, l2]:
        assert natural_residual(l, G, b, np.full(4, .5)) < 1e-10


@pytest.mark.parametrize('rho', [-.1, 0., .1])
def test_soft_branch_converges_to_its_selected_limit(rho):
    J, Mi, vf, seed = fixture()
    limit, vl, _ = selected(J, Mi, vf, seed, rho, 0)
    errors = []
    for eps in [1e-3, 1e-4, 1e-5, 1e-6]:
        _, v, _ = selected(J, Mi, vf, limit, rho, eps)
        errors.append(np.linalg.norm(v-vl))
    assert all(errors[i+1] < .11*errors[i] for i in range(3))
    assert errors[-1] < 1.1e-6
