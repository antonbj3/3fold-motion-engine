"""Stiffness-box corner values need not enclose NCP kinetic energy.

Four coplanar contacts on a rotating plate. The normal compliance selector
chooses one positive/sliding rigid solution for each ratio. This regression
uses the engine's actual residual; it asserts neither global uniqueness nor
material calibration. Exact local curvature: d²E/dd1² = 4095/654944 at d1=1.
"""
import numpy as np
from scipy.optimize import root
from motion_engine.ncp.ncp_ref import natural_residual


def selected_energy(d1):
    J = np.vstack([
        [[0, 0, 1, y, -x, 0], [1, 0, 0, 0, 0, -y], [0, 1, 0, 0, 0, x]]
        for x, y in [(12, 5.5), (12, -5.5), (-12, 5.5), (-12, -5.5)]
    ])
    Mi = np.diag([1, 1, 1, 12 / 125, 3 / 145, 12 / 697])
    vf = np.array([1519 / 130, 0, -4, 0, 0, 61343 / 45305])
    G, b = J @ Mi @ J.T, J @ vf
    v0 = np.array([10.5, 0, 0, 0, 0, 1])
    slip = (J @ v0).reshape(4, 3)[:, 1:]
    seed = np.column_stack([np.ones(4), -.5 * slip / np.linalg.norm(slip, axis=1)[:, None]]).ravel()
    D = np.array([d1, 1, 1, 1])
    rows = [i for i in range(12) if i != 9]

    def residual(lam):
        u = (G @ lam + b).reshape(4, 3)
        f = np.empty((4, 3))
        f[:, 0] = u[:, 0]
        f[:, 1:] = lam.reshape(4, 3)[:, 1:] + .5 * lam[::3, None] * (
            u[:, 1:] / np.linalg.norm(u[:, 1:], axis=1)[:, None]
        )
        return np.r_[f.ravel()[rows], np.array([1, -1, -1, 1]) @ (D * lam[::3])]

    lam = root(residual, seed, tol=1e-11).x
    assert np.max(np.abs(residual(lam))) < 1e-10
    assert np.min(lam[::3]) > .8
    assert natural_residual(lam, G, b, np.full(4, .5)) < 1e-10
    v = vf + Mi @ J.T @ lam
    return float(v @ np.linalg.solve(Mi, v) / 2)


def test_corner_energy_minimum_misses_interior_solution():
    center = selected_energy(1.)
    corners = [selected_energy(.9), selected_energy(1.1)]
    assert abs(center - 505 / 6) < 1e-10
    assert min(corners) - center > 2e-5


def test_refreshed_energy_curvature_matches_exact_reference():
    h = .01
    curvature = (selected_energy(1-h) - 2*selected_energy(1) + selected_energy(1+h)) / h**2
    assert abs(curvature - 4095/654944) < 2e-6
