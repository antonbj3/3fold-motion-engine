"""M2 -- rigid-body frictional contact for ``mpm_core`` (SPEC_MPM.md).

Plugs into ``MPMSolver.grid_update(contacts=...)``: the hook runs after gravity
and before the domain boundary, and edits ``solver.grid_v`` in place.

--------------------------------------------------------------------------
WHAT IS HERE
--------------------------------------------------------------------------
* ``RigidBody`` -- plane / box / sphere as a signed distance field with a
  6-DOF state (position, quaternion, linear + angular velocity, mass,
  body-frame inertia, Coulomb friction vs the MPM material).
* ``RigidContact`` -- the grid hook.  Two formulations, switchable:

  (a) ``mode="baseline"``  per-node Coulomb projection (Stomakhin/Hu):
      separate if v_n > 0, else remove v_n and clamp the tangential part by
      mu |v_n|.  One pass, no iteration.

  (b) ``mode="ncp"``  per contacting node, the exact second-order Coulomb
      cone complementarity with the de Saxce correction

          K_mu  ∋  lam  ⊥  (u + Gamma(u))  ∈  K_mu*,
          Gamma(u) = (mu ||u_t||, 0, 0)

      exactly as in ``../ncp_ref.py`` (whose ``proj_cone`` / ``desaxce`` /
      ``natural_residual`` are imported and used for the CPU reference path
      and for the residual report; the Warp ``_proj_cone`` below is a
      line-by-line transcription of ``ncp_ref.proj_cone`` and is pinned to it
      by ``test_mpm_contact.py::test_warp_cone_projection_matches_ncp_ref``).
      Solved by **projected Jacobi over the contacting nodes**, node mass as
      the Delassus diagonal (G_ii = I/m_i), the off-diagonal coupling being
      the rigid body itself (G_ij = M^-1 I + r_i^x I_w^-1 r_j^x,T).
      Per-node natural-map residual is reported.

  (c) ``mode="ncp_coupled"`` (H14)  the same exact cone + de Saxce NCP, but
      preconditioned with the per-node 3x3 diagonal block of the **coupled**
      Delassus operator of the body,

          G = diag(1 / m_i)  +  J_b M_b^-1 J_b^T,
          J_i = [ I , -[r_i]_x ]   (3x6 rigid Jacobian at the node),

      i.e. every node in contact with a body is coupled to every other node
      of that body through the body's 6x6 inverse inertia.  The operator is
      never assembled: the off-diagonal action is applied matrix-free through
      the int64 sum of the current lambdas (``_body_delta_kernel``), and the
      sweep is a projected Jacobi with ``W_ii^-1`` as the step, W_ii being
      the exact diagonal block in the contact frame (``rho_mode="block"``) or
      that block with the off-diagonal row sum folded in as a safeguard
      (``rho_mode="block_rowsum"``).  ``mode="ncp"`` differs only in the
      preconditioner: a scalar row-sum bound instead of the 3x3 block.
      For a **prescribed** body M_b^-1 = 0, so W_ii = I/m_i and
      ncp_coupled == ncp == baseline, bit for bit.

  Difference to Menager & Carpentier, arXiv 2602.02038 (implicit MPM + NCP,
  ADMM): there the NCP is assembled against an **implicit** grid-velocity
  solve, so the Delassus operator carries the elastic Hessian.  Ours is the
  **explicit** (symplectic-Euler) MPM of ``mpm_core``: the grid velocity after
  gravity is already the free velocity, and the Delassus operator is just the
  lumped node mass plus the rigid-body inverse inertia.  The cone algebra and
  the de Saxce shift are identical; the operator is not.

  Analytic note (proved in the measurement record 8.5 and pinned by
  ``test_ncp_equals_baseline_for_prescribed_bodies``): for a **prescribed**
  (kinematic / one-way) body the Delassus operator is exactly diagonal,
  G = I/m_i, and the exact-cone NCP with de Saxce reduces **algebraically**
  to the baseline Coulomb projection -- same normal impulse -m u_n^free, same
  tangential clamp max(0, 1 + mu u_n/|u_t|).  The two formulations can only
  differ when the body is two-way coupled (the nodes then see each other).

* two-way coupling: node impulses summed back onto the body as force and
  torque with **int64 fixed-point accumulation** (``mpm_core.MOM_SCALE``), so
  the rigid feedback is bit-reproducible; 6-DOF rigid integration in float64
  on the CPU (a handful of bodies).  ``two_way=False`` gives the one-way
  (prescribed-motion) option.

* ``FrozenParticles`` -- the "contact bubble" of scene (v): particles outside
  a radius r of a body's SDF are frozen into a static heightfield.  They keep
  scattering **mass** to the grid but with zero velocity, zero affine matrix
  and F = I (hence zero stress), and their state is restored after every
  ``g2p`` -- i.e. mass on the grid, no momentum, no G2P.

Determinism: every kernel here is a pure per-node map (no float atomics);
the only reductions are int64 fixed point.  The rigid integration is float64
numpy.  Two processes give identical bytes -- see the measurement record 8.1.

Not imported from, not copied from, Newton.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import warp as wp

from .sand_core import MOM_SCALE, MPMSolver  # noqa: E402
from ..ncp.ncp_ref import (  # noqa: E402  (exact-cone projection + de Saxce)
    desaxce, natural_residual, proj_cone, proj_dual_cone, solve_ncp_pgs,
)

__all__ = [
    "RigidBody", "RigidContact", "FrozenParticles", "SpringDamperRing",
    "plane", "box", "sphere",
    "SHAPE_PLANE", "SHAPE_BOX", "SHAPE_SPHERE",
    "sdf_numpy", "solve_node_ncp_numpy", "baseline_node_projection_numpy",
]

SHAPE_PLANE = 0
SHAPE_BOX = 1
SHAPE_SPHERE = 2

_MOM_SCALE_WP = wp.constant(wp.float64(MOM_SCALE))
_DIAG_SCALE = wp.constant(wp.float64(1.0e9))   # diagnostics only
_EPS = 1.0e-12


# ==========================================================================
# quaternion helpers (xyzw, float64, CPU)
# ==========================================================================
def _quat_to_R(q: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(a) for a in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def _quat_integrate(q: np.ndarray, w: np.ndarray, dt: float) -> np.ndarray:
    dq = _quat_mul(np.array([w[0], w[1], w[2], 0.0], dtype=np.float64), q)
    qn = q + 0.5 * dt * dq
    n = np.linalg.norm(qn)
    return qn / n if n > 0 else np.array([0.0, 0.0, 0.0, 1.0])


def _quat_from_axis_angle(axis: Sequence[float], angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1e-300)
    s = math.sin(0.5 * angle)
    return np.array([a[0] * s, a[1] * s, a[2] * s, math.cos(0.5 * angle)])


# ==========================================================================
# rigid bodies
# ==========================================================================
@dataclass
class RigidBody:
    """A rigid SDF body.

    shape          SHAPE_PLANE (half-space y_local >= 0 is *outside*),
                   SHAPE_BOX (half extents ``h``),
                   SHAPE_SPHERE (radius ``h[0]``).
    x, q           world position of the body origin and orientation (xyzw).
    v, w           linear / angular velocity (world).
    mass, inertia  rigid mass [kg] and body-frame principal inertia (3,).
    mu             Coulomb friction vs the MPM material.
    kinematic      True  -> prescribed motion, contact impulses do not move it
                            (the "one-way" option of the spec)
                   False -> two-way, integrated from the summed node impulses.
    gravity        apply gravity to the body when dynamic.
    lock_rot       freeze the orientation (I^-1 := 0); the body still collects
                   torque for reporting.
    motion         optional callable t -> (v, w) prescribing the velocity of a
                   kinematic body each step.
    """

    shape: int
    h: np.ndarray = field(default_factory=lambda: np.zeros(3))
    x: np.ndarray = field(default_factory=lambda: np.zeros(3))
    q: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    w: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mass: float = 1.0
    inertia: np.ndarray = field(default_factory=lambda: np.ones(3))
    mu: float = 0.3
    kinematic: bool = True
    gravity: bool = False
    lock_rot: bool = False
    motion: Optional[Callable[[float], tuple]] = None
    name: str = ""

    def __post_init__(self):
        self.h = np.asarray(self.h, dtype=np.float64).reshape(3)
        self.x = np.asarray(self.x, dtype=np.float64).reshape(3)
        self.q = np.asarray(self.q, dtype=np.float64).reshape(4)
        self.v = np.asarray(self.v, dtype=np.float64).reshape(3)
        self.w = np.asarray(self.w, dtype=np.float64).reshape(3)
        self.inertia = np.asarray(self.inertia, dtype=np.float64).reshape(3)
        # per-step diagnostics filled by RigidContact.advance()
        self.force = np.zeros(3)
        self.torque = np.zeros(3)

    # -- inertia helpers ----------------------------------------------------
    @property
    def minv(self) -> float:
        return 0.0 if self.kinematic else 1.0 / self.mass

    def I_world_inv(self) -> np.ndarray:
        if self.kinematic or self.lock_rot:
            return np.zeros((3, 3))
        R = _quat_to_R(self.q)
        return R @ np.diag(1.0 / self.inertia) @ R.T

    def I_world(self) -> np.ndarray:
        R = _quat_to_R(self.q)
        return R @ np.diag(self.inertia) @ R.T

    def point_velocity(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return self.v[None, :] + np.cross(self.w[None, :], X - self.x[None, :])

    def state(self) -> np.ndarray:
        return np.concatenate([self.x, self.q, self.v, self.w])


def plane(point=(0.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0), **kw) -> RigidBody:
    """Half-space; ``normal`` points away from the solid (outward)."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    y = np.array([0.0, 1.0, 0.0])
    c = float(np.dot(y, n))
    if c > 1.0 - 1e-12:
        q = np.array([0.0, 0.0, 0.0, 1.0])
    elif c < -1.0 + 1e-12:
        q = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        ax = np.cross(y, n)
        q = _quat_from_axis_angle(ax, math.acos(max(-1.0, min(1.0, c))))
    return RigidBody(shape=SHAPE_PLANE, x=np.asarray(point, dtype=np.float64), q=q, **kw)


def box(center, half_extents, **kw) -> RigidBody:
    h = np.asarray(half_extents, dtype=np.float64)
    kw.setdefault("inertia", np.ones(3))
    b = RigidBody(shape=SHAPE_BOX, h=h, x=np.asarray(center, dtype=np.float64), **kw)
    if "inertia" not in kw or np.allclose(b.inertia, 1.0):
        m = b.mass
        hx, hy, hz = h
        b.inertia = (m / 3.0) * np.array([hy ** 2 + hz ** 2,
                                          hx ** 2 + hz ** 2,
                                          hx ** 2 + hy ** 2])
    return b


def sphere(center, radius, **kw) -> RigidBody:
    b = RigidBody(shape=SHAPE_SPHERE, h=np.array([radius, radius, radius]),
                  x=np.asarray(center, dtype=np.float64), **kw)
    if np.allclose(b.inertia, 1.0):
        b.inertia = np.full(3, 0.4 * b.mass * radius ** 2)
    return b


# --------------------------------------------------------------------------
# numpy SDF (penetration measurement, bubble masks, tests)
# --------------------------------------------------------------------------
def sdf_numpy(body: RigidBody, X: np.ndarray, with_normal: bool = False):
    """Signed distance (and outward normal) of world points ``X`` (N,3)."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    R = _quat_to_R(body.q)
    P = (X - body.x[None, :]) @ R            # world -> local  (R^T v == v @ R)
    if body.shape == SHAPE_PLANE:
        phi = P[:, 1].copy()
        g = np.zeros_like(P)
        g[:, 1] = 1.0
    elif body.shape == SHAPE_SPHERE:
        r = np.linalg.norm(P, axis=1)
        phi = r - body.h[0]
        g = P / np.maximum(r, 1e-300)[:, None]
        deg = r < 1e-12
        if np.any(deg):
            g[deg] = np.array([0.0, 1.0, 0.0])
    elif body.shape == SHAPE_BOX:
        h = body.h
        Q = np.abs(P) - h[None, :]
        Qp = np.maximum(Q, 0.0)
        lo = np.linalg.norm(Qp, axis=1)
        inner = np.minimum(Q.max(axis=1), 0.0)
        phi = lo + inner
        g = np.zeros_like(P)
        out = lo > 1e-12
        g[out] = Qp[out] / lo[out][:, None]
        if np.any(~out):
            k = np.argmax(Q[~out], axis=1)
            gi = np.zeros((int((~out).sum()), 3))
            gi[np.arange(len(k)), k] = 1.0
            g[~out] = gi
        g = g * np.where(P < 0.0, -1.0, 1.0)
    else:
        raise ValueError(f"unknown shape {body.shape}")
    if not with_normal:
        return phi
    return phi, g @ R.T                      # local -> world


# --------------------------------------------------------------------------
# CPU reference solvers (used by the tests and by mode="ncp_cpu")
# --------------------------------------------------------------------------
def baseline_node_projection_numpy(u_free: np.ndarray, mu: float) -> np.ndarray:
    """Baseline Coulomb projection of one node's relative velocity."""
    u = np.asarray(u_free, dtype=np.float64).copy()
    if u[0] >= 0.0:
        return u
    ut = u[1:3]
    lt = np.linalg.norm(ut)
    s = max(0.0, 1.0 + mu * u[0] / lt) if lt > 1e-12 else 0.0
    return np.array([0.0, ut[0] * s, ut[1] * s])


def solve_node_ncp_numpy(u_free: np.ndarray, m: float, mu: float,
                         iters: int = 400, tol: float = 1e-14,
                         method: str = "exact"):
    """Single-node NCP with the Delassus diagonal G = I/m, exact cone + de
    Saxce.  Returns (lam, u, residual).

    ``method="exact"``  -> ``ncp_ref.solve_ncp_pgs`` (its exact 3x3 local case
                           analysis: take-off / stick / slip root find).
    ``method="jacobi"`` -> the same projected iteration the GPU kernel runs,
                           lam <- proj_K(lam - rho (u + Gamma(u))), rho = m.
                           Not always a contraction -- see the measurement record 8.5."""
    u_free = np.asarray(u_free, dtype=np.float64).reshape(3)
    G = np.eye(3) / m
    mu_v = np.array([float(mu)])
    if method == "exact":
        lam, _, _ = solve_ncp_pgs(G, u_free.copy(), mu_v, iters=iters, tol=tol)
        lam = np.asarray(lam, dtype=np.float64).reshape(3)
    else:
        lam = np.zeros(3)
        for _ in range(iters):
            u = u_free + lam / m
            new = proj_cone(lam - m * (u + desaxce(u, mu_v)), mu)
            if np.max(np.abs(new - lam)) < tol:
                lam = new
                break
            lam = new
    u = u_free + lam / m
    return lam, u, natural_residual(lam, G, u_free, mu_v)


# ==========================================================================
# Warp: SDF, cone algebra
# ==========================================================================
@wp.func
def _sdf_local(bt: int, p: wp.vec3, h: wp.vec3):
    """Signed distance + outward gradient of a body, in the body frame."""
    phi = float(0.0)
    g = wp.vec3(0.0, 1.0, 0.0)
    if bt == 0:                                            # plane
        phi = p[1]
        g = wp.vec3(0.0, 1.0, 0.0)
    elif bt == 2:                                          # sphere
        r = wp.length(p)
        phi = r - h[0]
        if r > 1.0e-12:
            g = p / r
    else:                                                  # box
        qx = wp.abs(p[0]) - h[0]
        qy = wp.abs(p[1]) - h[1]
        qz = wp.abs(p[2]) - h[2]
        qo = wp.vec3(wp.max(qx, 0.0), wp.max(qy, 0.0), wp.max(qz, 0.0))
        lo = wp.length(qo)
        phi = lo + wp.min(wp.max(qx, wp.max(qy, qz)), 0.0)
        gl = wp.vec3(0.0, 1.0, 0.0)
        if lo > 1.0e-12:
            gl = qo / lo
        else:
            if qx >= qy and qx >= qz:
                gl = wp.vec3(1.0, 0.0, 0.0)
            elif qy >= qz:
                gl = wp.vec3(0.0, 1.0, 0.0)
            else:
                gl = wp.vec3(0.0, 0.0, 1.0)
        sx = 1.0
        if p[0] < 0.0:
            sx = -1.0
        sy = 1.0
        if p[1] < 0.0:
            sy = -1.0
        sz = 1.0
        if p[2] < 0.0:
            sz = -1.0
        g = wp.vec3(gl[0] * sx, gl[1] * sy, gl[2] * sz)
    return phi, g


@wp.func
def _tangent_frame(n: wp.vec3):
    """Deterministic orthonormal tangents of ``n``."""
    a = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(n[0]) > 0.5773502691896258:
        a = wp.vec3(0.0, 1.0, 0.0)
    t1 = wp.normalize(wp.cross(n, a))
    t2 = wp.cross(n, t1)
    return t1, t2


@wp.func
def _proj_cone(x: wp.vec3, mu: float) -> wp.vec3:
    """Euclidean projection onto K_mu = {||x_t|| <= mu x_n}.

    Transcription of ``ncp_ref.proj_cone`` (same three branches, same
    formulas); pinned to it by the test suite."""
    if mu <= 0.0:
        return wp.vec3(wp.max(0.0, x[0]), 0.0, 0.0)
    nt = wp.sqrt(x[1] * x[1] + x[2] * x[2])
    if nt <= mu * x[0]:                       # inside K
        return x
    if mu * nt <= -x[0]:                      # inside the polar cone
        return wp.vec3(0.0, 0.0, 0.0)
    a = (mu * nt + x[0]) / (1.0 + mu * mu)
    s = mu * a / nt
    return wp.vec3(a, s * x[1], s * x[2])


@wp.func
def _node_pos(gi: int, res: int, dx: float, origin: wp.vec3) -> wp.vec3:
    iz = gi % res
    iy = (gi // res) % res
    ix = gi // (res * res)
    return origin + wp.vec3(float(ix) * dx, float(iy) * dx, float(iz) * dx)


@wp.func
def _acc6(acc: wp.array2d(dtype=wp.int64), b: int, f: wp.vec3, t: wp.vec3):
    wp.atomic_add(acc, b, 0, wp.int64(wp.round(wp.float64(f[0]) * _MOM_SCALE_WP)))
    wp.atomic_add(acc, b, 1, wp.int64(wp.round(wp.float64(f[1]) * _MOM_SCALE_WP)))
    wp.atomic_add(acc, b, 2, wp.int64(wp.round(wp.float64(f[2]) * _MOM_SCALE_WP)))
    wp.atomic_add(acc, b, 3, wp.int64(wp.round(wp.float64(t[0]) * _MOM_SCALE_WP)))
    wp.atomic_add(acc, b, 4, wp.int64(wp.round(wp.float64(t[1]) * _MOM_SCALE_WP)))
    wp.atomic_add(acc, b, 5, wp.int64(wp.round(wp.float64(t[2]) * _MOM_SCALE_WP)))


# ==========================================================================
# Warp kernels
# ==========================================================================
@wp.kernel
def _detect_kernel(
    grid_m: wp.array(dtype=wp.float32),
    grid_v: wp.array(dtype=wp.vec3),
    b_type: wp.array(dtype=wp.int32),
    b_x: wp.array(dtype=wp.vec3),
    b_q: wp.array(dtype=wp.quat),
    b_v: wp.array(dtype=wp.vec3),
    b_w: wp.array(dtype=wp.vec3),
    b_h: wp.array(dtype=wp.vec3),
    n_bodies: int, res: int, dx: float, origin: wp.vec3,
    band: float, dt: float, speculative: int,
    c_body: wp.array(dtype=wp.int32),
    c_n: wp.array(dtype=wp.vec3),
    c_phi: wp.array(dtype=wp.float32),
    c_lam: wp.array(dtype=wp.vec3),
    v_free: wp.array(dtype=wp.vec3),
    b_count: wp.array(dtype=wp.int32),
):
    gi = wp.tid()
    c_body[gi] = -1
    c_lam[gi] = wp.vec3(0.0, 0.0, 0.0)
    if grid_m[gi] <= 0.0:
        return
    xw = _node_pos(gi, res, dx, origin)
    vg = grid_v[gi]
    v_free[gi] = vg
    best = float(1.0e30)
    bi = int(-1)
    bn = wp.vec3(0.0, 1.0, 0.0)
    for b in range(n_bodies):
        pl = wp.quat_rotate_inv(b_q[b], xw - b_x[b])
        phi, gl = _sdf_local(b_type[b], pl, b_h[b])
        nw = wp.quat_rotate(b_q[b], gl)
        hit = int(0)
        if phi <= band:
            hit = 1
        elif speculative == 1:
            vb = b_v[b] + wp.cross(b_w[b], xw - b_x[b])
            vn = wp.dot(vg - vb, nw)
            if vn < 0.0 and phi + dt * vn <= band:
                hit = 1
        if hit == 1 and phi < best:
            best = phi
            bi = b
            bn = nw
    if bi >= 0:
        c_body[gi] = bi
        c_n[gi] = bn
        c_phi[gi] = best
        wp.atomic_add(b_count, bi, 1)


@wp.kernel
def _baseline_kernel(
    grid_m: wp.array(dtype=wp.float32),
    grid_v: wp.array(dtype=wp.vec3),
    v_free: wp.array(dtype=wp.vec3),
    c_body: wp.array(dtype=wp.int32),
    c_n: wp.array(dtype=wp.vec3),
    b_x: wp.array(dtype=wp.vec3),
    b_v: wp.array(dtype=wp.vec3),
    b_w: wp.array(dtype=wp.vec3),
    b_mu: wp.array(dtype=wp.float32),
    res: int, dx: float, origin: wp.vec3,
    imp: wp.array2d(dtype=wp.int64),
):
    gi = wp.tid()
    b = c_body[gi]
    if b < 0:
        return
    m = grid_m[gi]
    xw = _node_pos(gi, res, dx, origin)
    r = xw - b_x[b]
    vb = b_v[b] + wp.cross(b_w[b], r)
    n = c_n[gi]
    vr = v_free[gi] - vb
    vn = wp.dot(vr, n)
    if vn >= 0.0:                              # separating: leave it alone
        return
    vt = vr - vn * n
    lt = wp.length(vt)
    s = float(0.0)
    if lt > 1.0e-12:
        s = wp.max(0.0, 1.0 + b_mu[b] * vn / lt)
    vnew = vb + s * vt
    grid_v[gi] = vnew
    J = m * (vnew - v_free[gi])                # impulse applied to the material
    _acc6(imp, b, -J, -wp.cross(r, J))         # reaction on the rigid body


@wp.kernel
def _ncp_iter_kernel(
    grid_m: wp.array(dtype=wp.float32),
    v_free: wp.array(dtype=wp.vec3),
    c_body: wp.array(dtype=wp.int32),
    c_n: wp.array(dtype=wp.vec3),
    c_lam: wp.array(dtype=wp.vec3),
    b_x: wp.array(dtype=wp.vec3),
    b_v: wp.array(dtype=wp.vec3),
    b_w: wp.array(dtype=wp.vec3),
    b_mu: wp.array(dtype=wp.float32),
    b_minv: wp.array(dtype=wp.float32),
    b_iinv_max: wp.array(dtype=wp.float32),
    b_count: wp.array(dtype=wp.int32),
    b_dv: wp.array(dtype=wp.vec3),
    b_dw: wp.array(dtype=wp.vec3),
    res: int, dx: float, origin: wp.vec3,
    rho_mode: int,
    jac: wp.array2d(dtype=wp.int64),
):
    gi = wp.tid()
    b = c_body[gi]
    if b < 0:
        return
    m = grid_m[gi]
    xw = _node_pos(gi, res, dx, origin)
    r = xw - b_x[b]
    vb = (b_v[b] + b_dv[b]) + wp.cross(b_w[b] + b_dw[b], r)
    n = c_n[gi]
    t1, t2 = _tangent_frame(n)
    lam = c_lam[gi]
    lw = lam[0] * n + lam[1] * t1 + lam[2] * t2
    ur = (v_free[gi] + lw / m) - vb
    u = wp.vec3(wp.dot(ur, n), wp.dot(ur, t1), wp.dot(ur, t2))
    mu = b_mu[b]
    # de Saxce shift Gamma(u) = (mu ||u_t||, 0, 0)
    g = mu * wp.sqrt(u[1] * u[1] + u[2] * u[2])
    # Delassus diagonal: node mass.  rho_mode 1 adds the rigid-body row sum
    # so that the Jacobi sweep stays a contraction when the body is light.
    gii = 1.0 / m
    if rho_mode == 1:
        gii = gii + float(b_count[b]) * (b_minv[b]
                                         + wp.dot(r, r) * b_iinv_max[b])
    rho = 1.0 / gii
    lam_new = _proj_cone(lam - rho * (u + wp.vec3(g, 0.0, 0.0)), mu)
    c_lam[gi] = lam_new
    lw2 = lam_new[0] * n + lam_new[1] * t1 + lam_new[2] * t2
    _acc6(jac, b, lw2, wp.cross(r, lw2))


@wp.func
def _skew(r: wp.vec3) -> wp.mat33:
    return wp.mat33(0.0, -r[2], r[1],
                    r[2], 0.0, -r[0],
                    -r[1], r[0], 0.0)


@wp.func
def _coupled_block(m: float, r: wp.vec3, minv: float, iinv: wp.mat33,
                   n: wp.vec3, t1: wp.vec3, t2: wp.vec3,
                   extra: float) -> wp.mat33:
    """Diagonal 3x3 block of the COUPLED Delassus operator, contact frame.

        G_ii = I / m_i  +  J_i M_b^-1 J_i^T,
        J_i  = [ I , -[r_i]_x ]   (body twist -> point velocity)
        J_i M^-1 J_i^T = (1/M) I - [r_i]_x I_w^-1 [r_i]_x

    ``extra`` scales an extra copy of the body part, used as the row-sum
    safeguard (extra = n_c - 1 bounds the off-diagonal row sum by the block
    itself; extra = 0 is the exact diagonal block)."""
    Rx = _skew(r)
    body = minv * wp.identity(n=3, dtype=float) - Rx * iinv * Rx
    Gw = (1.0 / m) * wp.identity(n=3, dtype=float) + (1.0 + extra) * body
    B = wp.mat33(n[0], t1[0], t2[0],
                 n[1], t1[1], t2[1],
                 n[2], t1[2], t2[2])
    return wp.transpose(B) * Gw * B


@wp.kernel
def _ncp_coupled_iter_kernel(
    grid_m: wp.array(dtype=wp.float32),
    v_free: wp.array(dtype=wp.vec3),
    c_body: wp.array(dtype=wp.int32),
    c_n: wp.array(dtype=wp.vec3),
    c_lam: wp.array(dtype=wp.vec3),
    b_x: wp.array(dtype=wp.vec3),
    b_v: wp.array(dtype=wp.vec3),
    b_w: wp.array(dtype=wp.vec3),
    b_mu: wp.array(dtype=wp.float32),
    b_minv: wp.array(dtype=wp.float32),
    b_iinv: wp.array(dtype=wp.mat33),
    b_count: wp.array(dtype=wp.int32),
    b_dv: wp.array(dtype=wp.vec3),
    b_dw: wp.array(dtype=wp.vec3),
    res: int, dx: float, origin: wp.vec3,
    block_mode: int, omega: float,
    jac: wp.array2d(dtype=wp.int64),
):
    """One matrix-free projected-Jacobi sweep of the node-level NCP with the
    per-node 3x3 block of the COUPLED Delassus operator as preconditioner.

    The off-diagonal action (every node of a body seeing every other node
    through the body's 6x6 inverse inertia) is applied matrix-free through
    ``b_dv`` / ``b_dw``, which ``_body_delta_kernel`` recomputes from the
    int64 sum of the current lambdas after every sweep -- i.e. the operator
    really is G = diag(1/m_i) + J_b M_b^-1 J_b^T, never assembled."""
    gi = wp.tid()
    b = c_body[gi]
    if b < 0:
        return
    m = grid_m[gi]
    xw = _node_pos(gi, res, dx, origin)
    r = xw - b_x[b]
    vb = (b_v[b] + b_dv[b]) + wp.cross(b_w[b] + b_dw[b], r)
    n = c_n[gi]
    t1, t2 = _tangent_frame(n)
    lam = c_lam[gi]
    lw = lam[0] * n + lam[1] * t1 + lam[2] * t2
    ur = (v_free[gi] + lw / m) - vb
    u = wp.vec3(wp.dot(ur, n), wp.dot(ur, t1), wp.dot(ur, t2))
    mu = b_mu[b]
    g = mu * wp.sqrt(u[1] * u[1] + u[2] * u[2])
    extra = float(0.0)
    if block_mode == 1:
        extra = float(wp.max(b_count[b] - 1, 0))
    W = _coupled_block(m, r, b_minv[b], b_iinv[b], n, t1, t2, extra)
    step = wp.inverse(W) * (u + wp.vec3(g, 0.0, 0.0))
    lam_new = _proj_cone(lam - omega * step, mu)
    c_lam[gi] = lam_new
    lw2 = lam_new[0] * n + lam_new[1] * t1 + lam_new[2] * t2
    _acc6(jac, b, lw2, wp.cross(r, lw2))


@wp.kernel
def _diag_kernel(
    grid_m: wp.array(dtype=wp.float32),
    grid_v: wp.array(dtype=wp.vec3),
    c_body: wp.array(dtype=wp.int32),
    c_n: wp.array(dtype=wp.vec3),
    b_x: wp.array(dtype=wp.vec3),
    b_v: wp.array(dtype=wp.vec3),
    b_w: wp.array(dtype=wp.vec3),
    b_dv: wp.array(dtype=wp.vec3),
    b_dw: wp.array(dtype=wp.vec3),
    res: int, dx: float, origin: wp.vec3,
    acc: wp.array(dtype=wp.int64),
):
    """Post-solve contact diagnostics, int64 fixed point (scale 1e9):
    0 = sum |u_t| (tangential slip speed), 1 = sum max(0, -u_n) (residual
    approach speed), 2 = sum of the contacting node masses, 3 = node count."""
    gi = wp.tid()
    b = c_body[gi]
    if b < 0:
        return
    xw = _node_pos(gi, res, dx, origin)
    r = xw - b_x[b]
    vb = (b_v[b] + b_dv[b]) + wp.cross(b_w[b] + b_dw[b], r)
    n = c_n[gi]
    ur = grid_v[gi] - vb
    un = wp.dot(ur, n)
    ut = wp.length(ur - un * n)
    wp.atomic_add(acc, 0, wp.int64(wp.round(wp.float64(ut) * _DIAG_SCALE)))
    wp.atomic_add(acc, 1,
                  wp.int64(wp.round(wp.float64(wp.max(-un, 0.0)) * _DIAG_SCALE)))
    wp.atomic_add(acc, 2, wp.int64(wp.round(wp.float64(grid_m[gi]) * _DIAG_SCALE)))
    wp.atomic_add(acc, 3, wp.int64(1))


@wp.kernel
def _body_delta_kernel(
    jac: wp.array2d(dtype=wp.int64),
    b_minv: wp.array(dtype=wp.float32),
    b_iinv: wp.array(dtype=wp.mat33),
    b_dv: wp.array(dtype=wp.vec3),
    b_dw: wp.array(dtype=wp.vec3),
):
    b = wp.tid()
    P = wp.vec3(wp.float32(wp.float64(jac[b, 0]) / _MOM_SCALE_WP),
                wp.float32(wp.float64(jac[b, 1]) / _MOM_SCALE_WP),
                wp.float32(wp.float64(jac[b, 2]) / _MOM_SCALE_WP))
    L = wp.vec3(wp.float32(wp.float64(jac[b, 3]) / _MOM_SCALE_WP),
                wp.float32(wp.float64(jac[b, 4]) / _MOM_SCALE_WP),
                wp.float32(wp.float64(jac[b, 5]) / _MOM_SCALE_WP))
    # reaction on the body is -sum(lam); its velocity change is the response
    b_dv[b] = -b_minv[b] * P
    b_dw[b] = -(b_iinv[b] * L)


@wp.kernel
def _ncp_finalize_kernel(
    grid_m: wp.array(dtype=wp.float32),
    grid_v: wp.array(dtype=wp.vec3),
    v_free: wp.array(dtype=wp.vec3),
    c_body: wp.array(dtype=wp.int32),
    c_n: wp.array(dtype=wp.vec3),
    c_lam: wp.array(dtype=wp.vec3),
    b_x: wp.array(dtype=wp.vec3),
    res: int, dx: float, origin: wp.vec3,
    imp: wp.array2d(dtype=wp.int64),
):
    gi = wp.tid()
    b = c_body[gi]
    if b < 0:
        return
    m = grid_m[gi]
    xw = _node_pos(gi, res, dx, origin)
    r = xw - b_x[b]
    n = c_n[gi]
    t1, t2 = _tangent_frame(n)
    lam = c_lam[gi]
    lw = lam[0] * n + lam[1] * t1 + lam[2] * t2
    grid_v[gi] = v_free[gi] + lw / m
    _acc6(imp, b, -lw, -wp.cross(r, lw))


@wp.kernel
def _ncp_residual_kernel(
    grid_m: wp.array(dtype=wp.float32),
    v_free: wp.array(dtype=wp.vec3),
    c_body: wp.array(dtype=wp.int32),
    c_n: wp.array(dtype=wp.vec3),
    c_lam: wp.array(dtype=wp.vec3),
    b_x: wp.array(dtype=wp.vec3),
    b_v: wp.array(dtype=wp.vec3),
    b_w: wp.array(dtype=wp.vec3),
    b_mu: wp.array(dtype=wp.float32),
    b_dv: wp.array(dtype=wp.vec3),
    b_dw: wp.array(dtype=wp.vec3),
    res: int, dx: float, origin: wp.vec3,
    out_res: wp.array(dtype=wp.float32),
    out_lam: wp.array(dtype=wp.float32),
):
    gi = wp.tid()
    out_res[gi] = 0.0
    out_lam[gi] = 0.0
    b = c_body[gi]
    if b < 0:
        return
    m = grid_m[gi]
    xw = _node_pos(gi, res, dx, origin)
    r = xw - b_x[b]
    vb = (b_v[b] + b_dv[b]) + wp.cross(b_w[b] + b_dw[b], r)
    n = c_n[gi]
    t1, t2 = _tangent_frame(n)
    lam = c_lam[gi]
    lw = lam[0] * n + lam[1] * t1 + lam[2] * t2
    ur = (v_free[gi] + lw / m) - vb
    u = wp.vec3(wp.dot(ur, n), wp.dot(ur, t1), wp.dot(ur, t2))
    mu = b_mu[b]
    g = mu * wp.sqrt(u[1] * u[1] + u[2] * u[2])
    p = _proj_cone(lam - m * (u + wp.vec3(g, 0.0, 0.0)), mu)
    d = lam - p
    out_res[gi] = wp.max(wp.abs(d[0]), wp.max(wp.abs(d[1]), wp.abs(d[2])))
    out_lam[gi] = wp.max(wp.abs(lam[0]), wp.max(wp.abs(lam[1]), wp.abs(lam[2])))


@wp.kernel
def _freeze_kernel(
    idx: wp.array(dtype=wp.int32),
    x0: wp.array(dtype=wp.vec3),
    jp0: wp.array(dtype=wp.float32),
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    C: wp.array(dtype=wp.mat33),
    F: wp.array(dtype=wp.mat33),
    Jp: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    p = idx[i]
    x[p] = x0[i]
    v[p] = wp.vec3(0.0, 0.0, 0.0)
    C[p] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    F[p] = wp.identity(n=3, dtype=float)
    Jp[p] = jp0[i]


# ==========================================================================
# the hook
# ==========================================================================
class RigidContact:
    """``grid_update(contacts=...)`` hook for rigid frictional contact.

    mode        "baseline" | "ncp" | "ncp_coupled"
    iters       projected-Jacobi iterations (NCP modes)
    rho_mode    "node_mass" -- rho_i = m_i, the literal Delassus diagonal
                "jacobi"    -- rho_i = 1/(1/m_i + n_c (1/M + |r|^2 lmax(I^-1)))
                               (row-sum bound; identical to "node_mass" for a
                               prescribed body, where 1/M = I^-1 = 0)
                "block"     -- (ncp_coupled) exact 3x3 diagonal block of the
                               coupled Delassus operator, contact frame
                "block_rowsum" -- the same block plus (n_c - 1) copies of its
                               body part, a diagonal-dominance safeguard
    omega       relaxation on the coupled sweep (1.0 = plain block Jacobi)
    record_diag accumulate mean |u_t| (slip) and mean max(0, -u_n) at the
                contacting nodes; see diag_reset() / read_diag()
    band        a node is in contact where sdf <= band  (default 0 = inside)
    speculative also take a node whose sdf + dt v_n <= band (v_n < 0)
    two_way     False forces every body kinematic (prescribed motion)
    """

    def __init__(self, bodies: Sequence[RigidBody], mode: str = "baseline",
                 iters: int = 20, rho_mode: str = "jacobi",
                 band: float = 0.0, speculative: bool = False,
                 two_way: bool = True, record_residual: bool = False,
                 record_diag: bool = False, omega: float = 1.0,
                 device: str = "cuda:0"):
        if mode not in ("baseline", "ncp", "ncp_coupled"):
            raise ValueError(
                "mode must be 'baseline', 'ncp' or 'ncp_coupled'")
        if rho_mode not in ("node_mass", "jacobi", "block", "block_rowsum"):
            raise ValueError("rho_mode must be 'node_mass', 'jacobi', "
                             "'block' or 'block_rowsum'")
        if mode == "ncp_coupled" and rho_mode not in ("block", "block_rowsum"):
            rho_mode = "block"
        self.bodies = list(bodies)
        self.mode = mode
        self.iters = int(iters)
        self.rho_mode_name = rho_mode
        self.rho_mode = 1 if rho_mode == "jacobi" else 0
        self.block_mode = 1 if rho_mode == "block_rowsum" else 0
        self.omega = float(omega)
        self.band = float(band)
        self.speculative = 1 if speculative else 0
        self.two_way = bool(two_way)
        self.record_residual = bool(record_residual)
        self.record_diag = bool(record_diag)
        self.device = device
        self.t = 0.0
        self.n_contacts = 0
        self.residual = float("nan")
        self.residual_rel = float("nan")
        self.gravity = np.array([0.0, -9.81, 0.0])
        if not self.two_way:
            for b in self.bodies:
                b.kinematic = True
        self._alloc = False
        self._nb = len(self.bodies)

    # -- allocation ---------------------------------------------------------
    def _allocate(self, solver: MPMSolver) -> None:
        nb, nn = self._nb, solver.n_nodes
        with wp.ScopedDevice(self.device):
            self.b_type = wp.array(np.array([b.shape for b in self.bodies],
                                            dtype=np.int32), dtype=wp.int32)
            self.b_h = wp.array(np.array([b.h for b in self.bodies],
                                         dtype=np.float32), dtype=wp.vec3)
            self.b_x = wp.zeros(nb, dtype=wp.vec3)
            self.b_q = wp.zeros(nb, dtype=wp.quat)
            self.b_v = wp.zeros(nb, dtype=wp.vec3)
            self.b_w = wp.zeros(nb, dtype=wp.vec3)
            self.b_mu = wp.array(np.array([b.mu for b in self.bodies],
                                          dtype=np.float32), dtype=wp.float32)
            self.b_minv = wp.zeros(nb, dtype=wp.float32)
            self.b_iinv = wp.zeros(nb, dtype=wp.mat33)
            self.b_iinv_max = wp.zeros(nb, dtype=wp.float32)
            self.b_count = wp.zeros(nb, dtype=wp.int32)
            self.b_dv = wp.zeros(nb, dtype=wp.vec3)
            self.b_dw = wp.zeros(nb, dtype=wp.vec3)
            self.imp = wp.zeros((nb, 6), dtype=wp.int64)
            self.jac = wp.zeros((nb, 6), dtype=wp.int64)
            self.c_body = wp.zeros(nn, dtype=wp.int32)
            self.c_n = wp.zeros(nn, dtype=wp.vec3)
            self.c_phi = wp.zeros(nn, dtype=wp.float32)
            self.c_lam = wp.zeros(nn, dtype=wp.vec3)
            self.v_free = wp.zeros(nn, dtype=wp.vec3)
            self.r_res = wp.zeros(nn, dtype=wp.float32)
            self.r_lam = wp.zeros(nn, dtype=wp.float32)
            self.diag = wp.zeros(4, dtype=wp.int64)
        self._alloc = True
        self._upload_bodies()

    def sync_bodies(self) -> None:
        """Push a hand-edited body state (e.g. a velocity set from a scene
        driver between phases) to the device before the next step."""
        self._upload_bodies()

    def _upload_bodies(self) -> None:
        if not self._alloc:
            return
        self.b_x.assign(np.array([b.x for b in self.bodies], dtype=np.float32))
        self.b_q.assign(np.array([b.q for b in self.bodies], dtype=np.float32))
        self.b_v.assign(np.array([b.v for b in self.bodies], dtype=np.float32))
        self.b_w.assign(np.array([b.w for b in self.bodies], dtype=np.float32))
        self.b_minv.assign(np.array([b.minv for b in self.bodies], dtype=np.float32))
        Iinv = np.array([b.I_world_inv() for b in self.bodies], dtype=np.float32)
        self.b_iinv.assign(Iinv)
        self.b_iinv_max.assign(np.array(
            [float(np.max(np.abs(np.linalg.eigvalsh(b.I_world_inv()))))
             if not (b.kinematic or b.lock_rot) else 0.0
             for b in self.bodies], dtype=np.float32))

    # -- the hook -----------------------------------------------------------
    def apply_grid(self, solver: MPMSolver, dt: float) -> None:
        if not self._alloc:
            self._allocate(solver)
        nn = solver.n_nodes
        with wp.ScopedDevice(self.device):
            self.imp.zero_()
            self.b_count.zero_()
            self.b_dv.zero_()
            self.b_dw.zero_()
            wp.launch(_detect_kernel, dim=nn, inputs=[
                solver.grid_m, solver.grid_v, self.b_type, self.b_x, self.b_q,
                self.b_v, self.b_w, self.b_h, self._nb, solver.res, solver.dx,
                solver.origin, self.band, dt, self.speculative,
                self.c_body, self.c_n, self.c_phi, self.c_lam, self.v_free,
                self.b_count])
            if self.mode == "baseline":
                wp.launch(_baseline_kernel, dim=nn, inputs=[
                    solver.grid_m, solver.grid_v, self.v_free, self.c_body,
                    self.c_n, self.b_x, self.b_v, self.b_w, self.b_mu,
                    solver.res, solver.dx, solver.origin, self.imp])
            else:
                for _ in range(self.iters):
                    self.jac.zero_()
                    if self.mode == "ncp_coupled":
                        wp.launch(_ncp_coupled_iter_kernel, dim=nn, inputs=[
                            solver.grid_m, self.v_free, self.c_body, self.c_n,
                            self.c_lam, self.b_x, self.b_v, self.b_w,
                            self.b_mu, self.b_minv, self.b_iinv, self.b_count,
                            self.b_dv, self.b_dw, solver.res, solver.dx,
                            solver.origin, self.block_mode, self.omega,
                            self.jac])
                    else:
                        wp.launch(_ncp_iter_kernel, dim=nn, inputs=[
                            solver.grid_m, self.v_free, self.c_body, self.c_n,
                            self.c_lam, self.b_x, self.b_v, self.b_w,
                            self.b_mu, self.b_minv, self.b_iinv_max,
                            self.b_count, self.b_dv, self.b_dw, solver.res,
                            solver.dx, solver.origin, self.rho_mode, self.jac])
                    wp.launch(_body_delta_kernel, dim=self._nb, inputs=[
                        self.jac, self.b_minv, self.b_iinv,
                        self.b_dv, self.b_dw])
                if self.record_residual:
                    wp.launch(_ncp_residual_kernel, dim=nn, inputs=[
                        solver.grid_m, self.v_free, self.c_body, self.c_n,
                        self.c_lam, self.b_x, self.b_v, self.b_w, self.b_mu,
                        self.b_dv, self.b_dw, solver.res, solver.dx,
                        solver.origin, self.r_res, self.r_lam])
                    rr = self.r_res.numpy()
                    rl = self.r_lam.numpy()
                    self.residual = float(rr.max())
                    denom = float(rl.max())
                    self.residual_rel = (self.residual / denom
                                         if denom > 0 else 0.0)
                wp.launch(_ncp_finalize_kernel, dim=nn, inputs=[
                    solver.grid_m, solver.grid_v, self.v_free, self.c_body,
                    self.c_n, self.c_lam, self.b_x, solver.res, solver.dx,
                    solver.origin, self.imp])
            if self.record_diag:
                wp.launch(_diag_kernel, dim=nn, inputs=[
                    solver.grid_m, solver.grid_v, self.c_body, self.c_n,
                    self.b_x, self.b_v, self.b_w, self.b_dv, self.b_dw,
                    solver.res, solver.dx, solver.origin, self.diag])

    # -- contact diagnostics (int64, deterministic) -------------------------
    def diag_reset(self) -> None:
        """Zero the slip / approach accumulators (call at the start of the
        window you want the averages over)."""
        if self._alloc:
            self.diag.zero_()

    def read_diag(self) -> dict:
        """Averages over the accumulated node-steps since ``diag_reset``:
        ``slip`` = mean |u_t| [m/s], ``approach`` = mean max(0, -u_n) [m/s],
        ``node_mass`` = mean contacting node mass [kg], ``n`` = node-steps."""
        if not self._alloc:
            return {"slip": float("nan"), "approach": float("nan"),
                    "node_mass": float("nan"), "n": 0}
        a = self.diag.numpy().astype(np.float64)
        n = max(a[3], 1.0)
        return {"slip": a[0] / 1.0e9 / n, "approach": a[1] / 1.0e9 / n,
                "node_mass": a[2] / 1.0e9 / n, "n": int(a[3])}

    # -- rigid integration --------------------------------------------------
    def read_impulses(self) -> np.ndarray:
        """(nb, 6) float64 -- linear and angular impulse on each body [kg m/s]."""
        return self.imp.numpy().astype(np.float64) / MOM_SCALE

    def advance(self, dt: float) -> None:
        """Integrate the rigid 6-DOF state from the accumulated node impulses.

        Works before the first grid pass too (no impulses yet), so a body can
        be driven by prescribed motion alone."""
        if self._alloc:
            P = self.read_impulses()
            self.n_contacts = int(self.b_count.numpy().sum())
        else:
            P = np.zeros((self._nb, 6))
            self.n_contacts = 0
        for i, b in enumerate(self.bodies):
            lin, ang = P[i, 0:3], P[i, 3:6]
            b.force = lin / dt
            b.torque = ang / dt
            if b.kinematic:
                if b.motion is not None:
                    v, w = b.motion(self.t)
                    b.v = np.asarray(v, dtype=np.float64).reshape(3)
                    b.w = np.asarray(w, dtype=np.float64).reshape(3)
            else:
                b.v = b.v + lin / b.mass
                if b.gravity:
                    b.v = b.v + dt * self.gravity
                if not b.lock_rot:
                    Iw = b.I_world()
                    Iwi = b.I_world_inv()
                    b.w = b.w + Iwi @ (ang - dt * np.cross(b.w, Iw @ b.w))
                else:
                    b.w = np.zeros(3)
            b.x = b.x + dt * b.v
            if np.any(b.w != 0.0):
                b.q = _quat_integrate(b.q, b.w, dt)
        self.t += dt
        self._upload_bodies()

    def step(self, solver: MPMSolver, dt: Optional[float] = None) -> None:
        """One full MPM step with contact + rigid advance."""
        dt = solver.dt if dt is None else dt
        solver.step(contacts=self, dt=dt)
        self.advance(dt)

    # -- diagnostics --------------------------------------------------------
    def state_bytes(self) -> bytes:
        return b"".join(np.ascontiguousarray(b.state(), dtype=np.float64).tobytes()
                        for b in self.bodies)

    def free(self) -> None:
        for n in ("b_type", "b_h", "b_x", "b_q", "b_v", "b_w", "b_mu", "b_minv",
                  "b_iinv", "b_iinv_max", "b_count", "b_dv", "b_dw", "imp",
                  "jac", "c_body", "c_n", "c_phi", "c_lam", "v_free",
                  "r_res", "r_lam", "diag"):
            if hasattr(self, n):
                setattr(self, n, None)
        self._alloc = False


@wp.kernel
def _ring_kernel(
    idx: wp.array(dtype=wp.int32),
    x0: wp.array(dtype=wp.vec3),
    sig: wp.array(dtype=wp.float32),
    zeta: float, dt: float,
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
):
    """Spring-damper pull of a ring particle towards its rest position,
    semi-implicit so the damper cannot overshoot:

        v <- (v - dt sigma^2 (x - x0)) / (1 + 2 zeta sigma dt)

    sigma [1/s] is the per-particle damping RATE; k = m sigma^2 and
    c = 2 zeta m sigma, so the particle mass cancels and the layer is
    mass-independent."""
    i = wp.tid()
    p = idx[i]
    sg = sig[i]
    d = x[p] - x0[i]
    v[p] = (v[p] - (dt * sg * sg) * d) / (1.0 + 2.0 * zeta * sg * dt)


# ==========================================================================
# contact bubble (scene v)
# ==========================================================================
class SpringDamperRing:
    """H9b: an *absorbing* ring instead of the frozen ring of scene (v).

    Particles farther than ``r`` from the body stay ordinary MPM material --
    they carry mass, stress and momentum -- and additionally get a
    spring-damper towards their rest position:

        sigma = c_p / L          [1/s]   (L = ring thickness, c_p = P-wave
                                          speed of the material)
        c     = 2 zeta m_p sigma [N s/m]  zeta = 0.5  =>  c = m_p c_p / L,
                                          which distributed over the layer is
                                          exactly the Lysmer-Kuhlemeyer
                                          viscous traction rho c_p per unit
                                          area -- the impedance match
        k     = m_p sigma^2      [N/m]    the accompanying far-field spring

    ``ramp`` > 0 grades sigma as ((d - r) / L)^ramp so the impedance does not
    jump at the bubble surface; ramp = 0 is a uniform ring.

    The frozen ring of ``FrozenParticles`` is the sigma -> infinity limit of
    this family and the full domain is sigma = 0, so one knob spans both.
    """

    def __init__(self, solver: MPMSolver, ring_mask: np.ndarray,
                 dist: np.ndarray, r: float, c_p: float,
                 zeta: float = 0.5, ramp: float = 0.0,
                 sigma_scale: float = 1.0, L: Optional[float] = None,
                 device: str = "cuda:0"):
        mask = np.asarray(ring_mask).reshape(-1)
        idx = np.where(mask)[0].astype(np.int32)
        self.n_ring = int(idx.size)
        self.n_active = int(solver.n_particles - self.n_ring)
        self.device = device
        self.zeta = float(zeta)
        self.ramp = float(ramp)
        self.r = float(r)
        self.c_p = float(c_p)
        if self.n_ring == 0:
            self.idx = None
            self.L = 0.0
            self.sigma0 = 0.0
            self.sigma_scale = float(sigma_scale)
            return
        d = np.asarray(dist, dtype=np.float64).reshape(-1)[idx]
        self.L = float(d.max() - r) if L is None else float(L)
        self.sigma0 = sigma_scale * c_p / max(self.L, 1.0e-9)
        self.sigma_scale = float(sigma_scale)
        if ramp > 0.0:
            t = np.clip((d - r) / max(self.L, 1.0e-9), 0.0, 1.0)
            sig = self.sigma0 * t ** ramp
        else:
            sig = np.full(idx.size, self.sigma0)
        self.k_over_m = float(self.sigma0 ** 2)
        self.c_over_m = float(2.0 * zeta * self.sigma0)
        with wp.ScopedDevice(device):
            self.idx = wp.array(idx, dtype=wp.int32)
            self.x0 = wp.array(np.ascontiguousarray(solver.x.numpy()[idx]),
                               dtype=wp.vec3)
            self.sig = wp.array(np.ascontiguousarray(sig, dtype=np.float32),
                                dtype=wp.float32)

    def apply(self, solver: MPMSolver, dt: float) -> None:
        if self.idx is None:
            return
        with wp.ScopedDevice(self.device):
            wp.launch(_ring_kernel, dim=self.n_ring, inputs=[
                self.idx, self.x0, self.sig, self.zeta, dt,
                solver.x, solver.v])

    def params(self, m_p: float) -> dict:
        return {"ring_particles": self.n_ring, "ring_L_m": self.L,
                "sigma_scale": getattr(self, "sigma_scale", 1.0),
                "c_p_mps": self.c_p, "sigma_per_s": self.sigma0,
                "zeta": self.zeta, "ramp": self.ramp,
                "k_N_per_m": m_p * self.sigma0 ** 2,
                "c_Ns_per_m": 2.0 * self.zeta * m_p * self.sigma0,
                "rho_cp_Pa_s_per_m": None}

    def free(self) -> None:
        self.idx = None
        self.x0 = None
        self.sig = None


class FrozenParticles:
    """Freeze a subset of particles into a static heightfield.

    Frozen particles keep scattering **mass** to the grid (so they still carry
    the support under the active material) but with v = 0, C = 0 and F = I,
    which in the MLS-MPM P2G means zero momentum and zero stress.  Their state
    is restored after every ``g2p`` -- so they are never advected: "mass on
    the grid, velocity zero, no G2P".
    """

    def __init__(self, solver: MPMSolver, frozen_mask: np.ndarray,
                 device: str = "cuda:0"):
        idx = np.where(np.asarray(frozen_mask).reshape(-1))[0].astype(np.int32)
        self.n_frozen = int(idx.size)
        self.n_active = int(solver.n_particles - self.n_frozen)
        self.device = device
        if self.n_frozen == 0:
            self.idx = None
            return
        with wp.ScopedDevice(device):
            self.idx = wp.array(idx, dtype=wp.int32)
            self.x0 = wp.array(np.ascontiguousarray(solver.x.numpy()[idx]),
                               dtype=wp.vec3)
            self.jp0 = wp.array(np.ascontiguousarray(solver.Jp.numpy()[idx]),
                                dtype=wp.float32)
        self.apply(solver)

    def apply(self, solver: MPMSolver) -> None:
        if self.idx is None:
            return
        with wp.ScopedDevice(self.device):
            wp.launch(_freeze_kernel, dim=self.n_frozen, inputs=[
                self.idx, self.x0, self.jp0, solver.x, solver.v, solver.C,
                solver.F, solver.Jp])

    def free(self) -> None:
        self.idx = None
        self.x0 = None
        self.jp0 = None
