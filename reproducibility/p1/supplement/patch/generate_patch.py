#!/usr/bin/env python3
"""Patch-wrench and chain scenes: self-contained generator.

This file regenerates the patch/chain data shipped next to it:

    scenes.json      per-scene geometry (bodies, patches, free velocities, units)
    operators.npz    per-scene assembled operators (J_patch, J_points, C, G, b, mu)

Run:

    python generate_patch.py

Conventions (identical to the delivered experiments):

* One patch per body pair.  A patch is a rectangular support with a right-handed
  orthonormal frame (t1, t2, n), n = t1 x t2, half extents e1 > 0 (along t1) and
  e2 > 0 (along t2), and four point contacts at its corners.
* Bodies are rigid with a diagonal body-frame inertia I = m/3 * (hy^2+hz^2,
  hx^2+hz^2, hx^2+hy^2) for half extents (hx, hy, hz).  The generalized velocity
  column of a body is (v, omega) in world axes, translation first.
* The contact rows of a patch are ordered

      w = (F_n, M_t1, M_t2, F_t1, F_t2, M_n)

  with the dual twist rows

      (n.v(p0), e2 t1.omega, e1 t2.omega, t1.v(p0), t2.v(p0), ell n.omega),
      ell = sqrt((e1^2 + e2^2)/3).

  The moment rows carry the extent scaling; this is a diagonal gauge, not a
  physical change.  Point rows are (n, t1, t2) at each corner, in that order.
* The admissible patch cone is

      K(mu) = { F_n >= 0, |M_t1| <= F_n, |M_t2| <= F_n,
                ||(F_t1, F_t2, M_n)|| <= mu F_n }.

  The first three inequalities are exactly the image of the four non-negative
  corner normal loads (the centre of pressure lies inside the rectangle).  The
  last is the Contensou-type friction-plus-spin ellipsoid used by the solver.
* Sign convention: patch (a, b) has body a on the +1 side and body b on the -1
  side; b = -1 denotes the static ground (no dynamic mass block).  Contact
  directions enter the translational columns, and their cross product with the
  moment arm (p - body centre) enters the rotational columns, with opposite
  signs for the two incident bodies.  Ground contributes no inverse mass.
* Solver convention: the complementarity problem is u = G w + b, w in K(mu),
  with the de Saxce shift Gamma(u) = (mu ||(u_Ft1, u_Ft2, u_Mn)||, 0,0,0,0,0)
  and natural-map residual ||w - P_K(w - rho (u + Gamma(u)))||_inf.

No external data or paths are required.  Only NumPy is used.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# scene containers
# ---------------------------------------------------------------------------


def skew(r):
    r = np.asarray(r, float)
    return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])


def adjoint(r):
    """Twist transport: [v(p); omega] = [[I, -skew(r)],[0, I]] [v(c); omega],
    with r = p - c."""
    A = np.eye(6)
    A[:3, 3:] = -skew(r)
    return A


@dataclass
class Body:
    mass: float
    half: np.ndarray
    centre: np.ndarray

    def inertia(self):
        hx, hy, hz = self.half
        return self.mass / 3.0 * np.array(
            [hy ** 2 + hz ** 2, hx ** 2 + hz ** 2, hx ** 2 + hy ** 2]
        )

    def Minv(self):
        M = np.zeros((6, 6))
        M[:3, :3] = np.eye(3) / self.mass
        M[3:, 3:] = np.diag(1.0 / self.inertia())
        return M


@dataclass
class Patch:
    a: int
    b: int
    p0: np.ndarray
    n: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    e1: float
    e2: float
    pts: np.ndarray
    mu: float = 0.5

    @property
    def ell(self):
        return float(np.sqrt((self.e1 ** 2 + self.e2 ** 2) / 3.0))


@dataclass
class PatchScene:
    bodies: list
    patches: list
    v_free: np.ndarray
    dt: float = 0.004
    meta: dict = field(default_factory=dict)

    @property
    def nb(self):
        return len(self.bodies)

    @property
    def np_(self):
        return len(self.patches)

    @property
    def nv(self):
        return 6 * len(self.bodies)

    def Minv(self):
        M = np.zeros((self.nv, self.nv))
        for i, bd in enumerate(self.bodies):
            M[6 * i:6 * i + 6, 6 * i:6 * i + 6] = bd.Minv()
        return M

    def J_points(self):
        rows, pairs, mus = [], [], []
        for pa in self.patches:
            for p in pa.pts:
                blk = np.zeros((3, self.nv))
                for r, d in enumerate((pa.n, pa.t1, pa.t2)):
                    if pa.a >= 0:
                        rA = p - self.bodies[pa.a].centre
                        blk[r, 6 * pa.a:6 * pa.a + 3] = d
                        blk[r, 6 * pa.a + 3:6 * pa.a + 6] = np.cross(rA, d)
                    if pa.b >= 0:
                        rB = p - self.bodies[pa.b].centre
                        blk[r, 6 * pa.b:6 * pa.b + 3] = -d
                        blk[r, 6 * pa.b + 3:6 * pa.b + 6] = -np.cross(rB, d)
                rows.append(blk)
                pairs.append((pa.a, pa.b))
                mus.append(pa.mu)
        return np.vstack(rows), np.array(pairs, int), np.array(mus, float)

    def _patch_rows_local(self, pa):
        blk = np.zeros((6, self.nv))
        force_dirs = [(0, pa.n), (3, pa.t1), (4, pa.t2)]
        mom_dirs = [(1, pa.t1, pa.e2), (2, pa.t2, pa.e1), (5, pa.n, pa.ell)]
        for i, sgn in ((pa.a, 1.0), (pa.b, -1.0)):
            if i < 0:
                continue
            r = pa.p0 - self.bodies[i].centre
            for row, d in force_dirs:
                blk[row, 6 * i:6 * i + 3] = sgn * d
                blk[row, 6 * i + 3:6 * i + 6] = sgn * np.cross(r, d)
            for row, d, sc in mom_dirs:
                blk[row, 6 * i + 3:6 * i + 6] = sgn * d * sc
        return blk

    def J_patch(self):
        return np.vstack([self._patch_rows_local(pa) for pa in self.patches])

    def conductances(self, p_ref=None):
        p_ref = np.zeros(3) if p_ref is None else np.asarray(p_ref, float)
        out = []
        for bd in self.bodies:
            A = adjoint(p_ref - bd.centre)
            out.append(A @ bd.Minv() @ A.T)
        return out

    def incidence(self):
        S = np.zeros((self.np_, self.nb))
        for c, pa in enumerate(self.patches):
            if pa.a >= 0:
                S[c, pa.a] += 1.0
            if pa.b >= 0:
                S[c, pa.b] -= 1.0
        return S

    def laplacian_common(self, p_ref=None):
        W = self.conductances(p_ref)
        S = self.incidence()
        npatch = self.np_
        L = np.zeros((6 * npatch, 6 * npatch))
        for i in range(self.nb):
            for c in range(npatch):
                if S[c, i] == 0:
                    continue
                for d in range(npatch):
                    if S[d, i] == 0:
                        continue
                    L[6 * c:6 * c + 6, 6 * d:6 * d + 6] += S[c, i] * S[d, i] * W[i]
        return L

    def mu(self):
        return np.array([pa.mu for pa in self.patches], float)

    def half_extents(self):
        return np.array([(pa.e1, pa.e2, pa.ell) for pa in self.patches], float)


# ---------------------------------------------------------------------------
# scene constructors
# ---------------------------------------------------------------------------


def box_column(masses, half=(0.1, 0.1, 0.1), mu=0.5, dt=0.004, g=9.81,
               v_tan=0.0, v_spin=0.0, v_roll=0.0):
    """n boxes stacked on the static ground, one four-corner patch per interface."""
    hx, hy, hz = half
    m = np.asarray(masses, float)
    nb = len(m)
    bodies = [Body(float(m[i]), np.array(half, float),
                   np.array([0.0, 0.0, 2 * hz * (i + 0.5)])) for i in range(nb)]
    n = np.array([0.0, 0.0, 1.0])
    t1 = np.array([1.0, 0.0, 0.0])
    t2 = np.array([0.0, 1.0, 0.0])
    patches = []
    for k in range(nb):
        z = 2 * hz * k
        pts = np.array([[sx * hx, sy * hy, z] for sx in (-1, 1) for sy in (-1, 1)])
        patches.append(Patch(a=k, b=k - 1, p0=np.array([0.0, 0.0, z]),
                             n=n, t1=t1, t2=t2, e1=hx, e2=hy, pts=pts, mu=float(mu)))
    v = np.zeros(6 * nb)
    for i in range(nb):
        v[6 * i + 2] = -g * dt
        if v_tan:
            v[6 * i + 0] += v_tan * (1.0 if i % 2 == 0 else -1.0)
        if v_spin:
            v[6 * i + 5] += v_spin * (1.0 if i % 2 == 0 else -1.0)
        if v_roll:
            v[6 * i + 4] += v_roll * (1.0 if i % 2 == 0 else -1.0)
    return PatchScene(bodies, patches, v, dt,
                      meta={"kind": "box_column", "masses": m.tolist(),
                            "half": list(half), "v_roll": float(v_roll),
                            "v_tan": float(v_tan), "v_spin": float(v_spin)})


def box_lattice(nx=2, ny=2, nz=2, half=(0.1, 0.1, 0.1), mu=0.5, dt=0.004,
                g=9.81, mass=1.0):
    """Regular nx*ny*nz box grid touching face to face, plus a ground patch under
    every bottom-layer box.  Bodies shared by three or more patches break the
    unsigned Laplacian (branching)."""
    hx, hy, hz = half
    h = np.array(half, float)
    idx = {}
    bodies = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                idx[(i, j, k)] = len(bodies)
                bodies.append(Body(float(mass), h,
                                   np.array([(2 * i + 1) * hx, (2 * j + 1) * hy,
                                             (2 * k + 1) * hz])))
    patches = []
    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    ez = np.array([0.0, 0.0, 1.0])
    axes = [(0, ex, ey, ez, hy, hz), (1, ey, ez, ex, hz, hx), (2, ez, ex, ey, hx, hy)]
    for ax, n, t1, t2, e1, e2 in axes:
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    c = [i, j, k]
                    c2 = list(c)
                    c2[ax] += 1
                    if c2[ax] >= (nx, ny, nz)[ax]:
                        continue
                    lo, hi = idx[tuple(c)], idx[tuple(c2)]
                    p0 = 0.5 * (bodies[lo].centre + bodies[hi].centre)
                    pts = np.array([p0 + sa * e1 * t1 + sb * e2 * t2
                                    for sa in (-1, 1) for sb in (-1, 1)])
                    patches.append(Patch(a=hi, b=lo, p0=p0, n=n, t1=t1, t2=t2,
                                         e1=e1, e2=e2, pts=pts, mu=float(mu)))
    for i in range(nx):
        for j in range(ny):
            bi = idx[(i, j, 0)]
            p0 = bodies[bi].centre - np.array([0.0, 0.0, hz])
            pts = np.array([p0 + sa * hx * ex + sb * hy * ey
                            for sa in (-1, 1) for sb in (-1, 1)])
            patches.append(Patch(a=bi, b=-1, p0=p0, n=ez, t1=ex, t2=ey,
                                 e1=hx, e2=hy, pts=pts, mu=float(mu)))
    v = np.zeros(6 * len(bodies))
    for i in range(len(bodies)):
        v[6 * i + 2] = -g * dt
    return PatchScene(bodies, patches, v, dt,
                      meta={"kind": "box_lattice", "dims": [nx, ny, nz],
                            "half": list(half)})


def pushed_cube(mu=0.5, size=0.20, mass=15.0, force=147.15, dt=0.004):
    """One 0.20 m / 15 kg cube on the ground with a horizontal push; four coplanar
    corner contacts make the point problem statically indeterminate."""
    h = size / 2.0
    bodies = [Body(mass, np.array([h, h, h]), np.array([0.0, 0.0, h]))]
    pts = np.array([[sx * h, sy * h, 0.0] for sx in (-1, 1) for sy in (-1, 1)])
    patches = [Patch(a=0, b=-1, p0=np.zeros(3), n=np.array([0.0, 0.0, 1.0]),
                     t1=np.array([1.0, 0.0, 0.0]), t2=np.array([0.0, 1.0, 0.0]),
                     e1=h, e2=h, pts=pts, mu=float(mu))]
    v = np.zeros(6)
    v[2] = -9.81 * dt
    v[0] = force * dt / mass
    return PatchScene(bodies, patches, v, dt, meta={"kind": "pushed_cube"})


def lumping_matrix(scene: PatchScene):
    """C (6 n_p x 3 n_pts) with J_points^T lambda = J_patch^T (C lambda) exactly."""
    rows_tot = 6 * scene.np_
    k_tot = sum(len(pa.pts) for pa in scene.patches)
    C = np.zeros((rows_tot, 3 * k_tot))
    col = 0
    for c, pa in enumerate(scene.patches):
        for p in pa.pts:
            r = p - pa.p0
            for j, d in enumerate((pa.n, pa.t1, pa.t2)):
                f = d
                m = np.cross(r, d)
                blk = np.zeros(6)
                blk[0] = f @ pa.n
                blk[3] = f @ pa.t1
                blk[4] = f @ pa.t2
                blk[1] = (m @ pa.t1) / pa.e2
                blk[2] = (m @ pa.t2) / pa.e1
                blk[5] = (m @ pa.n) / pa.ell
                C[6 * c:6 * c + 6, col + j] = blk
            col += 3
    return C


# ---------------------------------------------------------------------------
# patch cone and projection (the contact law convention)
# ---------------------------------------------------------------------------


def proj_patch(z, mu):
    """Exact Euclidean projection of rows of z (n,6) onto K(mu), closed form."""
    Z = np.atleast_2d(np.asarray(z, float))
    m = np.broadcast_to(np.atleast_1d(np.asarray(mu, float)), (Z.shape[0],))
    out = np.zeros_like(Z)
    s1 = np.abs(Z[:, 1])
    s2 = np.abs(Z[:, 2])
    st = np.linalg.norm(Z[:, 3:6], axis=1)
    for i in range(Z.shape[0]):
        z0 = Z[i, 0]
        mu_i = m[i]
        cs = [1.0, 1.0] + ([mu_i] if mu_i > 0 else [])
        ss = [s1[i], s2[i]] + ([st[i]] if mu_i > 0 else [])
        bps = sorted([ss[j] / cs[j] for j in range(len(cs)) if cs[j] > 0])
        edges = [0.0] + [max(0.0, x) for x in bps] + [np.inf]
        best_f, best_g = 0.0, None
        for a, bnd in zip(edges[:-1], edges[1:]):
            mid = a + 1.0 if not np.isfinite(bnd) else 0.5 * (a + bnd)
            num, den = z0, 1.0
            for j in range(len(cs)):
                if ss[j] > cs[j] * mid:
                    num += cs[j] * ss[j]
                    den += cs[j] ** 2
            f = num / den
            f = min(max(f, a), bnd if np.isfinite(bnd) else f)
            f = max(f, 0.0)
            g = (f - z0) ** 2
            for j in range(len(cs)):
                g += max(0.0, ss[j] - cs[j] * f) ** 2
            if mu_i <= 0:
                g += st[i] ** 2
            if best_g is None or g < best_g:
                best_f, best_g = f, g
        f = best_f
        out[i, 0] = f
        out[i, 1] = min(max(Z[i, 1], -f), f)
        out[i, 2] = min(max(Z[i, 2], -f), f)
        if mu_i > 0 and st[i] > 0:
            out[i, 3:6] = Z[i, 3:6] * min(1.0, mu_i * f / st[i])
    return out[0] if np.asarray(z).ndim == 1 else out


def desaxce_patch(u, mu):
    """de Saxce shift on the friction-plus-spin block."""
    U = np.atleast_2d(np.asarray(u, float))
    m = np.broadcast_to(np.atleast_1d(np.asarray(mu, float)), (U.shape[0],))
    g = np.zeros_like(U)
    g[:, 0] = m * np.linalg.norm(U[:, 3:6], axis=1)
    return g


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def build_scenes():
    """Return the ordered (name, PatchScene) list delivered by this package."""
    out = []
    out.append(("three_box_column", box_column([1.0, 1.0, 1.0])))
    out.append(("eight_box_tower", box_column([1.0] * 8)))
    out.append(("mass_ratio_column", box_column([1.0, 1.0, 1000.0])))
    out.append(("box_lattice", box_lattice(2, 2, 2)))
    out.append(("pushed_cube", pushed_cube()))
    out.append(("eight_box_tower_rolling", box_column([1.0] * 8, v_roll=0.6)))
    out.append(("mass_ratio_column_rolling", box_column([1.0, 1.0, 1000.0], v_roll=0.6)))
    return out


def scene_to_dict(name, sc):
    return {
        "name": name,
        "kind": sc.meta.get("kind", "patch_scene"),
        "dt": float(sc.dt),
        "units": {"length": "m", "mass": "kg", "time": "s",
                  "impulse": "N s", "angle": "rad", "friction": "dimensionless"},
        "g": 9.81,
        "bodies": [
            {"mass": float(bd.mass),
             "half": [float(x) for x in bd.half],
             "centre": [float(x) for x in bd.centre],
             "inertia": [float(x) for x in bd.inertia()]}
            for bd in sc.bodies
        ],
        "patches": [
            {"a": int(pa.a), "b": int(pa.b),
             "p0": [float(x) for x in pa.p0],
             "n": [float(x) for x in pa.n],
             "t1": [float(x) for x in pa.t1],
             "t2": [float(x) for x in pa.t2],
             "e1": float(pa.e1), "e2": float(pa.e2), "ell": float(pa.ell),
             "mu": float(pa.mu),
             "pts": [[float(x) for x in p] for p in pa.pts]}
            for pa in sc.patches
        ],
        "v_free": [float(x) for x in sc.v_free],
        "meta": sc.meta,
    }


def main():
    scenes = build_scenes()
    doc = {"description": "Patch-wrench and chain scenes for the contact-graph supplement",
           "cone": {"normal": "F_n >= 0", "moments": "|M_t1| <= F_n, |M_t2| <= F_n",
                    "friction_spin": "||(F_t1,F_t2,M_n)|| <= mu F_n",
                    "wrench_order": ["F_n", "M_t1", "M_t2", "F_t1", "F_t2", "M_n"]},
           "scenes": [scene_to_dict(n, s) for n, s in scenes]}
    with open(os.path.join(HERE, "scenes.json"), "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write("\n")

    arrays = {}
    for name, sc in scenes:
        Jp = sc.J_patch()
        Jpts, pairs, mus = sc.J_points()
        C = lumping_matrix(sc)
        G = Jp @ sc.Minv() @ Jp.T
        b = Jp @ sc.v_free
        arrays[f"{name}__J_patch"] = Jp
        arrays[f"{name}__J_points"] = Jpts
        arrays[f"{name}__body_pairs"] = np.array(
            [(pa.a, pa.b) for pa in sc.patches], int)
        arrays[f"{name}__point_pairs"] = pairs
        arrays[f"{name}__C"] = C
        arrays[f"{name}__G"] = G
        arrays[f"{name}__b"] = b
        arrays[f"{name}__mu"] = sc.mu()
    np.savez(os.path.join(HERE, "operators.npz"), **arrays)
    print("wrote", len(scenes), "patch scenes to", HERE)


if __name__ == "__main__":
    main()
