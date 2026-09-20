"""U112 round 6 solver: Cloth5 + bending hysteresis + the two round-5 defect fixes.

Cloth6 inherits round 5's step (one-step NCP, lumped mass in the NCP, 0.25 h candidate margin)
untouched and replaces four pieces:

  * the hinge force, energy and tangent come from r6_kernels (Jenkin slider with the measured
    2HB, area floor in the tangent).  With theta_y == 0 and the floor not binding every
    floating-point operation is the round-5 one, which the bit-identity gate checks.
  * the contact frame comes from r6_contact.k_geom6 (previous-step side for a degenerate normal);
    with eps_n = 1e-30 it is the round-5 path.
  * gravity is a parameter, so the release test runs at g = 0.
  * theta_p is return-mapped once at the end of each step.

Nothing in r4_*.py, r5_*.py, energies.py or ncp_gpu.py is modified; the membrane force is still
energies.compute_pipkin_membrane_forces_kernel.
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np
import warp as wp

from . import energies as E
from . import r4_kernels as K
from . import r4_contact as CT
from . import r6_kernels as R6
from . import r6_contact as R6C
from .r4_solver import _f64, _i32, _v3, GRAV  # noqa: F401
from .r5_solver import Cloth5, tri_geometry, min_curvature_radius, min_triangle_area  # noqa: F401
from .r6_setup import hinge_bend_length, hinge_angles

SLOTS = 4


class Cloth6(Cloth5):
    def __init__(self, d, B, Eh, nu=.3, h=0., support=0, self_contact=False, mu_s=.5, mu_c=.2,
                 cap=None, kappa_res=0., ratio=20., area_floor=.1, eps_n=0., gscale=1.):
        super().__init__(d, B, Eh, nu, h, support, self_contact, mu_s, mu_c, cap)
        self.kappa_res = float(kappa_res)
        self.ratio = float(ratio)
        self.area_floor = float(area_floor)
        self.eps_n = float(eps_n) if eps_n > 0 else 1e-30
        self.gscale = float(gscale)
        nh = max(self.nh, 1)
        if self.nh:
            ell, self.ell_info = hinge_bend_length(d)
            th0, le, l1, l2 = hinge_angles(np.asarray(d['verts'], np.float64), d)
        else:
            ell, self.ell_info = np.zeros(1), {}
            l1 = l2 = np.zeros(1)
        self.ell_np = np.asarray(ell, np.float64)
        self.ky_np = self.kappa_res * self.ell_np
        self.ky = _f64(self.ky_np)
        self.thp = wp.zeros(nh, dtype=wp.float64)
        self.l1f = _f64(self.area_floor * np.asarray(l1, np.float64))
        self.l2f = _f64(self.area_floor * np.asarray(l2, np.float64))
        self.slip = wp.zeros(1, dtype=wp.int32)
        self.hth = wp.zeros(nh, dtype=wp.float64)
        self.hmom = wp.zeros(nh, dtype=wp.float64)
        nt = max(self.nt, 1)
        self.pkey = wp.zeros((nt, SLOTS), dtype=wp.int32)
        self.ckey = wp.zeros((nt, SLOTS), dtype=wp.int32)
        self.pval = wp.zeros((nt, SLOTS), dtype=wp.float64)
        self.cval = wp.zeros((nt, SLOTS), dtype=wp.float64)
        wp.launch(R6C.k_side_clear, dim=(nt, SLOTS), inputs=[self.pkey, self.pval])
        self.cside = wp.zeros(self.cap, dtype=wp.float64)
        self.cstat = wp.zeros(4, dtype=wp.int32)

    # ------------------------------------------------------------------ physics
    def forces(self, xa, acc):
        acc.zero_()
        if self.B and self.nh:
            wp.launch(R6.k_hinge_force6, dim=self.nh,
                      inputs=[xa, *self.hv, self.hw, self.h0, self.B, self.thp, self.ky,
                              self.ratio, acc])
        wp.launch(E.compute_pipkin_membrane_forces_kernel, dim=self.nt,
                  inputs=[xa, self.t0, self.t1, self.t2, self.tb, self.tc, self.ta,
                          self.Eh, self.nu, self.Eh / (1 - self.nu ** 2), acc])

    def energy(self, xa):
        wp.launch(K.k_tri_energy, dim=self.nt,
                  inputs=[xa, self.t0, self.t1, self.t2, self.tb, self.tc, self.ta,
                          self.Eh, self.nu, self.etri])
        self.reduce(self.etri, self.nt, self.s_e, 0)
        if self.B and self.nh:
            wp.launch(R6.k_hinge_energy6, dim=self.nh,
                      inputs=[xa, *self.hv, self.hw, self.h0, self.B, self.thp, self.ky,
                              self.ratio, self.ehin])
            self.reduce(self.ehin, self.nh, self.s_e, 1)
        else:
            wp.launch(K.k_copy, dim=1, inputs=[self.s_zero, 0, self.s_e, 1])
        wp.launch(R6.k_node_energy_g, dim=self.n,
                  inputs=[xa, self.m, self.tgt, self.dm, self.fc, self.free,
                          self.gscale * GRAV, self.enod])
        self.reduce(self.enod, self.n, self.s_e, 2)
        s = self.s_e.numpy()
        return float(s[0] + s[1] + s[2])

    def residual(self, xa):
        self.forces(xa, self.acc)
        wp.launch(R6.k_residual_g, dim=self.n,
                  inputs=[self.acc, self.m, self.dm, xa, self.tgt, self.fc, self.free,
                          self.gscale * GRAV, self.r])

    def matvec(self, pv, out):
        self.dacc.zero_()
        wp.launch(K.k_tri_matvec, dim=self.nt,
                  inputs=[self.xt, pv, self.t0, self.t1, self.t2, self.tb, self.tc, self.ta,
                          self.Eh, self.nu, self.dacc])
        if self.B and self.nh:
            wp.launch(R6.k_hinge_matvec6, dim=self.nh,
                      inputs=[self.xt, pv, *self.hv, self.hw, self.h0, self.B, self.thp,
                              self.ky, self.ratio, self.l1f, self.l2f, self.dacc])
        wp.launch(K.k_apply_A, dim=self.n, inputs=[self.dacc, pv, self.dm, self.free, out])

    def precond(self):
        self.dacc9.zero_()
        wp.launch(K.k_tri_diag3, dim=self.nt,
                  inputs=[self.xt, self.t0, self.t1, self.t2, self.tb, self.tc, self.ta,
                          self.Eh, self.nu, self.dacc9])
        if self.B and self.nh:
            wp.launch(R6.k_hinge_diag36, dim=self.nh,
                      inputs=[self.xt, *self.hv, self.hw, self.h0, self.B, self.thp, self.ky,
                              self.ratio, self.l1f, self.l2f, self.dacc9])
        wp.launch(K.k_precond3, dim=self.n, inputs=[self.dacc9, self.dm, self.free, self.pre3])

    def static_residual(self):
        self.forces(self.x, self.acc)
        f = self.acc.numpy().astype(float) / 1e12
        f[:, 1] -= self.mass_np * GRAV * self.gscale
        f = f[self.free_np == 1]
        return float(np.linalg.norm(f) / self.grav_norm)

    # ------------------------------------------------------------------ contact
    def contact_geometry(self, xa, dt=1e-3):
        self.dt_now = dt
        if self.C == 0:
            return
        store = xa is self.x0
        nt = max(self.nt, 1)
        if store:
            wp.launch(R6C.k_side_clear, dim=(nt, SLOTS), inputs=[self.ckey, self.cval])
            wp.launch(R6C.k_side_claim, dim=self.C,
                      inputs=[self.ckind, self.ca, self.cb, self.C, SLOTS, self.ckey])
        wp.launch(R6C.k_geom6, dim=self.C,
                  inputs=[xa, self.tri2d, self.ev, self.ckind, self.ca, self.cb, self.C,
                          self.h, self.mu_s, self.mu_c, self.eps_n, self.pkey, self.pval,
                          SLOTS, self.cids, self.cw, self.cn, self.ct1, self.ct2, self.cgap,
                          self.cmu, self.cside, self.cstat])
        if store:
            wp.launch(R6C.k_side_store, dim=self.C,
                      inputs=[self.ckind, self.ca, self.cb, self.C, SLOTS, self.ckey,
                              self.cside, self.cval])
            self.pkey, self.ckey = self.ckey, self.pkey
            self.pval, self.cval = self.cval, self.pval
        self.ncnt.zero_()
        wp.launch(CT.k_count_nodes, dim=self.C, inputs=[self.cids, self.C, self.ncnt])
        wp.launch(CT.k_rho, dim=self.C,
                  inputs=[self.cids, self.cw, self.C, self.comp, self.dt_now, self.ncnt,
                          self.crho])

    # ------------------------------------------------------------------ step
    def step(self, **kw):
        info = super().step(**kw)
        if self.nh and self.kappa_res > 0:
            self.slip.zero_()
            wp.launch(R6.k_hyst_update, dim=self.nh,
                      inputs=[self.x, *self.hv, self.h0, self.ky, self.ratio, self.thp,
                              self.slip])
            info['slip'] = int(self.slip.numpy()[0])
        return info

    # ------------------------------------------------------------------ diagnostics
    def hinge_state(self):
        """(theta, moment, theta_p) per hinge for the current x."""
        if not self.nh:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        wp.launch(R6.k_hinge_probe, dim=self.nh,
                  inputs=[self.x, *self.hv, self.hw, self.h0, self.B, self.thp, self.ky,
                          self.ratio, self.hth, self.hmom])
        return self.hth.numpy().copy(), self.hmom.numpy().copy(), self.thp.numpy().copy()

    def load_hysteresis(self, x_seq):
        """Drive the state through a sequence of prescribed configurations, return-mapping the
        slider at each one (kinematic loading; the return map is exact for monotone loading)."""
        for xs in x_seq:
            self.x.assign(np.ascontiguousarray(xs, np.float64))
            if self.nh and self.kappa_res > 0:
                wp.launch(R6.k_hyst_update, dim=self.nh,
                          inputs=[self.x, *self.hv, self.h0, self.ky, self.ratio, self.thp,
                                  self.slip])


def circle_fit(p):
    """Algebraic (Kasa) circle fit of 2-D points; returns (centre, radius)."""
    p = np.asarray(p, np.float64)
    A = np.column_stack([p[:, 0], p[:, 1], np.ones(len(p))])
    b = p[:, 0] ** 2 + p[:, 1] ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = .5 * sol[0], .5 * sol[1]
    r = float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 0.)))
    return np.array([cx, cy]), r


def bend_to_cylinder(verts, kappa, axis=0, up=1):
    """Isometric wrap of a flat strip onto a cylinder of curvature kappa (arc length preserved
    along `axis`), so the loading path adds no membrane strain."""
    v = np.asarray(verts, np.float64).copy()
    s = v[:, axis] - .5 * (v[:, axis].max() + v[:, axis].min())
    phi = kappa * s
    v[:, axis] = np.sin(phi) / kappa
    v[:, up] = v[:, up] + (1.0 - np.cos(phi)) / kappa
    return v
