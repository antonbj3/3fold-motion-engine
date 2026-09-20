"""U112 round 5 solver: the three measured round-4 defects removed.

Cloth5 subclasses round 4's Cloth4 (r4_solver.py is untouched) and replaces step().

(1) ONE-STEP NCP, NO DOUBLE COUNTING.  Round 4 ran a fixed point of `rounds` passes in which the
    contact force of the previous pass was HELD inside the implicit solve (fc in the residual and
    in the line-search energy) and then removed again from the end-of-step velocity with
    k_vfree, i.e. with the LUMPED compliance dt^2/m(1+dt damp).  The implicit solve does not
    respond to fc with the lumped compliance -- it responds with (H + M/dt^2)^-1 fc, which on a
    near-inextensible membrane is ~10^3 smaller -- so the subtraction removed more velocity than
    the held force put in.  The NCP then saw a receding gap, released the impulse, and the fixed
    point cycled between "impulse" and "no impulse".  Round 5 uses the plain one-step
    (Stewart-Trinkle / Anitescu) split: ONE elastic solve with fc == 0, then ONE NCP solve on the
    resulting free velocity, with g0 and the contact frame taken at the START of the step, then
    x = x0 + dt v.  Nothing is held, so nothing is double counted.

(2) LUMPED MASS IN THE NCP.  The Delassus operator of the NCP is W = sum_k w_k^2 / m_eff,k with
    m_eff = m (1 + dt damp), i.e. the lumped-mass collision response ncp_gpu.py is verified
    against -- not the block diagonal of the implicit tangent.  k_lumped_comp supplies it and
    r5_checks.py measures W against the analytic value.

(3) CANDIDATE MARGIN 0.25 h AND A BUFFER THAT DOES NOT OVERFLOW.  The cloth-cloth candidate
    threshold is h + 0.25 h; the buffer is sized from a measured first-step candidate count with
    headroom, and any step with dropped pairs > 0 marks the whole run as not contact-certified.
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np
import warp as wp

from . import r4_kernels as K
from . import r4_contact as CT
from .r4_solver import Cloth4, tri_geometry, edge_list, triangle_neighbours  # noqa: F401


class Cloth5(Cloth4):
    """Round-5 step: single elastic solve + single one-step NCP, lumped mass, 0.25 h margin."""

    cgap0 = None
    C0 = 0
    end_stats = True

    def step(self, dt=1e-3, damp=100., newton=5, tol=1e-3, cg_iter=200, cg_tol=1e-3,
             rounds=1, ls=12, c1=1e-4, ncp_iter=200, ncp_tol=1e-12, margin=None,
             self_margin=None):
        n = self.n
        wp.copy(self.x0, self.x)
        wp.launch(K.k_set_target, dim=n,
                  inputs=[self.x0, self.v, self.m, dt, damp, self.tgt, self.dm])
        # fc is identically zero for the whole step: no contact force is held in the implicit
        # solve, so the NCP cannot double count it (defect 1).
        wp.launch(K.k_zero3, dim=n, inputs=[self.fc])
        # lumped-mass compliance dt^2 / (m (1+dt damp)) for the NCP (defect 2)
        wp.launch(K.k_lumped_comp, dim=n, inputs=[self.minv, dt, damp, self.comp])
        self.iacc.zero_()
        wp.launch(K.k_zero3, dim=self.cap, inputs=[self.clam])
        contact_on = bool(self.support or self.self_contact)
        info = dict(newton=[], cg=[], alpha=[], contacts=0, overflow=0, ncp_iters=[],
                    ncp_res=[], rounds=1, dx_estimate_m=0.0)
        if contact_on:
            if margin is None:
                margin = self.h
            c, over = self.build_candidates(margin, self_margin)
            info['contacts'] = c
            info['overflow'] = over
            info['margin'] = float(margin)
            info['self_margin'] = float(self.h if self_margin is None else self_margin)
            self.contact_geometry(self.x0, dt)          # g0 and frame at the START of the step
            if self.cgap0 is None:
                self.cgap0 = wp.zeros(self.cap, dtype=wp.float64)
            wp.copy(self.cgap0, self.cgap)
            self.C0 = self.C

        # ---- elastic implicit solve, no contact force in the functional
        wp.copy(self.xt, self.x)
        hist = []
        alphas = []
        cgs = []
        for ni in range(newton):
            self.residual(self.xt)
            self.dot(self.r, self.r, self.s_rr)
            rel = float(np.sqrt(max(self.s_rr.numpy()[0], 0.0))) / self.grav_norm
            hist.append(rel)
            if rel <= tol:
                break
            self.precond()
            cit, crel = self.cg(maxiter=cg_iter, rtol=cg_tol)
            cgs.append([cit, crel])
            self.dot(self.r, self.sol, self.s_tmp)
            slope = float(self.s_tmp.numpy()[0])
            if slope <= 0.0:
                alphas.append(0.0)
                break
            eb = self.energy(self.xt)
            alpha = 1.0
            ok = False
            for _ in range(ls):
                wp.launch(K.k_lincomb, dim=n,
                          inputs=[self.xt, self.sol, alpha, self.free, self.xtry])
                et = self.energy(self.xtry)
                if np.isfinite(et) and et <= eb - c1 * alpha * slope:
                    ok = True
                    break
                alpha *= .5
            if not ok:
                alphas.append(0.0)
                break
            alphas.append(alpha)
            wp.copy(self.xt, self.xtry)
        self.residual(self.xt)
        self.dot(self.r, self.r, self.s_rr)
        hist.append(float(np.sqrt(max(self.s_rr.numpy()[0], 0.0))) / self.grav_norm)
        wp.launch(K.k_step_estimate, dim=n, inputs=[self.pre3, self.r, self.nodebuf])
        self.reduce(self.nodebuf, n, self.s_tmp, 0, mx=True)
        info['dx_estimate_m'] = float(self.s_tmp.numpy()[0])
        info['newton'].append(hist)
        info['alpha'].append(alphas)
        info['cg'].append(cgs)
        wp.copy(self.x, self.xt)

        # ---- one NCP solve on the free velocity, then position update
        if contact_on:
            wp.launch(K.k_velocity, dim=n,
                      inputs=[self.x, self.x0, dt, self.free, self.vfree])
            nit, nres = self.ncp_solve(dt, iters=ncp_iter, tol=ncp_tol)
            info['ncp_iters'].append(nit)
            info['ncp_res'].append(nres)
            wp.launch(K.k_advance, dim=n, inputs=[self.x0, self.v, dt, self.free, self.x])
            wp.launch(CT.k_contact_force, dim=n, inputs=[self.iacc, dt, self.fc])
        wp.launch(K.k_velocity, dim=n, inputs=[self.x, self.x0, dt, self.free, self.v])
        if contact_on and self.C and self.end_stats:
            # gap statistics on the END state (the nonlinear gap, not the linearised one)
            self.contact_geometry(self.x, dt)
            self.gstat.zero_()
            wp.launch(CT.k_gap_stats, dim=self.C,
                      inputs=[self.cgap, self.ckind, self.C, self.h, self.gstat])
            g = self.gstat.numpy()
            info['pen_support_m'] = float(g[0]) * 1e-12
            info['pen_vt_m'] = float(g[1]) * 1e-12
            info['pen_ee_m'] = float(g[2]) * 1e-12
            info['over_h2'] = [int(g[3]), int(g[4]), int(g[5])]
        info['residual'] = info['newton'][-1][-1] if info['newton'][-1] else 0.0
        info['converged'] = bool(info['residual'] <= tol)
        return info


def min_curvature_radius(x, d):
    """Smallest bending radius on the mesh: R = (h1 + h2) / |theta|, with h1, h2 the heights of
    the two hinge triangles over the shared edge (the surface span the dihedral angle turns
    through) and theta the dihedral angle of energies.compute_hinge_forces_kernel.
    Hinges with |theta| below 1e-9 rad are flat and excluded."""
    i1 = np.asarray(d['h_v1'], np.int64)
    i2 = np.asarray(d['h_v2'], np.int64)
    i3 = np.asarray(d['h_v3'], np.int64)
    i4 = np.asarray(d['h_v4'], np.int64)
    p1, p2, p3, p4 = x[i1], x[i2], x[i3], x[i4]
    e = p2 - p1
    le = np.linalg.norm(e, axis=1)
    ok = le > 1e-12
    n1 = np.cross(e, p3 - p1)
    n2 = np.cross(p4 - p1, e)
    l1 = np.linalg.norm(n1, axis=1)
    l2 = np.linalg.norm(n2, axis=1)
    ok &= (l1 > 1e-14) & (l2 > 1e-14)
    with np.errstate(invalid='ignore', divide='ignore'):
        u1 = n1 / l1[:, None]
        u2 = n2 / l2[:, None]
        cos_t = np.clip(np.einsum('ij,ij->i', u1, u2), -1., 1.)
        sin_t = np.einsum('ij,ij->i', np.cross(u1, u2), e / le[:, None])
        theta = np.abs(np.arctan2(sin_t, cos_t))
        span = (l1 + l2) / le          # h1 + h2 = (|n1| + |n2|) / |e|
        R = span / theta
    ok &= theta > 1e-9
    if not ok.any():
        return float('inf'), 0.
    return float(R[ok].min()), float(np.median(theta[ok]))


def min_triangle_area(x, tris):
    p = x[np.asarray(tris, np.int64)]
    a = .5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    return float(a.min())
