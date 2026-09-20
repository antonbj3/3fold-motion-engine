"""U112 round 4 implicit Lagrange cloth.

Backward Euler on nodal positions, Newton with PSD element tangents (matrix-free), CG with
Jacobi preconditioning on the GPU, line search on the TOTAL incremental energy (elastic +
gravity + inertia + the held contact potential), and an exact NCP contact solve coupled to the
elastic solve by a fixed point of at most `rounds` passes.

Forces are the verified kernels of energies.py; everything else is in r4_kernels/r4_contact.
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
from pathlib import Path
import numpy as np
import warp as wp

wp.config.kernel_cache_dir = str(Path(__file__).parent / 'cache')
wp.init()
from . import energies as E
from . import r4_kernels as K
from . import r4_contact as CT

T_BLOCK = 64
GRAV = 9.80665


def tri_geometry(d, axes=(0, 2)):
    """Rest shape-function derivatives in the material plane (same as implicit.geometry)."""
    x = d['verts'].astype(float)
    uv = x[:, axes]
    t = d['tris']
    a, b, c = uv[t[:, 0]], uv[t[:, 1]], uv[t[:, 2]]
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    d['tri_b'] = np.stack([b[:, 1] - c[:, 1], c[:, 1] - a[:, 1], a[:, 1] - b[:, 1]], 1) / det[:, None]
    d['tri_c'] = np.stack([c[:, 0] - b[:, 0], a[:, 0] - c[:, 0], b[:, 0] - a[:, 0]], 1) / det[:, None]
    d['tri_area'] = abs(det) * .5
    return d


def edge_list(tris):
    if len(tris) == 0:
        return np.zeros((0, 2), np.int32)
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0).astype(np.int32)


def triangle_neighbours(tris):
    """For each triangle, the opposite vertex of the triangle across each of its three edges
    (-1 on a boundary edge). Those vertices sit 1/sqrt(2) dx from the triangle in the flat rest
    state, i.e. closer than a few thicknesses, so they are excluded from the thickness test
    (Bridson-style adjacency exclusion); every other pair is tested."""
    nt = len(tris)
    out = np.full((nt, 3), -1, np.int32)
    if nt == 0:
        return out
    book = {}
    for t in range(nt):
        a, b, c = int(tris[t, 0]), int(tris[t, 1]), int(tris[t, 2])
        for k, (u, w, opp) in enumerate(((a, b, c), (b, c, a), (c, a, b))):
            key = (min(u, w), max(u, w))
            book.setdefault(key, []).append((t, k, opp))
    for key, lst in book.items():
        if len(lst) == 2:
            (t1, k1, o1), (t2, k2, o2) = lst
            out[t1, k1] = o2
            out[t2, k2] = o1
    return out


def _f64(v):
    return wp.array(np.ascontiguousarray(v, dtype=np.float64), dtype=wp.float64)


def _v3(v):
    return wp.array(np.ascontiguousarray(v, dtype=np.float64), dtype=wp.vec3d)


def _i32(v):
    return wp.array(np.ascontiguousarray(v, dtype=np.int32), dtype=wp.int32)


class Cloth4:
    def __init__(self, d, B, Eh, nu=.3, h=0., support=0, self_contact=False,
                 mu_s=.5, mu_c=.2, cap=None):
        self.d = d
        self.B = float(B)
        self.Eh = float(Eh)
        self.nu = float(nu)
        self.h = float(h)
        self.support = int(support)
        self.self_contact = bool(self_contact)
        self.mu_s = float(mu_s)
        self.mu_c = float(mu_c)
        n = len(d['verts'])
        self.n = n
        self.nt = len(d['tris'])
        self.nh = len(d['h_w'])
        self.x = _v3(d['verts'])
        self.x0 = _v3(d['verts'])
        self.xt = _v3(d['verts'])
        self.xtry = _v3(d['verts'])
        self.v = wp.zeros(n, dtype=wp.vec3d)
        self.vfree = wp.zeros(n, dtype=wp.vec3d)
        self.vtmp = wp.zeros(n, dtype=wp.vec3d)
        mass = np.asarray(d['masses'], dtype=np.float64)
        free = (np.asarray(d['fixed_flags']) == 0).astype(np.int32)
        self.mass_np = mass
        self.free_np = free
        self.m = _f64(mass)
        self.minv = _f64(np.where(free == 1, 1.0 / mass, 0.0))
        self.free = _i32(free)
        self.tgt = wp.zeros(n, dtype=wp.vec3d)
        self.dm = wp.zeros(n, dtype=wp.float64)
        self.fc = wp.zeros(n, dtype=wp.vec3d)
        self.acc = wp.zeros((n, 3), dtype=wp.int64)
        self.dacc = wp.zeros((n, 3), dtype=wp.int64)
        self.iacc = wp.zeros((n, 3), dtype=wp.int64)
        self.r = wp.zeros(n, dtype=wp.vec3d)
        self.rcg = wp.zeros(n, dtype=wp.vec3d)
        self.dir = wp.zeros(n, dtype=wp.vec3d)
        self.Ap = wp.zeros(n, dtype=wp.vec3d)
        self.z = wp.zeros(n, dtype=wp.vec3d)
        self.sol = wp.zeros(n, dtype=wp.vec3d)
        self.pre = wp.zeros(n, dtype=wp.vec3d)
        self.pre3 = wp.zeros(n, dtype=wp.mat33d)
        self.comp = wp.zeros(n, dtype=wp.mat33d)
        self.delassus = 'lumped'
        self.dacc9 = wp.zeros((n, 9), dtype=wp.int64)
        self._cg_graph = None
        self._cg_chunk = 0
        tris = np.asarray(d['tris'], dtype=np.int32)
        self.tris_np = tris
        self.t0, self.t1, self.t2 = _i32(tris[:, 0]), _i32(tris[:, 1]), _i32(tris[:, 2])
        self.tb = _v3(d['tri_b'])
        self.tc = _v3(d['tri_c'])
        self.ta = _f64(d['tri_area'])
        self.tri2d = wp.array(tris, dtype=wp.int32)
        self.tri_nbr = wp.array(triangle_neighbours(tris), dtype=wp.int32)
        self.hv = [_i32(d[k]) for k in ('h_v1', 'h_v2', 'h_v3', 'h_v4')]
        self.hw = _f64(d['h_w'])
        self.h0 = _f64(d['h_t0'])
        self.etri = wp.zeros(max(self.nt, 1), dtype=wp.float64)
        self.ehin = wp.zeros(max(self.nh, 1), dtype=wp.float64)
        self.enod = wp.zeros(n, dtype=wp.float64)
        # scalars, each its own 1-element array so kernels can consume them on device
        for name in ('s_rz', 's_rz2', 's_pAp', 's_alpha', 's_beta', 's_bb', 's_rr', 's_zero',
                     's_tmp'):
            setattr(self, name, wp.zeros(1, dtype=wp.float64))
        self.s_e = wp.zeros(3, dtype=wp.float64)
        self.grav_norm = float(np.linalg.norm(mass[free == 1] * GRAV))
        # contact
        self.edges = edge_list(tris)
        self.ne = len(self.edges)
        self.ev = wp.array(self.edges, dtype=wp.int32)
        self.xf = wp.zeros(n, dtype=wp.vec3)
        self.mid = wp.zeros(max(self.ne, 1), dtype=wp.vec3)
        self.cap = int(cap if cap is not None else max(8192, 10 * n))
        c = self.cap
        self.ckind = wp.zeros(c, dtype=wp.int32)
        self.ca = wp.zeros(c, dtype=wp.int32)
        self.cb = wp.zeros(c, dtype=wp.int32)
        self.cids = wp.zeros((c, 4), dtype=wp.int32)
        self.cw = wp.zeros((c, 4), dtype=wp.float64)
        self.cn = wp.zeros(c, dtype=wp.vec3d)
        self.ct1 = wp.zeros(c, dtype=wp.vec3d)
        self.ct2 = wp.zeros(c, dtype=wp.vec3d)
        self.cgap = wp.zeros(c, dtype=wp.float64)
        self.cmu = wp.zeros(c, dtype=wp.float64)
        self.crho = wp.zeros(c, dtype=wp.float64)
        self.clam = wp.zeros(c, dtype=wp.vec3d)
        self.cres = wp.zeros(c, dtype=wp.float64)
        self.ccount = wp.zeros(1, dtype=wp.int32)
        self.cover = wp.zeros(1, dtype=wp.int32)
        self.ncnt = wp.zeros(n, dtype=wp.int32)
        self.gstat = wp.zeros(6, dtype=wp.int64)
        self.nodebuf = wp.zeros(n, dtype=wp.float64)
        # reduction ladders
        self.rb = self._ladder(max(n, self.nt, self.nh, 1))
        self.rbc = self._ladder(c)
        rest = d['verts'].astype(float)
        if self.nt:
            p = rest[tris]
            cen = p.mean(1)
            self.tri_r = float(np.linalg.norm(p - cen[:, None, :], axis=2).max())
        else:
            self.tri_r = 0.
        self.edge_len = float(np.linalg.norm(rest[self.edges[:, 1]] - rest[self.edges[:, 0]],
                                             axis=1).max()) if self.ne else 0.
        self.grid_v = wp.HashGrid(64, 64, 64)
        self.grid_e = wp.HashGrid(64, 64, 64)
        self.C = 0
        self.dt_now = 1e-3
        self.log = {}

    @staticmethod
    def _ladder(nmax):
        out = []
        cur = nmax
        while True:
            cur = (cur + T_BLOCK - 1) // T_BLOCK
            out.append(wp.zeros(max(cur, 1), dtype=wp.float64))
            if cur <= 1:
                break
        return out

    # ------------------------------------------------------------------ reductions
    def _fold(self, n, bufs, dst, di, kern):
        cur = (n + T_BLOCK - 1) // T_BLOCK
        lvl = 0
        while cur > 1:
            nxt = (cur + T_BLOCK - 1) // T_BLOCK
            wp.launch(kern, dim=nxt, inputs=[bufs[lvl], cur, T_BLOCK, bufs[lvl + 1]])
            lvl += 1
            cur = nxt
        wp.launch(K.k_copy, dim=1, inputs=[bufs[lvl], 0, dst, di])

    def dot(self, a, b, dst, di=0):
        n = self.n
        nb = (n + T_BLOCK - 1) // T_BLOCK
        wp.launch(K.k_dot_block, dim=nb, inputs=[a, b, n, T_BLOCK, self.rb[0]])
        self._fold(n, self.rb, dst, di, K.k_sum_block)

    def reduce(self, src, n, dst, di=0, bufs=None, mx=False):
        bufs = self.rb if bufs is None else bufs
        kern = K.k_max_block if mx else K.k_sum_block
        nb = (n + T_BLOCK - 1) // T_BLOCK
        wp.launch(kern, dim=nb, inputs=[src, n, T_BLOCK, bufs[0]])
        self._fold(n, bufs, dst, di, kern)

    # ------------------------------------------------------------------ physics
    def forces(self, xa, acc):
        acc.zero_()
        if self.B and self.nh:
            wp.launch(E.compute_hinge_forces_kernel, dim=self.nh,
                      inputs=[xa, *self.hv, self.hw, self.h0, self.B, acc])
        wp.launch(E.compute_pipkin_membrane_forces_kernel, dim=self.nt,
                  inputs=[xa, self.t0, self.t1, self.t2, self.tb, self.tc, self.ta,
                          self.Eh, self.nu, self.Eh / (1 - self.nu ** 2), acc])

    def energy(self, xa):
        wp.launch(K.k_tri_energy, dim=self.nt,
                  inputs=[xa, self.t0, self.t1, self.t2, self.tb, self.tc, self.ta,
                          self.Eh, self.nu, self.etri])
        self.reduce(self.etri, self.nt, self.s_e, 0)
        if self.B and self.nh:
            wp.launch(K.k_hinge_energy, dim=self.nh,
                      inputs=[xa, *self.hv, self.hw, self.h0, self.B, self.ehin])
            self.reduce(self.ehin, self.nh, self.s_e, 1)
        else:
            wp.launch(K.k_copy, dim=1, inputs=[self.s_zero, 0, self.s_e, 1])
        wp.launch(K.k_node_energy, dim=self.n,
                  inputs=[xa, self.m, self.tgt, self.dm, self.fc, self.free, self.enod])
        self.reduce(self.enod, self.n, self.s_e, 2)
        s = self.s_e.numpy()
        return float(s[0] + s[1] + s[2])

    def residual(self, xa):
        self.forces(xa, self.acc)
        wp.launch(K.k_residual, dim=self.n,
                  inputs=[self.acc, self.m, self.dm, xa, self.tgt, self.fc, self.free, self.r])

    def matvec(self, pv, out):
        self.dacc.zero_()
        wp.launch(K.k_tri_matvec, dim=self.nt,
                  inputs=[self.xt, pv, self.t0, self.t1, self.t2, self.tb, self.tc, self.ta,
                          self.Eh, self.nu, self.dacc])
        if self.B and self.nh:
            wp.launch(K.k_hinge_matvec, dim=self.nh,
                      inputs=[self.xt, pv, *self.hv, self.hw, self.B, self.dacc])
        wp.launch(K.k_apply_A, dim=self.n, inputs=[self.dacc, pv, self.dm, self.free, out])

    def precond(self):
        self.dacc9.zero_()
        wp.launch(K.k_tri_diag3, dim=self.nt,
                  inputs=[self.xt, self.t0, self.t1, self.t2, self.tb, self.tc, self.ta,
                          self.Eh, self.nu, self.dacc9])
        if self.B and self.nh:
            wp.launch(K.k_hinge_diag3, dim=self.nh,
                      inputs=[self.xt, *self.hv, self.hw, self.B, self.dacc9])
        wp.launch(K.k_precond3, dim=self.n, inputs=[self.dacc9, self.dm, self.free, self.pre3])

    def _cg_body(self):
        n = self.n
        self.matvec(self.dir, self.Ap)
        self.dot(self.dir, self.Ap, self.s_pAp)
        wp.launch(K.k_ratio, dim=1, inputs=[self.s_rz, self.s_pAp, self.s_alpha, 0])
        wp.launch(K.k_axpy, dim=n, inputs=[self.sol, self.s_alpha, 0, 1.0, self.dir])
        wp.launch(K.k_axpy, dim=n, inputs=[self.rcg, self.s_alpha, 0, -1.0, self.Ap])
        wp.launch(K.k_jacobi3, dim=n, inputs=[self.rcg, self.pre3, self.z])
        self.dot(self.rcg, self.z, self.s_rz2)
        wp.launch(K.k_ratio, dim=1, inputs=[self.s_rz2, self.s_rz, self.s_beta, 0])
        wp.launch(K.k_xpby, dim=n, inputs=[self.dir, self.s_beta, 0, self.z])
        wp.launch(K.k_copy, dim=1, inputs=[self.s_rz2, 0, self.s_rz, 0])

    def cg(self, maxiter=200, rtol=1e-3, check=25, graph=True):
        n = self.n
        wp.launch(K.k_zero3, dim=n, inputs=[self.sol])
        wp.copy(self.rcg, self.r)
        wp.launch(K.k_jacobi3, dim=n, inputs=[self.rcg, self.pre3, self.z])
        wp.copy(self.dir, self.z)
        self.dot(self.rcg, self.rcg, self.s_bb)
        self.dot(self.rcg, self.z, self.s_rz)
        b2 = float(self.s_bb.numpy()[0])
        if b2 <= 0.0:
            return 0, 0.0
        if graph and self._cg_chunk != check:
            wp.synchronize()
            with wp.ScopedCapture() as cap:
                for _ in range(check):
                    self._cg_body()
            self._cg_graph = cap.graph
            self._cg_chunk = check
        it = 0
        rel = 1.0
        while it < maxiter:
            if graph:
                wp.capture_launch(self._cg_graph)
                it += check
            else:
                for _ in range(min(check, maxiter - it)):
                    self._cg_body()
                it += min(check, maxiter - it)
            self.dot(self.rcg, self.rcg, self.s_rr)
            rel = float(np.sqrt(max(self.s_rr.numpy()[0], 0.0) / b2))
            if rel <= rtol or not np.isfinite(rel):
                break
        return it, rel

    # ------------------------------------------------------------------ contact
    def build_candidates(self, margin, self_margin=None):
        """margin: how far a node may travel toward the rigid support inside one step (dt*v).
        self_margin: the same for cloth-cloth pairs, where the RELATIVE approach speed of two
        layers is what matters, not the common fall speed. Using dt*v_max for both makes the
        candidate threshold exceed the mesh spacing at large dt and the pair list explodes
        (round-4 first Cusick attempt: 1.6e6 pairs, 9e7 dropped). Dropped pairs are counted and
        reported; a run with overflow > 0 is not a contact-certified run."""
        self.ccount.zero_()
        self.cover.zero_()
        thr = self.h + (self.h if self_margin is None else self_margin)
        if self.support:
            wp.launch(CT.k_emit_support, dim=self.n,
                      inputs=[self.x0, self.free, self.support, margin, .5 * self.h,
                              self.ccount, self.cap, self.ckind, self.ca, self.cb, self.cover])
        if self.self_contact:
            wp.launch(K.k_to_f32, dim=self.n, inputs=[self.x0, self.xf])
            qr = float(self.tri_r + thr)
            self.grid_v.build(self.xf, qr)
            wp.launch(CT.k_emit_vt, dim=self.nt,
                      inputs=[self.grid_v.id, self.x0, self.xf, self.tri2d, self.tri_nbr,
                              qr, thr,
                              self.ccount, self.cap, self.ckind, self.ca, self.cb, self.cover])
            wp.launch(CT.k_edge_mid, dim=self.ne, inputs=[self.x0, self.ev, self.mid])
            qre = float(self.edge_len + thr)
            self.grid_e.build(self.mid, qre)
            wp.launch(CT.k_emit_ee, dim=self.ne,
                      inputs=[self.grid_e.id, self.x0, self.mid, self.ev, qre, thr,
                              self.ccount, self.cap, self.ckind, self.ca, self.cb, self.cover])
        c = int(self.ccount.numpy()[0])
        self.C = min(c, self.cap)
        return c, int(self.cover.numpy()[0])

    def contact_geometry(self, xa, dt=1e-3):
        self.dt_now = dt
        if self.C == 0:
            return
        wp.launch(CT.k_geom, dim=self.C,
                  inputs=[xa, self.tri2d, self.ev, self.ckind, self.ca, self.cb, self.C,
                          self.h, self.mu_s, self.mu_c, self.cids, self.cw, self.cn,
                          self.ct1, self.ct2, self.cgap, self.cmu])
        self.ncnt.zero_()
        wp.launch(CT.k_count_nodes, dim=self.C, inputs=[self.cids, self.C, self.ncnt])
        wp.launch(CT.k_rho, dim=self.C,
                  inputs=[self.cids, self.cw, self.C, self.comp, self.dt_now, self.ncnt,
                          self.crho])

    def ncp_solve(self, dt, iters=100, check=25, tol=1e-12, desaxce=1):
        if self.C == 0:
            wp.copy(self.v, self.vfree)
            return 0, 0.0
        n = self.n
        it = 0
        res = 0.0
        wp.launch(CT.k_apply, dim=n, inputs=[self.vfree, self.iacc, self.comp, dt, self.v])
        while it < iters:
            nrun = min(check, iters - it)
            for _ in range(nrun):
                wp.launch(CT.k_sweep, dim=self.C,
                          inputs=[self.cids, self.cw, self.C, self.cn, self.ct1, self.ct2,
                                  self.cgap, self.cmu, self.crho, self.v, dt, desaxce,
                                  self.clam, self.iacc])
                wp.launch(CT.k_apply, dim=n,
                          inputs=[self.vfree, self.iacc, self.comp, dt, self.v])
            it += nrun
            wp.launch(CT.k_resid, dim=self.C,
                      inputs=[self.cids, self.cw, self.C, self.cn, self.ct1, self.ct2,
                              self.cgap, self.cmu, self.crho, self.v, dt, desaxce,
                              self.clam, self.cres])
            self.reduce(self.cres, self.C, self.s_tmp, 0, bufs=self.rbc, mx=True)
            res = float(self.s_tmp.numpy()[0])
            if res <= tol:
                break
        return it, res

    # ------------------------------------------------------------------ one step
    def step(self, dt=1e-3, damp=100., newton=5, tol=1e-3, cg_iter=200, cg_tol=1e-3,
             rounds=3, ls=12, c1=1e-4, ncp_iter=100, ncp_tol=1e-12, margin=None,
             self_margin=None):
        n = self.n
        wp.copy(self.x0, self.x)
        wp.launch(K.k_set_target, dim=n,
                  inputs=[self.x0, self.v, self.m, dt, damp, self.tgt, self.dm])
        wp.launch(K.k_zero3, dim=n, inputs=[self.fc])
        wp.launch(K.k_lumped_comp, dim=n, inputs=[self.minv, dt, damp, self.comp])
        self.iacc.zero_()
        wp.launch(K.k_zero3, dim=self.cap, inputs=[self.clam])
        contact_on = bool(self.support or self.self_contact)
        info = dict(newton=[], cg=[], alpha=[], contacts=0, overflow=0, ncp_iters=[],
                    ncp_res=[], rounds=0, dx_estimate_m=0.0)
        if contact_on:
            if margin is None:
                margin = self.h
            c, over = self.build_candidates(margin, self_margin)
            info['contacts'] = c
            info['overflow'] = over
            info['margin'] = float(margin)
            # One-step NCP (Stewart-Trinkle / Anitescu form): gap and frame are taken at the
            # START of the step, so the constraint g0 + dt*u_n >= 0 is a fixed condition on the
            # end-of-step velocity. Re-evaluating the gap at an iterate that already carries the
            # previous impulse double counts it and makes the contact/elastic fixed point cycle.
            self.contact_geometry(self.x0, dt)
        nrounds = rounds if contact_on else 1
        for rnd in range(nrounds):
            info['rounds'] = rnd + 1
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
            if not contact_on:
                break
            wp.launch(K.k_velocity, dim=n, inputs=[self.x, self.x0, dt, self.free, self.vtmp])
            wp.launch(CT.k_vfree, dim=n,
                      inputs=[self.vtmp, self.iacc, self.comp, dt, self.vfree])
            nit, nres = self.ncp_solve(dt, iters=ncp_iter, tol=ncp_tol)
            info['ncp_iters'].append(nit)
            info['ncp_res'].append(nres)
            wp.launch(K.k_advance, dim=n, inputs=[self.x0, self.v, dt, self.free, self.x])
            wp.launch(CT.k_contact_force, dim=n, inputs=[self.iacc, dt, self.fc])
        wp.launch(K.k_velocity, dim=n, inputs=[self.x, self.x0, dt, self.free, self.v])
        if contact_on and self.C:
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

    # ------------------------------------------------------------------ diagnostics
    def kinetic(self):
        wp.launch(K.k_kinetic, dim=self.n, inputs=[self.v, self.m, self.nodebuf])
        self.reduce(self.nodebuf, self.n, self.s_tmp, 0)
        ke = float(self.s_tmp.numpy()[0])
        wp.launch(K.k_speed, dim=self.n, inputs=[self.v, self.nodebuf])
        self.reduce(self.nodebuf, self.n, self.s_tmp, 0, mx=True)
        return ke, float(self.s_tmp.numpy()[0])

    def static_residual(self):
        """||f_int + f_grav|| / ||Mg|| on free nodes: the equilibrium certificate."""
        self.forces(self.x, self.acc)
        f = self.acc.numpy().astype(float) / 1e12
        f[:, 1] -= self.mass_np * GRAV
        f = f[self.free_np == 1]
        return float(np.linalg.norm(f) / self.grav_norm)
