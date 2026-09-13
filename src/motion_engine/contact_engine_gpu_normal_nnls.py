"""Experimental bounded host normal solve before the frozen GPU pair sweeps."""
import numpy as np
from scipy.optimize import nnls

from motion_engine.contact_engine_gpu_colored import GraphColoredContactEngine


class NormalNNLSContactEngine(GraphColoredContactEngine):
    name = "normal_nnls_contact_gpu"

    def __init__(self, max_normal_bodies=64, **kw):
        super().__init__(manifold_reduce=True, graph_capture=True, pair_chunks=True, **kw)
        self.max_normal_bodies = max_normal_bodies
        self.normal_applied = 0
        self.normal_large_skipped = 0
        self.normal_max_kkt = 0.

    def _color_pairs(self, pairs):
        colors = super()._color_pairs(pairs)
        if self.N > self.max_normal_bodies:
            self.normal_large_skipped += 1
            return colors
        c = self.n_contacts_solved
        bi, bj = self.sbi.numpy()[:c], self.sbj.numpy()[:c]
        pa, pb = self.spA.numpy()[:c], self.spB.numpy()[:c]
        normals = self.sn.numpy()[:c].astype(float)
        center = self.xc.numpy().astype(float)
        inertia = self.invIw.numpy().astype(float)
        mass = self.invM.numpy().astype(float)
        velocity = np.concatenate((self.v.numpy(), self.w.numpy()), axis=1).astype(float)
        jn = self.jn.numpy()[:c].astype(float)
        jac = np.zeros((c, self.N, 6))
        for i in range(c):
            a, b, normal = bi[i], bj[i], normals[i]
            jac[i, a, :3] = normal
            jac[i, a, 3:] = np.cross(pa[i].astype(float)-center[a], normal)
            if b >= 0:
                jac[i, b, :3] = -normal
                jac[i, b, 3:] = -np.cross(pb[i].astype(float)-center[b], normal)
        eig, vec = np.linalg.eigh(inertia)
        if np.any(eig <= 0) or np.any(mass <= 0):
            raise RuntimeError("Positive dynamic inverse mass/inertia required")
        root_i = (vec * np.sqrt(eig)[:, None, :]) @ vec.transpose(0, 2, 1)
        weighted = jac.copy()
        weighted[:, :, :3] *= np.sqrt(mass)[None, :, None]
        weighted[:, :, 3:] = np.einsum("cni,nij->cnj", jac[:, :, 3:], root_i)
        a = weighted.reshape(c, -1).T
        u = velocity.copy()
        u[:, :3] /= np.sqrt(mass)[:, None]
        u[:, 3:] = np.linalg.solve(root_i, velocity[:, 3:, None])[:, :, 0]
        # Minimise final mass-weighted velocity over absolute nonnegative normal
        # impulses, with frozen tangential impulses. This is not a friction solve.
        rhs = a @ jn - u.ravel()
        solution, _ = nnls(a, rhs, maxiter=3*c)
        gradient = a.T @ (a @ solution-rhs)
        kkt = max(float(np.maximum(-gradient, 0).max()),
                  float(np.abs(solution*gradient).max()))
        self.normal_max_kkt = max(self.normal_max_kkt, kkt)
        if not np.isfinite(solution).all() or kkt > 1e-8:
            raise RuntimeError(f"Normal NNLS KKT gate failed: {kkt}")
        updated = solution.astype(np.float32)
        actual = updated.astype(float)-jn
        impulse = np.einsum("c,cni->ni", actual, jac)
        velocity[:, :3] += impulse[:, :3]*mass[:, None]
        velocity[:, 3:] += np.einsum("nij,nj->ni", inertia, impulse[:, 3:])
        wp = self.wp
        wp.copy(self.v, wp.array(velocity[:, :3], dtype=wp.vec3, device=self.dev))
        wp.copy(self.w, wp.array(velocity[:, 3:], dtype=wp.vec3, device=self.dev))
        wp.copy(self.jn, wp.array(updated, dtype=float, device=self.dev), count=c)
        self.normal_applied += 1
        return colors


if __name__ == "__main__":
    import runpy
    from pathlib import Path
    runpy.run_path(str(Path(__file__).resolve().parents[2]/"probes"/"innovation_stack_normal_nnls_probe.py"), run_name="__main__")
