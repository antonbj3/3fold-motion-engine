"""Host coarse normal correction beside the frozen GPU pair solver.

One coarse coordinate per body pair, obtained by averaging its contact Jacobians.
The dense prototype is limited to 64 bodies; larger scenes use the baseline.
"""
import numpy as np
from motion_engine.contact_engine_gpu_colored import GraphColoredContactEngine


class CoarsePairContactEngine(GraphColoredContactEngine):
    name = "coarse_pair_contact_gpu"

    def __init__(self, max_coarse_bodies=64, **kw):
        super().__init__(manifold_reduce=True, graph_capture=True, pair_chunks=True, **kw)
        self.max_coarse_bodies = max_coarse_bodies
        self.coarse_applied = 0
        self.coarse_singular = 0
        self.coarse_large_skipped = 0

    def _color_pairs(self, pairs):
        colors = super()._color_pairs(pairs)
        if self.N > self.max_coarse_bodies:
            self.coarse_large_skipped += 1
            return colors
        c = self.n_contacts_solved
        starts = self.pstart.numpy()[:pairs]
        counts = self.pcount.numpy()[:pairs]
        order = self.pvalP.numpy()[:c]
        bi, bj = self.sbi.numpy()[:c], self.sbj.numpy()[:c]
        point_a, point_b = self.spA.numpy()[:c], self.spB.numpy()[:c]
        normal = self.sn.numpy()[:c].astype(np.float64)
        center = self.xc.numpy().astype(np.float64)
        inertia = self.invIw.numpy().astype(np.float64)
        mass = self.invM.numpy().astype(np.float64)
        v, w = self.v.numpy(), self.w.numpy()
        jn = self.jn.numpy()[:c].copy()
        jac = np.zeros((c, self.N, 6))
        for i in range(c):
            jac[i, bi[i], :3] = normal[i]
            jac[i, bi[i], 3:] = np.cross(point_a[i]-center[bi[i]], normal[i])
            if bj[i] >= 0:
                jac[i, bj[i], :3] = -normal[i]
                jac[i, bj[i], 3:] = -np.cross(point_b[i]-center[bj[i]], normal[i])
        coarse = np.empty((pairs, self.N, 6))
        lower = np.empty(pairs)
        groups = []
        for p in range(pairs):
            indices = order[starts[p]:starts[p]+counts[p]]
            groups.append(indices)
            coarse[p] = jac[indices].mean(axis=0)
            lower[p] = -float(jn[indices].min())*len(indices)
        weighted = coarse.copy()
        weighted[:, :, :3] *= mass[None, :, None]
        weighted[:, :, 3:] = np.einsum("nij,pnj->pni", inertia, coarse[:, :, 3:])
        matrix = weighted.reshape(pairs, -1) @ coarse.reshape(pairs, -1).T
        velocity = np.concatenate((v, w), axis=1)
        rhs = -np.einsum("pni,ni->p", coarse, velocity)
        # A bounded active-set solve. Fix a violating coordinate at its lower bound;
        # re-solve the remaining block. No regularisation or tolerance inflation.
        active = np.zeros(pairs, dtype=bool)
        delta = np.zeros(pairs)
        try:
            for _ in range(pairs+1):
                free = ~active
                if free.any():
                    delta[free] = np.linalg.solve(matrix[np.ix_(free, free)],
                        rhs[free]-matrix[np.ix_(free, active)] @ delta[active])
                violated = free & (delta < lower)
                if not violated.any():
                    break
                active |= violated
                delta[violated] = lower[violated]
        except np.linalg.LinAlgError:
            self.coarse_singular += 1
            return colors
        if not np.isfinite(delta).all():
            raise RuntimeError("Non-finite coarse correction")
        increments = np.zeros(c)
        for p, indices in enumerate(groups):
            increments[indices] = delta[p]/len(indices)
        # Round to the delivered accumulated impulses first. Apply their actual
        # difference to velocities, so projection/rounding cannot inject an unmatched impulse.
        updated = np.maximum(0., jn.astype(np.float64)+increments).astype(np.float32)
        actual = updated.astype(np.float64)-jn.astype(np.float64)
        impulse = np.einsum("c,cni->ni", actual, jac)
        velocity[:, :3] += impulse[:, :3]*mass[:, None]
        velocity[:, 3:] += np.einsum("nij,nj->ni", inertia, impulse[:, 3:])
        wp = self.wp
        wp.copy(self.v, wp.array(velocity[:, :3], dtype=wp.vec3, device=self.dev))
        wp.copy(self.w, wp.array(velocity[:, 3:], dtype=wp.vec3, device=self.dev))
        wp.copy(self.jn, wp.array(updated, dtype=float, device=self.dev), count=c)
        self.coarse_applied += 1
        return colors


if __name__ == "__main__":
    import runpy
    from pathlib import Path
    import sys
    sys.argv = ["innovation_stack_coarse_probe.py"]
    runpy.run_path(str(Path(__file__).resolve().parents[2]/"probes"/"innovation_stack_coarse_probe.py"), run_name="__main__")
