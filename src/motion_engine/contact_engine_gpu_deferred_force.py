"""Applied-warm solver with deferred host force evaluation and complete steps."""
import numpy as np
from motion_engine.contact_engine_gpu_applied_warm import AppliedWarmContactEngine
from motion_engine.contact_engine_gpu_colored import _R, _MAXC


class DeferredForceContactEngine(AppliedWarmContactEngine):
    def _build(self):
        super()._build()
        self._force_before = self.wp.empty_like(self.v)
        self._force_dt = None

    def step(s, dt, substeps=1):
        if not s._built:
            s._build()
        wp = s.wp
        K = s.K
        KC = s.KC
        for _ in range(substeps):
            sdt = dt / substeps
            with s._stage("generate"):
                wp.launch(K["grav"], s.N, inputs=[s.v, s.invM, sdt], device=s.dev)
                wp.launch(K["invIw"], s.N, inputs=[s.q, s.IbInv, s._kin_wp, s.invIw], device=s.dev)
                wp.launch(K["worldp"], s.N * s.P,
                          inputs=[s.xc, s.q, s.rest, s.P, s.allp, s.owner], device=s.dev)
                s.grid.build(s.allp, 2.0 * _R)
                s.cnt.zero_()
                wp.launch(KC["gen_fid"], s.N * s.P,
                          inputs=[s.allp, s.owner, s.grid.id, s.cnt, s.cbi, s.cbj,
                                  s.cpA, s.cpB, s.cn, s.cpen, s.cfa, s.cfb], device=s.dev)
                C = min(int(s.cnt.numpy()[0]), _MAXC)
            s.n_contacts = C
            wp.copy(s._force_before, s.v)
            if C > 0:
                C = s._device_manifold(C)
                s.n_contacts_solved = C
                s.pv.zero_()
                s.po.zero_()
                with s._stage("pairs"):
                    P = s._build_pairs(C)
                s.n_pairs = P
                with s._stage("colour"):
                    ncol = s._color_pairs(P)
                s.n_colors = ncol
                with s._stage("solve"):
                    wp.capture_launch(s._solve_chunk_graph(P, ncol, sdt))
                if s.warm_start:
                    with s._stage("warm_store"):
                        s._store_warm_device(C)
            else:
                s.n_colors = 0
                s.n_contacts_solved = 0
                s.n_pairs = 0
                s.pv.zero_()
                s.po.zero_()
                s._ws_keys = np.zeros(0, np.int64)
                s._ws_imp = np.zeros((0, 3))
                s._ws_n = 0
            with s._stage("integrate"):
                wp.launch(K["integ"], s.N,
                          inputs=[s.xc, s.q, s.v, s.w, s.pv, s.po, s.invM, sdt], device=s.dev)
            s._force_dt = sdt
            s._prof_steps += 1
        wp.synchronize_device(s.dev)

    def contact_forces(self):
        if self._force_dt is None:
            return super().contact_forces()
        before = self._force_before.numpy()
        after = self.v.numpy()
        force = np.zeros((self.N, 3))
        dynamic = self._kinarr == 0
        force[dynamic] = self._M[dynamic, None] * (after[dynamic] - before[dynamic]) / self._force_dt
        return force


if __name__ == "__main__":
    import runpy
    from pathlib import Path
    runpy.run_path(str(Path(__file__).resolve().parents[2] /
                      "probes/innovation_stack_deferred_force.py"), run_name="__main__")
