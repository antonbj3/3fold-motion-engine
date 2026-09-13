"""Measured fixed128 launch for the unchanged noncontracting component kernel."""
from motion_engine.contact_engine_gpu_noncontracting_island import NoncontractingIslandContactEngine


class CompactIslandContactEngine(NoncontractingIslandContactEngine):
    name = "compact_island_contact_gpu"

    def _chunk_launch(self, ncol, dim, sdt):
        if self.max_island_pairs > 64:
            return super()._chunk_launch(ncol, dim, sdt)
        for ci in range(ncol):
            self.wp.launch(self._apply_warm, dim, inputs=[self.porder, self.pvalP, self.pstart, self.pcount, self.pbi, self.pbj, ci, self.cstart, self.ccount, self.spA, self.spB, self.sn, self.jn, self.jt1, self.jt2, self.xc, self.v, self.w, self.invIw, self.invM], device=self.dev)
        self.wp.launch(self._noncontracting_island_solve, self.island_count, inputs=[self._island_pairs, self._island_starts, self._island_counts, self.pvalP, self.pstart, self.pcount, self.pbi, self.pbj, self.sbi, self.sbj, self.spA, self.spB, self.sn, self.spen, self.jn, self.jt1, self.jt2, self.jp, self.xc, self.v, self.w, self.pv, self.po, self.invIw, self.invM, self.mu, self.t2_damp, sdt, self.vit, self.pit], device=self.dev, block_dim=128)
