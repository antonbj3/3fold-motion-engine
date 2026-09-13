"""Prepared contact selection with the unchanged noncontracting chunk solver."""
from motion_engine.contact_engine_gpu_prepared_planar import PreparedPlanarContactEngine
from motion_engine.contact_engine_gpu_applied_warm import AppliedWarmContactEngine
from motion_engine.contact_engine_gpu_unfused_solve import unfused_fixed
from motion_engine.contact_engine_gpu_unfused_position import unfused_position


class PreparedNoncontractingContactEngine(PreparedPlanarContactEngine):
    name = "prepared_noncontracting_contact_gpu"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.KC = dict(self.KC)
        self.KC["chunk_vel_r"] = unfused_fixed
        self.KC["chunk_pos_r"] = unfused_position

    def _chunk_launch(self, ncol, dim, sdt):
        # Preserve the baseline warm/velocity/position launch order. The inherited
        # island specialization uses a different floating-point contraction policy.
        return AppliedWarmContactEngine._chunk_launch(self, ncol, dim, sdt)
