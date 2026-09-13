"""Separate wide-lattice grid allocation; all frozen solver operations inherited."""
from motion_engine.contact_engine_gpu_colored import GraphColoredContactEngine


class WideGridContactEngine(GraphColoredContactEngine):
    name = "wide_grid_contact_gpu"

    def _build(self):
        super()._build()
        self.grid = self.wp.HashGrid(256, 256, 8, device=self.dev)


if __name__ == "__main__":
    import runpy
    from pathlib import Path
    runpy.run_path(str(Path(__file__).resolve().parents[2]/"probes"/"innovation_wide_grid_engine_probe.py"),run_name="__main__")
