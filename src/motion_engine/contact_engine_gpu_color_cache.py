"""Reuse pair colours only after exact graph-input comparison on the device."""
from motion_engine.contact_engine_gpu_deferred_force import DeferredForceContactEngine


def build_compare(wp):
    @wp.kernel
    def compare(pairs: int, bodies: int, a: wp.array(dtype=int), b: wp.array(dtype=int),
                old_a: wp.array(dtype=int), old_b: wp.array(dtype=int),
                mass: wp.array(dtype=float), old_mass: wp.array(dtype=float),
                mismatch: wp.array(dtype=int)):
        i = wp.tid()
        if i < pairs:
            if a[i] != old_a[i] or b[i] != old_b[i]:
                wp.atomic_max(mismatch, 0, 1)
        if i < bodies:
            if mass[i] != old_mass[i]:
                wp.atomic_max(mismatch, 0, 1)
    return compare


class ColorCacheContactEngine(DeferredForceContactEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._compare_graph = build_compare(self.wp)

    def _build(self):
        super()._build()
        self._cached_a = self.wp.empty_like(self.pbi)
        self._cached_b = self.wp.empty_like(self.pbj)
        self._cached_mass = self.wp.empty_like(self.invM)
        self._graph_mismatch = self.wp.zeros(1, dtype=int, device=self.dev)
        self._cached_pairs = -1
        self._cached_colors = 0
        self.color_cache_hits = 0
        self.color_cache_misses = 0

    def _color_pairs(self, pairs):
        wp = self.wp
        if pairs == self._cached_pairs:
            self._graph_mismatch.zero_()
            wp.launch(self._compare_graph, max(pairs, self.N),
                      inputs=[pairs, self.N, self.pbi, self.pbj, self._cached_a, self._cached_b,
                              self.invM, self._cached_mass, self._graph_mismatch], device=self.dev)
            if int(self._graph_mismatch.numpy()[0]) == 0:
                self.color_cache_hits += 1
                return self._cached_colors
        colors = super()._color_pairs(pairs)
        wp.copy(self._cached_a, self.pbi, count=pairs)
        wp.copy(self._cached_b, self.pbj, count=pairs)
        wp.copy(self._cached_mass, self.invM)
        self._cached_pairs = pairs
        self._cached_colors = colors
        self.color_cache_misses += 1
        return colors


if __name__ == "__main__":
    import runpy
    from pathlib import Path
    runpy.run_path(str(Path(__file__).resolve().parents[2] /
                      "probes/innovation_stack_color_cache.py"), run_name="__main__")
