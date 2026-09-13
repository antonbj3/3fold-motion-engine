"""Reuse pair grouping only after exact ordered endpoint validation."""
from motion_engine.contact_engine_gpu_island_solve import IslandSolveContactEngine


def build_layout_compare(wp):
    @wp.kernel
    def compare(contacts: int,bodies: int,a: wp.array(dtype=int),b: wp.array(dtype=int),
                old_a: wp.array(dtype=int),old_b: wp.array(dtype=int),mass: wp.array(dtype=float),
                old_mass: wp.array(dtype=float),changed: wp.array(dtype=int)):
        i=wp.tid()
        if i<contacts:
            if a[i]!=old_a[i] or b[i]!=old_b[i]:wp.atomic_max(changed,0,1)
        if i<bodies:
            if mass[i]!=old_mass[i]:wp.atomic_max(changed,1,1)
    return compare


class PairLayoutCacheContactEngine(IslandSolveContactEngine):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._layout_compare=build_layout_compare(self.wp)

    def _build(self):
        super()._build()
        self._layout_a=self.wp.empty_like(self.sbi)
        self._layout_b=self.wp.empty_like(self.sbj)
        self._layout_changed=self.wp.zeros(2,dtype=int,device=self.dev)
        self._layout_contacts=-1
        self._layout_pairs=0
        self._layout_reused=False
        self._layout_mass_changed=True
        self.pair_layout_hits=0
        self.pair_layout_misses=0

    def _build_pairs(self,contacts):
        self._layout_reused=False
        self._layout_mass_changed=True
        if contacts==self._layout_contacts:
            self._layout_changed.zero_()
            self.wp.launch(self._layout_compare,max(contacts,self.N),
                inputs=[contacts,self.N,self.sbi,self.sbj,self._layout_a,self._layout_b,
                        self.invM,self._cached_mass,self._layout_changed],device=self.dev)
            flags=self._layout_changed.numpy()
            self._layout_mass_changed=bool(flags[1])
            if flags[0]==0:
                self._layout_reused=True
                self.pair_layout_hits+=1
                return self._layout_pairs
        pairs=super()._build_pairs(contacts)
        self.wp.copy(self._layout_a,self.sbi,count=contacts)
        self.wp.copy(self._layout_b,self.sbj,count=contacts)
        self._layout_contacts=contacts
        self._layout_pairs=pairs
        self.pair_layout_misses+=1
        return pairs

    def _color_pairs(self,pairs):
        reuse=self._layout_reused and not self._layout_mass_changed and pairs==self._layout_pairs and pairs==self._cached_pairs
        self._layout_reused=False
        if reuse:
            self.color_cache_hits+=1
            return self._cached_colors
        return super()._color_pairs(pairs)


if __name__=='__main__':
    import runpy
    from pathlib import Path
    runpy.run_path(str(Path(__file__).resolve().parents[2]/'probes/innovation_pair_layout_cache.py'),run_name='__main__')
