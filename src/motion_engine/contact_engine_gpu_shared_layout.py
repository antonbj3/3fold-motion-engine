"""Share selection and exact pair-layout validation in one host readback."""
from motion_engine.contact_engine_gpu_colored import _MAXSEG
from motion_engine.contact_engine_gpu_compact_island import CompactIslandContactEngine
from motion_engine.contact_engine_gpu_pair_layout_cache import PairLayoutCacheContactEngine


def build_shared_layout_check(wp):
    @wp.kernel
    def compare(previous: int, bodies: int, flags: wp.array(dtype=int),
                a: wp.array(dtype=int), b: wp.array(dtype=int),
                old_a: wp.array(dtype=int), old_b: wp.array(dtype=int),
                mass: wp.array(dtype=float), old_mass: wp.array(dtype=float)):
        i = wp.tid()
        contacts = flags[0]
        if contacts != previous:
            if i == 0:
                wp.atomic_max(flags, 2, 1)
                wp.atomic_max(flags, 3, 1)
        else:
            if i < contacts:
                if a[i] != old_a[i] or b[i] != old_b[i]:
                    wp.atomic_max(flags, 2, 1)
            if i < bodies:
                if mass[i] != old_mass[i]:
                    wp.atomic_max(flags, 3, 1)
    return compare


class SharedLayoutContactEngine(CompactIslandContactEngine):
    name = "shared_layout_contact_gpu"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._shared_layout_check = build_shared_layout_check(self.wp)

    def _build(self):
        super()._build()
        self.dflags = self.wp.zeros(4, dtype=int, device=self.dev)
        self._layout_hint = None

    def _queue_layout_check(self, raw_contacts):
        self.wp.launch(self._shared_layout_check, max(raw_contacts, self.N),
            inputs=[self._layout_contacts, self.N, self.dflags,
                    self.sbi, self.sbj, self._layout_a, self._layout_b,
                    self.invM, self._cached_mass], device=self.dev)

    def _build_pairs(self, contacts):
        hint = self._layout_hint
        self._layout_hint = None
        if hint is None or hint[0] != contacts:
            return super()._build_pairs(contacts)
        self._layout_reused = False
        self._layout_mass_changed = hint[2]
        if contacts == self._layout_contacts and not hint[1]:
            self._layout_reused = True
            self.pair_layout_hits += 1
            return self._layout_pairs
        self._layout_mass_changed = True
        pairs = super(PairLayoutCacheContactEngine, self)._build_pairs(contacts)
        self.wp.copy(self._layout_a, self.sbi, count=contacts)
        self.wp.copy(self._layout_b, self.sbj, count=contacts)
        self._layout_contacts = contacts
        self._layout_pairs = pairs
        self.pair_layout_misses += 1
        return pairs

    def _device_manifold(s, C):
        """The grouping, the E-optimal selection and the warm start on the GPU: feature-key radix sort,
        pair-key radix sort, run-length segment flags, one thread per body pair for the selection, a scan
        and a compaction. Returns the number of kept contacts; the host readback contains kept count, oversize count,
        ordered endpoint mismatch and mass mismatch. Same selection as the host stage."""
        wp = s.wp; KD = s.KD; KC = s.KC; mx = s.manifold_max_points
        with s._stage("sort"):
            wp.launch(KD["fkey"], C, inputs=[C, s.cfa, s.cfb, s.N * s.P + 1, s.skey, s.sval], device=s.dev)
            wp.utils.radix_sort_pairs(s.skey, s.sval, C)          # stable: the order is a pure function of the keys
            wp.copy(s.order, s.sval, count=C)
            wp.launch(KC["gather"], C, inputs=[s.order, C, s.cbi, s.cbj, s.cpA, s.cpB, s.cn, s.cpen,
                                               s.tbi, s.tbj, s.tpA, s.tpB, s.tn, s.tpen], device=s.dev)
            wp.launch(KD["pkey_pair"], C, inputs=[C, s.tbi, s.tbj, s.N + 1, s.pkey_s, s.pval_s], device=s.dev)
            wp.utils.radix_sort_pairs(s.pkey_s, s.pval_s, C)
            wp.launch(KD["runflag"], C, inputs=[C, s.pkey_s, s.segflag], device=s.dev)
        with s._stage("select"):
            wn = s._ws_n if s.warm_start else 0
            wp.launch(KD["warm"], C, inputs=[C, s.skey, wn, s.ws_keys, s.ws_jn, s.ws_jt1, s.ws_jt2,
                                             s.jn_in, s.jt1_in, s.jt2_in], device=s.dev)
            s.keep.zero_(); s.dflags.zero_()
            wp.utils.array_scan(s.segflag[:C], s.segscan[:C], True)
            wp.launch(KD["runstart"], C, inputs=[C, s.segflag, s.segscan, s.runstart], device=s.dev)
            wp.launch(s._packed_select, C, inputs=[C, mx, s.runstart, s.segscan, s._planar_coords, s.pval_s, s.segflag, s.tbi, s.tpA, s.tn, s.tpen,
                                               s.xc, s.jn_in, s.keep, s.jn_red, s.dflags], device=s.dev)
            wp.utils.array_scan(s.keep[:C], s.segscan[:C], True)
            wp.launch(KD["compact"], C, inputs=[C, s.keep, s.segscan, s.skey, s.tbi, s.tbj, s.tpA, s.tpB,
                                                s.tn, s.tpen, s.jn_red, s.jt1_in, s.jt2_in, s.ckeys,
                                                s.sbi, s.sbj, s.spA, s.spB, s.sn, s.spen,
                                                s.jn, s.jt1, s.jt2, s.dflags], device=s.dev)
            s._queue_layout_check(C)
            fl = s.dflags.numpy(); Ck = int(fl[0])
            s._layout_hint = (Ck, bool(fl[2]), bool(fl[3]))
            if int(fl[1]) > 0:
                raise RuntimeError(f"contact_engine_gpu_colored._device_manifold: {int(fl[1])} body pairs carry "
                                   f"more than {_MAXSEG} contacts or more than the kept-point capacity; the "
                                   f"device selection kernel is fixed size -- use manifold_on_host=True")
            s.jp.zero_()
            if not s.trim_unused_priorities:
                wp.launch(KD["kept_runflag"], Ck, inputs=[Ck, s.sbi, s.sbj, s.N + 1, s.segflag], device=s.dev)
                wp.utils.array_scan(s.segflag[:Ck], s.segscan[:Ck], True)
                wp.launch(KD["runstart"], Ck, inputs=[Ck, s.segflag, s.segscan, s.runstart], device=s.dev)
                wp.launch(KD["prio"], Ck, inputs=[Ck, s.segscan, s.runstart, s.pkey], device=s.dev)
        return Ck

