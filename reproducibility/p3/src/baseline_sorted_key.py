#!/usr/bin/env python3
"""Exact AABB range counter WITHOUT an explicit prefix scan (radix-sorted cell key).

A second serious exact baseline with the same output semantics (exact strict D1 count per box),
built from the particles; it answers the range query by binary-searching a radix-sorted
(cell_key, particle_index) array. No global prefix scan is needed: the sorted key gives
variable-size equal-key runs directly. `wp.utils.radix_sort_pairs` is a first-party Warp two-pass
radix sort.

  BUILD:  one int64 key per particle = clamped cell index (pitch dx, res 64, same origin),
          then `radix_sort_pairs(keys, idx)`.
  QUERY:  per box, `lower_bound` for the first key >= lo_cell and the first key > hi_cell on the
          sorted key array (binary search in a per-thread loop), then the exact D1 predicate over
          that key range, with no dependence on the cell loop extent.

The comparator uses int64 keys, so the count is integer and order-independent (byte-identical). Same
role as `baseline_cell_list.py`; the key-sort range query is the standard sorted-grid variant of the
engine's fixed-radius broad phase."""
import numpy as np
import warp as wp

wp.config.module_options = {"fuse_fp": False, "fast_math": False}


@wp.kernel
def _k_key(X: wp.array(dtype=wp.vec3), n: wp.int32, res: wp.int32,
           ox: wp.float64, oy: wp.float64, oz: wp.float64, dx: wp.float64,
           valid: wp.array(dtype=wp.int32), keys: wp.array(dtype=wp.int64)):
    """Cell key per particle. `origin`/`inv_dx` are passed as three float64 scalars and divided by
    `dx`: passing a vec3d/float64 pair together miscompiles the expression to 0 on this device
    (verified: same arithmetic with scalar params or a division gives the correct keys)."""
    p = wp.tid()
    if p >= n:
        return
    x = X[p]
    ok = wp.int32(1)
    if not wp.isfinite(x[0]) or not wp.isfinite(x[1]) or not wp.isfinite(x[2]):
        ok = wp.int32(0)
    if ok == 1:
        xg0 = (wp.float64(x[0]) - ox) / dx
        xg1 = (wp.float64(x[1]) - oy) / dx
        xg2 = (wp.float64(x[2]) - oz) / dx
        c0 = wp.int32(wp.floor(xg0))
        c1 = wp.int32(wp.floor(xg1))
        c2 = wp.int32(wp.floor(xg2))
        if c0 < wp.int32(0):
            c0 = wp.int32(0)
        if c0 > res - wp.int32(1):
            c0 = res - wp.int32(1)
        if c1 < wp.int32(0):
            c1 = wp.int32(0)
        if c1 > res - wp.int32(1):
            c1 = res - wp.int32(1)
        if c2 < wp.int32(0):
            c2 = wp.int32(0)
        if c2 > res - wp.int32(1):
            c2 = res - wp.int32(1)
        keys[p] = (wp.int64(c0) * wp.int64(res) + wp.int64(c1)) * wp.int64(res) + wp.int64(c2)
    else:
        keys[p] = wp.int64(res) * wp.int64(res) * wp.int64(res)
    valid[p] = ok


@wp.func
def _lower_bound(keys: wp.array(dtype=wp.int64), nk: wp.int32, target: wp.int64) -> wp.int32:
    lo = wp.int32(0)
    hi = nk
    while lo < hi:
        mid = (lo + hi) / 2
        if keys[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


@wp.kernel
def _k_sorted_range_count(X: wp.array(dtype=wp.vec3), keys: wp.array(dtype=wp.int64),
                          idx: wp.array(dtype=wp.int32), valid: wp.array(dtype=wp.int32),
                          nk: wp.int32, res: wp.int32,
                          ox: wp.float64, oy: wp.float64, oz: wp.float64, inv_dx: wp.float64,
                          box_lo: wp.array(dtype=wp.vec3d), box_hi: wp.array(dtype=wp.vec3d),
                          out: wp.array(dtype=wp.int32), work: wp.array(dtype=wp.int64)):
    b = wp.tid()
    lo = box_lo[b]
    hi = box_hi[b]
    eps = wp.float64(1e-9)
    i0x = wp.int32(wp.floor((lo[0] - ox) * inv_dx - eps))
    i0y = wp.int32(wp.floor((lo[1] - oy) * inv_dx - eps))
    i0z = wp.int32(wp.floor((lo[2] - oz) * inv_dx - eps))
    i1x = wp.int32(wp.floor((hi[0] - ox) * inv_dx + eps))
    i1y = wp.int32(wp.floor((hi[1] - oy) * inv_dx + eps))
    i1z = wp.int32(wp.floor((hi[2] - oz) * inv_dx + eps))
    if i0x < wp.int32(0):
        i0x = wp.int32(0)
    if i0x > res - wp.int32(1):
        i0x = res - wp.int32(1)
    if i0y < wp.int32(0):
        i0y = wp.int32(0)
    if i0y > res - wp.int32(1):
        i0y = res - wp.int32(1)
    if i0z < wp.int32(0):
        i0z = wp.int32(0)
    if i0z > res - wp.int32(1):
        i0z = res - wp.int32(1)
    if i1x < wp.int32(0):
        i1x = wp.int32(0)
    if i1x > res - wp.int32(1):
        i1x = res - wp.int32(1)
    if i1y < wp.int32(0):
        i1y = wp.int32(0)
    if i1y > res - wp.int32(1):
        i1y = res - wp.int32(1)
    if i1z < wp.int32(0):
        i1z = wp.int32(0)
    if i1z > res - wp.int32(1):
        i1z = res - wp.int32(1)

    s = wp.int32(0)
    cand = wp.int64(0)
    # the key is (ix*res + iy)*res + iz; a contiguous key range is exact ONLY if all three indices
    # vary (each (ix,iy) row must be looked up for its own z-interval separately).
    for ix in range(i0x, i1x + 1):
        for iy in range(i0y, i1y + 1):
            klo = wp.int64((wp.int64(ix) * wp.int64(res) + wp.int64(iy)) * wp.int64(res)
                           + wp.int64(i0z))
            khi = wp.int64((wp.int64(ix) * wp.int64(res) + wp.int64(iy)) * wp.int64(res)
                           + wp.int64(i1z))
            a = _lower_bound(keys, nk, klo)
            e = _lower_bound(keys, nk, khi + wp.int64(1))
            for t in range(a, e):
                p = idx[t]
                if valid[p] == 0:
                    continue
                q = X[p]
                x0 = wp.float64(q[0])
                x1 = wp.float64(q[1])
                x2 = wp.float64(q[2])
                cand += wp.int64(1)
                if x0 >= lo[0] and x0 <= hi[0] and x1 >= lo[1] and x1 <= hi[1] \
                        and x2 >= lo[2] and x2 <= hi[2]:
                    s += 1
    out[b] = s
    wp.atomic_add(work, 0, cand)


class SortedKeyBaseline:
    """Exact range counter over a radix-sorted cell key (no explicit prefix scan)."""

    def __init__(self, res, origin, dx, device="cuda:0"):
        self.res = int(res)
        self.origin = np.asarray(origin, dtype=np.float64).reshape(3)
        self.dx = float(dx)
        self.device = device
        self.keys = None
        self.idx = None
        self.valid = None
        self.X = None
        self.n = 0

    def build(self, P, n=None):
        dev = self.device
        if isinstance(P, wp.array):
            Pw = P
            n = int(P.size) if n is None else int(n)
        else:
            Pw = wp.array(np.ascontiguousarray(P, dtype=np.float32), dtype=wp.vec3, device=dev)
            n = int(len(np.asarray(P).reshape(-1, 3))) if n is None else int(n)
        self.n = n
        # radix_sort_pairs needs backing storage for 2*count elements in this Warp version, and the
        # value array must hold the identity permutation before sorting.
        keys = wp.zeros(2 * n, dtype=wp.int64, device=dev)
        idx = wp.array(np.arange(2 * n, dtype=np.int32), dtype=wp.int32, device=dev)
        valid = wp.zeros(n, dtype=wp.int32, device=dev)
        with wp.ScopedDevice(dev):
            wp.launch(_k_key, dim=n, inputs=[Pw, n, self.res, float(self.origin[0]),
                                             float(self.origin[1]), float(self.origin[2]),
                                             self.dx, valid, keys], device=dev)
            wp.utils.radix_sort_pairs(keys, idx, n)
        self.X = Pw
        self.keys = keys
        self.idx = idx
        self.valid = valid
        return self

    def query(self, box_lo, box_hi):
        dev = self.device
        blo = np.asarray(box_lo, dtype=np.float64).reshape(-1, 3)
        bhi = np.asarray(box_hi, dtype=np.float64).reshape(-1, 3)
        nb = len(blo)
        out = wp.zeros(nb, dtype=wp.int32, device=dev)
        work = wp.zeros(1, dtype=wp.int64, device=dev)
        if nb:
            a = wp.array(blo, dtype=wp.vec3d, device=dev)
            b = wp.array(bhi, dtype=wp.vec3d, device=dev)
            with wp.ScopedDevice(dev):
                wp.launch(_k_sorted_range_count, dim=nb,
                          inputs=[self.X, self.keys, self.idx, self.valid, self.n, self.res,
                                  float(self.origin[0]), float(self.origin[1]),
                                  float(self.origin[2]), 1.0 / self.dx, a, b, out, work],
                          device=dev)
        return out.numpy().astype(np.int32), int(work.numpy()[0])
