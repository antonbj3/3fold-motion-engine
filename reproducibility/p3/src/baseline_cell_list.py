#!/usr/bin/env python3
"""Exact uniform-cell-list AABB range counter (a competent exact baseline).

Algorithm:
  * cell pitch = the producer's mass-grid pitch dx, res = 64 over the same origin/domain;
  * BUILD: one integer cell index per particle, atomic histogram over res^3 cells, exclusive
    prefix scan, atomic scatter into a compact particle order (a standard grid broad phase);
  * QUERY: per box, walk the clamped cell-index range overlapping [lo,hi] and test the exact D1
    predicate (particle centre in the closed box) for every particle in those cells only.

Output semantics are identical to the producer-grid gated path and to the strong direct kernel: an
exact strict D1 count per box; ties and order do not affect an integer count. This is the same
"grid broad phase + exact narrow phase" pattern that the engine's own fixed-radius hash-grid broad
phase uses, adapted to an AABB range. It imports neither the producer module nor the owned contract.

Determinism: integer counts only; the query kernel uses no atomics for the result (one thread per
box), so counts are byte-identical across processes."""
import os

import numpy as np
import warp as wp

wp.config.module_options = {"fuse_fp": False, "fast_math": False}


@wp.func
def _cell_index(x: wp.vec3, res: wp.int32, origin: wp.vec3d, inv_dx: wp.float64) -> wp.int32:
    # NOTE (Warp 1.13.0 sm_120 codegen): `wp.min(int, int)` / `wp.max(int, int)` overloads on
    # int32 operands silently produce 0 on this device. The clamp is therefore expressed as
    # `v + min(0, max(0, v))`, which is an exact no-op for v in [0, res) (all frozen particles are
    # inside the domain; the guard only protects out-of-domain inputs).
    xg = (wp.vec3d(wp.float64(x[0]), wp.float64(x[1]), wp.float64(x[2])) - origin) * inv_dx
    cx = wp.int32(wp.floor(xg[0]))
    cy = wp.int32(wp.floor(xg[1]))
    cz = wp.int32(wp.floor(xg[2]))
    cx = cx + wp.min(wp.int32(0), wp.max(wp.int32(0), cx))
    cy = cy + wp.min(wp.int32(0), wp.max(wp.int32(0), cy))
    cz = cz + wp.min(wp.int32(0), wp.max(wp.int32(0), cz))
    return (cx * res + cy) * res + cz


@wp.func
def _finite3(x: wp.vec3) -> wp.bool:
    return wp.isfinite(x[0]) and wp.isfinite(x[1]) and wp.isfinite(x[2])


@wp.kernel
def _k_hist(X: wp.array(dtype=wp.vec3), n: wp.int32, res: wp.int32, origin: wp.vec3d,
            inv_dx: wp.float64, counts: wp.array(dtype=wp.int32)):
    p = wp.tid()
    if p >= n:
        return
    x = X[p]
    if not _finite3(x):
        return
    c = _cell_index(x, res, origin, inv_dx)
    wp.atomic_add(counts, c, wp.int32(1))


@wp.kernel
def _k_scatter(X: wp.array(dtype=wp.vec3), n: wp.int32, res: wp.int32, origin: wp.vec3d,
               inv_dx: wp.float64, cursor: wp.array(dtype=wp.int32), order: wp.array(dtype=wp.int32)):
    p = wp.tid()
    if p >= n:
        return
    x = X[p]
    if not _finite3(x):
        return
    c = _cell_index(x, res, origin, inv_dx)
    pos = wp.atomic_add(cursor, c, wp.int32(1))
    order[pos] = wp.int32(p)


@wp.func
def _clampc(v: wp.int32, hi: wp.int32) -> wp.int32:
    if v < wp.int32(0):
        return wp.int32(0)
    if v > hi:
        return hi
    return v


@wp.kernel
def _k_range_count(X: wp.array(dtype=wp.vec3), starts: wp.array(dtype=wp.int32),
                   counts: wp.array(dtype=wp.int32), order: wp.array(dtype=wp.int32),
                   res: wp.int32, origin: wp.vec3d, inv_dx: wp.float64,
                   box_lo: wp.array(dtype=wp.vec3d), box_hi: wp.array(dtype=wp.vec3d),
                   n_chunks: wp.int32, out: wp.array(dtype=wp.int32),
                   work: wp.array(dtype=wp.int64)):
    """One thread per (box, x-slab chunk): parallel over the ix range, atomic add into the box."""
    t = wp.tid()
    b = t // n_chunks
    ch = t % n_chunks
    lo = box_lo[b]
    hi = box_hi[b]
    eps = wp.float64(1e-9)
    i0x = _clampc(wp.int32(wp.floor((lo[0] - origin[0]) * inv_dx - eps)), res - wp.int32(1))
    i0y = _clampc(wp.int32(wp.floor((lo[1] - origin[1]) * inv_dx - eps)), res - wp.int32(1))
    i0z = _clampc(wp.int32(wp.floor((lo[2] - origin[2]) * inv_dx - eps)), res - wp.int32(1))
    i1x = _clampc(wp.int32(wp.floor((hi[0] - origin[0]) * inv_dx + eps)), res - wp.int32(1))
    i1y = _clampc(wp.int32(wp.floor((hi[1] - origin[1]) * inv_dx + eps)), res - wp.int32(1))
    i1z = _clampc(wp.int32(wp.floor((hi[2] - origin[2]) * inv_dx + eps)), res - wp.int32(1))
    nx = i1x - i0x + 1
    start = i0x + (ch * nx) // n_chunks
    end = i0x + ((ch + 1) * nx) // n_chunks
    s = wp.int32(0)
    cand = wp.int64(0)
    for ix in range(start, end):
        for iy in range(i0y, i1y + 1):
            for iz in range(i0z, i1z + 1):
                c = (ix * res + iy) * res + iz
                a = starts[c]
                m = counts[c]
                for u in range(a, a + m):
                    q = X[order[u]]
                    x0 = wp.float64(q[0])
                    x1 = wp.float64(q[1])
                    x2 = wp.float64(q[2])
                    cand += wp.int64(1)
                    if x0 >= lo[0] and x0 <= hi[0] and x1 >= lo[1] and x1 <= hi[1] \
                            and x2 >= lo[2] and x2 <= hi[2]:
                        s += 1
    if s > 0:
        wp.atomic_add(out, b, s)
    wp.atomic_add(work, 0, cand)


class CellListBaseline:
    """Exact uniform cell-list AABB range counter. Build once per frame; query Q batches."""

    def __init__(self, res, origin, dx, device="cuda:0"):
        self.res = int(res)
        self.origin = np.asarray(origin, dtype=np.float64).reshape(3)
        self.dx = float(dx)
        self.inv_dx = 1.0 / float(dx)
        self.device = device
        self.ncells = self.res ** 3
        self.counts = None
        self.starts = None
        self.order = None
        self.X = None
        self.n = 0

    def build(self, P, n=None):
        """Build the cell list from a caller numpy/warp position array (charged as build cost)."""
        device = self.device
        if isinstance(P, wp.array):
            Pw = P
            n = int(P.size) if n is None else int(n)
        else:
            Pw = wp.array(np.ascontiguousarray(P, dtype=np.float32), dtype=wp.vec3, device=device)
            n = int(len(np.asarray(P).reshape(-1, 3))) if n is None else int(n)
        self.n = n
        counts = wp.zeros(self.ncells, dtype=wp.int32, device=device)
        order = wp.zeros(max(n, 1), dtype=wp.int32, device=device)
        with wp.ScopedDevice(device):
            wp.launch(_k_hist, dim=n, inputs=[Pw, n, self.res, wp.vec3d(*self.origin),
                                              self.inv_dx, counts], device=device)
            starts = wp.zeros(self.ncells, dtype=wp.int32, device=device)
            wp.utils.array_scan(counts, starts, False)
            cursor = wp.clone(starts)
            wp.launch(_k_scatter, dim=n, inputs=[Pw, n, self.res, wp.vec3d(*self.origin),
                                                 self.inv_dx, cursor, order], device=device)
        self.X = Pw
        self.counts = counts
        self.starts = starts
        self.order = order
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
            n_chunks = max(1, int(os.environ.get("CELLLIST_CHUNKS", "8")))
            with wp.ScopedDevice(dev):
                wp.launch(_k_range_count, dim=nb * n_chunks,
                          inputs=[self.X, self.starts, self.counts, self.order, self.res,
                                  wp.vec3d(*self.origin), self.inv_dx, a, b, n_chunks, out, work],
                          device=dev)
        return out.numpy().astype(np.int32), int(work.numpy()[0])
