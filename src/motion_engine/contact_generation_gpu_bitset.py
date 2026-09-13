"""Bounded point-local word bitsets emit canonical contacts and fused keys.

This separate prototype does not replace the frozen contact engine. At most 32
non-ground neighbours per point are supported; overflow fails before emission.
"""
import numpy as np
from motion_engine.contact_engine_gpu import _R


def build_kernels(wp):
    ids32 = wp.types.vector(length=32, dtype=wp.int32)

    @wp.kernel
    def count(p: wp.array(dtype=wp.vec3), owner: wp.array(dtype=int), grid: wp.uint64,
              counts: wp.array(dtype=int), maximum: wp.array(dtype=int)):
        i = wp.tid()
        xi = p[i]
        neighbours = int(0)
        query = wp.hash_grid_query(grid, xi, 2.0*_R)
        j = int(0)
        while wp.hash_grid_query_next(query, j):
            if j > i and owner[j] != owner[i]:
                distance = wp.length(xi-p[j])
                if distance < 2.0*_R and distance > 1e-9:
                    neighbours += 1
        ground = int(0)
        if xi[2]-_R < 0.0:
            ground = 1
        counts[i] = neighbours+ground
        wp.atomic_max(maximum, 0, neighbours)

    @wp.kernel
    def emit(p: wp.array(dtype=wp.vec3), owner: wp.array(dtype=int), grid: wp.uint64,
             offsets: wp.array(dtype=int), stride: wp.int64, body_stride: wp.int64,
             bi: wp.array(dtype=int), bj: wp.array(dtype=int),
             pa: wp.array(dtype=wp.vec3), pb: wp.array(dtype=wp.vec3),
             normal: wp.array(dtype=wp.vec3), pen: wp.array(dtype=float),
             fa: wp.array(dtype=int), fb: wp.array(dtype=int),
             feature: wp.array(dtype=wp.int64), pair: wp.array(dtype=wp.int64)):
        i = wp.tid()
        xi = p[i]
        out = offsets[i]
        if xi[2]-_R < 0.0:
            bi[out] = owner[i]
            bj[out] = -1
            pa[out] = xi
            pb[out] = wp.vec3(0.0, 0.0, 0.0)
            normal[out] = wp.vec3(0.0, 0.0, 1.0)
            pen[out] = -(xi[2]-_R)
            fa[out] = i
            fb[out] = -1
            feature[out] = wp.int64(i)*stride
            pair[out] = wp.int64(owner[i])*body_stride-wp.int64(1)
            out += 1
        candidates = ids32(0)
        n = int(0)
        query = wp.hash_grid_query(grid, xi, 2.0*_R)
        j = int(0)
        while wp.hash_grid_query_next(query, j):
            if j > i and owner[j] != owner[i]:
                distance = wp.length(xi-p[j])
                if distance < 2.0*_R and distance > 1e-9:
                    candidates[n] = j
                    n += 1
        # Hash traversal may reorder candidates. Select ascending global-id words,
        # then ascending occupied bits, independent of their temporary slot order.
        previous_word = int(-1)
        for group in range(n):
            word = int(2147483647)
            for slot in range(n):
                w = candidates[slot] // 32
                if w > previous_word and w < word:
                    word = w
            if word < 2147483647:
                mask = wp.uint32(0)
                for slot in range(n):
                    candidate = candidates[slot]
                    if candidate // 32 == word:
                        mask = mask | (wp.uint32(1) << wp.uint32(candidate % 32))
                for bit in range(32):
                    if (mask & (wp.uint32(1) << wp.uint32(bit))) != wp.uint32(0):
                        q = word*32+bit
                        dvec = xi-p[q]
                        distance = wp.length(dvec)
                        bi[out] = owner[i]
                        bj[out] = owner[q]
                        pa[out] = xi
                        pb[out] = p[q]
                        normal[out] = dvec/distance
                        pen[out] = 2.0*_R-distance
                        fa[out] = i
                        fb[out] = q
                        feature[out] = wp.int64(i)*stride+wp.int64(q)+wp.int64(1)
                        pair[out] = wp.int64(owner[i])*body_stride+wp.int64(owner[q])
                        out += 1
                previous_word = word

    return count, emit


def generate(points, owners):
    import warp as wp
    p = np.asarray(points, dtype=np.float32)
    o = np.asarray(owners, dtype=np.int32)
    if p.ndim != 2 or p.shape[1] != 3 or o.shape != (len(p),):
        raise ValueError("Expected points[N,3] and owners[N]")
    if not len(p) or not np.isfinite(p).all() or np.any(o < 0):
        raise ValueError("Require finite nonempty points and nonnegative owners")
    count, emit = build_kernels(wp)
    allp = wp.array(p, dtype=wp.vec3, device="cuda:0")
    owner = wp.array(o, dtype=int, device="cuda:0")
    grid = wp.HashGrid(64, 64, 64, device="cuda:0")
    grid.build(allp, 2*_R)
    counts = wp.zeros(len(p), dtype=int, device="cuda:0")
    maximum = wp.zeros(1, dtype=int, device="cuda:0")
    wp.launch(count, len(p), inputs=[allp, owner, grid.id, counts, maximum], device="cuda:0")
    max_neighbours = int(maximum.numpy()[0])
    if max_neighbours > 32:
        raise ValueError(f"Neighbour capacity exceeded: {max_neighbours} > 32")
    c = int(counts.numpy().sum())
    offsets = wp.empty(len(p), dtype=int, device="cuda:0")
    wp.utils.array_scan(counts, offsets, inclusive=False)
    arrays = [wp.empty(c, dtype=t, device="cuda:0") for t in
              (int, int, wp.vec3, wp.vec3, wp.vec3, float, int, int, wp.int64, wp.int64)]
    wp.launch(emit, len(p), inputs=[allp, owner, grid.id, offsets,
              wp.int64(len(p)+1), wp.int64(int(o.max())+2), *arrays], device="cuda:0")
    return [a.numpy() for a in arrays], max_neighbours


if __name__ == "__main__":
    import runpy
    from pathlib import Path
    runpy.run_path(str(Path(__file__).resolve().parents[2]/"probes"/"innovation_contact_gpu_bitset_probe.py"), run_name="__main__")
