"""Independent reference constants, box-geometry generators and D1 predicates.

Imports neither the producer module nor the owned contract. Written from the simulator semantics:
  * P2G: quadratic B-spline, int64 fixed-point scale 1e11, nodes outside [0,res) skipped,
    float32 output = int64_acc / scale.
  * D1 STRICT: breach iff a particle centre lies in [lo, hi] inclusive (float32 -> float64).
  * Occupancy: node with mass > theta*rho*dx^3.

D1 is implemented twice across the package: this NumPy predicate and a separate Warp kernel with a
different chunk size, so the held-out decision does not rest on the path under test."""
import hashlib

import numpy as np

RES = 64
DOMAIN = 0.80
DX = DOMAIN / RES
ORIGIN = np.array([0.0, -0.40, -0.05], dtype=np.float64)
RHO = 1600.0
DT = 1.0e-5
MASS_SCALE = 100000000000
MASS1 = 3.90625e-4
CELL_MASS_FULL = RHO * DX ** 3
KERNEL_ID = "quadratic_bspline_p2g_int64_scale1e11_round_half_away"

KEEPOUT_MIN = np.array([0.45, -0.25, -0.03], dtype=np.float64)
KEEPOUT_MAX = np.array([0.55, 0.10, 0.05], dtype=np.float64)

DOMAIN_LO = ORIGIN.copy()
DOMAIN_HI = ORIGIN + DOMAIN
SPARSE_MIN = ORIGIN + 1.0 * DX
SPARSE_MAX = ORIGIN + (RES - 1.0) * DX


def axis_w(f, j):
    if j == 0:
        return 0.5 * (1.5 - f) ** 2
    if j == 1:
        return 0.75 - (f - 1.0) ** 2
    return 0.5 * (f - 0.5) ** 2


def _aw(fa, j):
    if j == 0:
        return (0.5 * (1.5 - fa) * (1.5 - fa)).astype(np.float32)
    if j == 1:
        return (0.75 - (fa - 1.0) * (fa - 1.0)).astype(np.float32)
    return (0.5 * (fa - 0.5) * (fa - 0.5)).astype(np.float32)


def p2g_int64(x_at, m, res=RES, origin=ORIGIN, dx=DX, scale=MASS_SCALE, return_int64=False):
    """Independent vectorised flat int64 accumulation, then float32(acc/scale)."""
    P = np.asarray(x_at, dtype=np.float32).reshape(-1, 3)
    mass = np.asarray(m, dtype=np.float32).reshape(-1)
    if mass.size == 1 and len(P) != 1:
        mass = np.full(len(P), mass[0], dtype=np.float32)
    o = np.asarray(origin, dtype=np.float32)
    inv_dx = np.float32(1.0 / np.float32(dx))
    acc = np.zeros(res * res * res, dtype=np.int64)
    if len(P) == 0:
        return acc if return_int64 else (acc.astype(np.float64) / scale).astype(np.float32)
    xg = (P - o) * inv_dx
    bf = np.floor(xg - np.float32(0.5)).astype(np.float32)
    b = bf.astype(np.int64)
    f = (xg - bf).astype(np.float32)
    aw = [[_aw(f[:, d], j) for j in range(3)] for d in range(3)]
    for i in range(3):
        ix = b[:, 0] + i
        ox = (ix >= 0) & (ix < res)
        for j in range(3):
            iy = b[:, 1] + j
            oy = (iy >= 0) & (iy < res)
            for k in range(3):
                iz = b[:, 2] + k
                oz = (iz >= 0) & (iz < res)
                ok = ox & oy & oz
                if not ok.any():
                    continue
                w = (aw[0][i] * aw[1][j] * aw[2][k]).astype(np.float32)
                prod = (w * mass).astype(np.float64) * scale
                contrib = np.floor(prod + 0.5).astype(np.int64)
                flat = (ix[ok] * res + iy[ok]) * res + iz[ok]
                np.add.at(acc, flat, contrib[ok])
    return acc if return_int64 else (acc.astype(np.float64) / scale).astype(np.float32)


def best_node_weight(x, res=RES, origin=ORIGIN, dx=DX):
    """Max tensor B-spline weight over IN-GRID nodes for a single float32 position."""
    x = np.asarray(x, dtype=np.float32).reshape(3)
    o = np.asarray(origin, dtype=np.float32)
    inv_dx = np.float32(1.0 / np.float32(dx))
    xg = (x - o) * inv_dx
    bf = np.floor(xg - np.float32(0.5)).astype(np.float32)
    b = bf.astype(np.int64)
    f = (xg - bf).astype(np.float32)
    best = np.float32(0.0)
    for i in range(3):
        ix = b[0] + i
        if ix < 0 or ix >= res:
            continue
        wi = np.float32(axis_w(np.float32(f[0]), i))
        for j in range(3):
            iy = b[1] + j
            if iy < 0 or iy >= res:
                continue
            wj = np.float32(axis_w(np.float32(f[1]), j))
            for k in range(3):
                iz = b[2] + k
                if iz < 0 or iz >= res:
                    continue
                wk = np.float32(axis_w(np.float32(f[2]), k))
                w = np.float32(np.float32(np.float32(wi * wj)) * wk)
                if w > best:
                    best = w
    return best


def d1_counts(P, lo, hi):
    """NumPy D1: particle centre in [lo, hi] inclusive (float32 promoted to float64)."""
    p = np.asarray(P, dtype=np.float32).astype(np.float64).reshape(-1, 3)
    a = np.asarray(lo, dtype=np.float64).reshape(3)
    c = np.asarray(hi, dtype=np.float64).reshape(3)
    if p.size == 0:
        return 0
    return int(np.all((p >= a) & (p <= c), axis=1).sum())


def d1_counts_all(P, lo, hi):
    p = np.asarray(P, dtype=np.float32).astype(np.float64).reshape(-1, 3)
    blo = np.asarray(lo, dtype=np.float64).reshape(-1, 3)
    bhi = np.asarray(hi, dtype=np.float64).reshape(-1, 3)
    out = np.zeros(len(blo), dtype=np.int64)
    if p.size == 0:
        return out
    for b in range(len(blo)):
        out[b] = int(np.all((p >= blo[b]) & (p <= bhi[b]), axis=1).sum())
    return out


def boxes_for(n, seed, region, min_frac=0.15, max_frac=0.30):
    rng = np.random.default_rng(seed)
    lo = np.asarray(region[0], dtype=np.float64)
    hi = np.asarray(region[1], dtype=np.float64)
    span = hi - lo
    if n == 1:
        size = span * min_frac
        blo = lo + 0.5 * (span - size)
        return blo.reshape(1, 3), (blo + size).reshape(1, 3)
    size = span * rng.uniform(min_frac, max_frac, size=(n, 3))
    blo = lo + (span - size) * rng.random((n, 3))
    return blo, blo + size


def boxes_slab(n, seed, region, thin=0.02, min_frac=0.30, max_frac=0.60):
    """New geometry family: thin boxes in z, wide in x/y (stresses node-index bounds)."""
    rng = np.random.default_rng(seed)
    lo = np.asarray(region[0], dtype=np.float64)
    hi = np.asarray(region[1], dtype=np.float64)
    span = hi - lo
    size = np.empty((n, 3), dtype=np.float64)
    size[:, 0] = span[0] * rng.uniform(min_frac, max_frac, size=n)
    size[:, 1] = span[1] * rng.uniform(min_frac, max_frac, size=n)
    size[:, 2] = np.full(n, thin)
    size[:, 2] = np.minimum(size[:, 2], span[2])
    blo = lo + (span - size) * rng.random((n, 3))
    return blo, blo + size


def boxes_corner(n, seed, region, min_frac=0.10, max_frac=0.25):
    """New geometry family: boxes clustered at the lower domain corner."""
    rng = np.random.default_rng(seed)
    lo = np.asarray(region[0], dtype=np.float64)
    hi = np.asarray(region[1], dtype=np.float64)
    span = hi - lo
    size = span * rng.uniform(min_frac, max_frac, size=(n, 3))
    blo = lo + (span - size) * (rng.random((n, 3)) ** 3)
    return blo, blo + size


def boxes_empty(n, seed):
    """New geometry family: 256 boxes in a region with no particles (gate must skip all)."""
    return boxes_for(n, seed, (ORIGIN + np.array([0.60, 0.30, 0.55]),
                               ORIGIN + np.array([0.72, 0.38, 0.65])), 0.15, 0.30)


def sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
