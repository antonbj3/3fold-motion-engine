"""Independent NumPy reference for the simulator deposition and the strict-D1 predicate.

Written from the simulator semantics:
  * P2G: quadratic B-spline, int64 fixed-point with scale 1e11, round-half (positive), nodes
    outside [0,res) skipped, float32 output = int64_acc / scale.
  * D1 STRICT: breach iff a particle centre lies in [lo, hi] inclusive (float32 promoted to f64).
  * D3 CELLOCC: grid node with mass > theta*rho*dx^3.

Imports neither the producer module nor the owned contract. The producer's returned count must
equal D1 on valid input."""
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
    """Independent flat int64 accumulation, then float32(acc/scale)."""
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


def deposits_at_least_one(x, m, res=RES, origin=ORIGIN, dx=DX, scale=MASS_SCALE):
    """Independent predicate: does float32 mass m at x deposit >= 1 count on some in-grid node?"""
    best = best_node_weight(x, res, origin, dx)
    if best <= np.float32(0.0):
        return False
    prod = np.float64(np.float32(best * np.float32(m))) * np.float64(scale)
    return bool(prod >= 0.5)


def d1_counts_all(P, box_lo, box_hi):
    p = np.asarray(P, dtype=np.float32).astype(np.float64).reshape(-1, 3)
    blo = np.asarray(box_lo, dtype=np.float64).reshape(-1, 3)
    bhi = np.asarray(box_hi, dtype=np.float64).reshape(-1, 3)
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


def node_bounds(box_lo, box_hi, pad, res=RES, origin=ORIGIN, dx=DX):
    lo = np.asarray(box_lo, dtype=np.float64).reshape(-1, 3) - pad
    hi = np.asarray(box_hi, dtype=np.float64).reshape(-1, 3) + pad
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    r0 = np.ceil((lo - origin) / dx - 1e-12).astype(np.int64)
    r1 = np.floor((hi - origin) / dx + 1e-12).astype(np.int64)
    valid = (r1 >= 0).all(axis=1) & (r0 <= res - 1).all(axis=1) & (r1 >= r0).all(axis=1)
    ni = np.where(valid, (np.clip(r1, 0, res - 1) - np.clip(r0, 0, res - 1) + 1).prod(axis=1), 0)
    return np.clip(r0, 0, res - 1), np.clip(r1, 0, res - 1), valid, ni


def sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
