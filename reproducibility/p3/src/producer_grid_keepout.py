#!/usr/bin/env python3
"""Producer-bound GPU keep-out prefilter (reference implementation, not product-integrated).

This module implements a two-stage keep-out decision for a declared axis-aligned box: a conservative
occupancy prefilter built from the simulator's own deposited mass grid, followed by an exact
particle-centre count for every box the prefilter cannot safely reject. It is the path under test in
the matched-baseline study; it is not the truth reference.

Producer coupling. A `Snapshot` can only carry a grid that is P2G(x_at, m_at) by construction:
either `snapshot_from_particles` computes the int64 fixed-point P2G from its own owned copies of
`x/m/ids`, or `snapshot_from_solver_grid` verifies a supplied grid once, bit-for-bit, against a
fresh int64 P2G of the supplied particles. A supplied grid that does not match is rejected
(`GridNotBoundError`); an unverified snapshot is never usable (`reason='grid_unverified'`). A hash
of the grid alone is not accepted as evidence: the proof is the recomputation from the particles.

The gate skips work. `keepout_gated` sends only the flagged boxes to the exact particle query; safely
empty boxes return 0 with no particle sweep. `info['n_box_particle_tests']` counts the tests actually
executed. A locked policy falls back to the strong direct kernel when the gate work would dominate
or there are too few boxes.

Contract. `keepout_gated(state, cur, box_lo, box_hi, ...)` refuses the resident grid unless the
lattice scalars are finite and the kernel id is the known quadratic-B-spline kernel; the grid is
verified; `grid.size == res**3`; `cur.n == state.n` and the arrays are self-consistent; every
particle id is element-wise identical; `cur.version == state.version`; `0 <= cur.frame - state.frame
<= max_lag`; every deposited particle is finite, positive, in-domain and deposits at least one count
at the real int64 scale; and the position lag `D = max|x - x_at|` is finite and `<= max_disp`. On any
failure every box is counted exactly (`used=False`).

Gate soundness (within the tested scope). Every particle deposits at least one count, so an occupied
node exists within 1.5 dx of its deposited position; the pad `radie + 2 dx + D` covers the B-spline
support plus the full position lag. Hence a particle centre in the box implies an occupied node
centre in box+pad.

Determinism: int32 occupancy/count atomics, int64 fixed-point P2G; `fuse_fp=False`,
`fast_math=False`. Two processes must produce identical decision bytes."""
import hashlib
from dataclasses import dataclass, field

import numpy as np
import warp as wp

wp.config.module_options = {"fuse_fp": False, "fast_math": False}

CHUNK = 512          # exact-query chunk (particles per thread)
RED_CHUNK = 1024     # displacement-reduction chunk (particles per thread)

KERNEL_ID = "quadratic_bspline_p2g_int64_scale1e11_round_half_away"
MASS_SCALE = 100000000000

# ---- LOCKED POLICY (set by policy_preregistration.py before any hold-out) ---------------------------
# Fall back to the strong direct kernel when there are too few boxes for the gate to pay for
# itself, or when the gate node work would exceed a fixed fraction of the direct particle work.
# The rule uses only box geometry, n and the lattice -- never a measured result.
POLICY = {
    "min_nb_gated": 128,
    "max_gate_work_frac": 0.20,
    "amortize_prepare": True,
    "small_nb_route": "exact_box_counts",   # nb < min_nb_gated -> the strong direct kernel
    "production_route": "amortized",        # one prepare per frame, then query batch
}


class GridNotBoundError(ValueError):
    """Raised when a supplied grid is not bit-equal to P2G(x_at, m_at)."""


# --------------------------------------------------------------------------------------
# Warp kernels
# --------------------------------------------------------------------------------------
@wp.func
def _axis_w(f: wp.float32, j: wp.int32) -> wp.float32:
    if j == 0:
        return 0.5 * (1.5 - f) * (1.5 - f)
    elif j == 1:
        return 0.75 - (f - 1.0) * (f - 1.0)
    else:
        return 0.5 * (f - 0.5) * (f - 0.5)


@wp.kernel
def _occ_kernel(mass: wp.array(dtype=wp.float32), thr: wp.float32,
                occ: wp.array(dtype=wp.int32)):
    i = wp.tid()
    occ[i] = wp.int32(1) if mass[i] > thr else wp.int32(0)


@wp.kernel
def _grid_box_count_kernel(occ: wp.array(dtype=wp.int32), res: wp.int32,
                           i0: wp.array(dtype=wp.vec3i), i1: wp.array(dtype=wp.vec3i),
                           valid: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
    b = wp.tid()
    if valid[b] == 0:
        out[b] = 0
    else:
        a = i0[b]
        c = i1[b]
        s = wp.int32(0)
        for ix in range(a[0], c[0] + 1):
            for iy in range(a[1], c[1] + 1):
                for iz in range(a[2], c[2] + 1):
                    s += occ[(ix * res + iy) * res + iz]
        out[b] = s


@wp.kernel
def _particle_box_count_kernel(P: wp.array(dtype=wp.vec3), n: wp.int32, n_chunks: wp.int32,
                               lo: wp.array(dtype=wp.vec3d), hi: wp.array(dtype=wp.vec3d),
                               out: wp.array(dtype=wp.int32)):
    t = wp.tid()
    b = t // n_chunks
    ch = t % n_chunks
    a = lo[b]
    c = hi[b]
    s = wp.int32(0)
    start = ch * CHUNK
    end = wp.min(start + CHUNK, n)
    for p in range(start, end):
        x = P[p]
        x0 = wp.float64(x[0])
        x1 = wp.float64(x[1])
        x2 = wp.float64(x[2])
        if x0 >= a[0] and x0 <= c[0] and x1 >= a[1] and x1 <= c[1] \
                and x2 >= a[2] and x2 <= c[2]:
            s += 1
    if s > 0:
        wp.atomic_add(out, b, s)


@wp.kernel
def _chunk_disp_nonfinite_kernel(XN: wp.array(dtype=wp.vec3), XO: wp.array(dtype=wp.vec3),
                                 M: wp.array(dtype=wp.float32), n: wp.int32, chunk: wp.int32,
                                 out_disp: wp.array(dtype=wp.float32),
                                 out_bad: wp.array(dtype=wp.int32)):
    """Two-level reduction: per-chunk max displacement and non-finite count, no global atomic."""
    i = wp.tid()
    start = i * chunk
    end = wp.min(start + chunk, n)
    dmax = wp.float32(0.0)
    bad = wp.int32(0)
    for p in range(start, end):
        d = XN[p] - XO[p]
        s = wp.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        if s > dmax:
            dmax = s
        x = XN[p]
        if not wp.isfinite(x[0]) or not wp.isfinite(x[1]) or not wp.isfinite(x[2]) \
                or not wp.isfinite(M[p]):
            bad += 1
    out_disp[i] = dmax
    out_bad[i] = bad


@wp.kernel
def _ids_mismatch_kernel(A: wp.array(dtype=wp.int64), B: wp.array(dtype=wp.int64),
                         n: wp.int32, out: wp.array(dtype=wp.int32)):
    p = wp.tid()
    if p < n and A[p] != B[p]:
        wp.atomic_add(out, 0, wp.int32(1))


@wp.kernel
def _deposition_valid_kernel(X: wp.array(dtype=wp.vec3), M: wp.array(dtype=wp.float32),
                             n: wp.int32, res: wp.int32, origin: wp.vec3, inv_dx: wp.float32,
                             scale: wp.float64, fail: wp.array(dtype=wp.int32)):
    """fail[p]=1 unless particle p deposits >= 1 count on an in-grid node at the real scale."""
    p = wp.tid()
    if p >= n:
        return
    x = X[p]
    m = M[p]
    bad = wp.int32(0)
    for c in range(3):
        if not wp.isfinite(x[c]):
            bad = wp.int32(1)
    if not wp.isfinite(m) or m <= 0.0:
        bad = wp.int32(1)
    if bad == 0:
        xg = (x - origin) * inv_dx
        for c in range(3):
            if not wp.isfinite(xg[c]) or xg[c] < 0.0 or xg[c] > wp.float32(res):
                bad = wp.int32(1)
    if bad == 0:
        b = wp.vec3i(wp.int32(wp.floor(xg[0] - 0.5)), wp.int32(wp.floor(xg[1] - 0.5)),
                     wp.int32(wp.floor(xg[2] - 0.5)))
        f = wp.vec3(xg[0] - wp.float32(b[0]), xg[1] - wp.float32(b[1]),
                    xg[2] - wp.float32(b[2]))
        best = wp.float32(0.0)
        for i in range(3):
            ix = b[0] + i
            if ix < 0 or ix >= res:
                continue
            wi = _axis_w(f[0], i)
            for j in range(3):
                iy = b[1] + j
                if iy < 0 or iy >= res:
                    continue
                wj = _axis_w(f[1], j)
                for k in range(3):
                    iz = b[2] + k
                    if iz < 0 or iz >= res:
                        continue
                    wk = _axis_w(f[2], k)
                    w = wi * wj * wk
                    if w > best:
                        best = w
        if best <= 0.0:
            bad = wp.int32(1)
        else:
            prod = wp.float64(best * m) * scale
            if prod < 0.5:
                bad = wp.int32(1)
    fail[p] = bad


@wp.kernel
def _p2g_int64_kernel(X: wp.array(dtype=wp.vec3), M: wp.array(dtype=wp.float32), n: wp.int32,
                      res: wp.int32, origin: wp.vec3, inv_dx: wp.float32, scale: wp.float64,
                      grid_mi: wp.array(dtype=wp.int64)):
    p = wp.tid()
    if p >= n:
        return
    x = X[p]
    m = M[p]
    if not wp.isfinite(m) or m <= 0.0:
        return
    xg = (x - origin) * inv_dx
    b = wp.vec3i(wp.int32(wp.floor(xg[0] - 0.5)), wp.int32(wp.floor(xg[1] - 0.5)),
                 wp.int32(wp.floor(xg[2] - 0.5)))
    f = wp.vec3(xg[0] - wp.float32(b[0]), xg[1] - wp.float32(b[1]),
                xg[2] - wp.float32(b[2]))
    for i in range(3):
        ix = b[0] + i
        if ix < 0 or ix >= res:
            continue
        wi = _axis_w(f[0], i)
        for j in range(3):
            iy = b[1] + j
            if iy < 0 or iy >= res:
                continue
            wj = _axis_w(f[1], j)
            for k in range(3):
                iz = b[2] + k
                if iz < 0 or iz >= res:
                    continue
                wk = _axis_w(f[2], k)
                w = wi * wj * wk
                gi = (ix * res + iy) * res + iz
                wp.atomic_add(grid_mi, gi, wp.int64(wp.round(wp.float64(w * m) * scale)))


@wp.kernel
def _fixed_to_float_kernel(grid_mi: wp.array(dtype=wp.int64), scale: wp.float64,
                           grid_m: wp.array(dtype=wp.float32)):
    i = wp.tid()
    grid_m[i] = wp.float32(wp.float64(grid_mi[i]) / scale)


# --------------------------------------------------------------------------------------
# State contract
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Lattice:
    res: int
    origin: tuple
    dx: float
    rho: float
    scale: int = MASS_SCALE
    kernel_id: str = KERNEL_ID


@dataclass(frozen=True)
class Snapshot:
    """Immutable, producer-bound capture. `grid_m == P2G(x_at, m_at)` is a checked invariant."""
    lattice: Lattice
    frame: int
    version: int
    n: int
    grid_m: object
    x_at: object
    m_at: object
    ids_at: object
    grid_source: str            # "factory" (built here) or "solver_verified" (checked once)
    verified: bool
    grid_sha256: str = ""
    binding_sha256: str = ""


@dataclass(frozen=True)
class ParticleState:
    frame: int
    version: int
    x: object
    m: object
    ids: object
    n: int = -1


def _is_wp(a):
    return isinstance(a, wp.array)


def _owned(a, dtype, device):
    if _is_wp(a):
        return wp.clone(a)
    return wp.array(np.asarray(a), dtype=dtype, device=device)


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _lattice_fail(lat):
    if not isinstance(lat.res, int) or lat.res <= 0:
        return "invalid_res"
    if not np.isfinite(lat.dx) or lat.dx <= 0.0:
        return "invalid_dx"
    o = np.asarray(lat.origin, dtype=np.float64).reshape(-1)
    if o.size != 3 or not np.isfinite(o).all():
        return "invalid_origin"
    if not np.isfinite(lat.rho) or lat.rho <= 0.0:
        return "invalid_rho"
    if int(lat.scale) <= 0:
        return "invalid_scale"
    if lat.kernel_id != KERNEL_ID:
        return "unknown_kernel"
    return None


def _p2g_on_device(x, m, lat, device):
    res = int(lat.res)
    gi = wp.zeros(res ** 3, dtype=wp.int64, device=device)
    out = wp.zeros(res ** 3, dtype=wp.float32, device=device)
    n = int(m.size)
    if n > 0:
        with wp.ScopedDevice(device):
            wp.launch(_p2g_int64_kernel, dim=n,
                      inputs=[x, m, n, res, wp.vec3(*[float(o) for o in lat.origin]),
                              np.float32(1.0 / np.float32(lat.dx)), float(lat.scale), gi],
                      device=device)
            wp.launch(_fixed_to_float_kernel, dim=res ** 3, inputs=[gi, float(lat.scale), out],
                      device=device)
    return out


def snapshot_from_particles(x, m, ids, lattice, frame, version, device="cuda:0",
                            digest=False):
    """Producer factory: the grid is BUILT here as P2G(x_at, m_at); coupling by construction."""
    lf = _lattice_fail(lattice)
    if lf is not None:
        raise ValueError(lf)
    xs = _owned(x, wp.vec3, device)
    ms = _owned(m, wp.float32, device)
    isd = _owned(ids, wp.int64, device)
    n = int(ms.size)
    if int(xs.size) != n or int(isd.size) != n:
        raise ValueError("x / m / ids lengths differ")
    with wp.ScopedDevice(device):
        grid = _p2g_on_device(xs, ms, lattice, device)
    g_sha, b_sha = "", ""
    if digest:
        g_sha = _sha_bytes(np.ascontiguousarray(grid.numpy(), dtype=np.float32).tobytes())
        b_sha = _sha_bytes(b"".join([
            np.ascontiguousarray(xs.numpy(), dtype=np.float32).tobytes(),
            np.ascontiguousarray(ms.numpy(), dtype=np.float32).tobytes(),
            np.ascontiguousarray(isd.numpy(), dtype=np.int64).tobytes(),
            np.ascontiguousarray(grid.numpy(), dtype=np.float32).tobytes()]))
    return Snapshot(lattice=lattice, frame=int(frame), version=int(version), n=n,
                    grid_m=grid, x_at=xs, m_at=ms, ids_at=isd, grid_source="factory",
                    verified=True, grid_sha256=g_sha, binding_sha256=b_sha)


def snapshot_from_solver_grid(grid_m, x, m, ids, lattice, frame, version, verify=True,
                              device="cuda:0", digest=False):
    """Capture an EXISTING grid. It is usable only after a one-time bit-equality check against a
    fresh int64 P2G of the supplied particles. A mismatch is rejected (GridNotBoundError)."""
    lf = _lattice_fail(lattice)
    if lf is not None:
        raise ValueError(lf)
    gs = _owned(grid_m, wp.float32, device)
    xs = _owned(x, wp.vec3, device)
    ms = _owned(m, wp.float32, device)
    isd = _owned(ids, wp.int64, device)
    res = int(lattice.res)
    if int(gs.size) != res ** 3:
        raise ValueError("grid_size_mismatch")
    n = int(ms.size)
    if int(xs.size) != n or int(isd.size) != n:
        raise ValueError("x / m / ids lengths differ")
    verified = False
    if verify:
        with wp.ScopedDevice(device):
            recomputed = _p2g_on_device(xs, ms, lattice, device)
        a = np.ascontiguousarray(recomputed.numpy(), dtype=np.float32)
        b = np.ascontiguousarray(gs.numpy(), dtype=np.float32)
        ndiff = int((a != b).sum())
        if ndiff != 0:
            raise GridNotBoundError(
                "supplied grid is not P2G(x_at, m_at): ndiff=%d (rejected, no certificate)" % ndiff)
        verified = True
    g_sha, b_sha = "", ""
    if digest:
        g_sha = _sha_bytes(np.ascontiguousarray(gs.numpy(), dtype=np.float32).tobytes())
        b_sha = _sha_bytes(b"".join([
            np.ascontiguousarray(xs.numpy(), dtype=np.float32).tobytes(),
            np.ascontiguousarray(ms.numpy(), dtype=np.float32).tobytes(),
            np.ascontiguousarray(isd.numpy(), dtype=np.int64).tobytes(),
            np.ascontiguousarray(gs.numpy(), dtype=np.float32).tobytes()]))
    return Snapshot(lattice=lattice, frame=int(frame), version=int(version), n=n,
                    grid_m=gs, x_at=xs, m_at=ms, ids_at=isd, grid_source="solver_verified",
                    verified=verified, grid_sha256=g_sha, binding_sha256=b_sha)


def capture_solver_snapshot(solver, frame, version, verify=True, device="cuda:0"):
    """Documented producer hook for the the simulator checkpoint `MPMSolver`.

    The simulator's `grid_m` is written only by its own `p2g()` from `s.x`/`s.mass`; call this
    immediately after `solver.p2g()` (and before `grid_update`/`g2p`, which do not deposit).
    Binding is CHECKED by `snapshot_from_solver_grid` (recompute int64 P2G and require bit
    equality), not asserted by the caller. IDs are the producer's persistent integer identity;
    the simulator checkpoint appends emitted particles, so index identity is stable (documented assumption).
    """
    n = int(solver.n_particles)
    x = solver.x  # wp.array vec3, full capacity; take the live prefix
    ids = wp.array(np.arange(n, dtype=np.int64), dtype=wp.int64, device=device)
    origin = tuple(float(o) for o in np.asarray(solver.origin_np).reshape(3))
    lat = Lattice(res=int(solver.res), origin=origin, dx=float(solver.dx),
                  rho=float(getattr(solver, "rho", 1600.0)))
    with wp.ScopedDevice(device):
        xp = wp.zeros(n, dtype=wp.vec3, device=device)
        mp = wp.zeros(n, dtype=wp.float32, device=device)
        wp.copy(dest=xp, src=x, count=n)
        wp.copy(dest=mp, src=solver.mass, count=n)
        gm = wp.zeros(int(lat.res) ** 3, dtype=wp.float32, device=device)
        wp.copy(dest=gm, src=solver.grid_m, count=int(lat.res) ** 3)
    return snapshot_from_solver_grid(gm, xp, mp, ids, lat, frame, version, verify=verify,
                                     device=device)


def particle_state(x, m, ids, frame, version, device="cuda:0"):
    xs = x if _is_wp(x) else wp.array(np.asarray(x), dtype=wp.vec3, device=device)
    ms = m if _is_wp(m) else wp.array(np.asarray(m), dtype=wp.float32, device=device)
    isd = ids if _is_wp(ids) else wp.array(np.asarray(ids), dtype=wp.int64, device=device)
    return ParticleState(frame=int(frame), version=int(version), x=xs, m=ms, ids=isd,
                         n=int(ms.size))


# --------------------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------------------
def validate_state(state, cur, *, max_lag=1, max_disp=None, device="cuda:0", cache=None):
    info = {"used": False, "reason": "uninitialized", "D": 0.0, "n_failed_deposition": -1,
            "n_id_mismatch": -1, "frame_lag": -1, "version_ok": False, "lattice_ok": False,
            "n_nonfinite_particles": 0, "grid_verified": False, "cache_hit": False}
    if cur is None or cur.x is None:
        raise ValueError("current ParticleState with positions is required (documented rejection)")
    if state is None:
        info["reason"] = "no_reference_state"
        return info
    lat = state.lattice
    lf = _lattice_fail(lat)
    if lf is not None:
        info["reason"] = lf
        return info
    info["lattice_ok"] = True
    info["grid_verified"] = bool(state.verified)
    if not state.verified:
        info["reason"] = "grid_unverified"
        return info
    res = int(lat.res)
    if int(state.grid_m.size) != res ** 3:
        info["reason"] = "grid_size_mismatch"
        return info
    if state.n != cur.n and cur.n != -1:
        info["reason"] = "particle_set_size_changed"
        return info
    if state.n != int(cur.m.size) or int(cur.x.size) != state.n or int(cur.ids.size) != state.n:
        info["reason"] = "particle_arrays_inconsistent"
        return info
    if int(cur.version) != int(state.version):
        info["reason"] = "particle_version_changed"
        return info
    info["version_ok"] = True
    lag = int(cur.frame) - int(state.frame)
    info["frame_lag"] = lag
    if lag < 0:
        info["reason"] = "grid_newer_than_particles"
        return info
    if lag > int(max_lag):
        info["reason"] = "stale_grid"
        return info
    if int(state.n) == 0:
        info.update(used=True, reason="ok_empty")
        return info
    # amortization cache: keyed by the full public state identity; any change invalidates
    key = None
    if cache is not None:
        key = (int(cur.version), int(cur.frame), int(cur.n), id(cur.x), id(cur.m), id(cur.ids))
        if cache.get("key") == key:
            info.update(cache["info"])
            info["cache_hit"] = True
            return info
    with wp.ScopedDevice(device):
        idm = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(_ids_mismatch_kernel, dim=state.n,
                  inputs=[state.ids_at, cur.ids, state.n, idm], device=device)
        nm = int(idm.numpy()[0])
        info["n_id_mismatch"] = nm
        if nm != 0:
            info["reason"] = "particle_id_mismatch"
            return info
        fail = wp.zeros(state.n, dtype=wp.int32, device=device)
        inv_dx = np.float32(1.0 / np.float32(lat.dx))
        wp.launch(_deposition_valid_kernel, dim=state.n,
                  inputs=[state.x_at, state.m_at, state.n, res,
                          wp.vec3(*[float(o) for o in lat.origin]), inv_dx,
                          float(lat.scale), fail], device=device)
        nf = int(fail.numpy().sum())
        info["n_failed_deposition"] = nf
        if nf != 0:
            info["reason"] = "deposition_invalid_underflow_outside_or_nonfinite"
            return info
        n_ch = (state.n + RED_CHUNK - 1) // RED_CHUNK
        disp = wp.zeros(n_ch, dtype=wp.float32, device=device)
        bad = wp.zeros(n_ch, dtype=wp.int32, device=device)
        wp.launch(_chunk_disp_nonfinite_kernel, dim=n_ch,
                  inputs=[cur.x, state.x_at, cur.m, state.n, RED_CHUNK, disp, bad], device=device)
        d_arr = disp.numpy()
        nfpart = int(bad.numpy().sum())
    info["n_nonfinite_particles"] = nfpart
    if nfpart != 0:
        info["reason"] = "nonfinite_particle_or_mass"
        return info
    D = float(np.max(d_arr)) if d_arr.size else 0.0
    info["D"] = D
    limit = float(lat.dx) if max_disp is None else float(max_disp)
    if not np.isfinite(D):
        info["reason"] = "nonfinite_displacement"
        return info
    if D > limit:
        info["reason"] = "displacement_exceeds_bound"
        return info
    info.update(used=True, reason="ok")
    if cache is not None:
        cache["key"] = key
        cache["info"] = dict(info)
    return info


# --------------------------------------------------------------------------------------
# Occupancy / exact query
# --------------------------------------------------------------------------------------
def occupancy_from_massgrid(mass, lattice, theta=0.0, device="cuda:0"):
    thr = np.float32(float(theta) * float(lattice.rho) * float(lattice.dx) ** 3)
    occ = wp.zeros(int(mass.size), dtype=wp.int32, device=device)
    with wp.ScopedDevice(device):
        wp.launch(_occ_kernel, dim=int(mass.size), inputs=[mass, thr, occ], device=device)
    return occ


def exact_box_counts(P, box_lo, box_hi, *, n=None, device="cuda:0"):
    """Exact strict particle-centre count per box (D1). The strong direct kernel."""
    box_lo = np.asarray(box_lo, dtype=np.float64).reshape(-1, 3)
    box_hi = np.asarray(box_hi, dtype=np.float64).reshape(-1, 3)
    nb = len(box_lo)
    if n is None:
        n = int(P.size)
    n_chunks = max(1, (n + CHUNK - 1) // CHUNK)
    out = wp.zeros(nb, dtype=wp.int32, device=device)
    if nb and n > 0:
        with wp.ScopedDevice(device):
            a = wp.array(box_lo, dtype=wp.vec3d, device=device)
            b = wp.array(box_hi, dtype=wp.vec3d, device=device)
            wp.launch(_particle_box_count_kernel, dim=nb * n_chunks,
                      inputs=[P, n, n_chunks, a, b, out], device=device)
    return out.numpy().astype(np.int32)


def node_index_bounds(box_lo, box_hi, pad, res, origin, dx):
    lo = np.asarray(box_lo, dtype=np.float64).reshape(-1, 3) - pad
    hi = np.asarray(box_hi, dtype=np.float64).reshape(-1, 3) + pad
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    r0 = np.ceil((lo - origin) / dx - 1e-12).astype(np.int64)
    r1 = np.floor((hi - origin) / dx + 1e-12).astype(np.int64)
    valid = (r1 >= 0).all(axis=1) & (r0 <= res - 1).all(axis=1) & (r1 >= r0).all(axis=1)
    i0 = np.clip(r0, 0, res - 1).astype(np.int32)
    i1 = np.clip(r1, 0, res - 1).astype(np.int32)
    ni = np.where(valid, (i1 - i0 + 1).prod(axis=1), 0)
    return i0, i1, valid.astype(np.int32), ni


class GateSession:
    """Amortized gate for one immutable snapshot across a batch of queries on one frame.

    `prepare(cur)` runs the contract once and caches it keyed by the public particle-state
    identity (version, frame, n, array identities); any change invalidates the cache and re-runs
    the contract. `query` reuses the cache. Occupancy is computed once per session.
    """

    def __init__(self, snapshot, device="cuda:0"):
        self.snapshot = snapshot
        self.device = device
        self._occ = None
        self._cache = {}
        self.n_occ_builds = 0

    def _occupancy(self):
        if self._occ is None:
            self._occ = occupancy_from_massgrid(self.snapshot.grid_m, self.snapshot.lattice,
                                                theta=0.0, device=self.device)
            self.n_occ_builds += 1
        return self._occ

    def prepare(self, cur, **kw):
        info = validate_state(self.snapshot, cur, device=self.device, cache=self._cache, **kw)
        return info

    def query(self, cur, box_lo, box_hi, radie=0.0, **kw):
        return _gated(self.snapshot, cur, box_lo, box_hi, radie=radie, session=self, **kw)


def _gated(state, cur, box_lo, box_hi, radie=0.0, session=None, device="cuda:0",
           max_lag=1, max_disp=None, policy=None):
    pol = dict(POLICY if policy is None else policy)
    box_lo = np.asarray(box_lo, dtype=np.float64).reshape(-1, 3)
    box_hi = np.asarray(box_hi, dtype=np.float64).reshape(-1, 3)
    if len(box_lo) != len(box_hi):
        raise ValueError("box_lo / box_hi lengths differ")
    nb = len(box_lo)
    info = validate_state(state, cur, max_lag=max_lag, max_disp=max_disp, device=device,
                          cache=None if session is None else session._cache)
    if info["reason"] == "nonfinite_particle_or_mass":
        raise ValueError("current particle state contains NaN/inf; exact fallback undefined")
    n = int(cur.m.size)
    info.update(nb=nb, route="direct", n_exact_boxes=nb, n_gate_nodes=0,
                n_box_particle_tests=nb * n, gate_fn=0, gate_fp=-1)
    if not info["used"]:
        counts = np.zeros(nb, dtype=np.int32)
        if n > 0:
            counts = exact_box_counts(cur.x, box_lo, box_hi, n=n, device=device)
        gate = np.ones(nb, dtype=np.int32)
        info["route"] = "direct"
        info["n_box_particle_tests"] = nb * n
        return counts, gate, info
    lat = state.lattice
    res = int(lat.res)
    if session is not None:
        occ = session._occupancy()
    else:
        occ = occupancy_from_massgrid(state.grid_m, lat, theta=0.0, device=device)
    pad = float(radie) + 2.0 * float(lat.dx) + float(info["D"])
    info["pad_m"] = pad
    i0, i1, valid, ni = node_index_bounds(box_lo, box_hi, pad, res, lat.origin, lat.dx)
    info["n_gate_nodes"] = int(ni.sum())
    direct_work = nb * n
    gate_work = int(ni.sum())
    use_gated = (nb >= int(pol["min_nb_gated"])) and (
        direct_work == 0 or gate_work <= float(pol["max_gate_work_frac"]) * direct_work)
    if not use_gated:
        counts = np.zeros(nb, dtype=np.int32)
        if n > 0:
            counts = exact_box_counts(cur.x, box_lo, box_hi, n=n, device=device)
        gate = np.ones(nb, dtype=np.int32)
        info["route"] = "direct"
        info["n_box_particle_tests"] = nb * n
        return counts, gate, info
    with wp.ScopedDevice(device):
        a0 = wp.array(i0, dtype=wp.vec3i, device=device)
        a1 = wp.array(i1, dtype=wp.vec3i, device=device)
        vv = wp.array(valid, dtype=wp.int32, device=device)
        gate_buf = wp.zeros(nb, dtype=wp.int32, device=device)
        wp.launch(_grid_box_count_kernel, dim=nb,
                  inputs=[occ, res, a0, a1, vv, gate_buf], device=device)
    gate = gate_buf.numpy().astype(np.int32)
    flagged = np.nonzero(gate > 0)[0]
    counts = np.zeros(nb, dtype=np.int32)
    if len(flagged) and n > 0:
        fc = exact_box_counts(cur.x, box_lo[flagged], box_hi[flagged], n=n, device=device)
        counts[flagged] = fc
    info["route"] = "gated"
    info["n_exact_boxes"] = int(len(flagged))
    info["n_box_particle_tests"] = int(len(flagged)) * n
    return counts, gate, info


def keepout_gated(state, cur, box_lo, box_hi, radie=0.0, *, max_lag=1, max_disp=None,
                  device="cuda:0", policy=None):
    """Non-amortized: run the contract and the gate for this query. Returns (counts, gate, info)."""
    return _gated(state, cur, box_lo, box_hi, radie=radie, session=None, device=device,
                  max_lag=max_lag, max_disp=max_disp, policy=policy)
