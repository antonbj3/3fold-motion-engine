#!/usr/bin/env python3
"""Graph-coloured Gauss-Seidel GPU contact engine, behind the same ContactEngine Protocol.

Same bodies (voxelised boxes), same hash-grid broad phase and the same contact model as
`contact_engine_gpu.RelaxedJacobiContactEngine`; only the solve differs. Contacts are coloured so that no
two contacts in one colour share a dynamic body, and the colours are solved sequentially. Inside a colour
every contact touches a disjoint set of bodies, so the per-contact impulses are applied with PLAIN reads
and writes of the body velocity: no atomics, no fixed-point accumulator, no relaxation factor. That is
Gauss-Seidel between colours and Jacobi (order-free) inside a colour.

Determinism. Two sources of run-to-run variation are removed:
  - Inside a colour the written bodies are disjoint, so the result does not depend on the order in which
    the contacts of that colour execute. Bit-identity is asserted by the selftest, not assumed.
  - The contact ORDER coming out of the generation kernel is an atomic-counter order and does vary between
    runs; the colouring would then partition the contacts differently and the Gauss-Seidel sweep order
    would change the result. Contacts are therefore sorted by a stable FEATURE ID (owning point index of
    body A, and of body B or -1 for the ground) before colouring, so the colouring input, the colour
    assignment and the sweep order are all independent of the atomic-counter order.

The colouring itself runs ON THE GPU (this step): a parallel claim/win loop, one colour per round, where
each still-uncoloured contact claims its dynamic bodies with an atomic minimum over a Jones-Plassmann key
(a hash of its rank in the sorted contact list, packed with the rank, so the key is unique and a pure
function of the deterministic rank) and takes the colour when it won every body it needs. Only the number of
rounds and the colour offsets are read back to the host.

Default iteration counts are 20 velocity and 20 position iterations (40 velocity with the manifold
reduction on): measured, that is the lowest velocity
count at which the K=8 stack keeps M5 (max penetration / R) bounded over 600 steps. The Jacobi engine ships
with 40 velocity and 20 position iterations.

Warm start is by feature id across steps: the accumulated normal and friction impulses of a contact whose
feature id reappears in the next step are used as the starting impulses of that contact.

Manifold reduction (`manifold_reduce=True`, default off so the plain coloured numbers stay reproducible).
The voxelised boxes put ~9 contact points on every body pair, and contacts that share a body pair can never
share a colour, so the colour count of a stack is set by that redundancy. With the flag on, the contacts are
grouped by body pair after generation and each pair is reduced to at most 4 representative points before the
colouring. Which 4: the E-optimal rule of the repository's manifold-reduction cells. A face contact wrench is
3-dimensional (F_z, tau_x, tau_y), so the wrench map G = [1, r_y, -r_x] per point, taken about the projected
centre of mass of body A, must have rank 3 -- two points leave a torque unconstrained (the pair rocks about
the axis through them), three are the minimum. Among the subsets of that size the E-optimal choice maximises
sigma_min(G), the worst-direction wrench margin, which for points expressed about the reference equals
sqrt(min(npts, lambda_min(second-moment scatter))); that selects a round straddle of the reference, not the
largest-area triple. The deepest point of the pair is always kept and seeds a greedy that adds the three
points with the largest sigma_min(G) in turn. The pair's total normal impulse is conserved: every dropped
contact's warm-started normal impulse is added to the kept point nearest to it in the contact plane, so the
sum of the normal impulses over a pair is unchanged by the reduction and the reported contact force still
sums to Mg on a resting box. The reduction runs on the device by default: a stable radix sort by
feature key, a second stable radix sort by body-pair key, a run-length kernel for the segment boundaries,
one thread per body pair for the E-optimal selection and the impulse redistribution, a scan and a
compaction, with the warm start done by binary search in the kept keys of the previous step, so no contact
data crosses to the host. `manifold_on_host=True` keeps the numpy stage for A/B; the two paths select the
same points and produce bit-identical states.

Pair chunks (`pair_chunks=True`, default off; needs the device manifold reduction). With the reduction on,
a body pair carries at most 4 contacts, and those 4 can never share a colour, so a pair converges only
across colour sweeps. With this flag the COLOURING runs on the body-pair graph (a node per contacting body
pair, an edge where two pairs share a dynamic body) and one thread takes a whole pair: its <= 4 contacts are
solved one after the other in the thread, which is an exact sequential Gauss-Seidel sweep of the chunk (the
WY compression of the updates does not apply: each update is state dependent through max(0, .) on the normal
impulse and the projection onto the friction cone, so they are not a fixed product of rank-1 projectors).
`chunk_mode='registers'` (default) keeps the two bodies' velocities in registers over the chunk,
`chunk_mode='global'` calls the same per-contact warp function once per contact; both produce the same
states bit for bit. Bodies are still disjoint across the threads of a colour, so the writes stay plain. The
pair graph needs far fewer colours than the contact graph (measured: 3 against 8 on the K=8 stack), and the
grouping is built on the device (radix sort on the pair key, run-length flags, a scan, one thread per
segment). The whole chunk loop is captured as a CUDA graph exactly as the per-contact loop is.

Warp is imported lazily: the module imports without warp and only raises at instantiation.

Graph capture (`graph_capture=True`, default off). The solve is 8 colours x (40 velocity + 20 position)
sweeps = 480 kernel launches per step, each of them a fixed amount of host work, and at N = 1e4 that host
work, not the arithmetic, was the step. The whole loop is captured once as a CUDA graph and replayed with
`wp.capture_launch`. Nothing that changes between steps is baked into the capture: each colour's offset and
count are read from device arrays (`graph_mode='device'`, the default), which are built by a counting sort
on the colour id, a histogram and a scan, so the colour array no longer crosses to the host either. The
graph is re-captured only when the number of colours or the contact-count bucket changes.
`graph_mode='spans'` bakes the offsets into the launches and re-captures whenever the partition changes; it
is kept for comparison. Both produce the states of the uncaptured path bit for bit.

  python -u -m motion_engine.contact_engine_gpu_colored     # contract selftest (requires warp + CUDA)
"""
import time
from contextlib import contextmanager

import numpy as np

from motion_engine.contact_engine import BodyState, ContactEngine   # same data contract as the CPU engine
from motion_engine.contact_engine_gpu import _build_kernels, _voxbox, _G, _R, _BETA, _SLOP, _MAXC

_BIG = 2147483647          # "no claim" sentinel for the colouring pass


def _build_colored_kernels(wp):
    """Kernels specific to the coloured solver: contact generation with feature ids, the gather that puts
    the contacts in feature-id order, the colouring rounds and the two Gauss-Seidel sweeps.

    `k_gen_fid` is `contact_engine_gpu.k_gen` plus the two feature-index outputs (the generation kernel of
    the Jacobi engine is method-local to its kernel builder and does not emit them); the broad phase, the
    radius test, the normals and the penetrations are unchanged."""

    @wp.kernel
    def k_gen_fid(allp: wp.array(dtype=wp.vec3), owner: wp.array(dtype=int), grid: wp.uint64,
                  cnt: wp.array(dtype=int), cbi: wp.array(dtype=int), cbj: wp.array(dtype=int),
                  cpA: wp.array(dtype=wp.vec3), cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3),
                  cpen: wp.array(dtype=float), cfa: wp.array(dtype=int), cfb: wp.array(dtype=int)):
        gid = wp.tid(); xi = allp[gid]; bi = owner[gid]; s = xi[2] - _R
        if s < 0.0:
            idx = wp.atomic_add(cnt, 0, 1)
            if idx < _MAXC:
                cbi[idx] = bi; cbj[idx] = -1; cpA[idx] = xi; cpB[idx] = wp.vec3(0.0, 0.0, 0.0)
                cn[idx] = wp.vec3(0.0, 0.0, 1.0); cpen[idx] = -s; cfa[idx] = gid; cfb[idx] = -1
        qy = wp.hash_grid_query(grid, xi, 2.0 * _R); j = int(0)
        while wp.hash_grid_query_next(qy, j):
            if j > gid and owner[j] != bi:
                dvec = xi - allp[j]; dist = wp.length(dvec)
                if dist < 2.0 * _R and dist > 1e-9:
                    idx = wp.atomic_add(cnt, 0, 1)
                    if idx < _MAXC:
                        cbi[idx] = bi; cbj[idx] = owner[j]; cpA[idx] = xi; cpB[idx] = allp[j]
                        cn[idx] = dvec / dist; cpen[idx] = 2.0 * _R - dist
                        cfa[idx] = gid; cfb[idx] = j

    @wp.kernel
    def k_gather(perm: wp.array(dtype=int), C: int,
                 cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3),
                 cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3), cpen: wp.array(dtype=float),
                 obi: wp.array(dtype=int), obj: wp.array(dtype=int), opA: wp.array(dtype=wp.vec3),
                 opB: wp.array(dtype=wp.vec3), on: wp.array(dtype=wp.vec3), open_: wp.array(dtype=float)):
        t = wp.tid()
        if t >= C:
            return
        k = perm[t]
        obi[t] = cbi[k]; obj[t] = cbj[k]; opA[t] = cpA[k]; opB[t] = cpB[k]; on[t] = cn[k]; open_[t] = cpen[k]

    @wp.kernel
    def k_prio(C: int, pkey: wp.array(dtype=wp.int64)):
        """Jones-Plassmann priority: a hash of the contact rank, packed with the rank itself so the key is
        unique. A priority that increases monotonically along a chain of contacts (a stack) would colour one
        contact per round; the hash breaks that chain while staying a pure function of the rank, hence
        deterministic."""
        c = wp.tid()
        if c >= C:
            return
        st = wp.rand_init(7919, c); p = wp.int64(wp.randi(st))
        if p < wp.int64(0):
            p = -p
        pkey[c] = p * wp.int64(4294967296) + wp.int64(c)

    @wp.kernel
    def k_claim(C: int, col: wp.array(dtype=int), cbi: wp.array(dtype=int), cbj: wp.array(dtype=int),
                invM: wp.array(dtype=float), pkey: wp.array(dtype=wp.int64), claim: wp.array(dtype=wp.int64)):
        c = wp.tid()
        if c >= C or col[c] >= 0:
            return
        bi = cbi[c]; bj = cbj[c]; k = pkey[c]
        if invM[bi] > 0.0:
            wp.atomic_min(claim, bi, k)          # min over unique keys: order-independent
        if bj >= 0:
            if invM[bj] > 0.0:
                wp.atomic_min(claim, bj, k)

    @wp.kernel
    def k_win(C: int, cur: int, col: wp.array(dtype=int), cbi: wp.array(dtype=int), cbj: wp.array(dtype=int),
              invM: wp.array(dtype=float), pkey: wp.array(dtype=wp.int64), claim: wp.array(dtype=wp.int64),
              left: wp.array(dtype=int)):
        c = wp.tid()
        if c >= C or col[c] >= 0:
            return
        bi = cbi[c]; bj = cbj[c]; k = pkey[c]; ok = True
        if invM[bi] > 0.0:
            if claim[bi] != k:
                ok = False
        if bj >= 0:
            if invM[bj] > 0.0:
                if claim[bj] != k:
                    ok = False
        if ok:
            col[c] = cur
        else:
            wp.atomic_add(left, 0, 1)

    @wp.kernel
    def k_clear_claim(claim: wp.array(dtype=wp.int64)):
        claim[wp.tid()] = wp.int64(9223372036854775807)

    @wp.func
    def f_gs_vel(c: int,
                 cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3),
                 cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3), jn: wp.array(dtype=float),
                 jt1: wp.array(dtype=float), jt2: wp.array(dtype=float), xc: wp.array(dtype=wp.vec3),
                 v: wp.array(dtype=wp.vec3), w: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33),
                 invM: wp.array(dtype=float), mu: float, t2d: float):
        """One contact of the velocity sweep. Same expressions as before; the body is a function only so
        that the launch-offset form and the device-offset form of the sweep share one compiled body."""
        bi = cbi[c]; bj = cbj[c]; nrm = cn[c]; rA = cpA[c] - xc[bi]
        iMA = invM[bi]; IA = invIw[bi]; va = v[bi] + wp.cross(w[bi], rA)
        vb = wp.vec3(0.0, 0.0, 0.0); rB = wp.vec3(0.0, 0.0, 0.0); iMB = float(0.0)
        if bj >= 0:
            rB = cpB[c] - xc[bj]; vb = v[bj] + wp.cross(w[bj], rB); iMB = invM[bj]
        # 3D cone on a slip-direction basis: t1 = slip, t2 perpendicular -> circular mu*jn disc
        vrel = va - vb; vn = wp.dot(vrel, nrm); vt = vrel - vn * nrm; mt = wp.length(vt)
        t1 = vt / mt
        if mt <= 5e-3:
            aa = wp.vec3(1.0, 0.0, 0.0)
            if wp.abs(nrm[0]) >= 0.9:
                aa = wp.vec3(0.0, 1.0, 0.0)
            t1 = wp.normalize(aa - wp.dot(aa, nrm) * nrm)
        t2 = wp.cross(nrm, t1)
        rnA = wp.cross(rA, nrm); meff = iMA + wp.dot(rnA, IA * rnA)
        if bj >= 0:
            rnB = wp.cross(rB, nrm); meff = meff + iMB + wp.dot(rnB, invIw[bj] * rnB)
        if meff < 1e-12:
            return
        dj = -vn / meff; nw = wp.max(0.0, jn[c] + dj); dj = nw - jn[c]; jn[c] = nw; Jn = dj * nrm
        if iMA > 0.0:
            v[bi] = v[bi] + Jn * iMA; w[bi] = w[bi] + IA * wp.cross(rA, Jn)
        if bj >= 0:
            if iMB > 0.0:
                v[bj] = v[bj] - Jn * iMB; w[bj] = w[bj] - invIw[bj] * wp.cross(rB, Jn)
        # friction: own effective masses per tangent, vector clamp to the circular mu*jn disc
        va = v[bi] + wp.cross(w[bi], rA); vb = wp.vec3(0.0, 0.0, 0.0)
        if bj >= 0:
            vb = v[bj] + wp.cross(w[bj], rB)
        vrel = va - vb
        rtA1 = wp.cross(rA, t1); meft1 = iMA + wp.dot(rtA1, IA * rtA1)
        rtA2 = wp.cross(rA, t2); meft2 = iMA + wp.dot(rtA2, IA * rtA2)
        if bj >= 0:
            rtB1 = wp.cross(rB, t1); meft1 = meft1 + iMB + wp.dot(rtB1, invIw[bj] * rtB1)
            rtB2 = wp.cross(rB, t2); meft2 = meft2 + iMB + wp.dot(rtB2, invIw[bj] * rtB2)
        if meft1 < 1e-12 or meft2 < 1e-12:
            return
        o1 = jt1[c]; o2 = jt2[c]; a1 = o1 - wp.dot(vrel, t1) / meft1; a2 = o2 - wp.dot(vrel, t2) / meft2
        mag = wp.sqrt(a1 * a1 + a2 * a2); lim = mu * jn[c]
        if mag > lim and mag > 1e-12:
            a1 = a1 * lim / mag; a2 = a2 * lim / mag
        jt1[c] = a1; jt2[c] = a2; Jt = (a1 - o1) * t1 + t2d * (a2 - o2) * t2
        if iMA > 0.0:
            v[bi] = v[bi] + Jt * iMA; w[bi] = w[bi] + IA * wp.cross(rA, Jt)
        if bj >= 0:
            if iMB > 0.0:
                v[bj] = v[bj] - Jt * iMB; w[bj] = w[bj] - invIw[bj] * wp.cross(rB, Jt)

    @wp.kernel
    def k_gs_vel(order: wp.array(dtype=int), start: int, count: int,
                 cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3),
                 cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3), jn: wp.array(dtype=float),
                 jt1: wp.array(dtype=float), jt2: wp.array(dtype=float), xc: wp.array(dtype=wp.vec3),
                 v: wp.array(dtype=wp.vec3), w: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33),
                 invM: wp.array(dtype=float), mu: float, t2d: float):
        t = wp.tid()
        if t >= count:
            return
        f_gs_vel(order[start + t], cbi, cbj, cpA, cpB, cn, jn, jt1, jt2, xc, v, w, invIw, invM, mu, t2d)

    @wp.kernel
    def k_gs_vel_d(order: wp.array(dtype=int), ci: int, cstart: wp.array(dtype=int),
                   ccount: wp.array(dtype=int),
                 cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3),
                 cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3), jn: wp.array(dtype=float),
                 jt1: wp.array(dtype=float), jt2: wp.array(dtype=float), xc: wp.array(dtype=wp.vec3),
                 v: wp.array(dtype=wp.vec3), w: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33),
                 invM: wp.array(dtype=float), mu: float, t2d: float):
        """The same sweep with the colour's offset and count read from device arrays instead of from launch
        arguments, so one CUDA graph serves every step: the launch dimension is a fixed upper bound and the
        threads beyond the colour's count return."""
        t = wp.tid()
        if t >= ccount[ci]:
            return
        f_gs_vel(order[cstart[ci] + t], cbi, cbj, cpA, cpB, cn, jn, jt1, jt2, xc, v, w, invIw, invM, mu, t2d)

    @wp.func
    def f_gs_pos(c: int,
                 cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3),
                 cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3), cpen: wp.array(dtype=float),
                 jp: wp.array(dtype=float), xc: wp.array(dtype=wp.vec3), pv: wp.array(dtype=wp.vec3),
                 po: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33), invM: wp.array(dtype=float),
                 dt: float):
        """One contact of the position sweep; a function for the same reason as `f_gs_vel`."""
        bi = cbi[c]; bj = cbj[c]; nrm = cn[c]; rA = cpA[c] - xc[bi]; iMA = invM[bi]; IA = invIw[bi]
        rel = pv[bi] + wp.cross(po[bi], rA); rB = wp.vec3(0.0, 0.0, 0.0); iMB = float(0.0)
        if bj >= 0:
            rB = cpB[c] - xc[bj]; rel = rel - (pv[bj] + wp.cross(po[bj], rB)); iMB = invM[bj]
        rnA = wp.cross(rA, nrm); meff = iMA + wp.dot(rnA, IA * rnA)
        if bj >= 0:
            rnB = wp.cross(rB, nrm); meff = meff + iMB + wp.dot(rnB, invIw[bj] * rnB)
        if meff < 1e-12:
            return
        bias = _BETA * wp.max(cpen[c] - _SLOP, 0.0) / dt; dj = (bias - wp.dot(rel, nrm)) / meff
        nw = wp.max(0.0, jp[c] + dj); dj = nw - jp[c]; jp[c] = nw; Jn = dj * nrm
        if iMA > 0.0:
            pv[bi] = pv[bi] + Jn * iMA; po[bi] = po[bi] + IA * wp.cross(rA, Jn)
        if bj >= 0:
            if iMB > 0.0:
                pv[bj] = pv[bj] - Jn * iMB; po[bj] = po[bj] - invIw[bj] * wp.cross(rB, Jn)

    @wp.kernel
    def k_gs_pos(order: wp.array(dtype=int), start: int, count: int,
                 cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3),
                 cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3), cpen: wp.array(dtype=float),
                 jp: wp.array(dtype=float), xc: wp.array(dtype=wp.vec3), pv: wp.array(dtype=wp.vec3),
                 po: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33), invM: wp.array(dtype=float),
                 dt: float):
        t = wp.tid()
        if t >= count:
            return
        f_gs_pos(order[start + t], cbi, cbj, cpA, cpB, cn, cpen, jp, xc, pv, po, invIw, invM, dt)

    @wp.kernel
    def k_gs_pos_d(order: wp.array(dtype=int), ci: int, cstart: wp.array(dtype=int),
                   ccount: wp.array(dtype=int),
                 cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3),
                 cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3), cpen: wp.array(dtype=float),
                 jp: wp.array(dtype=float), xc: wp.array(dtype=wp.vec3), pv: wp.array(dtype=wp.vec3),
                 po: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33), invM: wp.array(dtype=float),
                 dt: float):
        t = wp.tid()
        if t >= ccount[ci]:
            return
        f_gs_pos(order[cstart[ci] + t], cbi, cbj, cpA, cpB, cn, cpen, jp, xc, pv, po, invIw, invM, dt)

    @wp.kernel
    def k_colkey(C: int, col: wp.array(dtype=int), keys: wp.array(dtype=wp.int64), vals: wp.array(dtype=int)):
        """Key/value pairs for the counting sort of the contacts by colour id."""
        i = wp.tid()
        if i >= C:
            return
        keys[i] = wp.int64(col[i]); vals[i] = i

    @wp.kernel
    def k_colcount(C: int, col: wp.array(dtype=int), ccount: wp.array(dtype=int)):
        """Histogram of the colour ids: the count of each colour, from which the scan gives the offsets."""
        i = wp.tid()
        if i >= C:
            return
        wp.atomic_add(ccount, col[i], 1)


    # ── pair chunks (step 3c): colour the BODY-PAIR graph, solve a pair's <= 4 contacts as one chunk ──

    @wp.kernel
    def k_pairkey(C: int, cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), N1: wp.int64,
                  keys: wp.array(dtype=wp.int64), vals: wp.array(dtype=int)):
        """Body-pair key of every kept contact, for the stable sort that makes a pair's contacts
        contiguous."""
        i = wp.tid()
        if i >= C:
            return
        keys[i] = wp.int64(cbi[i]) * N1 + wp.int64(cbj[i]) + wp.int64(1)
        vals[i] = i

    @wp.kernel
    def k_pairflag(C: int, keys: wp.array(dtype=wp.int64), flag: wp.array(dtype=int)):
        """1 where a new body pair starts in the pair-sorted list."""
        i = wp.tid()
        if i >= C:
            return
        if i == 0:
            flag[i] = 1
        else:
            if keys[i] != keys[i - 1]:
                flag[i] = 1
            else:
                flag[i] = 0

    @wp.kernel
    def k_pairbuild(C: int, flag: wp.array(dtype=int), scan: wp.array(dtype=int), pidx: wp.array(dtype=int),
                    cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), pstart: wp.array(dtype=int),
                    pcount: wp.array(dtype=int), pbi: wp.array(dtype=int), pbj: wp.array(dtype=int),
                    flags: wp.array(dtype=int)):
        """One thread per pair-segment start: the offset and the length of the segment in the pair-sorted
        contact index list and the two bodies of the pair. flags[0] = number of pairs, flags[1] = segments
        longer than the chunk capacity (reported to the host, which raises)."""
        i = wp.tid()
        if i >= C:
            return
        if i == C - 1:
            flags[0] = scan[i]
        if flag[i] == 0:
            return
        L = int(1)
        while i + L < C and flag[i + L] == 0:
            L += 1
        if L > _MAXCHUNK:
            wp.atomic_add(flags, 1, 1)
            L = _MAXCHUNK
        g = scan[i] - 1
        pstart[g] = i; pcount[g] = L
        c = pidx[i]; pbi[g] = cbi[c]; pbj[g] = cbj[c]

    @wp.kernel
    def k_chunk_vel_g(porder: wp.array(dtype=int), pidx: wp.array(dtype=int), pstart: wp.array(dtype=int),
                      pcount: wp.array(dtype=int), pbi: wp.array(dtype=int), pbj: wp.array(dtype=int),
                      ci: int, cstart: wp.array(dtype=int), ccount: wp.array(dtype=int),
                      cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3),
                      cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3), jn: wp.array(dtype=float),
                      jt1: wp.array(dtype=float), jt2: wp.array(dtype=float), xc: wp.array(dtype=wp.vec3),
                      v: wp.array(dtype=wp.vec3), w: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33),
                      invM: wp.array(dtype=float), mu: float, t2d: float):
        """Chunk velocity sweep, chunk_mode='global': one thread per body pair of the colour, the pair's
        contacts solved one after another through the same `f_gs_vel` the per-contact sweep uses, so the
        body velocity round-trips through global memory between the contacts of the chunk. Safe because the
        thread is the only writer of those two bodies within the colour."""
        t = wp.tid()
        if t >= ccount[ci]:
            return
        p = porder[cstart[ci] + t]
        st = pstart[p]; n = pcount[p]
        for k in range(n):
            f_gs_vel(pidx[st + k], cbi, cbj, cpA, cpB, cn, jn, jt1, jt2, xc, v, w, invIw, invM, mu, t2d)

    @wp.kernel
    def k_chunk_vel_r(porder: wp.array(dtype=int), pidx: wp.array(dtype=int), pstart: wp.array(dtype=int),
                      pcount: wp.array(dtype=int), pbi: wp.array(dtype=int), pbj: wp.array(dtype=int),
                      ci: int, cstart: wp.array(dtype=int), ccount: wp.array(dtype=int),
                      cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3),
                      cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3), jn: wp.array(dtype=float),
                      jt1: wp.array(dtype=float), jt2: wp.array(dtype=float), xc: wp.array(dtype=wp.vec3),
                      v: wp.array(dtype=wp.vec3), w: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33),
                      invM: wp.array(dtype=float), mu: float, t2d: float):
        """The same chunk, chunk_mode='registers': the two bodies of the pair are loaded once, every contact
        of the chunk updates the local velocities, and they are stored once at the end. Expression for
        expression and in the same order as `f_gs_vel`, so this is the exact sequential Gauss-Seidel result
        of the chunk (the WY compression of the updates does not apply here: each of the <= 4 updates is
        state dependent through max(0, .) on the normal impulse and the projection onto the friction cone,
        so they are not a fixed product of rank-1 projectors)."""
        t = wp.tid()
        if t >= ccount[ci]:
            return
        p = porder[cstart[ci] + t]
        st = pstart[p]; n = pcount[p]
        bi = pbi[p]; bj = pbj[p]
        iMA = invM[bi]; IA = invIw[bi]
        vA = v[bi]; wA = w[bi]
        iMB = float(0.0); IB = invIw[bi]
        vB = wp.vec3(0.0, 0.0, 0.0); wB = wp.vec3(0.0, 0.0, 0.0)
        if bj >= 0:
            iMB = invM[bj]; IB = invIw[bj]; vB = v[bj]; wB = w[bj]
        for k in range(n):
            c = pidx[st + k]
            nrm = cn[c]; rA = cpA[c] - xc[bi]
            va = vA + wp.cross(wA, rA)
            vb = wp.vec3(0.0, 0.0, 0.0); rB = wp.vec3(0.0, 0.0, 0.0)
            if bj >= 0:
                rB = cpB[c] - xc[bj]; vb = vB + wp.cross(wB, rB)
            vrel = va - vb; vn = wp.dot(vrel, nrm); vt = vrel - vn * nrm; mt = wp.length(vt)
            t1 = vt / mt
            if mt <= 5e-3:
                aa = wp.vec3(1.0, 0.0, 0.0)
                if wp.abs(nrm[0]) >= 0.9:
                    aa = wp.vec3(0.0, 1.0, 0.0)
                t1 = wp.normalize(aa - wp.dot(aa, nrm) * nrm)
            t2 = wp.cross(nrm, t1)
            rnA = wp.cross(rA, nrm); meff = iMA + wp.dot(rnA, IA * rnA)
            if bj >= 0:
                rnB = wp.cross(rB, nrm); meff = meff + iMB + wp.dot(rnB, IB * rnB)
            if meff >= 1e-12:
                dj = -vn / meff; nw = wp.max(0.0, jn[c] + dj); dj = nw - jn[c]; jn[c] = nw; Jn = dj * nrm
                if iMA > 0.0:
                    vA = vA + Jn * iMA; wA = wA + IA * wp.cross(rA, Jn)
                if bj >= 0:
                    if iMB > 0.0:
                        vB = vB - Jn * iMB; wB = wB - IB * wp.cross(rB, Jn)
                va = vA + wp.cross(wA, rA); vb = wp.vec3(0.0, 0.0, 0.0)
                if bj >= 0:
                    vb = vB + wp.cross(wB, rB)
                vrel = va - vb
                rtA1 = wp.cross(rA, t1); meft1 = iMA + wp.dot(rtA1, IA * rtA1)
                rtA2 = wp.cross(rA, t2); meft2 = iMA + wp.dot(rtA2, IA * rtA2)
                if bj >= 0:
                    rtB1 = wp.cross(rB, t1); meft1 = meft1 + iMB + wp.dot(rtB1, IB * rtB1)
                    rtB2 = wp.cross(rB, t2); meft2 = meft2 + iMB + wp.dot(rtB2, IB * rtB2)
                if meft1 >= 1e-12:
                    if meft2 >= 1e-12:
                        o1 = jt1[c]; o2 = jt2[c]
                        a1 = o1 - wp.dot(vrel, t1) / meft1; a2 = o2 - wp.dot(vrel, t2) / meft2
                        mag = wp.sqrt(a1 * a1 + a2 * a2); lim = mu * jn[c]
                        if mag > lim and mag > 1e-12:
                            a1 = a1 * lim / mag; a2 = a2 * lim / mag
                        jt1[c] = a1; jt2[c] = a2; Jt = (a1 - o1) * t1 + t2d * (a2 - o2) * t2
                        if iMA > 0.0:
                            vA = vA + Jt * iMA; wA = wA + IA * wp.cross(rA, Jt)
                        if bj >= 0:
                            if iMB > 0.0:
                                vB = vB - Jt * iMB; wB = wB - IB * wp.cross(rB, Jt)
        if iMA > 0.0:
            v[bi] = vA; w[bi] = wA
        if bj >= 0:
            if iMB > 0.0:
                v[bj] = vB; w[bj] = wB

    @wp.kernel
    def k_chunk_pos_g(porder: wp.array(dtype=int), pidx: wp.array(dtype=int), pstart: wp.array(dtype=int),
                      pcount: wp.array(dtype=int), pbi: wp.array(dtype=int), pbj: wp.array(dtype=int),
                      ci: int, cstart: wp.array(dtype=int), ccount: wp.array(dtype=int),
                      cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3),
                      cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3), cpen: wp.array(dtype=float),
                      jp: wp.array(dtype=float), xc: wp.array(dtype=wp.vec3), pv: wp.array(dtype=wp.vec3),
                      po: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33),
                      invM: wp.array(dtype=float), dt: float):
        """Chunk position sweep, chunk_mode='global'."""
        t = wp.tid()
        if t >= ccount[ci]:
            return
        p = porder[cstart[ci] + t]
        st = pstart[p]; n = pcount[p]
        for k in range(n):
            f_gs_pos(pidx[st + k], cbi, cbj, cpA, cpB, cn, cpen, jp, xc, pv, po, invIw, invM, dt)

    @wp.kernel
    def k_chunk_pos_r(porder: wp.array(dtype=int), pidx: wp.array(dtype=int), pstart: wp.array(dtype=int),
                      pcount: wp.array(dtype=int), pbi: wp.array(dtype=int), pbj: wp.array(dtype=int),
                      ci: int, cstart: wp.array(dtype=int), ccount: wp.array(dtype=int),
                      cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3),
                      cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3), cpen: wp.array(dtype=float),
                      jp: wp.array(dtype=float), xc: wp.array(dtype=wp.vec3), pv: wp.array(dtype=wp.vec3),
                      po: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33),
                      invM: wp.array(dtype=float), dt: float):
        """Chunk position sweep, chunk_mode='registers': same expressions as `f_gs_pos`, pseudo-velocities of
        the two bodies loaded once and stored once."""
        t = wp.tid()
        if t >= ccount[ci]:
            return
        p = porder[cstart[ci] + t]
        st = pstart[p]; n = pcount[p]
        bi = pbi[p]; bj = pbj[p]
        iMA = invM[bi]; IA = invIw[bi]
        pvA = pv[bi]; poA = po[bi]
        iMB = float(0.0); IB = invIw[bi]
        pvB = wp.vec3(0.0, 0.0, 0.0); poB = wp.vec3(0.0, 0.0, 0.0)
        if bj >= 0:
            iMB = invM[bj]; IB = invIw[bj]; pvB = pv[bj]; poB = po[bj]
        for k in range(n):
            c = pidx[st + k]
            nrm = cn[c]; rA = cpA[c] - xc[bi]
            rel = pvA + wp.cross(poA, rA); rB = wp.vec3(0.0, 0.0, 0.0)
            if bj >= 0:
                rB = cpB[c] - xc[bj]; rel = rel - (pvB + wp.cross(poB, rB))
            rnA = wp.cross(rA, nrm); meff = iMA + wp.dot(rnA, IA * rnA)
            if bj >= 0:
                rnB = wp.cross(rB, nrm); meff = meff + iMB + wp.dot(rnB, IB * rnB)
            if meff >= 1e-12:
                bias = _BETA * wp.max(cpen[c] - _SLOP, 0.0) / dt; dj = (bias - wp.dot(rel, nrm)) / meff
                nw = wp.max(0.0, jp[c] + dj); dj = nw - jp[c]; jp[c] = nw; Jn = dj * nrm
                if iMA > 0.0:
                    pvA = pvA + Jn * iMA; poA = poA + IA * wp.cross(rA, Jn)
                if bj >= 0:
                    if iMB > 0.0:
                        pvB = pvB - Jn * iMB; poB = poB - IB * wp.cross(rB, Jn)
        if iMA > 0.0:
            pv[bi] = pvA; po[bi] = poA
        if bj >= 0:
            if iMB > 0.0:
                pv[bj] = pvB; po[bj] = poB

    return dict(gen_fid=k_gen_fid, gather=k_gather, prio=k_prio, claim=k_claim, win=k_win,
                clear_claim=k_clear_claim, gs_vel=k_gs_vel, gs_pos=k_gs_pos, gs_vel_d=k_gs_vel_d,
                gs_pos_d=k_gs_pos_d, colkey=k_colkey, colcount=k_colcount,
                pairkey=k_pairkey, pairflag=k_pairflag, pairbuild=k_pairbuild,
                chunk_vel_g=k_chunk_vel_g, chunk_vel_r=k_chunk_vel_r,
                chunk_pos_g=k_chunk_pos_g, chunk_pos_r=k_chunk_pos_r)


_NCOLMAX = 512            # capacity of the per-colour offset/count arrays of the captured solve
_MAXCHUNK = 8              # capacity of one body-pair chunk of the pair-chunk solve (<= 4 after reduction)
_MAXSEG = 1024             # sanity cap on the number of contacts of one body pair in the selection kernel


def _build_device_manifold_kernels(wp, max_keep=8):
    """Kernels of the manifold stage on the device: feature key, pair key, run-length segment flags, the
    per-segment E-optimal selection, the compaction of the kept contacts and the colouring priority.

    Every kernel reproduces the arithmetic of the host stage expression by expression, including the
    float32 point difference promoted to float64 for the planar coordinates and the first-index tie break of
    `np.argmax` / `np.argmin`, so the kept-point sets are the same list, not merely an equivalent one.

    The selection kernel is compiled into its own module with floating-point contraction off
    (`fuse_fp=False`). A flat contact manifold is full of exact ties -- on a 3x3 ground patch seven of the
    eight candidates score identically in the first greedy round -- and a fused multiply-add changes the last
    bit of one candidate, so the tie breaks the other way and the kept set is a different (equally E-optimal)
    one. Without contraction the device scores are the same doubles as the numpy scores and the first-index
    tie break lands on the same point. The solve kernels keep the default compilation settings."""
    veck = wp.types.vector(length=max_keep, dtype=wp.int32)

    @wp.func
    def _planar(p: wp.vec3, ref: wp.vec3, u: wp.vec3d, w: wp.vec3d):
        """Planar coordinates of a contact point about the reference, in the (u, w) tangent basis."""
        dx = p[0] - ref[0]; dy = p[1] - ref[1]; dz = p[2] - ref[2]            # float32, as on the host
        ax = wp.float64(dx); ay = wp.float64(dy); az = wp.float64(dz)
        return wp.vec2d(ax * u[0] + ay * u[1] + az * u[2], ax * w[0] + ay * w[1] + az * w[2])

    @wp.kernel
    def k_fkey(C: int, cfa: wp.array(dtype=int), cfb: wp.array(dtype=int), stride: wp.int64,
               keys: wp.array(dtype=wp.int64), vals: wp.array(dtype=int)):
        i = wp.tid()
        if i >= C:
            return
        keys[i] = wp.int64(cfa[i]) * stride + wp.int64(cfb[i]) + wp.int64(1)
        vals[i] = i

    @wp.kernel
    def k_warm(C: int, keys: wp.array(dtype=wp.int64), wn: int, wkeys: wp.array(dtype=wp.int64),
               wjn: wp.array(dtype=float), wjt1: wp.array(dtype=float), wjt2: wp.array(dtype=float),
               jn: wp.array(dtype=wp.float64), jt1: wp.array(dtype=float), jt2: wp.array(dtype=float)):
        """Warm start by binary search in the kept keys of the previous step (they are sorted, so this is the
        device form of the host `searchsorted`). No host round trip."""
        i = wp.tid()
        if i >= C:
            return
        jn[i] = wp.float64(0.0); jt1[i] = 0.0; jt2[i] = 0.0
        if wn == 0:
            return
        k = keys[i]; lo = int(0); hi = wn - 1; hit = int(-1)
        while lo <= hi:
            mid = (lo + hi) / 2
            km = wkeys[mid]
            if km == k:
                hit = mid
                lo = hi + 1
            else:
                if km < k:
                    lo = mid + 1
                else:
                    hi = mid - 1
        if hit >= 0:
            jn[i] = wp.float64(wjn[hit]); jt1[i] = wjt1[hit]; jt2[i] = wjt2[hit]

    @wp.kernel
    def k_pkey_pair(C: int, cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), N1: wp.int64,
                    keys: wp.array(dtype=wp.int64), vals: wp.array(dtype=int)):
        i = wp.tid()
        if i >= C:
            return
        keys[i] = wp.int64(cbi[i]) * N1 + wp.int64(cbj[i]) + wp.int64(1)
        vals[i] = i

    @wp.kernel
    def k_runflag(C: int, keys: wp.array(dtype=wp.int64), flag: wp.array(dtype=int)):
        """Segment boundaries of the pair-sorted list: 1 where a new body pair starts."""
        i = wp.tid()
        if i >= C:
            return
        if i == 0:
            flag[i] = 1
        else:
            if keys[i] != keys[i - 1]:
                flag[i] = 1
            else:
                flag[i] = 0

    @wp.kernel(module="unique", module_options={"fuse_fp": False})
    def k_select(C: int, mx: int, gorder: wp.array(dtype=int), flag: wp.array(dtype=int),
                 cbi: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3),
                 cpen: wp.array(dtype=float), xc: wp.array(dtype=wp.vec3), jn_in: wp.array(dtype=wp.float64),
                 keep: wp.array(dtype=int), jn_out: wp.array(dtype=wp.float64), flags: wp.array(dtype=int)):
        """One thread per body pair: keep at most `mx` points, E-optimal, deepest point always kept, and move
        the normal impulse of every dropped point onto the kept point nearest to it in the contact plane."""
        i = wp.tid()
        if i >= C or flag[i] == 0:
            return
        L = int(1)
        while i + L < C and flag[i + L] == 0:
            L += 1
        if L <= mx:
            for t in range(L):
                g = gorder[i + t]
                keep[g] = 1; jn_out[g] = jn_in[g]
            return
        if L > _MAXSEG or mx > max_keep:
            wp.atomic_add(flags, 1, 1)               # reported to the host, which raises
            for t in range(L):
                g = gorder[i + t]
                keep[g] = 1; jn_out[g] = jn_in[g]
            return
        deep = int(0); dbest = cpen[gorder[i]]
        for t in range(1, L):
            p = cpen[gorder[i + t]]
            if p > dbest:
                dbest = p; deep = t
        gd = gorder[i + deep]; nr = cn[gd]
        ax = wp.float64(1.0); ay = wp.float64(0.0); az = wp.float64(0.0)
        if wp.abs(nr[0]) >= 0.9:
            ax = wp.float64(0.0); ay = wp.float64(1.0); az = wp.float64(0.0)
        n0 = wp.float64(nr[0]); n1 = wp.float64(nr[1]); n2 = wp.float64(nr[2])
        dn = ax * n0 + ay * n1 + az * n2
        u0 = ax - dn * n0; u1 = ay - dn * n1; u2 = az - dn * n2
        ln = wp.sqrt(u0 * u0 + u1 * u1 + u2 * u2)
        u = wp.vec3d(u0 / ln, u1 / ln, u2 / ln)
        w = wp.vec3d(n1 * u[2] - n2 * u[1], n2 * u[0] - n0 * u[2], n0 * u[1] - n1 * u[0])
        ref = xc[cbi[gd]]
        rd = _planar(cpA[gd], ref, u, w)
        S00 = rd[0] * rd[0]; S01 = rd[0] * rd[1]; S10 = rd[1] * rd[0]; S11 = rd[1] * rd[1]
        pick = veck(); pick[0] = deep
        for k in range(1, mx):
            sbest = wp.float64(-1.0e300); tbest = int(-1)
            c00 = wp.float64(0.0); c01 = wp.float64(0.0); c10 = wp.float64(0.0); c11 = wp.float64(0.0)
            for t in range(L):
                taken = int(0)
                for q in range(k):
                    if pick[q] == t:
                        taken = 1
                if taken == 0:
                    rt = _planar(cpA[gorder[i + t]], ref, u, w)
                    a00 = S00 + rt[0] * rt[0]; a01 = S01 + rt[0] * rt[1]
                    a10 = S10 + rt[1] * rt[0]; a11 = S11 + rt[1] * rt[1]
                    tr = a00 + a11
                    det = a00 * a11 - a01 * a10
                    disc = wp.max(tr * tr / wp.float64(4.0) - det, wp.float64(0.0))
                    lam = tr / wp.float64(2.0) - wp.sqrt(disc)          # sigma_min(G)^2 = min(n, lam_min)
                    sc = wp.min(wp.float64(k + 1), lam)
                    if sc > sbest:
                        sbest = sc; tbest = t
                        c00 = a00; c01 = a01; c10 = a10; c11 = a11
            pick[k] = tbest
            S00 = c00; S01 = c01; S10 = c10; S11 = c11
        for k in range(mx):
            g = gorder[i + pick[k]]
            keep[g] = 1; jn_out[g] = wp.float64(0.0)
        for t in range(L):
            rt = _planar(cpA[gorder[i + t]], ref, u, w)
            dbest2 = wp.float64(1.0e300); kbest = int(0)
            for k in range(mx):
                rk = _planar(cpA[gorder[i + pick[k]]], ref, u, w)
                e0 = rt[0] - rk[0]; e1 = rt[1] - rk[1]; d2 = e0 * e0 + e1 * e1
                if d2 < dbest2:
                    dbest2 = d2; kbest = k
            gk = gorder[i + pick[kbest]]
            jn_out[gk] = jn_out[gk] + jn_in[gorder[i + t]]

    @wp.kernel
    def k_compact(C: int, keep: wp.array(dtype=int), scan: wp.array(dtype=int),
                  keys: wp.array(dtype=wp.int64), cbi: wp.array(dtype=int), cbj: wp.array(dtype=int),
                  cpA: wp.array(dtype=wp.vec3), cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3),
                  cpen: wp.array(dtype=float), jn_in: wp.array(dtype=wp.float64), jt1_in: wp.array(dtype=float),
                  jt2_in: wp.array(dtype=float), okeys: wp.array(dtype=wp.int64), obi: wp.array(dtype=int),
                  obj: wp.array(dtype=int), opA: wp.array(dtype=wp.vec3), opB: wp.array(dtype=wp.vec3),
                  on: wp.array(dtype=wp.vec3), open_: wp.array(dtype=float), ojn: wp.array(dtype=float),
                  ojt1: wp.array(dtype=float), ojt2: wp.array(dtype=float), flags: wp.array(dtype=int)):
        """Stream-compaction of the kept contacts, feature-id order preserved (the scan is over the
        feature-sorted list, so the output index is the rank of the contact among the kept ones)."""
        i = wp.tid()
        if i >= C:
            return
        if i == C - 1:
            flags[0] = scan[i]
        if keep[i] == 0:
            return
        d = scan[i] - 1
        okeys[d] = keys[i]; obi[d] = cbi[i]; obj[d] = cbj[i]; opA[d] = cpA[i]; opB[d] = cpB[i]
        on[d] = cn[i]; open_[d] = cpen[i]; ojn[d] = wp.float32(jn_in[i])
        ojt1[d] = jt1_in[i]; ojt2[d] = jt2_in[i]

    @wp.kernel
    def k_kept_runflag(Ck: int, cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), N1: wp.int64,
                       flag: wp.array(dtype=int)):
        i = wp.tid()
        if i >= Ck:
            return
        p = wp.int64(cbi[i]) * N1 + wp.int64(cbj[i])
        if i == 0:
            flag[i] = 1
        else:
            if p != wp.int64(cbi[i - 1]) * N1 + wp.int64(cbj[i - 1]):
                flag[i] = 1
            else:
                flag[i] = 0

    @wp.kernel
    def k_runstart(Ck: int, flag: wp.array(dtype=int), scan: wp.array(dtype=int),
                   runstart: wp.array(dtype=int)):
        i = wp.tid()
        if i >= Ck or flag[i] == 0:
            return
        runstart[scan[i] - 1] = i

    @wp.kernel
    def k_prio_manifold(Ck: int, scan: wp.array(dtype=int), runstart: wp.array(dtype=int),
                        pkey: wp.array(dtype=wp.int64)):
        """The step-2 colouring priority, built on the device: slot within the body pair, then the parity of
        the pair's rank, then the contact rank."""
        i = wp.tid()
        if i >= Ck:
            return
        g = scan[i] - 1
        slot = wp.min(i - runstart[g], 255)
        pkey[i] = (wp.int64(slot) << wp.int64(48)) | (wp.int64(g % 2) << wp.int64(40)) \
            | wp.min(wp.int64(i), wp.int64(1099511627775))

    return dict(fkey=k_fkey, warm=k_warm, pkey_pair=k_pkey_pair, runflag=k_runflag, select=k_select,
                compact=k_compact, kept_runflag=k_kept_runflag, runstart=k_runstart, prio=k_prio_manifold)


class GraphColoredContactEngine:
    """GPU federation member (graph-coloured Gauss-Seidel, atomics-free solve). Satisfies the ContactEngine
    Protocol. Same bodies, broad phase and contact model as the relaxed-Jacobi GPU engine."""
    name = "graph_colored_gpu"

    def __init__(s, dims=(0.3, 0.3, 0.2), vit=None, pit=20, mu=0.5, warm_start=True, t2_damp=1.0,
                 manifold_reduce=False, manifold_max_points=4, manifold_on_host=False, profile=False,
                 graph_capture=False, graph_mode="device", pair_chunks=False, chunk_mode="registers", **kw):
        # vit=None takes the default of the selected solve path: 20 for the plain coloured solve, 40 with the
        # manifold reduction (measured: the lowest of the sweep at which the reduced K=8 stack keeps M5).
        import warp as wp                                  # lazy: warp + CUDA are only needed at instantiation
        s.wp = wp; wp.init(); s.dev = "cuda:0" if wp.is_cuda_available() else "cpu"
        s.K = _build_kernels(wp)                            # grav / invIw / worldp / integ reused as-is
        s.KC = _build_colored_kernels(wp)
        s.KD = _build_device_manifold_kernels(wp) if (manifold_reduce and not manifold_on_host) else None
        s.dims = dims; s.rest_np = _voxbox(*dims); s.P = len(s.rest_np)
        s.vit = (40 if manifold_reduce else 20) if vit is None else int(vit); s.pit = pit; s.mu = mu; s.warm_start = warm_start; s.t2_damp = t2_damp
        s.manifold_reduce = bool(manifold_reduce); s.manifold_max_points = int(manifold_max_points)
        s.manifold_on_host = bool(manifold_on_host)
        s.profile = bool(profile); s.prof_ms = {}; s._prof_steps = 0
        if graph_mode not in ("device", "spans"):
            raise ValueError("contact_engine_gpu_colored: graph_mode is 'device' or 'spans'")
        s.graph_capture = bool(graph_capture); s.graph_mode = graph_mode
        s.pair_chunks = bool(pair_chunks); s.chunk_mode = chunk_mode
        if s.pair_chunks:
            if chunk_mode not in ("registers", "global"):
                raise ValueError("contact_engine_gpu_colored: chunk_mode is 'registers' or 'global'")
            if not (manifold_reduce and not manifold_on_host):
                raise ValueError("contact_engine_gpu_colored: pair_chunks=True needs the device manifold "
                                 "reduction (manifold_reduce=True, manifold_on_host=False): a chunk is the "
                                 "<= manifold_max_points contacts of one body pair")
        s.n_pairs = 0
        s._graphs = {}; s.n_graph_captures = 0
        s._centers = []; s._dens = []; s._kin = []
        s._built = False
        s.n_colors = 0; s.n_contacts = 0; s.n_contacts_solved = 0
        s._ws_keys = np.zeros(0, np.int64); s._ws_imp = np.zeros((0, 3))

    # ── construction (data in) ──
    def add_body(s, center, density=700., kin=False):
        s._centers.append(list(center)); s._dens.append(float(density)); s._kin.append(1 if kin else 0)
        s._built = False; return len(s._centers) - 1

    def _build(s):
        wp = s.wp; N = len(s._centers); w, h, d = s.dims; vol = w * h * d
        s.N = N; mp = np.array(s._dens) * vol / s.P; M = mp * s.P
        s._M = M; s._kinarr = np.array(s._kin, int)
        invM = np.where(s._kinarr == 1, 0.0, 1.0 / M)
        Ib = sum(mp[0] * ((r @ r) * np.eye(3) - np.outer(r, r)) for r in s.rest_np)
        IbInv = np.array([np.linalg.inv((mp[i] / mp[0]) * Ib) if s._kinarr[i] == 0 else np.eye(3) for i in range(N)])
        if not np.all(np.isfinite(IbInv)):
            raise ValueError("contact_engine_gpu_colored._build: IbInv contains non-finite (NaN/inf) entries -- "
                             "degenerate body mass/inertia; refusing to build a corrupted rigid-body sim")
        s.xc = wp.array(np.array(s._centers, float), dtype=wp.vec3, device=s.dev)
        s.q = wp.array(np.tile([0, 0, 0, 1.], (N, 1)), dtype=wp.quat, device=s.dev)
        s.v = wp.zeros(N, dtype=wp.vec3, device=s.dev); s.w = wp.zeros(N, dtype=wp.vec3, device=s.dev)
        s.invM = wp.array(invM, dtype=float, device=s.dev)
        s.IbInv = wp.array(IbInv, dtype=wp.mat33, device=s.dev); s.invIw = wp.zeros(N, dtype=wp.mat33, device=s.dev)
        s.rest = wp.array(s.rest_np, dtype=wp.vec3, device=s.dev)
        s.allp = wp.zeros(N * s.P, dtype=wp.vec3, device=s.dev); s.owner = wp.zeros(N * s.P, dtype=int, device=s.dev)
        s.grid = wp.HashGrid(64, 64, 64, device=s.dev); s.cnt = wp.zeros(1, dtype=int, device=s.dev)
        z = lambda dt_: wp.zeros(_MAXC, dtype=dt_, device=s.dev)
        s.cbi = z(int); s.cbj = z(int); s.cpA = z(wp.vec3); s.cpB = z(wp.vec3); s.cn = z(wp.vec3); s.cpen = z(float)
        s.sbi = z(int); s.sbj = z(int); s.spA = z(wp.vec3); s.spB = z(wp.vec3); s.sn = z(wp.vec3); s.spen = z(float)
        s.cfa = z(int); s.cfb = z(int)
        s.jn = z(float); s.jt1 = z(float); s.jt2 = z(float); s.jp = z(float)
        s.col = z(int); s.order = z(int); s.pkey = z(wp.int64)
        s.claim = wp.zeros(N, dtype=wp.int64, device=s.dev); s.left = wp.zeros(1, dtype=int, device=s.dev)
        s.pv = wp.zeros(N, dtype=wp.vec3, device=s.dev); s.po = wp.zeros(N, dtype=wp.vec3, device=s.dev)
        s._kin_wp = wp.array(s._kinarr, dtype=int, device=s.dev)
        if s.KD is not None:                       # device manifold stage: sort buffers, flags, warm-start store
            s.tbi = z(int); s.tbj = z(int); s.tpA = z(wp.vec3); s.tpB = z(wp.vec3); s.tn = z(wp.vec3)
            s.tpen = z(float)
            s.skey = wp.zeros(2 * _MAXC, dtype=wp.int64, device=s.dev)      # radix sort needs 2*count
            s.sval = wp.zeros(2 * _MAXC, dtype=int, device=s.dev)
            s.pkey_s = wp.zeros(2 * _MAXC, dtype=wp.int64, device=s.dev)
            s.pval_s = wp.zeros(2 * _MAXC, dtype=int, device=s.dev)
            s.segflag = z(int); s.segscan = z(int); s.keep = z(int); s.runstart = z(int)
            s.jn_in = z(wp.float64); s.jt1_in = z(float); s.jt2_in = z(float); s.jn_red = z(wp.float64)
            s.ckeys = wp.zeros(_MAXC, dtype=wp.int64, device=s.dev)
            s.ws_keys = wp.zeros(_MAXC, dtype=wp.int64, device=s.dev)
            s.ws_jn = z(float); s.ws_jt1 = z(float); s.ws_jt2 = z(float); s._ws_n = 0
            s.dflags = wp.zeros(2, dtype=int, device=s.dev)                 # [kept count, oversize segments]
        if s.graph_capture or s.pair_chunks:       # colour counting sort + per-colour offsets on the device
            s.ckey_s = wp.zeros(2 * _MAXC, dtype=wp.int64, device=s.dev)
            s.cval_s = wp.zeros(2 * _MAXC, dtype=int, device=s.dev)
            s.ccount = wp.zeros(_NCOLMAX, dtype=int, device=s.dev)
            s.cstart = wp.zeros(_NCOLMAX, dtype=int, device=s.dev)
        if s.pair_chunks:                          # body-pair segments and the colouring of the pair graph
            s.pkeyP = wp.zeros(2 * _MAXC, dtype=wp.int64, device=s.dev)
            s.pvalP = wp.zeros(2 * _MAXC, dtype=int, device=s.dev)
            s.pflag = z(int); s.pscan = z(int)
            s.pstart = z(int); s.pcount = z(int); s.pbi = z(int); s.pbj = z(int)
            s.colP = z(int); s.prioP = z(wp.int64); s.porder = z(int)
            s.pflags = wp.zeros(2, dtype=int, device=s.dev)
        s._graphs = {}
        s._cforce = np.zeros((N, 3)); s._built = True

    def set_kinematic(s, i, xc):
        s._kin[i] = 1; s._centers[i] = list(xc)
        if s._built:
            wp = s.wp
            a = s.xc.numpy(); a[i] = xc; s.xc = wp.array(a, dtype=wp.vec3, device=s.dev)
            m = s.invM.numpy(); m[i] = 0.0; s.invM = wp.array(m, dtype=float, device=s.dev)
            vv = s.v.numpy(); vv[i] = [0.0, 0.0, 0.0]; s.v = wp.array(vv, dtype=wp.vec3, device=s.dev)
            ww = s.w.numpy(); ww[i] = [0.0, 0.0, 0.0]; s.w = wp.array(ww, dtype=wp.vec3, device=s.dev)
            s._kinarr[i] = 1; s._M[i] = np.inf
            s._kin_wp = wp.array(s._kinarr, dtype=int, device=s.dev)

    # ── contact ordering, warm start, colouring ──
    def _sort_by_feature(s, C):
        """Sort the generated contacts by feature id (point index on A, point index on B or -1) so that the
        atomic-counter order of `k_gen_fid` cannot reach the solve. Returns the feature keys, sorted."""
        wp = s.wp
        fa = s.cfa.numpy()[:C].astype(np.int64); fb = s.cfb.numpy()[:C].astype(np.int64)
        keys = fa * np.int64(s.N * s.P + 1) + (fb + 1)
        perm = np.argsort(keys, kind="stable").astype(np.int32)
        wp.copy(s.order, wp.array(perm, dtype=int, device=s.dev), count=C)
        wp.launch(s.KC["gather"], C, inputs=[s.order, C, s.cbi, s.cbj, s.cpA, s.cpB, s.cn, s.cpen,
                                             s.sbi, s.sbj, s.spA, s.spB, s.sn, s.spen], device=s.dev)
        if s.manifold_reduce:          # the reduction stage groups and selects on the host
            s._h_bi = s.cbi[:C].numpy()[perm]; s._h_bj = s.cbj[:C].numpy()[perm]
            s._h_pA = s.cpA[:C].numpy()[perm]; s._h_pB = s.cpB[:C].numpy()[perm]
            s._h_n = s.cn[:C].numpy()[perm]; s._h_pen = s.cpen[:C].numpy()[perm]
        return keys[perm]

    def _warm_start(s, C, keys):
        """Impulses of a feature id that was in contact in the previous step are the starting impulses here.
        Returns the three impulse arrays on the host; the caller uploads them (the manifold stage, when it is
        on, redistributes the normal impulses over the kept points first)."""
        jn = np.zeros(C); jt1 = np.zeros(C); jt2 = np.zeros(C)
        if s.warm_start and len(s._ws_keys):
            pos = np.searchsorted(s._ws_keys, keys)
            pos = np.clip(pos, 0, len(s._ws_keys) - 1)
            hit = s._ws_keys[pos] == keys
            jn[hit] = s._ws_imp[pos[hit], 0]; jt1[hit] = s._ws_imp[pos[hit], 1]; jt2[hit] = s._ws_imp[pos[hit], 2]
        return jn, jt1, jt2

    def _upload_impulses(s, C, jn, jt1, jt2):
        wp = s.wp
        wp.copy(s.jn, wp.array(jn, dtype=float, device=s.dev), count=C)
        wp.copy(s.jt1, wp.array(jt1, dtype=float, device=s.dev), count=C)
        wp.copy(s.jt2, wp.array(jt2, dtype=float, device=s.dev), count=C)
        s.jp.zero_()

    # -- manifold reduction (step 2) --
    @staticmethod
    def _plane_basis(nrm):
        """Deterministic tangent basis of a set of unit normals; the same rule the velocity sweep uses."""
        a = np.where(np.abs(nrm[:, :1]) >= 0.9, np.array([[0.0, 1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]))
        t1 = a - np.sum(a * nrm, axis=1, keepdims=True) * nrm
        t1 = t1 / np.linalg.norm(t1, axis=1, keepdims=True)
        return t1, np.cross(nrm, t1)

    def _reduce_manifold(s, C, keys, jn, jt1, jt2):
        """Group the sorted contacts by body pair and keep at most `manifold_max_points` per pair, chosen by
        the E-optimal rule (max sigma_min of the wrench map about the projected centre of mass of body A,
        deepest point always kept). The normal impulse of every dropped contact is added to the kept point
        nearest to it in the contact plane, so the pair's total normal impulse is preserved.

        Rewrites the device contact arrays in place with the kept contacts, in the same feature-id order, and
        returns (C_kept, keys_kept, jn, jt1, jt2)."""
        wp = s.wp; mx = s.manifold_max_points
        bi = s._h_bi; bj = s._h_bj; pA = s._h_pA; pB = s._h_pB; nn = s._h_n; pen = s._h_pen
        xcn = s.xc.numpy()
        pair = bi.astype(np.int64) * np.int64(s.N + 1) + (bj.astype(np.int64) + 1)
        gorder = np.argsort(pair, kind="stable")
        _, starts, counts = np.unique(pair[gorder], return_index=True, return_counts=True)
        big = counts > mx
        keep = [gorder[starts[i]:starts[i] + counts[i]] for i in np.flatnonzero(~big)]
        if not np.any(big):
            keep_idx = np.sort(np.concatenate(keep)) if keep else np.zeros(0, np.int64)
            return s._apply_keep(C, keys, jn, jt1, jt2, keep_idx, None, None)
        bst = starts[big]; bcn = counts[big]; P = len(bst); gmax = int(bcn.max())
        cols = np.arange(gmax)
        valid = cols[None, :] < bcn[:, None]
        idx = gorder[bst[:, None] + np.minimum(cols[None, :], (bcn - 1)[:, None])]          # (P, gmax)
        peng = np.where(valid, pen[idx], -np.inf)
        deep = np.argmax(peng, axis=1)
        rows = np.arange(P)
        nref = nn[idx[rows, deep]]
        t1, t2 = s._plane_basis(nref)
        ref = xcn[bi[idx[rows, deep]]]                                       # centre of mass of body A
        d = pA[idx] - ref[:, None, :]
        r = np.stack([np.sum(d * t1[:, None, :], axis=2), np.sum(d * t2[:, None, :], axis=2)], axis=2)
        outer = r[:, :, :, None] * r[:, :, None, :]                          # (P, gmax, 2, 2)
        sel = np.zeros((P, gmax), bool); sel[rows, deep] = True
        picks = [deep]
        S = outer[rows, deep]
        for k in range(1, mx):
            cand = S[:, None, :, :] + outer
            tr = cand[:, :, 0, 0] + cand[:, :, 1, 1]
            det = cand[:, :, 0, 0] * cand[:, :, 1, 1] - cand[:, :, 0, 1] * cand[:, :, 1, 0]
            disc = np.maximum(tr * tr / 4.0 - det, 0.0)
            lam_min = tr / 2.0 - np.sqrt(disc)                               # sigma_min(G)^2 = min(n, lam_min)
            score = np.where(valid & ~sel, np.minimum(float(k + 1), lam_min), -np.inf)
            pk = np.argmax(score, axis=1)
            sel[rows, pk] = True; picks.append(pk); S = cand[rows, pk]
        pick = np.stack(picks, axis=1)                                       # (P, mx)
        rk = r[rows[:, None], pick]                                          # kept planar coordinates
        dist2 = np.sum((r[:, :, None, :] - rk[:, None, :, :]) ** 2, axis=3)
        near = np.argmin(dist2, axis=2)                                      # (P, gmax) kept slot per contact
        flat = (rows[:, None] * mx + near).ravel()
        w = np.where(valid, jn[idx], 0.0).ravel()
        jn_kept = np.bincount(flat, weights=w, minlength=P * mx).reshape(P, mx)
        keep.append(idx[rows[:, None], pick].ravel())
        keep_idx = np.sort(np.concatenate(keep))
        return s._apply_keep(C, keys, jn, jt1, jt2, keep_idx,
                             idx[rows[:, None], pick].ravel(), jn_kept.ravel())

    def _apply_keep(s, C, keys, jn, jt1, jt2, keep_idx, red_idx, red_jn):
        """Upload the kept contacts (feature-id order preserved) to the solver arrays, and build the
        colouring priority of the reduced set.

        Priority. After the reduction the contacts of one pair are at most 4 and they all conflict with each
        other, so a body carrying two pairs needs at least 2*4 colours: the colour count is now set by the
        pair graph (pairs sharing a dynamic body), not by the length of the contact list. The key is ranked
        by the SLOT within the pair first, then by the parity of the pair's rank in body-pair order, then by
        the contact rank (which keeps it unique). Every Jones-Plassmann round then takes one slot over an
        independent set of pairs, and a chain of pairs -- a stack, where consecutive pair ranks are the
        neighbours -- is taken in two rounds per slot instead of the three a rank hash needs, so the round
        count comes out at (points per pair) x (colours of the pair graph). The key is a pure function of the
        sorted contact list, so the colouring stays independent of the generation order."""
        wp = s.wp
        if red_idx is not None:
            jn = jn.copy(); jn[red_idx] = red_jn
        k = keep_idx
        pair = s._h_bi[k].astype(np.int64) * np.int64(s.N + 1) + (s._h_bj[k].astype(np.int64) + 1)
        rank = np.arange(len(k), dtype=np.int64)
        first = np.concatenate([[True], pair[1:] != pair[:-1]]) if len(k) else np.zeros(0, bool)
        gid = np.cumsum(first) - 1
        slot = rank - np.flatnonzero(first)[gid] if len(k) else rank
        pr = gid.astype(np.int64)                       # rank of the pair in body-pair order
        pkey = ((np.minimum(slot, 255).astype(np.int64) << np.int64(48))
                | ((pr % np.int64(2)) << np.int64(40)) | np.minimum(rank, np.int64(1099511627775)))
        s._pkey_host = pkey
        wp.copy(s.sbi, wp.array(s._h_bi[k], dtype=int, device=s.dev), count=len(k))
        wp.copy(s.sbj, wp.array(s._h_bj[k], dtype=int, device=s.dev), count=len(k))
        wp.copy(s.spA, wp.array(s._h_pA[k], dtype=wp.vec3, device=s.dev), count=len(k))
        wp.copy(s.spB, wp.array(s._h_pB[k], dtype=wp.vec3, device=s.dev), count=len(k))
        wp.copy(s.sn, wp.array(s._h_n[k], dtype=wp.vec3, device=s.dev), count=len(k))
        wp.copy(s.spen, wp.array(s._h_pen[k], dtype=float, device=s.dev), count=len(k))
        return len(k), keys[k], jn[k], jt1[k], jt2[k]


    # -- manifold reduction on the device (step 3a) --
    @contextmanager
    def _stage(s, name):
        """Per-stage wall time, device-synchronised, accumulated over the steps of the run."""
        if not s.profile:
            yield
            return
        s.wp.synchronize_device(s.dev); t0 = time.perf_counter()
        yield
        s.wp.synchronize_device(s.dev)
        s.prof_ms[name] = s.prof_ms.get(name, 0.0) + (time.perf_counter() - t0) * 1e3

    def _device_manifold(s, C):
        """The grouping, the E-optimal selection and the warm start on the GPU: feature-key radix sort,
        pair-key radix sort, run-length segment flags, one thread per body pair for the selection, a scan
        and a compaction. Returns the number of kept contacts; the only host readback is the two-integer
        flag array (kept count, oversize-segment counter). Same selection as the host stage."""
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
            wp.launch(KD["select"], C, inputs=[C, mx, s.pval_s, s.segflag, s.tbi, s.tpA, s.tn, s.tpen,
                                               s.xc, s.jn_in, s.keep, s.jn_red, s.dflags], device=s.dev)
            wp.utils.array_scan(s.keep[:C], s.segscan[:C], True)
            wp.launch(KD["compact"], C, inputs=[C, s.keep, s.segscan, s.skey, s.tbi, s.tbj, s.tpA, s.tpB,
                                                s.tn, s.tpen, s.jn_red, s.jt1_in, s.jt2_in, s.ckeys,
                                                s.sbi, s.sbj, s.spA, s.spB, s.sn, s.spen,
                                                s.jn, s.jt1, s.jt2, s.dflags], device=s.dev)
            fl = s.dflags.numpy(); Ck = int(fl[0])
            if int(fl[1]) > 0:
                raise RuntimeError(f"contact_engine_gpu_colored._device_manifold: {int(fl[1])} body pairs carry "
                                   f"more than {_MAXSEG} contacts or more than the kept-point capacity; the "
                                   f"device selection kernel is fixed size -- use manifold_on_host=True")
            s.jp.zero_()
            wp.launch(KD["kept_runflag"], Ck, inputs=[Ck, s.sbi, s.sbj, s.N + 1, s.segflag], device=s.dev)
            wp.utils.array_scan(s.segflag[:Ck], s.segscan[:Ck], True)
            wp.launch(KD["runstart"], Ck, inputs=[Ck, s.segflag, s.segscan, s.runstart], device=s.dev)
            wp.launch(KD["prio"], Ck, inputs=[Ck, s.segscan, s.runstart, s.pkey], device=s.dev)
        return Ck

    def _store_warm_device(s, C):
        wp = s.wp
        wp.copy(s.ws_keys, s.ckeys, count=C); wp.copy(s.ws_jn, s.jn, count=C)
        wp.copy(s.ws_jt1, s.jt1, count=C); wp.copy(s.ws_jt2, s.jt2, count=C)
        s._ws_n = C

    def selection_debug(s):
        """The kept contacts of the last step as (feature key, body i, body j), for the device-vs-host
        identity check of the test suite. Host path and device path answer in the same layout."""
        C = s.n_contacts_solved
        if s.manifold_on_host:
            return np.stack([s._ws_keys[:C] if len(s._ws_keys) else np.zeros(0, np.int64),
                             s.sbi.numpy()[:C].astype(np.int64), s.sbj.numpy()[:C].astype(np.int64)], axis=1)
        return np.stack([s.ckeys.numpy()[:C], s.sbi.numpy()[:C].astype(np.int64),
                         s.sbj.numpy()[:C].astype(np.int64)], axis=1)

    def _color(s, C):
        """Parallel colouring on the GPU: one colour per round, claim by atomic minimum over the contact
        rank. Returns (order array grouped by colour, list of (start, count) per colour)."""
        wp = s.wp
        col = np.full(C, -1, np.int32)
        wp.copy(s.col, wp.array(col, dtype=int, device=s.dev), count=C)
        if s.manifold_reduce and s.manifold_on_host:
            wp.copy(s.pkey, wp.array(s._pkey_host, dtype=wp.int64, device=s.dev), count=C)
        elif s.manifold_reduce:
            pass                       # the device stage has already written s.pkey for the kept contacts
        else:
            wp.launch(s.KC["prio"], C, inputs=[C, s.pkey], device=s.dev)
        cur = 0
        while True:
            wp.launch(s.KC["clear_claim"], s.N, inputs=[s.claim], device=s.dev)
            s.left.zero_()
            wp.launch(s.KC["claim"], C, inputs=[C, s.col, s.sbi, s.sbj, s.invM, s.pkey, s.claim], device=s.dev)
            wp.launch(s.KC["win"], C, inputs=[C, cur, s.col, s.sbi, s.sbj, s.invM, s.pkey, s.claim, s.left],
                      device=s.dev)
            cur += 1
            if int(s.left.numpy()[0]) == 0:
                break
            if cur > C:                       # cannot happen: every round colours at least one contact
                raise RuntimeError("contact_engine_gpu_colored._color: colouring did not terminate")
        if s.graph_capture and s.graph_mode == "device":
            return cur, s._color_spans_device(C, cur)
        col = s.col.numpy()[:C]
        order = np.argsort(col, kind="stable").astype(np.int32)
        counts = np.bincount(col, minlength=cur)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        wp.copy(s.order, wp.array(order, dtype=int, device=s.dev), count=C)
        return cur, [(int(a), int(b)) for a, b in zip(starts, counts) if b > 0]

    def _color_spans_device(s, C, ncol):
        """Group the coloured contacts by colour on the device: a stable counting sort on the colour id
        (`wp.utils.radix_sort_pairs`, whose passes are counting sorts, so the permutation is the one a stable
        argsort by colour gives), a histogram of the colour ids and an exclusive scan of it for the offset of
        each colour. The colour array never crosses to the host and nothing is uploaded; the sweep kernels
        read the offset and the count of their colour from `cstart` / `ccount`, so a captured graph stays
        valid when the partition changes. Returns None in place of the host span list."""
        wp = s.wp
        if ncol > _NCOLMAX:
            raise RuntimeError(f"contact_engine_gpu_colored._color_spans_device: {ncol} colours exceeds the "
                               f"per-colour array capacity {_NCOLMAX}")
        wp.launch(s.KC["colkey"], C, inputs=[C, s.col, s.ckey_s, s.cval_s], device=s.dev)
        wp.utils.radix_sort_pairs(s.ckey_s, s.cval_s, C)
        wp.copy(s.order, s.cval_s, count=C)
        s.ccount.zero_()
        wp.launch(s.KC["colcount"], C, inputs=[C, s.col, s.ccount], device=s.dev)
        wp.utils.array_scan(s.ccount[:ncol], s.cstart[:ncol], False)          # exclusive scan: colour offsets
        return None

    # -- pair chunks (step 3c) --
    def _build_pairs(s, C):
        """Group the kept contacts by body pair on the device: a stable radix sort on the pair key, a
        run-length flag, a scan and one thread per segment start, which writes the segment offset, its
        length and the two bodies of the pair. Returns the number of pairs. The contact index list
        `pvalP` stays the sort permutation, so a chunk is a contiguous run of it."""
        wp = s.wp; KC = s.KC
        wp.launch(KC["pairkey"], C, inputs=[C, s.sbi, s.sbj, s.N + 1, s.pkeyP, s.pvalP], device=s.dev)
        wp.utils.radix_sort_pairs(s.pkeyP, s.pvalP, C)
        wp.launch(KC["pairflag"], C, inputs=[C, s.pkeyP, s.pflag], device=s.dev)
        wp.utils.array_scan(s.pflag[:C], s.pscan[:C], True)
        s.pflags.zero_()
        wp.launch(KC["pairbuild"], C, inputs=[C, s.pflag, s.pscan, s.pvalP, s.sbi, s.sbj, s.pstart,
                                              s.pcount, s.pbi, s.pbj, s.pflags], device=s.dev)
        fl = s.pflags.numpy(); P = int(fl[0])
        if int(fl[1]) > 0:
            raise RuntimeError(f"contact_engine_gpu_colored._build_pairs: {int(fl[1])} body pairs carry more "
                               f"than {_MAXCHUNK} contacts; the chunk solve is fixed size -- lower "
                               f"manifold_max_points or raise _MAXCHUNK")
        return P

    def _color_pairs(s, P):
        """Colour the PAIR graph (a node per body pair, an edge where two pairs share a dynamic body) with
        the same Jones-Plassmann rounds the contact colouring uses, then group the pairs by colour with the
        device counting sort. A pair's contacts are all in one chunk, so they no longer have to be in
        different colours: the colour count is the chromatic number of the pair graph, not of the contact
        graph. Returns the number of colours."""
        wp = s.wp; KC = s.KC
        wp.copy(s.colP, wp.array(np.full(P, -1, np.int32), dtype=int, device=s.dev), count=P)
        wp.launch(KC["prio"], P, inputs=[P, s.prioP], device=s.dev)
        cur = 0
        while True:
            wp.launch(KC["clear_claim"], s.N, inputs=[s.claim], device=s.dev)
            s.left.zero_()
            wp.launch(KC["claim"], P, inputs=[P, s.colP, s.pbi, s.pbj, s.invM, s.prioP, s.claim], device=s.dev)
            wp.launch(KC["win"], P, inputs=[P, cur, s.colP, s.pbi, s.pbj, s.invM, s.prioP, s.claim, s.left],
                      device=s.dev)
            cur += 1
            if int(s.left.numpy()[0]) == 0:
                break
            if cur > P:
                raise RuntimeError("contact_engine_gpu_colored._color_pairs: colouring did not terminate")
        if cur > _NCOLMAX:
            raise RuntimeError(f"contact_engine_gpu_colored._color_pairs: {cur} colours exceeds the "
                               f"per-colour array capacity {_NCOLMAX}")
        wp.launch(KC["colkey"], P, inputs=[P, s.colP, s.ckey_s, s.cval_s], device=s.dev)
        wp.utils.radix_sort_pairs(s.ckey_s, s.cval_s, P)
        wp.copy(s.porder, s.cval_s, count=P)
        s.ccount.zero_()
        wp.launch(KC["colcount"], P, inputs=[P, s.colP, s.ccount], device=s.dev)
        wp.utils.array_scan(s.ccount[:cur], s.cstart[:cur], False)
        return cur

    def _chunk_launch(s, ncol, dim, sdt):
        """The `vit` velocity and `pit` position chunk sweeps over the colours, one launch per colour per
        iteration, offsets and counts read from the device arrays. Used directly and inside the capture."""
        wp = s.wp; KC = s.KC
        kv = KC["chunk_vel_r"] if s.chunk_mode == "registers" else KC["chunk_vel_g"]
        kp = KC["chunk_pos_r"] if s.chunk_mode == "registers" else KC["chunk_pos_g"]
        for _ in range(s.vit):
            for ci in range(ncol):
                wp.launch(kv, dim, inputs=[s.porder, s.pvalP, s.pstart, s.pcount, s.pbi, s.pbj, ci,
                                           s.cstart, s.ccount, s.sbi, s.sbj, s.spA, s.spB, s.sn, s.jn,
                                           s.jt1, s.jt2, s.xc, s.v, s.w, s.invIw, s.invM, s.mu, s.t2_damp],
                          device=s.dev)
        for _ in range(s.pit):
            for ci in range(ncol):
                wp.launch(kp, dim, inputs=[s.porder, s.pvalP, s.pstart, s.pcount, s.pbi, s.pbj, ci,
                                           s.cstart, s.ccount, s.sbi, s.sbj, s.spA, s.spB, s.sn, s.spen,
                                           s.jp, s.xc, s.pv, s.po, s.invIw, s.invM, sdt], device=s.dev)

    def _solve_chunk_graph(s, P, ncol, sdt):
        """The chunk solve loop captured as one CUDA graph, exactly as the per-contact loop of step 3b: no
        host value that changes between steps is baked in (the per-colour offsets and counts are read from
        the device), so the graph is re-captured only when the colour count or the pair-count bucket
        changes."""
        wp = s.wp
        dim = 1
        while dim < max(P, 1):
            dim *= 2
        key = ("chunk", s.chunk_mode, ncol, dim, s.vit, s.pit, float(sdt))
        g = s._graphs.get(key)
        if g is not None:
            return g
        if len(s._graphs) >= 8:
            s._graphs.clear()
        wp.load_module(device=s.dev)
        with wp.ScopedCapture(device=s.dev) as cap:
            s._chunk_launch(ncol, dim, sdt)
        s._graphs[key] = cap.graph; s.n_graph_captures += 1
        return cap.graph

    def _solve_graph(s, C, ncol, spans, sdt):
        """The whole per-step solve loop (`vit` velocity sweeps and `pit` position sweeps over every colour)
        captured once as a CUDA graph and replayed with `wp.capture_launch`, which removes the per-launch
        host cost of the 480 kernel launches.

        graph_mode='device' (the default): the launches carry no host value that changes between steps --
        each colour's offset and count are read from `cstart` / `ccount` on the device and the launch
        dimension is the contact count rounded up to a power of two, an upper bound for any colour. The graph
        is therefore re-captured only when the colour COUNT or that bucket changes, not when the partition
        does. graph_mode='spans': the offsets and counts are baked into the launches, so the graph is
        re-captured whenever the span list changes (measured for comparison; `n_graph_captures` counts it)."""
        wp = s.wp; KC = s.KC
        if spans is None:
            dim = 1
            while dim < max(C, 1):
                dim *= 2
            key = ("device", ncol, dim, s.vit, s.pit, float(sdt))
        else:
            key = ("spans", tuple(spans), s.vit, s.pit, float(sdt))
        g = s._graphs.get(key)
        if g is not None:
            return g
        if len(s._graphs) >= 8:                      # bounded cache; a graph holds its 480 nodes
            s._graphs.clear()
        wp.load_module(device=s.dev)
        with wp.ScopedCapture(device=s.dev) as cap:
            if spans is None:
                for _ in range(s.vit):
                    for ci in range(ncol):
                        wp.launch(KC["gs_vel_d"], dim, inputs=[s.order, ci, s.cstart, s.ccount, s.sbi, s.sbj,
                                                               s.spA, s.spB, s.sn, s.jn, s.jt1, s.jt2, s.xc,
                                                               s.v, s.w, s.invIw, s.invM, s.mu, s.t2_damp],
                                  device=s.dev)
                for _ in range(s.pit):
                    for ci in range(ncol):
                        wp.launch(KC["gs_pos_d"], dim, inputs=[s.order, ci, s.cstart, s.ccount, s.sbi, s.sbj,
                                                               s.spA, s.spB, s.sn, s.spen, s.jp, s.xc, s.pv,
                                                               s.po, s.invIw, s.invM, sdt], device=s.dev)
            else:
                for _ in range(s.vit):
                    for st, cn_ in spans:
                        wp.launch(KC["gs_vel"], cn_, inputs=[s.order, st, cn_, s.sbi, s.sbj, s.spA, s.spB,
                                                             s.sn, s.jn, s.jt1, s.jt2, s.xc, s.v, s.w,
                                                             s.invIw, s.invM, s.mu, s.t2_damp], device=s.dev)
                for _ in range(s.pit):
                    for st, cn_ in spans:
                        wp.launch(KC["gs_pos"], cn_, inputs=[s.order, st, cn_, s.sbi, s.sbj, s.spA, s.spB,
                                                             s.sn, s.spen, s.jp, s.xc, s.pv, s.po,
                                                             s.invIw, s.invM, sdt], device=s.dev)
        s._graphs[key] = cap.graph; s.n_graph_captures += 1
        return cap.graph

    # ── advance ──
    def step(s, dt, substeps=1):
        if not s._built:
            s._build()
        wp = s.wp; K = s.K; KC = s.KC
        dev_path = s.manifold_reduce and not s.manifold_on_host
        for _ in range(substeps):
            sdt = dt / substeps
            with s._stage("generate"):
                wp.launch(K["grav"], s.N, inputs=[s.v, s.invM, sdt], device=s.dev)
                wp.launch(K["invIw"], s.N, inputs=[s.q, s.IbInv, s._kin_wp, s.invIw], device=s.dev)
                wp.launch(K["worldp"], s.N * s.P, inputs=[s.xc, s.q, s.rest, s.P, s.allp, s.owner], device=s.dev)
                s.grid.build(s.allp, 2.0 * _R); s.cnt.zero_()
                wp.launch(KC["gen_fid"], s.N * s.P, inputs=[s.allp, s.owner, s.grid.id, s.cnt, s.cbi, s.cbj,
                                                            s.cpA, s.cpB, s.cn, s.cpen, s.cfa, s.cfb], device=s.dev)
                C = min(int(s.cnt.numpy()[0]), _MAXC)
            s.n_contacts = C
            v_grav = s.v.numpy().copy()
            if C > 0:
                if dev_path:
                    C = s._device_manifold(C)
                    s.n_contacts_solved = C
                    s.pv.zero_(); s.po.zero_()
                else:
                    with s._stage("sort"):
                        keys = s._sort_by_feature(C)
                    with s._stage("select"):
                        jn0, jt10, jt20 = s._warm_start(C, keys)
                        if s.manifold_reduce:
                            C, keys, jn0, jt10, jt20 = s._reduce_manifold(C, keys, jn0, jt10, jt20)
                        s.n_contacts_solved = C
                        s._upload_impulses(C, jn0, jt10, jt20)
                        s.pv.zero_(); s.po.zero_()
                if s.pair_chunks:
                    with s._stage("pairs"):
                        P = s._build_pairs(C)
                    s.n_pairs = P
                    with s._stage("colour"):
                        ncol = s._color_pairs(P)
                    s.n_colors = ncol
                    with s._stage("solve"):
                        if s.graph_capture:
                            wp.capture_launch(s._solve_chunk_graph(P, ncol, sdt))
                        else:
                            dim = 1
                            while dim < max(P, 1):
                                dim *= 2
                            s._chunk_launch(ncol, dim, sdt)
                else:
                    with s._stage("colour"):
                        ncol, spans = s._color(C)
                    s.n_colors = ncol if spans is None else len(spans)
                    with s._stage("solve"):
                        if s.graph_capture:
                            wp.capture_launch(s._solve_graph(C, ncol, spans, sdt))
                        else:
                            s._solve_uncaptured(C, spans, sdt)
                if s.warm_start:
                    with s._stage("warm_store"):
                        if dev_path:
                            s._store_warm_device(C)
                        else:
                            s._ws_keys = keys
                            s._ws_imp = np.stack([s.jn.numpy()[:C], s.jt1.numpy()[:C], s.jt2.numpy()[:C]], axis=1)
            else:
                s.n_colors = 0; s.n_contacts_solved = 0; s.n_pairs = 0
                s.pv.zero_(); s.po.zero_()
                s._ws_keys = np.zeros(0, np.int64); s._ws_imp = np.zeros((0, 3)); s._ws_n = 0
            with s._stage("integrate"):
                wp.launch(K["integ"], s.N, inputs=[s.xc, s.q, s.v, s.w, s.pv, s.po, s.invM, sdt], device=s.dev)
                v_post = s.v.numpy()
                fin = np.zeros((s.N, 3)); nk = s._kinarr == 0
                fin[nk] = s._M[nk, None] * (v_post[nk] - v_grav[nk]) / sdt
                s._cforce = fin
            s._prof_steps += 1

    # ── data out ──
    def _solve_uncaptured(s, C, spans, sdt):
        """The solve loop as separate launches: `vit` velocity sweeps and `pit` position sweeps over the
        colours, one launch per colour per iteration. This is the path the captured graph must reproduce bit
        for bit."""
        wp = s.wp; KC = s.KC
        for _ in range(s.vit):
            for st, cn_ in spans:
                wp.launch(KC["gs_vel"], cn_, inputs=[s.order, st, cn_, s.sbi, s.sbj, s.spA, s.spB,
                                                     s.sn, s.jn, s.jt1, s.jt2, s.xc, s.v, s.w,
                                                     s.invIw, s.invM, s.mu, s.t2_damp], device=s.dev)
        for _ in range(s.pit):
            for st, cn_ in spans:
                wp.launch(KC["gs_pos"], cn_, inputs=[s.order, st, cn_, s.sbi, s.sbj, s.spA, s.spB,
                                                     s.sn, s.spen, s.jp, s.xc, s.pv, s.po,
                                                     s.invIw, s.invM, sdt], device=s.dev)

    def _quat2R(s, q):
        x, y, z, w = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

    def get_state(s):
        if not s._built:
            s._build()
        xc = s.xc.numpy(); q = s.q.numpy(); Rm = np.array([s._quat2R(q[i]) for i in range(s.N)])
        return BodyState(xc=xc, Rm=Rm, vc=s.v.numpy(), om=s.w.numpy())

    def contact_forces(s):
        return s._cforce.copy()

    def com(s, i):
        return s.xc.numpy()[i]


# ───────────────────────── contract selftest ─────────────────────────
def _selftest():
    e = GraphColoredContactEngine()
    assert isinstance(e, ContactEngine), "does not satisfy the ContactEngine Protocol"
    ok = True
    for k in range(8):
        e.add_body([0, 0, 0.101 + k * 0.205])
    for _ in range(400):
        e.step(1 / 240)
    st = e.get_state(); ke = float(np.sum(0.5 * e._M[:, None] * st.vc ** 2))
    zs = sorted(st.xc[:, 2]); seps = [zs[i + 1] - zs[i] for i in range(7)]
    s_ok = ke < 0.5 and all(0.15 < sp < 0.25 for sp in seps)
    print(f"  [stack K=8] KE={ke:.3f} sep {min(seps):.3f}-{max(seps):.3f} colours={e.n_colors} "
          f"contacts={e.n_contacts}  {'stable' if s_ok else 'FAIL'}"); ok = ok and s_ok
    e2 = GraphColoredContactEngine(); e2.add_body([0, 0, 0.101])
    for _ in range(200):
        e2.step(1 / 240)
    Fz = e2.contact_forces()[0, 2]; Mg = e2._M[0] * _G; err = abs(Fz - Mg) / Mg * 100
    f_ok = err < 8
    print(f"  [contact force, box at rest] Fz={Fz:.2f} Mg={Mg:.2f} error={err:.1f}%  {'ok' if f_ok else 'FAIL'}")
    ok = ok and f_ok
    e3 = GraphColoredContactEngine(); e3.add_body([0, 0, 0.5]); e3.set_kinematic(0, [0, 0, 0.5])
    for _ in range(120):
        e3.step(1 / 240)
    z_stay = e3.get_state().xc[0, 2]
    e3.set_kinematic(0, [0.1, 0, 0.5]); e3.step(1 / 240)
    x_moved = e3.get_state().xc[0, 0]
    k_ok = abs(z_stay - 0.5) < 1e-6 and abs(x_moved - 0.1) < 1e-6
    print(f"  [set_kinematic] stays z={z_stay:.4f} + live move x={x_moved:.4f}  {'ok' if k_ok else 'FAIL'}")
    ok = ok and k_ok

    em = GraphColoredContactEngine(manifold_reduce=True)
    for k in range(8):
        em.add_body([0, 0, 0.101 + k * 0.205])
    for _ in range(400):
        em.step(1 / 240)
    stm = em.get_state(); kem = float(np.sum(0.5 * em._M[:, None] * stm.vc ** 2))
    zm = sorted(stm.xc[:, 2]); sepm = [zm[i + 1] - zm[i] for i in range(7)]
    m_ok = kem < 0.5 and all(0.15 < sp < 0.25 for sp in sepm) and em.n_colors <= 8
    print(f"  [stack K=8, manifold_reduce] KE={kem:.3f} sep {min(sepm):.3f}-{max(sepm):.3f} "
          f"colours={em.n_colors} contacts={em.n_contacts}->{em.n_contacts_solved} solved  "
          f"{'stable' if m_ok else 'FAIL'}"); ok = ok and m_ok
    em2 = GraphColoredContactEngine(manifold_reduce=True); em2.add_body([0, 0, 0.101])
    for _ in range(200):
        em2.step(1 / 240)
    Fzm = em2.contact_forces()[0, 2]; Mgm = em2._M[0] * _G; errm = abs(Fzm - Mgm) / Mgm * 100
    mf_ok = errm < 1
    print(f"  [contact force, manifold_reduce] Fz={Fzm:.2f} Mg={Mgm:.2f} error={errm:.3f}%  "
          f"{'ok' if mf_ok else 'FAIL'}"); ok = ok and mf_ok

    eh = GraphColoredContactEngine(manifold_reduce=True, manifold_on_host=True)
    ed = GraphColoredContactEngine(manifold_reduce=True)
    for k in range(8):
        eh.add_body([0, 0, 0.101 + k * 0.205]); ed.add_body([0, 0, 0.101 + k * 0.205])
    same = True
    for _ in range(120):
        eh.step(1 / 240); ed.step(1 / 240)
        same = same and eh.n_contacts_solved == ed.n_contacts_solved and \
            np.array_equal(eh.selection_debug(), ed.selection_debug())
    d_ok = same and float(np.max(np.abs(eh.get_state().xc - ed.get_state().xc))) == 0.0
    print(f"  [manifold on device vs on host] kept sets equal over 120 steps: {same}, "
          f"solved={ed.n_contacts_solved}  {'identical' if d_ok else 'FAIL'}"); ok = ok and d_ok

    eu = GraphColoredContactEngine(manifold_reduce=True)
    eg = GraphColoredContactEngine(manifold_reduce=True, graph_capture=True)
    for k in range(8):
        eu.add_body([0, 0, 0.101 + k * 0.205]); eg.add_body([0, 0, 0.101 + k * 0.205])
    same_c = True
    for _ in range(400):
        eu.step(1 / 240); eg.step(1 / 240)
        same_c = same_c and eu.n_contacts_solved == eg.n_contacts_solved and eu.n_colors == eg.n_colors
    g_ok = same_c and float(np.max(np.abs(eu.get_state().xc - eg.get_state().xc))) == 0.0
    print(f"  [solve loop as a CUDA graph] same colours/contacts over 400 steps: {same_c}, "
          f"captures={eg.n_graph_captures}  {'identical' if g_ok else 'FAIL'}"); ok = ok and g_ok

    ec = GraphColoredContactEngine(manifold_reduce=True, graph_capture=True, pair_chunks=True)
    ecg = GraphColoredContactEngine(manifold_reduce=True, graph_capture=True, pair_chunks=True,
                                    chunk_mode="global")
    for k in range(8):
        ec.add_body([0, 0, 0.101 + k * 0.205]); ecg.add_body([0, 0, 0.101 + k * 0.205])
    for _ in range(400):
        ec.step(1 / 240); ecg.step(1 / 240)
    stc = ec.get_state(); kec = float(np.sum(0.5 * ec._M[:, None] * stc.vc ** 2))
    zc = sorted(stc.xc[:, 2]); sepc = [zc[i + 1] - zc[i] for i in range(7)]
    mode_same = float(np.max(np.abs(stc.xc - ecg.get_state().xc))) == 0.0
    c_ok = kec < 0.5 and all(0.15 < sp < 0.25 for sp in sepc) and ec.n_colors < 8 and mode_same
    print(f"  [pair chunks] KE={kec:.3f} sep {min(sepc):.3f}-{max(sepc):.3f} pairs={ec.n_pairs} "
          f"colours={ec.n_colors} (per-contact colouring: 8), registers == global: {mode_same}  "
          f"{'ok' if c_ok else 'FAIL'}"); ok = ok and c_ok

    def run_stack4():
        e = GraphColoredContactEngine()
        for k in range(4):
            e.add_body([0, 0, 0.101 + k * 0.205])
        for _ in range(200):
            e.step(1 / 240)
        return e.get_state().xc
    dmax = float(np.max(np.abs(run_stack4() - run_stack4())))
    det_ok = dmax == 0.0
    print(f"  [determinism] max |delta xc| over two runs = {dmax:.2e} m  device={e.dev}  "
          f"{'bit-identical' if det_ok else 'FAIL: runs differ'}"); ok = ok and det_ok
    print(f"  -> {'coloured GPU ContactEngine: all four contract checks pass' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    print("Graph-coloured GPU contact engine selftest (Gauss-Seidel over colours, atomics-free solve):")
    sys.exit(0 if _selftest() else 1)
