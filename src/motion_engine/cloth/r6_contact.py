"""U112 round 6 contact geometry: the degenerate normal at zero distance (round-5 defect 1).

Round 4/5 built the vertex-triangle normal as n = d/|d| with d the vector from the closest point
on the triangle to the vertex, falling back to the triangle normal only below |d| < 1e-30.  When
a vertex sits ON the triangle (round 5 measured self-penetration equal to the full thickness h,
i.e. |d| -> 0) the direction of d is numerical noise, so the impulse direction rotates between
steps and the pair never separates.

Fix: when |d| < eps_n the normal is the TRIANGLE normal signed with the side the pair had at the
previous step.  The side is kept in a per-triangle table of S slots (slot = vertex id mod S,
claimed by atomic_min on the vertex id, so the winner does not depend on thread order) which is
double buffered: the step reads `prev` and writes `cur`, and a degenerate pair copies the side it
read forward, so the history survives as long as the pair stays in the candidate list.  Pairs that
lose a slot to a smaller vertex id fall back to the raw triangle normal and are counted.

With eps_n = 1e-30 every path is the round-4/5 path bit for bit, which is how the ablation
"normal fix off" is run.

A degenerate pair with no stored side keeps round 5's direction (d/|d|, or the triangle normal
below |d| < 1e-30): the first version of this fix used the raw triangle normal there, whose
winding sign is unrelated to the side the vertex is on, and it drove tyg F's drape to DC 2.02 %
against round 5's 11.31 % (r6_out/F2h0_firstfix.json).  Edge-edge pairs keep round 5's direction
throughout; no side history is kept for edge pairs, so their degenerate count is a diagnostic,
not a repair.
"""
import warp as wp

from .r4_contact import (F0, F1, FH, FM1, TINY, KIND_SUPPORT, KIND_VT, KIND_EE,
                        sdf_support, closest_bary, closest_seg)

wp.set_module_options({'fuse_fp': False, 'fast_math': False})

IMAX = wp.constant(wp.int32(2147483647))


@wp.kernel
def k_side_clear(key: wp.array2d(dtype=wp.int32), val: wp.array2d(dtype=wp.float64)):
    t, s = wp.tid()
    key[t, s] = IMAX
    val[t, s] = F0


@wp.kernel
def k_side_claim(ckind: wp.array(dtype=wp.int32), ca: wp.array(dtype=wp.int32),
                 cb: wp.array(dtype=wp.int32), C: wp.int32, S: wp.int32,
                 key: wp.array2d(dtype=wp.int32)):
    c = wp.tid()
    if c >= C:
        return
    if ckind[c] != KIND_VT:
        return
    v = ca[c]
    t = cb[c]
    slot = v - (v / S) * S
    wp.atomic_min(key, t, slot, v)


@wp.kernel
def k_side_store(ckind: wp.array(dtype=wp.int32), ca: wp.array(dtype=wp.int32),
                 cb: wp.array(dtype=wp.int32), C: wp.int32, S: wp.int32,
                 key: wp.array2d(dtype=wp.int32), side: wp.array(dtype=wp.float64),
                 val: wp.array2d(dtype=wp.float64)):
    c = wp.tid()
    if c >= C:
        return
    if ckind[c] != KIND_VT:
        return
    s = side[c]
    if s == F0:
        return
    v = ca[c]
    t = cb[c]
    slot = v - (v / S) * S
    if key[t, slot] == v:
        val[t, slot] = s


@wp.kernel
def k_geom6(x: wp.array(dtype=wp.vec3d), tri: wp.array2d(dtype=wp.int32),
            ev: wp.array2d(dtype=wp.int32), ckind: wp.array(dtype=wp.int32),
            ca: wp.array(dtype=wp.int32), cb: wp.array(dtype=wp.int32), C: wp.int32,
            h: wp.float64, mu_s: wp.float64, mu_c: wp.float64, eps_n: wp.float64,
            pkey: wp.array2d(dtype=wp.int32), pval: wp.array2d(dtype=wp.float64), S: wp.int32,
            ids: wp.array2d(dtype=wp.int32), wts: wp.array2d(dtype=wp.float64),
            nrm: wp.array(dtype=wp.vec3d), tg1: wp.array(dtype=wp.vec3d),
            tg2: wp.array(dtype=wp.vec3d), gap: wp.array(dtype=wp.float64),
            mu: wp.array(dtype=wp.float64), side: wp.array(dtype=wp.float64),
            stat: wp.array(dtype=wp.int32)):
    c = wp.tid()
    if c >= C:
        return
    k = ckind[c]
    n = wp.vec3d(F0, F1, F0)
    g = wp.float64(1.0e30)
    side[c] = F0
    for s in range(4):
        ids[c, s] = -1
        wts[c, s] = F0
    if k == KIND_SUPPORT:
        i = ca[c]
        z = sdf_support(x[i], cb[c])
        n = wp.vec3d(z[0], z[1], z[2])
        g = z[3] - FH * h
        ids[c, 0] = i
        wts[c, 0] = F1
        mu[c] = mu_s
    if k == KIND_VT:
        v = ca[c]
        t = cb[c]
        i0 = tri[t, 0]
        i1 = tri[t, 1]
        i2 = tri[t, 2]
        a = x[i0]
        b = x[i1]
        cc = x[i2]
        w = closest_bary(x[v], a, b, cc)
        q = w[0] * a + w[1] * b + w[2] * cc
        d = x[v] - q
        L = wp.length(d)
        nt = wp.normalize(wp.cross(b - a, cc - a))
        if L > TINY:
            n = d / L
        else:
            n = nt
        dn = wp.dot(d, nt)
        if dn > F0:
            side[c] = F1
        if dn < F0:
            side[c] = FM1
        if L <= eps_n:
            # degenerate: |d| carries no direction.  Take the side this pair had at the
            # previous step if the table still holds it; otherwise keep the round-5 value,
            # which is noise but at least not a sign flip of a normal we never measured.
            wp.atomic_add(stat, 0, 1)
            slot = v - (v / S) * S
            sgn = F0
            if pkey[t, slot] == v:
                sgn = pval[t, slot]
            if sgn != F0:
                n = sgn * nt
                side[c] = sgn
                wp.atomic_add(stat, 1, 1)
            else:
                wp.atomic_add(stat, 2, 1)
        g = L - h
        ids[c, 0] = v
        ids[c, 1] = i0
        ids[c, 2] = i1
        ids[c, 3] = i2
        wts[c, 0] = F1
        wts[c, 1] = FM1 * w[0]
        wts[c, 2] = FM1 * w[1]
        wts[c, 3] = FM1 * w[2]
        mu[c] = mu_c
    if k == KIND_EE:
        e1 = ca[c]
        e2 = cb[c]
        a0 = ev[e1, 0]
        a1 = ev[e1, 1]
        b0 = ev[e2, 0]
        b1 = ev[e2, 1]
        st = closest_seg(x[a0], x[a1], x[b0], x[b1])
        pa = x[a0] + st[0] * (x[a1] - x[a0])
        pb = x[b0] + st[1] * (x[b1] - x[b0])
        d = pa - pb
        L = wp.length(d)
        if L > TINY:
            n = d / L
        else:
            n = wp.normalize(wp.cross(x[a1] - x[a0], x[b1] - x[b0]))
        if L <= eps_n:
            # edge-edge keeps the round-5 direction; no side history is kept for edge pairs,
            # so the count is a diagnostic, not a repair
            wp.atomic_add(stat, 3, 1)
        g = L - h
        ids[c, 0] = a0
        ids[c, 1] = a1
        ids[c, 2] = b0
        ids[c, 3] = b1
        wts[c, 0] = F1 - st[0]
        wts[c, 1] = st[0]
        wts[c, 2] = FM1 * (F1 - st[1])
        wts[c, 3] = FM1 * st[1]
        mu[c] = mu_c
    nrm[c] = n
    gap[c] = g
    a = wp.vec3d(F1, F0, F0)
    if wp.abs(n[0]) > FH:
        a = wp.vec3d(F0, F1, F0)
    t1 = wp.normalize(wp.cross(n, a))
    tg1[c] = t1
    tg2[c] = wp.cross(n, t1)
