"""U112 round 4 contact: candidates with thickness h + exact NCP.

Narrow phase is a point-triangle / segment-segment DISTANCE test (not a ray probe): the round-3
failure was that two normal probes miss a perpendicular thickness overlap, which a distance test
cannot miss. Candidates come from a uniform hash grid (wp.HashGrid); pair generation is one thread
per triangle (vertex-triangle) and one thread per edge (edge-edge), so each pair is produced once.

The NCP is the Coulomb cone complementarity problem at velocity level
    u = J v + b,   K_mu = {(zn,zt): |zt| <= mu zn},   -u - Gamma(u) in normal cone of K_mu at lam,
solved by a projected Jacobi sweep whose per-contact update and cone projection are copied from
ncp_gpu.py (f_proj, k_sweep Baumgarte path, f_swap int64 accumulation). rho carries a mass split
(1/(kappa*G)) so the simultaneous sweep is contractive; the natural-map residual is reported.
"""
import warp as wp

wp.set_module_options({'fuse_fp': False, 'fast_math': False})

F0 = wp.constant(wp.float64(0.0))
F1 = wp.constant(wp.float64(1.0))
FH = wp.constant(wp.float64(0.5))
FM1 = wp.constant(wp.float64(-1.0))
TINY = wp.constant(wp.float64(1.0e-30))
ISCALE = wp.constant(wp.float64(1.0e15))

KIND_SUPPORT = wp.constant(wp.int32(0))
KIND_VT = wp.constant(wp.int32(1))
KIND_EE = wp.constant(wp.int32(2))


# ---------------------------------------------------------------- cone algebra (ncp_gpu.py)


@wp.func
def f_proj(z: wp.vec3d, mu: wp.float64):
    """Exact projection onto K_mu = {(zn, zt): ||zt|| <= mu zn}.  Copied from ncp_gpu.py."""
    zn = z[0]
    zt = wp.sqrt(z[1] * z[1] + z[2] * z[2])
    res = wp.vec3d(F0, F0, F0)
    if zt <= mu * zn:
        res = z
    else:
        if mu * zt <= FM1 * zn:
            res = wp.vec3d(F0, F0, F0)
        else:
            s = (zn + mu * zt) / (F1 + mu * mu)
            f = mu * s / zt
            res = wp.vec3d(s, f * z[1], f * z[2])
    return res


@wp.func
def f_eigmax(A: wp.mat33d):
    """Largest eigenvalue of a symmetric 3x3, closed form. Copied from ncp_gpu.py (f_eigmax);
    the unary-minus-on-float64-constant trap noted there is avoided by using FM1 * x."""
    F3 = wp.float64(3.0)
    F6 = wp.float64(6.0)
    p1 = A[0, 1] * A[0, 1] + A[0, 2] * A[0, 2] + A[1, 2] * A[1, 2]
    q = (A[0, 0] + A[1, 1] + A[2, 2]) / F3
    out = wp.max(A[0, 0], wp.max(A[1, 1], A[2, 2]))
    if p1 > F0:
        p2 = ((A[0, 0] - q) * (A[0, 0] - q) + (A[1, 1] - q) * (A[1, 1] - q)
              + (A[2, 2] - q) * (A[2, 2] - q) + wp.float64(2.0) * p1)
        pp = wp.sqrt(p2 / F6)
        if pp > wp.float64(1.0e-14):
            B = (F1 / pp) * (A - q * wp.mat33d(F1, F0, F0, F0, F1, F0, F0, F0, F1))
            r = wp.determinant(B) / wp.float64(2.0)
            r = wp.clamp(r, FM1, F1)
            phi = wp.acos(r) / F3
            out = q + wp.float64(2.0) * pp * wp.cos(phi)
    return out


# ---------------------------------------------------------------- geometry


@wp.func
def sdf_support(p: wp.vec3d, kind: wp.int32):
    """vec4d(nx,ny,nz,distance) of the rigid support. kind 1 = Cusick pedestal, 2 = slab edge."""
    n = wp.vec3d(F0, F1, F0)
    dist = wp.float64(1.0e30)
    if kind == 1:
        r = wp.sqrt(p[0] * p[0] + p[2] * p[2])
        a = r - wp.float64(0.09)
        b = wp.abs(p[1] - wp.float64(0.08)) - wp.float64(0.08)
        nr = wp.vec3d(p[0] / wp.max(r, TINY), F0, p[2] / wp.max(r, TINY))
        ny = wp.vec3d(F0, wp.sign(p[1] - wp.float64(0.08)), F0)
        out = wp.sqrt(wp.max(a, F0) * wp.max(a, F0) + wp.max(b, F0) * wp.max(b, F0))
        dist = out + wp.min(wp.max(a, b), F0)
        n = ny
        if a > b:
            n = nr
        if a > F0 and b > F0:
            n = (a * nr + b * ny) / wp.max(out, TINY)
    if kind == 2:
        a = p[0] - wp.float64(0.008)
        b = p[1]
        out = wp.sqrt(wp.max(a, F0) * wp.max(a, F0) + wp.max(b, F0) * wp.max(b, F0))
        dist = out + wp.min(wp.max(a, b), F0)
        n = wp.vec3d(F0, F1, F0)
        if a > b:
            n = wp.vec3d(F1, F0, F0)
        if a > F0 and b > F0:
            n = wp.vec3d(a / wp.max(out, TINY), b / wp.max(out, TINY), F0)
    return wp.vec4d(n[0], n[1], n[2], dist)


@wp.func
def closest_bary(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d):
    """Barycentric weights of the closest point of triangle (a,b,c) to p (Ericson)."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    res = wp.vec3d(F0, F0, F0)
    done = wp.int32(0)
    if d1 <= F0 and d2 <= F0:
        res = wp.vec3d(F1, F0, F0)
        done = 1
    if done == 0 and d3 >= F0 and d4 <= d3:
        res = wp.vec3d(F0, F1, F0)
        done = 1
    if done == 0 and d6 >= F0 and d5 <= d6:
        res = wp.vec3d(F0, F0, F1)
        done = 1
    if done == 0 and vc <= F0 and d1 >= F0 and d3 <= F0:
        t = d1 / wp.max(d1 - d3, TINY)
        res = wp.vec3d(F1 - t, t, F0)
        done = 1
    if done == 0 and vb <= F0 and d2 >= F0 and d6 <= F0:
        t = d2 / wp.max(d2 - d6, TINY)
        res = wp.vec3d(F1 - t, F0, t)
        done = 1
    if done == 0 and va <= F0 and (d4 - d3) >= F0 and (d5 - d6) >= F0:
        t = (d4 - d3) / wp.max((d4 - d3) + (d5 - d6), TINY)
        res = wp.vec3d(F0, F1 - t, t)
        done = 1
    if done == 0:
        den = F1 / wp.max(va + vb + vc, TINY)
        v = vb * den
        w = vc * den
        res = wp.vec3d(F1 - v - w, v, w)
    return res


@wp.func
def closest_seg(p1: wp.vec3d, q1: wp.vec3d, p2: wp.vec3d, q2: wp.vec3d):
    """Parameters (s,t) of the closest points on segments p1q1 and p2q2 (Ericson)."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = wp.dot(d1, d1)
    e = wp.dot(d2, d2)
    f = wp.dot(d2, r)
    c = wp.dot(d1, r)
    b = wp.dot(d1, d2)
    den = a * e - b * b
    s = F0
    if den > TINY:
        s = wp.clamp((b * f - c * e) / den, F0, F1)
    t = (b * s + f) / wp.max(e, TINY)
    if t < F0:
        t = F0
        s = wp.clamp(FM1 * c / wp.max(a, TINY), F0, F1)
    if t > F1:
        t = F1
        s = wp.clamp((b - c) / wp.max(a, TINY), F0, F1)
    return wp.vec2d(s, t)


# ---------------------------------------------------------------- broad phase


@wp.kernel
def k_emit_support(x: wp.array(dtype=wp.vec3d), free: wp.array(dtype=wp.int32),
                   kind: wp.int32, thr: wp.float64, hh: wp.float64,
                   count: wp.array(dtype=wp.int32), cap: wp.int32,
                   ckind: wp.array(dtype=wp.int32), ca: wp.array(dtype=wp.int32),
                   cb: wp.array(dtype=wp.int32), over: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if free[i] == 0:
        return
    z = sdf_support(x[i], kind)
    if z[3] - hh < thr:
        j = wp.atomic_add(count, 0, 1)
        if j < cap:
            ckind[j] = KIND_SUPPORT
            ca[j] = i
            cb[j] = kind
        else:
            wp.atomic_add(over, 0, 1)


@wp.kernel
def k_emit_vt(grid: wp.uint64, x: wp.array(dtype=wp.vec3d), xf: wp.array(dtype=wp.vec3),
              tri: wp.array2d(dtype=wp.int32), nbr: wp.array2d(dtype=wp.int32),
              qr: wp.float32, thr: wp.float64,
              count: wp.array(dtype=wp.int32), cap: wp.int32,
              ckind: wp.array(dtype=wp.int32), ca: wp.array(dtype=wp.int32),
              cb: wp.array(dtype=wp.int32), over: wp.array(dtype=wp.int32)):
    t = wp.tid()
    i0 = tri[t, 0]
    i1 = tri[t, 1]
    i2 = tri[t, 2]
    a = x[i0]
    b = x[i1]
    c = x[i2]
    cen = wp.vec3(wp.float32((a[0] + b[0] + c[0]) / wp.float64(3.0)),
                  wp.float32((a[1] + b[1] + c[1]) / wp.float64(3.0)),
                  wp.float32((a[2] + b[2] + c[2]) / wp.float64(3.0)))
    query = wp.hash_grid_query(grid, cen, qr)
    v = wp.int32(0)
    while wp.hash_grid_query_next(query, v):
        if (v != i0 and v != i1 and v != i2
                and v != nbr[t, 0] and v != nbr[t, 1] and v != nbr[t, 2]):
            p = x[v]
            w = closest_bary(p, a, b, c)
            q = w[0] * a + w[1] * b + w[2] * c
            if wp.length(p - q) < thr:
                j = wp.atomic_add(count, 0, 1)
                if j < cap:
                    ckind[j] = KIND_VT
                    ca[j] = v
                    cb[j] = t
                else:
                    wp.atomic_add(over, 0, 1)


@wp.kernel
def k_edge_mid(x: wp.array(dtype=wp.vec3d), ev: wp.array2d(dtype=wp.int32),
               out: wp.array(dtype=wp.vec3)):
    e = wp.tid()
    m = FH * (x[ev[e, 0]] + x[ev[e, 1]])
    out[e] = wp.vec3(wp.float32(m[0]), wp.float32(m[1]), wp.float32(m[2]))


@wp.kernel
def k_emit_ee(grid: wp.uint64, x: wp.array(dtype=wp.vec3d), mid: wp.array(dtype=wp.vec3),
              ev: wp.array2d(dtype=wp.int32), qr: wp.float32, thr: wp.float64,
              count: wp.array(dtype=wp.int32), cap: wp.int32,
              ckind: wp.array(dtype=wp.int32), ca: wp.array(dtype=wp.int32),
              cb: wp.array(dtype=wp.int32), over: wp.array(dtype=wp.int32)):
    e1 = wp.tid()
    a0 = ev[e1, 0]
    a1 = ev[e1, 1]
    p1 = x[a0]
    q1 = x[a1]
    query = wp.hash_grid_query(grid, mid[e1], qr)
    e2 = wp.int32(0)
    while wp.hash_grid_query_next(query, e2):
        if e2 > e1:
            b0 = ev[e2, 0]
            b1 = ev[e2, 1]
            if b0 != a0 and b0 != a1 and b1 != a0 and b1 != a1:
                st = closest_seg(p1, q1, x[b0], x[b1])
                pa = p1 + st[0] * (q1 - p1)
                pb = x[b0] + st[1] * (x[b1] - x[b0])
                if wp.length(pa - pb) < thr:
                    j = wp.atomic_add(count, 0, 1)
                    if j < cap:
                        ckind[j] = KIND_EE
                        ca[j] = e1
                        cb[j] = e2
                    else:
                        wp.atomic_add(over, 0, 1)


# ---------------------------------------------------------------- contact geometry / NCP


@wp.kernel
def k_geom(x: wp.array(dtype=wp.vec3d), tri: wp.array2d(dtype=wp.int32),
           ev: wp.array2d(dtype=wp.int32), ckind: wp.array(dtype=wp.int32),
           ca: wp.array(dtype=wp.int32), cb: wp.array(dtype=wp.int32), C: wp.int32,
           h: wp.float64, mu_s: wp.float64, mu_c: wp.float64,
           ids: wp.array2d(dtype=wp.int32), wts: wp.array2d(dtype=wp.float64),
           nrm: wp.array(dtype=wp.vec3d), tg1: wp.array(dtype=wp.vec3d),
           tg2: wp.array(dtype=wp.vec3d), gap: wp.array(dtype=wp.float64),
           mu: wp.array(dtype=wp.float64)):
    c = wp.tid()
    if c >= C:
        return
    k = ckind[c]
    n = wp.vec3d(F0, F1, F0)
    g = wp.float64(1.0e30)
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
        if L > TINY:
            n = d / L
        else:
            n = wp.normalize(wp.cross(b - a, cc - a))
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


@wp.kernel
def k_count_nodes(ids: wp.array2d(dtype=wp.int32), C: wp.int32,
                  cnt: wp.array(dtype=wp.int32)):
    c = wp.tid()
    if c >= C:
        return
    for s in range(4):
        i = ids[c, s]
        if i >= 0:
            wp.atomic_add(cnt, i, 1)


@wp.kernel
def k_rho(ids: wp.array2d(dtype=wp.int32), wts: wp.array2d(dtype=wp.float64), C: wp.int32,
          pre: wp.array(dtype=wp.mat33d), dt: wp.float64, cnt: wp.array(dtype=wp.int32),
          rho: wp.array(dtype=wp.float64)):
    """rho = 1/(kappa * eigmax(W)) with W = sum_k w_k^2 A_kk^-1 / dt^2: the Delassus of the
    IMPLICIT system (block-diagonal of H + M(1+dt*damp)/dt^2), not of the lumped mass. On a
    near-inextensible membrane the two differ by three orders of magnitude."""
    c = wp.tid()
    if c >= C:
        return
    W = wp.mat33d()
    kap = wp.int32(1)
    for s in range(4):
        i = ids[c, s]
        if i >= 0:
            W = W + (wts[c, s] * wts[c, s] / (dt * dt)) * pre[i]
            kap = wp.max(kap, cnt[i])
    lm = f_eigmax(FH * (W + wp.transpose(W)))
    if lm > TINY:
        rho[c] = F1 / (wp.float64(kap) * lm)
    else:
        rho[c] = F0


@wp.kernel
def k_sweep(ids: wp.array2d(dtype=wp.int32), wts: wp.array2d(dtype=wp.float64), C: wp.int32,
            nrm: wp.array(dtype=wp.vec3d), tg1: wp.array(dtype=wp.vec3d),
            tg2: wp.array(dtype=wp.vec3d), gap: wp.array(dtype=wp.float64),
            mu: wp.array(dtype=wp.float64), rho: wp.array(dtype=wp.float64),
            v: wp.array(dtype=wp.vec3d), dt: wp.float64, desaxce: wp.int32,
            lam: wp.array(dtype=wp.vec3d), acc: wp.array2d(dtype=wp.int64)):
    """One projected Jacobi sweep. lam <- proj_K(lam - rho (u + Gamma(u))) with the int64 swap
    accumulation of ncp_gpu.f_swap: acc holds the per-node TOTAL J^T lam, not a running delta."""
    c = wp.tid()
    if c >= C:
        return
    rel = wp.vec3d(F0, F0, F0)
    for s in range(4):
        i = ids[c, s]
        if i >= 0:
            rel = rel + wts[c, s] * v[i]
    n = nrm[c]
    t1 = tg1[c]
    t2 = tg2[c]
    u = wp.vec3d(wp.dot(rel, n) + gap[c] / dt, wp.dot(rel, t1), wp.dot(rel, t2))
    m = mu[c]
    lo = lam[c]
    s0 = u
    if desaxce == 1:
        ut = wp.sqrt(u[1] * u[1] + u[2] * u[2])
        s0 = wp.vec3d(u[0] + m * ut, u[1], u[2])
    zn = f_proj(lo - rho[c] * s0, m)
    lam[c] = zn
    dn = zn[0] * n + zn[1] * t1 + zn[2] * t2
    do = lo[0] * n + lo[1] * t1 + lo[2] * t2
    for s in range(4):
        i = ids[c, s]
        if i >= 0:
            w = wts[c, s]
            for k in range(3):
                wp.atomic_add(acc, i, k,
                              wp.int64(wp.round(w * dn[k] * ISCALE))
                              - wp.int64(wp.round(w * do[k] * ISCALE)))


@wp.kernel
def k_apply(vfree: wp.array(dtype=wp.vec3d), acc: wp.array2d(dtype=wp.int64),
            pre: wp.array(dtype=wp.mat33d), dt: wp.float64, v: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    d = wp.vec3d(wp.float64(acc[i, 0]), wp.float64(acc[i, 1]), wp.float64(acc[i, 2])) / ISCALE
    v[i] = vfree[i] + (F1 / (dt * dt)) * (pre[i] * d)


@wp.kernel
def k_resid(ids: wp.array2d(dtype=wp.int32), wts: wp.array2d(dtype=wp.float64), C: wp.int32,
            nrm: wp.array(dtype=wp.vec3d), tg1: wp.array(dtype=wp.vec3d),
            tg2: wp.array(dtype=wp.vec3d), gap: wp.array(dtype=wp.float64),
            mu: wp.array(dtype=wp.float64), rho: wp.array(dtype=wp.float64),
            v: wp.array(dtype=wp.vec3d), dt: wp.float64, desaxce: wp.int32,
            lam: wp.array(dtype=wp.vec3d), res: wp.array(dtype=wp.float64)):
    c = wp.tid()
    if c >= C:
        res[c] = F0
        return
    rel = wp.vec3d(F0, F0, F0)
    for s in range(4):
        i = ids[c, s]
        if i >= 0:
            rel = rel + wts[c, s] * v[i]
    n = nrm[c]
    u = wp.vec3d(wp.dot(rel, n) + gap[c] / dt, wp.dot(rel, tg1[c]), wp.dot(rel, tg2[c]))
    m = mu[c]
    s0 = u
    if desaxce == 1:
        ut = wp.sqrt(u[1] * u[1] + u[2] * u[2])
        s0 = wp.vec3d(u[0] + m * ut, u[1], u[2])
    d = lam[c] - f_proj(lam[c] - rho[c] * s0, m)
    res[c] = wp.max(wp.abs(d[0]), wp.max(wp.abs(d[1]), wp.abs(d[2])))


@wp.kernel
def k_contact_force(acc: wp.array2d(dtype=wp.int64), dt: wp.float64,
                    fc: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    fc[i] = wp.vec3d(wp.float64(acc[i, 0]), wp.float64(acc[i, 1]),
                     wp.float64(acc[i, 2])) / (ISCALE * dt)


@wp.kernel
def k_gap_stats(gap: wp.array(dtype=wp.float64), ckind: wp.array(dtype=wp.int32), C: wp.int32,
                h: wp.float64, out: wp.array(dtype=wp.int64)):
    """out[0..2]: max penetration (1e12 fixed point) for support / VT / EE;
    out[3..5]: number of contacts with penetration > h/2."""
    c = wp.tid()
    if c >= C:
        return
    k = ckind[c]
    pen = wp.max(FM1 * gap[c], F0)
    wp.atomic_max(out, k, wp.int64(pen * wp.float64(1.0e12)))
    if pen > FH * h:
        wp.atomic_add(out, 3 + k, wp.int64(1))


@wp.kernel
def k_vfree(v: wp.array(dtype=wp.vec3d), acc: wp.array2d(dtype=wp.int64),
            pre: wp.array(dtype=wp.mat33d), dt: wp.float64, out: wp.array(dtype=wp.vec3d)):
    """Remove the impulse that is already carried by the held contact force, so the NCP sees the
    velocity the elastic solve would have produced without it."""
    i = wp.tid()
    d = wp.vec3d(wp.float64(acc[i, 0]), wp.float64(acc[i, 1]), wp.float64(acc[i, 2])) / ISCALE
    out[i] = v[i] - (F1 / (dt * dt)) * (pre[i] * d)
