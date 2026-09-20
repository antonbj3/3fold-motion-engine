"""U112 round 4 kernels.

Implicit backward Euler on the Lagrange mesh.
  * forces: the verified float64 kernels in energies.py (U93 hinge + Pipkin membrane) are used
    unchanged; nothing here recomputes a force that is already verified there.
  * tangent: matrix-free element Hessian action. Triangle tangent = geometric term dF S + material
    term F dS with S projected to PSD (2x2 eigen-clip, same eigenbasis as E) and the branch
    material tangent; hinge tangent = Gauss-Newton B w g g^T. Both are PSD by construction.
  * reductions: cross-thread scatter is int64 fixed point (forces 1e12, impulses 1e15).
    Inner products/energies use a fixed-order float64 tree reduction (no atomics, no thread-order
    dependence), so both are bit-reproducible.
  * contact: exact NCP with the Coulomb cone projection copied from ncp_gpu.py (f_proj), applied
    to node-support, vertex-triangle and edge-edge stencils with thickness h.
"""
import warp as wp

wp.set_module_options({'fuse_fp': False, 'fast_math': False})

F0 = wp.constant(wp.float64(0.0))
F1 = wp.constant(wp.float64(1.0))
F2 = wp.constant(wp.float64(2.0))
FH = wp.constant(wp.float64(0.5))
FM1 = wp.constant(wp.float64(-1.0))
TINY = wp.constant(wp.float64(1.0e-30))
FSCALE = wp.constant(wp.float64(1.0e12))
ISCALE = wp.constant(wp.float64(1.0e15))
GRAV = wp.constant(wp.float64(9.80665))


# ---------------------------------------------------------------- triangle state


@wp.struct
class TS:
    S11: wp.float64
    S12: wp.float64
    S22: wp.float64
    m11: wp.float64
    m12: wp.float64
    m22: wp.float64
    ep: wp.float64
    disc: wp.float64
    br: wp.int32
    en: wp.float64
    clip: wp.int32


@wp.func
def tri_state(wu: wp.vec3d, wv: wp.vec3d, Eh: wp.float64, nu: wp.float64):
    """Green strain, relaxed (Pipkin) PK2 stress, energy and the eigenprojector of E.

    Branches match energies.compute_pipkin_membrane_forces_kernel exactly:
      eps1<=0            -> slack   (S=0)
      eps2+nu*eps1>0     -> taut    (St Venant Kirchhoff)
      otherwise          -> wrinkle (uniaxial along the eps1 eigenvector)
    """
    r = TS()
    C11 = wp.dot(wu, wu)
    C12 = wp.dot(wu, wv)
    C22 = wp.dot(wv, wv)
    E11 = FH * (C11 - F1)
    E12 = FH * C12
    E22 = FH * (C22 - F1)
    trE = E11 + E22
    diff = E11 - E22
    disc = wp.sqrt(diff * diff + wp.float64(4.0) * E12 * E12)
    ep = FH * (trE + disc)
    em = FH * (trE - disc)
    m11 = FH
    m12 = F0
    m22 = FH
    if disc > wp.float64(1.0e-12):
        m11 = FH + FH * diff / disc
        m12 = E12 / disc
        m22 = FH - FH * diff / disc
    r.m11 = m11
    r.m12 = m12
    r.m22 = m22
    r.ep = ep
    r.disc = disc
    r.br = 0
    r.S11 = F0
    r.S12 = F0
    r.S22 = F0
    r.en = F0
    r.clip = 0
    if ep <= F0:
        return r
    if em + nu * ep > F0:
        r.br = 2
        coeff = Eh / (F1 - nu * nu)
        # eigenvalues of S in the eigenbasis of E; both are > 0 inside the taut branch.
        sa = coeff * (ep + nu * em)
        sb = coeff * (em + nu * ep)
        if sa < F0:
            sa = F0
            r.clip = 1
        if sb < F0:
            sb = F0
            r.clip = 1
        r.S11 = sa * m11 + sb * (F1 - m11)
        r.S12 = (sa - sb) * m12
        r.S22 = sa * m22 + sb * (F1 - m22)
        r.en = FH * (sa * ep + sb * em)
    else:
        r.br = 1
        s1 = Eh * ep
        r.S11 = s1 * m11
        r.S12 = s1 * m12
        r.S22 = s1 * m22
        r.en = FH * s1 * ep
    return r


@wp.func
def tri_dS(st: TS, dE11: wp.float64, dE12: wp.float64, dE22: wp.float64,
           Eh: wp.float64, nu: wp.float64):
    """Material tangent dS = C:dE for the active branch. PSD as a map dE -> dS in every branch."""
    out = wp.vec3d(F0, F0, F0)
    if st.br == 2:
        coeff = Eh / (F1 - nu * nu)
        out = wp.vec3d(coeff * (dE11 + nu * dE22), coeff * (F1 - nu) * dE12,
                       coeff * (dE22 + nu * dE11))
    if st.br == 1:
        m11 = st.m11
        m12 = st.m12
        m22 = st.m22
        dep = m11 * dE11 + F2 * m12 * dE12 + m22 * dE22
        # dM = (O dE M + M dE O)/(ep-em); ratio ep/(ep-em) <= 1 inside the wrinkle branch.
        p = m22 * dE11 - m12 * dE12
        q = m22 * dE12 - m12 * dE22
        rr = m11 * dE12 - m12 * dE11
        s = m11 * dE22 - m12 * dE12
        X11 = p * m11 + q * m12
        X12 = p * m12 + q * m22
        X21 = rr * m11 + s * m12
        X22 = rr * m12 + s * m22
        fac = Eh * st.ep / wp.max(st.disc, TINY)
        out = wp.vec3d(Eh * dep * m11 + fac * F2 * X11,
                       Eh * dep * m12 + fac * (X12 + X21),
                       Eh * dep * m22 + fac * F2 * X22)
    return out


# ---------------------------------------------------------------- energy


@wp.kernel
def k_tri_energy(x: wp.array(dtype=wp.vec3d), t0: wp.array(dtype=wp.int32),
                 t1: wp.array(dtype=wp.int32), t2: wp.array(dtype=wp.int32),
                 tb: wp.array(dtype=wp.vec3d), tc: wp.array(dtype=wp.vec3d),
                 ta: wp.array(dtype=wp.float64), Eh: wp.float64, nu: wp.float64,
                 out: wp.array(dtype=wp.float64)):
    t = wp.tid()
    i0 = t0[t]
    i1 = t1[t]
    i2 = t2[t]
    b = tb[t]
    c = tc[t]
    p0 = x[i0]
    d1 = x[i1] - p0
    d2 = x[i2] - p0
    wu = b[1] * d1 + b[2] * d2
    wv = c[1] * d1 + c[2] * d2
    st = tri_state(wu, wv, Eh, nu)
    out[t] = ta[t] * st.en


@wp.kernel
def k_hinge_energy(x: wp.array(dtype=wp.vec3d), v1: wp.array(dtype=wp.int32),
                   v2: wp.array(dtype=wp.int32), v3: wp.array(dtype=wp.int32),
                   v4: wp.array(dtype=wp.int32), hw: wp.array(dtype=wp.float64),
                   h0: wp.array(dtype=wp.float64), B: wp.float64,
                   out: wp.array(dtype=wp.float64)):
    h = wp.tid()
    out[h] = F0
    p1 = x[v1[h]]
    p2 = x[v2[h]]
    p3 = x[v3[h]]
    p4 = x[v4[h]]
    e = p2 - p1
    le = wp.length(e)
    if le < wp.float64(1.0e-7):
        return
    n1 = wp.cross(e, p3 - p1)
    l1 = wp.length(n1)
    if l1 < wp.float64(1.0e-7):
        return
    n2 = wp.cross(p4 - p1, e)
    l2 = wp.length(n2)
    if l2 < wp.float64(1.0e-7):
        return
    n1 = n1 / l1
    n2 = n2 / l2
    ct = wp.clamp(wp.dot(n1, n2), FM1, F1)
    stn = wp.dot(wp.cross(n1, n2), e / le)
    th = wp.atan2(stn, ct) - h0[h]
    out[h] = FH * B * hw[h] * th * th


@wp.kernel
def k_node_energy(x: wp.array(dtype=wp.vec3d), m: wp.array(dtype=wp.float64),
                  tgt: wp.array(dtype=wp.vec3d), dm: wp.array(dtype=wp.float64),
                  fc: wp.array(dtype=wp.vec3d), free: wp.array(dtype=wp.int32),
                  out: wp.array(dtype=wp.float64)):
    """Gravity + backward-Euler inertia + the linear potential of the held contact force."""
    i = wp.tid()
    if free[i] == 0:
        out[i] = F0
        return
    d = x[i] - tgt[i]
    out[i] = GRAV * m[i] * x[i][1] + FH * dm[i] * wp.dot(d, d) - wp.dot(fc[i], x[i])


# ---------------------------------------------------------------- tangent action


@wp.kernel
def k_tri_matvec(x: wp.array(dtype=wp.vec3d), dp: wp.array(dtype=wp.vec3d),
                 t0: wp.array(dtype=wp.int32), t1: wp.array(dtype=wp.int32),
                 t2: wp.array(dtype=wp.int32), tb: wp.array(dtype=wp.vec3d),
                 tc: wp.array(dtype=wp.vec3d), ta: wp.array(dtype=wp.float64),
                 Eh: wp.float64, nu: wp.float64, acc: wp.array2d(dtype=wp.int64)):
    t = wp.tid()
    i0 = t0[t]
    i1 = t1[t]
    i2 = t2[t]
    b = tb[t]
    c = tc[t]
    p0 = x[i0]
    wu = b[1] * (x[i1] - p0) + b[2] * (x[i2] - p0)
    wv = c[1] * (x[i1] - p0) + c[2] * (x[i2] - p0)
    st = tri_state(wu, wv, Eh, nu)
    if st.br == 0:
        return
    q0 = dp[i0]
    du = b[0] * q0 + b[1] * dp[i1] + b[2] * dp[i2]
    dv = c[0] * q0 + c[1] * dp[i1] + c[2] * dp[i2]
    dE11 = wp.dot(wu, du)
    dE22 = wp.dot(wv, dv)
    dE12 = FH * (wp.dot(wu, dv) + wp.dot(wv, du))
    dS = tri_dS(st, dE11, dE12, dE22, Eh, nu)
    g0 = du * st.S11 + wu * dS[0] + dv * st.S12 + wv * dS[1]
    g1 = du * st.S12 + wu * dS[1] + dv * st.S22 + wv * dS[2]
    a = ta[t]
    f0 = a * (b[0] * g0 + c[0] * g1)
    f1 = a * (b[1] * g0 + c[1] * g1)
    f2 = a * (b[2] * g0 + c[2] * g1)
    for k in range(3):
        wp.atomic_add(acc, i0, k, wp.int64(f0[k] * FSCALE))
        wp.atomic_add(acc, i1, k, wp.int64(f1[k] * FSCALE))
        wp.atomic_add(acc, i2, k, wp.int64(f2[k] * FSCALE))


@wp.kernel
def k_tri_diag(x: wp.array(dtype=wp.vec3d), t0: wp.array(dtype=wp.int32),
               t1: wp.array(dtype=wp.int32), t2: wp.array(dtype=wp.int32),
               tb: wp.array(dtype=wp.vec3d), tc: wp.array(dtype=wp.vec3d),
               ta: wp.array(dtype=wp.float64), Eh: wp.float64, nu: wp.float64,
               acc: wp.array2d(dtype=wp.int64)):
    t = wp.tid()
    i0 = t0[t]
    i1 = t1[t]
    i2 = t2[t]
    b = tb[t]
    c = tc[t]
    p0 = x[i0]
    wu = b[1] * (x[i1] - p0) + b[2] * (x[i2] - p0)
    wv = c[1] * (x[i1] - p0) + c[2] * (x[i2] - p0)
    st = tri_state(wu, wv, Eh, nu)
    if st.br == 0:
        return
    a = ta[t]
    for vtx in range(3):
        bv = b[vtx]
        cv = c[vtx]
        idx = i0
        if vtx == 1:
            idx = i1
        if vtx == 2:
            idx = i2
        for comp in range(3):
            dE11 = bv * wu[comp]
            dE22 = cv * wv[comp]
            dE12 = FH * (bv * wv[comp] + cv * wu[comp])
            dS = tri_dS(st, dE11, dE12, dE22, Eh, nu)
            g0 = bv * st.S11 + cv * st.S12 + wu[comp] * dS[0] + wv[comp] * dS[1]
            g1 = bv * st.S12 + cv * st.S22 + wu[comp] * dS[1] + wv[comp] * dS[2]
            wp.atomic_add(acc, idx, comp, wp.int64(a * (bv * g0 + cv * g1) * FSCALE))


@wp.kernel
def k_hinge_matvec(x: wp.array(dtype=wp.vec3d), dp: wp.array(dtype=wp.vec3d),
                   v1: wp.array(dtype=wp.int32), v2: wp.array(dtype=wp.int32),
                   v3: wp.array(dtype=wp.int32), v4: wp.array(dtype=wp.int32),
                   hw: wp.array(dtype=wp.float64), B: wp.float64,
                   acc: wp.array2d(dtype=wp.int64)):
    h = wp.tid()
    i1 = v1[h]
    i2 = v2[h]
    i3 = v3[h]
    i4 = v4[h]
    p1 = x[i1]
    p2 = x[i2]
    p3 = x[i3]
    p4 = x[i4]
    e = p2 - p1
    le = wp.length(e)
    if le < wp.float64(1.0e-7):
        return
    n1 = wp.cross(e, p3 - p1)
    l1 = wp.length(n1)
    if l1 < wp.float64(1.0e-7):
        return
    n2 = wp.cross(p4 - p1, e)
    l2 = wp.length(n2)
    if l2 < wp.float64(1.0e-7):
        return
    g3 = (n1 / l1) * (FM1 * le / l1)
    g4 = (n2 / l2) * (FM1 * le / l2)
    al1 = wp.dot(p2 - p3, e) / (le * le)
    al2 = wp.dot(p3 - p1, e) / (le * le)
    be1 = wp.dot(p2 - p4, e) / (le * le)
    be2 = wp.dot(p4 - p1, e) / (le * le)
    g1 = FM1 * al1 * g3 - be1 * g4
    g2 = FM1 * al2 * g3 - be2 * g4
    s = wp.dot(g1, dp[i1]) + wp.dot(g2, dp[i2]) + wp.dot(g3, dp[i3]) + wp.dot(g4, dp[i4])
    k = B * hw[h] * s
    for c in range(3):
        wp.atomic_add(acc, i1, c, wp.int64(k * g1[c] * FSCALE))
        wp.atomic_add(acc, i2, c, wp.int64(k * g2[c] * FSCALE))
        wp.atomic_add(acc, i3, c, wp.int64(k * g3[c] * FSCALE))
        wp.atomic_add(acc, i4, c, wp.int64(k * g4[c] * FSCALE))


@wp.kernel
def k_hinge_diag(x: wp.array(dtype=wp.vec3d), v1: wp.array(dtype=wp.int32),
                 v2: wp.array(dtype=wp.int32), v3: wp.array(dtype=wp.int32),
                 v4: wp.array(dtype=wp.int32), hw: wp.array(dtype=wp.float64),
                 B: wp.float64, acc: wp.array2d(dtype=wp.int64)):
    h = wp.tid()
    i1 = v1[h]
    i2 = v2[h]
    i3 = v3[h]
    i4 = v4[h]
    p1 = x[i1]
    p2 = x[i2]
    p3 = x[i3]
    p4 = x[i4]
    e = p2 - p1
    le = wp.length(e)
    if le < wp.float64(1.0e-7):
        return
    n1 = wp.cross(e, p3 - p1)
    l1 = wp.length(n1)
    if l1 < wp.float64(1.0e-7):
        return
    n2 = wp.cross(p4 - p1, e)
    l2 = wp.length(n2)
    if l2 < wp.float64(1.0e-7):
        return
    g3 = (n1 / l1) * (FM1 * le / l1)
    g4 = (n2 / l2) * (FM1 * le / l2)
    al1 = wp.dot(p2 - p3, e) / (le * le)
    al2 = wp.dot(p3 - p1, e) / (le * le)
    be1 = wp.dot(p2 - p4, e) / (le * le)
    be2 = wp.dot(p4 - p1, e) / (le * le)
    g1 = FM1 * al1 * g3 - be1 * g4
    g2 = FM1 * al2 * g3 - be2 * g4
    k = B * hw[h]
    for c in range(3):
        wp.atomic_add(acc, i1, c, wp.int64(k * g1[c] * g1[c] * FSCALE))
        wp.atomic_add(acc, i2, c, wp.int64(k * g2[c] * g2[c] * FSCALE))
        wp.atomic_add(acc, i3, c, wp.int64(k * g3[c] * g3[c] * FSCALE))
        wp.atomic_add(acc, i4, c, wp.int64(k * g4[c] * g4[c] * FSCALE))


# ---------------------------------------------------------------- vector algebra


@wp.kernel
def k_residual(fint: wp.array2d(dtype=wp.int64), m: wp.array(dtype=wp.float64),
               dm: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.vec3d),
               tgt: wp.array(dtype=wp.vec3d), fc: wp.array(dtype=wp.vec3d),
               free: wp.array(dtype=wp.int32), r: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    if free[i] == 0:
        r[i] = wp.vec3d(F0, F0, F0)
        return
    f = wp.vec3d(wp.float64(fint[i, 0]), wp.float64(fint[i, 1]), wp.float64(fint[i, 2])) / FSCALE
    f = f + wp.vec3d(F0, FM1 * GRAV * m[i], F0) + fc[i]
    r[i] = f - dm[i] * (x[i] - tgt[i])


@wp.kernel
def k_precond(acc: wp.array2d(dtype=wp.int64), dm: wp.array(dtype=wp.float64),
              free: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    if free[i] == 0:
        out[i] = wp.vec3d(F1, F1, F1)
        return
    d = wp.vec3d(wp.float64(acc[i, 0]), wp.float64(acc[i, 1]), wp.float64(acc[i, 2])) / FSCALE
    out[i] = wp.vec3d(wp.max(d[0] + dm[i], dm[i]), wp.max(d[1] + dm[i], dm[i]),
                      wp.max(d[2] + dm[i], dm[i]))


@wp.kernel
def k_apply_A(acc: wp.array2d(dtype=wp.int64), p: wp.array(dtype=wp.vec3d),
              dm: wp.array(dtype=wp.float64), free: wp.array(dtype=wp.int32),
              out: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    if free[i] == 0:
        out[i] = p[i]
        return
    d = wp.vec3d(wp.float64(acc[i, 0]), wp.float64(acc[i, 1]), wp.float64(acc[i, 2])) / FSCALE
    out[i] = d + dm[i] * p[i]


@wp.kernel
def k_jacobi(r: wp.array(dtype=wp.vec3d), d: wp.array(dtype=wp.vec3d),
             out: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    out[i] = wp.vec3d(r[i][0] / d[i][0], r[i][1] / d[i][1], r[i][2] / d[i][2])


@wp.kernel
def k_dot_block(a: wp.array(dtype=wp.vec3d), b: wp.array(dtype=wp.vec3d), n: wp.int32,
                T: wp.int32, out: wp.array(dtype=wp.float64)):
    """Fixed-partition, fixed-order float64 reduction: no atomics, bit-reproducible."""
    j = wp.tid()
    s = F0
    for k in range(T):
        i = j * T + k
        if i < n:
            s = s + wp.dot(a[i], b[i])
    out[j] = s


@wp.kernel
def k_sum_block(src: wp.array(dtype=wp.float64), n: wp.int32, T: wp.int32,
                out: wp.array(dtype=wp.float64)):
    j = wp.tid()
    s = F0
    for k in range(T):
        i = j * T + k
        if i < n:
            s = s + src[i]
    out[j] = s


@wp.kernel
def k_max_block(src: wp.array(dtype=wp.float64), n: wp.int32, T: wp.int32,
                out: wp.array(dtype=wp.float64)):
    j = wp.tid()
    s = F0
    for k in range(T):
        i = j * T + k
        if i < n:
            s = wp.max(s, src[i])
    out[j] = s


@wp.kernel
def k_axpy(y: wp.array(dtype=wp.vec3d), a: wp.array(dtype=wp.float64), ai: wp.int32,
           sign: wp.float64, xv: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    y[i] = y[i] + sign * a[ai] * xv[i]


@wp.kernel
def k_xpby(y: wp.array(dtype=wp.vec3d), a: wp.array(dtype=wp.float64), ai: wp.int32,
           xv: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    y[i] = xv[i] + a[ai] * y[i]


@wp.kernel
def k_ratio(num: wp.array(dtype=wp.float64), den: wp.array(dtype=wp.float64),
            out: wp.array(dtype=wp.float64), oi: wp.int32):
    if den[0] > F0 or den[0] < F0:
        out[oi] = num[0] / den[0]
    else:
        out[oi] = F0


@wp.kernel
def k_copy(src: wp.array(dtype=wp.float64), si: wp.int32, dst: wp.array(dtype=wp.float64),
           di: wp.int32):
    dst[di] = src[si]


@wp.kernel
def k_lincomb(base: wp.array(dtype=wp.vec3d), step: wp.array(dtype=wp.vec3d),
              alpha: wp.float64, free: wp.array(dtype=wp.int32),
              out: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    if free[i] == 0:
        out[i] = base[i]
    else:
        out[i] = base[i] + alpha * step[i]


@wp.kernel
def k_zero3(a: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    a[i] = wp.vec3d(F0, F0, F0)


@wp.kernel
def k_set_target(x0: wp.array(dtype=wp.vec3d), v0: wp.array(dtype=wp.vec3d),
                 m: wp.array(dtype=wp.float64), dt: wp.float64, damp: wp.float64,
                 tgt: wp.array(dtype=wp.vec3d), dm: wp.array(dtype=wp.float64)):
    i = wp.tid()
    tgt[i] = x0[i] + (dt / (F1 + dt * damp)) * v0[i]
    dm[i] = m[i] * (F1 + dt * damp) / (dt * dt)


@wp.kernel
def k_velocity(x: wp.array(dtype=wp.vec3d), x0: wp.array(dtype=wp.vec3d), dt: wp.float64,
               free: wp.array(dtype=wp.int32), v: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    if free[i] == 0:
        v[i] = wp.vec3d(F0, F0, F0)
    else:
        v[i] = (x[i] - x0[i]) / dt


@wp.kernel
def k_advance(x0: wp.array(dtype=wp.vec3d), v: wp.array(dtype=wp.vec3d), dt: wp.float64,
              free: wp.array(dtype=wp.int32), x: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    if free[i] == 1:
        x[i] = x0[i] + dt * v[i]


@wp.kernel
def k_to_f32(x: wp.array(dtype=wp.vec3d), out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    out[i] = wp.vec3(wp.float32(x[i][0]), wp.float32(x[i][1]), wp.float32(x[i][2]))


@wp.kernel
def k_kinetic(v: wp.array(dtype=wp.vec3d), m: wp.array(dtype=wp.float64),
              out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    out[i] = FH * m[i] * wp.dot(v[i], v[i])


@wp.kernel
def k_speed(v: wp.array(dtype=wp.vec3d), out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    out[i] = wp.length(v[i])


# ---------------------------------------------------------------- block-Jacobi preconditioner


@wp.kernel
def k_tri_diag3(x: wp.array(dtype=wp.vec3d), t0: wp.array(dtype=wp.int32),
                t1: wp.array(dtype=wp.int32), t2: wp.array(dtype=wp.int32),
                tb: wp.array(dtype=wp.vec3d), tc: wp.array(dtype=wp.vec3d),
                ta: wp.array(dtype=wp.float64), Eh: wp.float64, nu: wp.float64,
                acc: wp.array2d(dtype=wp.int64)):
    """Per-node 3x3 diagonal block of the triangle tangent (9 int64 lanes per node)."""
    t = wp.tid()
    i0 = t0[t]
    i1 = t1[t]
    i2 = t2[t]
    b = tb[t]
    c = tc[t]
    p0 = x[i0]
    wu = b[1] * (x[i1] - p0) + b[2] * (x[i2] - p0)
    wv = c[1] * (x[i1] - p0) + c[2] * (x[i2] - p0)
    st = tri_state(wu, wv, Eh, nu)
    if st.br == 0:
        return
    a = ta[t]
    for vtx in range(3):
        bv = b[vtx]
        cv = c[vtx]
        idx = i0
        if vtx == 1:
            idx = i1
        if vtx == 2:
            idx = i2
        for j in range(3):
            dE11 = bv * wu[j]
            dE22 = cv * wv[j]
            dE12 = FH * (bv * wv[j] + cv * wu[j])
            dS = tri_dS(st, dE11, dE12, dE22, Eh, nu)
            for i in range(3):
                geo = F0
                if i == j:
                    geo = bv * st.S11 + cv * st.S12
                g0 = geo + wu[i] * dS[0] + wv[i] * dS[1]
                geo2 = F0
                if i == j:
                    geo2 = bv * st.S12 + cv * st.S22
                g1 = geo2 + wu[i] * dS[1] + wv[i] * dS[2]
                wp.atomic_add(acc, idx, i * 3 + j, wp.int64(a * (bv * g0 + cv * g1) * FSCALE))


@wp.kernel
def k_hinge_diag3(x: wp.array(dtype=wp.vec3d), v1: wp.array(dtype=wp.int32),
                  v2: wp.array(dtype=wp.int32), v3: wp.array(dtype=wp.int32),
                  v4: wp.array(dtype=wp.int32), hw: wp.array(dtype=wp.float64),
                  B: wp.float64, acc: wp.array2d(dtype=wp.int64)):
    h = wp.tid()
    i1 = v1[h]
    i2 = v2[h]
    i3 = v3[h]
    i4 = v4[h]
    p1 = x[i1]
    p2 = x[i2]
    p3 = x[i3]
    p4 = x[i4]
    e = p2 - p1
    le = wp.length(e)
    if le < wp.float64(1.0e-7):
        return
    n1 = wp.cross(e, p3 - p1)
    l1 = wp.length(n1)
    if l1 < wp.float64(1.0e-7):
        return
    n2 = wp.cross(p4 - p1, e)
    l2 = wp.length(n2)
    if l2 < wp.float64(1.0e-7):
        return
    g3 = (n1 / l1) * (FM1 * le / l1)
    g4 = (n2 / l2) * (FM1 * le / l2)
    al1 = wp.dot(p2 - p3, e) / (le * le)
    al2 = wp.dot(p3 - p1, e) / (le * le)
    be1 = wp.dot(p2 - p4, e) / (le * le)
    be2 = wp.dot(p4 - p1, e) / (le * le)
    g1 = FM1 * al1 * g3 - be1 * g4
    g2 = FM1 * al2 * g3 - be2 * g4
    k = B * hw[h]
    for i in range(3):
        for j in range(3):
            wp.atomic_add(acc, i1, i * 3 + j, wp.int64(k * g1[i] * g1[j] * FSCALE))
            wp.atomic_add(acc, i2, i * 3 + j, wp.int64(k * g2[i] * g2[j] * FSCALE))
            wp.atomic_add(acc, i3, i * 3 + j, wp.int64(k * g3[i] * g3[j] * FSCALE))
            wp.atomic_add(acc, i4, i * 3 + j, wp.int64(k * g4[i] * g4[j] * FSCALE))


@wp.kernel
def k_precond3(acc: wp.array2d(dtype=wp.int64), dm: wp.array(dtype=wp.float64),
               free: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.mat33d)):
    i = wp.tid()
    if free[i] == 0:
        out[i] = wp.mat33d(F0, F0, F0, F0, F0, F0, F0, F0, F0)
        return
    A = wp.mat33d()
    for a in range(3):
        for b in range(3):
            A[a, b] = wp.float64(acc[i, a * 3 + b]) / FSCALE
    A = FH * (A + wp.transpose(A))
    for a in range(3):
        A[a, a] = A[a, a] + dm[i]
    det = wp.determinant(A)
    tr = A[0, 0] + A[1, 1] + A[2, 2]
    if det > wp.float64(1.0e-24) * tr * tr * tr and tr > F0:
        out[i] = wp.inverse(A)
    else:
        out[i] = wp.mat33d(F1 / wp.max(A[0, 0], dm[i]), F0, F0,
                           F0, F1 / wp.max(A[1, 1], dm[i]), F0,
                           F0, F0, F1 / wp.max(A[2, 2], dm[i]))


@wp.kernel
def k_jacobi3(r: wp.array(dtype=wp.vec3d), pre: wp.array(dtype=wp.mat33d),
              out: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    out[i] = pre[i] * r[i]


@wp.kernel
def k_lumped_comp(minv: wp.array(dtype=wp.float64), dt: wp.float64, damp: wp.float64,
                  out: wp.array(dtype=wp.mat33d)):
    """Compliance of the implicit system WITHOUT the elastic coupling: A_kk = m(1+dt*damp)/dt^2.
    This is the lumped-mass collision response (the contract ncp_gpu.py is verified against)."""
    i = wp.tid()
    c = dt * dt * minv[i] / (F1 + dt * damp)
    out[i] = wp.mat33d(c, F0, F0, F0, c, F0, F0, F0, c)


@wp.kernel
def k_step_estimate(pre: wp.array(dtype=wp.mat33d), r: wp.array(dtype=wp.vec3d),
                    out: wp.array(dtype=wp.float64)):
    """|A_kk^-1 r_k| with A_kk the diagonal block of the implicit tangent INCLUDING elasticity:
    the position correction the residual implies if the neighbours were held fixed. Reported next
    to the residual because the membrane tangent is ~10^3 stiffer than gravity, so a force
    residual alone does not say how far the state is from the implicit solution. It is a lower
    bound: off-diagonal coupling adds compliance."""
    i = wp.tid()
    out[i] = wp.length(pre[i] * r[i])
