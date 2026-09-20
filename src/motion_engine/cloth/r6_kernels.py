"""U112 round 6 kernels: bending hysteresis (KES-FB2 2HB) as a state-dependent hinge energy.

MODEL.  KES-FB2 measures, per fabric, the bending rigidity B [gf*cm^2/cm] (slope of the
moment-curvature line per unit width) and the bending hysteresis 2HB [gf*cm/cm] (the WIDTH of
the moment loop at kappa = +-0.5 cm^-1).  Kawabata's bilinear idealisation of the loop is two
parallel lines of slope B separated by 2HB:

    M(kappa) = B*kappa +- HB,      HB = 2HB/2,

with the branch sign set by the direction of the last curvature change.  Implemented here as a
Jenkin element (spring k_p in series with a Coulomb slider of yield moment M_y = HB per unit
width) in PARALLEL with the elastic spring B, which is the standard rate-independent
elastoplastic reading of that loop:

    M(theta) = K_e*(theta - theta_rest) + clamp(K_p*(theta - theta_p), -M_y, +M_y)

with K_e = B*w_h the hinge's elastic stiffness (w_h = Grinspun weight 1.5|e|^2/(A1+A2), the same
one the U93 hinge energy is verified with), K_p = ratio*K_e the pre-slip stiffness, and theta_p
the per-hinge slip offset = the last reversal point.  On monotone bending the slider slips and
the branch slope is K_e with the offset +-M_y; on reversal the slider sticks, the moment falls
by 2*M_y at slope K_e+K_p, and the loop width is exactly 2*M_y = 2HB per unit width.

YIELD ANGLE.  M_y is set per hinge from the measured 2HB as M_y = K_e * theta_y with
theta_y = kappa_res * ell_h, kappa_res = HB/B = 2HB/(2B) [1/m] and ell_h = dtheta_h/dkappa for a
cylindrical bend whose curvature direction is the hinge's own in-plane normal (computed in the
rest mesh, r6_setup.hinge_bend_length).  A hinge therefore yields exactly when the fabric
curvature across it reaches kappa_res, so a strip bent and released sticks at kappa_res, which
is the analytic residual 2HB/(2B).

INCREMENTAL POTENTIAL.  theta_p is frozen at the value it had at the START of the step, so the
step solves a convex incremental problem with the C^1 potential

    Psi(theta) = 1/2 K_e (theta-theta_rest)^2 + huber(theta-theta_p),
    huber(u)   = 1/2 K_p u^2                    if |u| <= u_y = M_y/K_p
               = M_y|u| - 1/2 M_y u_y           otherwise,

whose derivative is the moment law above and whose second derivative is K_e+K_p (stick) or K_e
(slip) -- both PSD, so the Gauss-Newton hinge tangent stays PSD.  theta_p is returned-mapped once
at the end of the step by k_hyst_update, one thread per hinge, no atomics: the state is
bit-deterministic and the forces still accumulate in the int64 fixed point of energies.py.

AREA FLOOR (round-5 defect 2).  The hinge gradient carries 1/h_i with h_i = |n_i|/|e| the
triangle height over the hinge edge, so a collapsing triangle makes the PSD tangent blow up
(round 5 measured triangles down to 5.8e-10 m^2 against a nominal 1.73e-6 m^2).  The tangent
kernels here floor |n_i| = 2*A_i at floor_frac of its REST value.  The floor is applied only in
the tangent (matvec/diagonal), never in the force or the energy, so the gradient stays the exact
gradient of the energy and the line search is unaffected; when the floor does not bind the
arithmetic is bit-identical to round 4/5.

GRAVITY SCALE.  k_residual_g/k_node_energy_g take g as an argument instead of the constant
9.80665 so the release test (gate 2) can run at zero gravity; at g = 9.80665 the expressions are
bit-identical to r4_kernels.k_residual/k_node_energy.
"""
import warp as wp

wp.set_module_options({'fuse_fp': False, 'fast_math': False})

F0 = wp.constant(wp.float64(0.0))
F1 = wp.constant(wp.float64(1.0))
FH = wp.constant(wp.float64(0.5))
FM1 = wp.constant(wp.float64(-1.0))
FSCALE = wp.constant(wp.float64(1.0e12))
GRAV = wp.constant(wp.float64(9.80665))
HTINY = wp.constant(wp.float64(1.0e-07))


@wp.func
def hyst_moment(u: wp.float64, Kp: wp.float64, My: wp.float64):
    """clamp(Kp*u, -My, +My): the Jenkin slider moment."""
    m = Kp * u
    if m > My:
        m = My
    if m < -My:
        m = -My
    return m


# ---------------------------------------------------------------- force (gradient)


@wp.kernel
def k_hinge_force6(positions: wp.array(dtype=wp.vec3d), hinge_v1: wp.array(dtype=wp.int32),
                   hinge_v2: wp.array(dtype=wp.int32), hinge_v3: wp.array(dtype=wp.int32),
                   hinge_v4: wp.array(dtype=wp.int32), hinge_weights: wp.array(dtype=wp.float64),
                   hinge_rest_angles: wp.array(dtype=wp.float64), k_bend: wp.float64,
                   theta_p: wp.array(dtype=wp.float64), theta_y: wp.array(dtype=wp.float64),
                   ratio: wp.float64, force_accum: wp.array2d(dtype=wp.int64)):
    """energies.compute_hinge_forces_kernel with the Jenkin moment added to coeff.

    With theta_y[h] == 0 the branch is skipped and every floating-point operation is the one in
    energies.py, so the no-hysteresis run is bit-identical to round 5."""
    h = wp.tid()
    i1 = hinge_v1[h]
    i2 = hinge_v2[h]
    i3 = hinge_v3[h]
    i4 = hinge_v4[h]
    p1 = positions[i1]
    p2 = positions[i2]
    p3 = positions[i3]
    p4 = positions[i4]
    e = p2 - p1
    len_e = wp.length(e)
    if len_e < HTINY:
        return
    e_hat = e / len_e
    n1_raw = wp.cross(e, p3 - p1)
    len_n1 = wp.length(n1_raw)
    if len_n1 < HTINY:
        return
    n1 = n1_raw / len_n1
    n2_raw = wp.cross(p4 - p1, e)
    len_n2 = wp.length(n2_raw)
    if len_n2 < HTINY:
        return
    n2 = n2_raw / len_n2
    cos_t = wp.clamp(wp.dot(n1, n2), -wp.float64(1.0), wp.float64(1.0))
    sin_t = wp.dot(wp.cross(n1, n2), e_hat)
    theta = wp.atan2(sin_t, cos_t)
    d_theta = theta - hinge_rest_angles[h]
    weight = hinge_weights[h]
    coeff = k_bend * weight * d_theta
    ty = theta_y[h]
    if ty > F0:
        Ke = k_bend * weight
        coeff = coeff + hyst_moment(d_theta - theta_p[h], ratio * Ke, Ke * ty)
    h1 = len_n1 / len_e
    h2 = len_n2 / len_e
    g3 = -n1 / h1
    g4 = -n2 / h2
    alpha1 = wp.dot(p2 - p3, e) / (len_e * len_e)
    alpha2 = wp.dot(p3 - p1, e) / (len_e * len_e)
    beta1 = wp.dot(p2 - p4, e) / (len_e * len_e)
    beta2 = wp.dot(p4 - p1, e) / (len_e * len_e)
    g1 = -alpha1 * g3 - beta1 * g4
    g2 = -alpha2 * g3 - beta2 * g4
    f1 = -coeff * g1
    f2 = -coeff * g2
    f3 = -coeff * g3
    f4 = -coeff * g4
    scale = wp.float64(FSCALE)
    wp.atomic_add(force_accum, i1, 0, wp.int64(f1[0] * scale))
    wp.atomic_add(force_accum, i1, 1, wp.int64(f1[1] * scale))
    wp.atomic_add(force_accum, i1, 2, wp.int64(f1[2] * scale))
    wp.atomic_add(force_accum, i2, 0, wp.int64(f2[0] * scale))
    wp.atomic_add(force_accum, i2, 1, wp.int64(f2[1] * scale))
    wp.atomic_add(force_accum, i2, 2, wp.int64(f2[2] * scale))
    wp.atomic_add(force_accum, i3, 0, wp.int64(f3[0] * scale))
    wp.atomic_add(force_accum, i3, 1, wp.int64(f3[1] * scale))
    wp.atomic_add(force_accum, i3, 2, wp.int64(f3[2] * scale))
    wp.atomic_add(force_accum, i4, 0, wp.int64(f4[0] * scale))
    wp.atomic_add(force_accum, i4, 1, wp.int64(f4[1] * scale))
    wp.atomic_add(force_accum, i4, 2, wp.int64(f4[2] * scale))


# ---------------------------------------------------------------- energy


@wp.kernel
def k_hinge_energy6(x: wp.array(dtype=wp.vec3d), v1: wp.array(dtype=wp.int32),
                    v2: wp.array(dtype=wp.int32), v3: wp.array(dtype=wp.int32),
                    v4: wp.array(dtype=wp.int32), hw: wp.array(dtype=wp.float64),
                    h0: wp.array(dtype=wp.float64), B: wp.float64,
                    theta_p: wp.array(dtype=wp.float64), theta_y: wp.array(dtype=wp.float64),
                    ratio: wp.float64, out: wp.array(dtype=wp.float64)):
    h = wp.tid()
    out[h] = F0
    p1 = x[v1[h]]
    p2 = x[v2[h]]
    p3 = x[v3[h]]
    p4 = x[v4[h]]
    e = p2 - p1
    le = wp.length(e)
    if le < HTINY:
        return
    n1 = wp.cross(e, p3 - p1)
    l1 = wp.length(n1)
    if l1 < HTINY:
        return
    n2 = wp.cross(p4 - p1, e)
    l2 = wp.length(n2)
    if l2 < HTINY:
        return
    n1 = n1 / l1
    n2 = n2 / l2
    ct = wp.clamp(wp.dot(n1, n2), -wp.float64(1.0), wp.float64(1.0))
    stn = wp.dot(wp.cross(n1, n2), e / le)
    th = wp.atan2(stn, ct) - h0[h]
    en = FH * B * hw[h] * th * th
    ty = theta_y[h]
    if ty > F0:
        Ke = B * hw[h]
        Kp = ratio * Ke
        My = Ke * ty
        uy = My / Kp
        u = th - theta_p[h]
        au = wp.abs(u)
        if au <= uy:
            en = en + FH * Kp * u * u
        else:
            en = en + My * au - FH * My * uy
    out[h] = en


# ---------------------------------------------------------------- tangent


@wp.kernel
def k_hinge_matvec6(x: wp.array(dtype=wp.vec3d), dp: wp.array(dtype=wp.vec3d),
                    v1: wp.array(dtype=wp.int32), v2: wp.array(dtype=wp.int32),
                    v3: wp.array(dtype=wp.int32), v4: wp.array(dtype=wp.int32),
                    hw: wp.array(dtype=wp.float64), h0: wp.array(dtype=wp.float64),
                    B: wp.float64, theta_p: wp.array(dtype=wp.float64),
                    theta_y: wp.array(dtype=wp.float64), ratio: wp.float64,
                    l1f: wp.array(dtype=wp.float64), l2f: wp.array(dtype=wp.float64),
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
    if le < HTINY:
        return
    n1 = wp.cross(e, p3 - p1)
    l1 = wp.length(n1)
    if l1 < HTINY:
        return
    n2 = wp.cross(p4 - p1, e)
    l2 = wp.length(n2)
    if l2 < HTINY:
        return
    l1c = wp.max(l1, l1f[h])
    l2c = wp.max(l2, l2f[h])
    g3 = (n1 / l1c) * (FM1 * le / l1c)
    g4 = (n2 / l2c) * (FM1 * le / l2c)
    al1 = wp.dot(p2 - p3, e) / (le * le)
    al2 = wp.dot(p3 - p1, e) / (le * le)
    be1 = wp.dot(p2 - p4, e) / (le * le)
    be2 = wp.dot(p4 - p1, e) / (le * le)
    g1 = FM1 * al1 * g3 - be1 * g4
    g2 = FM1 * al2 * g3 - be2 * g4
    kt = B * hw[h]
    ty = theta_y[h]
    if ty > F0:
        Ke = B * hw[h]
        Kp = ratio * Ke
        My = Ke * ty
        n1u = n1 / l1
        n2u = n2 / l2
        ct = wp.clamp(wp.dot(n1u, n2u), -wp.float64(1.0), wp.float64(1.0))
        stn = wp.dot(wp.cross(n1u, n2u), e / le)
        u = (wp.atan2(stn, ct) - h0[h]) - theta_p[h]
        if wp.abs(Kp * u) <= My:
            kt = kt + Kp
    s = wp.dot(g1, dp[i1]) + wp.dot(g2, dp[i2]) + wp.dot(g3, dp[i3]) + wp.dot(g4, dp[i4])
    k = kt * s
    for c in range(3):
        wp.atomic_add(acc, i1, c, wp.int64(k * g1[c] * FSCALE))
        wp.atomic_add(acc, i2, c, wp.int64(k * g2[c] * FSCALE))
        wp.atomic_add(acc, i3, c, wp.int64(k * g3[c] * FSCALE))
        wp.atomic_add(acc, i4, c, wp.int64(k * g4[c] * FSCALE))


@wp.kernel
def k_hinge_diag36(x: wp.array(dtype=wp.vec3d), v1: wp.array(dtype=wp.int32),
                   v2: wp.array(dtype=wp.int32), v3: wp.array(dtype=wp.int32),
                   v4: wp.array(dtype=wp.int32), hw: wp.array(dtype=wp.float64),
                   h0: wp.array(dtype=wp.float64), B: wp.float64,
                   theta_p: wp.array(dtype=wp.float64), theta_y: wp.array(dtype=wp.float64),
                   ratio: wp.float64, l1f: wp.array(dtype=wp.float64),
                   l2f: wp.array(dtype=wp.float64), acc: wp.array2d(dtype=wp.int64)):
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
    if le < HTINY:
        return
    n1 = wp.cross(e, p3 - p1)
    l1 = wp.length(n1)
    if l1 < HTINY:
        return
    n2 = wp.cross(p4 - p1, e)
    l2 = wp.length(n2)
    if l2 < HTINY:
        return
    l1c = wp.max(l1, l1f[h])
    l2c = wp.max(l2, l2f[h])
    g3 = (n1 / l1c) * (FM1 * le / l1c)
    g4 = (n2 / l2c) * (FM1 * le / l2c)
    al1 = wp.dot(p2 - p3, e) / (le * le)
    al2 = wp.dot(p3 - p1, e) / (le * le)
    be1 = wp.dot(p2 - p4, e) / (le * le)
    be2 = wp.dot(p4 - p1, e) / (le * le)
    g1 = FM1 * al1 * g3 - be1 * g4
    g2 = FM1 * al2 * g3 - be2 * g4
    k = B * hw[h]
    ty = theta_y[h]
    if ty > F0:
        Ke = B * hw[h]
        Kp = ratio * Ke
        My = Ke * ty
        n1u = n1 / l1
        n2u = n2 / l2
        ct = wp.clamp(wp.dot(n1u, n2u), -wp.float64(1.0), wp.float64(1.0))
        stn = wp.dot(wp.cross(n1u, n2u), e / le)
        u = (wp.atan2(stn, ct) - h0[h]) - theta_p[h]
        if wp.abs(Kp * u) <= My:
            k = k + Kp
    for i in range(3):
        for j in range(3):
            wp.atomic_add(acc, i1, i * 3 + j, wp.int64(k * g1[i] * g1[j] * FSCALE))
            wp.atomic_add(acc, i2, i * 3 + j, wp.int64(k * g2[i] * g2[j] * FSCALE))
            wp.atomic_add(acc, i3, i * 3 + j, wp.int64(k * g3[i] * g3[j] * FSCALE))
            wp.atomic_add(acc, i4, i * 3 + j, wp.int64(k * g4[i] * g4[j] * FSCALE))


# ---------------------------------------------------------------- state update


@wp.kernel
def k_hyst_update(x: wp.array(dtype=wp.vec3d), v1: wp.array(dtype=wp.int32),
                  v2: wp.array(dtype=wp.int32), v3: wp.array(dtype=wp.int32),
                  v4: wp.array(dtype=wp.int32), h0: wp.array(dtype=wp.float64),
                  theta_y: wp.array(dtype=wp.float64), ratio: wp.float64,
                  theta_p: wp.array(dtype=wp.float64), slip: wp.array(dtype=wp.int32)):
    """Return mapping of the slider: theta_p += (|u| - u_y)_+ * sign(u), u = theta - theta_p.

    One thread per hinge, one read and one write of theta_p[h]: no atomics, no thread-order
    dependence."""
    h = wp.tid()
    ty = theta_y[h]
    if ty <= F0:
        return
    p1 = x[v1[h]]
    p2 = x[v2[h]]
    p3 = x[v3[h]]
    p4 = x[v4[h]]
    e = p2 - p1
    le = wp.length(e)
    if le < HTINY:
        return
    n1 = wp.cross(e, p3 - p1)
    l1 = wp.length(n1)
    if l1 < HTINY:
        return
    n2 = wp.cross(p4 - p1, e)
    l2 = wp.length(n2)
    if l2 < HTINY:
        return
    n1 = n1 / l1
    n2 = n2 / l2
    ct = wp.clamp(wp.dot(n1, n2), -wp.float64(1.0), wp.float64(1.0))
    stn = wp.dot(wp.cross(n1, n2), e / le)
    th = wp.atan2(stn, ct) - h0[h]
    uy = ty / ratio
    u = th - theta_p[h]
    au = wp.abs(u)
    if au > uy:
        s = F1
        if u < F0:
            s = FM1
        theta_p[h] = theta_p[h] + s * (au - uy)
        wp.atomic_add(slip, 0, 1)


@wp.kernel
def k_hinge_probe(x: wp.array(dtype=wp.vec3d), v1: wp.array(dtype=wp.int32),
                  v2: wp.array(dtype=wp.int32), v3: wp.array(dtype=wp.int32),
                  v4: wp.array(dtype=wp.int32), hw: wp.array(dtype=wp.float64),
                  h0: wp.array(dtype=wp.float64), B: wp.float64,
                  theta_p: wp.array(dtype=wp.float64), theta_y: wp.array(dtype=wp.float64),
                  ratio: wp.float64, th_out: wp.array(dtype=wp.float64),
                  m_out: wp.array(dtype=wp.float64)):
    """Diagnostic: the hinge angle and the total hinge moment K_e*(theta) + slider."""
    h = wp.tid()
    th_out[h] = F0
    m_out[h] = F0
    p1 = x[v1[h]]
    p2 = x[v2[h]]
    p3 = x[v3[h]]
    p4 = x[v4[h]]
    e = p2 - p1
    le = wp.length(e)
    if le < HTINY:
        return
    n1 = wp.cross(e, p3 - p1)
    l1 = wp.length(n1)
    if l1 < HTINY:
        return
    n2 = wp.cross(p4 - p1, e)
    l2 = wp.length(n2)
    if l2 < HTINY:
        return
    n1 = n1 / l1
    n2 = n2 / l2
    ct = wp.clamp(wp.dot(n1, n2), -wp.float64(1.0), wp.float64(1.0))
    stn = wp.dot(wp.cross(n1, n2), e / le)
    th = wp.atan2(stn, ct) - h0[h]
    Ke = B * hw[h]
    m = Ke * th
    ty = theta_y[h]
    if ty > F0:
        m = m + hyst_moment(th - theta_p[h], ratio * Ke, Ke * ty)
    th_out[h] = th
    m_out[h] = m


# ---------------------------------------------------------------- gravity-scaled


@wp.kernel
def k_residual_g(fint: wp.array2d(dtype=wp.int64), m: wp.array(dtype=wp.float64),
                 dm: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.vec3d),
                 tgt: wp.array(dtype=wp.vec3d), fc: wp.array(dtype=wp.vec3d),
                 free: wp.array(dtype=wp.int32), g: wp.float64,
                 r: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    if free[i] == 0:
        r[i] = wp.vec3d(F0, F0, F0)
        return
    f = wp.vec3d(wp.float64(fint[i, 0]), wp.float64(fint[i, 1]), wp.float64(fint[i, 2])) / FSCALE
    f = f + wp.vec3d(F0, FM1 * g * m[i], F0) + fc[i]
    r[i] = f - dm[i] * (x[i] - tgt[i])


@wp.kernel
def k_node_energy_g(x: wp.array(dtype=wp.vec3d), m: wp.array(dtype=wp.float64),
                    tgt: wp.array(dtype=wp.vec3d), dm: wp.array(dtype=wp.float64),
                    fc: wp.array(dtype=wp.vec3d), free: wp.array(dtype=wp.int32),
                    g: wp.float64, out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    if free[i] == 0:
        out[i] = F0
        return
    d = x[i] - tgt[i]
    out[i] = g * m[i] * x[i][1] + FH * dm[i] * wp.dot(d, d) - wp.dot(fc[i], x[i])
