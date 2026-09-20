import numpy as np
import math
import warp as wp
FORCE_SCALE=1e12
wp.set_module_options({"fuse_fp":False,"fast_math":False})
@wp.kernel
def compute_hinge_forces_kernel(positions: wp.array(dtype=wp.vec3d), hinge_v1: wp.array(dtype=wp.int32), hinge_v2: wp.array(dtype=wp.int32), hinge_v3: wp.array(dtype=wp.int32), hinge_v4: wp.array(dtype=wp.int32), hinge_weights: wp.array(dtype=wp.float64), hinge_rest_angles: wp.array(dtype=wp.float64), k_bend: wp.float64, force_accum: wp.array2d(dtype=wp.int64)):
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
    if len_e < wp.float64(1e-07):
        return
    e_hat = e / len_e
    n1_raw = wp.cross(e, p3 - p1)
    len_n1 = wp.length(n1_raw)
    if len_n1 < wp.float64(1e-07):
        return
    n1 = n1_raw / len_n1
    n2_raw = wp.cross(p4 - p1, e)
    len_n2 = wp.length(n2_raw)
    if len_n2 < wp.float64(1e-07):
        return
    n2 = n2_raw / len_n2
    cos_t = wp.clamp(wp.dot(n1, n2), -wp.float64(1.0), wp.float64(1.0))
    sin_t = wp.dot(wp.cross(n1, n2), e_hat)
    theta = wp.atan2(sin_t, cos_t)
    d_theta = theta - hinge_rest_angles[h]
    weight = hinge_weights[h]
    coeff = k_bend * weight * d_theta
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
    scale = wp.float64(FORCE_SCALE)
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

@wp.kernel
def compute_pipkin_membrane_forces_kernel(positions: wp.array(dtype=wp.vec3d), tri_v0: wp.array(dtype=wp.int32), tri_v1: wp.array(dtype=wp.int32), tri_v2: wp.array(dtype=wp.int32), tri_b: wp.array(dtype=wp.vec3d), tri_c: wp.array(dtype=wp.vec3d), tri_area: wp.array(dtype=wp.float64), Eh: wp.float64, nu: wp.float64, coeff_taut: wp.float64, force_accum: wp.array2d(dtype=wp.int64)):
    t = wp.tid()
    i0 = tri_v0[t]
    i1 = tri_v1[t]
    i2 = tri_v2[t]
    p0 = positions[i0]
    p1 = positions[i1]
    p2 = positions[i2]
    b = tri_b[t]
    c = tri_c[t]
    area = tri_area[t]
    w_u = b[1] * (p1 - p0) + b[2] * (p2 - p0)
    w_v = c[1] * (p1 - p0) + c[2] * (p2 - p0)
    C11 = wp.dot(w_u, w_u)
    C12 = wp.dot(w_u, w_v)
    C22 = wp.dot(w_v, w_v)
    E11 = wp.float64(0.5) * (C11 - wp.float64(1.0))
    E12 = wp.float64(0.5) * C12
    E22 = wp.float64(0.5) * (C22 - wp.float64(1.0))
    trE = E11 + E22
    diff = E11 - E22
    disc = wp.sqrt(diff * diff + wp.float64(4.0) * E12 * E12)
    eps1 = wp.float64(0.5) * (trE + disc)
    eps2 = wp.float64(0.5) * (trE - disc)
    if eps1 <= wp.float64(0.0):
        return
    S11 = wp.float64(wp.float64(0.0))
    S12 = wp.float64(wp.float64(0.0))
    S22 = wp.float64(wp.float64(0.0))
    if eps2 + nu * eps1 > wp.float64(0.0):
        S11 = coeff_taut * (E11 + nu * E22)
        S12 = coeff_taut * (wp.float64(1.0) - nu) * E12
        S22 = coeff_taut * (E22 + nu * E11)
    else:
        s1 = Eh * eps1
        if disc > wp.float64(1e-12):
            inv_disc = wp.float64(1.0) / disc
            m11 = wp.float64(0.5) + wp.float64(0.5) * diff * inv_disc
            m12 = E12 * inv_disc
            m22 = wp.float64(0.5) - wp.float64(0.5) * diff * inv_disc
        else:
            m11 = wp.float64(0.5)
            m12 = wp.float64(0.0)
            m22 = wp.float64(0.5)
        S11 = s1 * m11
        S12 = s1 * m12
        S22 = s1 * m22
    FS0 = w_u * S11 + w_v * S12
    FS1 = w_u * S12 + w_v * S22
    f0 = -area * (b[0] * FS0 + c[0] * FS1)
    f1 = -area * (b[1] * FS0 + c[1] * FS1)
    f2 = -area * (b[2] * FS0 + c[2] * FS1)
    scale = wp.float64(FORCE_SCALE)
    wp.atomic_add(force_accum, i0, 0, wp.int64(f0[0] * scale))
    wp.atomic_add(force_accum, i0, 1, wp.int64(f0[1] * scale))
    wp.atomic_add(force_accum, i0, 2, wp.int64(f0[2] * scale))
    wp.atomic_add(force_accum, i1, 0, wp.int64(f1[0] * scale))
    wp.atomic_add(force_accum, i1, 1, wp.int64(f1[1] * scale))
    wp.atomic_add(force_accum, i1, 2, wp.int64(f1[2] * scale))
    wp.atomic_add(force_accum, i2, 0, wp.int64(f2[0] * scale))
    wp.atomic_add(force_accum, i2, 1, wp.int64(f2[1] * scale))
    wp.atomic_add(force_accum, i2, 2, wp.int64(f2[2] * scale))
