"""U112 round 6 setup: KES-FB2 data, the SI conversion, and the per-hinge yield angle.

FABRIC DATA.  B and 2HB are the MEASURED KES values for the same three fabrics the reference set
comes from: Matsudaira Mitsuo, Masuda Tomoe, Wada Minami, Yokura Hiroko (2015), "Shape Factor of
Flared Skirts Compared with That of Circular Fabrics", Journal of Textile Engineering 61(6)
69-73, DOI 10.4188/jte.61.69, Table 1 ("Mechanical parameters related to fabric drapability
measured by KES-system", KES-FB2 for B and 2HB) and Table 2 (drape coefficients).  They are read
from the published table, which carries
B_gf, 2HB_gf, G_gf, W_mg, T_mm, DC_pub, nodes_pub for A, B, C and F; round 1-5 used every column
except 2HB.  No value here is invented or taken from another fabric.

UNITS.  1 gf = 9.80665e-3 N.
    B   [gf*cm^2/cm] -> N*m   : *9.80665e-3 * 1e-4 / 1e-2 = *9.80665e-5
    2HB [gf*cm/cm]   -> N*m/m : *9.80665e-3 * 1e-2 / 1e-2 = *9.80665e-3
    W   [mg/cm^2]    -> kg/m^2: *1e-2
Residual curvature of the Kawabata bilinear loop, unloading to zero moment:
    kappa_res = HB/B = 2HB/(2B)   [1/m],   HB = 2HB/2.
"""
import numpy as np

GF = 9.80665e-3

# id: (B_gf*cm^2/cm, 2HB_gf*cm/cm, W_mg/cm^2, T_mm, DC_pub_%, nodes_pub)
FABRICS = {
    'A': (0.118, 0.124, 12.38, 0.497, 82.9, 2),   # Cotton Broad, 100% cotton plain weave
    'B': (0.067, 0.033, 16.29, 0.407, 49.8, 5),   # Wool Tropical, 100% wool plain weave
    'F': (0.019, 0.007, 10.43, 0.226, 28.3, 7),   # Polyester Faille
    'C': (0.077, 0.029, 11.65, 0.340, 50.7, 5),   # Polyester Tropical (reference only)
}
SOURCE = ('Matsudaira, Masuda, Wada, Yokura 2015, J. Textile Eng. 61(6) 69-73, Table 1 (KES-FB2 '
          'B and 2HB) and Table 2 (DC), via the published table')


def fabric_si(fid, nu=0.3):
    """SI parameters of fabric `fid`, including the bending hysteresis."""
    b_gf, twohb_gf, w_mg, t_mm, dc_pub, nodes_pub = FABRICS[fid]
    B = b_gf * GF * 1e-2            # N*m   (gf*cm^2/cm)
    twoHB = twohb_gf * GF           # N*m/m (gf*cm/cm)
    HB = 0.5 * twoHB
    W = w_mg * 1e-2                 # kg/m^2
    h = t_mm * 1e-3                 # m
    return dict(fabric=fid, B_gf=b_gf, twoHB_gf=twohb_gf, B_Nm=B, twoHB_Npm=twoHB, HB_Npm=HB,
                kappa_res_1pm=HB / B, R_res_m=B / HB, W_kgm2=W, h_m=h,
                Eh_Npm=12 * (1 - nu ** 2) * B / h ** 2, DC_pub=dc_pub, nodes_pub=nodes_pub,
                source=SOURCE)


def hinge_angles(x, d):
    """The dihedral angle of every hinge, exactly as energies.compute_hinge_forces_kernel
    computes it (atan2 of the sine on the edge against the clamped cosine)."""
    x = np.asarray(x, np.float64)
    p1 = x[np.asarray(d['h_v1'], np.int64)]
    p2 = x[np.asarray(d['h_v2'], np.int64)]
    p3 = x[np.asarray(d['h_v3'], np.int64)]
    p4 = x[np.asarray(d['h_v4'], np.int64)]
    e = p2 - p1
    le = np.linalg.norm(e, axis=1)
    n1 = np.cross(e, p3 - p1)
    n2 = np.cross(p4 - p1, e)
    l1 = np.linalg.norm(n1, axis=1)
    l2 = np.linalg.norm(n2, axis=1)
    u1 = n1 / l1[:, None]
    u2 = n2 / l2[:, None]
    ct = np.clip(np.einsum('ij,ij->i', u1, u2), -1., 1.)
    st = np.einsum('ij,ij->i', np.cross(u1, u2), e / le[:, None])
    return np.arctan2(st, ct), le, l1, l2


def plane_axes(verts):
    """The two in-plane axes and the out-of-plane axis of a flat rest mesh."""
    ext = verts.max(0) - verts.min(0)
    up = int(np.argmin(ext))
    ax = [k for k in range(3) if k != up]
    return ax, up, float(ext[up])


def hinge_bend_length(d, amp=1e-5):
    """ell_h = dtheta_h/dkappa for a cylindrical bend whose curvature direction is the hinge's own
    in-plane normal, evaluated in the rest mesh by a central difference.

    The hinge then yields exactly when the fabric curvature ACROSS it reaches kappa_res, which
    makes the yield moment M_y = B*w_h*kappa_res*ell_h independent of how the discrete hinge
    weight w_h relates to the continuum B.  For the strip mesh the hinges perpendicular to the
    bend get ell_h = dx and the hinges along it and the diagonals get their own dx/dz, so a
    cylindrical release sticks at kappa_res on every hinge that carries angle."""
    v = np.asarray(d['verts'], np.float64)
    ax, up, flat = plane_axes(v)
    idx = [np.asarray(d[k], np.int64) for k in ('h_v1', 'h_v2', 'h_v3', 'h_v4')]
    P = [v[i] for i in idx]
    e2 = (P[1] - P[0])[:, ax]
    le2 = np.linalg.norm(e2, axis=1)
    t = e2 / le2[:, None]
    nh = np.stack([-t[:, 1], t[:, 0]], 1)            # in-plane normal of the hinge
    cen = sum(p[:, ax] for p in P) / 4.0
    u = [np.einsum('ij,ij->i', p[:, ax] - cen, nh) for p in P]
    # span (h1+h2) in the rest mesh sets the amplitude so that theta ~ amp rad
    th0, le, l1, l2 = hinge_angles(v, d)
    span = (l1 + l2) / le
    kap = amp / np.maximum(span, 1e-12)
    out = []
    for sgn in (1.0, -1.0):
        y = [0.5 * sgn * kap * ui ** 2 for ui in u]
        q = np.empty((4, len(le), 3))
        for j in range(4):
            q[j] = v[idx[j]]
            q[j][:, up] = q[j][:, up] + y[j]
        # evaluate the dihedral on the deformed 4-vertex patches directly
        p1, p2, p3, p4 = q
        ee = p2 - p1
        lee = np.linalg.norm(ee, axis=1)
        n1 = np.cross(ee, p3 - p1)
        n2 = np.cross(p4 - p1, ee)
        u1 = n1 / np.linalg.norm(n1, axis=1)[:, None]
        u2 = n2 / np.linalg.norm(n2, axis=1)[:, None]
        ct = np.clip(np.einsum('ij,ij->i', u1, u2), -1., 1.)
        st = np.einsum('ij,ij->i', np.cross(u1, u2), ee / lee[:, None])
        out.append(np.arctan2(st, ct))
    ell = np.abs((out[0] - out[1]) / (2.0 * kap))
    return ell, dict(flat_extent_m=flat, up_axis=up, span_min=float(span.min()),
                     span_max=float(span.max()), ell_min=float(ell.min()),
                     ell_max=float(ell.max()), ell_median=float(np.median(ell)),
                     amp_rad=amp)


def bending_energy_ratio(d, kappa=1.0, axis=0):
    """Diagnostic: discrete hinge energy of a cylindrical bend over the continuum 0.5*B*kappa^2*A
    (per unit B).  Says how the Grinspun weight 1.5|e|^2/(A1+A2) maps onto the continuum on THIS
    mesh; the cantilever gate is the calibrated statement, this is the local one."""
    v = np.asarray(d['verts'], np.float64).copy()
    ax, up, _ = plane_axes(v)
    a = ax[axis]
    c = 0.5 * (v[:, a].max() + v[:, a].min())
    x = v.copy()
    x[:, up] = x[:, up] + 0.5 * kappa * (v[:, a] - c) ** 2
    th, le, l1, l2 = hinge_angles(x, d)
    w = np.asarray(d['h_w'], np.float64)
    e_disc = float((0.5 * w * th ** 2).sum())
    area = float(np.asarray(d['tri_area'], np.float64).sum())
    e_cont = 0.5 * kappa ** 2 * area
    return e_disc / e_cont, e_disc, e_cont


def cantilever_facit(W, g, L, B, twoHB=0.0):
    """Tip deflection of a self-weight cantilever strip per unit width.

    Elastic (twoHB = 0):            y = W g L^4 / (8 B)                 (the round 1-5 facit)
    With the Coulomb bending moment HB = 2HB/2 opposing the bending, small deflection:
        B kappa(u) = W g u^2/2 - HB  for u > u_c = sqrt(2 HB/(W g)), zero closer to the tip,
        y = (W g / (8 B)) * (L^2 - 2HB/(W g))^2,   valid for L^2 >= 2HB/(W g).
    Both are integrals of the same beam equation; the second reduces to the first at 2HB = 0."""
    if twoHB <= 0.0:
        return W * g * L ** 4 / (8 * B)
    c = twoHB / (W * g)
    if L * L <= c:
        return 0.0
    return W * g / (8 * B) * (L * L - c) ** 2


def hinge_guard_count(x, d, tol=1e-7):
    """How many hinges the U93 hinge kernels SKIP because a guard fires.

    energies.compute_hinge_forces_kernel (and the r4/r6 energy and tangent kernels copied from
    it) return without contributing when |e| < 1e-7, |n1| < 1e-7 or |n2| < 1e-7, with
    |n_i| = 2 * area_i.  The guard is an absolute area, so on the Cusick mesh it sits at 2.9 % of
    the rest triangle area at dx 2 mm and at 11.6 % at dx 1 mm: a squashed triangle silently
    loses its bending resistance.  Counted here, never changed."""
    x = np.asarray(x, np.float64)
    p1 = x[np.asarray(d['h_v1'], np.int64)]
    p2 = x[np.asarray(d['h_v2'], np.int64)]
    p3 = x[np.asarray(d['h_v3'], np.int64)]
    p4 = x[np.asarray(d['h_v4'], np.int64)]
    e = p2 - p1
    le = np.linalg.norm(e, axis=1)
    l1 = np.linalg.norm(np.cross(e, p3 - p1), axis=1)
    l2 = np.linalg.norm(np.cross(p4 - p1, e), axis=1)
    drop = (le < tol) | (l1 < tol) | (l2 < tol)
    return int(drop.sum()), float(min(l1.min(), l2.min()))
