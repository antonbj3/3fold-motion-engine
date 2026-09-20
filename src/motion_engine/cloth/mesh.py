import numpy as np
import math
import warp as wp
def build_strip_mesh(L=0.017, w=0.006, nx=17, nz=4, W_si=0.1629, orient="horizontal"):
    """Build rectangular strip mesh with triangles, shape functions, and hinges."""
    x_lin = np.linspace(0.0, L, nx + 1)
    z_lin = np.linspace(0.0, w, nz + 1)
    dx = x_lin[1] - x_lin[0]
    dz = z_lin[1] - z_lin[0]

    verts = []
    v_map = {}
    for i in range(nx + 1):
        for j in range(nz + 1):
            v_map[(i, j)] = len(verts)
            if orient == "horizontal":
                # Cantilever: along X, Z is width, Y is vertical
                verts.append([x_lin[i], 0.0, z_lin[j]])
            elif orient == "vertical":
                # Tension strip: along Y (0 to L), X is width, Z=0
                # Clamped at y = L, free at y = 0
                verts.append([z_lin[j], x_lin[i], 0.0])

    verts = np.array(verts, dtype=np.float32)
    n_verts = len(verts)

    tris = []
    edges = {}
    for i in range(nx):
        for j in range(nz):
            v0 = v_map[(i, j)]
            v1 = v_map[(i + 1, j)]
            v2 = v_map[(i + 1, j + 1)]
            v3 = v_map[(i, j + 1)]

            t1 = len(tris)
            tris.append((v0, v1, v2))
            t2 = len(tris)
            tris.append((v0, v2, v3))

            for e, opp in [((min(v0, v1), max(v0, v1)), v2),
                           ((min(v1, v2), max(v1, v2)), v0),
                           ((min(v2, v0), max(v2, v0)), v1)]:
                edges.setdefault(e, []).append((t1, opp))
            for e, opp in [((min(v0, v2), max(v0, v2)), v3),
                           ((min(v2, v3), max(v2, v3)), v0),
                           ((min(v3, v0), max(v3, v0)), v2)]:
                edges.setdefault(e, []).append((t2, opp))

    tris = np.array(tris, dtype=np.int32)
    n_tris = len(tris)

    # Triangle shape function derivatives b, c and area
    tri_b = np.zeros((n_tris, 3), dtype=np.float32)
    tri_c = np.zeros((n_tris, 3), dtype=np.float32)
    tri_area = np.zeros(n_tris, dtype=np.float32)
    v_areas = np.zeros(n_verts, dtype=np.float32)

    for t in range(n_tris):
        i0, i1, i2 = tris[t]
        if orient == "horizontal":
            u0, v0_coord = verts[i0, 0], verts[i0, 2]
            u1, v1_coord = verts[i1, 0], verts[i1, 2]
            u2, v2_coord = verts[i2, 0], verts[i2, 2]
        else:
            u0, v0_coord = verts[i0, 0], verts[i0, 1]
            u1, v1_coord = verts[i1, 0], verts[i1, 1]
            u2, v2_coord = verts[i2, 0], verts[i2, 1]

        det = (u1 - u0) * (v2_coord - v0_coord) - (u2 - u0) * (v1_coord - v0_coord)
        area = 0.5 * abs(det)
        tri_area[t] = area

        # Cyclic shape function derivatives
        b = np.array([v1_coord - v2_coord, v2_coord - v0_coord, v0_coord - v1_coord]) / det
        c = np.array([u2 - u1, u0 - u2, u1 - u0]) / det

        tri_b[t] = b
        tri_c[t] = c

        v_areas[i0] += area / 3.0
        v_areas[i1] += area / 3.0
        v_areas[i2] += area / 3.0

    masses = v_areas * float(W_si)

    # Hinges
    h_v1, h_v2, h_v3, h_v4, h_w, h_t0 = [], [], [], [], [], []
    for e, t_list in edges.items():
        if len(t_list) == 2:
            (t1, opp1), (t2, opp2) = t_list
            v1_e, v2_e = e
            p1 = verts[v1_e]
            p2 = verts[v2_e]
            p3 = verts[opp1]
            p4 = verts[opp2]

            e_vec = p2 - p1
            len_e = np.linalg.norm(e_vec)
            if len_e < 1e-7:
                continue

            n1 = np.cross(e_vec, p3 - p1)
            len_n1 = np.linalg.norm(n1)
            n2 = np.cross(p4 - p1, e_vec)
            len_n2 = np.linalg.norm(n2)
            if len_n1 < 1e-7 or len_n2 < 1e-7:
                continue

            A1 = 0.5 * len_n1
            A2 = 0.5 * len_n2
            # Grinspun: |e| / h_e = 1.5 * |e|^2 / (A1 + A2)
            w_h = 1.5 * (len_e ** 2) / (A1 + A2)

            h_v1.append(v1_e)
            h_v2.append(v2_e)
            h_v3.append(opp1)
            h_v4.append(opp2)
            h_w.append(w_h)
            h_t0.append(0.0)

    # Boundary conditions
    fixed_flags = np.zeros(n_verts, dtype=np.int32)
    if orient == "horizontal":
        # Clamp first column x = 0 (and x = dx for slope)
        for j in range(nz + 1):
            fixed_flags[v_map[(0, j)]] = 1
            fixed_flags[v_map[(1, j)]] = 1
        tip_indices = [v_map[(nx, j)] for j in range(nz + 1)]
        L_free = L - dx
    else:
        # Tension strip: clamp top edge y = L
        for j in range(nz + 1):
            fixed_flags[v_map[(nx, j)]] = 1
        tip_indices = [v_map[(0, j)] for j in range(nz + 1)]
        L_free = L

    return {
        "verts": verts,
        "tris": tris,
        "n_verts": n_verts,
        "n_tris": n_tris,
        "masses": masses,
        "fixed_flags": fixed_flags,
        "tip_indices": tip_indices,
        "L_free": L_free,
        "dx": dx,
        "dz": dz,
        "tri_b": tri_b,
        "tri_c": tri_c,
        "tri_area": tri_area,
        "h_v1": np.array(h_v1, dtype=np.int32),
        "h_v2": np.array(h_v2, dtype=np.int32),
        "h_v3": np.array(h_v3, dtype=np.int32),
        "h_v4": np.array(h_v4, dtype=np.int32),
        "h_w": np.array(h_w, dtype=np.float32),
        "h_t0": np.array(h_t0, dtype=np.float32),
        "n_hinges": len(h_w),
    }