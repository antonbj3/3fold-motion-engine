"""Vectorized circular cloth mesh and hinge builder using NumPy."""
import numpy as np
import time

def make_circular_cloth_mesh_fast(
    spacing: float = 0.002,
    radius: float = 0.15,
    centre = (0.25, 0.1605, 0.25),
):
    t0 = time.time()
    c = np.asarray(centre, dtype=np.float64)
    n_step = int(np.ceil(2.0 * radius / spacing))
    u = np.linspace(-radius, radius, n_step + 1)
    du = u[1] - u[0]
    
    gx, gz = np.meshgrid(u, u, indexing="ij")
    
    # Grid coordinates
    i_idx, j_idx = np.meshgrid(np.arange(n_step), np.arange(n_step), indexing="ij")
    i_flat = i_idx.ravel()
    j_flat = j_idx.ravel()
    
    # 4 vertices per cell
    # T1: (i, j), (i+1, j), (i+1, j+1)
    p0_x = gx[i_flat, j_flat]; p0_z = gz[i_flat, j_flat]
    p1_x = gx[i_flat + 1, j_flat]; p1_z = gz[i_flat + 1, j_flat]
    p2_x = gx[i_flat + 1, j_flat + 1]; p2_z = gz[i_flat + 1, j_flat + 1]
    p3_x = gx[i_flat, j_flat + 1]; p3_z = gz[i_flat, j_flat + 1]
    
    # Centroids
    c1_x = (p0_x + p1_x + p2_x) / 3.0
    c1_z = (p0_z + p1_z + p2_z) / 3.0
    r1 = np.hypot(c1_x, c1_z)
    keep1 = r1 <= radius
    
    c2_x = (p0_x + p2_x + p3_x) / 3.0
    c2_z = (p0_z + p2_z + p3_z) / 3.0
    r2 = np.hypot(c2_x, c2_z)
    keep2 = r2 <= radius
    
    v0 = i_flat * (n_step + 1) + j_flat
    v1 = (i_flat + 1) * (n_step + 1) + j_flat
    v2 = (i_flat + 1) * (n_step + 1) + (j_flat + 1)
    v3 = i_flat * (n_step + 1) + (j_flat + 1)
    
    # Form triangles
    t1_v = np.stack([v0[keep1], v1[keep1], v2[keep1]], axis=1)
    t2_v = np.stack([v0[keep2], v2[keep2], v3[keep2]], axis=1)
    raw_tris = np.vstack([t1_v, t2_v])
    
    # Map to compact vertex indices
    used_verts, inv_v = np.unique(raw_tris, return_inverse=True)
    tris = inv_v.reshape((-1, 3)).astype(np.int32)
    
    # Vertex coordinates
    gx_flat = gx.ravel()
    gz_flat = gz.ravel()
    verts = np.zeros((len(used_verts), 3), dtype=np.float32)
    verts[:, 0] = c[0] + gx_flat[used_verts]
    verts[:, 1] = c[1]
    verts[:, 2] = c[2] + gz_flat[used_verts]
    
    # Build edges: 3 half-edges per triangle
    # e0: (v0, v1), e1: (v1, v2), e2: (v2, v0)
    # Store: (min_v, max_v), tri_idx, opp_vert, edge_idx
    n_tri = len(tris)
    e_all = np.vstack([
        np.stack([tris[:, 0], tris[:, 1]], axis=1),
        np.stack([tris[:, 1], tris[:, 2]], axis=1),
        np.stack([tris[:, 2], tris[:, 0]], axis=1),
    ])
    e_sorted = np.sort(e_all, axis=1)
    opp_verts = np.concatenate([tris[:, 2], tris[:, 0], tris[:, 1]])
    tri_indices = np.concatenate([np.arange(n_tri), np.arange(n_tri), np.arange(n_tri)])
    
    # Encode edge into int64 for fast sorting
    # key = min_v * (max_v_max + 1) + max_v
    max_v = len(verts)
    e_keys = e_sorted[:, 0].astype(np.int64) * max_v + e_sorted[:, 1].astype(np.int64)
    sort_order = np.argsort(e_keys)
    
    sorted_keys = e_keys[sort_order]
    sorted_opp = opp_verts[sort_order]
    sorted_tri = tri_indices[sort_order]
    sorted_e = e_sorted[sort_order]
    
    # Find hinges: pairs where sorted_keys[k] == sorted_keys[k+1]
    is_pair = (sorted_keys[:-1] == sorted_keys[1:])
    # Avoid triple edges (boundary shouldn't have them on 2-manifold)
    hinge_idx1 = np.where(is_pair)[0]
    
    h_v1 = sorted_e[hinge_idx1, 0]
    h_v2 = sorted_e[hinge_idx1, 1]
    h_v3 = sorted_opp[hinge_idx1]
    h_v4 = sorted_opp[hinge_idx1 + 1]
    
    # Orient edges consistently: cross(v2 - v1, v3 - v1) should have y > 0
    p1 = verts[h_v1]
    p2 = verts[h_v2]
    p3 = verts[h_v3]
    p4 = verts[h_v4]
    
    e_vec = p2 - p1
    n1 = np.cross(e_vec, p3 - p1)
    flip = n1[:, 1] < 0.0
    h_v1_final = np.where(flip, h_v2, h_v1)
    h_v2_final = np.where(flip, h_v1, h_v2)
    
    # Weights: 1.5 * |e|^2 / (A1 + A2)
    len_e = np.linalg.norm(verts[h_v2_final] - verts[h_v1_final], axis=1)
    A1 = 0.5 * np.linalg.norm(np.cross(verts[h_v2_final] - verts[h_v1_final], verts[h_v3] - verts[h_v1_final]), axis=1)
    A2 = 0.5 * np.linalg.norm(np.cross(verts[h_v4] - verts[h_v1_final], verts[h_v2_final] - verts[h_v1_final]), axis=1)
    h_w = (1.5 * (len_e ** 2) / (A1 + A2)).astype(np.float32)
    h_t0 = np.zeros(len(h_w), dtype=np.float32)
    
    # Unique edges for membrane
    unique_keys, first_idx = np.unique(sorted_keys, return_index=True)
    m_v1 = sorted_e[first_idx, 0]
    m_v2 = sorted_e[first_idx, 1]
    m_l0 = np.linalg.norm(verts[m_v2] - verts[m_v1], axis=1).astype(np.float32)
    
    # Lumped masses
    v_areas = np.zeros(len(verts), dtype=np.float32)
    t_v0, t_v1, t_v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    t_areas = 0.5 * np.linalg.norm(np.cross(t_v1 - t_v0, t_v2 - t_v0), axis=1).astype(np.float32)
    np.add.at(v_areas, tris[:, 0], t_areas / 3.0)
    np.add.at(v_areas, tris[:, 1], t_areas / 3.0)
    np.add.at(v_areas, tris[:, 2], t_areas / 3.0)
    
    # Clamp mask (pedestal top)
    ped_radius = 0.09
    r_v = np.hypot(verts[:, 0] - c[0], verts[:, 2] - c[2])
    clamp_flags = (r_v <= ped_radius).astype(np.int32)
    
    wall = time.time() - t0
    print(f"Spacing {spacing*1e3:4.1f} mm built in {wall*1e3:6.1f} ms:")
    print(f"  Vertices: {len(verts):6d}, Triangles: {len(tris):6d}, Hinges: {len(h_w):6d}, Edges: {len(m_l0):6d}, Clamped: {np.sum(clamp_flags):5d}")
    
    return {
        "verts": verts,
        "tris": tris,
        "h_v1": h_v1_final.astype(np.int32),
        "h_v2": h_v2_final.astype(np.int32),
        "h_v3": h_v3.astype(np.int32),
        "h_v4": h_v4.astype(np.int32),
        "h_w": h_w,
        "h_t0": h_t0,
        "m_v1": m_v1.astype(np.int32),
        "m_v2": m_v2.astype(np.int32),
        "m_l0": m_l0,
        "v_areas": v_areas,
        "clamp_flags": clamp_flags,
        "cloth_centre": c,
        "ped_radius": ped_radius,
        "cloth_radius": radius,
    }

if __name__ == "__main__":
    for sp in [0.002, 0.001, 0.0005]:
        make_circular_cloth_mesh_fast(spacing=sp)
