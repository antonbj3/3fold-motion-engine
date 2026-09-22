#!/usr/bin/env python3
"""Loader and reconstruction for the contact-graph reproduction supplement.

Run `python loader.py` to load every exported file, rebuild each operator from
its geometry, and print a verification report.  No external paths are used; all
files are read relative to this file.

Dependencies: numpy, scipy (scipy only for sparse flow solves on large graphs).
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _json(rel):
    with open(os.path.join(HERE, rel)) as f:
        return json.load(f)


def _npz(rel):
    return np.load(os.path.join(HERE, rel))


def skew(r):
    r = np.asarray(r, float)
    return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])


def adjoint(r):
    A = np.eye(6)
    A[:3, 3:] = -skew(r)
    return A


def body_minv(mass, half):
    hx, hy, hz = half
    inertia = mass / 3.0 * np.array([hy ** 2 + hz ** 2, hx ** 2 + hz ** 2,
                                     hx ** 2 + hy ** 2])
    M = np.zeros((6, 6))
    M[:3, :3] = np.eye(3) / mass
    M[3:, 3:] = np.diag(1.0 / inertia)
    return M


# ---------------------------------------------------------------------------
# patch family
# ---------------------------------------------------------------------------
def patch_scenes():
    return _json("patch/scenes.json")


def patch_scene(name):
    for s in patch_scenes()["scenes"]:
        if s["name"] == name:
            return s
    raise KeyError(name)


def patch_operator(name):
    z = _npz("patch/operators.npz")
    return {k[len(name) + 2:]: z[k] for k in z.files if k.startswith(name + "__")}


def assemble_patch(scene):
    """Rebuild J_points, J_patch, C, G, b, incidence and the graph assembly."""
    bodies = scene["bodies"]
    patches = scene["patches"]
    nb = len(bodies)
    nv = 6 * nb
    npatch = len(patches)

    Jpts = []
    pairs = []
    mus = []
    for pa in patches:
        a, b = pa["a"], pa["b"]
        for p in pa["pts"]:
            blk = np.zeros((3, nv))
            for r, d in enumerate((pa["n"], pa["t1"], pa["t2"])):
                if a >= 0:
                    rA = np.asarray(p) - np.asarray(bodies[a]["centre"])
                    blk[r, 6 * a:6 * a + 3] = d
                    blk[r, 6 * a + 3:6 * a + 6] = np.cross(rA, d)
                if b >= 0:
                    rB = np.asarray(p) - np.asarray(bodies[b]["centre"])
                    blk[r, 6 * b:6 * b + 3] = -np.asarray(d)
                    blk[r, 6 * b + 3:6 * b + 6] = -np.cross(rB, d)
            Jpts.append(blk)
        pairs.append((a, b))
        mus.append(pa["mu"])
    J_points = np.vstack(Jpts)
    pairs = np.array(pairs, int)
    mu = np.array(mus, float)

    Jp = []
    for pa in patches:
        a, b = pa["a"], pa["b"]
        ell = float(np.sqrt((pa["e1"] ** 2 + pa["e2"] ** 2) / 3.0))
        blk = np.zeros((6, nv))
        force_dirs = [(0, pa["n"]), (3, pa["t1"]), (4, pa["t2"])]
        mom_dirs = [(1, pa["t1"], pa["e2"]), (2, pa["t2"], pa["e1"]), (5, pa["n"], ell)]
        for i, sgn in ((a, 1.0), (b, -1.0)):
            if i < 0:
                continue
            r = np.asarray(pa["p0"]) - np.asarray(bodies[i]["centre"])
            for row, d in force_dirs:
                blk[row, 6 * i:6 * i + 3] = sgn * np.asarray(d)
                blk[row, 6 * i + 3:6 * i + 6] = sgn * np.cross(r, np.asarray(d))
            for row, d, sc in mom_dirs:
                blk[row, 6 * i + 3:6 * i + 6] = sgn * np.asarray(d) * sc
        Jp.append(blk)
    J_patch = np.vstack(Jp)

    # lumping map C
    C = np.zeros((6 * npatch, 3 * sum(len(pa["pts"]) for pa in patches)))
    col = 0
    for c, pa in enumerate(patches):
        ell = float(np.sqrt((pa["e1"] ** 2 + pa["e2"] ** 2) / 3.0))
        for p in pa["pts"]:
            r = np.asarray(p) - np.asarray(pa["p0"])
            for j, d in enumerate((pa["n"], pa["t1"], pa["t2"])):
                f = np.asarray(d)
                m = np.cross(r, f)
                blk = np.zeros(6)
                blk[0] = f @ np.asarray(pa["n"])
                blk[3] = f @ np.asarray(pa["t1"])
                blk[4] = f @ np.asarray(pa["t2"])
                blk[1] = (m @ np.asarray(pa["t1"])) / pa["e2"]
                blk[2] = (m @ np.asarray(pa["t2"])) / pa["e1"]
                blk[5] = (m @ np.asarray(pa["n"])) / ell
                C[6 * c:6 * c + 6, col + j] = blk
            col += 3

    Minv = np.zeros((nv, nv))
    for i, bd in enumerate(bodies):
        Minv[6 * i:6 * i + 6, 6 * i:6 * i + 6] = body_minv(bd["mass"], bd["half"])
    v_free = np.array(scene["v_free"], float)
    G = J_patch @ Minv @ J_patch.T
    b = J_patch @ v_free

    # graph assembly in the common gauge (reference point at the origin)
    W = []
    for bd in bodies:
        W.append(adjoint(-np.asarray(bd["centre"])) @ body_minv(bd["mass"], bd["half"])
                 @ adjoint(-np.asarray(bd["centre"])).T)
    S = np.zeros((npatch, nb))
    for c, pa in enumerate(patches):
        if pa["a"] >= 0:
            S[c, pa["a"]] += 1.0
        if pa["b"] >= 0:
            S[c, pa["b"]] -= 1.0
    L_graph = np.zeros((6 * npatch, 6 * npatch))
    for i in range(nb):
        for c in range(npatch):
            if S[c, i] == 0:
                continue
            for d in range(npatch):
                if S[d, i] == 0:
                    continue
                L_graph[6 * c:6 * c + 6, 6 * d:6 * d + 6] += S[c, i] * S[d, i] * W[i]

    # common gauge: relative twist at one world point, world axes
    J_common = np.zeros((6 * npatch, nv))
    for c, pa in enumerate(patches):
        for i, sgn in ((pa["a"], 1.0), (pa["b"], -1.0)):
            if i < 0:
                continue
            J_common[6 * c:6 * c + 6, 6 * i:6 * i + 6] = sgn * adjoint(
                -np.asarray(bodies[i]["centre"]))
    G_common = J_common @ Minv @ J_common.T

    return {"J_points": J_points, "J_patch": J_patch, "C": C, "G": G, "b": b,
            "mu": mu, "body_pairs": pairs, "incidence": S, "conductances": W,
            "L_graph": L_graph, "J_common": J_common, "G_common": G_common}


def verify_patch(name):
    sc = patch_scene(name)
    op = patch_operator(name)
    r = assemble_patch(sc)
    out = {"name": name, "n_patches": len(sc["patches"]),
           "n_point_contacts": r["J_points"].shape[0] // 3,
           "dG": float(np.max(np.abs(r["G"] - op["G"]))),
           "db": float(np.max(np.abs(r["b"] - op["b"]))),
           "dmu": float(np.max(np.abs(r["mu"] - op["mu"]))),
           "dJ_patch": float(np.max(np.abs(r["J_patch"] - op["J_patch"]))),
           "dC": float(np.max(np.abs(r["C"] - op["C"]))),
           "assembly_rel_inf": float(np.max(np.abs(r["G_common"] - r["L_graph"]))
                                     / max(np.max(np.abs(r["G_common"])), 1e-300)),
           "rank_J_points": int(np.linalg.matrix_rank(r["J_points"])),
           "rows_J_points": int(r["J_points"].shape[0]),
           "rank_J_patch": int(np.linalg.matrix_rank(r["J_patch"])),
           "rows_J_patch": int(r["J_patch"].shape[0])}
    # vertex pencil: spec(G) = spec(Lv, M) with Lv = J_common^T J_common
    bodies = sc["bodies"]
    nb = len(bodies)
    nv = 6 * nb
    J_common = np.zeros((6 * len(sc["patches"]), nv))
    for c, pa in enumerate(sc["patches"]):
        for i, sgn in ((pa["a"], 1.0), (pa["b"], -1.0)):
            if i < 0:
                continue
            J_common[6 * c:6 * c + 6, 6 * i:6 * i + 6] = sgn * adjoint(
                -np.asarray(bodies[i]["centre"]))
    Lv = J_common.T @ J_common
    # vertex pencil: spec(G)\{0} == spec(M^{-1/2} Lv M^{-1/2})
    Minv_sqrt = np.zeros((nv, nv))
    for i, bd in enumerate(bodies):
        B = body_minv(bd["mass"], bd["half"])
        w, V = np.linalg.eigh(B)
        Minv_sqrt[6 * i:6 * i + 6, 6 * i:6 * i + 6] = (V * np.sqrt(w)) @ V.T
    evG = np.linalg.eigvalsh(r["G_common"])
    evP = np.linalg.eigvalsh(Minv_sqrt @ Lv @ Minv_sqrt)
    evG = np.sort(evG[evG > 1e-9 * max(evG[-1], 1e-300)])
    evP = np.sort(evP[evP > 1e-9 * max(evP[-1], 1e-300)])
    n = min(len(evG), len(evP))
    out["pencil_rel_spectral"] = (float(np.max(np.abs(evG[-n:] - evP[-n:])) / evG[-1])
                                  if n else float("nan"))
    return out


# ---------------------------------------------------------------------------
# incidence family
# ---------------------------------------------------------------------------
def incidence_scenes():
    return _json("patch/incidence_scenes.json")


def incidence_system(name):
    z = _npz("patch/incidence.npz")
    return {"body_pairs": z[f"{name}__body_pairs"],
            "inv_mass": z[f"{name}__inv_mass"],
            "rhs": z[f"{name}__rhs"],
            "L_indptr": z[f"{name}__L_indptr"],
            "L_indices": z[f"{name}__L_indices"],
            "L_data": z[f"{name}__L_data"]}


def scalar_incidence(body_pairs, inv_mass):
    """Dense L = S diag(1/m) S^T from the signed contact-body incidence."""
    pairs = np.asarray(body_pairs, int)
    w = np.asarray(inv_mass, float)
    n_c, n_b = pairs.shape[0], len(w)
    S = np.zeros((n_c, n_b))
    for c in range(n_c):
        a, b = int(pairs[c, 0]), int(pairs[c, 1])
        if a >= 0:
            S[c, a] += 1.0
        if b >= 0:
            S[c, b] -= 1.0
    return S @ np.diag(w) @ S.T


def verify_incidence(name):
    s = incidence_system(name)
    L = scalar_incidence(s["body_pairs"], s["inv_mass"])
    Lref = _csr_dense(s["L_indptr"], s["L_indices"], s["L_data"], L.shape[0])
    return {"name": name, "n_contacts": int(L.shape[0]),
            "dL": float(np.max(np.abs(L - Lref))),
            "symmetry": float(np.max(np.abs(L - L.T)))}


def _csr_dense(indptr, indices, data, n):
    L = np.zeros((n, n))
    for i in range(n):
        for k in range(indptr[i], indptr[i + 1]):
            L[i, indices[k]] = data[k]
    return L


# ---------------------------------------------------------------------------
# load-ordering family
# ---------------------------------------------------------------------------
def load_ordering_scenes():
    return _json("load_ordering/scenes.json")


def load_ordering_scene(name):
    z = _npz("load_ordering/scenes.npz")
    keys = [k for k in z.files if k.startswith(name + "__")]
    d = {k[len(name) + 2:]: z[k] for k in keys}
    return d


def graph_laplacian(edges, weights, num_nodes):
    """Contact-graph Laplacian L = B^T diag(w) B, assembled exactly as in the
    delivered experiment (scipy sparse incidence product)."""
    import scipy.sparse as sp
    edges = np.asarray(edges, int)
    weights = np.asarray(weights, float)
    n_c = len(edges)
    rows = np.repeat(np.arange(n_c), 2)
    cols = edges.reshape(-1)
    vals = np.tile([1.0, -1.0], n_c)
    B = sp.csr_matrix((vals, (rows, cols)), shape=(n_c, num_nodes))
    L = (B.T @ sp.diags(weights) @ B).tocsr()
    return L, B


def _connected_components(edges, num_nodes):
    """Connected components of the (undirected) contact graph."""
    import scipy.sparse as sp
    import scipy.sparse.csgraph as csg
    e = np.asarray(edges, int)
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])),
                      shape=(num_nodes, num_nodes))
    return csg.connected_components(A, directed=False)


def _current_injection(num_nodes, n_grains, node_floor, node_foot, kind):
    I = np.zeros(num_nodes)
    if kind == "foot":
        I[node_foot] = 1.0
        I[node_floor] = -1.0
    else:
        I[:n_grains] = 1.0
        I[node_floor] = -float(n_grains)
    return I


def solve_flow(edges, weights, num_nodes, n_grains, node_floor, node_foot, kind,
               method="direct", rtol=1e-5, maxiter=3000):
    """Grounded Laplacian current flow.

    ``method="direct"`` is the independent reference: a sparse LU of the
    grounded matrix with ONE gauge node pinned per connected component.  The
    grounded matrix is singular on every component that does not contain the
    grounded node (isolated grains, the wall), so each such component needs its
    own reference potential; the edge currents are invariant under that gauge.
    ``method="minres"`` is the legacy iterative solve, kept for the convergence
    sweep only (its stopping criterion is on the residual, which does not bound
    the current error on these ill-conditioned Laplacians).
    """
    import scipy.sparse.linalg as spla

    L, _ = graph_laplacian(edges, weights, num_nodes)
    I = _current_injection(num_nodes, n_grains, node_floor, node_foot, kind)
    edges = np.asarray(edges, int)
    if method == "direct":
        L = L.tocsr()
        nc, lab = _connected_components(edges, num_nodes)
        pins = np.array([np.nonzero(lab == c)[0][0] for c in range(nc)], int)
        pin = np.zeros(num_nodes, bool)
        pin[pins] = True
        free = ~pin
        V = np.zeros(num_nodes)
        V[free] = spla.spsolve(L[free][:, free].tocsc(), I[free])
    elif method == "minres":
        V, _ = spla.minres(L, I, rtol=rtol, maxiter=maxiter)
    else:
        raise ValueError(method)
    I_edge = np.abs(weights * (V[edges[:, 0]] - V[edges[:, 1]]))
    return I_edge, V


def solve_flow_diag(edges, weights, num_nodes, n_grains, node_floor, node_foot, kind):
    """Direct solve plus residual, backward error and gauge/component record."""
    import scipy.sparse.linalg as spla

    L, _ = graph_laplacian(edges, weights, num_nodes)
    L = L.tocsr()
    I = _current_injection(num_nodes, n_grains, node_floor, node_foot, kind)
    edges = np.asarray(edges, int)
    nc, lab = _connected_components(edges, num_nodes)
    pins = np.array([np.nonzero(lab == c)[0][0] for c in range(nc)], int)
    pin = np.zeros(num_nodes, bool)
    pin[pins] = True
    free = ~pin
    V = np.zeros(num_nodes)
    V[free] = spla.spsolve(L[free][:, free].tocsc(), I[free])
    I_edge = np.abs(weights * (V[edges[:, 0]] - V[edges[:, 1]]))
    r = L @ V - I
    rf = r[free]
    rhs = I[free]
    rel = float(np.linalg.norm(rf) / max(np.linalg.norm(rhs), 1e-300))
    # componentwise backward error ||r||_inf / (||A||_inf ||V||_inf + ||I||_inf)
    A = L[free][:, free]
    anorm = float(np.max(np.abs(A.data))) if A.nnz else 0.0
    denom = anorm * float(np.max(np.abs(V[free]))) + float(np.max(np.abs(rhs)))
    bwd = float(np.max(np.abs(rf)) / max(denom, 1e-300)) if rf.size else 0.0
    comp_sizes = np.bincount(lab).tolist()
    return I_edge, {"rel_residual": rel, "backward_error": bwd,
                    "n_components": int(nc), "component_sizes": comp_sizes,
                    "gauge_nodes": [int(x) for x in pins],
                    "n_gauge_nodes": int(len(pins))}


def _rankdata(a, atol=0.0):
    """Average ranks under leader clustering at ``atol``.

    The series is sorted stably (``kind="mergesort"``, deterministic); scanning
    in that order, a value joins the current cluster while it is at most
    ``atol`` above the cluster's FIRST value, otherwise it starts a new cluster.
    All members of a cluster receive the average of the ranks they cover.  The
    tie radius is dimensionless in the caller: ``atol = tie_rel * max|a|``.
    """
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(len(a), dtype=float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and (sa[j] - sa[i]) <= atol:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = ranks[order[i:j]].mean()
        i = j
    return ranks


def _spearman(a, b, tie_rel=0.0):
    ta = tie_rel * float(np.max(np.abs(a))) if tie_rel else 0.0
    tb = tie_rel * float(np.max(np.abs(b))) if tie_rel else 0.0
    ra = _rankdata(a, ta)
    rb = _rankdata(b, tb)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / denom) if denom > 0 else 0.0


def metrics(pred, truth, tie_rel=0.0):
    """Spearman and precision@5/10% with a dimensionless tie radius ``tie_rel``.

    ``tie_rel`` is relative to each series' own max magnitude.  Ranks use leader
    clustering: after a stable sort, a value joins the current cluster while it
    is at most ``tie_rel * max|series|`` above the cluster's FIRST value, and
    cluster members share the average rank (see ``_rankdata``).  ``tie_rel=0``
    recovers the exact-equality convention used by the delivered measurements.
    """
    n = len(truth)
    k5 = max(1, int(round(n * 0.05)))
    k10 = max(1, int(round(n * 0.10)))
    t5 = set(np.argsort(truth)[-k5:])
    t10 = set(np.argsort(truth)[-k10:])
    p5 = len(t5 & set(np.argsort(pred)[-k5:])) / k5
    p10 = len(t10 & set(np.argsort(pred)[-k10:])) / k10
    return {"spearman": _spearman(pred, truth, tie_rel),
            "prec_5": float(p5), "prec_10": float(p10),
            "tie_rel": float(tie_rel),
            "n_ties_pred": int(len(pred) - len(np.unique(pred))),
            "n_ties_truth": int(len(truth) - len(np.unique(truth)))}


def load_ordering_measurements():
    return _json("load_ordering/measurements.json")


# ---------------------------------------------------------------------------
# ramp family
# ---------------------------------------------------------------------------
def ramp_geometry():
    return _json("ramp/geometry.json")


def ramp_measurements():
    return _json("ramp/measurements.json")


def rect_tip(a_u, a_v, u0, v0, phi0, h, phi, g=9.81):
    c = np.cos(phi - phi0)
    s = np.sin(phi - phi0)
    lim = []
    if abs(c) > 1e-14:
        lim.append((a_u - np.copysign(u0, c)) / abs(c))
    if abs(s) > 1e-14:
        lim.append((a_v - np.copysign(v0, s)) / abs(s))
    return g / h * min(lim)


def ramp_static_radius(scene, phi):
    """Proposition 7 radius: min of the interface and ground patch candidates."""
    cand = {}
    if scene["support"] >= 0:
        p = scene["interface_patch"]
        cand["interface_tip"] = rect_tip(p["a_u"], p["a_v"], p["u0"], p["v0"],
                                         p["phi0"], p["h"], phi)
        cand["interface_slide"] = p["mu"] * 9.81
    pg = scene["ground_patch"]
    cand["ground_tip"] = rect_tip(pg["a_u"], pg["a_v"], pg["u0"], pg["v0"],
                                  pg["phi0"], pg["h"], phi)
    cand["ground_slide"] = pg["mu"] * 9.81
    label = min(cand, key=cand.get)
    return float(cand[label]), label


# ---------------------------------------------------------------------------
# verification entry point
# ---------------------------------------------------------------------------
def main():
    print("patch scenes:")
    pdoc = patch_scenes()
    for s in pdoc["scenes"]:
        v = verify_patch(s["name"])
        ok = (v["dG"] == 0.0 and v["db"] == 0.0 and v["dmu"] == 0.0)
        print(f"  {s['name']:28s} n_p={v['n_patches']:2d} "
              f"dG={v['dG']:.1e} db={v['db']:.1e} dC={v['dC']:.1e} "
              f"assembly={v['assembly_rel_inf']:.1e} ok={ok}")
    print("incidence systems:")
    for s in incidence_scenes()["scenes"]:
        v = verify_incidence(s["name"])
        print(f"  {s['name']:28s} n_c={v['n_contacts']:5d} dL={v['dL']:.1e} "
              f"sym={v['symmetry']:.1e}")
    print("load-ordering scenes:")
    for s in load_ordering_scenes()["scenes"]:
        d = load_ordering_scene(s["name"])
        print(f"  {s['name']:20s} grains={s['n_grains']:6d} "
              f"contacts={s['n_contacts']:6d} lam_sha={s['lam_n_sha256'][:16]}")
    print("ramp scenes:")
    rg = ramp_geometry()
    for tag, sc in rg["scenes"].items():
        r0, lab = ramp_static_radius(sc, 0.0)
        print(f"  {tag}: {sc['scene']:16s} radius(phi=0)={r0:.6f} ({lab})")
    print("loader: all files loaded")


if __name__ == "__main__":
    main()
