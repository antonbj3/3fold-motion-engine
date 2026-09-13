"""grasp_wrench_cert -- force-closure / grasp-wrench-feasibility certificate for a hand grasping a tool
or screw during simulated assembly. A feasibility check on a given contact set (no physics simulation)
anchored on classical grasp mechanics.

THE PHYSICS (externally anchored, Nguyen 1988 / Ferrari-Canny 1992):
A grasp is a set of K contacts on the grasped object. Each contact i (at position p_i, inward surface
normal n_i, Coulomb friction mu) can apply a force inside its FRICTION CONE {f : f.n_i >= 0,
|f_tangential| <= mu (f.n_i)}. The contact WRENCH about the object COM is w_i = [f_i ; r_i x f_i] with
r_i = p_i - com. The set of net wrenches the grasp can resist is the convex cone of all contact
wrenches. Linearising each friction cone into `n_edges` unit-force generators e_ij gives wrench
PRIMITIVES w_ij = [e_ij ; r_i x e_ij].

FORCE CLOSURE (can resist ANY external wrench) <=> the ORIGIN lies in the INTERIOR of the convex hull
of the (force-normalised) wrench primitives. The Ferrari-Canny quality EPSILON = the radius of the
largest 6-D ball centred at the origin inscribed in that hull = the SMALLEST-magnitude external
wrench that breaks the grasp (>0 iff force closure). This is the standard grasp-quality metric.

NOT substrate: no dynamics/contact simulation -- a pure wrench-space feasibility test on given contacts.
Composes-via-contract: feed contacts from a grasp planner / MANO-SMPL-X hand pose; returns a verdict a
motion cert can gate on before admitting a generated grasp.
"""
import numpy as np

__all__ = ["grasp_force_closure_cert"]


def _friction_edges(n, mu, n_edges):
    """Unit-force generators of the linearised Coulomb friction cone around inward normal n."""
    n = np.asarray(n, float)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return None
    n = n / nn
    # a tangent basis (t1, t2) orthonormal to n
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, a); t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    edges = []
    for j in range(n_edges):
        th = 2.0 * np.pi * j / n_edges
        e = n + mu * (np.cos(th) * t1 + np.sin(th) * t2)   # a boundary generator of the cone
        edges.append(e / np.linalg.norm(e))
    return np.array(edges)


def grasp_force_closure_cert(contacts, mu=0.5, com=None, n_edges=8, torque_scale=None, eps_floor=1e-6):
    """Certify whether a grasp (list/array of contacts) is FORCE-CLOSURE + its Ferrari-Canny quality.

    contacts : list of (position(3,), inward_normal(3,)) OR an array of shape (K, 6) = [px,py,pz, nx,ny,nz].
               inward_normal points INTO the object (the direction the contact can PUSH).
    mu       : Coulomb friction coefficient (>=0). mu=0 -> frictionless point contacts.
    com      : object centre of mass (3,); defaults to the centroid of the contact positions.
    n_edges  : friction-cone linearisation resolution (>=4). Larger = tighter (conservative) hull.
    torque_scale : characteristic length L to make torque commensurate with force (wrench = [f ; (r x f)/L]).
               Defaults to the max |r_i| (object radius). REQUIRED for a dimensionally-honest epsilon.

    Returns {force_closure (bool), epsilon_quality (float, >=0; 0 iff not closure), n_contacts,
             n_primitives, verdict, degenerate (bool)}.
      verdict: FORCE-CLOSURE (eps>0) / NOT-FORCE-CLOSURE (origin on/outside hull) / ABSTAIN-DEGENERATE.
    """
    # ---- normalise input ----
    P, N = [], []
    arr = np.asarray(contacts, float) if not isinstance(contacts, (list, tuple)) else None
    if arr is not None and arr.ndim == 2 and arr.shape[1] == 6:
        P = arr[:, :3]; N = arr[:, 3:]
    else:
        for c in contacts:
            P.append(np.asarray(c[0], float)); N.append(np.asarray(c[1], float))
        P = np.asarray(P, float); N = np.asarray(N, float)
    K = len(P)
    fail = {"force_closure": False, "epsilon_quality": 0.0, "n_contacts": int(K), "n_primitives": 0,
            "degenerate": True}
    if K < 2 or not np.all(np.isfinite(P)) or not np.all(np.isfinite(N)):
        return {**fail, "verdict": "ABSTAIN-DEGENERATE (need >=2 finite contacts)"}
    if not (np.isfinite(mu) and mu >= 0.0) or int(n_edges) < 4:
        return {**fail, "verdict": "ABSTAIN-DEGENERATE (mu>=0 finite and n_edges>=4 required)"}
    com = np.asarray(com, float) if com is not None else P.mean(axis=0)
    R = P - com
    L = float(torque_scale) if torque_scale is not None else max(float(np.max(np.linalg.norm(R, axis=1))), 1e-9)
    if not (np.isfinite(L) and L > 0):
        return {**fail, "verdict": "ABSTAIN-DEGENERATE (non-positive torque scale)"}

    # ---- build wrench primitives ----
    W = []
    for i in range(K):
        edges = _friction_edges(N[i], mu, int(n_edges))
        if edges is None:
            return {**fail, "verdict": "ABSTAIN-DEGENERATE (zero-length contact normal)"}
        for e in edges:
            tau = np.cross(R[i], e) / L                     # scaled torque, commensurate with force
            W.append(np.concatenate([e, tau]))
    W = np.asarray(W)                                       # (n_prim, 6)

    # ---- force closure <=> origin strictly inside conv-hull of primitives; eps = min facet distance ----
    try:
        from scipy.spatial import ConvexHull, QhullError
    except Exception:
        return {**fail, "n_primitives": int(len(W)), "verdict": "ABSTAIN-DEGENERATE (scipy unavailable)"}
    try:
        hull = ConvexHull(W)                                # in 6-D
    except Exception:
        # primitives do not span 6-D (degenerate span) -> cannot enclose the origin -> not force closure
        return {"force_closure": False, "epsilon_quality": 0.0, "n_contacts": int(K),
                "n_primitives": int(len(W)), "degenerate": False,
                "verdict": "NOT-FORCE-CLOSURE (wrench primitives do not span 6-D; a wrench axis is unresisted)"}
    # hull facets: A x + b <= 0 for interior points (scipy 'equations' = [A | b], normal outward). Origin distance
    # to facet k = b_k (since A_k.0 + b_k = b_k); interior => all b_k < 0; epsilon = -max_k b_k (closest facet).
    b = hull.equations[:, -1]
    eps = float(-np.max(b))                                 # >0 iff origin strictly inside all facets
    force_closure = bool(eps > eps_floor)
    return {"force_closure": force_closure,
            "epsilon_quality": float(max(eps, 0.0)),
            "n_contacts": int(K), "n_primitives": int(len(W)), "degenerate": False,
            "verdict": ("FORCE-CLOSURE (eps=%.4g resists any wrench)" % eps) if force_closure
                       else "NOT-FORCE-CLOSURE (origin on/outside wrench hull; some external wrench breaks the grasp)"}


def _selftest():
    ok = True
    # a canonical FORCE-CLOSURE grasp: 4 contacts on the +-x, +-y faces of a unit cube, normals inward
    cube = [([1, 0, 0], [-1, 0, 0]), ([-1, 0, 0], [1, 0, 0]),
            ([0, 1, 0], [0, -1, 0]), ([0, -1, 0], [0, 1, 0]),
            ([0, 0, 1], [0, 0, -1]), ([0, 0, -1], [0, 0, 1])]
    r = grasp_force_closure_cert(cube, mu=0.5)
    ok &= r["force_closure"] and r["epsilon_quality"] > 0
    print("  [%s] 6-face enclosing grasp -> FORCE-CLOSURE (eps=%.3g)" % ("PASS" if ok else "FAIL", r["epsilon_quality"]))
    # NOT closure: all contacts on ONE side (parallel inward normals) -> the object can be pulled out the open side
    onesided = [([1, y, z], [-1, 0, 0]) for y in (-0.3, 0.3) for z in (-0.3, 0.3)]
    r2 = grasp_force_closure_cert(onesided, mu=0.5)
    c2 = not r2["force_closure"] and r2["epsilon_quality"] == 0.0
    print("  [%s] one-sided parallel-normal grasp -> NOT-FORCE-CLOSURE" % ("PASS" if c2 else "FAIL"))
    ok &= c2
    # frictionless single-side pair cannot resist tangential -> not closure
    r3 = grasp_force_closure_cert([([1, 0, 0], [-1, 0, 0]), ([-1, 0, 0], [1, 0, 0])], mu=0.0)
    c3 = not r3["force_closure"]
    print("  [%s] frictionless antipodal pair (2 pts) -> NOT-FORCE-CLOSURE (unresisted torque axes)" % ("PASS" if c3 else "FAIL"))
    ok &= c3
    # degenerate: 1 contact -> abstain
    r4 = grasp_force_closure_cert([([1, 0, 0], [-1, 0, 0])], mu=0.5)
    c4 = r4["verdict"].startswith("ABSTAIN")
    print("  [%s] single contact -> ABSTAIN-DEGENERATE" % ("PASS" if c4 else "FAIL"))
    ok &= c4
    print("grasp_wrench_cert selftest: %s" % ("ALL PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
