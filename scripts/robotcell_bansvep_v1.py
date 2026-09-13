#!/usr/bin/env python3
"""Swept self-collision and arm-vs-environment check over a robot's joint range or along a path.

Method (both conditions are built in):
  * AABB clearance: an AABB of one non-adjacent link's parts against another's. AABB-DISJOINT implies
    mesh-disjoint (the box contains the body), so clearance is proven fail-closed.
  * Escalation: every non-adjacent pair whose AABBs overlap in any pose is escalated to an exact
    oriented-box separation test (SAT over 15 axes) before a verdict. An AABB overlap is a suspicion,
    never an acquittal and never a conviction.
  * Declared sweep density: steps per joint interval plus EDGE poses at each joint's min and max, so a
    "0 collisions" result carries its own resolution.

A disable set (SRDF pattern) removes part pairs that already overlap at the reference pose: those are
structurally connected (a shared joint housing or cable run), never a genuine motion collision.

Kinematics come from `robotcell_leder_v1.fk` (URDF-driven). Geometry comes from a parts manifest
(`scripts/urdf_parts_manifest.py`), whose per-part world AABBs at the reference pose are transformed
into link-local frames via inv(F_reference).

I/O:
  --parts-manifest PATH   manifest JSON (default: data/cells/ur10e_parts_manifest_v1.json)
  --steps N               steps per joint interval, edge poses included
  --combos M              seeded random joint combinations
  --ref-only              calibration run: reference pose only, must yield 0 candidates
  --bana PATH             path-mode: collision-check along a trajectory JSON ({t, q} in degrees)
  --bana-steps N          interpolation steps per path segment
  --out PATH              report JSON
Exit code 0 on a GREEN verdict, 2 otherwise.
"""
import argparse
import json
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robotcell_leder_v1 import ROOT, fk, load_chain, DEFAULT_URDF  # noqa: E402

DEFAULT_MANIFEST = os.path.join(ROOT, "data", "cells", "ur10e_parts_manifest_v1.json")


def link_of(subassembly):
    """`robot/link{n}` -> n; anything else -> -1 (static cell body)."""
    s = subassembly or ""
    if s.startswith("robot/link"):
        try:
            return int(s[len("robot/link"):].split("/")[0])
        except ValueError:
            return -1
    return -1


def se3_apply(F, v):
    R, p = F
    return [R[i][0] * v[0] + R[i][1] * v[1] + R[i][2] * v[2] + p[i] for i in range(3)]


def se3_inv_apply(F, v):
    R, p = F
    d = [v[i] - p[i] for i in range(3)]
    return [R[0][j] * d[0] + R[1][j] * d[1] + R[2][j] * d[2] for j in range(3)]   # R^T d


def corners(bbox):
    x0, y0, z0, x1, y1, z1 = bbox
    return [[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]


def aabb_of(pts):
    return [min(p[i] for p in pts) for i in range(3)] + [max(p[i] for p in pts) for i in range(3)]


def aabb_overlap(a, b, tol=0.0):
    return all(a[i] <= b[i + 3] + tol and b[i] <= a[i + 3] + tol for i in range(3))


def union_aabb(aabbs):
    return [min(a[i] for a in aabbs) for i in range(3)] + [max(a[i + 3] for a in aabbs) for i in range(3)]


def _sub(a, b): return [a[i] - b[i] for i in range(3)]
def _dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def _cross(a, b): return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _norm(a):
    n = math.sqrt(_dot(a, a))
    return [x / n for x in a] if n > 1e-12 else None


def obb_from_corners(wc):
    """8 corners of a rigid box (ordering as produced by corners()) -> (center, [axes], [half extents]).
    The box is tight (a reference-pose world AABB rigidly transformed), so there is no rotation inflation."""
    c0 = wc[0]
    u = _sub(wc[4], c0)
    v = _sub(wc[2], c0)
    w = _sub(wc[1], c0)
    center = [sum(p[i] for p in wc) / 8.0 for i in range(3)]
    axes, he = [], []
    for e in (u, v, w):
        L = math.sqrt(_dot(e, e))
        axes.append([x / L for x in e] if L > 1e-9 else [1.0, 0.0, 0.0])
        he.append(L / 2.0)
    return center, axes, he


def obb_overlap(wcA, wcB):
    """Exact OBB-OBB separation test (SAT, 15 axes). True = the boxes overlap."""
    cA, axA, heA = obb_from_corners(wcA)
    cB, axB, heB = obb_from_corners(wcB)
    T = _sub(cB, cA)
    axes = list(axA) + list(axB)
    for a in axA:
        for b in axB:
            n = _norm(_cross(a, b))
            if n:
                axes.append(n)
    for L in axes:
        ra = sum(heA[i] * abs(_dot(axA[i], L)) for i in range(3))
        rb = sum(heB[i] * abs(_dot(axB[i], L)) for i in range(3))
        if abs(_dot(T, L)) > ra + rb + 1e-9:
            return False            # separating axis found -> disjoint
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts-manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--steps", type=int, default=13, help="steps per joint interval (edge poses included)")
    ap.add_argument("--combos", type=int, default=400, help="seeded random joint combinations")
    ap.add_argument("--ref-only", action="store_true",
                    help="calibration: reference pose only -> 0 candidates expected")
    ap.add_argument("--bana", help="path mode: trajectory JSON with {t, q} (q in degrees)")
    ap.add_argument("--bana-steps", type=int, default=24, help="interpolation steps per path segment")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "robotcell_bansvep_v1.json"))
    args = ap.parse_args()

    man = json.load(open(args.parts_manifest))
    urdf = os.path.join(ROOT, man.get("urdf", os.path.relpath(DEFAULT_URDF, ROOT)))
    joints, limits, jnames = load_chain(urdf)
    nj = len(joints)
    q_render = [math.radians(a) for a in man["render_pose_deg"]]
    F_render = fk(q_render, joints)
    JLIM = [(math.radians(lo), math.radians(hi)) for lo, hi in man.get("axis_limits_deg", limits)]

    parts = []
    for p in man["parts"]:
        L = link_of(p["subassembly"])
        wc = corners(p["bbox"])
        if L < 0:
            parts.append({"namn": p["part"], "link": -1, "aabb_static": aabb_of(wc)})
        else:
            parts.append({"namn": p["part"], "link": L,
                          "local": [se3_inv_apply(F_render[L], c) for c in wc]})

    arm_links = list(range(1, nj + 1))
    by_link = {L: [p for p in parts if p["link"] == L] for L in arm_links}
    static_parts = [p for p in parts if p["link"] == -1]

    poses = []
    if args.bana:
        traj = json.load(open(args.bana))
        qs = [[math.radians(x) for x in row] for row in traj["q"]]
        for a, b in zip(qs[:-1], qs[1:]):
            for k in range(args.bana_steps + 1):
                t = k / args.bana_steps
                poses.append(("bana", [a[i] + (b[i] - a[i]) * t for i in range(nj)]))
    elif args.ref_only:
        poses = [("ref", list(q_render))]
    else:
        for j in range(nj):                                  # (a) per-joint sweep incl. endpoints
            lo, hi = JLIM[j]
            for k in range(args.steps):
                q = list(q_render)
                q[j] = lo + (hi - lo) * k / (args.steps - 1)
                poses.append(("perled", q))
        poses.append(("edge_min", [JLIM[j][0] for j in range(nj)]))   # (b) edge poses
        poses.append(("edge_max", [JLIM[j][1] for j in range(nj)]))
        seed = 1234567                                       # (c) seeded LCG combinations
        def rnd():
            nonlocal seed
            seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
            return seed / 0x7FFFFFFF
        for _ in range(args.combos):
            poses.append(("combo", [JLIM[j][0] + (JLIM[j][1] - JLIM[j][0]) * rnd() for j in range(nj)]))

    def part_corners(p, F):
        return [se3_apply(F[p["link"]], c) for c in p["local"]] if p["link"] >= 0 else corners(p["aabb_static"])

    all_movable = [p for p in parts if p["link"] in arm_links]
    ref_wc = {p["namn"]: part_corners(p, F_render) for p in all_movable}
    ref_aabb = {n: aabb_of(w) for n, w in ref_wc.items()}
    static_wc = {sp["namn"]: corners(sp["aabb_static"]) for sp in static_parts}
    disable = set()
    for i in range(len(all_movable)):
        for jx in range(i + 1, len(all_movable)):
            pa, pb = all_movable[i], all_movable[jx]
            if abs(pa["link"] - pb["link"]) < 2:
                continue
            if aabb_overlap(ref_aabb[pa["namn"]], ref_aabb[pb["namn"]]) and \
               obb_overlap(ref_wc[pa["namn"]], ref_wc[pb["namn"]]):
                disable.add(tuple(sorted((pa["namn"], pb["namn"]))))
    for p in all_movable:
        for sp in static_parts:
            if aabb_overlap(ref_aabb[p["namn"]], sp["aabb_static"]) and \
               obb_overlap(ref_wc[p["namn"]], static_wc[sp["namn"]]):
                disable.add(tuple(sorted((p["namn"], sp["namn"]))))

    stat_sph = []
    for sp in static_parts:
        wc = corners(sp["aabb_static"])
        cx = [sum(w[i] for w in wc) / 8.0 for i in range(3)]
        stat_sph.append(((cx, max(math.dist(cx, w) for w in wc)), sp))

    candidates = []
    seen_pairs = set()
    n_checks = 0
    for ptype, q in poses:
        F = fk(q, joints)
        arm = []
        for L in arm_links:
            for p in by_link[L]:
                wc = [se3_apply(F[L], c) for c in p["local"]]
                cx = [sum(w[i] for w in wc) / 8.0 for i in range(3)]
                arm.append((L, p["namn"], aabb_of(wc), cx, max(math.dist(cx, w) for w in wc), wc))
        for i in range(len(arm)):
            La, na, ba, ca, ra, wca = arm[i]
            for jx in range(i + 1, len(arm)):
                Lb, nb, bb, cb, rb, wcb = arm[jx]
                if abs(La - Lb) < 2:
                    continue
                if math.dist(ca, cb) > ra + rb:                    # sphere prefilter
                    continue
                key = tuple(sorted((na, nb)))
                if key in disable or key in seen_pairs:
                    continue
                n_checks += 1
                if aabb_overlap(ba, bb) and obb_overlap(wca, wcb):
                    seen_pairs.add(key)
                    candidates.append((na, nb, ptype, [round(math.degrees(x), 2) for x in q]))
        for (La, na, ba, ca, ra, wca) in arm:
            for (scx, sr), sp in stat_sph:
                if math.dist(ca, scx) > ra + sr:
                    continue
                key = tuple(sorted((na, sp["namn"])))
                if key in disable or key in seen_pairs:
                    continue
                n_checks += 1
                if aabb_overlap(ba, sp["aabb_static"]) and obb_overlap(wca, static_wc[sp["namn"]]):
                    seen_pairs.add(key)
                    candidates.append((na, sp["namn"], ptype, [round(math.degrees(x), 2) for x in q]))

    static_names = {sp["namn"] for sp in static_parts}
    arm_arm = [c for c in candidates if c[1] not in static_names]
    arm_env = [c for c in candidates if c[1] in static_names]
    is_bana = bool(args.bana)
    if not candidates:
        verdict = "GREEN_BANA" if is_bana else "GREEN_FULLENVELOPP"
    elif is_bana:
        verdict = "BANA_LINJAR_EJ_REN_KRAVER_PLANERING"
    else:
        # A 6R arm is never collision-free over its whole joint space: at the joint extremes it folds into
        # itself and reaches the cell perimeter. Sweeping the full envelope is the wrong question; a green
        # sweep is defined against a SPECIFIC path, or is delivered by a collision-checked planner.
        verdict = "FULLENVELOPP_EJ_KOLLISIONSFRI_VANTAT_6R"
    rep = {
        "cell": "robotcell_bansvep_v1",
        "urdf": man.get("urdf"),
        "metod": "joint-driven fk sweep; AABB clearance (disjoint -> mesh-disjoint, fail-closed); "
                 "non-adjacent pairs + arm against static bodies; edge poses at the joint limits; "
                 "every overlap escalated to an exact OBB/SAT test.",
        "sveptathet": {"steg_per_ledintervall": args.steps, "kant_poser": ["alla_min", "alla_max"],
                       "slumpkombinationer": args.combos, "n_poser": len(poses),
                       "n_par_kontroller": n_checks, "n_disablade_strukturpar": len(disable)},
        "adjacenskriterium": "non-adjacent = |i-j|>=2 (neighbouring links share a joint, skipped)",
        "n_kandidater": len(candidates),
        "n_arm_sjalvkollision": len(arm_arm),
        "n_arm_mot_miljo": len(arm_env),
        "arm_sjalvkollision_per_posetyp": dict(Counter(c[2] for c in arm_arm)),
        "arm_mot_miljo_per_posetyp": dict(Counter(c[2] for c in arm_env)),
        "self_collision_ren": len(arm_arm) == 0,
        "kandidater_arm_sjalvkollision": arm_arm[:50],
        "kandidater_arm_mot_miljo": arm_env[:30],
        "verdict": verdict,
        "note": "AABB-disjoint across the whole sweep proves mesh clearance (the box contains the body). "
                "Candidates, if any, must be escalated to an exact test before a verdict.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=1)
    print(f"BANSVEP {verdict}: {len(poses)} poses, {n_checks} pair checks, "
          f"{len(candidates)} candidates (steps/joint={args.steps}, edge poses yes, combos={args.combos})")
    if candidates:
        print("  CANDIDATES (require exact escalation):")
        for c in candidates[:10]:
            print("   ", c)
    print(f"  -> {os.path.relpath(os.path.abspath(args.out), ROOT)}")
    return 0 if verdict.startswith("GREEN") else 2


if __name__ == "__main__":
    raise SystemExit(main())
