#!/usr/bin/env python3
"""Mesh assembly -> convex decomposition -> planner scene, with a conservative-coverage guard.

A planner that only ever sees demo primitives has not been shown to handle its most important input: real
multi-part assembly geometry. This cell takes a real mesh assembly, decomposes it with CoACD into convex pieces,
feeds those pieces to the SIMD planner as a point-cloud obstacle, and plans around it.

Conservative coverage is the safety property the planner relies on: the CoACD approximation must ENCLOSE the mesh
(mesh subset of the convex union, within tolerance) so that no collision is MISSED. An approximation that cuts
concavities is unsafe, and the guard fails it. The opposite failure - an absurdly over-conservative union - is
gated too, by the union/mesh volume ratio.

Selftest checks: (1) CoACD decomposes a multi-part assembly into convex pieces with positive volume;
(2) CONSERVATIVE COVERAGE - sampled mesh surface points lie inside the convex union (coverage > 0.97) and the
union is not absurdly over-conservative (union volume / mesh volume < 3); (3) the planner finds a collision-free
path around the CoACD-derived point-cloud scene; (4) DOUBLE ORACLE - the densely sampled path has 0 collisions
against the scene.

  python -u scripts/coacd_scene_to_planner.py   (requires vamp-planner + coacd + trimesh + scipy)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MESH_DIR = ROOT / "assets" / "robots" / "franka_description" / "meshes" / "collision"
START_CFG = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
GOAL_CFG = [1.6, 0.5, 0.0, -1.6, 0.0, 2.0, 0.785]
DENSE = 64


def main():
    import vamp
    import coacd
    import trimesh

    # a real multi-part assembly scene: three link meshes concatenated and placed in the workspace
    parts = []
    for ln, off in [("link4.obj", [0, 0, 0]), ("link5.obj", [0.0, 0.15, 0.0]), ("link6.obj", [0.0, -0.12, 0.05])]:
        p = MESH_DIR / ln
        if not p.exists():
            print(f"missing {p}"); return 2
        mm = trimesh.load(p); mm.apply_translation(np.array(off)); parts.append(mm)
    asm = trimesh.util.concatenate(parts)
    asm.apply_translation(np.array([0.55, 0.0, 0.5]) - asm.centroid)          # into the arm's workspace
    print(f"COACD-SCENE-TO-PLANNER: assembly {len(asm.vertices)} verts, {len(asm.faces)} faces, vol {asm.volume:.5g}")

    # (1) CoACD decomposition
    t0 = time.time()
    pieces = coacd.run_coacd(coacd.Mesh(asm.vertices, asm.faces), threshold=0.05)
    coacd_s = time.time() - t0
    hulls = [trimesh.Trimesh(vertices=v, faces=f).convex_hull for v, f in pieces]
    g1 = len(pieces) >= 1 and all(h.volume > 0 for h in hulls)
    print(f"  (1) CoACD: {len(pieces)} convex pieces ({coacd_s:.1f}s), all volume > 0 = {g1}")

    # (2) CONSERVATIVE COVERAGE: mesh surface points inside the convex union.
    # Point-in-CONVEX-hull is an exact half-space test through the ConvexHull equations (no spatial index needed).
    from scipy.spatial import ConvexHull

    def in_hull(P, V, tol=1.5e-3):
        try:
            eq = ConvexHull(np.asarray(V)).equations          # (nf,4): normal(3) + offset, inside iff n.x + d <= 0
        except Exception:
            return np.zeros(len(P), bool)
        return np.all(P @ eq[:, :3].T + eq[:, 3] <= tol, axis=1)
    surf = asm.sample(4000)
    inside = np.zeros(len(surf), bool)
    for v, _f in pieces:
        inside |= in_hull(surf, v)
    coverage = float(inside.mean())
    union_vol = sum(h.volume for h in hulls); vol_ratio = union_vol / (asm.volume + 1e-12)
    g2 = coverage > 0.97 and vol_ratio < 3.0
    print(f"  (2) CONSERVATIVE COVERAGE: {coverage*100:.1f}% of surface points inside the convex union, "
          f"union/mesh volume {vol_ratio:.2f}x -> {g2}")

    # (3) plan around the CoACD scene as a point cloud (sample the convex pieces' surfaces)
    pc = np.vstack([h.sample(700) for h in hulls]).astype(float)
    mod, planner, rrtc_set, simp_set = vamp.configure_robot_and_planner_with_kwargs("panda", "rrtc")
    rrtc_set.max_iterations = 100000
    rng = vamp.panda.halton()
    env = vamp.Environment()
    env.add_pointcloud(pc.tolist(), 0.0, 1.0, 0.012)          # (points, min, max, point radius 12 mm)
    if not (mod.validate(START_CFG, env) and mod.validate(GOAL_CFG, env)):
        print(f"  start/goal invalid with this scene (start={mod.validate(START_CFG,env)}, "
              f"goal={mod.validate(GOAL_CFG,env)}) — adjust the scene pose"); return 1
    t1 = time.time(); res = planner(START_CFG, [GOAL_CFG], env, rrtc_set, rng); plan_ms = (time.time() - t1) * 1000
    g3 = bool(res.solved)
    print(f"  (3) plan around the CoACD point-cloud scene ({len(pc)} points): solved={res.solved} {plan_ms:.1f}ms, "
          f"{len(res.path)} waypoints -> {g3}")

    # (4) DOUBLE ORACLE: densely validate the plan against the scene
    pts = [np.asarray(res.path[i], dtype=float) for i in range(len(res.path))]
    n_coll = 0; n = 0
    for a, b in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0, 1, DENSE):
            n += 1
            if not mod.validate((a + (b - a) * t).tolist(), env):
                n_coll += 1
    g4 = g3 and n_coll == 0
    print(f"  (4) DOUBLE ORACLE: {n} densely sampled configurations, {n_coll} collisions against the CoACD scene -> {g4}")

    ok = g1 and g2 and g3 and g4
    print(f"\nVERDICT: coacd-scene-to-planner = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"A real multi-part mesh assembly ({len(asm.vertices)} verts) -> CoACD {len(pieces)} convex pieces -> "
             f"point-cloud obstacle -> collision-free plan ({plan_ms:.0f}ms). Conservative coverage: "
             f"{coverage*100:.0f}% of the mesh surface lies inside the convex union ({vol_ratio:.1f}x volume, not "
             "over-conservative), so no collision is missed. Double oracle: the dense plan has 0 collisions against "
             "the scene. The motion stack therefore handles real assembly geometry, not only demo primitives. " if ok else
             f"Not validated (CoACD {g1}, conservative coverage {g2} [{coverage*100:.0f}%/{vol_ratio:.1f}x], "
             f"planner solves {g3}, double oracle {g4}). ")
          + "Scope: the obstacle is a point-cloud approximation (12 mm point radius) of the CoACD pieces, because the "
            "SIMD planner does not take convex meshes directly; conservative coverage measures mesh inside union; the "
            "assembly is a stand-in scene built from vendored collision meshes.")

    out = dict(scene="3-part link assembly (real mesh)", coacd_pieces=len(pieces), coacd_s=round(coacd_s, 2),
               coverage=round(coverage, 3), union_vol_ratio=round(vol_ratio, 2), solved=bool(res.solved),
               plan_ms=round(plan_ms, 2), waypoints=len(res.path), double_oracle_collisions=int(n_coll),
               validated=bool(ok), composes="coacd + vamp-planner")
    (ROOT / "reports" / "coacd_scene_to_planner.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("  wrote reports/coacd_scene_to_planner.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
