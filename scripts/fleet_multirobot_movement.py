#!/usr/bin/env python3
"""Exercises the motion stack on more than one robot vendor, not just one arm.

A planner that is structurally robot-agnostic still has to be EXERCISED on more than the robot it was written
against. This cell runs the actual stack (pinocchio FK + exact-mesh coal collision + RRT-Connect through the
planner contract) on one representative per vendor description that ships with collision meshes, so planning plus
self- and environment collision are shown to work per robot rather than assumed.

Scope of the shipped default: this repository redistributes collision meshes for two vendor descriptions
(`ur_description`, `franka_description`); the 165 fleet XMLs reference `package://<vendor>_support/meshes/...`
mesh packages that are not redistributed here, so they are usable for kinematics, limits and inertials but not
for mesh collision. The representatives are therefore auto-discovered from the descriptions that do carry
meshes, and `--robots` takes explicit URDF paths when more vendor mesh packages are available locally.

Selftest checks: (1) every representative loads (model + geometry) and RRT finds a path; (2) every planned path
is collision-free (self + environment, exact coal); (3) collision BINDS - a configuration forced against a large
workspace block is detected, per robot; (4) at least two distinct vendor descriptions are exercised. A robot that
fails to load or plan is reported, not hidden.

  python -u scripts/fleet_multirobot_movement.py   (requires pinocchio + hppfcl/coal)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import hppfcl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fleet_agnostic_planner import in_collision, edge_free, rrt_connect

ROOT = Path(__file__).resolve().parents[1]
ROBOTS_DIR = ROOT / "assets" / "robots"


def _pkgs():
    """Every package root under assets/robots, so `package://` and relative mesh paths resolve per vendor."""
    return sorted({str(ROBOTS_DIR)} | {str(p) for p in ROBOTS_DIR.iterdir() if p.is_dir()})


def _mesh_descriptions():
    """URDFs whose vendor description directory actually carries collision meshes (auto-discovered)."""
    out = []
    for d in sorted(p for p in ROBOTS_DIR.iterdir() if p.is_dir()):
        if not any(d.rglob("meshes")):
            continue
        out += sorted(str(u) for u in d.glob("*.urdf"))
    return out


def _build(urdf, pkgs, obstacle, box_pose):
    model = pin.buildModelFromUrdf(str(urdf))
    gm = pin.buildGeomFromUrdf(model, str(urdf), pin.GeometryType.COLLISION, package_dirs=pkgs)
    data = model.createData()
    nrobot = gm.ngeoms
    pj = [g.parentJoint for g in gm.geometryObjects]
    cand = [(i, j) for i in range(nrobot) for j in range(i + 1, nrobot) if abs(pj[i] - pj[j]) >= 2]
    for (i, j) in cand:
        gm.addCollisionPair(pin.CollisionPair(i, j))
    # Prune "always colliding" self pairs, the same rule motion_stack.py uses: some non-adjacent links have nested
    # or oversized COLLISION meshes by design (wrist and hand links), so a naive adjacency rule leaves 0% free
    # configurations on those robots. Pairs colliding in nearly every sample are disabled (MoveIt practice).
    gdtmp = gm.createData()
    K = 80; cnt = np.zeros(len(cand), int); rng0 = np.random.default_rng(0)
    loq, hiq = model.lowerPositionLimit, model.upperPositionLimit
    for _ in range(K):
        q = loq + rng0.random(model.nq) * (hiq - loq)
        pin.computeCollisions(model, data, gm, gdtmp, q, False)
        for k in range(len(cand)):
            if gdtmp.collisionResults[k].isCollision():
                cnt[k] += 1
    always = {cand[k] for k in range(len(cand)) if cnt[k] >= 0.95 * K}
    if always:
        gm.removeAllCollisionPairs()
        for (i, j) in cand:
            if (i, j) not in always:
                gm.addCollisionPair(pin.CollisionPair(i, j))
    obj = pin.GeometryObject("obstacle", 0, pin.SE3(np.eye(3), np.asarray(box_pose)), hppfcl.Box(*obstacle))
    oid = gm.addGeometryObject(obj)
    for g in range(nrobot):
        gm.addCollisionPair(pin.CollisionPair(g, oid))
    return model, data, gm, gm.createData(), nrobot


def exercise(urdf, pkgs, rng):
    urdf = Path(urdf)
    name = urdf.stem
    if not urdf.exists():
        return dict(robot=name, loaded=False, err="no URDF")
    try:
        model, data, gm, gd, ngeoms = _build(urdf, pkgs, [0.2, 0.2, 0.5], [0.5, 0.0, 0.3])
    except Exception as e:
        return dict(robot=name, loaded=False, err=f"{type(e).__name__}:{str(e)[:50]}")
    nq = model.nq
    lo, hi = model.lowerPositionLimit[:nq], model.upperPositionLimit[:nq]

    def free(q): return not in_collision(model, data, gm, gd, q)
    q0 = next((q for q in (lo + rng.random(nq) * (hi - lo) for _ in range(800)) if free(q)), None)
    qg = next((q for q in (lo + rng.random(nq) * (hi - lo) for _ in range(800)) if free(q)), None)
    if q0 is None or qg is None:
        return dict(robot=name, loaded=True, ngeoms=ngeoms, nq=nq, planned=False, err="no free start/goal")
    t0 = time.time()
    path = rrt_connect(model, data, gm, gd, q0, qg, lo, hi, rng)
    dt = time.time() - t0
    path_free = path is not None and all(edge_free(model, data, gm, gd, path[i], path[i + 1], res=0.05)
                                         for i in range(len(path) - 1))
    # collision binds: a configuration forced against a large workspace block must be detected
    mB, dB, gmB, gdB, _ = _build(urdf, pkgs, [2.0, 2.0, 2.0], [0.0, 0.0, 0.3])
    binds = any(in_collision(mB, dB, gmB, gdB, lo + rng.random(nq) * (hi - lo)) for _ in range(50))
    return dict(robot=name, vendor=urdf.parent.name, loaded=True, ngeoms=ngeoms, nq=nq, planned=path is not None,
                nodes=len(path) if path else 0, plan_s=round(dt, 2), collision_free=bool(path_free),
                collision_binds=bool(binds))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", nargs="*", default=None, help="URDF paths (default: descriptions with meshes)")
    a = ap.parse_args(argv)
    reps = a.robots if a.robots else _mesh_descriptions()
    pkgs = _pkgs()
    rng = np.random.default_rng(0)
    print(f"FLEET-MULTIROBOT-MOVEMENT — stack exercised on {len(reps)} vendor descriptions; package_dirs={len(pkgs)}")
    res = [exercise(r, pkgs, rng) for r in reps]
    for r in res:
        if r.get("planned"):
            print(f"  OK  {r['robot']:24s} dof{r['nq']} {r['ngeoms']}geoms | path {r['nodes']}n {r['plan_s']}s | "
                  f"collision-free={r['collision_free']} binds={r['collision_binds']}")
        else:
            print(f"  --  {r['robot']:24s} loaded={r.get('loaded')} — {r.get('err','not planned')}")

    ok_robots = [r for r in res if r.get("planned") and r.get("collision_free")]
    vendors = sorted({r.get("vendor", "?") for r in ok_robots})
    g1 = bool(res) and len(ok_robots) == len(res)               # every representative plans collision-free
    g2 = all(r["collision_free"] for r in res if r.get("planned"))
    g3 = all(r.get("collision_binds") for r in ok_robots) and bool(ok_robots)
    g4 = len(vendors) >= 2
    print(f"  (1) every representative plans collision-free: {g1} ({len(ok_robots)}/{len(res)})")
    print(f"  (2) all planned paths collision-free: {g2}")
    print(f"  (3) collision binds per robot (workspace block detected): {g3}")
    print(f"  (4) at least two vendor descriptions exercised: {g4} ({', '.join(vendors)})")

    ok = g1 and g2 and g3 and g4
    print(f"\nVERDICT: fleet-multirobot-movement = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"The motion stack (pinocchio + coal RRT, self and environment collision) is EXERCISED on "
             f"{len(ok_robots)} vendor descriptions ({', '.join(vendors)}); all plan collision-free against exact "
             "coal and collision binds on each. Fleet generalisation of the planner is exercised, not only claimed. "
             if ok else
             f"Not validated ({len(ok_robots)}/{len(res)} planned and free; g1 {g1} g2 {g2} g3 {g3} g4 {g4}). ")
          + "Scope: planning and collision are KINEMATIC/geometric here; per-robot DYNAMICS needs per-robot effort "
            "limits and validated friction. The obstacle is placed generically. The number of vendors is bounded by "
            "the collision meshes redistributed with this repository - pass --robots to exercise more. Composes "
            "fleet_agnostic_planner (coal + RRT).")

    rel = [str(Path(r).resolve().relative_to(ROOT)) if str(r).startswith(str(ROOT)) else str(r) for r in reps]
    out = dict(layer="multi-robot movement-stack exercise", representatives=rel,
               results=res, n_planned_free=len(ok_robots), vendors_exercised=vendors,
               note="planning + collision exercised per vendor description that ships collision meshes; per-robot "
                    "dynamics (torque gate) still structural (effort-limit and validated-friction gaps)",
               validated=bool(ok))
    (ROOT / "reports" / "fleet_multirobot_movement.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("  wrote reports/fleet_multirobot_movement.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
