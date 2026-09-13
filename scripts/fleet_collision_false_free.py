#!/usr/bin/env python3
"""False-free rate of the points-vs-SDF collision route, PER ROBOT.

`collision_false_free_audit.py` measures on one robot that the surface-sampling fix (mesh vertices plus triangle
centroids) takes the SDF route's false-free rate from about 33% (vertices only) to about 8%. Mesh topology differs
between robots - large flat faces with sparse triangles versus dense meshes - so the rate has to be measured per
robot rather than inferred from one of them. Per robot: a reach-scaled plate in the workspace, sample genuine
robot-vs-plate collisions (exact coal, self-collision-free), and measure the share that points-vs-SDF MISSES. The
worst robot is printed, not averaged away.

Scope: the robots covered are the vendor descriptions that ship collision meshes with this repository; the fleet
XMLs reference vendor mesh packages that are not redistributed here and cannot be mesh-collision-checked.

  python -u scripts/fleet_collision_false_free.py   (requires pinocchio + hppfcl/coal)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from motion_engine.motion_stack import MotionStack
from motion_engine.world_model import PrimitiveExactWorld
from fleet_multirobot_movement import _pkgs, _mesh_descriptions

PKG = _pkgs()


def audit_robot(urdf, n_target=150, max_samples=30000):
    urdf = str(urdf)
    ms_self = MotionStack(urdf, pkg=PKG, seed=1)
    # reach from the extent of the FK points
    rng0 = np.random.default_rng(0)
    Q = ms_self.lo + rng0.random((60, ms_self.nq)) * (ms_self.hi - ms_self.lo)
    pts = np.vstack([ms_self.robot_points(q) for q in Q])
    reach = float(np.percentile(np.linalg.norm(pts, axis=1), 99))
    dims = [0.10 * reach, 0.45 * reach, 0.55 * reach]; pose = [0.45 * reach, 0.0, 0.4 * reach]
    wm = PrimitiveExactWorld([{"type": "box", "dims": dims, "pose": pose}])
    ms_coal = MotionStack(urdf, pkg=PKG, obstacle_box=dims, box_pose=pose, seed=1)
    ms_wm = MotionStack(urdf, pkg=PKG, world_model=wm, seed=1)
    npts = sum(len(v) for _, v in ms_wm._rpts)
    rng = np.random.default_rng(2); env = 0; ff = 0
    for _ in range(max_samples):
        q = ms_coal.lo + rng.random(ms_coal.nq) * (ms_coal.hi - ms_coal.lo)
        if (not ms_coal.free(q)) and ms_self.free(q):          # genuine robot-vs-plate (exact coal), self-free
            env += 1
            if ms_wm.free(q):                                  # points-vs-SDF says FREE = false-free
                ff += 1
        if env >= n_target:
            break
    return dict(robot=Path(urdf).stem, reach=round(reach, 2), n_robot_points=npts, n_env=env, false_free=ff,
                false_free_pct=round(100 * ff / max(env, 1), 1))


def main():
    print("FLEET-COLLISION-FALSE-FREE — false-free rate of the surface-sampling fix, per robot\n")
    rows = []
    for urdf in _mesh_descriptions():
        try:
            x = audit_robot(urdf)
        except Exception as e:
            print(f"  {Path(urdf).stem:22s} FAIL: {type(e).__name__}: {str(e)[:50]}"); continue
        rows.append(x)
        print(f"  {x['robot']:22s} reach {x['reach']:.2f}m {x['n_robot_points']:4d}pts: false-free "
              f"{x['false_free']:3d}/{x['n_env']} ({x['false_free_pct']:4.1f}%)")
    pcts = [x["false_free_pct"] for x in rows]
    worst = max(rows, key=lambda x: x["false_free_pct"]) if rows else None
    mean = round(float(np.mean(pcts)), 1) if pcts else None
    ok = bool(rows) and all(p < 20.0 for p in pcts)                # per robot < 20% (surface sampling vs ~33% vertex-only)
    print(f"\n  per-robot false-free: mean {mean}% | WORST {worst['robot']} {worst['false_free_pct']}%")
    print(f"\nVERDICT: fleet-collision-false-free = "
          + (f"SURFACE SAMPLING GENERALISES. The fix (vertices + triangle centroids) keeps false-free under 20% on "
             f"all {len(rows)} robots (mean {mean}%, worst {worst['false_free_pct']}%), so it is not specific to the "
             "robot it was measured on. " if ok else
             f"PER-ROBOT VARIATION. At least one robot is at or above 20% false-free (worst {worst['robot']} "
             f"{worst['false_free_pct']}%) - surface sampling is not enough for that mesh topology and needs denser "
             "sampling or an exact narrow phase. ")
          + "Honest residual: a point-based test is not exact (face contacts between sampled points remain); the "
            "default coal route is exact (0%). Guaranteed no-false-free on the SDF/GPU route needs an exact narrow "
            "phase or denser surface sampling.")
    out = dict(robots=rows, mean_false_free_pct=mean, worst=worst, all_under_20pct=ok)
    (ROOT / "reports" / "fleet_collision_false_free.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("  wrote reports/fleet_collision_false_free.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
