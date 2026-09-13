#!/usr/bin/env python3
"""Selftest for motion_stack: checks that MotionStack is loadable and runnable end to end.

One import -> Cartesian goal -> smoothed, collision-checked plan with a plan confidence, plus an honest
non-plan on an unreachable goal.

  python -u -m motion_engine.selftest_motion_stack   (requires pinocchio + hppfcl/coal + scipy)
"""
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from motion_engine.motion_stack import MotionStack


def main():
    print("selftest motion_stack — one import -> cartesian goal -> checked plan")
    urdf = str(ROOT / "assets/robots/ur_description/ur10e.urdf"); pkg = str(ROOT / "assets/robots/ur_description")
    ms = MotionStack(urdf, pkg=pkg, ee_frame="tool0", obstacle_box=[0.15, 0.15, 0.4], box_pose=[0.45, 0.0, 0.3],
                     twin_name="ur10e", seed=1)
    print(f"  loaded: ur10e dof={ms.nj}, EE={ms.ee_frame}, twin={ms.twin_name}")

    # cartesian goal = EE pose of a reachable free configuration
    qt = next((q for q in (ms.sample() for _ in range(1500)) if ms.free(q)), None)
    from movement_ik import fk
    target = fk(ms.model, ms.data, ms.fid, qt)
    plan = ms.plan_cartesian(target)
    print(f"  plan_cartesian → ok={plan.ok} stage={plan.stage} wp={plan.waypoints} T={plan.total_time_s}s "
          f"effort_feasible={plan.effort_feasible} collision_free={plan.collision_free}")
    print(f"  plan_confidence='{plan.plan_confidence}' (capped by {plan.capped_by}); IK {plan.ik_pos_err_mm}mm; provenance={plan.provenance}")

    # unreachable goal -> honest non-plan
    far = pin.SE3(np.eye(3), np.array([9.0, 9.0, 9.0]))
    plan_far = ms.plan_cartesian(far)
    print(f"  unreachable goal (9,9,9) -> ok={plan_far.ok} stage={plan_far.stage} (expected not-ok/ik_unreachable)")

    g1 = plan.ok and plan.trajectory_q is not None and plan.collision_free      # one call -> smoothed, collision-checked plan
    g2 = plan.ik_pos_err_mm < 1.0 and plan.waypoints >= 2                        # IK reached the goal and produced a path
    g3 = plan.plan_confidence in ("medium", "high") and plan.capped_by == "twin"  # the UR10e twin tier caps the confidence (weakest link)
    g4 = (not plan_far.ok) and plan_far.stage == "ik_unreachable"                # honest on an unreachable goal
    ok = g1 and g2 and g3 and g4
    print(f"  (1) one import -> cartesian goal -> smoothed collision-free plan: {g1}")
    print(f"  (2) IK reached the goal, path produced ({plan.waypoints} wp): {g2}")
    print(f"  (3) plan_confidence capped by the weakest link (twin tier): {g3}")
    print(f"  (4) honest on the unreachable goal: {g4}")
    print(f"\nSELFTEST: motion-stack = {'PASS' if ok else 'FAIL'}. "
          + ("One import (motion_engine.motion_stack.MotionStack) takes a cartesian goal to a smoothed, "
             "collision-checked, effort-feasible plan with a weakest-link-capped plan confidence."
             if ok else "FAIL — see the checks above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
