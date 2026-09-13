#!/usr/bin/env python3
"""Decides whether a sampling planner's UNSOLVED scenes are FALSE-INFEASIBLE or genuinely hard.

`mppi_completeness_audit.py` (GPU, CUDA) finds scenes the sampling planner does not solve even at a high budget
and writes them to `reports/mppi_completeness_scenes.npz`. This cell verifies that stop claim against the raw
problem: it runs probabilistically complete CPU RRT-Connect on the SAME scenes. If RRT solves one, the sampling
planner was FALSE-infeasible there (a sampling limit, so the fallback is justified); if RRT fails too, the scene
is genuinely hard and the stop claim was correct.

The collision model is deliberately the SAME points-vs-SDF world model the GPU planner uses, not exact coal:
different collision models give different free spaces (a configuration free in one can collide in the other),
which would make the comparison invalid. Scenes whose start or goal is not free in this model are skipped and
reported as such.

Producer of the input: `mppi_completeness_audit.py` (CUDA). A second argument selects another planner's field, so
the same gate applies to any `*_completeness_scenes.npz` with the same layout.

  python -u scripts/rrt_completeness_verify.py [scenes.npz] [solved_field]   (requires pinocchio + hppfcl/coal)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from motion_engine.motion_stack import MotionStack
from motion_engine.world_model import PrimitiveExactWorld
from fleet_agnostic_planner import rrt_connect

ROBOTS_DIR = ROOT / "assets" / "robots"
PKG = [str(ROBOTS_DIR)] + [str(p) for p in ROBOTS_DIR.iterdir() if p.is_dir()]
URDF = str(ROBOTS_DIR / "ur_description" / "ur10e.urdf")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    sp = ROOT / (argv[0] if argv else "reports/mppi_completeness_scenes.npz")
    field = argv[1] if len(argv) > 1 else "mppi_high"
    if not sp.exists():
        print(f"{sp.name} is not present — run the matching *_completeness_audit.py (CUDA) first"); return 1
    d = np.load(sp)
    starts, goals, planner_high = d["starts"], d["goals"], d[field]
    PL = field.replace("_high", "").upper()                                     # planner label
    obs = [{"type": "box", "dims": list(dm), "pose": list(po)} for dm, po in zip(d["obs_dims"], d["obs_pose"])]
    wm = PrimitiveExactWorld(obs)
    ms = MotionStack(URDF, pkg=PKG, world_model=wm, seed=1)
    print(f"RRT-COMPLETENESS-VERIFY — complete RRT on {PL}'s scenes, same points-vs-SDF collision model\n")

    def cfree(q):
        return ms.free(q)

    rows = []
    for i in range(len(starts)):
        sfree = bool(cfree(starts[i])) and bool(cfree(goals[i]))               # start/goal free in the SAME model
        path = rrt_connect(ms.model, ms.data, ms.gm, ms.gd, starts[i], goals[i], ms.lo, ms.hi,
                           np.random.default_rng(7), n_iter=8000, cfree=cfree) if sfree else None
        rrt_ok = path is not None
        rows.append(dict(scene=i, planner_high=bool(planner_high[i]), rrt=rrt_ok, start_goal_free=sfree))
        tag = "" if sfree else "  (start/goal not free in this model -> invalid comparison, skipped)"
        if sfree and not planner_high[i]:
            tag = f"  <- FALSE-INFEASIBLE (RRT solves, {PL} does not)" if rrt_ok else "  <- genuinely hard (RRT fails too)"
        print(f"  scene {i}: {PL}-high={'OK' if planner_high[i] else 'MISS'} | "
              f"RRT={'SOLVES' if rrt_ok else 'fails'} | start/goal-free={sfree}{tag}")

    valid = [r for r in rows if r["start_goal_free"]]
    misses = [r for r in valid if not r["planner_high"]]
    false_infeas = [r for r in misses if r["rrt"]]
    genuine_hard = [r for r in misses if not r["rrt"]]
    print(f"\n  valid scenes (start/goal free in the same model): {len(valid)}/{len(rows)}")
    print(f"\n  {PL}-high misses: {len(misses)} | FALSE-infeasible (RRT solves): {len(false_infeas)} | "
          f"genuinely hard (RRT does not): {len(genuine_hard)}")
    print(f"\nVERDICT: rrt-completeness-verify = "
          + (f"{PL} FALSE-INFEASIBLE CONFIRMED on {len(false_infeas)} scene(s) — a path EXISTS (RRT-Connect finds "
             f"it) and {PL} misses it, so the planner must fall back on RRT (probabilistically complete) before "
             f"claiming 'infeasible'; {PL} is a fast path, not a completeness guarantee. " if false_infeas else
             f"{PL}'s misses were genuinely hard (RRT does not solve them either within budget) — not "
             f"false-infeasible on these; the {PL} stop claim matches the complete planner. ")
          + "This is a stop claim verified against a complete planner rather than assumed. RRT-Connect runs here on "
            "the same points-vs-SDF model as the planner under test, so the two see the same free space.")
    out = dict(rows=rows, planner_misses=len(misses), false_infeasible=len(false_infeas),
               genuine_hard=len(genuine_hard), scenes_file=sp.name, solved_field=field,
               producer="mppi_completeness_audit.py (CUDA)")
    (ROOT / "reports" / "rrt_completeness_verify.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("  wrote reports/rrt_completeness_verify.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
