#!/usr/bin/env python3
"""PERCEPTION → MOTION SAFETY LOOP — a plan on the occlusion-aware world is ground-truth-SAFE; the naive plan crashes.

Closes the loop across the session: the motion planner consumes a WorldModel occupancy; the perception observed-mask
(perception_observed_mask_occlusion.py) produces one. A NAIVE world (obstacles = observed surfaces only) calls the
occlusion shadow FREE, so the shortest plan cuts through it — straight into a HIDDEN obstacle the sensor never saw.
The OBSERVED-MASK world (unknown shadow → obstacle) forces the plan to route through SEEN-free space → ground-truth
safe. This is robot motion at the trajectory layer (plan→validate), graded against the analytic ground truth.

GATES (null/control each):
 (G1) NAIVE PLAN CRASHES — the path planned on the naive world collides with the GT (hidden) obstacle it could not
      see. [the perception error becomes a motion catastrophe]
 (G2) OBSERVED-MASK PLAN IS SAFE — the path planned on the observed-mask world has ZERO GT-collisions (it routes
      through seen-free space). NULL = the naive plan.
 (G3) NOT VACUOUS — the observed-mask plan still REACHES the goal (a safe route exists and is found); it is not
      trivially safe by refusing to move.
 (G4) ROBUST — naive-crashes / observed-mask-safe holds under ±sensor-pose shifts.

"""
import sys, os, heapq
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from perception_observed_mask_occlusion import raycast_visibility

NX, NY = 80, 48
START, GOAL = (5, 40), (74, 12)


def gt_scene():
    occ = np.zeros((NX, NY), bool)
    occ[36:42, 14:23] = True                      # small occluder OFF the start→goal path; it shadows the region up-right of it
    occ[50:62, 18:33] = True                      # HIDDEN obstacle, in that shadow, ON the naive geodesic
    sensor = np.array([2.0, 2.0])                 # bottom-left sensor
    return occ, sensor


def astar(free, start, goal):
    if not (free[start] and free[goal]):
        return None
    nbr = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]
    h = lambda a, b: np.hypot(a[0] - b[0], a[1] - b[1])
    openq = [(h(start, goal), 0.0, start)]; came = {}; g = {start: 0.0}; seen = set()
    while openq:
        _, gc, cur = heapq.heappop(openq)
        if cur == goal:
            p = [cur]
            while cur in came:
                cur = came[cur]; p.append(cur)
            return p[::-1]
        if cur in seen:
            continue
        seen.add(cur)
        for di, dj in nbr:
            ni, nj = cur[0] + di, cur[1] + dj
            if 0 <= ni < NX and 0 <= nj < NY and free[ni, nj]:
                if di and dj and not (free[cur[0] + di, cur[1]] and free[cur[0], cur[1] + dj]):
                    continue                                  # no diagonal corner-cutting
                ng = gc + np.hypot(di, dj)
                if ng < g.get((ni, nj), 1e18):
                    g[(ni, nj)] = ng; came[(ni, nj)] = cur
                    heapq.heappush(openq, (ng + h((ni, nj), goal), ng, (ni, nj)))
    return None


def gt_collisions(path, occ_gt):
    return 0 if path is None else int(sum(occ_gt[i, j] for i, j in path))


def worlds(occ_gt, sensor):
    of, surf, unk = raycast_visibility(occ_gt, sensor)
    naive_free = ~surf                                        # naive: only seen surfaces block
    obs_free = ~(surf | unk)                                  # observed-mask: unknown shadow blocks
    return naive_free, obs_free


def main():
    print("=" * 96)
    print("PERCEPTION → MOTION SAFETY LOOP — the occlusion-aware plan is GT-safe; the naive plan crashes")
    print("=" * 96)
    occ_gt, sensor = gt_scene()
    of_, surf_, unk_ = raycast_visibility(occ_gt, sensor)
    hid = np.zeros((NX, NY), bool); hid[50:62, 18:33] = True
    print(f"  [diag] hidden obstacle in shadow: {100*np.mean(unk_[hid&occ_gt]):.0f}% | goal observed: {not unk_[GOAL]} | start free: {(~surf_)[START]}")
    naive_free, obs_free = worlds(occ_gt, sensor)
    p_naive = astar(naive_free, START, GOAL)
    p_obs = astar(obs_free, START, GOAL)
    col_naive = gt_collisions(p_naive, occ_gt)
    col_obs = gt_collisions(p_obs, occ_gt)
    print(f"\n  start {START} → goal {GOAL}.  naive plan: {len(p_naive) if p_naive else 0} cells, {col_naive} GT-collisions | "
          f"observed-mask plan: {len(p_obs) if p_obs else 0} cells, {col_obs} GT-collisions")

    g1_ok = col_naive > 0
    print(f"\n(G1) NAIVE PLAN CRASHES — the plan on the naive world drives through {col_naive} ground-truth obstacle cells")
    print(f"     it never observed (the occlusion shadow it assumed free): {'PASS' if g1_ok else 'FAIL'}")

    g2_ok = (p_obs is not None) and col_obs == 0
    print(f"\n(G2) OBSERVED-MASK PLAN IS SAFE — the plan on the occlusion-aware world has {col_obs} GT-collisions (NULL = naive {col_naive})")
    print(f"     -> treating unknown as obstacle routes the robot through SEEN-free space, ground-truth-safe: {'PASS' if g2_ok else 'FAIL'}")

    g3_ok = (p_obs is not None) and p_obs[-1] == GOAL
    print(f"\n(G3) NOT VACUOUS — the observed-mask plan still REACHES the goal (a safe route exists and is found): "
          f"{'PASS' if g3_ok else 'FAIL'}")

    holds = []
    for ds in ([0, 3], [0, -3], [3, 0]):
        nf, of2 = worlds(occ_gt, sensor + np.array(ds, float))
        pn, po = astar(nf, START, GOAL), astar(of2, START, GOAL)
        holds.append(gt_collisions(pn, occ_gt) > 0 and po is not None and gt_collisions(po, occ_gt) == 0)
    g4_ok = all(holds)
    print(f"\n(G4) ROBUST — naive-crashes / observed-mask-safe-and-reaches holds under ±sensor shifts {sum(holds)}/3")

    allok = g1_ok and g2_ok and g3_ok and g4_ok
    print("\n" + "=" * 96)
    if allok:
        print("VERDICT: PERCEPTION × MOTION composes into a SAFETY LOOP. A motion planner on a NAIVE reconstruction drives")
        print(f"  through {col_naive} ground-truth obstacle cells hidden in the occlusion shadow (it assumed unseen = free). On the")
        print(f"  OBSERVED-MASK world (ray-cast free-space carving; unknown → obstacle) the SAME planner finds a {len(p_obs)}-cell route")
        print(f"  through seen-free space with ZERO GT-collisions, still REACHING the goal — not trivially-safe-by-freezing.")
        print(f"  Epistemic-safe perception → safe motion, end-to-end through the WorldModel contract, GT-anchored.")
    else:
        print(f"VERDICT: NOT all pass — G1 {g1_ok} G2 {g2_ok} G3 {g3_ok} G4 {g4_ok}. Fix at SOURCE.")
    print("=" * 96)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
