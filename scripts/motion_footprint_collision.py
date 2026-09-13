"""FOOTPRINT-AWARE collision — an elongated robot fits gaps its circumscribed circle cannot.

The σ-safe planner treats the robot as a POINT inflated by a margin — for a circular robot that is exact, but for an
ELONGATED one (a car, an AGV, a forklift) the circumscribed-circle margin is badly over-conservative: it blocks a gap the
robot drives straight through, because the conservative radius is the half-DIAGONAL while only the half-WIDTH faces the gap.
Checking the actual ORIENTED footprint (the robot rectangle aligned to its heading) against the obstacles is exact — the
robot is feasible iff its swept footprint is clear — and unlocks paths the point+circle model refuses.

GATES (null/control each):
 (G0) ELONGATED ROBOT + NARROW GAP — a 0.6×0.3 m robot driving a corridor of width 0.4 m (walls at ±0.2 m).
 (G1) CIRCUMSCRIBED CIRCLE COLLIDES — the point+circle model (radius = half-diagonal 0.335 m) overlaps the walls → it calls the corridor INFEASIBLE.
 (G2) FOOTPRINT CLEARS — the oriented footprint (half-width 0.15 m faces the gap) clears the 0.2 m walls → feasible, the path the circle refused.
 (G3) EXACT, NOT PERMISSIVE — a wider robot (0.5 m, half-width 0.25 > 0.2) is correctly REJECTED by the same footprint check.

"""
import sys
import numpy as np

L, W = 0.6, 0.3                                                   # robot length, width
GAP = 0.4                                                        # corridor width (walls at ±GAP/2)
PATH = np.column_stack([np.linspace(-1, 1, 21), np.zeros(21)])  # straight through the corridor, heading +x


def wall_points():
    x = np.linspace(-1, 1, 60)
    return np.vstack([np.column_stack([x, np.full(60, GAP / 2)]), np.column_stack([x, np.full(60, -GAP / 2)])])


def in_rect(pt, center, heading, hL, hW):
    d = pt - center; cos, sin = np.cos(heading), np.sin(heading)
    return abs(d[0] * cos + d[1] * sin) < hL and abs(-d[0] * sin + d[1] * cos) < hW


def footprint_collides(walls, hL, hW):
    for c in PATH:
        if any(in_rect(w, c, 0.0, hL, hW) for w in walls):       # heading +x along the corridor
            return True
    return False


def circle_collides(walls, radius):
    return any(min(np.linalg.norm(w - c) for c in PATH) < radius for w in walls)


def main():
    print("=" * 98)
    print("FOOTPRINT-AWARE collision — an elongated robot fits gaps its circumscribed circle can't")
    print("=" * 98)
    walls = wall_points()
    r_circ = 0.5 * np.hypot(L, W)
    g0 = L > W and GAP > W and GAP < 2 * r_circ
    print(f"\n(G0) ELONGATED ROBOT + NARROW GAP — robot {L}×{W}m, corridor {GAP}m (walls ±{GAP/2}m); circumscribed radius {r_circ:.3f}m: {'PASS' if g0 else 'FAIL'}")

    g1 = circle_collides(walls, r_circ)
    print(f"\n(G1) CIRCLE COLLIDES — point+circle (radius {r_circ:.3f}m) overlaps the ±{GAP/2}m walls → calls the corridor INFEASIBLE: {'PASS' if g1 else 'FAIL'}")

    g2 = not footprint_collides(walls, L / 2, W / 2)
    print(f"\n(G2) FOOTPRINT CLEARS — oriented footprint (half-width {W/2}m < wall {GAP/2}m) is collision-free → FEASIBLE (the path the circle refused): {'PASS' if g2 else 'FAIL'}")

    g3 = footprint_collides(walls, 0.6 / 2, 0.5 / 2)
    print(f"\n(G3) EXACT — a wider 0.5m robot (half-width 0.25 > {GAP/2}m wall) IS rejected by the same footprint check: {'PASS' if g3 else 'FAIL'}")

    allok = g0 and g1 and g2 and g3
    print("\n" + "=" * 98)
    if allok:
        print("VERDICT: the conservative point+circle model blocks a corridor the robot drives straight through — its radius is the half-DIAGONAL")
        print(f"  ({r_circ:.2f}m), but only the half-WIDTH ({W/2}m) faces the {GAP}m gap. The oriented-footprint check is exact: the {W}m-wide robot clears the")
        print(f"  ±{GAP/2}m walls and is feasible, while a {0.5}m-wide robot is correctly rejected. Footprint-aware collision unlocks tight-space paths for")
        print(f"  elongated robots that a circular margin refuses — the body, not a point, swept along the path.")
    else:
        print(f"VERDICT: NOT all pass — G0 {g0} G1 {g1} G2 {g2} G3 {g3}. Fix at SOURCE.")
    print("=" * 98)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
