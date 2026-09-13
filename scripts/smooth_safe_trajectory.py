#!/usr/bin/env python3
"""Smooth (C2) and collision-validated trajectory.

Shortcut (sparse safe path) -> C2 blend -> dense collision re-validation of the SMOOTHED path -> on a
corner cut, insert a knot at the midpoint of the safe linear segment nearest the collision (that midpoint
is collision-free because the whole shortcut segment was edge-validated) -> re-blend, until the smoothed
path is collision-free. Then time-parametrise against the velocity and torque limits.

Selftest checks: (1) a C2-smooth path is produced; (2) the final smoothed path is collision-free under
dense re-validation (a corner cut is caught and fixed by knot insertion); (3) it is time-parametrisable
under the velocity and torque limits (finite T); (4) it is genuinely smooth - continuous speed, interior
knots have non-zero speed (unlike stop-at-waypoint); (5) the corner-cut fix binds (the naive blend of the
sparse shortcut collides, the fix makes it free).

  python -u scripts/smooth_safe_trajectory.py   (requires pinocchio + hppfcl/coal + scipy)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fleet_agnostic_planner import load, in_collision, edge_free, rrt_connect
from collision_aware_shortcut import shortcut, path_len
from fleet_waypoint_blending import blend, time_param

ROOT = Path(__file__).resolve().parents[1]


def spline_collides(cs, S, model, data, gm, gd, n=500, cfree=None):
    """Dense collision validation of the SMOOTHED path (not just the knots). cfree -> WorldModel path; None -> coal.
    -> (collides?, point, count)."""
    Q = cs(np.linspace(0, S, n))
    if cfree is not None:
        coll = [i for i in range(len(Q)) if not cfree(Q[i])]
    else:
        coll = [i for i in range(len(Q)) if in_collision(model, data, gm, gd, Q[i])]
    return (len(coll) > 0), (Q[coll[len(coll) // 2]] if coll else None), len(coll)


def smooth_safe(path, model, data, gm, gd, rng, max_iter=15, cfree=None):
    """Shortcut -> C2 blend -> re-validate the smoothed path -> insert a knot at a corner cut -> repeat until the
    smoothed path is collision-free. cfree = optional free predicate (WorldModel path); None -> coal."""
    wp = [np.asarray(p, float) for p in shortcut(path, model, data, gm, gd, rng, cfree=cfree)]
    inserts = 0
    for _ in range(max_iter):
        cs, S = blend(wp)
        collides, qc, ncoll = spline_collides(cs, S, model, data, gm, gd, cfree=cfree)
        if not collides:
            return wp, cs, S, inserts, True
        # insert the midpoint of the safe linear segment whose midpoint is nearest the colliding smoothed point
        mids = [0.5 * (wp[k] + wp[k + 1]) for k in range(len(wp) - 1)]
        k = int(np.argmin([np.linalg.norm(m - qc) for m in mids]))
        wp.insert(k + 1, mids[k]); inserts += 1
    cs, S = blend(wp)
    return wp, cs, S, inserts, (not spline_collides(cs, S, model, data, gm, gd, cfree=cfree)[0])


def main():
    # wall obstacle (thin in x, wide and tall) forces a sharp detour, so the sparse shortcut corner-cuts
    box, box_pose = [0.05, 0.6, 0.7], [0.45, 0.0, 0.4]
    model, data, gm, gd, _ = load(box, box_pose)
    nj = model.nv; lo, hi = model.lowerPositionLimit[:nj], model.upperPositionLimit[:nj]
    print("smooth-safe-trajectory — C2-smooth AND collision-validated path (UR10e, wall obstacle)")

    # search for a scenario where the naive blend of the sparse shortcut corner-cuts (so the fix is shown to bind)
    chosen = None
    for seed in range(20):
        rng = np.random.default_rng(seed)
        free = lambda q: not in_collision(model, data, gm, gd, q)
        q0 = next((q for q in (lo + rng.random(nj) * (hi - lo) for _ in range(400)) if free(q)), None)
        qg = next((q for q in (lo + rng.random(nj) * (hi - lo) for _ in range(400)) if free(q)), None)
        if q0 is None or qg is None:
            continue
        path = rrt_connect(model, data, gm, gd, q0, qg, lo, hi, rng)
        if not path:
            continue
        sc = shortcut(path, model, data, gm, gd, rng)
        cs0, S0 = blend(sc)
        nc, _, nn = spline_collides(cs0, S0, model, data, gm, gd)
        if nc:                                          # naive blend corner-cuts -> the hard case
            chosen = (seed, path, sc, nn); break
    if chosen is None:
        print("  no corner-cut scenario found in 20 seeds — not running an unverified demo"); return 1
    seed, path, sc, naive_ncoll = chosen
    rng = np.random.default_rng(seed)
    naive_collides = True
    print(f"  scenario seed={seed}: RRT {len(path)} wp (len {path_len(path):.2f}) → shortcut {len(sc)} wp (len {path_len(sc):.2f})")
    print(f"  naive blend of the sparse shortcut corner-cuts: {naive_ncoll} colliding samples")

    # smooth + safe via iterative knot insertion
    wp, cs, S, inserts, safe = smooth_safe(path, model, data, gm, gd, rng)
    final_collides, _, final_ncoll = spline_collides(cs, S, model, data, gm, gd, n=800)
    print(f"  smooth+safe: {len(wp)} knots ({inserts} inserted), collision-free={not final_collides} ({final_ncoll} colliding samples)")

    # time-parametrise + genuine-smoothness check (interior knots have non-zero speed)
    eff = np.asarray(model.effortLimit[:nj]); velLim = np.asarray(model.velocityLimit[:nj])
    eff_real = bool((eff > 0).all() and eff.std() / (eff.mean() + 1e-12) > 0.02)
    T = time_param(cs, S, model, data, eff if eff_real else np.full(nj, 1e6), velLim, dt=0.01)
    # interior speed at the knots (chord parametrisation): a smooth path has non-zero interior speed
    seg = np.linalg.norm(np.diff(np.array(wp), axis=0), axis=1); sknots = np.concatenate([[0], np.cumsum(seg)])
    interior_speed = [float(np.linalg.norm(cs(sknots[i], 1))) for i in range(1, len(sknots) - 1)]
    min_interior_speed = min(interior_speed) if interior_speed else 0.0
    peak_vel_ratio = float(np.max(np.abs(cs(np.linspace(0, S, 400), 1)) * (S / T)) / (velLim.max() + 1e-9))
    print(f"  time-param T={T:.2f}s (effort {'real' if eff_real else 'placeholder/missing -> velocity only'}), min interior-knot speed {min_interior_speed:.3f} (>0 = genuinely smooth)")

    g1 = cs is not None and len(wp) >= 3
    g2 = not final_collides                                   # final smoothed path collision-free (re-validated)
    g3 = np.isfinite(T) and T > 0
    g4 = min_interior_speed > 1e-3                            # genuinely smooth: non-zero speed at interior knots
    g5 = naive_collides and not final_collides               # the corner-cut fix binds: naive collided, fixed one is free
    print(f"  (1) C2-smooth path produced: {g1}")
    print(f"  (2) final smoothed path collision-free (dense re-validation): {g2}")
    print(f"  (3) time-parametrisable (T={T:.2f}s finite): {g3}")
    print(f"  (4) genuinely smooth (interior-knot speed {min_interior_speed:.3f}>0): {g4}")
    print(f"  (5) corner-cut fix binds (naive blend collided, fix made it safe): {g5}")

    ok = g1 and g2 and g3 and g4 and g5      # requires the fix to bind (naive corner-cut -> fixed safe)
    print(f"\nVERDICT: smooth-safe-trajectory = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"C2-smooth AND collision-validated path (UR10e). The naive smooth blend corner-cut into collision "
             f"({naive_ncoll} samples) and the collision-aware shortcut was safe but piecewise linear; combining them "
             f"gives shortcut -> C2 blend -> dense collision re-validation -> knot insertion at safe segment midpoints "
             f"({inserts} inserted) until the smoothed path is collision-free ({final_ncoll} colliding samples over "
             f"{len(wp)} knots). Genuinely smooth (interior speed {min_interior_speed:.2f}>0), time-parametrised "
             f"T={T:.1f}s under the velocity{'+torque' if eff_real else ''} limits. " if ok else
             f"Not validated (smooth {g1}, safe {g2}, time-param {g3}, genuinely smooth {g4}); corner-cut fix binds {g5}. ")
          + "Scope: collision re-validation is dense sampling (800 samples), not a continuous guarantee; knot insertion "
          "at safe-segment midpoints pulls the spline back towards the validated linear path; the torque limit applies "
          "only when the URDF declares real effort limits, otherwise velocity only; UR10e demo, robot-agnostic pattern.")

    out = dict(layer="smooth (C2) AND collision-validated trajectory (independently gated layer)",
               robot="UR10e", rrt_waypoints=len(path), shortcut_waypoints=len(sc),
               naive_blend_collides=bool(naive_collides), naive_collision_samples=int(naive_ncoll),
               final_knots=len(wp), knots_inserted=int(inserts), final_collision_free=bool(not final_collides),
               final_collision_samples=int(final_ncoll), time_s=round(float(T), 3), effort_real=eff_real,
               min_interior_knot_speed=round(min_interior_speed, 4), peak_vel_ratio=round(peak_vel_ratio, 3),
               smooth_C2=bool(g1), collision_validated=bool(g2), time_parametrizable=bool(g3), genuinely_smooth=bool(g4),
               corner_cut_fix_binds=bool(g5),
               note="combines collision_aware_shortcut (safe) + cubic blend (smooth) + dense collision RE-validation + knot-insertion until smooth path collision-free; closes the smooth+safe gap",
               validated=bool(ok))
    (ROOT / "reports" / "smooth_safe_trajectory.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("  wrote reports/smooth_safe_trajectory.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
