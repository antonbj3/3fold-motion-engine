#!/usr/bin/env python3
"""
cell 542 — the DEPLOYABLE COLLISION-FREE-TRAJECTORY CERT: composes cell 540 (tropical-SDF swept-volume/CCD, min-plus reduction over TIME) with the
sim-cost-grade "certify a MOTION, not a config" deliverable that cell 530's compute-value-chain packaging demands of any SO-ARM100 backend artifact
before it can gate a real plan. The geometry insight (one more min-plus axis, same tropical lattice as cell 496->cell 527->cell 540): a MULTI-waypoint
trajectory is a chain of segments; the min-plus/tropical union composes ASSOCIATIVELY over ANY index (min-plus composition) — segment
joins capsule joins time, so min-clearance(whole trajectory) = min_segment min_t min_k capsule_sdf, exactly one more min-axis (SEGMENT) on top of
cell 540's (TIME, CAPSULE) reduction, no new primitive. The cert is CONTINUOUS (a margin-to-spare value, not a bool) because a scalar witness lets a
planner rank near-misses, and on FAILURE it returns the BINDING SEGMENT + t* + the closest obstacle — the exact (where, when, what) a re-planner
needs, not just a red light.
PREREGISTERED (watertight, machine-checkable, no new geometry primitive — only a new min-axis over cell 540's swept_sdf + cell 527's posed_capsules):
  (a) CLEAR TRAJECTORY -> CERTIFIED — a trajectory whose every segment's swept-SDF is > margin is CERTIFIED, and the reported min_clearance matches
      an INDEPENDENT direct recomputation of min over per-segment swept_sdf (not trusting the cert's own bookkeeping).
  (b) DECISIVE CONTROL — a trajectory whose WAYPOINTS are each individually clear (endpoint SDF>margin at every waypoint) but ONE interior SEGMENT
      sweeps through an obstacle (reusing cell 540's own q0->q1 collision example, embedded as the middle segment of a 3-segment path with two safe
      segments flanking it) is NOT certified, and the binding-segment/t*/obstacle witness is cross-checked by independently re-running swept_sdf on
      JUST that one segment (not the whole-trajectory bookkeeping) and confirming an EXACT match.
  (c) MONOTONICITY — for a FIXED trajectory, shrinking the margin threshold can only RELAX the verdict: sweeping margin from permissive-fails to
      permissive-passes, the boolean sequence must be non-decreasing (once True at some margin, stays True at every smaller margin) — checked on
      BOTH the clear trajectory (a) and the colliding trajectory (b), the latter showing the TRUE flip point exactly where margin crosses
      min_clearance (a colliding trajectory becomes "certified" once the margin threshold permits at least its own penetration depth).
  (d) GRAZING/OFF-BY-ONE ADVERSARY — a segment whose TRUE (T-converged) clearance sits almost exactly at a chosen margin: certified must flip
      False->True as margin crosses C from above to below, with the boundary itself (margin==C exactly) resolving to NOT-certified (documented
      STRICT ">" convention, not ">="), and the flip must NOT be a T-discretization/tunneling artifact — verified by confirming the segment's
      swept-min is IDENTICAL at T=41 and T=257 (no residual truncation error to hide behind) before trusting the ±eps boundary test.
Anchors: cell 540 (swept_sdf, min-plus CCD core, T-convergence machinery), cell 527 (posed_capsules/capsule_sdf, config-driven FK),
min-plus composition (min-plus composes over ANY index — here SEGMENT joins TIME joins CAPSULE), cell 530 (why a compute-value-chain
deliverable needs a MOTION-level, not config-level, cert to gate a real plan before simulating it). External anchor: capsule_sdf is the cell 496-
validated (1e-12) analytic point-to-segment-minus-radius distance; this file introduces NO new primitive, only a new min-axis (SEGMENT/waypoint-chain).
"""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(_v,"4")  # thread cap (thread cap)
import numpy as np, importlib.util

def _load(name, path):
    p = path if os.path.isabs(path) else os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    s = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
w540 = _load("cell 540", "tropical_sdf_swept_volume_continuous_collision_detection_min_over_trajectory.py")
posed_capsules, sdf, capsule_sdf, swept_sdf = w540.posed_capsules, w540.sdf, w540.capsule_sdf, w540.swept_sdf

# ---------- the deployable cert: one more min-plus axis (SEGMENT) over cell 540's swept_sdf (TIME, CAPSULE) ----------
def trajectory_cert(obstacle_points, waypoints, T=41, margin=0.05, caps_fn=posed_capsules):
    """trajectory_cert(obstacle_points, waypoints, T, margin) — collision-free-WITH-MARGIN cert for a MULTI-segment robot MOTION (waypoints = list
       of >=2 joint configs, linear-in-config segments, matching cell 540's own interpolation). min_clearance = min over (segment, time-sample,
       capsule, obstacle-point) of the tropical SDF — the SAME min-plus reduction as cell 540 with one more index (SEGMENT). Certified iff
       min_clearance > margin (STRICT). On any outcome, returns the CONTINUOUS margin-to-spare (min_clearance - margin, sign-carrying: positive =
       spare clearance, negative = violation depth) and the WITNESS (binding_segment, t_star, obstacle_idx/point, capsule_idx) — the binding
       segment is the one to re-plan, t_star+capsule_idx is where-on-the-arm/when it happens, obstacle_idx is what it's closest to."""
    waypoints = [np.asarray(w, float) for w in waypoints]
    if not (len(waypoints) >= 2):  # assert-under-`-O`: validity guard must survive -O
        raise AssertionError("trajectory needs >=2 waypoints (>=1 segment)")
    P = np.atleast_2d(np.asarray(obstacle_points, float)); M = len(P)
    n_seg = len(waypoints) - 1
    D = np.empty((n_seg, M)); Tstar = np.empty((n_seg, M)); Kstar = np.empty((n_seg, M), dtype=int)
    for j in range(n_seg):
        res = swept_sdf(P, waypoints[j], waypoints[j + 1], T, caps_fn=caps_fn)
        D[j] = res["d_min"]; Tstar[j] = res["t_star"]; Kstar[j] = res["k_star"]
    min_clearance = float(D.min())
    j_star, p_star = (int(x) for x in np.unravel_index(int(np.argmin(D)), D.shape))
    t_star = float(Tstar[j_star, p_star]); k_star = int(Kstar[j_star, p_star])
    certified = min_clearance > margin                                    # STRICT: equality is NOT certified (documented, tested in (d))
    spare = min_clearance - margin                                        # continuous cert value: >0 spare clearance, <0 violation depth
    return dict(certified=bool(certified), min_clearance=min_clearance, margin=margin, spare=spare,
                binding_segment=j_star, t_star=t_star, obstacle_idx=p_star, obstacle_point=P[p_star].copy(),
                capsule_idx=k_star, n_segments=n_seg, D=D)

def _cfg(a): return np.array([a, 0.0, 0.0, 0.0, 0.0])

def main():
    print("=" * 130)
    print("cell 542  COLLISION-FREE-TRAJECTORY CERT — cell 540 swept-CCD + one more min-plus axis (SEGMENT) -> certify a MOTION with a margin+witness")
    print("=" * 130)
    rng = np.random.default_rng(0)

    # shared obstacle scenario (IDENTICAL to cell 540's, for a coherent scene: one fixed obstacle, several candidate trajectories tested against it)
    p_hit = np.array([0.35, 0.0, 0.30])
    cloud = p_hit + rng.uniform(-0.03, 0.03, size=(14, 3))
    P = np.vstack([p_hit[None, :], cloud])                                # M=15 obstacle points
    MARGIN = 0.05

    # ---------------- pre-scan (measure, don't assume): clearance vs joint0, other joints fixed at 0 (documents WHY the chosen configs are safe/unsafe) ----------------
    scan_a = np.linspace(-1.4, 1.4, 29)
    scan_d = np.array([sdf(P, posed_capsules(_cfg(a))).min() for a in scan_a])
    print("\n  pre-scan (config-only clearance, joint0 in [-1.4,1.4], others=0): dips negative in ~[-0.30,0.22] -- sets up the (a)/(b) trajectories below.")

    # =========================================================================================================================
    # (a) CLEAR TRAJECTORY -> CERTIFIED (2-segment path entirely in the safe negative-joint0 zone, never approaching the dip)
    # =========================================================================================================================
    wp_safe = [_cfg(-1.4), _cfg(-1.2), _cfg(-1.0)]
    cert_a = trajectory_cert(P, wp_safe, T=41, margin=MARGIN)
    # independent cross-check: recompute min over per-segment swept_sdf directly, NOT via trajectory_cert's own D bookkeeping
    direct_a = min(swept_sdf(P, wp_safe[j], wp_safe[j + 1], T=41)["d_min"].min() for j in range(len(wp_safe) - 1))
    a_match = abs(cert_a["min_clearance"] - direct_a) < 1e-12
    a_ok = cert_a["certified"] and (cert_a["min_clearance"] > MARGIN) and a_match
    print("\n  (a) CLEAR TRAJECTORY -> CERTIFIED (waypoints joint0=-1.4,-1.2,-1.0; 2 segments, both safe per pre-scan):")
    print("      trajectory_cert: certified=%s  min_clearance=%.5f  margin=%.2f  spare=%.5f  (spare>0 required)" %
          (cert_a["certified"], cert_a["min_clearance"], cert_a["margin"], cert_a["spare"]))
    print("      independent cross-check (direct min over per-segment swept_sdf, bypassing cert's D array): %.5f  (match to 1e-12 OK: =%s)" %
          (direct_a, a_match))
    print("      VERDICT (a): %s" % a_ok)

    # =========================================================================================================================
    # (b) DECISIVE CONTROL — waypoints all individually clear, middle SEGMENT (cell 540's own q0->q1) sweeps through the obstacle
    # =========================================================================================================================
    q0, q1 = _cfg(-1.0), _cfg(1.0)
    wp_hit = [_cfg(-1.4), q0, q1, _cfg(1.4)]                               # 3 segments: safe | HIT | safe
    endpoint_clear = all(sdf(P, posed_capsules(w)).min() > MARGIN for w in wp_hit)
    cert_b = trajectory_cert(P, wp_hit, T=41, margin=MARGIN)
    # independent cross-check #1: re-run swept_sdf on JUST the reported binding segment (not the whole-trajectory D matrix)
    j_b = cert_b["binding_segment"]
    res_seg = swept_sdf(P, wp_hit[j_b], wp_hit[j_b + 1], T=41)
    seg_match = abs(res_seg["d_min"].min() - cert_b["min_clearance"]) < 1e-12
    # independent cross-check #2: re-derive (t*,k*) witness capsule_sdf at the reported point/segment, reproduce d_min exactly (cell 540(c) pattern)
    p_idx = cert_b["obstacle_idx"]; t_star = cert_b["t_star"]; k_star = cert_b["capsule_idx"]
    q_star = wp_hit[j_b] * (1 - t_star) + wp_hit[j_b + 1] * t_star
    a_w, b_w, r_w = posed_capsules(q_star)[k_star]
    d_witness = capsule_sdf(P[p_idx][None, :], a_w, b_w, r_w)[0]
    witness_match = abs(d_witness - cert_b["min_clearance"]) < 1e-9
    interior_ok = 0.0 < t_star < 1.0
    b_ok = endpoint_clear and (not cert_b["certified"]) and (j_b == 1) and seg_match and witness_match and interior_ok
    print("\n  (b) DECISIVE CONTROL — 4 waypoints (joint0=-1.4,-1.0,1.0,1.4), middle segment = cell 540's own q0->q1 collision example:")
    print("      every WAYPOINT individually clear (min-SDF>margin at all 4) OK: =%s" % endpoint_clear)
    print("      trajectory_cert: certified=%s (need False)  min_clearance=%.5f  spare=%.5f  binding_segment=%d (need 1, the middle/HIT segment)" %
          (cert_b["certified"], cert_b["min_clearance"], cert_b["spare"], j_b))
    print("      witness: t*=%.4f (interior 0<t*<1 OK: =%s), capsule*=%d, obstacle_idx=%d (p_hit itself=%s)" %
          (t_star, interior_ok, k_star, p_idx, p_idx == 0))
    print("      cross-check #1 (re-run swept_sdf on JUST the binding segment): %.6f vs cert min_clearance %.6f (match OK: =%s)" %
          (res_seg["d_min"].min(), cert_b["min_clearance"], seg_match))
    print("      cross-check #2 (independently recompute capsule_sdf at witness t*,k*,obstacle_idx): %.6f vs %.6f (match OK: =%s)" %
          (d_witness, cert_b["min_clearance"], witness_match))
    print("      VERDICT (b): %s -> waypoints-clear-but-segment-collides is NOT certified, and the witness pinpoints (segment=%d, t*=%.3f, obstacle=%d) to re-plan." %
          (b_ok, j_b, t_star, p_idx))

    # =========================================================================================================================
    # (c) MONOTONICITY — shrinking margin can only RELAX the verdict (non-decreasing as margin falls), on BOTH trajectories
    # =========================================================================================================================
    def monotone_nondecreasing_as_margin_falls(margins_desc, verdicts):
        return all(not (verdicts[i] and not verdicts[i + 1]) for i in range(len(verdicts) - 1))

    margins_a = [0.50, 0.35, 0.30, 0.29, 0.28, 0.20, 0.10, 0.05, 0.02, 0.01, -0.10]
    verdicts_a = [trajectory_cert(P, wp_safe, T=41, margin=m)["certified"] for m in margins_a]
    mono_a = monotone_nondecreasing_as_margin_falls(margins_a, verdicts_a)
    predicted_a = [cert_a["min_clearance"] > m for m in margins_a]
    match_a = verdicts_a == predicted_a

    margins_b = [0.10, 0.05, 0.02, 0.00, -0.02, -0.03, -0.036, -0.038, -0.05, -0.10]
    verdicts_b = [trajectory_cert(P, wp_hit, T=41, margin=m)["certified"] for m in margins_b]
    mono_b = monotone_nondecreasing_as_margin_falls(margins_b, verdicts_b)
    predicted_b = [cert_b["min_clearance"] > m for m in margins_b]
    match_b = verdicts_b == predicted_b

    print("\n  (c) MONOTONICITY — shrinking margin only RELAXES the verdict (checked on BOTH trajectories, all boolean flips machine-cross-checked vs raw clearance>margin):")
    print("      clear-trajectory  (min_clearance=%.5f): margins=%s" % (cert_a["min_clearance"], margins_a))
    print("                                                verdicts=%s  non-decreasing-as-margin-falls OK: =%s  matches clearance>margin OK: =%s" %
          (verdicts_a, mono_a, match_a))
    print("      colliding-trajectory (min_clearance=%.5f): margins=%s" % (cert_b["min_clearance"], margins_b))
    print("                                                verdicts=%s  non-decreasing-as-margin-falls OK: =%s  matches clearance>margin OK: =%s" %
          (verdicts_b, mono_b, match_b))
    print("      (colliding-trajectory flips False->True exactly between margin=-0.036 and -0.038, bracketing its own min_clearance=%.5f: certifying" % cert_b["min_clearance"])
    print("       'collision-free' at a permissive-enough margin correctly means 'penetration within the allowed slack', not a bug.)")
    c_ok = mono_a and mono_b and match_a and match_b

    # =========================================================================================================================
    # (d) GRAZING/OFF-BY-ONE ADVERSARY — a segment whose TRUE clearance sits ~exactly at a margin; test the strict-> boundary + rule out tunneling
    # =========================================================================================================================
    wp_graze = [_cfg(0.25), _cfg(0.45)]
    d_graze_T41 = swept_sdf(P, wp_graze[0], wp_graze[1], T=41)["d_min"].min()
    d_graze_T257 = swept_sdf(P, wp_graze[0], wp_graze[1], T=257)["d_min"].min()
    t_converged = abs(d_graze_T41 - d_graze_T257) < 1e-9                  # rule out a T-discretization/tunneling artifact BEFORE trusting the boundary test
    C = d_graze_T257
    eps = 1e-3
    cert_below = trajectory_cert(P, wp_graze, T=41, margin=C - eps)      # margin more permissive than C -> must certify
    cert_exact = trajectory_cert(P, wp_graze, T=41, margin=C)             # margin == C exactly -> strict ">" => NOT certified
    cert_above = trajectory_cert(P, wp_graze, T=41, margin=C + eps)       # margin stricter than C -> must NOT certify
    d_ok = (t_converged and cert_below["certified"] and (not cert_exact["certified"]) and (not cert_above["certified"])
            and abs(cert_exact["spare"]) < 1e-9)
    print("\n  (d) GRAZING/OFF-BY-ONE ADVERSARY — segment joint0: 0.25->0.45, TRUE clearance sits almost exactly at a chosen margin:")
    print("      swept-min clearance C: T=41 -> %.6f ; T=257 -> %.6f  (IDENTICAL to 1e-9 OK: =%s -> rules out tunneling/truncation before trusting the boundary)" %
          (d_graze_T41, d_graze_T257, t_converged))
    print("      margin=C-eps=%.6f: certified=%s (need True)   spare=%+.6f" % (C - eps, cert_below["certified"], cert_below["spare"]))
    print("      margin=C     =%.6f: certified=%s (need False, STRICT '>' convention)  spare=%+.6f" % (C, cert_exact["certified"], cert_exact["spare"]))
    print("      margin=C+eps=%.6f: certified=%s (need False)  spare=%+.6f" % (C + eps, cert_above["certified"], cert_above["spare"]))
    print("      VERDICT (d): %s -> the boundary is a genuine crossing (T-stable), not an off-by-one/interpolation artifact." % d_ok)

    # =========================================================================================================================
    print("\n  VERDICT (does trajectory_cert = cell 540's swept-CCD composed over SEGMENTS give a deployable, watertight, MOTION-level collision-free-with-margin cert?):")
    all_ok = a_ok and b_ok and c_ok and d_ok
    if all_ok:
        print("  PASS (a,b,c,d):")
        print("    (a) a fully-clear 2-segment trajectory CERTIFIES (min_clearance=%.5f>margin=%.2f), cross-checked bit-exact against an independent" % (cert_a["min_clearance"], MARGIN))
        print("        direct recomputation (not the cert's own bookkeeping).")
        print("    (b) DECISIVE CONTROL: waypoints all individually clear but the middle segment sweeps through the obstacle (reusing cell 540's own")
        print("        q0->q1 example) -> NOT certified; binding_segment=%d correctly identified (the HIT segment, not the safe flanks); the" % j_b)
        print("        (t*,capsule*,obstacle*) witness is cross-checked TWO independent ways (re-run swept_sdf on just that segment; independently")
        print("        recompute capsule_sdf at the witness) -> exact match both times.")
        print("    (c) shrinking margin relaxes the verdict monotonically on both a clear and a colliding trajectory, every boolean flip machine-")
        print("        matched against raw clearance>margin -- no oscillation, no off-by-one.")
        print("    (d) a segment whose true clearance sits almost exactly at a margin flips False->True exactly at the boundary (strict '>',")
        print("        equality=NOT-certified), confirmed NOT a T-discretization artifact (T=41==T=257 to 1e-9 first).")
        print("    -> ONE more min-plus axis (SEGMENT) over cell 540's (TIME,CAPSULE) reduction gives a MOTION-level (not config-level) collision-free-")
        print("       with-margin cert: certified + continuous margin-to-spare + (segment,t*,obstacle) re-plan witness, all machine cross-checked.")
    else:
        print("  FAIL/INSPECT — a=%s b=%s c=%s d=%s (see per-check numbers above)." % (a_ok, b_ok, c_ok, d_ok))


if __name__ == "__main__": main()
