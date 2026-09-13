#!/usr/bin/env python3
"""IDENTIFIABILITY GOVERNOR (Component 13) — the generative loop's certification gate.

Contract (DATA in/out):
  Input:  J  (n_obs, n_param)  — constraint Jacobian ∂c/∂x evaluated at candidate design x.
          Each row is ONE linearized constraint (a FEA load case, manufacturing bound, or
          active-subspace projection row). Each column is one design variable.
  Output: IdResult with pre-registered VERDICT in {VERIFIABLE, UNVERIFIABLE, DEGENERATE}.

Rule (pre-registered, do not tune post-hoc):
  UNVERIFIABLE  iff  σ_min(J) / σ_max(J) < EPS_ID_RELATIVE  OR  DOF = n_obs − n_param < DOF_MIN
  DEGENERATE    iff  σ_max(J) < 1e-14  (fully zero J — design not connected to any constraint)
  VERIFIABLE    otherwise

Integration point in the generative loop:
  1. Candidate x is proposed (WFC / AGZ-residual / σ_min-ascent step).
  2. Caller builds J by stacking ∂c_i/∂x for each active constraint c_i at x
     (finite-difference or adjoint from the FEA surrogate).
  3. governor_gate(J) is called BEFORE any σ_min-ascent certification.
  4. If UNVERIFIABLE → reject candidate; push null_directions onto the hard-exclusion
     graph (these are directions the constraint set cannot distinguish — adding them as
     SPOF nodes prevents the search from drifting into unauditable sub-manifolds).
  5. If VERIFIABLE → proceed to σ_min-ascent certification.

The governor ties auditability directly to over-constrained-ness (DOF ≥ 1):
  an over-determined system (n_obs > n_param, full-column-rank J) implies every design
  direction is uniquely observed. A rank-deficient J means the certification is a tautology
  in the null directions — the engine refuses to say PASS on tautological evidence.

Pre-registered thresholds (watertight):
  EPS_ID_RELATIVE = 1e-3   (σ_min / σ_max; relative condition floor)
  DOF_MIN         = 1      (n_obs − n_param ≥ 1; strict over-determinacy)

Falsifier:
  Inject a J with a known null space (rank < n_param) → must return UNVERIFIABLE.
  If governor returns VERIFIABLE for such a J, the test FAILS (not just warns).

  python -u -m motion_engine.identifiability_governor   # selftest (3 cases + adversarial)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# ── Pre-registered thresholds — do NOT change post-hoc ──────────────────────────────
EPS_ID_RELATIVE: float = 1e-3   # σ_min / σ_max < this → UNVERIFIABLE
DOF_MIN: int = 1                 # n_obs − n_param must be ≥ DOF_MIN (strict over-det.)
# ────────────────────────────────────────────────────────────────────────────────────

Verdict = Literal["VERIFIABLE", "UNVERIFIABLE", "DEGENERATE"]


@dataclass
class IdResult:
    """Identifiability certificate for one candidate design point."""
    verdict: Verdict
    sigma_min: float          # smallest singular value of J
    sigma_max: float          # largest singular value of J
    condition: float          # σ_max / σ_min  (inf if σ_min = 0)
    dof: int                  # n_obs − n_param  (negative = grossly under-constrained)
    null_rank: int            # dim(null space) = number of sv below threshold
    null_directions: np.ndarray  # (n_param, null_rank); columns = unobservable directions
    eps_used: float = EPS_ID_RELATIVE
    dof_min_used: int = DOF_MIN
    null_kind: str = "NONE"   # remedy split: RANK_NULL/CONDITIONING_NULL/KIND_UNCERTAIN/INSUFFICIENT_DOF/NONE

    def summary(self) -> str:
        return (
            f"[{self.verdict}] σ_min={self.sigma_min:.3e} σ_max={self.sigma_max:.3e} "
            f"cond={self.condition:.2e} dof={self.dof} null_rank={self.null_rank} kind={self.null_kind}"
        )


def governor_gate(
    J: np.ndarray,
    eps_id: float = EPS_ID_RELATIVE,
    dof_min: int = DOF_MIN,
) -> IdResult:
    """Gate every candidate design through the identifiability governor.

    Parameters
    ----------
    J : (n_obs, n_param) constraint Jacobian at candidate x.
    eps_id : relative σ threshold (default = pre-registered EPS_ID_RELATIVE).
    dof_min : minimum required DOF (default = pre-registered DOF_MIN).

    Returns
    -------
    IdResult with verdict VERIFIABLE / UNVERIFIABLE / DEGENERATE.
    Caller must NOT certify a design when verdict != VERIFIABLE.

    ★PRECONDITION (declared-not-measured; the gate CANNOT check it): EVERY row of J must be a REAL active
    constraint's derivative — NOT a regularizer/Tikhonov row, NOT an uninformative padding/leftover row. The
    gate computes σ_min/σ_max of the J it is GIVEN; it cannot tell a real constraint from a spurious row.
    Appending m>=n_param uninformative rows at scale α >= eps_id·σ_max(J_true) lifts EVERY singular value by
    Θ(α) (Weyl) — including the true-null directions — and FALSELY flips UNVERIFIABLE→VERIFIABLE (measured
    2026-07-15: rank-4-null J + 0.01·randn pad -> null_rank 4→0, verdict flips; reproduces on a 7-null instance
    too). So strip regularizer/FD-adjoint padding rows from J BEFORE calling; if you cannot, treat a VERIFIABLE
    that depends on small-scale rows as UNPROVEN.
    """
    J = np.asarray(J, dtype=float)
    if J.ndim != 2:
        raise ValueError(f"J must be 2-D, got shape {J.shape}")

    n_obs, n_param = J.shape
    dof = n_obs - n_param

    # Full SVD — right singular vectors give design-space geometry
    _, sv, Vt = np.linalg.svd(J, full_matrices=True)

    # Trim sv to min(n_obs, n_param) (SVD convention: sv length = min dim)
    sv = sv[:min(n_obs, n_param)]
    sigma_max = float(sv[0]) if len(sv) > 0 else 0.0
    sigma_min = float(sv[-1]) if len(sv) > 0 else 0.0

    # Degenerate: J is numerically zero
    if sigma_max < 1e-14:
        return IdResult(
            verdict="DEGENERATE",
            sigma_min=0.0,
            sigma_max=0.0,
            condition=float("inf"),
            dof=dof,
            null_rank=n_param,
            null_directions=np.eye(n_param),
            eps_used=eps_id,
            dof_min_used=dof_min,
            null_kind="RANK_NULL",   # fully unobserved J = extreme rank deficiency
        )

    condition = sigma_max / max(sigma_min, 1e-300)

    # Null-space: rows of Vt whose corresponding sv is below threshold
    # For rectangular J with n_obs < n_param the trailing (n_param - n_obs) rows of Vt
    # are always in the null space (no sv assigned).
    null_mask_sv = sv < eps_id * sigma_max          # among the min(n_obs,n_param) svs
    null_rank_sv = int(np.sum(null_mask_sv))

    # Structural null from shape (n_param - n_obs columns of V with no sv)
    structural_null = max(0, n_param - n_obs)       # these rows of Vt are always null
    null_rank = null_rank_sv + structural_null

    # Collect null directions: Vt rows for near-zero svs + structural null rows
    near_zero_indices = np.where(null_mask_sv)[0]    # indices into the computed sv block
    struct_indices = np.arange(n_obs, n_param)       # rows of Vt beyond the sv block
    null_row_indices = np.concatenate([near_zero_indices, struct_indices]).astype(int)
    null_directions = Vt[null_row_indices].T         # (n_param, null_rank)

    # UNVERIFIABLE if any null direction exists OR DOF is insufficient
    if null_rank > 0 or dof < dof_min:
        verdict: Verdict = "UNVERIFIABLE"
    else:
        verdict = "VERIFIABLE"

    # Null KIND — the remedy split (acquisition-null taxonomy). ADVISORY ONLY: the verdict
    # above is unchanged by this label (both null kinds stay UNVERIFIABLE), so a mislabel near
    # the boundary can never flip the gate — that is why we emit KIND_UNCERTAIN rather than a
    # sharp rank/conditioning flip.
    #   RANK_NULL         exactly-unobserved direction (or structural under-obs) → unliftable by
    #                     more of the SAME design; ACQUIRE a new modality/constraint.
    #   CONDITIONING_NULL weakly-observed (small-but-nonzero) → liftable by more/better-conditioned
    #                     samples of the same design (1/N partial lift).
    #   KIND_UNCERTAIN    a null sv sits within ~2 decades of the rank floor → rank-vs-conditioning
    #                     is genuinely ambiguous; DISAMBIGUATE before prescribing a remedy.
    #   INSUFFICIENT_DOF  full column rank but DOF < dof_min → identifiable yet not strictly
    #                     over-determined; acquire more observations to reach auditability.
    RANK_TOL = 1e-9          # sv/σ_max below this = numerically rank-deficient (exact null)
    UNC_HI = 1e-7            # [RANK_TOL, UNC_HI) = ambiguous band just above the rank floor
    null_sv = sv[null_mask_sv]
    rel = (null_sv / sigma_max) if len(null_sv) else np.zeros(0)
    has_rank = structural_null > 0 or bool(np.any(rel < RANK_TOL))
    has_uncertain = bool(np.any((rel >= RANK_TOL) & (rel < UNC_HI)))
    has_cond = bool(np.any(rel >= UNC_HI))
    if verdict == "VERIFIABLE":
        null_kind = "NONE"
    elif has_rank:
        null_kind = "RANK_NULL"          # worst-case binds: mixed rank+conditioning → RANK
    elif has_uncertain:
        null_kind = "KIND_UNCERTAIN"
    elif has_cond:
        null_kind = "CONDITIONING_NULL"
    else:
        null_kind = "INSUFFICIENT_DOF"   # null_rank == 0 but dof < dof_min

    return IdResult(
        verdict=verdict,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        condition=condition,
        dof=dof,
        null_rank=null_rank,
        null_directions=null_directions,
        eps_used=eps_id,
        dof_min_used=dof_min,
        null_kind=null_kind,
    )


# ── Generative-loop integration helper ──────────────────────────────────────────────

def exclusion_vectors(result: IdResult) -> np.ndarray:
    """Return the (n_param, null_rank) null directions to push onto the hard-exclusion
    graph when a candidate is UNVERIFIABLE.  Caller adds these as SPOF nodes so the
    search cannot re-enter the unauditable sub-manifold.

    Usage (generative loop sketch):
        r = governor_gate(J)
        if r.verdict != "VERIFIABLE":
            exclusion_graph.add_hard_nodes(exclusion_vectors(r))
            continue   # do not certify
        sigma_min_certify(x)
    """
    return result.null_directions


# ── Selftest ─────────────────────────────────────────────────────────────────────────

def _selftest() -> bool:
    rng = np.random.default_rng(42)
    ok = True
    n_param = 8

    # ── Case 1: rank-4 J (4-dim null space, n_obs = n_param = 8) ──
    # Construct as outer product of two rank-4 matrices → rank ≤ 4
    A = rng.standard_normal((n_param, 4))
    B = rng.standard_normal((4, n_param))
    J_under = A @ B   # (8, 8), rank ≤ 4
    r1 = governor_gate(J_under)
    pass1 = r1.verdict == "UNVERIFIABLE" and r1.null_rank >= 4
    ok = ok and pass1
    print(f"  Case 1 (rank-4 J, n_obs=n_param=8): {r1.summary()}")
    print(f"    -> {'PASS' if pass1 else 'FAIL — expected UNVERIFIABLE with null_rank>=4'}")

    # ── Case 2: full-rank over-determined J (n_obs = n_param + 4 = 12) ──
    J_over = rng.standard_normal((n_param + 4, n_param))
    r2 = governor_gate(J_over)
    pass2 = r2.verdict == "VERIFIABLE" and r2.null_rank == 0 and r2.dof == 4
    ok = ok and pass2
    print(f"  Case 2 (full-rank over-det, n_obs=12, n_param=8): {r2.summary()}")
    print(f"    -> {'PASS' if pass2 else 'FAIL — expected VERIFIABLE, null_rank=0, dof=4'}")

    # ── Case 3: adversarial — one sv = 5e-4 * sv_max (below 1e-3 threshold) ──
    Q, _ = np.linalg.qr(rng.standard_normal((n_param + 2, n_param)))
    sv_target = np.ones(n_param)
    sv_target[-1] = 5e-4           # ratio = 5e-4 < EPS_ID_RELATIVE = 1e-3
    J_adv = Q[:n_param, :] @ np.diag(sv_target)   # (n_param, n_param), full n_obs but near-singular
    # Make it over-determined: stack two independent copies with small perturbation
    J_adv_od = np.vstack([J_adv, J_adv + rng.standard_normal((n_param, n_param)) * 1e-8])
    r3 = governor_gate(J_adv_od)
    ratio3 = r3.sigma_min / max(r3.sigma_max, 1e-300)
    pass3 = r3.verdict == "UNVERIFIABLE" and ratio3 < EPS_ID_RELATIVE
    ok = ok and pass3
    print(f"  Case 3 (adversarial sv_ratio=5e-4): {r3.summary()}")
    print(f"    sv_ratio={ratio3:.2e}  -> {'PASS' if pass3 else 'FAIL — expected UNVERIFIABLE'}")

    # ── Case 4: structurally under-determined (n_obs < n_param) ──
    J_struct = rng.standard_normal((4, n_param))   # 4 constraints, 8 params → dof = -4
    r4 = governor_gate(J_struct)
    pass4 = r4.verdict == "UNVERIFIABLE" and r4.dof == -4 and r4.null_rank >= (n_param - 4)
    ok = ok and pass4
    print(f"  Case 4 (structural under-det, n_obs=4, n_param=8, dof=-4): {r4.summary()}")
    print(f"    -> {'PASS' if pass4 else 'FAIL — expected UNVERIFIABLE, dof=-4'}")

    # ── Case 5: degenerate (zero J) ──
    J_zero = np.zeros((n_param + 2, n_param))
    r5 = governor_gate(J_zero)
    pass5 = r5.verdict == "DEGENERATE"
    ok = ok and pass5
    print(f"  Case 5 (zero J): {r5.summary()}")
    print(f"    -> {'PASS' if pass5 else 'FAIL — expected DEGENERATE'}")

    # ── Case 6: null_kind REMEDY SPLIT — rank-null vs conditioning-null must be DISTINGUISHED ──
    # (D acquisition-null taxonomy: the two need different remedies — acquire-new-modality vs
    #  more-same-data. The gate verdict is UNVERIFIABLE for both; null_kind must separate them.)
    U6, _, Vt6 = np.linalg.svd(rng.standard_normal((n_param + 4, n_param)), full_matrices=False)
    #   rank-null: one column exactly unobserved (sv = 0) → RANK_NULL
    A6 = rng.standard_normal((n_param + 4, n_param - 1))
    J_rank = np.hstack([A6, np.zeros((n_param + 4, 1))])
    r6a = governor_gate(J_rank)
    #   conditioning-null: full rank, one sv small-but-nonzero (ratio 5e-4) → CONDITIONING_NULL
    sv6 = np.linspace(5.0, 1.0, n_param); sv6[-1] = 5e-4 * 5.0
    J_cond = U6 @ np.diag(sv6) @ Vt6
    r6b = governor_gate(J_cond)
    pass6 = (r6a.verdict == "UNVERIFIABLE" and r6a.null_kind == "RANK_NULL"
             and r6b.verdict == "UNVERIFIABLE" and r6b.null_kind == "CONDITIONING_NULL"
             and r1.null_kind == "RANK_NULL"          # rank-4 J from Case 1
             and r2.null_kind == "NONE")              # verifiable → no null
    ok = ok and pass6
    print(f"  Case 6 (null_kind remedy split): rank-null={r6a.null_kind}  cond-null={r6b.null_kind}")
    print(f"    -> {'PASS' if pass6 else 'FAIL — rank vs conditioning null not distinguished'}")

    # ── Falsifier check: if VERIFIABLE returned for rank-deficient J, fail loudly ──
    if r1.verdict == "VERIFIABLE":
        print("  FALSIFIED: rank-deficient J returned VERIFIABLE — governor is broken.")
        ok = False

    summary = "PASS: all 6 cases correct" if ok else "FAIL: see above"
    print(f"\n  -> {summary}")
    print(
        f"  Thresholds: EPS_ID_RELATIVE={EPS_ID_RELATIVE} DOF_MIN={DOF_MIN}"
        f"  (pre-registered, not tuned to pass)"
    )
    return ok


if __name__ == "__main__":
    import sys
    print("IDENTIFIABILITY GOVERNOR selftest (Component 13):")
    sys.exit(0 if _selftest() else 1)
