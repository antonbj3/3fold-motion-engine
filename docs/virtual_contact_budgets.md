# Global coverage with local virtual-work budgets

`motion_engine.ncp.virtual_budget.virtual_budget_enclosure` is an opt-in,
solver-free rational postprocessor for `SparseCoulombBox`. It consumes an
already verified global energy reserve plus caller-supplied virtual fields.
No numerical optimizer is required by this API. A caller can discover fields
with an ordinary convex QP/SOCP and then convert proposed coefficients to
explicit exact rationals; optimizer convergence is not certificate validity.

```python
from motion_engine.ncp.virtual_budget import (
    virtual_budget_enclosure, verify_virtual_budget,
)
answer = virtual_budget_enclosure(
    problem, verified_global_reserve,
    energy_fields=[rational_energy_field],
    budget_fields=[rational_local_field],
    clusters=[("body-normal-load", contact_indices)],
)
assert verify_virtual_budget(problem, answer)
```

Every field is checked against every contact. A field that meets the local
target but gives a negative exterior margin is rejected. Clusters name sets
of contact indices; the returned quantity is the sum of normal force
magnitudes in N, rather than a signed vector resultant. Clusters may overlap,
so their bounds must not be summed to infer a global total without accounting
for contacts counted more than once.

The theorem uses frozen J, positive diagonal M, h>0, gaps in a hard box,
nonnegative normal compliance, and circular Coulomb or a dissipative law with
the same componentwise friction bound. Let v_f be free velocity and

    a_i(f) = J_ni f - μ_i (|J_t1i f| + |J_t2i f|),
    B_i = max(0, -g_lo_i/h).

Normal complementarity and tangential dissipation give

    vᵀM(v-v_f) ≤ Σ B_i p_ni.

For an energy field e with every a_i(e)>=B_i, virtual work bounds the right
side by eᵀM(v-v_f). Completing the square gives

    ||v-c||_M ≤ R,
    c=(v_f+e)/2, R=||v_f-e||_M/2.

Each global ball covers every solution at every gap, including every circular
slip direction. Coordinate projections are intersected with the initial
reserve. R is an exact rational upper square-root bound, never an unchecked
floating point norm.

For a budget field f with all a_i(f)>=0,

    Σ a_i(f) p_ni ≤ fᵀM(v-v_f)
                  ≤ S(f)=fᵀM(c-v_f)+R||f||_M.

The minimum S across all verified balls is a valid budget. For a_i(f)>0 this
gives p_ni<=S/a_i(f). For a cluster C with positive minimum margin it gives
Σ_C p_ni<=S/min_C a_i(f). Zero fields and exterior zero margins require no
division. Per-contact bounds and each cluster sum are intersected with every
available budget. Thus adding verified fields can only tighten the returned
intervals. This is ordinary weak duality and energy completion; method novelty
is not claimed.

The status remains `OUTER_ENCLOSURE`, `all_solutions_covered=True`,
`existence=UNPROVED`, `sharp=False`. Neither global gap-wise existence nor
sharp extrema nor simultaneously realizable marginal endpoints follow from
this relaxation. Unsupported or contradictory supplied certificates raise
`ValueError`; there is no fallback that silently admits invalid local work.
`verify_virtual_budget` performs solver-free replay of every proof and output.

Rigid body geometry, candidate inventory and frames are frozen. Parameters
μ and compliance remain constitutive closures; no anatomical calibration,
continuum trajectory guarantee or new-contact guarantee is supplied. The
heap study improved selected cluster load caps but failed its practical force
and motion widths; standard QP/SOCP with rational postchecking supplied the
same guarantees. This API exposes a useful representation, without a claim of
tenfold speedup or superiority to those methods.
