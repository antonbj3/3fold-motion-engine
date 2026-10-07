# Exact frozen planar contact inventory

This optional oracle answers a concrete regression question: **how many distinct admissible contact states does this declared reduced model have, and which modes describe each state?** It can supply a complete expected branch set for a solver test, including singular states that a regular local root checker cannot certify. It does not select a physical branch, accept a time step, or export an arbitrary Motion Scene into this model. Existing Krawczyk and step-bundle behavior is unchanged.

Run from a source checkout with SymPy available:

```sh
PYTHONPATH=src python examples/frozen_planar_contact_demo.py
```

The example supplies a true residual and an exact outward derivative enclosure to the existing `certify_root_box`. At its fold the checker correctly returns `UNKNOWN`; the new inventory finds two physical states, one with algebraic multiplicity two. This is useful as a regression oracle for a solver **whose input law matches the model below**. A solver contracted to return one solution has not necessarily failed by returning only one. Completeness is an additional, explicit test requirement.

## Admitted models and API

```python
from fractions import Fraction as Q
from motion_engine.ncp.frozen_planar_contact import (
    ExactEvent, ParameterInterval, inventory_single, inventory_coupled,
)

ordinary = inventory_single(Q(3, 4))
fold = inventory_single(Q(27, 32), eta=Q(27, 64))
canonical_fold = inventory_single(ExactEvent("one_fold"))
seam = inventory_coupled(ExactEvent("pair_PT_TT_seam"))
uncertain = inventory_single(ParameterInterval(Q(84, 100), Q(85, 100)))
```

Single contact binds frozen Gram `[[1,1],[1,2]]`, friction coefficient 2, free tangent velocity 3/4, zero elastic stiffness `e=0`, and fixed normalized step/activation width `h=w=1`. The source law defines eta as an energy and e as elastic stiffness (not restitution). Forces and step impulses have the same numerical values only because this model fixes h=1; this is not a unit-independent identification. For positive gap, the updated normal force is

`f(g) = 2 eta (1-g)^2/g^2` for `0<g<1`, and zero for `g>=1`.

This is the `e=0,w=1` specialization of the updated force documented in [cubic_barrier_risk.md](cubic_barrier_risk.md). `eta` is a strictly positive exact rational; the default is 1/8. Positive slip P has `tau=-2f`, `vt=3/4-3f>=0`, and `g=q-f`. Stick T has `tau=-3/8-f/2`, `vt=0`, `f>=1/4`, and `g=q-3/8+f/2`. Negative slip would give `vt=3/4+5f>0` and is impossible. Inactive I exists at `g=q>=1`. P/T equality describes the same physical state and is deduplicated. A gap and a mode determine the force, tangential impulse and tangent velocity uniquely. At the shared seam those recovered quantities agree.

The coupled model fixes `eta=1/8`, `q2=7/10`, `q1 in [7/10,4/5]`, with coordinate order `(g1,v1,g2,v2)` and Gram

```
1    1    1/10 0
1    2    0    0
1/10 0    1    1
0    0    1    2
```

The two gap equations are `g_i=q_i+f_i+tau_i+(1/10)f_j`; each contact has the single-contact tangential law above. All four active modes PP/PT/TP/TT are covered. Inactive modes are excluded throughout this admitted parameter segment: the other active force is at most 7/20, hence the would-be inactive gap is at most `4/5+(1/10)(7/20)=167/200<1`. Both-inactive and negative slip are excluded as well. This bound must not be reused outside the admitted segment. The Gram is SPD, but that does not imply uniqueness of this nonlinear force model.

## Exact events and distinct states

Event names select internal fixed polynomials and rational isolators, validated on use. Caller-provided names or floating estimates cannot assert event equality. `one_fold` binds eta=1/8. The following coupled inventories resolve the five exact event points inside the previously unresolved guard bands of this fixed one-dimensional family. They do not certify an entire uncertainty band; uncertain interval requests remain `UNKNOWN`:

| Exact event | Distinct physical states | Meaning |
|---|---:|---|
| `pair_inactive_fold` | 7 | Collision in an inadmissible inactive-mode polynomial |
| `pair_PT_TT_seam` | 6 | PT/TT descriptions meet |
| `pair_PP_TP_seam_low` | 4 | PP/TP descriptions meet |
| `pair_PP_TP_seam_high` | 2 | PP/TP descriptions meet |
| `pair_inactive_seam` | 1 | Inadmissible inactive-mode seam at q1=3/4 |

These are not five smooth physical folds. The active roots at these five events are simple; shared mode labels are deduplicated. A genuine smooth fold is `eta=27/64,q=27/32`: the P polynomial is `(4g-3)^2(2g+3)/32`, with admissible double root `g=3/4`, plus a distinct stick state. Thus there are **two**, not three, physical states. Parameters q minus/plus 1/100 give one/three states. The canonical eta=1/8 fold is also admitted exactly as `one_fold`, with two states. Phase diagrams depend on eta: for example eta=2,q=1 has three states, including an inactive state and two active states.

## Certificate and input contract

A `COMPLETE` result means the inventory is complete **for its model ID, exact parameter and eta**, not for any external scene. It returns rational isolating gap intervals, a square-free defining polynomial, exact rational reconstruction of the second gap where applicable, and all mode aliases with their original algebraic multiplicities. Polynomial arrays are descending gap powers; each coefficient array is descending powers of the named parameter root. Rational coefficients have one entry. An inactive single branch instead binds its gap directly to the exact input parameter. Mode counts count admissible roots before cross-mode deduplication, including I for the single model; their sum may exceed the physical count at a seam.

The implementation uses square-free decomposition and exact ordered-field Sturm isolation, checks the original mode domains and nonzero denominators, and merges roots by polynomial gcd plus equality of the reconstructed second gap. The exact field for an event is determined by its polynomial **and its particular real-root isolator**. No numerical root tolerance decides multiplicity, physical admission or merging.

Only Python `int` and `Fraction` are accepted as numerical parameters, with 128-bit numerator/denominator admission bounds. Floats, unknown event names, unsupported q2/domain/eta bindings, uncertain `ParameterInterval` inputs and exhausted work budgets return `UNKNOWN` with no partial branch inventory. `ParameterInterval` is a deliberate unsupported-input marker; it does not certify a surrounding stratum or implement rounding. Converting a sensor float to a Fraction only records that float exactly and does not establish physical parameter equality.

SymPy is imported lazily. If unavailable, inventory requests return `UNKNOWN`; importing Motion and using the existing local checker continue to work. No core dependency or solver hot path changes. `max_work` bounds counted exact isolation/sign operations; fixed degrees and input bit limits limit the admitted workload. It is not a wall-clock deadline for an individual SymPy operation. Applications needing a hard deadline should use a bounded worker process.

Every result has `step_acceptance="NOT_ASSESSED"`. A future adapter must establish equality to this exact physical law, preserve all admitted states unless a separate selection rule is supplied, and propagate `UNKNOWN` on an unmatched model or uncertain parameter. The immediate integration target is a deterministic regression fixture for reduced contact solvers; no full-scene exporter is provided.

## P1–P4 compatibility boundary

P1's Delassus results concern `G=J M^-1 J^T`; the fixed SPD Gram here is compatible with that structural object. P1 does not turn it into uniqueness of the total nonlinear gap/force residual. P2's local sensitivity regime assumes a persistent mode and nonsingular Jacobian; this fold is a useful negative sensitivity regression only after a same-law residual binding. A derivative at the fold cannot be inferred from a regular P2 formula.

P3's strict particle-center counting concerns a different observable and has no direct certification bridge here. P4 distinguishes geometric tangency, cap grazing, reached branches and macroscopic support. Its linear-spring/history/geometric laws differ from this rational updated-force model. This module does not certify P4's law or select a reached P4 branch. These are explicit compatibility boundaries, not a claim that the four papers have been jointly implemented.

## Validation and costs

Tests independently reconstruct the original rational contact equations for every returned mode, check exact root isolation and physical inequalities, cover all named events, multiplicity, shared seams, eta variation, malformed/uncertain inputs, budgets, and optional-dependency absence. The runnable example compares the unchanged native Krawczyk result with the complete singular inventory. Exact enumeration has substantially more work than a local enclosure test. This feature adds a bounded completeness oracle; it makes no solver speedup claim.
