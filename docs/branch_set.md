# Optional finite branch-set composition

`motion_engine.ncp.branch_set` is an opt-in relational composition kernel beside
the numerical solvers. It does not discover or certify all roots of an NCP.
The default solver and its API are unchanged.

The caller supplies a transition oracle returning `Step(children, complete=True)`
ONLY when it has proved exhaustion of its declared transition relation, and a
`verify(parent, child)` witness checker. `Step(children)` defaults to incomplete.
Passing two Newton roots does not provide completeness. Invalid witnesses,
unknown/singular modes and frontier budgets return UNKNOWN and the previous
completed frontier, never a truncated set labelled complete.

`propagate(initial, oracle, N, verify=check)` returns layers, predecessor edges,
state counts, split/merge counts and path multiplicities. Alternative witnesses
for the same parent and full child state retain all certificate edges but count
as one physical transition. Generated/split/merge statistics count these distinct
state transitions; `witness_edges` counts all retained certificates. Equality must compare
the complete Markov state, including geometry, all material memory and any
future-relevant parameters not fixed in the oracle context. Multiplicities are
history counts, not probabilities. Every allowed exact merge keeps all edges.

`observe(states, readout, value, error)` performs closed bounded-error intersection.
No entropy heuristic removes admissible states. `rank_measurements` minimizes
the worst survivor count among supplied sensors, with the disjoint-band
gap/resolution/cost as secondary score. Resolution must already be included in
the error bound when it contributes quantization error. This is not an entropy
or optimal continuous sensor-design claim.

Nonfinite measurement values, readouts and sensor parameters raise `ValueError`;
overflow in sensor bands or scores also raises. Such invalid evidence must not
be used to delete admissible states. Exact integers and rational values are
supported even when their magnitude exceeds the floating-point range.

The accompanying research lane formalizes the conditional set theorem in Lean.
Python execution, physical closure and oracle completeness remain separate
obligations. The rotating Coulomb box has a continuum already at its first step
and is unsupported by this finite oracle interface. No full rotating rod/box
trajectory certificate or automatic adapter from the engine solvers is supplied.
