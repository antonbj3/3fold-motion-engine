# Motion evidence snapshot, 2026-09-13

The separate noncontracting contact solver passes the recorded four-device
identity comparison. The submillisecond throughput target and complete
repository verification remain open. This snapshot reconciles existing
receipts and source dependencies; it adds no numerical implementation or new
performance measurement. Per-module status remains in [RUNNING.md](RUNNING.md).

| Result | Decisive evidence | Scope |
|---|---|---|
| Cross-device contact identity | 12/12 solver/backend rows; complete state and force bytes identical across two independent runs per device | Separate velocity/position noncontracting candidate; frozen matrix remains 10/12. [Matrix](../reports/position_solve_matrix/matrix.json) |
| K16 physical gates | 7/7; penetration/R 0.1911702752 at 40 velocity iterations | ColorCache branch; measured step 3.724/3.683 ms, no submillisecond result. [Physical receipt](../reports/innovation_position_physics.json) |
| Mass ratio 1000 | Residual 6.707800995e-8; spacing 0.1998999715; total budget 40 rounds; four-device exact traces | Bounded hybrid with CPU normal coupling, at most 32 bodies/256 contacts. Fully GPU-resident JGS2 and throughput remain unimplemented. [Matrix](../reports/complement_normal_matrix.json) |
| Persistent context | 10/10; 100 pair/color cache hits, zero rebuilds/captures and zero pair churn in the measured window | N10000 step 2.48736614/2.47570147 ms. This fixture already reuses its context. [Receipt](../reports/persistent_context_probe.json) |
| Substep quality/cost | Observer 6/6; at 4 ms K16 selects S2/V16, penetration/R 0.140916407 and worst measured cost 3.958035 ms | Separate N10000 cost selects S1/V40 at 3.602970 ms. No valid row at 0.5/1/2 ms. [Experiment](SUBSTEP_PARETO_EXPERIMENT.md) |
| RT body-pair search | 4/5; exact pairs but slower full rebuild than the frozen hash-grid baseline | No adoption or speed promotion. [Experiment](RT_BODY_PAIR_EXPERIMENT.md) |
| Midpoint free fall | 4/5; observed order -1.650360381 | Roundoff-dominated constant-acceleration fixture; no second-order certificate. [Receipt](../reports/midpoint_sigma_probe.json) |
| Full repository verification | 10/161 fresh declarations mapped; 151 unmapped, zero children launched by the complete target | Existing selected target passes eight commands in two exact local runs. Extended cloud setup failed before observed numerical execution. [Contract and limits](VERIFICATION.md) |

The first original differences were the compound tangential impulse expressions.
Velocity-only noncontraction left a second divergence in the position solve at
steps 29/11. Disabling contraction in the additional position kernel closes the
recorded matrix. This does not identify an individual machine instruction or
certify arbitrary scenes, future devices or different toolchains.

## Outstanding optimization proposals

The late proposals below were not implemented before the numerical freeze.
Their proposed speedups are not measured results of this snapshot.

| Proposal | Source finding | Required next evidence |
|---|---|---|
| Remove the deferred-force snapshot copy | `DeferredForceContactEngine.step()` copies the post-gravity, pre-solve velocity. The public `contact_forces()` subtracts that snapshot from the final velocity and multiplies by mass/time. The copy is not diagnostic-only. | A sibling must preserve force queries after every completed step, including repeated queries and multiple substeps. Omitting or making the snapshot stale cannot inherit the existing force-parity gates. Measure the copy's actual full-step cost before selecting a replacement. |
| Adopt the wide grid | `AppliedWarmContactEngine._build()` already creates the 256/256/8 grid. DeferredForce, ColorCache, PackedSelect, IslandSolve, PairLayoutCache and PreparedPlanar inherit it. The standalone WideGrid candidate still fails its own K16/4 ms gates. | Establish the intended target class and measure a remaining difference. Replacing the frozen base default is not an unimplemented optimization of the existing PreparedPlanar chain. |
| Remove direct-pair count readbacks | The raw-count readback determines capacity refusal and later launch sizes. The selected-count/flag readback determines selection refusal and downstream work size. | A separate implementation must preserve both refusal semantics and exact public outputs, then beat the matched full-step baseline. An isolated synchronization count is insufficient. |
| Cache tangent frames on chip | No new on-chip candidate was implemented. The fixed-frame prototype retains its K16 penetration and timing failures. | Recheck full physical gates and cross-device identity in addition to any measured solve-stage benefit. |

## Freeze checks

All 207 current status paths exist and are unique: 161 VERIFIED-FRESH,
36 OWN-GATE-FAIL and 10 SYNTHETIC-ONLY. Existing selected source/input pins
validate, and the two retained local selected reports are byte-identical.
These checks do not extend numerical coverage to the unmapped declarations.

The wording change preserves the complete executable AST when docstrings are
excluded. The CPU-hidden pytest invocation returned 68 passed, 36 skipped and
one compose setup failure from a sibling bytecode cache. With an isolated cache,
that unchanged test reached its CUDA requirement; it then passed separately
under the GPU lock in 1.21 s, including its original two-run state and physical
gates. This is not a single-environment full-suite-green result. Test-generated
reports were saved separately and the archived comparison bytes restored.
