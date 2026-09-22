# Exact scope, remaining obligations, and what is not claimed

## Exact scope

* **Question.** On the same captured solver state and the same strict particle-in-box (D1) count,
  does reuse of the solver's mass grid beat a competent exact spatial accelerator, including its
  build/update cost?
* **Answer.** No, on every tested cell. The grid path is 1.4–4.9× slower than the best exact
  accelerator (cell list) on complete per-frame cost, and still slower in the query-only regime.
  Measured on a shared GPU; same-run paired medians and ratios are the carried quantities.
* **States.** Five frozen states (121 000–161 000 particles, 64³ resident grid, `dx = 0.0125 m`,
  `rho = 1600 kg/m³`, origin `(0, −0.40, −0.05)`), plus two state assemblies built deterministically
  for the independent audit.
* **Families.** `sparse`, `allnear`, `slab`, `corner`, `empty`; box counts 1/32/256/1024; query
  multiplicity Q 1/4/16. Box seed generator `81000 + 7000*ci + 7919*j` (`ci` = filtered selection
  index, `j` = query batch).
* **Correctness.** All exact paths return identical strict-D1 integers. Producer: 99 held-out
  batches, 0 gate false negatives, 0 count mismatch, canonical decision hash identical across two
  processes. This study: 16 cases × 6 paths, all agree. Independent audit: 10 new-state cases, three
  independent oracles, zero divergence, exact boundary inclusivity. Gate is exact for one-step
  displacement up to 0.9·dx; stale/future grids fall back to the exact sweep.
* **Contract.** The owned immutable contract protects the documented public surface only. It is a
  prototype depending on an unapplied producer patch; `integration_ready = false`.

## Remaining obligations

1. The field-detector per-frame series that originally motivated the line has **no producer** and the
   field-versus-hull claim remains withdrawn; it is not part of this study.
2. A production ownership patch and its integration review remain open.
3. The Warp codegen caveats in the engine's version (int32 `min`/`max` and `vec3d`+`float64`
   parameter pairs miscompiling on the test device) must be re-checked if the Warp version changes.

## Not claimed here

* No novel acceleration and no methodological-novelty claim; the cell list is a textbook broad phase
  and the engine already ships a fixed-radius hash grid.
* No claim about occupied material volume, surface clearance, continuous collision, arbitrary
  (non-axis-aligned) query geometry, or production integration.
* No claim beyond the tested hardware; all timings carry the shared-device caveat.
* No claim that the mass-grid threshold alone is a correct decision; it is a measured negative.
* A secondary mass-underflow bound was excluded from the claim map because the corrected manuscript
  does not assert it and the supporting source is not in this package. No numeric bound is carried.
* The historical 84.66, 4.85 and 66.52 ms timing values are archival reports without their original
  timing runs. They cannot be regenerated here. The retained later incremental-flow raw has median
  70.2591 ms; its 3.5 % within-10 ms fraction is recomputable.

## Time and device labels used in every table

* `device`: `shared GPU` for all timings.
* `time_boundary`: `paired 5-repeat median` (per-frame or per-query) with `wp.synchronize` around
  each stage; absolute milliseconds vary between passes and are not carried in isolation.
