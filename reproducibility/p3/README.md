# Reusing a solver grid for an exact point-in-box particle count — release package

This is a self-contained, public package for a bounded comparative study. It answers one question:

> On the same captured solver state and the same strict particle-in-box (D1) count, does reuse of the
> solver's mass grid beat a competent exact spatial accelerator, including the accelerator's
> build/update cost?

**Measured answer: no, on every tested cell.** The producer-side grid reuse gives a real gain of
0.647 against the *direct* unindexed exact kernel on a sparse 1024-box cell, but against the best
exact accelerator (a uniform cell list built from the same particles) the grid path is 1.4–4.9×
slower, and it loses in the reusable query-only regime too. The mechanisms measured are the gate's
filter inflation and candidate density, and the conversion/ownership cost the grid path must pay.

## Contents

```
README.md
requirements.txt
MANIFEST.sha256                         full sha256 of every packaged file
docs/
  comparative_study.md                  the comparative-study manuscript
  CLAIM_MAP.csv                         every displayed number -> source artefact + reproducibility
  SOURCE_MAP.md                         public file -> table/state; how the package was assembled
  LIMITS.md                             exact scope, remaining obligations, what is not claimed
src/
  producer_grid_keepout.py              producer grid prefilter + strong direct kernel (path under test)
  owned_keepout_contract.py             owned immutable contract over the prefilter kernels (prototype)
  keepout_reference.py                  independent NumPy deposition / threshold reference
  range_reference.py                    independent constants, box generators, D1 predicates
  baseline_cell_list.py                 exact uniform cell-list accelerator (baseline A)
  baseline_sorted_key.py                exact radix-sorted-key accelerator (baseline B)
  bench_common.py                       independent D1 oracle and frozen-state loader
data/
  states/                               five frozen particle states + state_hashes.json
  measurements/                         frozen raw measurement JSON/CSV (numeric payloads byte-identical
                                        to the private originals; names only are descriptive)
  prose_controls/                       frozen raw inputs for the numerical manuscript prose controls
                                        (threshold fuzz/matched, mass-grid column edges,
                                        dynamic-occupancy costs) + an archived-reported-values record
scripts/
  regenerate_tables.py                  CPU: rebuild every displayed table from frozen raw JSON
  verify_package.py                     CPU: hashes, state loader, table match, prose-control check
  verify_prose_controls.py              CPU: recompute the numerical prose controls from raw inputs
  replay_correctness.py                 GPU: exact-count agreement matrix on the frozen states
  replay_benchmark.py                   GPU: matched complete-cost replay
  run_all_cpu.sh                        run the CPU scripts
  run_gpu_replay.sh                     run the two GPU scripts under a device lock
expected/
  table_*.csv                           the recomputed tables (output of regenerate_tables.py)
```

## Two kinds of reproduction

These are different commands and should not be confused.

**Table regeneration (CPU, no device needed).** Recompute every table from the frozen raw
measurements in isolation:

```
python scripts/regenerate_tables.py
python scripts/verify_package.py        # verifies the regenerated tables match expected/
python scripts/verify_prose_controls.py # recomputes the numerical prose controls from raw inputs
```

`regenerate_tables.py` computes the decisive ratio itself:
`t_gate_over_best = t_gate / min(t_cell_list, t_sorted_key)` from the **same-run** medians, where a
value above 1 means the grid path is slower. It never combines medians from different runs.

**New physics / benchmark replay (GPU).** Re-run the physical measurements on a GPU, e.g.

```
bash scripts/run_gpu_replay.sh
# or individually:
python scripts/replay_correctness.py --out replay_out/correctness.json
python scripts/replay_benchmark.py --tag a --out replay_out/benchmark_a.json
```

A fresh GPU benchmark is **not** required to check the tables. All measured times were taken on a
shared GPU. Recorded device memory usage ranged from 2043 to 2353 MiB (approximately 2.00–2.30 GiB)
across the bundled measurements; this is total memory reported by the device, not memory isolated to
other processes.
The carried quantities are the paired 5-repeat medians and the same-run ratios, labelled `device`
and `time_boundary` in every table.

## Data semantics (important)

* All paths return the identical strict-D1 integer count per box: a breach exists iff a particle
  **centre** lies in the closed box `[lo, hi]`. This is the only output.
* `t_gate_over_best` is recomputed from same-run inputs. The producer's stored
  `gain_vs_best_exact_baseline` field was found by the independent audit to carry inconsistent signs
  and is **not** quoted here.
* The box seed generator is `seed = 81000 + 7000*ci + 7919*j`, where `ci` is the index of the cell in
  the filtered selection order and `j` the query-batch index; the same cell label can use different
  boxes in different runs. The correctness replay uses a deterministic
  `seed = 90000 + nb + crc32("frame|family") mod 1000`.
* The owned/immutable contract and the exact fallback are **prototype properties**. They depend on an
  unapplied producer patch; `integration_ready = false`. Only the documented public surface is
  protected.
* Nothing here measures occupied material volume, surface clearance, continuous collision, arbitrary
  query geometry, production integration, or other hardware.
* Measurement conditions (shared device, time boundary) and denominators are attached to every
  displayed number in `docs/CLAIM_MAP.csv`.

## Provenance and integrity

All frozen states and raw measurements are the producer/auditor artefacts with full sha256 recorded
in `MANIFEST.sha256`. Descriptive file names were substituted for private ones; a private crosswalk
that maps public artefacts back to their originals and originals' hashes is kept **outside** this
package. `verify_package.py` checks the manifest, loads every state, and confirms the regenerated
tables and the numerical prose controls match their expected values. Absence of internal identifiers
in file names, text and binary payloads is checked by a separate auditor that lives outside this
package and that exempts no member.

The historical 84.66, 4.85 and 66.52 ms values in the manuscript are carried archival reports:
their original timing runs are not retained and cannot be regenerated from this package. The
retained later incremental-flow record has median 70.2591 ms and supports the reported 3.5 %
within-10 ms fraction. `verify_prose_controls.py` checks the archived record's labels and recomputes
the quantities supported by retained raw; it does not turn the three archival values into raw data.
