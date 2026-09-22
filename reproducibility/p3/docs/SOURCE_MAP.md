# Source map

Public artefact -> what it is -> which displayed table consumes it -> how it is reproduced.
Descriptive file names are used throughout; a crosswalk that maps each public artefact back to its
frozen source and that source's full sha256 is kept separately from this package. No member names any
internal identifier; an external private auditor scans the final tree (file names, text, raw bytes
and container metadata) and exempts no member.

## Frozen states (`data/states/`)

| Public file | Content | sha256 |
|---|---|---|
| `state_frame_120.npz` | 121 000 particles at the intrusion window; `x_at/x_now/mass/ids/grid_resident` | in `state_hashes.json` |
| `state_frame_140.npz` | 141 000 particles | in `state_hashes.json` |
| `state_frame_150.npz` | 151 000 particles (known frame used in the win cell) | in `state_hashes.json` |
| `state_frame_160.npz` | 161 000 particles (new frame) | in `state_hashes.json` |
| `state_frame_160_shifted.npz` | 161 000 particles, shifted geometry (baffle −20 mm z) | in `state_hashes.json` |

`bench_common.load_frame` verifies each state's hash at load time against `state_hashes.json`.

## Raw measurements (`data/measurements/`)

| Public file | Origin of the numbers | Consumed by |
|---|---|---|
| `producer_complete_cost_run_a.json`, `..._run_b.json` | producer complete cost vs the direct kernel | `table_producer_complete_cost.csv` |
| `matched_cost_run_a.json`, `..._run_b.json` | producer matched cost, all five paths, per-frame | `table_matched_comparison.csv`, `table_audit_reproduction.csv` |
| `matched_cost_query1.json`, `matched_cost_query4.json` | producer matched cost at Q=1 and Q=4 | `table_query_sweep.csv` |
| `matched_cost_repeat_a.json`, `..._b.json` | producer two-process repeat (counts identical) | `table_matched_comparison.csv` |
| `correctness_matrix.json` | 16 cases × 6 exact paths, all agree | `table_correctness_summary.csv` |
| `correctness_and_controls_run_a.json`, `..._b.json` | 99 held-out batches, ownership controls, frame semantics | `table_correctness_summary.csv` (via the audit) |
| `audit_independent_cost_run_a.json`, `..._b.json` | independent audit re-timing (own harness, own charging) | `table_audit_reproduction.csv` |
| `audit_gate_stress.json` | gate safety under displacement / stale / future grid | `table_gate_stress.csv` |
| `audit_new_state_cases.json` | 10 new-state cases, three oracles, work-count equality | `table_correctness_summary.csv` |
| `audit_oracle_crosscheck.json` | three oracles agree on 74 boxes per state | `table_correctness_summary.csv` |
| `audit_reconcile.json` | CSV-vs-raw reconciliation; documents the inconsistent stored `gain_vs_best` label | `table_correctness_summary.csv` |
| `audit_freeze.json` | hash freeze; records that the original producer manifest did not self-verify (`out_selftest.json`) | provenance only |
| `matched_baseline.csv` | the producer's own CSV summary (arithmetically re-derived by the audit) | cross-check only |
| `box_geometry_families.json` | the five box families and lattice used by the generators | all replays |

## Prose-control raw inputs (`data/prose_controls/`)

The numerical claims that appear only in the manuscript prose (not in a displayed table) are backed
by their actual frozen raw inputs, bundled here under descriptive names, plus one explicit record for
values whose original raw is not retained:

| Public file | Raw input | Backs |
|---|---|---|
| `threshold_fuzz_summary.json` | threshold study fuzz summary, 300 scenes | D3 false negatives/positives by `theta` |
| `threshold_matched_summary.json` | threshold study matched summary, 20 cases | `theta=0.5` matched false negatives |
| `threshold_matched_cases.csv` | threshold study per-case table | the matched controls |
| `massgrid_column_edges.json` | mass-grid column-edge probe on a real 96³ grid | empty columns `239/8458`, `150` column-frames, p95/one-cell counts |
| `incremental_frame_costs.json` | exact incremental flow, per-frame cost (199 frames, later reproduce pass) | the `3.5 %` within-10 ms fraction |
| `incremental_paired_ratio.json` | paired full-vs-incremental ratio probe | `35.32 ms` scaled incremental, carried `84.66`/`4.85` ms |
| `archived_reported_values.json` | **derived record, not raw** | values whose original raw is not retained (`84.66`, `4.85`, `66.52`), each labelled `archived_reported` with the gap named |

`scripts/verify_prose_controls.py` recomputes every recomputable prose number from these inputs and
requires the matching literal phrases to remain in `docs/comparative_study.md`.

## Source modules (`src/`)

| Module | Role |
|---|---|
| `producer_grid_keepout.py` | producer grid prefilter; strong direct exact kernel |
| `owned_keepout_contract.py` | owned immutable contract (prototype) over the prefilter kernels |
| `keepout_reference.py` | independent NumPy deposition / threshold reference |
| `range_reference.py` | independent constants, box families, D1 predicates |
| `baseline_cell_list.py` | exact uniform cell list (histogram + prefix scan + scatter) |
| `baseline_sorted_key.py` | exact radix-sorted cell key + binary search |
| `bench_common.py` | independent D1 oracle and frozen-state loader |

## The manifest defect, resolved

An earlier frozen bundle was frozen with a manifest whose `out_selftest.json` member had been
written after the manifest and did not verify. This package's `MANIFEST.sha256` is generated
**after** all outputs freeze, so it self-verifies; `scripts/verify_package.py` checks every member.
The frozen source of the corrected measurements is recorded in the separate crosswalk.

## Reproduction status legend (`docs/CLAIM_MAP.csv`)

* `frozen_raw_recompute` — the number is recomputed by `regenerate_tables.py` from a frozen raw
  artefact; the command and artefact hash are recorded.
* `producer_measured` — the number is a producer measurement carried with its run and device label.
* `audit_reproduced` — the number is an independent audit measurement.
* `archived_reported` — a retained value whose original raw is not in this package; it is a carried
  reported value, not a replayable raw measurement, and the gap is named in
  `data/prose_controls/archived_reported_values.json`.
* `withdrawn` — a claim removed because its producer never existed.

Claim identifiers use the neutral `claim-###` sequence. The secondary mass-underflow bound has no
entry because it is absent from the corrected manuscript and has no packaged source; the public map
asserts no number for it.
