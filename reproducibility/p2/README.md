# Contact-sensitivity reproduction package

This package regenerates the tables (A–AB) of the contact-sensitivity manuscript from the bundled raw measurement records.  Cells with bundled raw measurements are regenerated from those records. Thirty printed numeric measurement cells lack a bundled original per-run source, eighteen em-dash cells are unavailable placeholders, and one Table P digest identifies the original pre-cleanup serialization. These exceptions are declared in `SOURCE_AVAILABILITY.json`; fixed settings in Tables K and N are protocol parameters.  It is a re-run of *stored* measurements and of the packaged producer code, not a fresh GPU benchmark; no physics was rerun to assemble it.

Per-cell provenance is in `tables/cell_sources.json` (cell → raw source and aggregation) and the unresolved cells are enumerated in `SOURCE_AVAILABILITY.json`.

Layout:

```
manuscript.md              the paper text (public copy)
data/contact_scenes/       contact-scene-v1 package (JSON + packed HDF5, loader/verify)
data/reference/            measured reference records for the tables
data/matched_accumulation/ matched accumulation study (CSV + raw cell records)
data/matched_single_graph/ single-graph matched record (CSV + per-launch JSONL)
data/identification/       identification runs, bounds and series records
data/accepted_state/       accepted-state records and residual-metric checks
data/pile_runs/            integrated pile-run records
data/slip_validation/      single-contact slip sweep records
data/chunk_state/          observation-window chunk controls
data/replay_inputs/        large immutable inputs; files >100 MB ship as deterministic .gz
solver/                    producer code (Warp modules and drivers)
tables/table_<ID>/         regenerate.py, expected.json and output/table_<ID>.csv
tables/cell_sources.json   cell -> raw source and aggregation map
SOURCE_AVAILABILITY.json   unresolved cells and exactly which claims depend on them
run_tables.py              run every table and write tables/report.json
```

## Transport of the large inputs

Three replay inputs exceed 100 MB.  They are shipped as deterministic standard-library
gzip members below 90 MB (`*.npz.gz`); `data/replay_inputs/INPUT_MANIFEST.json`
records each original byte count and full SHA-256, and
`data/replay_inputs/reconstruct_inputs.py` (stdlib only) rebuilds them and verifies
full hash identity.  Reconstruction needs extra disk equal to the original file size
until the `.gz` member is removed.  No data is dropped and no external file or LFS
pointer is required.

## Run

```sh
cd release
python run_tables.py
```

Each table script reads only files under this directory (relative paths), uses NumPy only, and compares its regenerated values against the manuscript values extracted into `expected.json` at the manuscript's displayed precision. The comparator uses a relative tolerance (5e-4) with no absolute floor, so mutations of the small residual cells (1e-13 … 1e-17) are not masked. Cells whose raw input is not part of the package are reported as explicit gaps, never silently passed.

## Tables

| Appendix | directory | subject |
| --- | --- | --- |
| A | `tables/table_A` | Sliding contact gap under exact and relaxed laws |
| B | `tables/table_B` | Bilateral derivative reference (Pinocchio parity) |
| C | `tables/table_C` | Slipping derivative reference, faulty vs corrected kernel |
| D | `tables/table_D` | Executed solve costs, PGS vs ADMM |
| E | `tables/table_E` | Batch budgets and rates |
| F | `tables/table_F` | Repeated batch bytes (cross-device) |
| G | `tables/table_G` | Identification under observation noise |
| H | `tables/table_H` | Excitation and model consistency |
| I | `tables/table_I` | Synthetic crossover |
| J | `tables/table_J` | Matched accumulation study |
| K | `tables/table_K` | Matched protocol |
| L | `tables/table_L` | Matched integer digests |
| M | `tables/table_M` | Packing exclusions |
| N | `tables/table_N` | Identification protocol |
| O | `tables/table_O` | Standard-trajectory estimation |
| P | `tables/table_P` | Identification parameter digests |
| Q | `tables/table_Q` | Parameter spread between repetitions |
| R | `tables/table_R` | Identification elapsed time |
| S | `tables/table_S` | Excitation and local information |
| T | `tables/table_T` | Observation-derivative diagnostic |
| U | `tables/table_U` | Selected-excitation estimation and failure control |
| V | `tables/table_V` | Integrated pile failures |
| W | `tables/table_W` | Single-contact slip validation |
| X | `tables/table_X` | Chunk-state control |
| Y | `tables/table_Y` | Single-graph matched record |
| Z | `tables/table_Z` | Accepted-state and gate diagnostics |
| AA | `tables/table_AA` | Provenance, residual metric and verification cost |
| AB | `tables/table_AB` | Full parameter tail for the repaired estimator |

## Limits

* Regeneration reads stored measurements; it does not rerun the optimization physics.  `solver/` and `data/replay_inputs/` are supplied so that a bounded physics replay is possible on a GPU host; that replay is not executed here.
* Timing records are observed elapsed times, some on a shared GPU; the shared device caveat is kept wherever the source carried it.
* Full SHA-256 digests are retained; no hash is truncated.
