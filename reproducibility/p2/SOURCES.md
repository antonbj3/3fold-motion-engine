# Release sources

Every file is derived from the audited measurement records of the contact-sensitivity study. Original source paths and their full SHA-256 hashes are recorded in a private crosswalk kept outside this package; inside the package provenance uses descriptive experiment names only. Numeric payloads are unchanged by the public renaming.

| public area | contents |
| --- | --- |
| `data/accepted_state/` | accepted-state records and residual-metric/provenance checks |
| `data/chunk_state/` | observation-window chunk controls |
| `data/contact_scenes/` | contact-scene-v1 package: JSON + compact symmetric-packed HDF5, standalone loader and verifier |
| `data/identification/` | identification runs, information bounds, chunk and series records |
| `data/matched_accumulation/` | matched accumulation study: delivered table, per-cell records, integer digests and packing exclusions |
| `data/matched_single_graph/` | single-graph matched record: matched table and per-launch JSONL records |
| `data/pile_runs/` | integrated pile-run records from the initial state |
| `data/reference/` | measured reference records, derived protocol tables and cross-device digest records |
| `data/replay_inputs/` | large immutable inputs for the bounded physics replay entry point |
| `data/slip_validation/` | single-contact slip sweep records |
| `solver/` | producer code: Warp contact/derivative modules and drivers |
| `tables/` | per-table regeneration scripts, expected values and outputs |

`data/contact_scenes` is the companion public scene package with fresh containers and descriptive aliases; the remaining areas are public-named copies of the measurement records.
The bounded physics replay entry point is `solver/` with `data/replay_inputs/`; it requires a GPU host and is not executed here.

## Recovered raw sources

Three former provenance gaps are now closed with genuine archived producer records
(not the manuscript or `expected.json`):

| release file | original archived producer |
| --- | --- |
| `data/reference/lattice_producer_shared.txt` | lattice-contact producer stdout (PGS 1000-sweep / ADMM 25-iteration residuals) |
| `data/reference/bilateral_a1_producer.txt` | Pinocchio-parity producer stdout for the Unitree A1 row |
| `data/accepted_state/branch_reference.json` | CPU-adjoint branch-reference record (smoke and series states) |

Cells whose raw source could not be recovered are listed in
`SOURCE_AVAILABILITY.json`; they are never filled from the manuscript values.

Internal build/session identifiers and environment-variable selectors were
replaced with descriptive experiment labels in shipped code, commands, loaders
and raw metadata; numeric payloads are unchanged. Some historical diagnostics
refer to archived reports whose supporting traces are not bundled. The scene index, JSON, HDF5 metadata and nested hashes now use the descriptive scene aliases. Historical run hashes in the saved records still identify the original producer bytes. The release manifest identifies the present public files.
