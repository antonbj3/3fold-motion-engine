# Table data

Clean CSVs from the measured studies, each with a regeneration script (where
the stored measurements allow it) and a SOURCE.md that names the manuscript
table, whether the numbers are regenerated or exported-only, and whether the
physics can be rerun here.

| directory | manuscript | status |
| --- | --- | --- |
| solver_relaxation | Appendix G | regenerated from stored measurements |
| adaptive_penalty | Appendix F | regenerated from stored measurements |
| incidence_solvers | Appendix C, D | regenerated from stored measurements |
| refresh_schedule | Section 4.4 refresh schedule | measured rows and 15-cell SI extraction |
| shift_ablation | Section 4.2 | recovered rows (archived 252-cell ablation) |
| spectral_grid | Appendix E | recovered rows (archived 21-row sweep) |
| load_ordering_conductance | Section 5 | recovered rows (archived 8-scene benchmark) |
| dense_block | Appendix B | recovered rows (row source in supplement/patch) |
| siconos_external | Appendix H | verified export only |
| gpu_batch_accuracy | P2 GPU batch table | verified export only |
| gpu_adjoint_identification | P2 GPU-adjoint identification | verified export only |

`source_map.json` gives the machine-readable mapping. No physics is rerun in
this package: every CSV is either regenerated from stored per-run measurements
or copied as an exported measurement. When a table's row-level input is a stored
measurement set, it is included under `<table>/inputs/`; when the measurement
came from an external solver or GPU batch, it is marked export-only.

Internal provenance strings (build paths, report names, GPU provider) were
replaced by public descriptions; no numeric value was changed. The full table
inventory (manuscript table, status, command, source file) is in
`source_map.json` beside this file and in the release manifest.
