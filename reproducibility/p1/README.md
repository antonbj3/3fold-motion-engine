# Contact reproduction bundle

This is the release copy of the contact-operator manuscript's reproduction
bundle. It is a repaired copy of the audited public bundle; every change is
documented in `RELEASE_CHANGES.md` beside this file.

Components:

```
data/ncp/               contact-scene-v1 package: 21 scenes (JSON + compact
                        symmetric-packed HDF5), standalone loader.py / verify.py /
                        negative_controls.py / iso_run.py, manifest.json,
                        SHA256SUMS, scene table
supplement/             patch-wrench, load-ordering and ramp reproduction
                        supplement (N1-corrected), with SHA-256
tables/                 clean CSVs of the tabulated studies, each with a
                        regeneration script (where stored measurements exist)
                        and a SOURCE.md; includes the 63-row §4.4 refresh source
dimensionless_stop/     absolute / physical / dimensionless stop comparison on
                        three scenes x five unit systems, with the bundled
                        standalone solver (solver.py), the reproduction driver
                        (rerun.py), the structure test (structure_check.json)
SHA256SUMS              SHA-256 of every file in this release
```

## Verify the bundle

```sh
cd release/data/ncp && python loader.py && python verify.py --all
cd release/supplement && sha256sum -c SHA256SUMS
cd release/dimensionless_stop && python rerun.py && python rerun.py --linear-solver sparse_lu
cd release/tables/incidence_solvers && python regenerate.py
cd release && sha256sum -c SHA256SUMS
```

`dimensionless_stop/rerun.py` matches 44/45 stored outer counts with dense
Cholesky and 45/45 with `--linear-solver sparse_lu`; the single differing cell is
rounding-sensitive and is discussed in `dimensionless_stop/SOURCE.md`.

Dependencies: NumPy, SciPy (sparse), h5py. No project build tree is imported.

## Limits

The scene package is a metadata-preserving repackaging of numeric payloads that
are unchanged from their source; the table CSVs are regenerated from stored
measurements or delivered as labelled aggregates; the dimensionless comparison
is a bounded consistency check on exported operators, not a convergence theorem
and not a speed claim. No physics is rerun beyond solves on the exported
operators.
