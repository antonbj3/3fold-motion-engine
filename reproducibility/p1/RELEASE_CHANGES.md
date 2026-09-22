# Contact reproduction bundle — release changes

Sections A and B document the prior release copy relative to its audited
predecessor bundle (425 files, release-wide predecessor `SHA256SUMS` sha256
`0efc8fc541f7e90f361d281412dbb6b581af5d88331e93ef423908a6b2280dc8`).
The predecessor repairs are restated in section A for a reader of this
standalone bundle; the edits in this copy are in section B with their exact
before/after anchors.

In the present public copy, numeric datasets, array payloads and measured
values remain unchanged. Scene identifiers and two yes/no flags in table CSVs
are expressed in public English; the source and delivery hashes are rebuilt.

## A. Repairs inherited from the audited predecessor bundle

1. **Fresh HDF5 containers.** All 21 `data/ncp/scenes/NN.hdf5` are rebuilt from
   scratch (`h5py.File(path, "w")`) from the original numeric datasets; no
   internal build-tree string survives in the file bytes. Numeric dtype, shape
   and raw bytes are identical to the source container; `W` attributes are
   identical.
2. **Private build identifier removed.** `tables/incidence_solvers/SOURCE.md` no longer
   contains a private build identifier; two duplicate files were renamed with unchanged content.
3. **Row-level table sources delivered** for the five former
   `historical_aggregate` groups (Section 4.2 shift ablation, Section 5
   conductance, Appendix B, Appendix E, central 5/21 count), each with a public
   builder. Row identity to the printed tables is recorded in the predecessor
   recovery report.
4. **Nested and global `SHA256SUMS`** rebuilt over the expanded file set.

## B. Edits in this copy

All edits are text-only or metadata-only; the anchors are exact.

| # | file | defect | repair |
| --- | --- | --- | --- |
| B1 | `data/ncp/README.md` | the delivery-level numeric-comparison sentence was duplicated | keep one occurrence |
| B2 | `tables/README.md` | directory table listed only the six pre-recovery tables | list all ten delivered table directories, with the four recovered-row groups added |
| B3 | `tables/README.md` | referenced `../PACKAGING.md`, which is not delivered | point to `source_map.json` and the release manifest instead |
| B4 | `tables/source_map.json` | conductance group labelled `Section 5.4`; the manuscript has no 5.4 heading | relabel `Section 5` |
| B5 | `dimensionless_stop/SOURCE.md` | arm descriptions did not state that the absolute arm also changes the inner-block target | state the inner/outer target scope of each arm |

B5 matches the manuscript, whose Section 4.4 states that the physical and
dimensionless arms share the dimensionless inner-block target while the absolute
arm uses an inner target written in the current unit.

## C. Public scene names and missing source in this copy

- Descriptive English scene aliases replace archive keys in JSON keys and values,
  HDF5 text, CSV, scripts, documentation, and the scene manifest. The private
  crosswalk is outside the public release. All 21 HDF5 files are written afresh.
- `tables/spectral_grid/spectral_grid.csv` expresses the two yes/no columns in
  English; its table builder reads those English values. Numeric values are unchanged.
- `tables/refresh_schedule/refresh_contrast.json` delivers the 63 measured rows
  for the Section 4.4 refresh comparison. `regenerate.py` extracts the 15 SI
  cells, and `tables/source_map.json` maps that table to this source.
- `data/ncp/numeric_identity.json` records the 210-dataset source comparison.
  The former dangling numerical-difference link is replaced by this file.
- No compiled Python caches are included. Scene, supplement, nested and global
  file hashes are recomputed after all edits.

## Verify the bundle

```sh
cd data/ncp && python loader.py && python verify.py --all
cd supplement && sha256sum -c SHA256SUMS
cd dimensionless_stop && python rerun.py && python rerun.py --linear-solver sparse_lu
cd tables/incidence_solvers && python regenerate.py
cd .. && sha256sum -c SHA256SUMS
```

`dimensionless_stop/rerun.py` matches 44/45 stored outer counts with dense
Cholesky and 45/45 with `--linear-solver sparse_lu`; the single differing cell is
rounding-sensitive and is discussed in `dimensionless_stop/SOURCE.md`.

## Limits

The scene package is a metadata-preserving repackaging of numeric payloads that
are unchanged from their source; the table CSVs are regenerated from stored
measurements or delivered as labelled aggregates or recovered rows; the
dimensionless comparison is a bounded consistency check on exported operators,
not a convergence theorem and not a speed claim. No physics is rerun beyond
solves on the exported operators.
