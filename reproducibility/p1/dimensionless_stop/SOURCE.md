# Dimensionless stop comparison

Measured/audited comparison of three stopping rules on cube, eight_box_tower and the
mass-ratio column, in five unit systems:

- `absolute`: fixed SI tolerance `1e-10` (the delivered SI variant). The
  absolute arm also writes the inner-block target in the current unit, so it
  varies both the outer stop and the inner target;
- `physical`: fixed physical tolerance `1e-10 N s`, transformed per unit
  (the delivered normalized variant); shares the dimensionless inner-block
  target with the dimensionless arm,
- `dimensionless`: `r / lambda_ref < 1e-10` (new measurement); shares the
  dimensionless inner-block target with the physical arm, so the
  physical-versus-dimensionless contrast isolates the outer stop.

The three variants are kept explicit and are not merged.

## Contents

- `dimensionless_stop.json` - stored measured rows (45 = 3 scenes x 5 units x
  3 stops), plus the physical/SI constants. Numeric row fields equal the audited source. Scene names use
  public aliases; `source_content_sha256` identifies the original archived
  measurement before aliasing.
- `dimensionless_stop.csv` - tidy export of the rows.
- `equivalence.json` - check that the operators (G, b, mu) reconstructed by the
  packaged `data/ncp/loader.py` equal the operators used by this measurement on
  the three scenes.
- `solver.py` - the bundled standalone solver (fixed-penalty proximal ADMM with
  exact Coulomb-cone projection, the de Saxce outer fixed point, the
  dimensionless inner-block target, and the natural-map residual). It has a
  dense-Cholesky x-update and a `sparse_lu` option matching the measurement's
  `archived ADMM solver` linear algebra.
- `rerun.py` - reproduces the 45 stored rows from the packaged scene operators.
- `regenerate.py` - rebuilds the CSV from the stored JSON (no physics).

## Reproduction

    cd release/dimensionless_stop
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python rerun.py --out rerun_dense.json
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python rerun.py --linear-solver sparse_lu --out rerun_lu.json

The dense-Cholesky path matches **44/45** stored cells. The single remaining
cell, `three_body_column_normal` absolute `Mm3`, is rounding-sensitive: dense Cholesky plateaus
at its own residual floor (no convergence within the cap), while the `sparse_lu`
path matches the stored count **51** and gives **45/45** exact cells. The
penalty is the measurement's fixed structural value (`lam_min+` for structural
scenes, `sqrt(lam_min+ lam_max)` otherwise); the packaged scene structure is
re-derived from the public JSON in `structure_check.json`.

Reproduction is a bounded consistency check on the exported operators, not a
new measurement and not a convergence theorem. The conditioning is reported as
the quotient of the positive spectrum (`lambda_max/lambda_min+`), never
`np.linalg.cond` on the singular operator.
