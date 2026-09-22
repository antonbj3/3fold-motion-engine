# Appendix B - dense and block elimination (row source)

Row-level source for the four-row Appendix B table. The measurement file
`supplement/patch/measurements_appendix_b.json` is already delivered in the
supplement; this entry adds the public row builder and records the row identity.

## Delivered files

- `../supplement/patch/measurements_appendix_b.json` - the archived solver-run
  aggregate (outer iteration counts, dense-vs-block impulse difference, status).
- `table_appendix_b_rows.py` - public builder: prints the four printed rows.
- `../supplement/patch/table_appendix_b.py` - recomputes the block-tridiagonal
  property and the dense-vs-block linear solve from the exported geometry.

## Row identity

| scene | dense updates | block updates | max abs impulse difference [N s] | status |
| --- | --- | --- | --- | --- |
| Eight-box tower, gravity | 140 | 140 | 1.67e-16 | converged |
| Eight-box tower, rolling | 1848 | 1848 | 2.37e-15 | converged |
| Mass-ratio column (1000:1), gravity | 296 | 296 | 4.39e-12 | converged |
| Mass-ratio column (1000:1), rolling | >30000 | >30000 | 1.22e+01 | both capped; iterate difference |

The final row compares unconverged iterates and does not measure solution error.

## Limits

- The outer iteration counts are the exported measurement; only the linear-solve
  agreement is recomputed from the delivered geometry.
