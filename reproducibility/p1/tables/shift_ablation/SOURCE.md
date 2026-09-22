# Section 4.2 - de Saxce shift ablation

Row-level source for the two printed Section 4.2 tables. All 252 cells
(21 scenes x 4 penalty rules x 3 shift modes) contain the archived measurement
values of the ablation study, with public scene aliases; the manuscript medians, convergence counts and
penalty-change totals are read out of these rows, not from the displayed
numbers.

## Delivered files

- `results.csv` - 21 rows x (4 rules x 3 shifts) raw iteration counts, converged
  flags and penalty-change counts.
- `summary.json` - the archived per-(rule, shift) aggregates (median, converged
  count, total penalty changes).
- `table_shift_ablation.py` - public builder: prints the 12-row aggregate table
  and the 4-row dense-random table from the two files above.

## Source identity

| file | sha256 |
| --- | --- |
| archived `results.csv` | `cbadf374260edf086ae954171e056f4c2df459b35cdf415ce3bbcdc732d00c98` |
| archived `summary.json` | `e7bb612b1125cdf6e83f951df427616baa65c6f2c294481e529d53c1a2bd3033` |

The producer executes the de Saxce iteration and a sparse-LU solve; it is not
bundled and the physics cannot be rerun from this directory. The delivered rows
are the exported measurement. Two independent process runs produced identical
record and solution sha256 for all 252 cells.

## Limits

- The `off` and `fixed` arms change the inner target as well as the outer shift
  handling; this is the shift-schedule comparison, not an isolated outer-stop
  comparison.
- Capped counts (`C`) remain in the accounting; a ratio of two capped counts is
  not a cost ratio.
- One archived cell (scene 9, He-structure, fixed shift) took 72.9 s in the
  archived run and is retained as a reported failure, not smoothed.
