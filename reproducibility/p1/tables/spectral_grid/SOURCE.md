# Appendix E - spectral penalties and grid minima

Row-level source for the 21-row Appendix E table and for the "five of
twenty-one scenes" central spectral-rule statement. The rows are the archived
output of the penalty-grid sweep and match a second process run. Public scene aliases and English
+yes/no flags replace the archived labels in this copy.

## Delivered files

- `spectral_grid.csv` - 21 rows: lambda_min+, lambda_max, rho*, iterations,
  selected rule, parameter distance, boundary flag, capped/nonfinite counts and
  the per-scene operator sha256.
- `table_spectral_grid.py` - public builder: prints the printed rows and the
  `within_2_selected` count that yields the 5/21 rule result.

## Source identity

| file | sha256 |
| --- | --- |
| archived sweep run 1 | `b9487ea579085675c12ef2fd8bc41b420b0bf1515c1951b89558370d6a8249b7` |
| archived sweep run 2 | `b9487ea579085675c12ef2fd8bc41b420b0bf1515c1951b89558370d6a8249b7` |
| second run identical | `True` |

## Limits

- The sweep used the internal ADMM iteration; the physics is not bundled and is
  not rerun here. The delivered rows are the exported measurement.
- Capped and nonfinite candidates remain in the accounting (`censored_grid_points`,
  `numerical_failures`).
- The eigenvalues refer to the operator used in the penalty experiment, including
  its declared compliance; the condition number is `lambda_max/lambda_min+`, not a
  nonsingularity claim.
