# Source: incidence_solvers

Manuscript: Appendix C (scalar incidence linear systems: Ruge-Stüben, smoothed
aggregation, direct, Jacobi-CG, tridiagonal) and Appendix D (preconditioned
contact linear solves).

Study: the scalar incidence-system solver comparison. `inputs/scaling.json` is
the scaling study on chains and random geometric graphs; `inputs/blockthomas.json`
is the block-Thomas comparison; `inputs/scenes.json` is the sanitized scene
table; `inputs/bit_exact_rerun.json` is the bit-exact large-packing
reproduction.

- Status: `regenerated` (from stored measurements).
- Regenerate here writes **all three** CSVs from the stored measurement JSONs:
  `incidence_solvers.csv` (Appendix C, AC1–AC6), `incidence_scaling.csv`
  (Appendix C, AC7, from `inputs/scaling.json`) and `incidence_blockthomas.csv`
  (Appendix D, AD1–AD3, from `inputs/blockthomas.json`). The command
  `python regenerate.py` reproduces the three delivered hashes exactly:
  `a64e9e4c…` (solvers), `7081a131…` (scaling) and `4111a2fe…` (block-Thomas).
- Internal scene provenance was replaced by public descriptions; no numeric
  value changed.
- Rerunning the physics is NOT possible from this directory (internal solvers);
  the stored `relative_residual` values are the evidence.
