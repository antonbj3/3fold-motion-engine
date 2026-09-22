# Source: solver_relaxation

Manuscript: Appendix G (APGD, convex projected Gauss-Seidel, exact-cone ADMM).

Study: the convex-relaxation vs exact-cone solver comparison. Each scene was
solved by three methods in two independent process rounds (63 solver rows per
round). `solver_relaxation.csv` is a regeneration from the stored per-solver
measurements in `inputs/`; it does not rerun physics.

- Status: `regenerated` (from stored measurements).
- Rerunning the physics is NOT possible from this directory: it needs the
  internal reference cone-ADMM solver. The stored solution SHA-256 values are
  the bit-identity evidence (63/63 identical between the two rounds).
- Regenerate here: `python regenerate.py`.
