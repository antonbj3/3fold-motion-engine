# Source: gpu_adjoint_identification

Manuscript: P2 (identification via the GPU adjoint; int64 determinism vs
float32 accumulation, CRB calibration, win7 OED excitation).

Study: identification via the GPU adjoint (int64 determinism vs float32
accumulation, CRB calibration, OED excitation).

`gpu_adjoint_identification.csv` aggregates the delivered `inputs/summary.json`;
the other `inputs/` JSONs are the delivered CRB results.

- Status: `verified_export_only`. All wall times carry the shared-GPU caveat
  (runs 163-349 s). Rerunning needs the Warp/GPU module and a hosted GPU account;
  it is not part of this package.
