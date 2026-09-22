# Source: gpu_batch_accuracy

Manuscript: the accuracy x repeatability x cost table on the corrected GPU
batch-adjoint chain (int64 fixed-point vs float32/float64 accumulation).

Study: the GPU batch-adjoint accuracy, repeatability and cost table.
`gpu_batch_accuracy.csv` is the delivered table with the device string replaced
by a public description; all numbers unchanged.

- Status: `verified_export_only`. GPU runs were measured on a dedicated L4;
  local RTX timings carry the shared-GPU caveat and are not in this table.
- Rerunning needs the Warp/GPU module and a hosted GPU account; the exported
  measurements are the evidence.
