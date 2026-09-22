# Source: siconos_external

Manuscript: Appendix H (external friction-contact solutions: NSGS, ACLMFP,
PROX, ADMM, DSFP, NSN_AC on the same operators, biases and friction).

Study: external friction-contact solver comparison. `siconos_external.csv` is
the delivered table with the internal reference-residual column renamed to
`reference_residual`; all values are unchanged.

- Status: `verified_export_only`. The runs used the external Siconos 4.3.1
  bindings with a NumPy<2 environment; the raw reaction arrays are stored in the
  source report.
- Rerunning is NOT part of this package (external dependency + long wall times,
  17 runs hit the 30 s cap); the exported measurements are the evidence.
