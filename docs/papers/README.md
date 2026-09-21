# Preprints

Both manuscripts are under revision. The PDFs here are revision 2 of 21 September 2026: the results are unchanged from the first version and stay locked to the tests and data in this repository; the text was rewritten for readers, the measurement register was replaced by tables, and the convex-relaxation, adaptive-step, Laplacian-solver and Siconos baselines were added. Revision 3, with a reproducibility package (scene definitions, one script per table) and the matched accuracy, repeatability and cost measurements, follows.

| | Title | Download |
|---|---|---|
| P1 | Contact Delassus operators as weighted Laplacians on the contact line graph | [P1.pdf](https://raw.githubusercontent.com/antonbj3/3fold-motion-engine/main/docs/papers/P1.pdf) |
| P2 | Deterministic differentiable exact-cone contact on GPUs: bit-identical adjoints, batched routing and system identification | [P2.pdf](https://raw.githubusercontent.com/antonbj3/3fold-motion-engine/main/docs/papers/P2.pdf) |

What can be verified today: `tests/test_ncp_*.py` recompute the locked reference values in `data/ncp/` (batch determinism, adjoint exactness, Pinocchio parity, friction identification); `SHA256SUMS` lists the digest of each PDF and `tests/test_papers_integrity.py` recomputes both.
