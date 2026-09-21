# Preprints

Both manuscripts are under revision. The PDFs here are the first versions of 20 September 2026: the results are measured and locked to the tests and data in this repository, but the text is written as a measurement record and is being rewritten for readers. A revised version with a reproducibility package (scene definitions, one script per table) follows.

| | Title | Download |
|---|---|---|
| P1 | Contact Delassus operators as weighted Laplacians on the contact line graph | [P1.pdf](https://raw.githubusercontent.com/antonbj3/3fold-motion-engine/main/docs/papers/P1.pdf) |
| P2 | Deterministic differentiable exact-cone contact on GPUs: bit-identical adjoints, batched routing and system identification | [P2.pdf](https://raw.githubusercontent.com/antonbj3/3fold-motion-engine/main/docs/papers/P2.pdf) |

What can be verified today: `tests/test_ncp_*.py` recompute the locked reference values in `data/ncp/` (batch determinism, adjoint exactness, Pinocchio parity, friction identification); `SHA256SUMS` lists the digest of each PDF and `tests/test_papers_integrity.py` recomputes both.
