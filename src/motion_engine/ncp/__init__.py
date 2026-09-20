"""Exact-cone (de Saxce) contact solvers, their adjoints, and the router over them.

`ncp_ref`   CPU float64 reference: exact cone, PGS and ADMM, analytical sensitivities.
`ncp_gpu`   GPU coloured PGS on the same formulation, int64 fixed point, bit-identical.

Each module's docstring carries the numbers its test file locks, and the audit
reservations attached to them.
"""
