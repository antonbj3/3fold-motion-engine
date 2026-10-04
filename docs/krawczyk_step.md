# Local root certificates for complete step residuals

`motion_engine.ncp.krawczyk_step.certify_root_box` checks outward enclosures of
the actual complete step force and Jacobian with exact rational arithmetic.
It includes every free neighbour and the entire pair list supplied by the
producer. PSD/projected/lagged Newton curvature is an invalid Jacobian input.

The producer identifies the residual and supplies centre-force bounds, box
radii, whole-box Jacobian bounds and a fixed point preconditioner. Float
endpoints are exact binary rationals; outward rounding is the producer's duty.
All accepted rational numerators and denominators are normalized to unbounded
Python integers, including NumPy integers and Fraction objects containing them.
Strict Krawczyk inclusion plus contraction yields
`UNIQUE_IN_BOX_IF_ENCLOSURES_VALID`. It is conditional on those bounds being
valid, and proves one root in the supplied box only. `global_root_count` always
remains `UNKNOWN`; global reachability, topology coverage and physical branch
continuation need separate proofs. Invalid input or a failed test is UNKNOWN.

No optional numerical package is needed. This checker is not a replacement
for a residual exporter, interval AD, solver, time-step controller or global
domain proof. Research basis: lane SOL_FALT_BARRIARSTEG_20261002; integration
base d073f8b616fc72017190fd95761f8fab9302d88e; independent review pending.
