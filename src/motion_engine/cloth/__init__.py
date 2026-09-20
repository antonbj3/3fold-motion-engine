"""Lagrangian cloth: membrane, hinge bending with measured hysteresis, and exact-cone
self-contact. No background grid.

`r4_kernels` / `r4_contact` / `r4_solver`   membrane, hinge, tension field, the
    point-triangle and edge-edge contact set solved as the same cone complementarity
    problem as rigid contact.
`r5_solver`   the one-step contact fixed point, lumped mass, spill-free candidate buffer.
`r6_kernels` / `r6_contact` / `r6_setup` / `r6_solver`   the two-line bending moment law
    from a measured KES loop, as a Jenkin element in parallel with the elastic spring.
"""
