# Certified gap-box extensions

These are opt-in APIs on top of `motion_engine.ncp.gap_box`. Frozen rational
input models are required. Frames, body geometry and candidate inventories
must remain fixed over the step. Mass, friction and compliance are explicit
constitutive inputs, without a claim of physical calibration.

## Sharp bounds for parallel compliant normal contacts

`shared_gap.shared_gap_enclosure(pb)` and `verify_shared_gap(pb,answer)` accept
normal contact with one translational DOF, every J row equal to 1, inverse mass
positive, and strictly positive normal compliances. All original velocity,
displacement, contact/net force and per-contact normal-force readouts are
returned with sharp rational extrema and original-space gap/impulse witnesses.

For fixed g, p_i=max(0,(-v-g_i/h)/d_i). The scalar equation
v=v_free+im Σp_i has exactly one root: its left-minus-right derivative is
1+im Σ_active 1/d_i>0, its limits have opposite signs, and the function is
continuous. On each active region,

    ∂v/∂g_j = -im/(h d_j (1+im Σ_active 1/d_i)).

An inactive j has derivative zero. An active p_i decreases in its own g_i
and increases in every other active gap; inactive outputs have zero derivative.
The continuous piecewise affine extension preserves these signs through
boundaries. This is the license for prescribed extreme-gap queries. It applies
to these output coordinates, not arbitrary energy or arbitrary J.

The complete projected feasible set is described by momentum plus

    max(0,(-v-g_hi_i/h)/d_i) ≤ p_i ≤ max(0,(-v-g_lo_i/h)/d_i).

For p_i>0 lift with g_i=-h(v+d_i p_i); for p_i=0 choose any
g_i∈[max(g_lo_i,-h v),g_hi_i]. This proves exact projection, including its
existence and joint consistency. There are at most 2n+1 velocity intervals
using shared cuts -g_lo_i/h,-g_hi_i/h. The API returns those cuts and sharp
marginal support witnesses, rather than a false list of few constant networks.
The full independent-gap box can still have all 2^n strict contact networks.
This is a standard scalar water-filling/parametric contact construction;
method novelty is not claimed.

## Complete declared polygon friction in 3D

`polyhedral_gap.PolyhedralGapProblem.make` uses three J rows per contact:
normal,tangent1,tangent2. Polygon vertices must be rational, strictly convex,
counterclockwise, with the origin strictly inside and no collinear vertices.
`enclose_gap_box` and `verify_gap_enclosure` can consume this object directly.
`solve_polyhedral_gap` offers a fail-closed dimension gate before sparse-to-dense
conversion. The exact arithmetic budget remains 128 impulse+gap variables.

The declared tangential law is minimum u_t·p_t over μp_n P. For each contact
the complete closures are open, stick, one per vertex, and one per edge.
For a vertex a: p_t=μp_n a and u_t·(a-b)≤0 for every polygon vertex b.
For an edge with outward normal n and support ρ: n·p_t=μp_nρ,
u_t parallel to n, and n·u_t≤0. Polygon membership supplies the edge segment.
The normal complementarity law is unchanged. Every polygon minimizer lies
in a face; zero u_t is stick; otherwise the negative normal cone of the
minimizer face gives exactly these closures. Degenerate zero-normal solutions
are covered by open. Thus all directions are covered for this polygon law.

The default diamond satisfies |t_x|+|t_y|≤μp_n and is inside the circular cone.
`SQUARE` satisfies max(|t_x|,|t_y|)≤μp_n and contains the circular cone.
The diamond contains a disk of radius μp_n/√2; the square's radial extent is
at most √2 μp_n. These are coarse constitutive approximations. **Nested force
cones do not nest dynamic solution sets.** Sharp polygon solutions are not a
universal enclosure of the circular Coulomb solutions. Use the following
separate theorem for that claim.

## Global circular-Coulomb outer bounds

`coulomb_outer.SparseCoulombBox.make` takes sparse rows as `(DOF_index,Fraction)`
pairs, diagonal positive inverse mass, free velocity, hard gap intervals, h>0,
μ≥0 and optional d≥0. Floats must be converted explicitly to exact binary
rationals or caller-chosen rational input; silent rationalization is prohibited.
`energy_enclosure(pb,escape)` returns `OUTER_ENCLOSURE` only after verifying
all theorem assumptions and a strictly positive escape margin. An unsupported
separator returns `OSÄKER` with no bounds. The solver-free
`verify_energy_enclosure` rebuilds margins, checks square-root upper bounds,
and recomputes every output relation.

Let e be any virtual vector, not an assumed physical solution. Suppose
α=min_i[J_ni e−μ_i(|J_t1i e|+|J_t2i e|)]>0. For circular Coulomb, or any
dissipative law with |p_tk|≤μp_n, eᵀJᵀp≥αΣp_n. Define
B=max(0,max_i(-g_lo_i/h)), k=B/α. Normal complementarity and tangent
dissipation imply

    vᵀM(v-v_free) = pᵀJv ≤ BΣp_n ≤ k eᵀM(v-v_free).

Completing the square yields

    ||v-(v_free+ke)/2||_M ≤ ||v_free-ke||_M/2.

Cauchy-Schwarz and momentum then yield
Σp_n≤||e||_M ||v_free-ke||_M/α. Rational square-root upper bounds give
finite coordinate bounds and normal loads. The certificate covers **every
solution for every gap in the box**, including all circular slip directions.
Each per-contact normal-load upper bound reuses the global bound on the sum
of normal impulses divided by the step. These marginal bounds cannot be
added to infer a sharper global load, or read as attainable contact loads.
It neither proves a solution exists at every gap nor returns sharp extrema,
networks or witnesses. `existence=UNPROVED` is mandatory. Marginal intervals
must not be treated as independent realizable states. The separate sharp
mode API remains the route for a complete exact outcome set on small models.

The sparse operation is linear in J entries plus DOFs/contacts. It does not
enumerate networks. It may be very conservative; geometry/material uncertainty
outside the frozen input box and omitted new contacts are not covered.
