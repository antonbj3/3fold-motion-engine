# One shared latent cause through normal contact and a prescribed path

Opt-in modules `motion_engine.ncp.latent_contact` and `family_sweep` add a
solver-free replay contract for a conventional parametric normal LCP.
They do not replace the existing Coulomb solver or the existing CCD API.

The model is p>=0, w=G p+q0+q1*z>=0, p_i*w_i=0, z in a closed interval.
G is rational, symmetric positive definite and constant, with at most eight
normal impulses. Compliance, mass projection and the frozen contact frame
are already included in G. An exact LDL test licenses SPD. For SI input,
q has m/s, G has s/kg, p has N s, and each readout is c.p/h in N. External
benchmarks may instead explicitly use EXTERNAL_UNSPECIFIED; this never
becomes N by transport. Physical parameter/bound calibration is UNKNOWN.

SPD gives existence and uniqueness for every z: the coercive strictly convex
QP min_{p>=0} (p^T G p)/2+q(z)^T p has a unique minimizer, and its KKT
conditions are this LCP. For an active set S, G_SS is invertible and
p_S(z)=-G_SS^-1 q_S(z), p_notS=0. Its validity region is an interval because
p(z) and w(z) are affine. The bounded producer enumerates active sets and
stores an ordered cell cover. The verifier never enumerates or solves. It
rechecks SPD, cell coverage, endpoint inequalities and all three coefficients
of p_i(z)w_i(z). These coefficients must vanish. A singleton domain checks
point KKT instead. Uniqueness ensures compatible values at shared boundaries.

Each linear readout is affine on a cell, so its extrema occur at cell
endpoints. The original latent box's two endpoints alone are insufficient.
For Siconos's own G=[[2,1],[1,2]], q0=(-5,-6), with the explicitly added
q1=(7,-7) on [-1,1], the three cells have cuts -1/3 and 4/21. The sum p0+p1
has minimum 11/3 in the central cell, while the two outer endpoints give
6 and 13/2. All input information is also supplied to the standard control.

`port_family` accepts exactly one PORT v1 latent cause, a complete affine
contact inventory and hard MODEL_BOUND gaps in meters. It checks the affine
range against the declared bands and hashes the whole source packet. Sigma
alone, blocked contacts, duplicate identities and unconverted mm are rejected.
A response is UNIQUE_PER_LATENT_VALUE, not an assertion that the whole
family is one state. Prior source branches and physical status are not promoted.

For the swept query the caller declares a complete finite list of PT/EE pairs
and, for every force cell, endpoint vertices X0(z),X1(z) affine in the SAME z.
The path is explicitly X(t,z)=(1-t)X0(z)+t X1(z). This prescribed path must
not be mistaken for a certified trajectory of continuous mechanical equations.
For a fixed direction n, every projected primitive relative displacement is
multiaffine separately in t,z and each primitive's barycentric weights.
Its extrema occur on the product's vertices. A cell is safely separated when
its exact minimum m>0 and m^2>skin^2*||n||^2. Skin is the combined radius.
The proof covers degenerate primitive convex hulls as well. Failure to find
one fixed direction is UNKNOWN, not a collision or an impossibility proof.
All declared pairs and cells must be covered; an empty inventory is rejected.

A translation shared by all primitive vertices cancels before projection.
Independent vertex boxes lose this information. The candidate and the strongest
equally informed control both use ordinary relative scalar support. Changes
to force coefficients, source identity, geometry, pairing, cell cover, scope,
units or margins invalidate replay. The library does not derive completeness
of a physical broadphase, a source uncertainty family, or upstream affine
geometry from metadata. Those are explicit caller preconditions.

Only rational arithmetic and the stated finite model are certified. The lane
measured two standard-method ties. Neither a general 10x advantage nor method
novelty, anatomical validation or full circular Coulomb coverage is established.
