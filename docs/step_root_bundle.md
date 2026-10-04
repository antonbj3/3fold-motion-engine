# Conditional root proof bundle and free linear paths

Two disjoint existence boxes of the same complete residual prove at least two
roots. A negative energy variation proves instability of that energy; one
local Krawczyk box cannot exclude other roots. A failed uniqueness bound
returns UNKNOWN.

`motion_engine.ncp.step_root_bundle.certify_step_roots` reruns the reviewed
exact Krawczyk checker, verifies problem identity, domain inclusion and closed
box disjointness, and checks geometry for every endpoint in each box. All
arithmetic is exact rational arithmetic. The true residual/Jacobian enclosures
and correct SHA-256 binding remain explicit producer premises.

Geometry is defined as nonempty filled XY rectangles with positive size,
zero thickness and an affine translation map from the complete coordinate
vector. For each pair one strict separating axis keeps the same order at the
initial state and throughout the endpoint box. Affinity preserves that order
along every linear path to the box. An axis change can conservatively decline.
Arbitrary meshes, deformation, nonlinear continuation, thickness and changing
contact features are outside this API. A rectangle is a defined geometry;
this function does not infer complete mesh topology from bounding boxes.

The caller's SHA-256 binds source, full step state, h, true residual, degrees
of freedom, domain, geometry and path law. Every force enclosure is at the
supplied centre; the true Jacobian enclosure covers its whole box. This
function does not create or independently verify those producer data. PSD or
lagged Newton matrices fail the mandatory TRUE_RESIDUAL role.

Outputs separate mathematical multiplicity and the count of roots with free
paths. `accepted_step` stays false: this API does not choose a physical branch
or bound integration error. One box always leaves global_root_count UNKNOWN.
No original Ando scene has the required full producer data in this delivery.

Base: d073f8b616fc72017190fd95761f8fab9302d88e plus reviewed
locally frozen code/sources/PRIOR_FROM_BASELINE.diff SHA-256
99060b2a2ef4e211f379834ad86ec404a8e0831742cb133dd68ccd0ebc755419. The local prior patch reconstructs the exported reviewed
BARRIARSTEG files byte-for-byte; PATCH_BASE.json records changing parent hashes.
