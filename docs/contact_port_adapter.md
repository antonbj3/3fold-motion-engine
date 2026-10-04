# Opt-in motion contact port adapter

Requires the canonical `field_engine.contact_port_v1` field patch/package.
The adapter normalizes accepted rational numerators/denominators to unbounded
Python ints, including prebuilt Fractions containing fixed-width NumPy storage,
before unit conversion and scalar-law arithmetic. The input policy is unchanged.
The existing Scene contract and all default solver behavior remain intact.

Motion contact dictionaries provide `gap` in metres. Scene.b is J*v_free plus
an optional offset, in m/s, and must never be imported as a length. `solver_gap`
requires an explicit mapping: RIGID_SPECULATIVE uses max(g,0)/dt, as build_scene;
CLOTH_SIGNED uses g/dt, as cloth contact. No mapping is inferred from a label.
`point_scene` realizes one point in the band in a copied Scene and rejects an
existing normal offset, mismatched time step or unresolved completion. The
caller retains the interval request; a point solve is not a band certificate.

`from_cloth_distance` uses h/2 against SUPPORT and h for VERTEX_TRIANGLE or
EDGE_EDGE self-contact. `from_mpm_cells` requires physical cell size in metres;
it transports the stated cell interval without claiming a continuum bound.

`solve_normal_gap` is restricted to one frictionless normal with G_nn>0 (1/kg).
For a positive dt, lambda(g)=max(0,-(v+offset(g))/G_nn) is nonincreasing. Hence
its exact endpoint interval contains the whole gap interval. Output is in N s.
The license is proved before routing: it is not inferred from samples. σ/dt
is a pre-clipping scale; clipping makes it unsuitable as a Gaussian sigma for
rigid offset. Conditions and shared causes remain with the Gap/evidence.
Frictions or general coupled NCPs require their own coverage oracle and are
rejected by this helper. No CCD or nonpenetration claim is made.

`identical_normal_impulse_difference` handles only identical affine gaps, the
same specified scalar law and resolved completion. Identity gives exactly zero
for every shared latent realization. Unequal relations and bare independent
marginals are refused. This is an equality license, not general corner routing.

`from_global_verdict` preserves the upstream observable scope: unique velocity
is not unique individual impulse. Its impulse existence witness stays in the
witness payload, outside the purportedly unique state. `from_reachability`
retains all edge witnesses and multiplicities; UNKNOWN never becomes complete.
Imported verdicts remain upstream claims until caller-side replay. Neither
adapter supplies a deformable collision backend or a clinical model lift.
