# Optional exact gap-box contact API
Import `GapProblem`, `enclose_gap_box`, `solve_gap_box` and `verify_gap_enclosure` from `motion_engine.ncp.gap_box`. Existing NCP/ADMM/Newton entry points are unchanged.

```python
from motion_engine.ncp.gap_box import solve_gap_box
answer = solve_gap_box(
    J=[[1]], inverse_mass=[1], free_velocity=[-1],
    gaps=[[0, '1/50']], step='1/100',
)
assert answer['status'] == 'MÄNGD'
assert answer['bounds']['normal_force:0']['hi'] == 100
```

Use only integers, Fraction or exact rational strings. A floating scanner gap must first receive a declared upstream enclosure; rounding its nominal value to a rational is not evidence of physical coverage. No `complete=True` caller assertion is accepted.

Normal-only uses one J row per contact and requires mu=0. `planar_coulomb` uses two rows `(normal,tangent)` per contact. Both accept positive diagonal `inverse_mass`, `initial_velocity` (default zero), and `compliance` (default zero). With coherent SI coordinates, momentum is `v=v_free+Minv J.T p`; normal closure is `w_n=J_n v+g/h+d p_n`. d is a declared physical normal compliance in s/kg. Plane friction is exact Coulomb: `|p_t|<=mu p_n`, sticking `u_t=0`, sliding `p_t=-sign(u_t)*mu*p_n`. J, all normals and inventory are frozen for this single step. A fixed J need not correspond to every contact candidate that moving geometry could generate.

`ENTYDIG`: every reported readout is constant over all allowed gaps and all contact solutions. `MÄNGD`: a complete bounded relation, with exact extremal solution witnesses. This includes continuum variation and force fibres at fixed motion; it does not assert multiple dynamical roots at a single gap. `OSÄKER`: invalid/unsupported input, exhausted prefix/dimension budget, unbounded readout, empty model or failed arithmetic discovery. Its `bounds` is empty. Partial records may aid debugging and never imply completeness.

Bounds contain generalized velocity, h*v displacement, contact generalized force J.T*p/h, net generalized force M*(v-v_initial)/h, and individual normal loads p_n/h. The consumer supplies body labels and which DOFs are translation or rotation; corresponding units are m/s or rad/s, m or rad, and N or N m. Body net includes external force already encoded in v_free. A consumer can sum oriented translational body readouts to obtain system net force; internal contact contributions cancel only if its J correctly encodes both bodies.

Marginal intervals are sharp within this model. They are not an independent joint output box. `records` retain full shared-gap mode polyhedra via replayable prefixes, feasible points, and endpoint LP certificates; rebuilding the system from the same GapProblem reconstructs their correlation. Open/closed and stick/slip closures overlap at zero, so mode counts are not distinct physical histories.

`contact_candidates` gives sharp possible/necessary load and conservative geometry investigation flags. Uncertain force can also come from another gap or a force fibre; these flags are not a claim that measuring that one gap alone resolves the problem. Paired witnesses with all other gaps equal can prove that a gap changes output.

HiGHS/SciPy discovers bases only. Accepted points, bounds and rejected prefixes pass exact rational checks. `verify_gap_enclosure(problem, answer)` independently checks input hash, equality/Farkas identities, primal/dual extrema, full tree coverage, bound-attaining witnesses and candidate summaries, without calling an LP solver. Serialized rational certificates can be replayed through the same verifier.

Defaults: max_nodes=2048; more than 128 total impulse+gap variables returns OSÄKER before building the dense arithmetic system. The limit controls implementation cost, not mathematical validity. Full 3D Coulomb has continuous slip directions and is unsupported. Large heaps, changing contact frames, history/rolling/restitution laws, non-diagonal mass, continuous-time trajectories and calibrated clinical/physical guarantees require a different certified oracle.

The approach uses standard mode-polyhedron enumeration, Farkas and LP duality. No general speedup or new mathematical method is claimed. Performance is per gap-box question; caching endpoint objectives within each leaf is included in the measured solve count. Python/SciPy startup and a separate replay add costs. Broadening a box can make many previously prunable modes feasible.
