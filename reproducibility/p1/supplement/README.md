# Contact-graph reproduction supplement

Standalone inputs, generators and aggregation scripts for the three experiment
families that the accompanying manuscript does **not** cover with its exported
operator dataset:

1. **Patch-wrench systems** (`patch/`) — the patch/chain operators, the scalar
   incidence systems and the block-elimination and linear-solver tables.
2. **Load-ordering packings** (`load_ordering/`) — the contact graphs and the
   converged contact loads of the eight scenes behind the effective-resistance
   load-ordering results.
3. **Ramp geometry** (`ramp/`) — the three quasistatic support scenes, the 24
   load directions, the solver/detection protocol and the measured brackets.

Everything needed to reconstruct the delivered problems is in this directory.
No file outside it is required.

## Quick start

```sh
python loader.py                 # load every exported file, rebuild operators
python verify.py                 # representative round-trip per family
python verify.py --all           # every scene
python patch/table_appendix_a.py
python patch/table_appendix_b.py
python patch/table_appendix_c.py
python load_ordering/build_corrected.py --write   # regenerate corrected reference
python load_ordering/table_load_ordering.py
python ramp/table_ramp.py
```

`dependencies.txt` lists the requirements. Only NumPy is needed for the patch and
ramp families; SciPy is needed for the sparse flow solve on the largest
load-ordering graph. `pyamg` is optional and only used if the Appendix C AMG
counts are to be recomputed rather than read from the saved measurements.

## What is recomputed and what is aggregated

| Script | Recomputes | Aggregates (saved measurements) |
| --- | --- | --- |
| `patch/table_appendix_a.py` | assembly identity, ranks, vertex pencil | — |
| `patch/table_appendix_b.py` | block bandwidth, dense-vs-block linear solve | outer iteration counts, impulse difference |
| `patch/table_appendix_c.py` | direct solve, Jacobi-CG solve | AMG iteration counts |
| `load_ordering/build_corrected.py` | independent direct flows, diagnostics, corrected ranks | k=64 resistance sketch (not shipped) |
| `load_ordering/table_load_ordering.py` | resistance flows, geometric baselines, Spearman/precision | k=64 resistance-sketch metrics |
| `ramp/table_ramp.py` | static radius from geometry | ramp bracket midpoints |

No contact physics is rerun. The load-ordering contact graphs and loads, and the
ramp bracket midpoints, are the exported outputs of the delivered runs.

## Conventions

### Patch and chain systems

* One patch per body pair, at the patch centroid `p0`, frame `(t1, t2, n)`,
  `n = t1 x t2`, half extents `e1 > 0` along `t1` and `e2 > 0` along `t2`, four
  corner point contacts.
* Generalized velocity per body is `(v, omega)`, translation first.
* Patch wrench order `w = (F_n, M_t1, M_t2, F_t1, F_t2, M_n)` with dual twist
  rows `(n.v(p0), e2 t1.omega, e1 t2.omega, t1.v(p0), t2.v(p0), ell n.omega)`,
  `ell = sqrt((e1^2 + e2^2)/3)`.
* Admissible patch cone

  ```text
  K(mu) = { F_n >= 0, |M_t1| <= F_n, |M_t2| <= F_n,
            ||(F_t1, F_t2, M_n)|| <= mu F_n }.
  ```

  The first three inequalities are exactly the image of four non-negative corner
  normal loads; the last is the Contensou-type friction-plus-spin ellipsoid.
* Sign convention: patch `(a, b)` has body `a` on the `+1` side and body `b` on
  the `-1` side; `b = -1` is the static ground, which contributes no inverse mass.
  Contact directions enter the translational columns and their cross product with
  the moment arm enters the rotational columns, with opposite signs for the two
  incident bodies.
* Solver convention: `u = G w + b`, `w` in `K(mu)`, de Saxce shift
  `Gamma(u) = (mu ||(u_Ft1, u_Ft2, u_Mn)||, 0, 0, 0, 0, 0)`, natural-map residual
  `||w - P_K(w - rho (u + Gamma(u)))||_inf`.
* Units: metre, kilogram, second; impulses in N s; angles in radians; friction
  dimensionless; gravity 9.81 m/s^2.

### Load ordering

* Nodes `0..N-1` are grains; the remaining nodes are floor, foot and wall as
  listed per scene.
* One edge per contact `(grain, destination node)`.
* Conductance models: `inverse_mass` (2 between two grains, 1 otherwise),
  `unweighted` (1), `hertz_stiffness` (`sqrt(0.5)` between two grains, 1
  otherwise).
* Load injection: foot scenes drive a unit current from the foot node to the
  floor node; gravity scenes drive a unit current into every grain, returned at
  the floor node.
* `lam_n` is the converged normal contact impulse (contact row 0) of the
  delivered solver; `lam_mag` is the full impulse norm.
* Reference solution: `solve_flow(method="direct")` uses a sparse LU of the
  grounded Laplacian with one gauge node pinned per connected component.  The
  grounded matrix is singular on components without the grounded node (isolated
  grains, the wall), so each component gets its own reference potential; edge
  currents are gauge-invariant.  `method="minres"` is the legacy iterative
  branch and is kept only for the convergence sweep: its residual-based stop
  does not bound the current error on these ill-conditioned Laplacians.
* Ties rule (fixed before comparison): the series is sorted stably
  (deterministic order) and clustered by leader scanning with radius
  `tol = tie_rel * max|series|`: scanning in sorted order, a value joins the
  current cluster while it is at most `tol` above that cluster's FIRST value,
  otherwise it starts a new cluster; every member of a cluster receives the
  average of the ranks it covers.  `loader._rankdata` implements exactly this.
  Rationale: the direct LU backward error is at most ~1e-11 and two independent
  solves of the same grounded system (different gauge pin and different sparse
  solver) agree to <=5.54e-12 relative to the maximum edge current, while the
  upper admissible radius 1e-8 is still far below the physical resolution;
  `measurements_corrected.json` reports the interval over the grid
  `{0, 1e-12, 1e-10, 1e-9, 1e-8}` and the breakdown scale 1e-6.
* The reported load-ordering number is convention-sensitive; it is not a
  policy-independent confidence bound.  For the `hertz_foot_dw8` inverse-mass
  series at `tie_rel = 1e-9`, the leader clustering above gives
  0.3992709577045221, the transitive-closure reading of the same radius (a
  value shares a rank with any earlier value connected through a chain of
  gaps <= `tol`) gives 0.31899045477279486, and exact-equality ties
  (`tie_rel = 0`) give 0.3996791630889775.  Only the leader value is the one
  this package reports; the other two are stated as the sensitivity to the
  rank convention.

### Ramp

* One rectangular support patch per target body. A horizontal acceleration field
  of magnitude `L` in direction `phi` is applied to every body, together with
  gravity.
* Static directional radius (Proposition 7):

  ```text
  rect_tip(a_u,a_v,u0,v0,phi0,h,phi)
      = g/h * min( (a_u - sign(cos)*u0)/|cos|, (a_v - sign(sin)*v0)/|sin| )
  rect_slide = mu * g
  radius(phi) = min(interface_tip, interface_slide, ground_tip, ground_slide)
  ```

  with the interface terms present only when the target rests on a support body.
* `wrench_radius[scene]` is indexed by geometry direction `0..23`; the field
  `direction` carries that index and `source_row` preserves the index in the
  source wrench table (`0,3,...,69`).  (Releases before this one labelled
  `direction` with the source row, which did not match the geometry order.)
* Detection protocol (retained from the experiment): time step 0.005 s, 60 PGS
  sweeps per step, contact residual target 1e-8 N s, displacement threshold
  5 mm, tilt threshold 5 degrees, acceleration ladder `0.25 * 2^k` for
  `k = 0..7`, 8 bisections, step horizon 0.6 s.

## Files

```text
loader.py                     load and rebuild every operator
verify.py                     standalone round-trip verification
dependencies.txt              requirements
manifest.json                 file list, sizes, SHA-256
SHA256SUMS                    full SHA-256 of every file
patch/generate_patch.py       self-contained patch/chain generator
patch/scenes.json             patch/chain geometry
patch/operators.npz           assembled J_patch, J_points, C, G, b, mu
patch/incidence.npz           scalar incidence systems L, body pairs, rhs
patch/incidence_scenes.json   incidence scene metadata
patch/measurements_appendix_b.json   saved dense/block iteration counts
patch/measurements_appendix_c.json   saved AMG solver counts
patch/table_appendix_a.py     assembly/rank/pencil table
patch/table_appendix_b.py     dense-vs-block elimination table
patch/table_appendix_c.py     incidence linear-solver table
load_ordering/scenes.json     load-ordering scene metadata
load_ordering/scenes.npz      graphs, contact points, converged loads
load_ordering/measurements.json          delivered (aggregated) metrics
load_ordering/measurements_corrected.json  direct-solve metrics + tie interval
load_ordering/build_corrected.py         rebuild the corrected reference
load_ordering/table_load_ordering.py
ramp/geometry.json            ramp scenes, directions, protocol
ramp/measurements.json        saved geometry/wrench radii and brackets
ramp/table_ramp.py
```

## Known gaps

* The patch/chain generator reproduces the delivered geometry exactly (bit
  identical operators). The load-ordering contact graphs and the ramp brackets
  are exported inputs, not regenerated from the grain-scale physics; the grain
  positions are included for the foot scenes only where the delivered loader
  used them.
* The k=64 resistance-sketch metrics are aggregated only; the sketch
  implementation is an external module and is not shipped.
* The delivered load-ordering ranks in `load_ordering/measurements.json` were
  computed with a solver-dependent reference (MINRES at `rtol=1e-5` has up to
  27% relative flow error on the largest scene).  `measurements_corrected.json`
  gives the independent direct-solve reference with the fixed tie rule and the
  explicit interval; for the resistance flows the tie-rule interval is at most
  1.6e-3, and the largest interval over all predictors is 3.4e-3 (the contact-
  degree baseline on `granular_step`, which is dominated by exact ties).
* The long-horizon ramp rows (box, direction zero, 50 s and 100 s) are exported
  as saved measurements; the 24-direction sweep covers ramp times 0.05–20 s.
* The patch outer-iteration counts (Appendix B) and the AMG counts (Appendix C)
  are saved solver runs; the linear algebra behind them is recomputed, the
  iteration schedule is not.

## Licensing and sources

The code and data in this directory are the author's own work from the
accompanying study; no third-party source code is copied into the package.
Third-party numerical libraries are only imported at run time: NumPy
(BSD-3-Clause), SciPy (BSD-3-Clause, bundling OpenBLAS/LAPACK under
BSD-3-Clause and GCC runtime libraries under GPL-3.0-with-GCC-exception /
LGPL-2.1) and optionally PyAMG (MIT).  Because no library source is
redistributed here, only the author's own code is licensed by this package;
the runtime libraries keep their own licenses.  The contact-law and solver
conventions follow the references cited in the manuscript (Delassus; de Saxce
and Feng; Acary, Cadoux, Lemarechal and Malick; Cadoux; Contensou; Boyd et al.;
He, Yang and Wang; Wohlberg); no code from those works is included.
