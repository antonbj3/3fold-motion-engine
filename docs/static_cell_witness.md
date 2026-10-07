# Exact static placement-cell witness

`motion_engine.ncp.static_cell_witness.certify_static_cell` checks one proposed
contact pair and normal-force weight. `STATIC_FEASIBLE_AT_LEVEL` proves static
wrench feasibility for every placement in the specified closed square, every
horizontal load direction, every allowed pelvis height and horizontal offset,
and every nonnegative load up to the supplied level. Equality is included; this
does not assert a strictly larger feasible load. `UNKNOWN` means that this
candidate did not establish feasibility. It never means infeasible or unsafe.

This opt-in module has no solver-path integration and uses only Python's standard
library. It neither selects pairs nor imports a physical Scene. Candidate search
may be approximate, because every accepted candidate is checked exactly.

## Declared model and inputs

- Eight exact rational point contacts on `z=0`, with upward normals. Exactly four
  rotate by the specified integer yaw about their arithmetic mean and translate
  together; the other four remain fixed. Contact indices are zero-based.
- The translation lies in a square centered at `cell_center_xy`, with
  `|u_x-center_x|, |u_y-center_y| <= cell_side/2`. The COM projection is fixed
  throughout this square. Yaw is a Python integer from −180 through 180 degrees.
- Exact gravity `981/100`, friction coefficient `mu=1`, and the ideal regular
  16-ray polygon with radial scale `2/(1+cos(pi/16))`. This polygon straddles the
  circular Coulomb cone. The witness uses its inscribed disk, whose radius factor
  `tau=2*cos(pi/16)/(1+cos(pi/16))<1`, so these particular forces also satisfy the
  circular cone; the polygon and circular feasible sets are not identified.
- Any positive mass; the mass cancels from the sufficient inequalities. The
  horizontal load is an acceleration in m/s² applied at the pelvis with
  `0 < h <= height_max` and `|pelvis_xy-com_xy| <= pelvis_offset_max`.
- Two distinct contact indices and an exact weight `0 < weight < 1/2`.

All positions, cell dimensions, heights and offsets are in metres. The level is
acceleration/load per mass in m/s², not a force in newtons.

Scalar inputs accept Python `int`, `fractions.Fraction`, or integer/ratio strings
such as `"11/10"`. Numerators and denominators are limited to 256 bits and strings
to 160 characters. Decimal/exponent strings, floats, bools and NumPy scalars are
rejected. Lists/tuples must have the specified dimensions. Side, level and offset
are nonnegative; height is positive. Unsupported inputs raise `TypeError` or
  `ValueError`. These checks precede geometry arithmetic. The module does not accept
caller-provided rotation errors or an unchecked interval certificate.

The eight points need not form rectangles. This is an explicit extension of the
P4 sole-box presentation: its convex-combination and pair-force argument requires
only coplanarity, not rectangularity. There is no extension to noncoplanar points,
variable yaw, variable COM, other friction laws, measured uncertainty, joint torque,
compliance, dynamics, reached branches, physical support loss or safe trajectories.

## Why the accepting check is exact

Let `w` denote the weight, `p_i,p_k` the chosen pair, and
`E = w(p_i+p_k)+(1-2w)conv{p_j:j not in {i,k}}`. P4's constructive witness fixes
the pair normals at `w*m*g`, distributes the remaining normal force to realize
the required center of pressure, then adds an equal-and-opposite tangential pair
to supply the pelvis-offset yaw torque. Sufficient conditions are

```
B(com, height_max*level/g) subset E(u) for every translation u in the cell
level*(w + pelvis_offset_max/ell_lower) <= tau*w*g
ell_lower > 0.
```

Machin's identity and consecutive alternating rational sums enclose pi. Exact
interval Taylor recurrences with analytic remainders enclose the ideal yaw sine
and cosine. Every rounded endpoint is directed onto a 96-bit dyadic grid.
Integer square roots provide proved lower and upper norm bounds. A rational
surrogate for each rotated point has a computed common error bound `err`.

Each vertex of `E` is a convex combination of contact points, so its rotation
error is at most `err`. Its translation is `gamma*delta_u`, with `0<=gamma<=1`.
Thus all support functions change by at most `err+rho`, where the computed upper
bound `rho >= cell_side/sqrt(2)` holds exactly. This argument covers every support
direction and survives changes of hull topology; it does not sample directions.

The rational surrogate hull is built with exact orientation predicates. On every
CCW edge `e=b-a`, the signed quantity `A=cross(e,com-a)` must be positive and
`A*A >= R*R*dot(e,e)`, with `R=height_max*level/g+rho+err`. The sign check prevents
a squared-distance test from accepting a point outside the hull. These tests
prove ball containment after erosion. The pair-distance lower bound is
`sqrt_lower(|p_i-p_k|²)-rho-2*err`. The friction factor is bounded below using
nested exact square-root bounds for `cos(pi/16)` and the monotonic function
`2*c/(1+c)`. All accepting comparisons use rational arithmetic, without a fixed
epsilon or floating-point trust. These are executable arithmetic checks with a
written mathematical argument, not a proof-assistant formalization.

## Source relation, example and limitations

The construction is derived from
[`reproducibility/p4/WITNESS_LEMMA.md`](../reproducibility/p4/WITNESS_LEMMA.md),
especially its Contract, Witness and Whole cell sections. The frozen P4 manuscript
records that its historical outward interval implementation did not establish
strict outward rounding. This module supplies a new exact verifier for the stated
model; it does not alter that historical record or certify the published areas.

Run `PYTHONPATH=src python examples/static_cell_witness_demo.py`. The example binds
the released `raw/certified_regions.json` by SHA256 and explicitly interprets its
stored binary64 coordinates as exact rationals. Ideal +20°, +60° and −70° rotations
are enclosed internally. Gravity is the newly declared rational `981/100`, not an
attempt to recover a historical binary64 gravity constant. The new envelope is
`h<=11/10`, pelvis offset `<=1/10`, level `1/20`, cell side `1/50`, and weight
`1/26`. All three center cells return `STATIC_FEASIBLE_AT_LEVEL`.

The associated bounded research supplement checked 75 preselected cells using a
floating search over all 28 pairs and 12 newly declared weights, then exact checks:
53 passed (17,18,18 by yaw) and 22 remained UNKNOWN. The same floating witness scan
proposed 53 positive cells. A separate direct-LP diagnostic found sampled
feasibility even in three selected UNKNOWN cells; the sufficient witness is
conservative. The LP samples cannot prove a whole-cell/all-direction statement.
No speed advantage, optimal coverage, full-grid classification, historical region
reproduction, generic contact theorem or physical trajectory guarantee is claimed.

Returned rational strings are diagnostic receipts for this supplied input. To
verify an untrusted receipt, rerun the function with the exact bound inputs;
merely trusting a saved status string is not verification.
