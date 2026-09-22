# Constructive full-LP primal witness (real arithmetic)

This note is self-contained and uses no external code. It states the witness
lemma for the coplanar fixed-yaw double-support family and the exact inequality
that makes it a valid lower bound on the full 6D wrench LP.

**Status.** The lemma is a **conditional** real-arithmetic statement: whenever
its exact containment and capacity inequalities hold, the full 6D LP is feasible
at the certified level. It was independently re-derived and its safe regions
reproduced by a separate audit. The reported regions, cell areas and the
smallest safe margin are float-computed observations over the discrete `w0`
grid; they are not a verified enclosure of the inequality premises over whole
cells. An attempted outward-rounded interval implementation does not establish
its strict outward obligation, and the float64 implementation with a fixed guard
is likewise **not** accepted as a machine-checked certificate. The release
therefore ships the lemma as a mathematical argument, not as a verified interval
certificate.

## Contract

Eight coplanar point contacts on `z = 0`, four sole-box corners per foot. The
movable foot is rotated by a fixed yaw about its own centre and then translated
by `u`; the fixed foot does not move. CoM projection `c = com_xy`; the load is
applied at the pelvis `(pelvis_xy, h)` with `h = pelvis_z`; `g = 9.81`,
`mu = 1`; the tangential directions use the frozen 16-ray polygon with
`POLY_SCALE = 2/(1+cos(pi/16))` and inradius factor `tau_f = POLY_SCALE*cos(pi/16)`.
Write `FRIC = g*tau_f` and `FACTOR = g/h`.

The full 6D LP maximises `L` subject to `W f = f0 + L F d`, `f >= 0` on the
rays, where `f0` is the weight-support wrench, `F d` is the prescribed load
wrench (`d` a unit horizontal direction), and the moment arm is `r_i = p_i - c`.

## Witness

Write `perp(v) = (-v_y, v_x)` for the counter-clockwise in-plane rotation. Let
`(i,k)` run over all distinct contact pairs — the actual selection scans all
`28` pairs, not only cross-foot pairs — and `w0 in (0, 1/2)`. Define the
pairwise centre-of-pressure polygon

> `E_ik = w0 (p_i + p_k) + (1 - 2 w0) conv{ p_j : j != i,k }`,

whose generating vertices are the other six contacts only (the pair itself is
excluded), and let `d_E` be the minimum signed support margin of `E_ik` about
`c` (the least signed distance from `c` to the boundary of `E_ik`, positive
inside). The necessary centre of pressure for load `L d` is
`q = c + (h/g) L d`. Place

* `N_i = N_k = w0 m g`, and distribute the remaining `(1-2 w0) m g` over the
  other contacts so that its centre of pressure is
  `qq = (q - w0 (p_i + p_k)) / (1 - 2 w0)`. This distribution exists exactly when
  `qq in conv{ p_j : j != i,k }`, equivalently `q in E_ik`. Because `q` depends on
  `L`, this is **not** a condition at one point: for a fixed `L` and all unit
  directions `d` it is the ball containment `B(c, (h/g)L) subset E_ik`,
  equivalently `(h/g)L <= d_E`. The bare `d_E >= 0` is only the `L = 0`
  (pair-existence) condition and does not admit a nonzero level by itself;
* the reduced tangential field `T^0_j = -(N_j/g) L d` at every contact;
* one equal-and-opposite tangential pair: `+beta wper` at `i`, `-beta wper` at
  `k`, with `wper = perp(p_k - p_i)/|p_i - p_k|` and
  `beta = -Delta/|p_i - p_k|`, where `Delta = -m L (rp x d)_z` is the required
  contact yaw and `rp = pelvis_xy - c`.

Both legs of the pair lie in `z = 0`, so the pair changes neither net tangential
force nor the roll/pitch moment; its moment about `c` is purely yaw and equals
`-beta |p_i - p_k| = Delta`, cancelling the drive yaw.

## Identities

* **Force.** `sum_j N_j = m g`, `sum_j T_j = -(m g/g) L d = -m L d`; the pair
  cancels, so the total wrench force is `m g e_z - m L d`.
* **Centre of pressure.** By construction `sum_j N_j (p_j - c) = m g (q - c)`,
  so the normals realise the centre of pressure `q` and the roll/pitch balance.
* **Yaw.** `T^0` contributes zero yaw because `sum_j N_j (p_j - c)` is parallel
  to `d`; the pair contributes exactly `Delta = -beta |p_i - p_k|`, cancelling
  the drive yaw.
* **Friction (containment).** `qq in conv{others}` implies all normal forces are
  nonnegative. If `|T_j| <= tau_f N_j` then the tangential force lies inside the
  circle of radius `tau_f N_j` inscribed in the 16-ray polygon, so the contact
  force is inside the polygon (and, when `tau_f < mu`, inside the original
  circular cone); this is a sufficient disk-inclusion condition, not an
  equivalence to the original Coulomb cone. Since
  `|T_i| <= |T^0_i| + |beta| = w0 m L + m L |rp|/|p_i-p_k|`
  and `|T^0_j| = (N_j/g) L <= tau_f N_j` for `L <= FRIC`, a sufficient condition is

  > **(capacity)** `L (w0 + |rp| / |p_i - p_k|) <= tau_f w0 g`.

  The inequality itself implies `L <= FRIC`.

## Whole cell and all directions

Each movable-foot vertex moves by the same translation `tau = u - u_c` with
`|tau| <= rho = side/sqrt(2)`. The vertices `q_j = w0(p_i+p_k) + (1-2w0)p_j`
each move by `gamma_j tau`, `gamma_j in [0,1]`, so every support function changes
by at most `rho` and `d_E(u) >= d_E(u_c) - rho`. The containment region is the
ball `B(c, (h/g)L)`; once `B(c,(h/g)L) subset E_ik(u_c;w0)` holds after eroding
by `rho`, it holds for every placement in the cell. Here `ell_c = |p_i-p_k|` is
the pair distance at the cell centre and `ell_lo = ell_c - rho` is its lower
bound over the cell (the erosion `rho` is applied once to the support margin and
once to the pair distance, never twice to the same primitive). Hence for each
cell, pair and `w0`,

* `r0 = d_E - rho`, `L_wall = FACTOR r0`;
* `ell_lo = ell_c - rho`, `L_cap = FRIC w0 / (w0 + |rp|/ell_lo)`;
* `L_cert = max over admissible candidates of min(L_wall, L_cap)`.

A candidate `(i,k,w0)` is **admissible** only if `ell_lo > 0` and
`d_E - rho >= 0`; inadmissible candidates are discarded. `L_cert` is a rigorous
lower bound on the full-LP radius for every placement in the cell and every
direction. If there is **no** admissible candidate, `L_cert = 0`, reported as
**UNKNOWN/unresolved**, not as a negative certificate. A cell is **safe** iff
`L_cert > tau` (a sufficient classification rule for feasibility, not a claim
that physical safety holds iff this inequality holds), **unsafe** iff the
reduced valid upper bound is `<= tau` (likewise a sufficient rule for
infeasibility), otherwise **unresolved**; equality at the threshold uses the
`<=` convention and is classified unsafe. The certified region is the union of
safe cells. Every classification carries the lemma's assumptions: a
sufficient-not-necessary capacity condition untested near `L ~ FRIC`, a discrete
12-point `w0` grid, the exact real-rotation (ideal-trigonometry) model rather
than the frozen float64 coordinates, `mu = 1` and the frozen 16-ray polygon.

## What this is not

The certificate is a *lower* bound (primal feasibility) and is conservative: a
discrete 12-point `w0` grid, a sufficient-not-necessary capacity condition, no
dynamics, joint torque, compliance, history or biology, and only the static
coplanar fixed-yaw family. It is not an outward-rounded interval certificate and
does not cover the exact `L = FRIC` regime.
