# Static support boundaries, reached branches and local events: a measured scope map with a real-arithmetic full-LP witness

**Anton Björkegren**

*22 September 2026 · revision 4.*

*A bounded, reproducible scope map with independent audits. Independent verification here means a separate implementation and separate raw sources, not external human peer review. The closest primary theory is cited in the source index.*

---

## Abstract

Contact-wrench feasibility for a rigid body on frictional contacts is a classical static question: does a contact-force vector inside the friction cones balance the body's weight and a prescribed inertial wrench? We separate that question from the branch a quasistatic solver reaches, from the first local contact event (gap closing, stick-to-slip, or unloading), and from the macro loss of support, and assemble a reproducible, independently audited map of when the static wrench-cone boundary is and is not the reached boundary, with counterexamples and exact scopes.

The static set uses an LP over a minimax 16-ray friction polygon that straddles the circular cone (balanced radial deviation `0.9701 %`); on the same discretized law it reproduces exhaustive subset enumeration to `≤ 1.5e-12`. Isotropic slow-ramp slip approaches the static radius (`1.31–1.46 %` at `T = 50 s`, `0.82 %` at `T = 100 s`; finite ramps do not prove the quasistatic limit); anisotropic scenes miss it by up to `7.26 %` and the compliant-stiff branch by `1.36–8.82 %`, through a history-dependent tangential residual. A reduced coplanar scalar is certifiable over a placement cell and all directions but is **not** a 6D wrench-LP certificate.

For the coplanar fixed-yaw double-support family the witness lemma is a **conditional** real-arithmetic argument: whenever its exact containment and capacity inequalities hold, the full 6D LP is feasible at the certified level. The reported safe regions and the smallest computed safe margin (`2.3e-5 m/s²`) are float-computed estimates over a fixed `w0` grid, not verified inequality premises over whole cells; an outward-rounded interval implementation was separately audited and its strict outward obligation was **not** established. First quasi-static **closes** are reproduced to `≤ 7e-6`; near-cap first slips and macro collapse are not. The contribution is a measured scope map with a conditional static lemma for one family, not a new theorem.

---

## 1. Introduction

### 1.1 Four different questions

Consider a rigid body in quasistatic contact with the environment through a set of point or patch contacts, each obeying Coulomb friction. At least four distinct questions arise, and the central methodological claim of this paper is that they must not be identified:

1. **Static wrench feasibility.** For a given candidate contact set and a prescribed horizontal load, does there exist a vector of contact forces inside the friction cones that balances the body's weight and the applied wrench? The set of feasible contact wrenches is a convex cone — the *contact wrench cone* (CWC) [S1]. At fixed weight, geometry and wrench map, the set of feasible *horizontal-load* vectors is an affine slice/projection of that cone, not the cone itself. Define the radial feasible load in unit direction `d` as
   `r(d) = sup { L ≥ 0 : the wrench (weight + L d) is feasible }`.
   This is a different object from the *support function* `max_x d·x` of a body or hull, and the two must not share notation. This question is a *possibility* question about a set.
2. **Reached quasistatic branch.** When a solver integrates a compliant (penalty/NCP) contact model under a slowly increasing load, which force vector does it *actually* settle on? This is a solver- and history-dependent *selection* from the feasible set; Coulomb contact problems are known to be non-unique [S3, S4], so the selected branch need not attain the static boundary.
3. **First local event.** Along the reached branch, where does the first local event occur — the first contact gap closing, the first stick-to-slip transition, or the first unloading? This is a *continuation* question about the branch.
4. **Macro support loss.** Where does the assembled support structure as a whole lose the ability to hold the load (tipping, collapse, loss of closed contacts)? This is a *global* question whose answer is, in general, a different load from the first local event.

The literature states the static CWC precisely [S1, S2], and the zero-moment point and support polygon describe the tipping boundary [S5]. The non-uniqueness of Coulomb contact and its time-stepping consequences are also known [S3, S4]. What is less often made explicit, in a single audited form, is *where the static boundary coincides with the reached physical boundary and where it does not*, and what can and cannot be certified for a whole cell of configurations and all generic load directions.

### 1.2 Prior art and what is not claimed as new

We do not claim novelty for:

- the contact wrench cone or its decomposition into Coulomb friction, centre-of-pressure/ZMP inside the support area, and yaw-torque bounds — this is the closed-form result of Caron, Pham & Nakamura [S1], and the ZMP/support-polygon boundary is classical [S5];
- penalty/compliant contact and "cap stiffness" — standard contact mechanics [S6];
- the implicit-function (material) tangent of a penalty equilibrium — standard differentiation;
- the non-uniqueness of Coulomb contact and its time-stepping treatment — [S3, S4];
- the statement that a static feasible set need not equal a solver-reached branch: this follows from the known CWC plus known contact non-uniqueness.

The auditable, independent contribution of this work is a **measured scope map**: a reproducible separation of the four quantities above, with explicit numerical counterexamples, exact scopes, independent audits, and — for the stated static coplanar fixed-yaw family — a constructive **conditional** real-arithmetic lemma whose exact inequalities imply full 6D wrench-LP feasibility over a whole placement cell and all directions, with float-computed region estimates (an outward interval verification was attempted but **not** established by independent audit).

### 1.3 Contributions

- **C1 (method parity).** A 16-ray minimax-polygon LP reproduces exhaustive subset enumeration of the *same discretized* wrench cone on frozen geometry to a maximum relative difference of `≤ 1.5e-12` (§3, T1, F1). The polygon straddles the circular cone with a balanced radial deviation of `0.9701 %`; it is a discretization choice, not an inner approximation, so C1 is not a statement about the continuous circular cone.
- **C2 (scope of the naive identification).** The reached slow-ramp slip boundary approaches the static radius on isotropic supports with a decreasing measured discrepancy at the tested finite rates (1.31–1.46 % at 50 s; 0.82 % at 100 s; these finite samples do not establish a limiting identity), and in the tested coplanar double-support stance the static radius agrees with the tipping/ZMP boundary at the tested placements within the reported numerical precision — but that stance result is audited only with reservations (§4.3) and is not a continuous-direction certificate. The identification **fails** anisotropically by up to 7.26 %, with the compliant-stiff branch 1.36–8.82 % below the static radius (§4, F2, F3).
- **C3 (four quantities kept distinct).** The four definitions above, with first-local-event and macro-collapse kept separate in every definition, table and figure; the frozen-normal macro bound is falsified 18/18 and min-norm/active-set certificates are falsified (§5, §6, F7).
- **C4 (certified reduced scalar and its hard limit).** A whole-cell, all-direction enclosure exists for the *reduced* scalar, but it is not a full-LP certificate (per-direction gap `6.69e-3`, inradius overestimate `8.85e-8`) (§7, F6).
- **C5 (conditional full 6D LP witness, real arithmetic).** For the static coplanar fixed-yaw double-support family, the witness lemma states a **conditional** real-arithmetic result: if its exact containment and capacity inequalities hold, then the full 6D LP is feasible at the certified level over the whole placement cell and all directions. The lemma is stated and proved in real arithmetic (§7.3, `WITNESS_LEMMA.md`): a normal-force floor `w0` on one contact pair, the remaining normal force distributed to realise the required centre of pressure, and one equal-and-opposite tangential pair whose moment about the centre of mass is the required `Δ`. The reported cell classifications and areas, the `4092` checked leaves and the witness residual `≤ 1.25e-12` are float-computed observations over a discrete `w0` grid; they are not a verified enclosure of the inequality premises over whole cells. **An independent audit of an outward-rounded interval implementation finds the strict outward obligation not established (`outward_rounding_verified = false`): the bound is valid for the tested geometries but rests on a fixed `1e-12` floor.** The float64-plus-fixed-guard version is reported separately and **not** accepted as a machine-checked certificate (§7.3).
- **C6 (local events, narrow scope).** A full residual Jacobian including geometric stiffness and the Coulomb cap reproduces an independent settle secant on accepted at-cap sticking states, and first quasi-static **closes** are confirmed; exact **path** derivatives and near-cap first slips are **not** (§6.2, §6.3).

---

## 2. Definitions and contracts

Throughout, unless stated otherwise, quantities are expressed in `m/s²` (an acceleration load level `L`), contacts are point contacts with Coulomb friction coefficient `μ`, and the drive is a prescribed horizontal acceleration of the base/CoM.

**Definition 2.1 (static wrench feasibility and radial load).** Given a frozen candidate contact set `C` with contact positions `p_i`, inward normals `n_i`, and friction coefficients `μ_i`, and a prescribed load wrench `F_ext(L; d)` of magnitude `L` in unit direction `d`, the load is *statically wrench-feasible* at level `L` if there exist contact forces `f_i ∈ K_i` (the friction cone at `i`) whose combined body wrench balances the body's weight and `F_ext`. The image of the product of the per-contact friction cones under the wrench map is the *contact wrench cone* `CWC = sum_i W_i K_i`, a cone in the six-dimensional wrench space (not the three-dimensional local force set `Minkowski sum_i K_i`); its intersection with the affine slice of feasible external wrenches is the feasible *load* set, whose radial limit is `r(d) = sup { L ≥ 0 : feasible }`. The radial limit `r(d)` is not the support function of the contact set. We compute `r(d)` with a linear program over the minimax-scaled 16-ray polygon (scale `2/(1+cos(π/16))`), which straddles the circular cone, and, for validation only, with exhaustive subset enumeration over the **same** polygon.

**Definition 2.2 (reached quasistatic branch).** A compliant (linear-spring) contact model with normal stiffness `k_n` and tangential stiffness `k_t = k_n(1-ν)/(2-ν)` (`ν = 0.3`) and a Coulomb cap is integrated under a prescribed load ramp of duration `T`. The *reached boundary* in a direction is the load level at which the simulated tangential velocity exceeds a fixed detection threshold (calibrated; `5e-3 m/s` unless stated). This is a property of the solver, the contact model, the ramp rate and the history — not of the static cone.

**Definition 2.3 (candidate and confirmed local events).** Along the accepted compliant equilibrium branch we distinguish three event notions. A **candidate event** is a load at which a Newton continuation predicts that (a) an initially open contact gap decreases to zero (an *open gap approaching close*), (b) a sticking contact's tangential force reaches the Coulomb cap, or (c) a contact's normal force reaches zero. A **confirmed branch event** is a candidate whose crossing is confirmed on an independently walked branch and which is transversal (the relevant scalar crosses with nonzero rate, so the event is locally isolated on the branch; transversality is local and does not by itself make all branch roots globally unique). A **measured ramp event** is the first event observed in an explicit dynamic ramp; it is a property of the ramp rate and is not, by itself, a branch event. A contact reaching the friction cap is not automatically a transition to sustained sliding: transversality and the path condition matter, as the grazes below demonstrate. Only *first* events are considered here.

**Definition 2.4 (macro support loss).** Macro support loss (tipping, collapse, or loss of enough closed contacts to carry the wrench) is the load level at which the assembled body-level support fails. It is measured, where available, by a fixed support-collapse detector and by the loss of closed contacts. It is a different quantity from Definition 2.3; the paper never infers one from the other.

**Definition 2.5 (reduced geometric scalar).** For coplanar contacts at one height with fixed foot yaw, one translated foot, and a drive at one height, define the reduced inradius
`r_red(u) = min( g·μ·POLY_SCALE·cos(π/K), (g/h)·max(0, s(u)) )`,
where `h` is the pelvis (load-point) height, `g` is gravity, `μ` the friction coefficient, `K = 16`, and
`s(u) = min_w ( σ_P(w) − w·c )` is the **minimum signed support margin** of the support hull `P` of the closed contacts about the centre-of-mass projection `c`: the minimum over unit directions `w` of the hull support function `σ_P(w) = max_{x∈P} w·x` minus `w·c`. The first term is the friction-limited inradius (the polygon inradius factor `cos(π/K)` makes it conservative for the polygon size), and the second is the tipping/ZMP-limited inradius. This scalar is the subject of §7; it is **not** the full 6D LP in general.

Assumptions that are carried by every number (§8): yaw is fixed in the reduced/full-LP family; geometry is frozen (the placement cell is sampled/enclosed, not optimized); history is the zero-history accepted branch; the friction law is Coulomb with a 16-ray minimax polygon that straddles the circular cone; the load-point lever is the pelvis height, not the CoM height (changing the convention moves the reduced scalar by `0.247 m/s²`); contacts are coplanar for §7; and static results carry no dynamics, joint-torque, compliance or biological terms.

---

## 3. Static feasibility and its numerical parity (known theory, audited)

### 3.1 Method

For a frozen candidate contact set we solve the discrete wrench-feasibility LP with the contact forces restricted to the 16-ray minimax polygon, and compare with exhaustive enumeration of all nonempty contact subsets (the Minkowski-sum characterisation), both assembled independently. The polygon scale is `κ = 2/(1+cos(π/16)) > 1`, so its circumradius `κ·μ` is **larger** than the circle radius `μ` and its inradius `κ·cos(π/16)·μ` is **smaller**; the polygon **straddles** the circular cone, and its radial deviation is balanced on both sides at `±(1−cos(π/16))/(1+cos(π/16)) = ±0.9701 %`. Consequently the polygon LP is neither an inner nor an outer approximation of the circular-cone problem, and it cannot be transferred to circular Coulomb without a separate argument. The enumeration uses the same polygon, so the comparison below tests the LP realization on one discretized law, not the cone discretization.

### 3.2 Results

The LP and the enumeration differ by at most the float64 round-off floor:

| scene | contacts | max relative difference LP vs enumeration |
|---|---|---|
| box | 4 | `2.908209251e-15` |
| stack | 8 | `3.072270253e-15` |
| pallet | 8 | `1.499866223e-12` |

These are empirical numerical-parity results on the same discretized law. The polygon's balanced radial deviation from the circular cone is `0.970055654 %` analytically and `0.9700556994879544 %` measured against a second-order-cone reference (box). For the `n_c = 12/20/30` scenes the LP over 72 directions costs `0.122/0.173/0.228 s` (CPU, shared machine), against `5.23 s` for the `2^12` enumeration and extrapolated `1.23e3/1.08e6 s` for the larger sets (the last two are extrapolations, not measurements). See T1 and F1.

This part is a reproducible implementation of the known CWC, whose value is the audited numerical parity and the removal of the exponential enumeration — not new mathematics. It is reported because the scope claims below depend on the static boundary being computed correctly. A self-contained re-assembly of the wrench map from the frozen contact geometry and an independent LP/enumeration reference are shipped in the release package (`scenes.py`, `lp_reference.py`, `verify_static_lp.py`).

---

## 4. Where the static radius is, and is not, the reached boundary

### 4.1 Isotropic supports: decreasing discrepancy under a slow ramp

Under a slowly ramped horizontal load on isotropic supports (on the four-contact box control the 24 sampled static LP radii agree to `6.1e-10` relative — the box support is friction-limited — so that ramp test is treated as a scalar test), the measured slip boundary approaches the static radius:

| scene | `T = 20 s` | `T = 50 s` | `T = 100 s` |
|---|---|---|---|
| isotropic N12 | — | 1.307 % | — |
| isotropic N20 | — | 1.385 % | — |
| box (S1) | 2.415 % | 1.459 % | 0.822 % |
| box/stack/pallet 3-scene mean (72 directions) | 2.731 % | — | — |

The N12/N20 rows have only the `T = 50 s` scalar run; there is no N12 or N20 `T = 20 s` run, hence the dashes. The `2.731 %` entry is the **three-scene mean** over box/stack/pallet, a separately labelled aggregate and not an N12 value; the box `T = 20 s` value is the box per-scene mean `2.415 %`. The measured discrepancy decreases with ramp duration (F3). At `T = 0.05/0.2/1/5/20 s` the three-scene means are `442.16/94.23/24.03/7.49/2.73 %`; the mismatch at short ramps is a **dynamic** effect of the finite ramp. These are finite-duration measurements; the `T = 50/100 s` values approach but do not prove the quasistatic limit. A widely quoted earlier fast-ramp figure (`23–28 %`) used `T ≈ 1 s` and a wrench denominator; at `T = 0.6 s` the same scene gives `29.49 %` under the wrench denominator and `22.96 %` under the original denominator, so the difference is a `T` and denominator convention, not a contradiction.

### 4.2 Anisotropic supports: the identification fails

On anisotropic support scenes the reached cold-ramp boundary falls **below** the static radius in weak directions:

| scene | max rel err | median rel err | max/min directional LP radius ratio |
|---|---|---|---|
| A | 7.264066 % | 5.696911 % | 1.705593 |
| B | 2.980434 % | 2.207335 % | 1.520508 |
| box (S1), 24 sampled LP radii | — | — | 1 (constant to 6.07e-10; friction-limited) |
| N12 platform (12 dirs, reached `T = 50 s`) | 1.230239 % | 1.230239 % | 1.019349 |
| N20 (exact 3D scalar LP) | 0.071274 % | — | not defined (not a 12-direction sample) |

Here the fourth column is the ratio of the **largest to the smallest directional LP radius** `max_k r(d_k)/min_k r(d_k)` within a scene, independently recomputed as max/min of that scene's `lp_radii` vector (an anisotropy measure of the same LP radii, not a reached ratio); the second and fourth columns are therefore different statistics of the same scene. Three distinct isotropic controls are shown and are **not** pooled: (i) the four-contact box scene (S1), whose 24 sampled static LP radii are constant to `6.07e-10` relative (friction-limited), which fixes a directional radius ratio of `1`; (ii) a genuine 12-contact elliptical 3-fold-perturbed ring platform, whose reached `T = 50 s` boundary is compared with its own LP radius (max/median relative error `1.230239 %`, its own directional LP ratio `1.019349`); and (iii) an N20 scene whose isotropic control value is an **exact 3D scalar LP** radius compared with its measured boundary (`0.071274 %`); the N20 control is not a 12-direction sample and no directional radius ratio is defined for it. No control's ratio is transferred to another; the previous revision's single "N12 sampled" row was the box sample, and is corrected here.

The mechanism is a **history-dependent tangential residual** in the reached branch, not a minimum-norm selection: the normal block of the cold-start solution is exactly minimum-norm, but the tangential part is locked by stuck anchors and does not relax to minimum-norm as `k_n → ∞`. The measured convergence slope is `-0.60`, not `-1`, with a floor of `5.48e-3` (relative `9.1e-5`) for `k_n ≥ 1e8`. Consequently the earlier hypothesis that the cold branch *is* the exact minimum-norm solution, and that the compliant-stiff limit converges as `1/k_n`, is **falsified** and reported as such (§5.2).

The physical compliant-stiffness branch (linear penalty, `k_t = k_n(1-ν)/(2-ν)`, `ν = 0.3`, Coulomb cap) lies at the cold branch (within `0.06–2.88 %` at `k_n = 1e6`) and **1.36–8.82 %** below the static LP radius (`T = 150 s` moves it further down). In this tested anisotropic family, therefore, the static radius is an **upper bound** on the reached boundary, not an identity (F2). This is the calibrated validity boundary of the naive identification.

### 4.3 A humanoid double support: static = tipping/ZMP (audited with reservations)

On a coplanar fixed-yaw double-support stance of a humanoid model, the static wrench radius matches the ZMP/tipping boundary at the tested placements to an absolute `4.7e-14 m/s²` (relative tip-vs-ZMP `6.7e-16`), with gliding far above (`μ·g` is at least `5.83×` the maximum tipping radius; max/min directional ratio `1.760533`). This is consistent with the classical result that the ZMP condition is part of the CWC.

An independent audit of the associated placement field reproduced the stored LP field to `9.0e-15 m/s²` and classified `374/441` placements as having static double support using an independent **exact geometric margin**, so the equality `LP = ZMP margin` holds at the tested placements. The audit also qualifies the result by naming its two references separately, following the audit's own definitions: the **exact geometric margin** is `r_cont = g·(minimum CoM-to-support-hull-edge distance)/pelvis_z`, whereas the field's own reference is a **12-direction sampled** radius. The largest sampled-versus-`r_cont` discrepancy over the feasible grid is `3.44e-2 m/s²` (`3.27 %` of the global maximum `R*`), and the 12-direction sampled reference itself deviates from the exact geometric margin by `3.29 %` of `R*`. Separate cost claims made for the field **fail** audit and are not used here. The stance equality is therefore reported as audited **with reservations**, not as an unqualified positive claim, and F4 is labelled accordingly.

---

## 5. Counterexamples kept as first-class results

The paper treats falsified equalities as results, not footnotes.

### 5.1 The active-set certificate is not the boundary

An early construction identified the stability boundary with a minimum over active sets of a per-set certificate. This is falsified: the certificate lies outside the measured bracket in `144/144` direction–scene pairs (mean relative error `0.198–2.979`), is negative in all 24 tested directions at `0.20 g` on one scene, and the minimum degenerates — two certificate definitions disagree by `0.976` vs `0.049` on the *same* shapes. The explanation is that a point-contact Delassus/`G` matrix is singular, so the certificate reduces to a minimum-norm selection, whereas the physically meaningful lumped contact-point quantity is unique.

### 5.2 Minimum-norm and `1/k_n` are falsified

Cold start is not minimum-norm (2/20 and 20/20 random restarts fall below the cold solution; the cold solution is `1.0 %` above exact minimum-norm, the LP `11.7 %`), and the compliant-stiff limit does not converge as `1/k_n` (slope `-0.60`, floor `5.48e-3`). The *normal* block is exactly minimum-norm; the deviation is the history-dependent tangential residual of §4.2.

### 5.3 The support rectangle alone is not the boundary

The support-rectangle polar radius alone has a mean relative error of `54.7/86.5/70.5 %` (box/stack/pallet), with `0/3` scenes below `10 %`. The full per-patch tip-and-slide geometry reduces this to `0.66/0.66/1.13 %`, and the mode agrees `24/24`. The rectangle is therefore not the boundary; the full geometry is.

### 5.4 Reduced scalar is not a full-LP certificate

See §7.2: the reduced scalar's whole-cell enclosure is valid, but the per-direction yaw-moment gap (`6.69e-3 m/s²`) and the inradius overestimate (`8.85e-8 m/s²`) refute promoting it to a full 6D LP certificate. `full_lp_certificate_supported = false`.

### 5.5 Frozen-normal macro bound and the fixed detector are falsified

Frozen observed normal reactions overestimate the measured macro holdout limit in `18/18` cases, from `+5.2 %` (`h = 0.10`, `0°`) to `+1462 %` (`h = 0.35`, `150°`); at the high cases the support falls to two closed contacts. The fixed velocity detector `v·d > 5e-3` does not prove history independence: a `1e-5` micro-creep threshold shifts the detected cycle by `-6.28 %` at `90°`, while a stiffness/dt change moves it by `+0.25 %`. These are negative results about the *macro* quantity and are kept visible.

---

## 6. Local events versus macro collapse

### 6.1 First-event prediction on a strict branch

A continuation of the accepted compliant equilibrium predicts the first close/lift/slip *identity* on a strict regular branch (no contact at the Coulomb cap) correctly (`type 10/10`, `contact 10/10`) with a load error of `5.4e-6` to `2.99 %` (median `0.51 %`), and event-driven re-solve predicts the second event to `0.36–3.35 %` in `6/6`. There is a genuine first-event miss: for one case at stance height 0.15 m and direction 0°, the candidate predictor reports a slip at `3.666` where the ramp first closes a different contact at `0.507` (`+623 %` off). This miss is reported, not suppressed.

### 6.2 The full tangent: what is exact and what is not

An independent re-derivation of the full nonlinear residual Jacobian — including contact geometry, geometric stiffness (`f_n dn_i` and `dZ_i × f_i`), the normal-load Coulomb cap term (`-μ_i û_i df_{n,i}`) and the stick/slip active-set law — matches a central-difference Jacobian to `≤ 2.5e-10`. Against an independent one-sided settle secant, the consistent tangent agrees to `≤ 4e-8` on accepted at-cap states where the contact **sticks** (the two height-control states at 0.30 m and 0.40 m and direction 15°), whereas the material (implicit-function) tangent errs by `1.26–1.68e-2`. Geometric stiffness explains essentially the whole discrepancy; cap following is secondary. **However**, this tangent is exact only under a *frozen* slip direction: the non-associated slip-direction/history term is not derived, and on one case (stance height 0.15 m, direction 45°) the tangent differs from the one-sided path secant by `1.5e-2`. The exact *path* derivative is therefore **not** claimed.

### 6.3 First events: closes confirmed, near-cap slips are grazes

Independent reconstruction confirms every predicted first **close** and the one genuine lift to `≤ 7e-6`: for stance height 0.15 m at directions 0° and 270° the predicted first closes are `0.4932093` and `0.1455178`, and the independently walked branch intervals are `[0.490, 0.495]` and `[0.145, 0.150]`; three additional held-out states give closes confirmed to `≤ 7e-6`. By contrast, every predicted first **slip** near the friction cap is a *nontransversal graze*: the branch ratio `|f_t|/(μ f_n)` reaches `0.9926` (0.30 m, 15°), `0.9996` (0.40 m, 15°), `0.9916` and `0.9766` (two further held-out states) without crossing, so there is no unique branch load and the candidate is not a confirmed branch event. The apparent `0.01 %` agreement of the predictor with the dynamic ramp for the 0.30 m and 0.40 m states is an agreement with a *rate-set dynamic transient*, not with a validated quasi-static branch event. The rate study at 0.15 m and 0° shows the dynamic first event moving `-1.386 %` on a halved ramp rate while the predictor targets the quasi-static branch close (`~0.4932`). The allowed claim is therefore: first quasi-static **closes** are predicted to `≤ ~0.001 %` on the confirmed subset; first **slips** are not.

### 6.4 Local event ≠ macro collapse

The first local event and macro collapse are different loads. For the 0.30 m, 15° stance, a rate-set dynamic ramp shows a first observed local slip at `0.6513`, a support collapse at `2.7824`, and a macro threshold at `2.985` (a factor `4.6×` above the first slip). The predicted first **branch** slip for that state does not cross the Coulomb cap (maximum ratio `0.9926`) and is an unconfirmed graze, so `0.6513` is a measured ramp event, not a confirmed branch event. Macro support loss is overestimated by both the static model `P0` and the frozen-normal model `P1` by `1.91–20.87×`; the continuation stops at support collapse with only two closed contacts. The dynamic first-event load is rate-stable to `≤ 0.011 %` in the tested ramp while the macro load moves `-3.1` to `-6.8 %` under dt/rate refinement. No local-to-macro inference is made anywhere in the paper (F7).

---

## 7. Regions: the reduced scalar and the conditional full-LP witness

### 7.1 The reduced scalar and its enclosure

For coplanar contacts at one height, fixed foot yaw, one translated foot and a drive at one height, the reduced inradius of Definition 2.5 is certified over a whole placement cell and all directions by a directional support-value interval plus a derived direction-cover remainder `(g/h)·R_max·ε` with the load factor `g/h = 9.624516 s⁻²` (at `M = 2880`). The remainder is geometry-dependent: for the three rendered reduced-scalar regions it is `0.004999 m/s²` (right foot +20°), `0.005095 m/s²` (left foot +60°) and `0.005020 m/s²` (right foot −70°); the single `0.004999` value is the right +20° geometry and is not a global bound. Against the global Lipschitz fallback `(g/h)·hs·√2 = 0.136111 m/s²` (cell half-diagonal, `hs = 0.01 m`); `0` bound violations were found. The bare product `R_max·ε = 5.19e-4 m` is not the remainder: `R_max` is a length cover and the lever `g/h` carries it into the acceleration load level. This is the *reduced* scalar.

### 7.2 The reduced scalar is not a full-LP certificate

Promoting the reduced scalar to a full 6D wrench-LP certificate is refuted. The per-direction yaw-moment gap reaches `6.6893387773934165e-3 m/s²`; the reduced inradius overestimates the full-LP inradius by up to `8.848317192833299e-8 m/s²`; and `24/25/23` rigorous-safe cells on the three geometries lie inside the yaw gap. The gap is real and placement-dependent: it vanishes to `~1e-16` when the horizontal pelvis offset is zeroed, and its mechanism is the yaw moment from that offset. An independent audit reproduces the earlier `8.848e-8` at the same placement (`8.848318e-8`) and finds a larger local maximum `2.365e-7` nearby, and a separate audit's own code finds per-direction gaps up to `1.697e-3` on another placement. **No universal per-cell model-error bound was established here**; the reduced scalar must not be used as a full-LP safety margin. The cost Pareto is also not one-sided: the adaptive depth-4 gate leaves less unresolved area than the primary gate above `~0.3 s`, so no one-sided "dominance" is claimed.

### 7.3 A conditional full 6D LP witness lemma, real arithmetic, with an attempted outward interval check

Instead of bounding the yaw gap, one can construct a **primal force witness** that establishes full 6D LP feasibility **whenever the lemma's exact inequalities hold**. The construction and its exact inequality are as follows (a self-contained statement is in the release `WITNESS_LEMMA.md`).

**Contract.** Eight coplanar point contacts on `z = 0`, four sole-box corners per foot; the movable foot is rotated by a fixed yaw about its own centre and then translated by `u`; the fixed foot does not move. Let `c` be the CoM projection, `h = pelvis_z` the load-point height, `g = 9.81`, `μ = 1`, `POLY_SCALE = 2/(1+cos(π/16))`, inradius factor `τ_f = POLY_SCALE·cos(π/16)`, `FRIC = g·τ_f`, `FACTOR = g/h`. The full 6D LP maximises `L` subject to `W f = f0 + L F d`, `f ≥ 0` on the rays.

**Witness.** Write `perp(v) = (−v_y, v_x)` for the counter-clockwise in-plane rotation. Let `(i,k)` run over all distinct contact pairs — the actual selection scans all `28` pairs, not only cross-foot pairs — and `w0 ∈ (0, 1/2)`. Define the pairwise centre-of-pressure polygon `E_ik = w0(p_i + p_k) + (1 − 2 w0)·conv{p_j : j ≠ i,k}`, whose generating vertices are the other six contacts only (the pair itself is excluded), and let `d_E` be the minimum signed support margin of `E_ik` about `c` (the least signed distance from `c` to the boundary of `E_ik`, positive inside). The necessary centre of pressure for load `L d` is `q = c + (h/g) L d`, which **depends on `L`**. Place `N_i = N_k = w0 m g` and distribute the remaining `(1 − 2 w0) m g` over the other contacts so that its centre of pressure is `qq = (q − w0(p_i + p_k))/(1 − 2 w0)`. This distribution exists exactly when `qq ∈ conv{p_j : j ≠ i,k}`, equivalently `q ∈ E_ik`. Because `q` depends on `L`, the condition is **not** `qq ∈ conv{…}` at one point: for a fixed `L` and **all** unit directions `d` it is the ball containment `B(c, (h/g)L) ⊆ E_ik`, equivalently `(h/g)L ≤ d_E`. The bare statement `d_E ≥ 0` is only the `L = 0` (pair-existence) condition and does not by itself admit a nonzero level; set the reduced tangential field `T⁰_j = −(N_j/g) L d` at every contact; and add one equal-and-opposite tangential pair `+β w_perp` at `i`, `−β w_perp` at `k`, with `w_perp = perp(p_k − p_i)/|p_i − p_k|` and `β = −Δ/|p_i − p_k|`, where `Δ = −m L (rp × d)_z` is the required contact yaw and `rp = pelvis_xy − c`. Both legs lie in `z = 0`, so the pair changes neither the net tangential force nor the roll/pitch moment; its moment about `c` is purely yaw and equals `−β |p_i − p_k| = Δ`, cancelling the drive yaw.

**Identities.** Force: `Σ_j N_j = m g` and `Σ_j T_j = −m L d`, and the pair cancels, so the total wrench force is `m g e_z − m L d`. Centre of pressure: by construction `Σ_j N_j (p_j − c) = m g (q − c)`, which balances the roll/pitch moment. Yaw: `T⁰` contributes zero yaw because `Σ_j N_j (p_j − c)` is parallel to `d`, and the pair contributes exactly `Δ = −β|p_i − p_k|`. Friction: if `|T_j| ≤ τ_f N_j` then the tangential force lies inside the circle of radius `τ_f N_j` inscribed in the 16-ray polygon, so the contact force is inside the polygon (and, when `τ_f < μ`, inside the original circular cone); this is therefore a *sufficient disk-inclusion condition*, not an equivalence to the original Coulomb cone. Since `|T_i| ≤ |T⁰_i| + |β| = w0 m L + m L |rp|/|p_i−p_k|` and `|T⁰_j| = (N_j/g) L ≤ τ_f N_j` for `L ≤ FRIC`, a sufficient condition is the **capacity inequality**
`L (w0 + |rp|/|p_i−p_k|) ≤ τ_f w0 g`
(which itself implies `L ≤ FRIC`).

**Whole cell and all directions.** Each movable-foot vertex translates by `δu = u − u_c`, `|δu| ≤ ρ = side/√2`, and each vertex `q_j = w0(p_i+p_k) + (1−2w0)p_j` moves by `γ_j δu`, `γ_j ∈ [0,1]`, so every support function changes by at most `ρ` and `d_E(u) ≥ d_E(u_c) − ρ`. The containment region is the ball `B(c, (h/g)L)`; once `B(c,(h/g)L) ⊆ E_ik(u_c;w0)` holds after eroding by `ρ`, it holds for every placement in the cell, with the pair distance bounded below by `|p_i−p_k|_c − ρ`. Here `ell_c = |p_i − p_k|` is the pair distance at the cell centre and `ell_lo = ell_c − ρ` is its lower bound over the cell (the erosion `ρ` is applied once to the support margin and once to the pair distance, never twice to the same primitive). For an **admissible candidate pair** `(i,k,w0)` — one with `ell_lo > 0` and `d_E(u_c) − ρ ≥ 0` — the level `min(FACTOR·(d_E − ρ), FRIC·w0/(w0 + |rp|/ell_lo))` is a rigorous lower bound on the full-LP radius for every placement in the cell and every direction; inadmissible candidates are discarded. The cell lower bound is the maximum over admissible candidates, and if there is **no** admissible candidate it is `0`, which is reported as **UNKNOWN/unresolved**, not as a negative certificate. A cell is labelled **safe** iff `L_cert > τ` — a sufficient classification rule for feasibility, not a claim that physical safety holds iff this inequality holds — **unsafe** iff the reduced *valid upper bound* is `≤ τ` (a sufficient rule that the all-direction radius does not exceed the threshold), and **unresolved** otherwise; equality at the threshold uses the `≤` convention. Here unsafe means failure of the strict margin above `τ`; it does not necessarily imply infeasibility at load `τ`. The classifications inherit the lemma's assumptions: a sufficient-not-necessary capacity condition untested near `L ≈ FRIC`, a discrete `12`-point `w0` grid, the exact real-rotation (ideal-trigonometry) model rather than the frozen float64 coordinates, `μ = 1` and the frozen 16-ray polygon.

**Numerical findings (float classifications, not a validated enclosure).** An independent audit reproduces the reported float classifications exactly — `right foot +20° 359.4375/69.8125/11.75`, `left foot +60° 406.625/23.25/11.125`, `right foot −70° 388.5625/41.3125/11.125` (safe/unsafe/unresolved cell areas, total 441 each) — with `0` false-safe leaves, a minimum computed estimate `(LP − certified level)` of `1.58e-2 m/s²`, and new frozen yaw holdouts (`−45°`, `+55°`) that yield nonempty full-LP regions (`0.9903`/`0.9894` of the reduced safe area). These are observed float classifications over the discrete `w0` grid, not a machine-checked enclosure of the lemma's inequality premises over whole cells; the audit's own implementation uses float64 plus a fixed `1e-9 m` guard and is explicitly **not** an outward-rounded interval certificate.

A second re-derivation addressed exactly that rounding gap with outward interval rounding on every primitive (contact geometry, polygon supports, inradius/erosion, norm, pair distance, `w0`, friction capacity, `g/h`, using `mpmath.iv` at 60 dps and `fractions.Fraction` for `g/h`). It recovered the same safe areas and an interval-versus-float64 gap of about `3.0e-12 m/s²` (largest over the locked cases; four orders below the smallest computed safe-margin estimate `2.304e-5 m/s²` at the left-foot +60° geometry), with no boundary cell flipping. **An independent audit does not support the outward claim: the strict outward obligation is not established.** That audit confirms the witness lemma and the ball-containment/erosion argument in real arithmetic, reproduces the four locked levels and all five safe regions exactly (0 false-safe / 0 false-unsafe, 0 label flips over 4092 leaves), and finds that the bound stays valid *for the tested geometries* only because a fixed `1e-12` Lipschitz floor dominates the interval-width error. But it shows the inradius primitive uses the *upper* interval endpoints `σ_hi`/`wc_hi`: for a lower bound on the signed margin `min_w(σ_P(w) − w·c)` one needs the lower endpoint on the support and the upper endpoint on the subtracted centre term (`σ_lo − wc_hi`), and a repair that merely swaps in `σ_lo`/`wc_lo` would not be a lower bound either, since it subtracts too little and can exceed the true margin (max wrong-direction effect `4.0e-15` over 240 high-precision samples), that the `g/h` factor uses nearest rounding without a downward `nextafter` (`1.09e-16` low for the frozen `h`), that `ρ = nextafter(side/√2, +inf)` is still `~9e-20` below exact, that `rest = 1−2w0` is treated as exact, and that the `1e-12` angular floor is a fixed guard rather than a derived interval bound. The outward obligation is therefore **not established as documented**: *the witness lemma holds in real arithmetic and the safe region is nonempty and empirically confirmed, but no machine-checked outward interval certificate exists for this family, and the float64-plus-guard proxy is likewise not a machine-checked certificate.* No status flag substitutes for the argument.

**What the certificate is not.** It covers only the static coplanar fixed-yaw double-support family (`μ = 1`, frozen 16-ray polygon), with a discrete 12-point `w0` grid (hence conservative), a capacity condition that is sufficient not necessary and untested near `L ≈ FRIC`, and no dynamics, joint-torque, compliance, history or biology. Its lemma implies *static feasibility over a cell* whenever the exact inequalities hold; it does not certify a dynamically reached boundary. The safe region is ~`1.3 %` smaller than the reduced region because it uses the inradius of the ball instead of a direction cover; all reported areas and margins are computed estimates, not validated enclosures. No speed advantage is claimed: the full-LP gate costs `~8.52 s` weighted on `right foot +20°` against `~0.17–0.23 s` for the reduced gate, and leaves `11.75` vs `6.9375` unresolved cell area; a direct-LP sweep of the same cells (`~441 s`) is sampling, not proof.

---

## 8. Assumptions, tolerances, rate controls and complete same-target cost

Every number carries the following scope:

- **Yaw/geometry/history/friction law.** The reduced and full-LP family uses fixed foot yaw, coplanar contacts at one height, one translated foot, a frozen 16-ray minimax friction polygon that straddles the circular cone (`μ = 1`), and zero-history accepted states. The load-point lever is the pelvis height.
- **Proof versus sample.** The full-LP witness lemma (§7.3) is an analytic argument with numerical verification (this is a *conditional* real-arithmetic result: it certifies a cell only if the containment and capacity inequalities hold exactly; the reported areas and the `2.304e-5 m/s²` margin are computed estimates without a validated enclosure); the method parity (§3) is a numerical parity result on one discretized law; the reached-boundary and event results (§4–§6) are bounded empirical measurements over the stated scenes and directions, not proofs. A finite direction sweep is never presented as a continuous certificate; finite ramps are never presented as the quasistatic limit.
- **Solver tolerances and censoring.** The velocity detector threshold is calibrated (`5e-3 m/s` unless stated); micro-creep thresholds shift detected macro cycles (§5.5). Event guards and active-set iteration counts were checked (no `maxit` hits on the audited states, no dead-zone contacts, no lightly-loaded closed contacts); three new batch states were correctly rejected for having no accepted equilibrium. Grazes are flagged, not turned into fabricated loads.
- **dt/rate controls.** Isotropic discrepancy is shown versus ramp duration `T`; the measured first-event load is rate-stable to `≤ 0.011 %` while macro loads move `-3.1` to `-6.8 %` under dt/rate refinement; the dynamic first close at 0.15 m, 0° moves `-1.386 %` on a halved rate. These are measurements over the tested refinements, not a proof of rate independence.
- **Complete same-target cost.** The reduced-scalar gate and the full-6D-LP gate answer different questions and are never compared as the same task. The full-LP density-sweep and boundary checks are sampling; the certificate is the analytic lemma plus its numerical gate; the attempted outward interval gate did not pass independent audit. All timings are CPU on a shared machine; GPU statements would carry a "shared GPU" caveat.
- **No LP speed ratio as a different-event acceleration.** The exponential-enumeration removal is a *method* statement about the same static query; it is not presented as a speedup for a different (e.g. dynamic or event) query.

---

## 9. Reproducibility and data availability

The figures and tables are reproduced from the frozen input data shipped in the release package, and the independent audits are cited in the text. `PROVENANCE.md` and `CLAIM_SOURCE_MAP.csv` map every retained numeric claim to a shipped raw field and generator, or declare it an archived reported value; the declared gaps are the audit-only numbers listed there (for example the compliant-stiff `1.36–8.82 %`, the support-rectangle `54.7/86.5/70.5 %`, the macro `18/18` and `+5.2…+1462 %`, the tangent/path errors, and the N12/N20 `T = 50 s` scalar reached-boundary percentages). The closest primary theory is Caron, Pham & Nakamura (2015) [S1]. Figures F1–F7 are rendered by a standalone script from public CSV payloads; the CSV payloads themselves are regenerated from frozen public JSON inputs by a second script. The release package is self-contained: a standalone renderer, a data generator, a table generator, the static scene/wrench-map module, an independent LP/enumeration reference, the witness lemma, a provenance map, and a public-name/reproduction self-check. It runs from a copied directory with no private workspace, absolute imports, symlinks or network:

```sh
python3 gen_data.py && python3 gen_tables.py && python3 render_figures.py
python3 verify_static_lp.py && python3 verify_regions.py && python3 check_release.py
```

Tables T1 and the reached-boundary, anisotropic, certified-region and event tables are shipped as machine-readable CSV with their regeneration script. `PROVENANCE.md` maps every retained numeric claim to its public raw field and generator, and lists the exact gaps where only an archived reported value (not a raw rerun) exists.

---

## 10. Limitations

- The full-LP certificate (§7.3) covers one static family with a conservative discrete `w0` grid and a sufficient (not necessary) capacity condition; it is not a dynamic or product result.
- The outward interval re-derivation was independently audited and its strict outward obligation was **not** established (`outward_rounding_verified = false`); the bound is valid for the tested geometries but rests on a fixed `1e-12` floor. The float64-plus-guard version is explicitly not accepted as a proof either.
- The method parity (§3) is a numerical result on the minimax polygon, which straddles the circular cone; it is not a statement about the continuous circular-cone boundary.
- The event results (§6) are bounded to the compliant penalty point-contact scenes, CPU float64, accepted zero-history states and a frozen slip direction; exact path derivatives and near-cap slips are out of reach, and macro support loss is unpredicted.
- The humanoid double-support result (§4.3) is audited only with reservations: the placement field reproduces to `9.0e-15` and `374/441`, but its 12-direction sampling and its discretized reference bound the achievable agreement, and the associated cost claim fails audit.
- No claim of a new general theorem is made; the contribution is a measured, independently audited boundary/counterexample scope map plus a real-arithmetic static certificate for a stated family.

---

## 11. Conclusion

The static wrench-cone boundary is reproducible to numerical parity on a stated discretization, and at the tested finite rates and directions the reached slow-ramp boundary approaches it; those finite samples do not establish a limiting identity. Anisotropic and compliant-stiff regimes reach *below* the static radius, explained by a history-dependent tangential residual, so the naive identification must be narrowed rather than asserted. Four quantities — static feasibility, reached branch, first local event, macro collapse — are distinct through definitions, tables and figures. For a static coplanar fixed-yaw family, the witness lemma is a *conditional* real-arithmetic result: its exact containment and capacity inequalities imply full 6D wrench-LP feasibility over a whole placement cell and all directions, and the reported cell areas and margins are float-computed estimates without a validated enclosure. The attempted outward interval check was independently audited and its strict outward obligation was **not** established, and the float64-plus-guard proxy is not accepted as a proof either. First quasi-static **closes** are predicted on a strict branch, but exact path derivatives and near-cap slips are not, and macro collapse remains a separate, unpredicted quantity. This is a boundary/counterexample scope map and a conditional static lemma, not a new general theorem.

---

## 12. Figures (release assets and captions)

The seven vector figures are shipped in the release package under `figures/` and are rendered from the public CSV payloads in `data/` by `render_figures.py`. The captions below are the public captions (`CAPTIONS.md`).

- **Figure 1 — Static LP method and polygon approximation** (`figures/f1_exactness.pdf`). (a) Maximum relative difference between the 16-ray contact-wrench LP and exhaustive subset enumeration over 72 directions in each of the box, stack, and pallet scenes; the logarithmic axis is dimensionless. (b) Maximum radial deviation of the minimax-scaled 16-ray polygon from the circular friction cone, in percent; the polygon straddles the circle, so this is a balanced deviation, not an inner approximation. The dashed line is the analytic maximum, 0.970055654 %. These are static method checks, not reached-motion errors.
- **Figure 2 — Static feasibility versus reached branch** (`figures/f2_reached.pdf`). Twelve sampled load directions in each anisotropic support. Blue circles give the static contact-wrench LP radius; red squares give the midpoint of the cold-ramp reached bracket, with its measured half-width. The largest absolute relative gaps are 7.264 % (A) and 2.980 % (B). Lines connect samples for reading only; they do not establish a continuous-direction bound. This compares the same frozen geometry within each panel and does not identify a universal physical selection rule.
- **Figure 3 — Ramp duration and dynamic excess load** (`figures/f3_ramp_rate.pdf`). Mean absolute relative error is `|L_ramp − r_static|/r_static`. Red circles average 72 directions across three planar scenes at each duration from 0.05 to 20 s. Blue squares are a separate, audited box-direction control at 50 and 100 s; they are not pooled with the 72-direction series. Both axes are logarithmic. The finite-duration ramp measures the load needed to accumulate the specified drift within the horizon, so the fast-ramp excess is a dynamic effect in these scenes.
- **Figure 4 — Humanoid double-support static boundary, audited with reservations** (`figures/f4_humanoid.pdf`). Static LP radii and geometric tipping/ZMP limits at 12 sampled directions for one fixed coplanar double-support stance. The two markers coincide within the reported numerical precision (absolute `4.7e-14 m/s²`; relative tip-vs-ZMP `6.7e-16`); connecting dashes do not certify intermediate directions. An independent audit of the associated placement field reproduces the stored static LP field to `9.0e-15 m/s²` and classifies 374/441 placements by an exact geometric margin (`r_cont = g·(min CoM-to-hull-edge distance)/pelvis_z`), so the equality holds at the tested placements, but the 12 directions are only a sample of the exact geometric margin (up to 3.27 % of the global maximum) and the field's own 12-direction sampled reference deviates from it by 3.29 % of the global maximum; the result is audited **with reservations**, not as an unqualified positive claim. Joint-moment limits and dynamics are excluded.
- **Figure 5 — Placement field, audited with reservations** (`figures/f5_placement.pdf`). The 21×21 moving-foot offset grid (0.02 m spacing). Color gives the weakest of 12 sampled static LP radii in m/s² at each placement. Hatching marks 67 placements with no static double support; these cells are a separate category, not measured zero radius. An independent audit reproduces the stored field to `9.0e-15 m/s²` and the 374/441 classification, but the 12-direction sampling (up to 3.27 % of the global maximum) and the field's own 12-direction sampled reference (which deviates from the exact geometric margin by 3.29 % of the global maximum) bound the achievable agreement, and the associated cost claim fails audit. It is a sampled static field, not a dynamic or full continuous-direction certificate.
- **Figure 6 — Reduced scalar and full-LP real-arithmetic witness** (`figures/f6_regions.pdf`). Placement-cell areas at threshold 0.05 m/s² for three fixed-yaw coplanar geometries: right foot +20°, left foot +60°, and right foot −70° (the moving foot is rotated about its centre before translation). Each geometry has total area 441 grid-cell equivalents. Solid green means safe **only by the method named on that row**; red crosshatching means unsafe; grey diagonal hatching means unresolved, not zero or safe. The reduced-scalar and full six-dimensional LP witness rows answer different questions. The witness lemma is a **conditional** real-arithmetic statement: its exact inequalities imply feasibility, and whole-cell erosion is supported in real arithmetic. The plotted regions and margins are float-computed estimates over a discrete `w0` grid, not a validated enclosure; the interval implementation is **not** outward verified and no machine-checked floating certificate is claimed.
- **Figure 7 — First local contact event versus macro outcome** (`figures/f7_events.pdf`). (a) Candidate loads from the compliant equilibrium continuation (filled green circles). The two contact-close candidates at stance height 0.15 m lie within the independently walked branch intervals (short black bars: 0.490–0.495 and 0.145–0.150 m/s²); open squares at 0.30 m and 0.40 m mark predicted slips that do not cross the Coulomb cap on the checked branch and are therefore unconfirmed grazes, not branch-event loads. (b) In a separate rate-set dynamic ramp at 0.30 m and direction 15°, the first observed local slip, support collapse, and macro threshold occur at distinct loads. The dynamic first slip is not a validated quasistatic branch slip. The first local event does not predict macro stability.

---

## Source index (closest primary theory and exact addition)

- **[S1]** Caron, Pham & Nakamura, *Stability of Surface Contacts for Humanoid Robots: Closed-Form Formulae of the Contact Wrench Cone for Rectangular Support Areas*, ICRA 2015 (arXiv:1501.04719). The CWC, including the yaw-torque condition. **Addition here:** the audited scope separation (§4–§7) and the real-arithmetic full-LP witness for the stated static family (§7.3). The CWC itself is theirs.
- **[S2]** Caron, *Computational Foundation for Planner-in-the-Loop Multi-Contact Whole-Body Control of Humanoid Robots*, PhD thesis, University of Tokyo, 2016 (ZMP and contact-wrench representations).
- **[S3]** Stewart & Trinkle, *An Implicit Time-Stepping Scheme for Rigid Body Dynamics with Inelastic Collisions and Coulomb Friction*, International Journal for Numerical Methods in Engineering 39(15):2673–2691, 1996. Complementarity time stepping; linearised cone.
- **[S4]** Preclik & Rüde, *Solution Existence and Non-Uniqueness of Coulomb Friction*, 2011. Non-uniqueness underlying the reached-branch separation.
- **[S5]** Vukobratović & Borovac, *Zero-Moment Point — Thirty Five Years of Its Life*, International Journal of Humanoid Robotics, 2004. ZMP/support boundary.
- **[S6]** Johnson, *Contact Mechanics*, Cambridge University Press, 1985. Penalty/cap contact mechanics (standard).
- **[S7]** Stewart & Trinkle, *An Implicit Time-Stepping Scheme for Rigid Body Dynamics with Coulomb Friction*, Proc. IEEE International Conference on Robotics and Automation (ICRA), 2000. Companion conference item to [S3]; the two titles/venues are distinct.

No numeric claim in this paper is taken from [S1]–[S7]; they are used only for attribution and novelty scoping. All numerical results refer to this project's experiments; the release distinguishes regenerated measurements from archived reported values whose underlying runs are unavailable.

---

## Appendix A. Supplementary audited values (not all required by the main text)

These values are in the source ledger with their source lines and hashes. They are listed so that every ledger quantity is traceable from the paper; some are supporting detail of a claim already stated in the main text.

| quantity | value |
|---|---|
| nominal `μ_min·g` for box/stack/pallet (continuous cone); computed polygon LP radius differs: box `4.8586`–`4.9526`, stack `2.4293`–`2.4763`, pallet `2.2130`–`3.9621` m/s² | `4.905`, `2.4525`, `3.924 m/s²` |
| support-seed misses (5/6 identical to cold) | `1.59–9.01 %` |
| LP-lambda seed reaches | median `0.746 %`, max `4.19 %` |
| scaling `α = 0.01/0.1/0.5/1.0` | `8.71/8.42/7.32/1.26 %` |
| cold vs exact min-norm; LP above | `1.0 %`; `11.7 %` |
| compliant `k_n = 1e6` vs cold; vs LP | within `0.06–2.88 %`; `1.36–7.93 %` below LP |
| compliant `T = 150 s` vs LP | `1.70–8.82 %` below LP |
| humanoid `max|tip−ZMP|/ZMP`; `μ·g` / max tipping radius | `6.7e-16`; `5.8294×` |
| humanoid ramp controls | `0.9805/1.0966/0.0237 %` |
| field reconstruction at internal sampling parameter `b = 0.04`, stored `3686/7056 ≈ 52.24 %` of cells; `0.04` is a sampling cap, **not** a 4 % budget; conditional on the reference field already being available and **not** a cost win (with the reference charged, the Pareto allocation costs `1.53×` the whole field) | Pareto max `0.6945 %`/RMS `0.1269 %`; uniform max `1.4959 %`/RMS `0.1927 %` |
| per-direction cf−LP; LP−cf; noncoplanar mismatch | `6.2790465650595095e-3`; `4.580669177300933e-12`; `8.368e-3..5.639e-2` |
| `R_max` (m, right +20°); `ε` (rad); cover remainder `(g/h)·R_max·ε` (m/s², `g/h = 9.624516`) per geometry right +20°/left +60°/right −70°; global fallback (m/s²) | `0.476191387006602`; `0.001090830782496456`; `0.004999399292698012` / `0.005094542880481823` / `0.00501990790985952`; `0.1361112120652152` |
| adaptive depth 4; primary | `4.1914 at 0.3149 s`; `6.9375 at 0.1726 s` |
| active-set mean rel err; per-scene | `0.198–2.979`; `0.617/0.198/2.979` |
| at-cap discrepancy; FD vs settle secant | `0.6–8.6 %`; `2e-10..1.2e-5` |
| micro-creep `1e-5` @90°; `5e-3`; stiffness/dt | `-6.28 %`; `-0.05 %`; `+0.25 %` |
| macro overestimate; first-event/macro rate | `1.91–20.87×`; `≤0.011 %/0.58 %` vs `-3.1..-6.8 %` |
| 0°/120° cold vs seed | `0.405119 %` vs `13.534951 %`; `0.101410 %` vs `92.777868 %` |
| certified full-LP safe fractions (right +20°/left +60°/right −70°/holdouts −45°/+55°) | `0.986788/0.990108/0.989495/0.990312/0.989415` |
| tangent-vs-secant; path gap; at-cap material error | `3–4e-8`; `1.505e-2`; `1.26–1.68e-2` |
