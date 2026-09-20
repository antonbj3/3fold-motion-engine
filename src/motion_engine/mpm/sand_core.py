"""MLS-MPM 3D core (M1) -- Warp, GPU, deterministic int64 fixed-point P2G.

H28 COPY.  This file is the pressure-dependent branch, a copy of the generic material-point core
(md5 d5e3737d9571c76b77cbb0c41163dea4) with the sand model extended; the
original is NOT touched.  Diff, in full:
  * N_PARAM 8 -> 14; new columns 8 P_BETA (dilatancy coefficient), 9..12
    P_H0..P_H3 (Klar et al. 2016 sec. 4.3.1 friction hardening), 13 P_HARD
    (0/1 switch).  Columns 0..7 unchanged.
  * new particle array ``qh`` (N,) f32: accumulated plastic deviatoric
    strain q of Klar 2016 eq. (30).  Zero-initialised, carried through g2p,
    included in ``state_bytes``.
  * ``plastic_return`` for MAT_SAND: (a) friction angle phi_F(q) =
    h0 + (h1 q - h3) exp(-h2 q) when P_HARD=1, alpha recomputed per particle
    per step; (b) non-associated Drucker-Prager flow with dilatancy
    coefficient beta: the plastic strain increment is dgamma*(n_hat + beta*I)
    instead of dgamma*n_hat, so
        dgamma = [||e_hat|| + alpha*kappa*tr(e)] / [1 + 3*alpha*beta*kappa],
        kappa  = (3 lam + 2 mu)/(2 mu),
        eps_new_i = e_i - (dgamma/||e_hat||)*h_i - dgamma*beta.
    beta = 0 reproduces the original expression exactly (same code path,
    same rounding); beta = alpha is associated flow.  The Klar sec. 4.3.2
    volume correction (the banked log-volume in ``Jp``) is unchanged.
  * ``set_material`` gains psi_deg / associated / hardening / h0..h3;
    ``add_particles`` already carried rho, which is the packing-density knob.
  * ``g2p_kernel`` takes and writes ``qh``.  Nothing else changed.

H29 COPY (this file).  the pressure-dependent branch (md5 d3d595965661043550913c3d61514785)
with the sand's ELASTICITY made pressure-dependent; h28 is not touched.  Diff:
  * N_PARAM 14 -> 24; new columns 14 P_PDEP (0/1), 15 P_G0, 16 P_PREF,
    17 P_NEXP, 18 P_PMIN, 19 P_PMAX, 20 P_PMODE, 21 P_FPIT, 22 P_CPHI,
    23 P_CG.  Columns 0..13 unchanged; P_PDEP = 0 runs H28's code path and
    H28's expressions verbatim.
  * Hardin-type elasticity:  G(p) = G0 (p/p_ref)^n,
    K(p) = 2 G(p)(1+nu)/(3(1-2nu)),  lam = K - 2G/3, evaluated PER PARTICLE
    PER STEP inside the constitutive evaluation (both the p2g stress and the
    return map).  p is the mean compressive stress -tr(sigma)/3 of the Hencky
    law, so it is coupled to the modulus that produces it; the coupling is
    closed in one of two ways (P_PMODE):
      "trial" (1, default): p is the EXACT fixed point of p = K(p) ec on the
          trial elastic volumetric compression ec = -tr(eps_trial),
              p = p_ref (K0 ec / p_ref)^{1/(1-n)},  K0 = K(p_ref),
          so no iteration is needed for the elastic part; P_FPIT extra passes
          then re-evaluate the moduli at the RETURNED pressure and redo the
          return map (a Picard fixed point on the plastic part as well).
      "lag" (0): p is the previous step's converged particle pressure, read
          from the new ``pp`` array (one step of lag, no iteration).
    Outside [p_min, p_max] the MODULUS is frozen (the response is linear in
    ec again).  p_max therefore bounds the wave speed, which is what the CFL
    time step is taken from (``wave_speed_at``); the yield surface is NOT
    capped.
  * Cohesionless yield at p -> 0 (free surface) is H28's apex projection,
    stated explicitly: any trial state with tr(eps)+Jp >= 0 (trial pressure
    <= 0) is projected onto the cone APEX, sigma = 0 -- a full tension
    cut-off, no tensile stress anywhere -- and the discarded expansion is
    banked in Jp (Klar sec. 4.3.2, unchanged) and restored on recompression.
    p_min keeps G finite there so the step size stays bounded.
  * Klar hardening and the dilatancy/beta flow rule of H28 are unchanged.
  * Compaction coupling (optional, off by default): ``evp`` accumulates the
    plastic volumetric strain (positive = dilation); P_CPHI shifts phi_F and
    P_CG scales G0 by the compaction -evp.
  * New particle arrays ``pp`` (pressure) and ``evp``; both carried through
    p2g/g2p/energy and appended to ``state_bytes``.  ``state_bytes_h28``
    reproduces H28's layout, ``state_bytes_m1`` M1's.
  * Back-compat wp.func overloads kept for ``plastic_return`` (4- and
    5-argument), ``particle_pk1`` and ``particle_energy`` (4-argument), so
    the implicit stepper still resolves against this core.


Method: MLS-MPM (Hu et al. 2018) with APIC transfer, quadratic B-spline
weights, explicit symplectic Euler.  3D only (a thin domain with slip walls in
one axis is the "2D mode" used by the sand tests).

--------------------------------------------------------------------------
DATA LAYOUT (contract -- M2 contact and M3 cloth build against these names)
--------------------------------------------------------------------------
Particles (``MPMSolver.<name>``, all Warp device arrays of length N):

    x     wp.array(dtype=wp.vec3)     (N,3) f32   position            [m]
    v     wp.array(dtype=wp.vec3)     (N,3) f32   velocity            [m/s]
    F     wp.array(dtype=wp.mat33)    (N,3,3) f32 elastic deformation gradient
    C     wp.array(dtype=wp.mat33)    (N,3,3) f32 APIC affine velocity matrix
    mass  wp.array(dtype=wp.float32)  (N,)        particle mass       [kg]
    vol0  wp.array(dtype=wp.float32)  (N,)        initial volume      [m^3]
    mat   wp.array(dtype=wp.int32)    (N,)        material / model id
    Jp    wp.array(dtype=wp.float32)  (N,)        plastic state
                                                  (snow: plastic volume J_p;
                                                   sand: banked log-volume
                                                   correction, Klar 2016 4.3.2)
    q     wp.array(dtype=wp.vec4)     (N,4)       RESERVED for M3 (codim frame)
    d     wp.array(dtype=wp.mat33)    (N,3,3)     RESERVED for M3 (codim dirs)

Grid (dense, ``res**3`` nodes, flat index ``gi = (ix*res + iy)*res + iz``):

    grid_m   wp.array(dtype=wp.float32) (res**3,)   node mass       [kg]
    grid_mv  wp.array(dtype=wp.vec3)    (res**3,3)  node momentum   [kg m/s]
    grid_v   wp.array(dtype=wp.vec3)    (res**3,3)  node velocity   [m/s]

``grid_m``/``grid_mv`` are valid after ``p2g()``.  ``grid_v`` is valid after
``grid_update()`` and is the array that contact solvers modify.
Node world position: ``origin + dx * (ix, iy, iz)``.

Material / model ids (``mat``), parameters in ``MPMSolver.params`` (4,8) f32:

    0 MAT_COROTATED   fixed-corotated (E, nu)
    1 MAT_SAND        Drucker-Prager, Klar et al. 2016 (friction angle phi,
                      cohesion 0), Hencky elasticity
    2 MAT_SNOW        Stomakhin et al. 2013 (theta_c, theta_s, xi)
    3 MAT_NEOHOOKEAN  Neo-Hookean (E, nu)

Parameter columns: 0 mu, 1 lambda, 2 alpha (DP), 3 theta_c, 4 theta_s, 5 xi,
6 E, 7 nu, 8 beta (DP dilatancy), 9..12 h0..h3 (friction hardening), 13 hard
on/off.  Set them with ``set_material()``.

--------------------------------------------------------------------------
DETERMINISM -- int64 fixed-point P2G scatter
--------------------------------------------------------------------------
Floating-point atomics are order-dependent, so two runs of the same scene give
different bytes.  The default scatter path (``scatter="int64"``) converts every
particle->node contribution to a fixed-point integer and accumulates it with
``wp.atomic_add`` on int64 arrays; integer addition is associative and
commutative, so the result does not depend on thread order.

    mass      scale  MASS_SCALE = 1e11       (1 count = 1e-11 kg)
    momentum  scale  MOM_SCALE  = 1e14       (1 count = 1e-14 kg m/s)
                                              LITERAL, not 1e6 * MASS_SCALE
    energy    scale  ENERGY_SCALE = 1e9      (diagnostics only)

Conversion is ``int64(round(float64(value) * SCALE))`` (round-half-away-from-
zero), done per particle-node pair, in float64.  int64 range +-9.223e18 gives
headroom to node mass 9.2e7 kg and node momentum 9.2e4 kg m/s; ``p2g()``
raises above half of that (``check_overflow=True``), i.e. at node mass
4.6e7 kg / node momentum 4.6e4 kg m/s.

The mass scale was 1e8 until the resting-block defect of the resting-block measurement: a
single particle->node contribution is ``w * m_p`` with w down to ~1e-3, so at
dx = 2.2 mm, rho = 1000, 8 ppc (m_p = 1.3e-6 kg) the smallest contributions
rounded to **zero** -- 2 185 live nodes reported mass 0 while momentum, kept
at the far finer MOM_SCALE, survived; the node velocity was then zeroed, the
momentum silently discarded, and a stress-free block at rest diverged to
4.6 m/s in 2 500 steps.  1e11 puts the quantum at 1e-11 kg, 1e5 contributions
below the smallest one that occurs there, and MOM_SCALE is kept at its literal
1e14 so the momentum headroom (9.2e4 kg m/s) is unchanged.  ``scatter="float32"``
uses plain float32 atomics and is provided only for the throughput comparison;
it is NOT bit-reproducible.

--------------------------------------------------------------------------
CONTACT HOOK (contract for M2 -- mpm_contact.py)
--------------------------------------------------------------------------
``MPMSolver.grid_update(contacts=None, dt=None)`` runs, in this order:

    1. node velocity        grid_v = grid_mv / grid_m   (grid_m > 0, else 0)
    2. gravity              grid_v += dt * gravity
    3. CONTACT HOOK         <-- rigid-body / external node constraints here
    4. domain boundary      sticky / slip / separate per face

The hook argument may be:
    * ``None``                        -- no hook
    * a callable                      -- ``contacts(solver, dt)``
    * an object with ``apply_grid``   -- ``contacts.apply_grid(solver, dt)``
    * a list/tuple of the above       -- applied in order

Canonical signature:

    def apply_grid(solver: MPMSolver, dt: float) -> None

The hook must modify ``solver.grid_v`` **in place** (launch a Warp kernel over
``solver.n_nodes``).  Everything it needs is on the solver:

    solver.grid_v     wp.array(dtype=wp.vec3), shape (res**3,)   -- read/write
    solver.grid_m     wp.array(dtype=wp.float32), shape (res**3,) -- read only
                      (node mass = Delassus diagonal for a node-level NCP)
    solver.grid_mv    wp.array(dtype=wp.vec3)   -- momentum before step 1
    solver.res, solver.dx, solver.inv_dx, solver.origin (wp.vec3),
    solver.n_nodes, solver.device, solver.frame, solver.dt

A hook that wants bit-reproducible impulse accumulation should use the same
int64 fixed-point trick; ``MOM_SCALE`` is exported for that purpose.
Node mass and velocity are already final when the hook runs, so a hook can
compute an impulse ``m_i * dv_i`` and sum it back onto a rigid body.

--------------------------------------------------------------------------
Not imported from, not copied from, Newton's implicit_mpm.

AUDITED ANCHORS AND LIMITS (tests/test_mpm_sand.py, data/mpm/*.json).
  Holds: triaxial compression returns the input friction angle. phi = 40 deg comes back
  as 40.49 / 40.35 / 40.38 / 40.22 deg at grid resolution 32 / 64 / 96 / 128 with
  dt = 10 us, maximum error +0.49 deg against a 2 deg gate, and halving dt again moves
  it by at most 0.02 deg. In triaxial compression the Drucker-Prager cone is the
  Mohr-Coulomb one exactly, M = 6 sin phi / (3 - sin phi).
  Falls: the earlier claim that shear-band localisation made the angle grid-dependent.
  The fall at resolution 96 (38.53 deg) and the blow-up at 128 were the time step
  exceeding CFL, not the grid. Rule for every run on this core: dt <= 10 us at
  resolution >= 96. The 4.5 % band-width spread offered as evidence is not
  threshold-independent -- doubling the threshold gives 336 %.
  Non-local regularisation is built and is bit-identical to the local model at ell = 0,
  but at resolution 128 with ell = 12.5 mm it drops the angle to 31.2 deg and it leaves
  the repose plateau (28.73 against 28.9 deg) and the bearing capacity (0.72x Terzaghi)
  where they were. Both are material-model defects, not numerics.
  Bearing capacity is wrong from three directions and the numbers are in
  data/mpm/sand_bearing_reference.json: crater prefactor 2.63x published on a soft bed
  and 0.25x on a stiff one, strip footing 4.8 to 9.3x Vesic local failure on the stiff
  bed and 0.72x Terzaghi on the soft one, and a discrete-element comparison that falls
  from 1.07 to 0.42 of Terzaghi as the grains are refined -- towards this model's side,
  not away from it. Loose sand fails locally, so the reference is Vesic local failure;
  the two carriers still do not meet within 20 %.
  Parameter-gated branches (rate dependence, cubic spline transfer, tension cut-off,
  elastic-return initialisation) are off by default and carry no anchor of their own.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Sequence

import numpy as np
import warp as wp

# Quiet by default so scene drivers can emit clean JSON on stdout.
if not os.environ.get("MPM_WARP_VERBOSE"):
    wp.config.quiet = True

# --------------------------------------------------------------------------
# fixed-point scales (see module docstring)
# --------------------------------------------------------------------------
MASS_SCALE = 1.0e11
MOM_SCALE = 1.0e14  # literal, NOT 1e6 * MASS_SCALE (see the docstring)
ENERGY_SCALE = 1.0e9

INT64_MAX = 9.223372036854775807e18

# Guard thresholds: half of the int64 range, so a single accumulation cannot
# wrap even if the absolute maxima above are reached simultaneously.
MAX_NODE_MASS = 0.5 * INT64_MAX / MASS_SCALE          # 4.61e7 kg
MAX_NODE_MOMENTUM = 0.5 * INT64_MAX / MOM_SCALE       # 4.61e4 kg m/s

_MASS_SCALE = wp.constant(wp.float64(MASS_SCALE))
_MOM_SCALE = wp.constant(wp.float64(MOM_SCALE))
_ENERGY_SCALE = wp.constant(wp.float64(ENERGY_SCALE))

# --------------------------------------------------------------------------
# material ids
# --------------------------------------------------------------------------
MAT_COROTATED = 0
MAT_SAND = 1
MAT_SNOW = 2
MAT_NEOHOOKEAN = 3
N_MAT = 4

# parameter columns
P_MU = 0
P_LAM = 1
P_ALPHA = 2
P_THETA_C = 3
P_THETA_S = 4
P_XI = 5
P_E = 6
P_NU = 7
# --- H28 additions ---
P_BETA = 8      # dilatancy coefficient, sqrt(2/3)*2 sin(psi)/(3 - sin(psi))
P_H0 = 9        # Klar 2016 sec 4.3.1 hardening: phi_F(q) = h0 + (h1 q - h3) e^{-h2 q}
P_H1 = 10
P_H2 = 11
P_H3 = 12
P_HARD = 13     # 0 = fixed phi (original), 1 = hardening on
# --- H29 additions: pressure-dependent (Hardin) elasticity ---------------
P_PDEP = 14     # 0 = fixed moduli (H28 behaviour), 1 = G = G0 (p/p_ref)^n
P_G0 = 15       # shear modulus at p = p_ref [Pa]
P_PREF = 16     # reference pressure [Pa]
P_NEXP = 17     # exponent n (0.5 = Hardin/Hertz)
P_PMIN = 18     # pressure floor for the modulus [Pa] (free surface)
P_PMAX = 19     # pressure cap for the modulus [Pa] (bounds the CFL wave speed)
P_PMODE = 20    # 0 = lagged p (previous step), 1 = trial-strain closed form
P_FPIT = 21     # extra outer fixed-point passes on the RETURNED pressure
P_CPHI = 22     # compaction coupling: phi += CPHI * ev_compaction [deg]
P_CG = 23       # compaction coupling: G0 *= max(1 + CG*(-evp), 0.05)
# U10: free-surface candidates. Columns 0..23 unchanged; all default to
# behaviour identical to h29 (coh=0, pfloor=0, flip=0).
P_COH = 24      # cohesion: sustained isotropic tension [Pa]; apex at tr=coh/K
P_PFLOOR = 25   # floor on the pressure used for the moduli [Pa]; 0 = off
P_FLIP = 26     # g2p blend: 0 = APIC (legacy), 1 = full FLIP
P_DPMATCH = 27  # 0 = TXC (Klar default), 1 = TXE (extension), 2 = Plane strain, 3 = shear, 4 = Lode/MC
P_PHI = 28      # base friction angle [deg]
P_MUI = 29
P_GRAIN = 30
P_SPLINE = 31  # 0 quadratic, 1 cubic; APIC moment dx^2/4 or dx^2/3
P_TIP = 32  # floor in Cauchy yield pressure; constant-modulus DP only
N_PARAM = 33

# boundary condition modes
BC_STICKY = 0
BC_SLIP = 1
BC_SEPARATE = 2
_BC_NAMES = {"sticky": BC_STICKY, "slip": BC_SLIP, "separate": BC_SEPARATE}
# face order: 0:-x 1:+x 2:-y 3:+y 4:-z 5:+z


# ==========================================================================
# Warp device functions -- constitutive models
# ==========================================================================
_DP_C = 1.6329931618554521  # sqrt(2/3) * 2


@wp.func
def _dp_alpha(phi_deg: float) -> float:
    """Drucker-Prager cone coefficient from a friction angle in degrees (TXC default)."""
    s = wp.sin(phi_deg * 0.017453292519943295)
    return _DP_C * s / (3.0 - s)


@wp.func
def _dp_alpha_matched(phi_deg: float, match_mode: int) -> float:
    """Drucker-Prager cone coefficient matched to Mohr-Coulomb under different Lode states."""
    s = wp.sin(phi_deg * 0.017453292519943295)
    if match_mode == 1:
        # Triaxial Extension (TXE): inner inscribed cone
        return _DP_C * s / (3.0 + s)
    elif match_mode == 2:
        # Plane strain match: sqrt(2)*tan(phi)/sqrt(9 + 12*tan^2(phi))
        t = wp.tan(phi_deg * 0.017453292519943295)
        return 1.41421356 * t / wp.sqrt(9.0 + 12.0 * t * t)
    elif match_mode == 3:
        # Simple shear
        t = wp.tan(phi_deg * 0.017453292519943295)
        return 0.81649658 * t
    elif match_mode == 5:
        # Scale 1.10
        return 1.10 * _DP_C * s / (3.0 - s)
    elif match_mode == 6:
        # Scale 1.20
        return 1.20 * _DP_C * s / (3.0 - s)
    elif match_mode == 7:
        # Scale 1.30
        return 1.30 * _DP_C * s / (3.0 - s)
    elif match_mode == 8:
        # Scale 1.40
        return 1.40 * _DP_C * s / (3.0 - s)
    else:
        # Triaxial Compression (TXC, Klar default): outer circumscribed cone
        return _DP_C * s / (3.0 - s)


@wp.func
def _safe_log(a: float) -> float:
    return wp.log(wp.max(a, 1.0e-8))


# --------------------------------------------------------------------------
# H29: Hardin-type pressure-dependent elasticity
#
#   G(p) = G0 (p/p_ref)^n ,  K(p) = 2 G(p) (1+nu) / (3 (1-2 nu))
#
# The elastic law is the same Hencky law as before, but with (mu, lam) rebuilt
# per particle per step from the CURRENT pressure.  p itself comes out of the
# law (p = -tr(tau)/3 = K(p) * ec, ec = -tr(eps) the elastic volumetric
# compression), so the two are coupled.  ``_hardin_p_closed`` is the EXACT
# fixed point of that coupling:
#
#   p = K0 (p/p_ref)^n ec   =>   p = p_ref (K0 ec / p_ref)^{1/(1-n)}
#
# with K0 = K(p_ref).  Outside [p_min, p_max] the modulus is frozen and the
# response is linear in ec again (p then grows linearly, only the STIFFNESS is
# capped -- the yield surface alpha*tr(sigma) is not).
# --------------------------------------------------------------------------
@wp.func
def _hardin_g_of_p(p: float, g0: float, pref: float, n: float,
                   pmin: float, pmax: float) -> float:
    pe = wp.clamp(p, pmin, pmax)
    return g0 * wp.pow(pe / pref, n)


@wp.func
def _hardin_p_closed(ec: float, g0: float, pref: float, n: float,
                     nu: float) -> float:
    """Exact fixed point of p = K(p) * ec.  ec <= 0 -> 0 (tension side)."""
    if ec <= 0.0:
        return 0.0
    k0 = 2.0 * g0 * (1.0 + nu) / (3.0 * (1.0 - 2.0 * nu))
    return pref * wp.pow(k0 * ec / pref, 1.0 / (1.0 - n))


@wp.func
def _hardin_mu_lam(g: float, nu: float):
    """(mu, lam) from a shear modulus and Poisson ratio."""
    k = 2.0 * g * (1.0 + nu) / (3.0 * (1.0 - 2.0 * nu))
    return g, k - 2.0 * g / 3.0


@wp.func
def _sand_moduli(F: wp.mat33, pprev: float, evp: float,
                 params: wp.array2d(dtype=wp.float32), m: int, Jp: float):
    """Pressure-dependent (mu, lam, p_used) for the sand at deformation F.

    P_PMODE = 1: p from the trial elastic volumetric strain, closed form.
    P_PMODE = 0: p lagged -- last step's converged particle pressure.
    """
    nu = params[m, P_NU]
    g0 = params[m, P_G0]
    if params[m, P_CG] != 0.0:
        g0 = g0 * wp.max(1.0 + params[m, P_CG] * wp.max(-evp, 0.0), 0.05)
    pref = params[m, P_PREF]
    n = params[m, P_NEXP]
    pmin = params[m, P_PMIN]
    pmax = params[m, P_PMAX]
    p = pprev
    if params[m, P_PMODE] > 0.5:
        U = wp.mat33()
        sig = wp.vec3()
        V = wp.mat33()
        U, sig, V = wp.svd3(F)
        tr = _safe_log(wp.abs(sig[0])) + _safe_log(wp.abs(sig[1])) \
            + _safe_log(wp.abs(sig[2])) + Jp
        p = _hardin_p_closed(-tr, g0, pref, n, nu)
    g = _hardin_g_of_p(p, g0, pref, n, pmin, pmax)
    mu, lam = _hardin_mu_lam(g, nu)
    return mu, lam, wp.clamp(p, pmin, pmax)


@wp.func
def pk1_corotated(F: wp.mat33, mu: float, lam: float) -> wp.mat33:
    """First Piola-Kirchhoff of the fixed-corotated model (Stomakhin 2012)."""
    U = wp.mat33()
    sig = wp.vec3()
    V = wp.mat33()
    U, sig, V = wp.svd3(F)
    R = U * wp.transpose(V)
    J = sig[0] * sig[1] * sig[2]
    # J * F^-T  == cofactor(F), built from the SVD to stay well defined
    cof = U * wp.mat33(
        sig[1] * sig[2], 0.0, 0.0,
        0.0, sig[0] * sig[2], 0.0,
        0.0, 0.0, sig[0] * sig[1],
    ) * wp.transpose(V)
    return 2.0 * mu * (F - R) + lam * (J - 1.0) * cof


@wp.func
def energy_corotated(F: wp.mat33, mu: float, lam: float) -> float:
    U = wp.mat33()
    sig = wp.vec3()
    V = wp.mat33()
    U, sig, V = wp.svd3(F)
    J = sig[0] * sig[1] * sig[2]
    e = 0.0
    for i in range(3):
        e += (sig[i] - 1.0) * (sig[i] - 1.0)
    return mu * e + 0.5 * lam * (J - 1.0) * (J - 1.0)


@wp.func
def pk1_neohookean(F: wp.mat33, mu: float, lam: float) -> wp.mat33:
    U = wp.mat33()
    sig = wp.vec3()
    V = wp.mat33()
    U, sig, V = wp.svd3(F)
    J = sig[0] * sig[1] * sig[2]
    Finv_T = U * wp.mat33(
        1.0 / wp.max(sig[0], 1.0e-6), 0.0, 0.0,
        0.0, 1.0 / wp.max(sig[1], 1.0e-6), 0.0,
        0.0, 0.0, 1.0 / wp.max(sig[2], 1.0e-6),
    ) * wp.transpose(V)
    return mu * (F - Finv_T) + lam * _safe_log(J) * Finv_T


@wp.func
def energy_neohookean(F: wp.mat33, mu: float, lam: float) -> float:
    U = wp.mat33()
    sig = wp.vec3()
    V = wp.mat33()
    U, sig, V = wp.svd3(F)
    J = sig[0] * sig[1] * sig[2]
    I1 = sig[0] * sig[0] + sig[1] * sig[1] + sig[2] * sig[2]
    lj = _safe_log(J)
    return 0.5 * mu * (I1 - 3.0) - mu * lj + 0.5 * lam * lj * lj


@wp.func
def pk1_hencky(F: wp.mat33, mu: float, lam: float) -> wp.mat33:
    """PK1 of the Hencky (log-strain) St.Venant-Kirchhoff energy used for sand.

    Psi = mu ||eps||^2 + lam/2 tr(eps)^2,  eps = log(sigma).
    P = U diag( (2 mu eps_i + lam tr(eps)) / sigma_i ) V^T
    """
    U = wp.mat33()
    sig = wp.vec3()
    V = wp.mat33()
    U, sig, V = wp.svd3(F)
    e0 = _safe_log(sig[0])
    e1 = _safe_log(sig[1])
    e2 = _safe_log(sig[2])
    tr = e0 + e1 + e2
    t0 = (2.0 * mu * e0 + lam * tr) / wp.max(sig[0], 1.0e-6)
    t1 = (2.0 * mu * e1 + lam * tr) / wp.max(sig[1], 1.0e-6)
    t2 = (2.0 * mu * e2 + lam * tr) / wp.max(sig[2], 1.0e-6)
    return U * wp.mat33(t0, 0.0, 0.0, 0.0, t1, 0.0, 0.0, 0.0, t2) * wp.transpose(V)


@wp.func
def energy_hencky(F: wp.mat33, mu: float, lam: float) -> float:
    U = wp.mat33()
    sig = wp.vec3()
    V = wp.mat33()
    U, sig, V = wp.svd3(F)
    e0 = _safe_log(sig[0])
    e1 = _safe_log(sig[1])
    e2 = _safe_log(sig[2])
    tr = e0 + e1 + e2
    return mu * (e0 * e0 + e1 * e1 + e2 * e2) + 0.5 * lam * tr * tr


@wp.func
def pk1_hencky_pdep(F: wp.mat33, Jp: float, pprev: float, evp: float,
                    params: wp.array2d(dtype=wp.float32), m: int) -> wp.mat33:
    """H29: the same Hencky PK1 as ``pk1_hencky``, one SVD, with (mu, lam)
    rebuilt from the pressure the strain itself implies (P_PMODE = 1) or from
    the previous step's particle pressure (P_PMODE = 0)."""
    U = wp.mat33()
    sig = wp.vec3()
    V = wp.mat33()
    U, sig, V = wp.svd3(F)
    e0 = _safe_log(sig[0])
    e1 = _safe_log(sig[1])
    e2 = _safe_log(sig[2])
    tr = e0 + e1 + e2
    nu = params[m, P_NU]
    g0 = params[m, P_G0]
    if params[m, P_CG] != 0.0:
        g0 = g0 * wp.max(1.0 + params[m, P_CG] * wp.max(-evp, 0.0), 0.05)
    p = pprev
    if params[m, P_PMODE] > 0.5:
        p = _hardin_p_closed(-(tr + Jp), g0, params[m, P_PREF],
                             params[m, P_NEXP], nu)
    g = _hardin_g_of_p(p, g0, params[m, P_PREF], params[m, P_NEXP],
                       params[m, P_PMIN], params[m, P_PMAX])
    mu, lam = _hardin_mu_lam(g, nu)
    t0 = (2.0 * mu * e0 + lam * tr) / wp.max(sig[0], 1.0e-6)
    t1 = (2.0 * mu * e1 + lam * tr) / wp.max(sig[1], 1.0e-6)
    t2 = (2.0 * mu * e2 + lam * tr) / wp.max(sig[2], 1.0e-6)
    return U * wp.mat33(t0, 0.0, 0.0, 0.0, t1, 0.0, 0.0, 0.0, t2) * wp.transpose(V)


@wp.func
def particle_pk1(F: wp.mat33, m: int, Jp: float,
                 params: wp.array2d(dtype=wp.float32)) -> wp.mat33:
    """H29 back-compat 3+1-argument form (callers of the generic core, mpm_implicit.py
    line 1877/1847).  Forwards with pprev = 0 and evp = 0, which is exact for
    P_PMODE = 1 (the pressure is a function of F alone) and means 'start from
    the floor modulus' for P_PMODE = 0."""
    return particle_pk1(F, m, Jp, float(0.0), float(0.0), params)


@wp.func
def particle_pk1(F: wp.mat33, m: int, Jp: float, pprev: float, evp: float,
                 params: wp.array2d(dtype=wp.float32)) -> wp.mat33:
    P = wp.mat33()
    mu = params[m, 0]
    lam = params[m, 1]
    if m == 2:  # snow hardening
        h = wp.exp(params[m, 5] * (1.0 - Jp))
        mu = mu * h
        lam = lam * h
        P = pk1_corotated(F, mu, lam)
    elif m == 1:  # sand
        if params[m, P_PDEP] > 0.5:
            P = pk1_hencky_pdep(F, Jp, pprev, evp, params, m)
        else:
            P = pk1_hencky(F, mu, lam)
    elif m == 3:
        P = pk1_neohookean(F, mu, lam)
    else:
        P = pk1_corotated(F, mu, lam)
    return P


@wp.func
def particle_energy(F: wp.mat33, m: int, Jp: float,
                    params: wp.array2d(dtype=wp.float32)) -> float:
    return particle_energy(F, m, Jp, float(0.0), float(0.0), params)


@wp.func
def particle_energy(F: wp.mat33, m: int, Jp: float, pprev: float, evp: float,
                    params: wp.array2d(dtype=wp.float32)) -> float:
    e = float(0.0)
    mu = params[m, 0]
    lam = params[m, 1]
    if m == 2:
        h = wp.exp(params[m, 5] * (1.0 - Jp))
        e = energy_corotated(F, mu * h, lam * h)
    elif m == 1:
        if params[m, P_PDEP] > 0.5:
            # DIAGNOSTIC ONLY: in this form the Hardin law is hypoelastic, so
            # this is the Hencky energy at the current secant moduli, NOT an
            # exact potential of the stress p2g uses.
            mu, lam, pu = _sand_moduli(F, pprev, evp, params, m, Jp)
        e = energy_hencky(F, mu, lam)
    elif m == 3:
        e = energy_neohookean(F, mu, lam)
    else:
        e = energy_corotated(F, mu, lam)
    return e


@wp.func
def _dp_face(e0: float, e1: float, e2: float, Jp: float, mu: float, lam: float,
             alpha: float, beta: float, coh_tr: float, match_mode: int, phi_deg: float):
    """One Drucker-Prager / Mohr-Coulomb return with FROZEN moduli."""
    tr = e0 + e1 + e2 + Jp - coh_tr
    h0 = e0 - (e0 + e1 + e2 + Jp) / 3.0
    h1 = e1 - (e0 + e1 + e2 + Jp) / 3.0
    h2 = e2 - (e0 + e1 + e2 + Jp) / 3.0
    hn = wp.sqrt(h0 * h0 + h1 * h1 + h2 * h2) + 1.0e-20
    n0 = float(0.0)
    n1 = float(0.0)
    n2 = float(0.0)
    Jpn = float(0.0)
    dq = float(0.0)
    tip = float(0.0)
    if tr >= 0.0:
        # cone tip / tension cut-off: stress-free, bank the expansion
        Jpn = tr
        dq = wp.sqrt(e0 * e0 + e1 * e1 + e2 * e2)
        tip = 1.0
        if coh_tr > 0.0:
            # cohesive apex: keep uniform tension coh_tr on the diagonal
            n0 = coh_tr / 3.0
            n1 = coh_tr / 3.0
            n2 = coh_tr / 3.0
    else:
        alpha_eff = alpha
        if match_mode == 4:
            # Candidate (b): Mohr-Coulomb with Lode angle
            s = wp.sin(phi_deg * 0.017453292519943295)
            # sin(3*theta) = - 3*sqrt(6)*J3 / hn^3
            j3 = h0 * h1 * h2
            sin3th = wp.clamp(- 7.348469228349534 * j3 / (hn * hn * hn), -1.0, 1.0)
            th = wp.asin(sin3th) / 3.0
            # K(th) = cos(th) - 1/sqrt(3)*sin(th)*sin(phi)
            k_th = wp.cos(th) - 0.577350269 * wp.sin(th) * s
            alpha_eff = 0.47140452 * s / wp.max(k_th, 0.05)
            
        kap = (3.0 * lam + 2.0 * mu) / (2.0 * mu)
        dgamma = hn + kap * tr * alpha_eff
        dgamma = wp.max(dgamma, 0.0) / (1.0 + 3.0 * alpha_eff * beta * kap)
        g = dgamma / hn
        n0 = e0 - g * h0 - dgamma * beta
        n1 = e1 - g * h1 - dgamma * beta
        n2 = e2 - g * h2 - dgamma * beta
        dq = dgamma
    return n0, n1, n2, Jpn, dq, tip


@wp.func
def plastic_return(F: wp.mat33, m: int, Jp: float,
                   params: wp.array2d(dtype=wp.float32)):
    """H28 back-compat overload: the ORIGINAL four-argument form of
    the generic material-point core, for callers written against it (mpm_implicit.py
    line 1846).  Runs with qh = 0, i.e. hardening disabled -- which is what
    P_HARD = 0 means anyway -- and drops the qh output.  H29: also drops the
    pressure/compaction state (pprev = 0, evp = 0), which is exact for
    P_PMODE = 1."""
    Fn, Jpn, qhn, pn, evn = plastic_return(F, m, Jp, float(0.0), float(0.0),
                                           float(0.0), params)
    return Fn, Jpn


@wp.func
def plastic_return(F: wp.mat33, m: int, Jp: float, qh: float,
                   params: wp.array2d(dtype=wp.float32)):
    """H28 five-argument form, kept so h28 drivers run unchanged."""
    Fn, Jpn, qhn, pn, evn = plastic_return(F, m, Jp, qh, float(0.0),
                                           float(0.0), params)
    return Fn, Jpn, qhn


@wp.func
def mui_number(rate: float, pressure: float, grain: float) -> float:
    # Jop et al. 2006 eqs. 1-3. Numerical p floor only in I, not yield pressure.
    return rate * grain / wp.sqrt(wp.max(pressure, 1.0e-6) / 2500.0)

@wp.func
def shear_rate(C: wp.mat33) -> float:
    D = 0.5 * (C + wp.transpose(C))
    D = D - (wp.trace(D) / 3.0) * wp.identity(n=3, dtype=float)
    return wp.sqrt(2.0 * wp.ddot(D, D))

@wp.func
def plastic_return(F: wp.mat33, m: int, Jp: float, qh: float, pprev: float,
                   evp: float, params: wp.array2d(dtype=wp.float32)):
    return plastic_return_rate(F, m, Jp, qh, pprev, evp, params, 0.0)

@wp.func
def plastic_return_rate(F: wp.mat33, m: int, Jp: float, qh: float, pprev: float,
                   evp: float, params: wp.array2d(dtype=wp.float32), rate: float):
    """Return-mapping.  Returns (F_elastic_new, Jp_new, qh_new, p_new, evp_new).

    H29: for MAT_SAND with P_PDEP = 1 the moduli (mu, lam) are rebuilt per
    particle per step from the pressure -- G = G0 (p/p_ref)^n -- and the trial
    stress is taken with those moduli.  Consistency of the return map with the
    pressure it assumed is handled by P_PMODE / P_FPIT (see the module notes).
    """
    Fn = F
    Jpn = Jp
    qhn = qh
    pn = pprev
    evn = evp
    if m == 1:
        # Drucker-Prager, Klar et al. 2016 (cohesion 0), WITH the volume
        # correction of sec. 4.3.2 (unchanged), H28's hardening + dilatancy,
        # and H29's pressure-dependent moduli.
        U, sig, V = wp.svd3(F)
        e0 = _safe_log(wp.abs(sig[0]))
        e1 = _safe_log(wp.abs(sig[1]))
        e2 = _safe_log(wp.abs(sig[2]))
        mu = params[m, 0]
        lam = params[m, 1]
        alpha = params[m, 2]
        beta = params[m, 8]
        nu = params[m, P_NU]
        pdep = params[m, P_PDEP]
        g0 = params[m, P_G0]
        if params[m, P_CG] != 0.0:
            g0 = g0 * wp.max(1.0 + params[m, P_CG] * wp.max(-evp, 0.0), 0.05)
        pref = params[m, P_PREF]
        nex = params[m, P_NEXP]
        pmin = params[m, P_PMIN]
        pmax = params[m, P_PMAX]
        match_mode = int(params[m, P_DPMATCH])
        phi_f = params[m, P_PHI]
        if params[m, 13] > 0.5:
            # phi_F(q) = h0 + (h1 q - h3) exp(-h2 q)   [degrees]
            phi_f = params[m, 9] + (params[m, 10] * qh - params[m, 12]) \
                * wp.exp(-params[m, 11] * qh)
            phi_f = phi_f + params[m, P_CPHI] * wp.max(-evp, 0.0)
            phi_f = wp.clamp(phi_f, 1.0, 70.0)
            alpha = _dp_alpha_matched(phi_f, match_mode)
        trial_tr = e0 + e1 + e2 + Jp
        if params[m, P_MUI] > 0.5:
            # U118 uses constant elasticity, zero dilatancy and no hardening.
            # Isochoric return preserves this mean Cauchy pressure.
            pressure = wp.max(-(lam + 2.0*mu/3.0) * trial_tr /
                              wp.max(wp.determinant(F), 1.0e-6), 0.0)
            inertial = mui_number(rate, pressure, params[m, P_GRAIN])
            mus = wp.tan(21.0 * 0.017453292519943295)
            mu2 = wp.tan(33.0 * 0.017453292519943295)
            friction = mus + (mu2-mus) * inertial / (0.28+inertial)
            phi_f = wp.atan(friction) * 57.29577951308232
            alpha = _dp_alpha_matched(phi_f, match_mode)
        if pdep > 0.5:
            # p used for the moduli: P_PMODE 1 = the closed-form fixed point of
            # p = K(p)*ec on the TRIAL strain; P_PMODE 0 = last step's p.
            p_use = pprev
            if params[m, P_PMODE] > 0.5:
                p_use = _hardin_p_closed(-trial_tr, g0, pref, nex, nu)
            pfloor = params[m, P_PFLOOR]
            if pfloor > 0.0:
                p_use = wp.max(p_use, pfloor)
            coh = params[m, P_COH]
            nfp = int(params[m, P_FPIT])
            n0 = float(0.0)
            n1 = float(0.0)
            n2 = float(0.0)
            jb = float(0.0)
            dq = float(0.0)
            tip = float(0.0)
            for it in range(nfp + 1):
                gsh = _hardin_g_of_p(p_use, g0, pref, nex, pmin, pmax)
                mu, lam = _hardin_mu_lam(gsh, nu)
                kk = lam + 2.0 * mu / 3.0
                coh_tr = 0.0
                if coh > 0.0:
                    coh_tr = coh / wp.max(kk, 1.0e-9)
                n0, n1, n2, jb, dq, tip = _dp_face(e0, e1, e2, Jp, mu, lam,
                                                   alpha, beta, coh_tr, match_mode, phi_f)
                # p implied by the RETURNED state; the next pass uses it.
                p_use = _hardin_p_closed(-(n0 + n1 + n2 + jb), g0, pref, nex, nu)
                if pfloor > 0.0:
                    p_use = wp.max(p_use, pfloor)
            Jpn = jb
            qhn = qh + dq
            if tip > 0.5:
                pn = 0.0
            else:
                gsh = _hardin_g_of_p(p_use, g0, pref, nex, pmin, pmax)
                mu, lam = _hardin_mu_lam(gsh, nu)
                kk = lam + 2.0 * mu / 3.0
                pn = wp.max(-kk * (n0 + n1 + n2), 0.0)
                evn = evp + ((trial_tr - Jp) - (n0 + n1 + n2))
            s0 = wp.exp(n0)
            s1 = wp.exp(n1)
            s2 = wp.exp(n2)
            Fn = U * wp.mat33(s0, 0.0, 0.0, 0.0, s1, 0.0, 0.0, 0.0, s2) * wp.transpose(V)
        else:
            # ---- H28 path, byte-for-byte the original expressions when coh=0 ----
            tr = e0 + e1 + e2 + Jp
            coh = params[m, P_COH]
            kk0 = lam + 2.0 * mu / 3.0
            coh_tr = 0.0
            if coh > 0.0:
                coh_tr = coh / wp.max(kk0, 1.0e-9)
            trs = tr - coh_tr
            h0 = e0 - tr / 3.0
            h1 = e1 - tr / 3.0
            h2 = e2 - tr / 3.0
            hn = wp.sqrt(h0 * h0 + h1 * h1 + h2 * h2) + 1.0e-20
            s0 = float(1.0)
            s1 = float(1.0)
            s2 = float(1.0)
            if trs >= 0.0:
                Jpn = trs
                qhn = qh + wp.sqrt(e0 * e0 + e1 * e1 + e2 * e2)
                if coh_tr > 0.0:
                    c3 = coh_tr / 3.0
                    s0 = wp.exp(c3)
                    s1 = wp.exp(c3)
                    s2 = wp.exp(c3)
            else:
                Jpn = 0.0
                kap = (3.0 * lam + 2.0 * mu) / (2.0 * mu)
                dgamma = hn + kap * trs * alpha
                dgamma = wp.max(dgamma, 0.0) / (1.0 + 3.0 * alpha * beta * kap)
                g = dgamma / hn
                s0 = wp.exp(e0 - g * h0 - dgamma * beta)
                s1 = wp.exp(e1 - g * h1 - dgamma * beta)
                s2 = wp.exp(e2 - g * h2 - dgamma * beta)
                qhn = qh + dgamma
            if params[m, P_TIP] > 0.0:
                # No tensile hydrostatic stress. Retain deviatoric stress up to
                # q=M max(p,p_floor), with J converting Cauchy to Kirchhoff.
                raw_tr = e0 + e1 + e2
                mean = raw_tr / 3.0
                d0 = e0-mean; d1 = e1-mean; d2 = e2-mean
                norm = wp.sqrt(d0*d0+d1*d1+d2*d2) + 1.0e-20
                returned_tr = wp.min(tr, 0.0)
                J = wp.exp(returned_tr)
                pressure = -kk0 * returned_tr / J
                limit = alpha * 3.0 * wp.max(pressure, params[m, P_TIP]) * J / (2.0*mu)
                ratio = wp.min(1.0, limit/norm)
                s0 = wp.exp(returned_tr/3.0 + ratio*d0)
                s1 = wp.exp(returned_tr/3.0 + ratio*d1)
                s2 = wp.exp(returned_tr/3.0 + ratio*d2)
                Jpn = wp.max(tr, 0.0)
                qhn = qh + (1.0-ratio)*norm
            Fn = U * wp.mat33(s0, 0.0, 0.0, 0.0, s1, 0.0, 0.0, 0.0, s2) * wp.transpose(V)
    elif m == 2:
        # snow: clamp singular values, move the rest into the plastic volume
        U, sig, V = wp.svd3(F)
        tc = params[m, 3]
        ts = params[m, 4]
        Jt = float(1.0)
        Jc = float(1.0)
        c0 = wp.clamp(sig[0], 1.0 - tc, 1.0 + ts)
        c1 = wp.clamp(sig[1], 1.0 - tc, 1.0 + ts)
        c2 = wp.clamp(sig[2], 1.0 - tc, 1.0 + ts)
        Jt = sig[0] * sig[1] * sig[2]
        Jc = c0 * c1 * c2
        Fn = U * wp.mat33(c0, 0.0, 0.0, 0.0, c1, 0.0, 0.0, 0.0, c2) * wp.transpose(V)
        Jpn = wp.clamp(Jp * Jt / wp.max(Jc, 1.0e-8), 0.05, 20.0)
    return Fn, Jpn, qhn, pn, evn


# ==========================================================================
# Warp kernels -- P2G
# ==========================================================================
@wp.func
def _bspline_w(fx: wp.vec3):
    w0 = wp.vec3(0.5 * (1.5 - fx[0]) * (1.5 - fx[0]),
                 0.5 * (1.5 - fx[1]) * (1.5 - fx[1]),
                 0.5 * (1.5 - fx[2]) * (1.5 - fx[2]))
    w1 = wp.vec3(0.75 - (fx[0] - 1.0) * (fx[0] - 1.0),
                 0.75 - (fx[1] - 1.0) * (fx[1] - 1.0),
                 0.75 - (fx[2] - 1.0) * (fx[2] - 1.0))
    w2 = wp.vec3(0.5 * (fx[0] - 0.5) * (fx[0] - 0.5),
                 0.5 * (fx[1] - 0.5) * (fx[1] - 0.5),
                 0.5 * (fx[2] - 0.5) * (fx[2] - 0.5))
    return w0, w1, w2


@wp.func
def _cubic1(t: float) -> wp.vec4:
    u = t - 1.0
    return wp.vec4((1.0-u)*(1.0-u)*(1.0-u)/6.0,
                   (3.0*u*u*u-6.0*u*u+4.0)/6.0,
                   (-3.0*u*u*u+3.0*u*u+3.0*u+1.0)/6.0,
                   u*u*u/6.0)


@wp.kernel
def p2g_int64_kernel(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    F: wp.array(dtype=wp.mat33),
    C: wp.array(dtype=wp.mat33),
    mass: wp.array(dtype=wp.float32),
    vol0: wp.array(dtype=wp.float32),
    mat: wp.array(dtype=wp.int32),
    Jp: wp.array(dtype=wp.float32),
    pp: wp.array(dtype=wp.float32),
    evp: wp.array(dtype=wp.float32),
    params: wp.array2d(dtype=wp.float32),
    grid_mi: wp.array(dtype=wp.int64),
    grid_mvi: wp.array2d(dtype=wp.int64),
    res: int,
    dx: float,
    inv_dx: float,
    origin: wp.vec3,
    dt: float,
):
    p = wp.tid()
    m = mat[p]
    Fp = F[p]
    P = particle_pk1(Fp, m, Jp[p], pp[p], evp[p], params)
    factor = float(4.0)
    if params[m, P_SPLINE] > 0.5:
        factor = 3.0
    stress = (-dt * vol0[p] * factor * inv_dx * inv_dx) * (P * wp.transpose(Fp))
    affine = stress + mass[p] * C[p]
    mv_p = mass[p] * v[p]

    cubic = params[m, P_SPLINE] > 0.5
    support = int(3)
    if cubic:
        support = 4
    xg = (x[p] - origin) * inv_dx
    bf = wp.vec3(wp.floor(xg[0] - 0.5), wp.floor(xg[1] - 0.5), wp.floor(xg[2] - 0.5))
    if cubic:
        bf = wp.vec3(wp.floor(xg[0])-1.0, wp.floor(xg[1])-1.0, wp.floor(xg[2])-1.0)
    fx = xg - bf
    w0, w1, w2 = _bspline_w(fx)
    cx = _cubic1(fx[0]); cy = _cubic1(fx[1]); cz = _cubic1(fx[2])
    b0 = int(bf[0])
    b1 = int(bf[1])
    b2 = int(bf[2])

    for i in range(support):
        wi = w0[0]
        if i == 1:
            wi = w1[0]
        if i == 2:
            wi = w2[0]
        if cubic:
            wi = cx[i]
        ix = b0 + i
        if ix < 0 or ix >= res:
            continue
        for j in range(support):
            wj = w0[1]
            if j == 1:
                wj = w1[1]
            if j == 2:
                wj = w2[1]
            if cubic:
                wj = cy[j]
            iy = b1 + j
            if iy < 0 or iy >= res:
                continue
            for k in range(support):
                wk = w0[2]
                if k == 1:
                    wk = w1[2]
                if k == 2:
                    wk = w2[2]
                if cubic:
                    wk = cz[k]
                iz = b2 + k
                if iz < 0 or iz >= res:
                    continue
                wgt = wi * wj * wk
                dpos = wp.vec3(float(i) - fx[0], float(j) - fx[1], float(k) - fx[2]) * dx
                contrib = wgt * (mv_p + affine * dpos)
                gi = (ix * res + iy) * res + iz
                wp.atomic_add(grid_mi, gi,
                              wp.int64(wp.round(wp.float64(wgt * mass[p]) * _MASS_SCALE)))
                for c in range(3):
                    wp.atomic_add(grid_mvi, gi, c,
                                  wp.int64(wp.round(wp.float64(contrib[c]) * _MOM_SCALE)))


@wp.kernel
def p2g_float32_kernel(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    F: wp.array(dtype=wp.mat33),
    C: wp.array(dtype=wp.mat33),
    mass: wp.array(dtype=wp.float32),
    vol0: wp.array(dtype=wp.float32),
    mat: wp.array(dtype=wp.int32),
    Jp: wp.array(dtype=wp.float32),
    pp: wp.array(dtype=wp.float32),
    evp: wp.array(dtype=wp.float32),
    params: wp.array2d(dtype=wp.float32),
    grid_m: wp.array(dtype=wp.float32),
    grid_mv: wp.array(dtype=wp.vec3),
    res: int,
    dx: float,
    inv_dx: float,
    origin: wp.vec3,
    dt: float,
):
    p = wp.tid()
    m = mat[p]
    Fp = F[p]
    P = particle_pk1(Fp, m, Jp[p], pp[p], evp[p], params)
    factor = float(4.0)
    if params[m, P_SPLINE] > 0.5:
        factor = 3.0
    stress = (-dt * vol0[p] * factor * inv_dx * inv_dx) * (P * wp.transpose(Fp))
    affine = stress + mass[p] * C[p]
    mv_p = mass[p] * v[p]

    cubic = params[m, P_SPLINE] > 0.5
    support = int(3)
    if cubic:
        support = 4
    xg = (x[p] - origin) * inv_dx
    bf = wp.vec3(wp.floor(xg[0] - 0.5), wp.floor(xg[1] - 0.5), wp.floor(xg[2] - 0.5))
    if cubic:
        bf = wp.vec3(wp.floor(xg[0])-1.0, wp.floor(xg[1])-1.0, wp.floor(xg[2])-1.0)
    fx = xg - bf
    w0, w1, w2 = _bspline_w(fx)
    cx = _cubic1(fx[0]); cy = _cubic1(fx[1]); cz = _cubic1(fx[2])
    b0 = int(bf[0])
    b1 = int(bf[1])
    b2 = int(bf[2])

    for i in range(support):
        wi = w0[0]
        if i == 1:
            wi = w1[0]
        if i == 2:
            wi = w2[0]
        if cubic:
            wi = cx[i]
        ix = b0 + i
        if ix < 0 or ix >= res:
            continue
        for j in range(support):
            wj = w0[1]
            if j == 1:
                wj = w1[1]
            if j == 2:
                wj = w2[1]
            if cubic:
                wj = cy[j]
            iy = b1 + j
            if iy < 0 or iy >= res:
                continue
            for k in range(support):
                wk = w0[2]
                if k == 1:
                    wk = w1[2]
                if k == 2:
                    wk = w2[2]
                if cubic:
                    wk = cz[k]
                iz = b2 + k
                if iz < 0 or iz >= res:
                    continue
                wgt = wi * wj * wk
                dpos = wp.vec3(float(i) - fx[0], float(j) - fx[1], float(k) - fx[2]) * dx
                gi = (ix * res + iy) * res + iz
                wp.atomic_add(grid_m, gi, wgt * mass[p])
                wp.atomic_add(grid_mv, gi, wgt * (mv_p + affine * dpos))


@wp.kernel
def fixed_to_float_kernel(
    grid_mi: wp.array(dtype=wp.int64),
    grid_mvi: wp.array2d(dtype=wp.int64),
    grid_m: wp.array(dtype=wp.float32),
    grid_mv: wp.array(dtype=wp.vec3),
):
    gi = wp.tid()
    grid_m[gi] = wp.float32(wp.float64(grid_mi[gi]) / _MASS_SCALE)
    grid_mv[gi] = wp.vec3(
        wp.float32(wp.float64(grid_mvi[gi, 0]) / _MOM_SCALE),
        wp.float32(wp.float64(grid_mvi[gi, 1]) / _MOM_SCALE),
        wp.float32(wp.float64(grid_mvi[gi, 2]) / _MOM_SCALE),
    )


# ==========================================================================
# Warp kernels -- grid update
# ==========================================================================
@wp.kernel
def grid_normalize_kernel(
    grid_m: wp.array(dtype=wp.float32),
    grid_mv: wp.array(dtype=wp.vec3),
    grid_v: wp.array(dtype=wp.vec3),
    gravity: wp.vec3,
    dt: float,
):
    gi = wp.tid()
    m = grid_m[gi]
    if m > 0.0:
        grid_v[gi] = grid_mv[gi] / m + dt * gravity
    else:
        grid_v[gi] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def grid_snapshot_kernel(
    grid_v: wp.array(dtype=wp.vec3),
    grid_v_old: wp.array(dtype=wp.vec3),
):
    """U10 (c): snapshot grid velocity before the update (FLIP increment)."""
    gi = wp.tid()
    grid_v_old[gi] = grid_v[gi]


@wp.kernel
def grid_floor_kernel(
    grid_m: wp.array(dtype=wp.float32),
    grid_mv: wp.array(dtype=wp.vec3),
    grid_v: wp.array(dtype=wp.vec3),
    gravity: wp.vec3,
    dt: float,
    m_full: float,
    eps: float,
    mode: int,
    res: int,
):
    """U10 (a): free-surface mass floor. Nodes with 0 < m < eps*m_full:
    mode 1 = freeze (v = 0); mode 2 = neighbour average over face neighbours
    with m >= eps*m_full (own mv/m fallback when none qualifies);
    mode 3 = ghost mass: v = mv/(m + eps*m_full) + dt*g. mode 0 = off.
    Per-node independent writes, no atomics."""
    gi = wp.tid()
    m = grid_m[gi]
    thr = eps * m_full
    if m <= 0.0 or m >= thr or mode == 0:
        return
    if mode == 1:
        grid_v[gi] = wp.vec3(0.0, 0.0, 0.0)
    elif mode == 3:
        grid_v[gi] = grid_mv[gi] / (m + thr) + dt * gravity
    else:
        iz = gi % res
        iy = (gi // res) % res
        ix = gi // (res * res)
        acc = wp.vec3(0.0, 0.0, 0.0)
        cnt = 0
        if ix > 0:
            j = gi - res * res
            if grid_m[j] >= thr:
                acc += grid_v[j]
                cnt += 1
        if ix < res - 1:
            j = gi + res * res
            if grid_m[j] >= thr:
                acc += grid_v[j]
                cnt += 1
        if iy > 0:
            j = gi - res
            if grid_m[j] >= thr:
                acc += grid_v[j]
                cnt += 1
        if iy < res - 1:
            j = gi + res
            if grid_m[j] >= thr:
                acc += grid_v[j]
                cnt += 1
        if iz > 0:
            j = gi - 1
            if grid_m[j] >= thr:
                acc += grid_v[j]
                cnt += 1
        if iz < res - 1:
            j = gi + 1
            if grid_m[j] >= thr:
                acc += grid_v[j]
                cnt += 1
        if cnt > 0:
            grid_v[gi] = acc / float(cnt)


@wp.func
def bc_velocity(v: wp.vec3, n: wp.vec3, mode: int, mu: float) -> wp.vec3:
    """Apply one axis-aligned wall.  ``n`` = inward normal."""
    out = v
    if mode == 0:  # sticky
        out = wp.vec3(0.0, 0.0, 0.0)
    elif mode == 1:  # slip: kill the normal component in both directions
        out = v - wp.dot(v, n) * n
    else:  # separate: only block motion into the wall
        vn = wp.dot(v, n)
        if vn < 0.0:
            vt = v - vn * n
            lt = wp.length(vt)
            if mu > 0.0 and lt > 1.0e-12:
                vt = vt * wp.max(0.0, 1.0 + mu * vn / lt)
            out = vt
    return out


@wp.kernel
def grid_boundary_kernel(
    grid_m: wp.array(dtype=wp.float32),
    grid_v: wp.array(dtype=wp.vec3),
    bc_mode: wp.array(dtype=wp.int32),
    bc_mu: wp.array(dtype=wp.float32),
    res: int,
    bound: int,
):
    gi = wp.tid()
    if grid_m[gi] <= 0.0:
        return
    iz = gi % res
    iy = (gi // res) % res
    ix = gi // (res * res)
    v = grid_v[gi]
    if ix < bound:
        v = bc_velocity(v, wp.vec3(1.0, 0.0, 0.0), bc_mode[0], bc_mu[0])
    if ix >= res - bound:
        v = bc_velocity(v, wp.vec3(-1.0, 0.0, 0.0), bc_mode[1], bc_mu[1])
    if iy < bound:
        v = bc_velocity(v, wp.vec3(0.0, 1.0, 0.0), bc_mode[2], bc_mu[2])
    if iy >= res - bound:
        v = bc_velocity(v, wp.vec3(0.0, -1.0, 0.0), bc_mode[3], bc_mu[3])
    if iz < bound:
        v = bc_velocity(v, wp.vec3(0.0, 0.0, 1.0), bc_mode[4], bc_mu[4])
    if iz >= res - bound:
        v = bc_velocity(v, wp.vec3(0.0, 0.0, -1.0), bc_mode[5], bc_mu[5])
    grid_v[gi] = v


# ==========================================================================
# Warp kernels -- G2P
# ==========================================================================
@wp.kernel
def g2p_kernel(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    F: wp.array(dtype=wp.mat33),
    C: wp.array(dtype=wp.mat33),
    mat: wp.array(dtype=wp.int32),
    Jp: wp.array(dtype=wp.float32),
    qh: wp.array(dtype=wp.float32),
    pp: wp.array(dtype=wp.float32),
    evp: wp.array(dtype=wp.float32),
    params: wp.array2d(dtype=wp.float32),
    grid_v: wp.array(dtype=wp.vec3),
    grid_v_old: wp.array(dtype=wp.vec3),
    res: int,
    dx: float,
    inv_dx: float,
    origin: wp.vec3,
    dt: float,
):
    p = wp.tid()
    cubic = params[mat[p], P_SPLINE] > 0.5
    support = int(3)
    if cubic:
        support = 4
    xg = (x[p] - origin) * inv_dx
    bf = wp.vec3(wp.floor(xg[0] - 0.5), wp.floor(xg[1] - 0.5), wp.floor(xg[2] - 0.5))
    if cubic:
        bf = wp.vec3(wp.floor(xg[0])-1.0, wp.floor(xg[1])-1.0, wp.floor(xg[2])-1.0)
    fx = xg - bf
    w0, w1, w2 = _bspline_w(fx)
    cx = _cubic1(fx[0]); cy = _cubic1(fx[1]); cz = _cubic1(fx[2])
    b0 = int(bf[0])
    b1 = int(bf[1])
    b2 = int(bf[2])

    new_v = wp.vec3(0.0, 0.0, 0.0)
    old_v = wp.vec3(0.0, 0.0, 0.0)
    new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for i in range(support):
        wi = w0[0]
        if i == 1:
            wi = w1[0]
        if i == 2:
            wi = w2[0]
        if cubic:
            wi = cx[i]
        ix = b0 + i
        if ix < 0 or ix >= res:
            continue
        for j in range(support):
            wj = w0[1]
            if j == 1:
                wj = w1[1]
            if j == 2:
                wj = w2[1]
            if cubic:
                wj = cy[j]
            iy = b1 + j
            if iy < 0 or iy >= res:
                continue
            for k in range(support):
                wk = w0[2]
                if k == 1:
                    wk = w1[2]
                if k == 2:
                    wk = w2[2]
                if cubic:
                    wk = cz[k]
                iz = b2 + k
                if iz < 0 or iz >= res:
                    continue
                wgt = wi * wj * wk
                dpos = wp.vec3(float(i) - fx[0], float(j) - fx[1], float(k) - fx[2])
                gv = grid_v[(ix * res + iy) * res + iz]
                new_v += wgt * gv
                old_v += wgt * grid_v_old[(ix * res + iy) * res + iz]
                factor = float(4.0)
                if cubic:
                    factor = 3.0
                new_C += (factor * inv_dx * wgt) * wp.outer(gv, dpos)

    # U10 (c): PIC/FLIP blend. flip == 0 runs the legacy APIC assignment
    # bit-for-bit; otherwise v = (1-f)*v_pic + f*(v_old + dv_pic).
    flip = params[mat[p], P_FLIP]
    if flip <= 0.0:
        v[p] = new_v
    else:
        v[p] = (1.0 - flip) * new_v + flip * (v[p] + (new_v - old_v))
    C[p] = new_C
    x[p] = x[p] + dt * v[p]
    Fnew = (wp.identity(n=3, dtype=float) + dt * new_C) * F[p]
    Fnew, jp_new, qh_new, p_new, ev_new = plastic_return_rate(
        Fnew, mat[p], Jp[p], qh[p], pp[p], evp[p], params, shear_rate(new_C))
    F[p] = Fnew
    Jp[p] = jp_new
    qh[p] = qh_new
    pp[p] = p_new
    evp[p] = ev_new


# ==========================================================================
# diagnostics
# ==========================================================================
@wp.kernel
def energy_kernel(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    F: wp.array(dtype=wp.mat33),
    mass: wp.array(dtype=wp.float32),
    vol0: wp.array(dtype=wp.float32),
    mat: wp.array(dtype=wp.int32),
    Jp: wp.array(dtype=wp.float32),
    pp: wp.array(dtype=wp.float32),
    evp: wp.array(dtype=wp.float32),
    params: wp.array2d(dtype=wp.float32),
    gravity: wp.vec3,
    acc: wp.array(dtype=wp.int64),
):
    p = wp.tid()
    ke = 0.5 * mass[p] * wp.dot(v[p], v[p])
    pe = vol0[p] * particle_energy(F[p], mat[p], Jp[p], pp[p], evp[p], params)
    gpe = -mass[p] * wp.dot(gravity, x[p])
    wp.atomic_add(acc, 0, wp.int64(wp.round(wp.float64(ke) * _ENERGY_SCALE)))
    wp.atomic_add(acc, 1, wp.int64(wp.round(wp.float64(pe) * _ENERGY_SCALE)))
    wp.atomic_add(acc, 2, wp.int64(wp.round(wp.float64(gpe) * _ENERGY_SCALE)))


@wp.kernel
def grid_absmax_kernel(
    grid_m: wp.array(dtype=wp.float32),
    grid_mv: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=wp.float32),
):
    gi = wp.tid()
    wp.atomic_max(out, 0, wp.abs(grid_m[gi]))
    mv = grid_mv[gi]
    wp.atomic_max(out, 1, wp.max(wp.max(wp.abs(mv[0]), wp.abs(mv[1])), wp.abs(mv[2])))


# ==========================================================================
# solver
# ==========================================================================
class MPMSolver:
    """MLS-MPM solver.  See module docstring for the data layout contract."""

    def __init__(
        self,
        res: int = 64,
        domain_size: float = 1.0,
        domain_lower: Sequence[float] = (0.0, 0.0, 0.0),
        gravity: Sequence[float] = (0.0, -9.81, 0.0),
        dt: float = 1.0e-4,
        scatter: str = "int64",
        bc: Sequence[str] = ("sticky",) * 6,
        bc_friction: Sequence[float] = (0.0,) * 6,
        bound: int = 3,
        device: str = "cuda:0",
    ):
        if scatter not in ("int64", "float32"):
            raise ValueError("scatter must be 'int64' or 'float32'")
        wp.init()
        self.device = device
        self.res = int(res)
        self.n_nodes = self.res ** 3
        self.domain_size = float(domain_size)
        self.dx = self.domain_size / self.res
        self.inv_dx = 1.0 / self.dx
        self.origin = wp.vec3(*[float(a) for a in domain_lower])
        self.origin_np = np.array(domain_lower, dtype=np.float64)
        self.gravity = wp.vec3(*[float(g) for g in gravity])
        self.gravity_np = np.array(gravity, dtype=np.float64)
        self.dt = float(dt)
        self.scatter = scatter
        self.bound = int(bound)
        self.frame = 0

        with wp.ScopedDevice(self.device):
            self.grid_m = wp.zeros(self.n_nodes, dtype=wp.float32)
            self.grid_mv = wp.zeros(self.n_nodes, dtype=wp.vec3)
            self.grid_v = wp.zeros(self.n_nodes, dtype=wp.vec3)
            # U10 (c): previous-step grid velocity for the FLIP increment.
            self.grid_v_old = wp.zeros(self.n_nodes, dtype=wp.vec3)
            if scatter == "int64":
                self.grid_mi = wp.zeros(self.n_nodes, dtype=wp.int64)
                self.grid_mvi = wp.zeros((self.n_nodes, 3), dtype=wp.int64)
            else:
                self.grid_mi = None
                self.grid_mvi = None
            self.bc_mode = wp.array(
                np.array([_BC_NAMES[b] for b in bc], dtype=np.int32), dtype=wp.int32)
            self.bc_mu = wp.array(np.array(bc_friction, dtype=np.float32),
                                  dtype=wp.float32)
            self.params = wp.zeros((N_MAT, N_PARAM), dtype=wp.float32)
            self._e_acc = wp.zeros(3, dtype=wp.int64)
            self._absmax = wp.zeros(2, dtype=wp.float32)

        self._params_np = np.zeros((N_MAT, N_PARAM), dtype=np.float32)
        self.set_material(MAT_COROTATED, E=1.0e4, nu=0.2)
        # H29: the Hardin columns get non-degenerate defaults even when
        # pdep is off, so p_ref is never 0 in a pow().
        self.set_material(MAT_SAND, E=3.537e5, nu=0.3, phi_deg=30.0,
                          pdep=False, G0=3.537e5 / 2.6, p_ref=1.0e3,
                          n_exp=0.5, p_min=10.0, p_max=1.0e4,
                          p_mode="trial", fp_iters=0, c_phi=0.0, c_g=0.0)
        self.set_material(MAT_SNOW, E=1.4e5, nu=0.2,
                          theta_c=2.5e-2, theta_s=7.5e-3, xi=10.0)
        self.set_material(MAT_NEOHOOKEAN, E=1.0e4, nu=0.2)

        self._px: list[np.ndarray] = []
        self._pv: list[np.ndarray] = []
        self._pm: list[np.ndarray] = []
        self._pvol: list[np.ndarray] = []
        self._pmat: list[np.ndarray] = []
        self.n_particles = 0
        self._finalized = False
        # U10 (a): mass floor, solver-level. mode 0 = off, 1 = freeze,
        # 2 = neighbour average, 3 = ghost mass; meps scales m_full = rho*dx^3.
        self._u10_mmode = 0
        self._u10_meps = 0.0
        self._u10_mfull = 0.0

    # ---------------- materials ----------------
    def set_material(self, mat_id: int, E: Optional[float] = None,
                     nu: Optional[float] = None, phi_deg: Optional[float] = None,
                     theta_c: Optional[float] = None, theta_s: Optional[float] = None,
                     xi: Optional[float] = None,
                     psi_deg: Optional[float] = None,
                     associated: Optional[bool] = None,
                     hardening: Optional[bool] = None,
                     h0: Optional[float] = None, h1: Optional[float] = None,
                     h2: Optional[float] = None, h3: Optional[float] = None,
                      pdep: Optional[bool] = None, G0: Optional[float] = None,
                      p_ref: Optional[float] = None, n_exp: Optional[float] = None,
                      p_min: Optional[float] = None, p_max: Optional[float] = None,
                      p_mode: Optional[str] = None, fp_iters: Optional[int] = None,
                      c_phi: Optional[float] = None, c_g: Optional[float] = None,
                      coh: Optional[float] = None, p_floor: Optional[float] = None,
                      flip: Optional[float] = None, dp_match: Optional[int] = None) -> None:
        row = self._params_np[mat_id]
        if dp_match is not None:
            row[P_DPMATCH] = float(dp_match)
        match_mode = int(row[P_DPMATCH])
        if E is not None:
            row[P_E] = E
        if nu is not None:
            row[P_NU] = nu
        E_ = float(row[P_E])
        nu_ = float(row[P_NU])
        row[P_MU] = E_ / (2.0 * (1.0 + nu_))
        row[P_LAM] = E_ * nu_ / ((1.0 + nu_) * (1.0 - 2.0 * nu_))
        if phi_deg is not None:
            row[P_PHI] = float(phi_deg)
            s = math.sin(math.radians(phi_deg))
            if match_mode == 1:
                # TXE
                row[P_ALPHA] = math.sqrt(2.0 / 3.0) * 2.0 * s / (3.0 + s)
            elif match_mode == 2:
                # Plane strain
                t = math.tan(math.radians(phi_deg))
                row[P_ALPHA] = math.sqrt(2.0) * t / math.sqrt(9.0 + 12.0 * t * t)
            elif match_mode == 3:
                # Simple shear
                t = math.tan(math.radians(phi_deg))
                row[P_ALPHA] = math.sqrt(2.0 / 3.0) * t
            else:
                # TXC
                row[P_ALPHA] = math.sqrt(2.0 / 3.0) * 2.0 * s / (3.0 - s)
        if theta_c is not None:
            row[P_THETA_C] = theta_c
        if theta_s is not None:
            row[P_THETA_S] = theta_s
        if xi is not None:
            row[P_XI] = xi
        # ---- H28: dilatancy + hardening ----
        if psi_deg is not None:
            sp = math.sin(math.radians(psi_deg))
            row[P_BETA] = math.sqrt(2.0 / 3.0) * 2.0 * sp / (3.0 - sp)
        if associated:
            row[P_BETA] = row[P_ALPHA]
        if h0 is not None:
            row[P_H0] = h0
        if h1 is not None:
            row[P_H1] = h1
        if h2 is not None:
            row[P_H2] = h2
        if h3 is not None:
            row[P_H3] = h3
        if hardening is not None:
            row[P_HARD] = 1.0 if hardening else 0.0
        # ---- H29: Hardin pressure-dependent elasticity ----
        if G0 is not None:
            row[P_G0] = G0
            # keep the legacy (fixed-modulus) fields consistent at p = p_ref,
            # so wave_speed()/suggest_dt() and every non-pdep path still work
            nu_g = float(row[P_NU])
            row[P_E] = 2.0 * G0 * (1.0 + nu_g)
            row[P_MU] = G0
            row[P_LAM] = 2.0 * G0 * nu_g / (1.0 - 2.0 * nu_g)
        if p_ref is not None:
            row[P_PREF] = p_ref
        if n_exp is not None:
            row[P_NEXP] = n_exp
        if p_min is not None:
            row[P_PMIN] = p_min
        if p_max is not None:
            row[P_PMAX] = p_max
        if p_mode is not None:
            row[P_PMODE] = {"lag": 0.0, "trial": 1.0}[p_mode]
        if fp_iters is not None:
            row[P_FPIT] = float(int(fp_iters))
        if c_phi is not None:
            row[P_CPHI] = c_phi
        if c_g is not None:
            row[P_CG] = c_g
        # ---- U10: cohesion / p-floor / FLIP blend ----
        if coh is not None:
            row[P_COH] = coh
        if p_floor is not None:
            row[P_PFLOOR] = p_floor
        if flip is not None:
            row[P_FLIP] = flip
        if pdep is not None:
            row[P_PDEP] = 1.0 if pdep else 0.0
        self.params.assign(self._params_np)

    def wave_speed(self, mat_id: int, rho: float) -> float:
        mu = float(self._params_np[mat_id, P_MU])
        lam = float(self._params_np[mat_id, P_LAM])
        return math.sqrt((lam + 2.0 * mu) / rho)

    def shear_modulus_at(self, mat_id: int, p: float) -> float:
        """H29: G(p) = G0 (p/p_ref)^n, clamped to [p_min, p_max]."""
        r = self._params_np[mat_id]
        if float(r[P_PDEP]) < 0.5:
            return float(r[P_MU])
        pe = min(max(float(p), float(r[P_PMIN])), float(r[P_PMAX]))
        return float(r[P_G0]) * (pe / float(r[P_PREF])) ** float(r[P_NEXP])

    def wave_speed_at(self, mat_id: int, rho: float, p: float) -> float:
        """H29: dilatational wave speed at pressure p.  Because the modulus is
        frozen above P_PMAX, wave_speed_at(p_max) BOUNDS the wave speed of the
        whole simulation, which is what the CFL step is taken from."""
        r = self._params_np[mat_id]
        if float(r[P_PDEP]) < 0.5:
            return self.wave_speed(mat_id, rho)
        g = self.shear_modulus_at(mat_id, p)
        nu = float(r[P_NU])
        lam = 2.0 * g * nu / (1.0 - 2.0 * nu)
        return math.sqrt((lam + 2.0 * g) / rho)

    def particle_pressure(self) -> np.ndarray:
        """H29: the per-particle pressure written by the last g2p (Pa)."""
        return self.pp.numpy()

    def suggest_dt(self, mat_id: int, rho: float, cfl: float = 0.4) -> float:
        return cfl * self.dx / max(self.wave_speed(mat_id, rho), 1.0e-9)

    # ---------------- particles ----------------
    def add_particles(self, positions: np.ndarray, mat: int = MAT_COROTATED,
                      rho: float = 1000.0, velocity: Sequence[float] = (0.0, 0.0, 0.0),
                      ppc: int = 8) -> None:
        """Add particles.  Particle volume = cell volume / ppc."""
        if self._finalized:
            raise RuntimeError("add_particles() after finalize()")
        pos = np.ascontiguousarray(positions, dtype=np.float32)
        n = pos.shape[0]
        vol = (self.dx ** 3) / float(ppc)
        self._px.append(pos)
        self._pv.append(np.tile(np.array(velocity, dtype=np.float32), (n, 1)))
        self._pm.append(np.full(n, rho * vol, dtype=np.float32))
        self._pvol.append(np.full(n, vol, dtype=np.float32))
        self._pmat.append(np.full(n, mat, dtype=np.int32))
        self.n_particles += n

    def finalize(self) -> None:
        if self._finalized:
            return
        if self.n_particles == 0:
            raise RuntimeError("no particles")
        x = np.concatenate(self._px, axis=0)
        v = np.concatenate(self._pv, axis=0)
        m = np.concatenate(self._pm, axis=0)
        vol = np.concatenate(self._pvol, axis=0)
        mat = np.concatenate(self._pmat, axis=0)
        n = self.n_particles
        eye = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
        with wp.ScopedDevice(self.device):
            self.x = wp.array(x, dtype=wp.vec3)
            self.v = wp.array(v, dtype=wp.vec3)
            self.F = wp.array(eye, dtype=wp.mat33)
            self.C = wp.zeros(n, dtype=wp.mat33)
            self.mass = wp.array(m, dtype=wp.float32)
            self.vol0 = wp.array(vol, dtype=wp.float32)
            self.mat = wp.array(mat, dtype=wp.int32)
            self.Jp = wp.array(np.where(mat == MAT_SNOW, 1.0, 0.0).astype(np.float32),
                               dtype=wp.float32)
            # H28: accumulated plastic deviatoric strain (Klar 2016 hardening)
            self.qh = wp.zeros(n, dtype=wp.float32)
            # H29: per-particle pressure (previous step) and accumulated
            # plastic volumetric strain (positive = dilation)
            self.pp = wp.zeros(n, dtype=wp.float32)
            self.evp = wp.zeros(n, dtype=wp.float32)
            # reserved for M3 (codimensional)
            self.q = wp.zeros(n, dtype=wp.vec4)
            self.d = wp.zeros(n, dtype=wp.mat33)
        self._finalized = True

    # ---------------- steps ----------------
    def p2g(self, dt: Optional[float] = None, check_overflow: bool = False) -> None:
        if not self._finalized:
            self.finalize()
        dt = self.dt if dt is None else dt
        with wp.ScopedDevice(self.device):
            if self.scatter == "int64":
                # grid_m / grid_mv are fully overwritten by fixed_to_float,
                # so only the fixed-point accumulators need clearing.
                self.grid_mi.zero_()
                self.grid_mvi.zero_()
                wp.launch(p2g_int64_kernel, dim=self.n_particles, inputs=[
                    self.x, self.v, self.F, self.C, self.mass, self.vol0, self.mat,
                    self.Jp, self.pp, self.evp, self.params,
                    self.grid_mi, self.grid_mvi,
                    self.res, self.dx, self.inv_dx, self.origin, dt])
                wp.launch(fixed_to_float_kernel, dim=self.n_nodes, inputs=[
                    self.grid_mi, self.grid_mvi, self.grid_m, self.grid_mv])
            else:
                self.grid_m.zero_()
                self.grid_mv.zero_()
                wp.launch(p2g_float32_kernel, dim=self.n_particles, inputs=[
                    self.x, self.v, self.F, self.C, self.mass, self.vol0, self.mat,
                    self.Jp, self.pp, self.evp, self.params,
                    self.grid_m, self.grid_mv,
                    self.res, self.dx, self.inv_dx, self.origin, dt])
            if check_overflow:
                self._absmax.zero_()
                wp.launch(grid_absmax_kernel, dim=self.n_nodes,
                          inputs=[self.grid_m, self.grid_mv, self._absmax])
                mm, mmv = [float(a) for a in self._absmax.numpy()]
                if mm > MAX_NODE_MASS or mmv > MAX_NODE_MOMENTUM:
                    raise OverflowError(
                        f"fixed-point overflow risk: max node mass {mm:.3e} kg "
                        f"(limit {MAX_NODE_MASS:.3e}), max node momentum "
                        f"{mmv:.3e} kg m/s (limit {MAX_NODE_MOMENTUM:.3e})")

    def grid_update(self, contacts=None, dt: Optional[float] = None) -> None:
        """Momentum -> velocity, gravity, CONTACT HOOK, domain boundary.

        ``contacts``: None | callable(solver, dt) | obj.apply_grid(solver, dt)
        | list of those.  See the module docstring (CONTACT HOOK).
        """
        dt = self.dt if dt is None else dt
        with wp.ScopedDevice(self.device):
            # U10 (c): keep last step's grid velocity for the FLIP increment.
            wp.launch(grid_snapshot_kernel, dim=self.n_nodes, inputs=[
                self.grid_v, self.grid_v_old])
            wp.launch(grid_normalize_kernel, dim=self.n_nodes, inputs=[
                self.grid_m, self.grid_mv, self.grid_v, self.gravity, dt])
            # U10 (a): free-surface mass floor (off by default).
            if self._u10_mmode != 0:
                wp.launch(grid_floor_kernel, dim=self.n_nodes, inputs=[
                    self.grid_m, self.grid_mv, self.grid_v, self.gravity, dt,
                    self._u10_mfull, self._u10_meps, self._u10_mmode, self.res])
            if contacts is not None:
                hooks = contacts if isinstance(contacts, (list, tuple)) else [contacts]
                for h in hooks:
                    if hasattr(h, "apply_grid"):
                        h.apply_grid(self, dt)
                    elif callable(h):
                        h(self, dt)
                    else:
                        raise TypeError(
                            "contacts entry must be callable(solver, dt) or have "
                            "apply_grid(solver, dt)")
            wp.launch(grid_boundary_kernel, dim=self.n_nodes, inputs=[
                self.grid_m, self.grid_v, self.bc_mode, self.bc_mu,
                self.res, self.bound])

    def g2p(self, dt: Optional[float] = None) -> None:
        dt = self.dt if dt is None else dt
        with wp.ScopedDevice(self.device):
            wp.launch(g2p_kernel, dim=self.n_particles, inputs=[
                self.x, self.v, self.F, self.C, self.mat, self.Jp, self.qh,
                self.pp, self.evp, self.params,
                self.grid_v, self.grid_v_old, self.res, self.dx, self.inv_dx,
                self.origin, dt])

    def step(self, contacts=None, dt: Optional[float] = None,
             check_overflow: bool = False) -> None:
        self.p2g(dt, check_overflow=check_overflow)
        self.grid_update(contacts=contacts, dt=dt)
        self.g2p(dt)
        self.frame += 1

    def run(self, n_steps: int, contacts=None, dt: Optional[float] = None) -> None:
        for _ in range(n_steps):
            self.step(contacts=contacts, dt=dt)
        wp.synchronize_device(self.device)

    # ---------------- diagnostics ----------------
    def energies(self) -> dict:
        """Deterministic (int64 fixed point) energy reduction, in J."""
        with wp.ScopedDevice(self.device):
            self._e_acc.zero_()
            wp.launch(energy_kernel, dim=self.n_particles, inputs=[
                self.x, self.v, self.F, self.mass, self.vol0, self.mat, self.Jp,
                self.pp, self.evp, self.params, self.gravity, self._e_acc])
            a = self._e_acc.numpy()
        ke, pe, gpe = [float(t) / ENERGY_SCALE for t in a]
        return {"kinetic": ke, "elastic": pe, "gravity": gpe,
                "mech": ke + pe, "total": ke + pe + gpe}

    def state_bytes(self) -> bytes:
        """Raw bytes of the particle state (for the bit-identity test)."""
        parts = [self.x.numpy(), self.v.numpy(), self.F.numpy(),
                 self.C.numpy(), self.Jp.numpy(), self.qh.numpy(),
                 self.pp.numpy(), self.evp.numpy()]
        return b"".join(np.ascontiguousarray(p).tobytes() for p in parts)

    def state_bytes_h28(self) -> bytes:
        """H29: the H28 byte layout -- x, v, F, C, Jp, qh (no pp/evp)."""
        parts = [self.x.numpy(), self.v.numpy(), self.F.numpy(),
                 self.C.numpy(), self.Jp.numpy(), self.qh.numpy()]
        return b"".join(np.ascontiguousarray(p).tobytes() for p in parts)

    def state_bytes_m1(self) -> bytes:
        """H28: the ORIGINAL (the generic core) byte layout -- x, v, F, C, Jp, no qh.
        Used to show that the extended core is bit-identical to the original
        one at default parameters (beta = 0, hardening off)."""
        parts = [self.x.numpy(), self.v.numpy(), self.F.numpy(),
                 self.C.numpy(), self.Jp.numpy()]
        return b"".join(np.ascontiguousarray(p).tobytes() for p in parts)

    def free(self) -> None:
        """Release device memory (shared GPU: call before exit)."""
        for name in ("x", "v", "F", "C", "mass", "vol0", "mat", "Jp", "qh",
                      "pp", "evp", "q", "d",
                      "grid_m", "grid_mv", "grid_v", "grid_v_old",
                      "grid_mi", "grid_mvi",
                      "params", "bc_mode", "bc_mu", "_e_acc", "_absmax"):
            if hasattr(self, name):
                setattr(self, name, None)
        wp.synchronize_device(self.device)


# ==========================================================================
# sampling helpers (deterministic)
# ==========================================================================
def sample_box(lower: Sequence[float], upper: Sequence[float], dx: float,
               ppc: int = 8, seed: int = 0, jitter: float = 0.0) -> np.ndarray:
    """Regular ppc-per-cell lattice inside a box.  ppc must be a perfect cube
    (1, 8, 27) unless jitter sampling is used."""
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    k = round(ppc ** (1.0 / 3.0))
    if k ** 3 != ppc:
        raise ValueError("ppc must be a perfect cube for sample_box")
    step = dx / k
    axes = [np.arange(lo[i] + 0.5 * step, hi[i] - 1e-12, step) for i in range(3)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        g = g + rng.uniform(-jitter * step, jitter * step, size=g.shape)
    return np.ascontiguousarray(g, dtype=np.float32)


def sample_sphere(center: Sequence[float], radius: float, dx: float,
                  ppc: int = 8, seed: int = 0, jitter: float = 0.0) -> np.ndarray:
    c = np.asarray(center, dtype=np.float64)
    box = sample_box(c - radius, c + radius, dx, ppc=ppc, seed=seed, jitter=jitter)
    r = np.linalg.norm(box.astype(np.float64) - c, axis=1)
    return np.ascontiguousarray(box[r <= radius], dtype=np.float32)


def gpu_mem_used_mib() -> float:
    import subprocess
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    return float(out)
