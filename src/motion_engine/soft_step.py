"""TGS soft-step constraint softness: the contact-softness primitive of a temporal
Gauss-Seidel contact solver, isolated so the math can be machine-checked on its own.

Math adopted (cited, re-derived, not lifted) from box3d v0.1.0 / box2d v3.2.0 TGS-soft:
  - MakeSoft:            box3d solver.h:264-306  (== box2d solver.h:239-281)
  - bias-velocity clamp: box3d contact_solver.c:429-452 (box2d :300-323)
  - hertz clamp:         box3d physics_world.c:1114-1116

API: `make_soft(hertz, zeta, h) -> Softness(bias_rate, mass_scale, impulse_scale)`,
`contact_hertz(inv_h, world_hertz)` (stiffness capped at a quarter of the substep Nyquist)
and `bias_velocity(separation, soft, use_bias, inv_h) -> (velocity_bias, mass_scale,
impulse_scale)` for the three solver passes: speculative (separation > 0), biased
penetrating (clamped at -contact_speed) and relax (no bias, unit scales).
`python -m motion_engine.soft_step` runs the invariant selftest; exit 0 on pass.

This module is a standalone primitive here: no solver in this repository calls it. See
docs/RUNNING.md for the three call sites a contact solver needs to wire it in.
"""
from dataclasses import dataclass
import math

# box3d/box2d defaults (types.c:19-21 / types.c:18-20, identical)
CONTACT_HERTZ = 30.0        # Hz
CONTACT_DAMPING_RATIO = 10.0   # ζ
CONTACT_SPEED = 3.0         # ·lengthUnits (max push-out speed / bias clamp)


@dataclass(frozen=True)
class Softness:
    bias_rate: float
    mass_scale: float
    impulse_scale: float


def make_soft(hertz: float, zeta: float, h: float) -> Softness:
    """box3d solver.h:264-306. Discrete implicit spring (Baumgarte is the ζ→∞
    degenerate limit). Guarantees mass_scale + impulse_scale == 1 (I-2)."""
    if hertz <= 0.0:                       # solver.h:266 short-circuit = rigid / no bias
        return Softness(0.0, 0.0, 0.0)
    omega = 2.0 * math.pi * hertz
    a1 = 2.0 * zeta + h * omega
    a2 = h * omega * a1
    a3 = 1.0 / (1.0 + a2)
    return Softness(bias_rate=omega / a1, mass_scale=a2 * a3, impulse_scale=a3)


def contact_hertz(inv_h: float, world_hertz: float = CONTACT_HERTZ) -> float:
    """physics_world.c:1114 — never stiffer than ¼ of the substep Nyquist."""
    return min(world_hertz, 0.125 * inv_h)


def bias_velocity(separation: float, soft: Softness, use_bias: bool, inv_h: float,
                  contact_speed: float = CONTACT_SPEED):
    """box3d contact_solver.c:429-452. Returns (velocity_bias, mass_scale, impulse_scale).
      separation > 0 : speculative — pull to contact at inv_h, NO soft scaling (I-3).
      separation < 0 & use_bias : penetrating biased pass — soft push-out, CLAMPED to
                                  -contact_speed so deep overlaps don't explode.
      relax pass (use_bias False): no bias, unit scales."""
    if separation > 0.0:                                   # I-3 speculative sign
        return separation * inv_h, 1.0, 0.0
    if use_bias:
        vb = max(soft.mass_scale * soft.bias_rate * separation, -contact_speed)  # THE CLAMP
        return vb, soft.mass_scale, soft.impulse_scale
    return 0.0, 1.0, 0.0                                    # relax


# ───────────────────────── invariant self-tests (I-2, I-3, clamp) ─────────────
def _selftest() -> bool:
    ok = True
    # I-2: mass_scale + impulse_scale == 1 across a wide (hertz, ζ, h) grid
    for hz in (5, 15, 30, 60, 120):
        for z in (1.0, 5.0, 10.0):
            for h in (1/60, 1/240, 1/960):
                s = make_soft(hz, z, h)
                d = abs(s.mass_scale + s.impulse_scale - 1.0)
                if d > 1e-12:
                    print(f"  I-2 FAIL hz={hz} z={z} h={h}: sum-1={d:.2e}"); ok = False
    print(f"  [I-2 mass_scale+impulse_scale==1] max-dev over grid checked  {'✓' if ok else '✗'}")

    # hertz==0 short-circuits to rigid (0,0,0)
    r = make_soft(0.0, 10.0, 1/240)
    z_ok = (r.bias_rate, r.mass_scale, r.impulse_scale) == (0.0, 0.0, 0.0)
    print(f"  [hertz=0 ⇒ rigid (0,0,0)] {'✓' if z_ok else '✗'}"); ok = ok and z_ok

    # I-3: speculative (s>0) uses inv_h, unit mass, zero impulse-scale (no spring pre-load)
    inv_h = 240.0; soft = make_soft(contact_hertz(inv_h), CONTACT_DAMPING_RATIO, 1/inv_h)
    vb, ms, is_ = bias_velocity(0.01, soft, True, inv_h)
    spec_ok = abs(vb - 0.01*inv_h) < 1e-12 and ms == 1.0 and is_ == 0.0
    print(f"  [I-3 speculative s>0 uses inv_h] vb={vb:.3f}=={0.01*inv_h}  {'✓' if spec_ok else '✗'}"); ok = ok and spec_ok

    # clamp: a deep penetration's push-out speed is capped at contact_speed
    vb_deep, _, _ = bias_velocity(-10.0, soft, True, inv_h)     # absurdly deep
    clamp_ok = abs(vb_deep + CONTACT_SPEED) < 1e-12             # == -contact_speed
    print(f"  [bias clamp @ -contact_speed] vb(deep)={vb_deep:.3f}=={-CONTACT_SPEED}  {'✓' if clamp_ok else '✗'}"); ok = ok and clamp_ok

    # relax pass: no bias, unit scales
    vb_r, ms_r, is_r = bias_velocity(-0.05, soft, False, inv_h)
    relax_ok = vb_r == 0.0 and ms_r == 1.0 and is_r == 0.0
    print(f"  [relax pass no-bias] {'✓' if relax_ok else '✗'}"); ok = ok and relax_ok

    # shallow penetration (unclamped): monotone in hertz (stiffer spring → faster push-out)
    push = []
    for hz in (15, 30, 60):
        s = make_soft(contact_hertz(inv_h, hz), CONTACT_DAMPING_RATIO, 1/inv_h)
        vb_s, _, _ = bias_velocity(-1e-3, s, True, inv_h)
        push.append(-vb_s)
    mono_ok = push[0] <= push[1] <= push[2]
    print(f"  [push-out monotone ↑ in hertz] {[round(p,4) for p in push]}  {'✓' if mono_ok else '✗'}"); ok = ok and mono_ok

    print(f"  → {'✓✓ soft-step primitive: invariants I-2/I-3 + clamp verified' if ok else '✗ invariant regression'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
