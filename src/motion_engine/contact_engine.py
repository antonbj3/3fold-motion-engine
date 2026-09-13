#!/usr/bin/env python3
"""Hot-swappable rigid-body contact dynamics behind one data-oriented Protocol.

Distinct from WorldModel (the collision QUERY: sd/grad/free); this is contact RESOLUTION: gravity +
contact + Coulomb friction -> rigid-body motion (Newton-Euler). Same hot-swap philosophy as world_model:
the contract is data in, data out (numpy arrays, poses) and never leaks engine-internal objects, so
another backend (GPU, Rust, C++) drops in behind the same signature.

Reference backend: SplitImpulseEngine — Catto split impulse (velocity pass with accumulated normal impulse
jn and tangential impulse jt capped at mu*jn along the slip tangent) plus a SEPARATE pseudo-velocity
position pass (Baumgarte outside the velocity solve, so no bounce or friction leak). Grid-hash broad phase.
Validated 8/8 on stick/slide at atan(mu), deep-stack stable, contact forces = Mg*{cos(theta), sin(theta)}
to 0% error. Two decisive details: the torque lever cross(r - com, J), and using the SLIP-DIRECTION tangent
(a gravity-derived tangent degenerates for horizontal contact and gives a lateral-load bug).

A GPU backend implements the same Protocol (see contact_engine_gpu.py).

  python -u -m motion_engine.contact_engine        # selftest (ramp stick/slide + stack + contact forces)
"""
import math
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np

G = 9.81
R = 0.05            # corner-particle radius (half the voxel resolution)
BETA = 0.2          # Baumgarte position-bias
SLOP = 2e-4         # allowed penetration (no bias below it)

# TGS soft-step (opt-in). make_soft/bias_velocity = the adopted contact softness.
try:                                                              # package import
    from .soft_step import make_soft, bias_velocity, contact_hertz, CONTACT_DAMPING_RATIO, CONTACT_SPEED
except ImportError:                                               # direct-script execution (selftest)
    from soft_step import make_soft, bias_velocity, contact_hertz, CONTACT_DAMPING_RATIO, CONTACT_SPEED


# ───────────────────────── data contract (no engine internals) ─────────────────────────
@dataclass
class BodyState:
    """Rigid-body state as plain data. xc=(N,3) com, Rm=(N,3,3) orientation, vc=(N,3) linear velocity, om=(N,3) angular velocity."""
    xc: np.ndarray
    Rm: np.ndarray
    vc: np.ndarray
    om: np.ndarray


@runtime_checkable
class ContactEngine(Protocol):
    """Contract surface (the consumer never sees an implementation). Data in, data out, so the backend is interchangeable (CPU / Warp GPU / Rust / ...)."""
    name: str
    def step(self, dt: float, substeps: int = 1) -> None: ...     # advance one time step under gravity + contact + friction
    def get_state(self) -> BodyState: ...                         # current state as data
    def set_kinematic(self, i: int, xc) -> None: ...              # force body i's com (actuation / grasp)
    def contact_forces(self) -> np.ndarray: ...                   # (N,3) net contact force per body (sum of impulses / dt): load and wear


# ───────────────────────── geometry helpers ─────────────────────────
def _voxbox(w, h, d, res=R * 2):
    a = lambda L: np.arange(-L / 2 + res / 2, L / 2, res)
    return np.array([[x, y, z] for z in a(d) for y in a(h) for x in a(w)])

def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def _rodr(w, dt):
    th = np.linalg.norm(w) * dt
    if th < 1e-9:
        return np.eye(3)
    k = w / (np.linalg.norm(w) + 1e-12); K = _skew(k)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


class _Body:
    def __init__(s, w, h, d, c, density=700., kin=False):
        s.rest = _voxbox(w, h, d); n = len(s.rest)
        s.mp = density * w * h * d / n; s.M = s.mp * n
        s.Ib = sum(s.mp * ((r @ r) * np.eye(3) - np.outer(r, r)) for r in s.rest)
        s.xc = np.array(c, float); s.Rm = np.eye(3); s.vc = np.zeros(3); s.om = np.zeros(3); s.kin = kin
    def pw(s):
        return s.rest @ s.Rm.T + s.xc
    def Iwi(s):
        # np.linalg.inv does not sanitise NaN: it returns a partially-NaN inverse for a degenerate or corrupt
        # inertia tensor instead of raising (an exactly-zero Ib raises "Singular matrix", a NaN-corrupt one does
        # not). Iwi() feeds the angular-velocity update in the impulse loop repeatedly, so a corrupt inverse would
        # silently poison the whole simulation with NaN.
        Iw = np.linalg.inv(s.Rm @ s.Ib @ s.Rm.T)
        if not np.all(np.isfinite(Iw)):
            raise ValueError("Iwi: world-frame inverse inertia is non-finite (NaN/inf) -- degenerate body "
                             "inertia tensor; refusing to propagate a corrupted angular-velocity update")
        return Iw


class SplitImpulseEngine:
    """Reference CPU backend: Catto split impulse + accumulated friction + a separate position pass.
    Implements ContactEngine. `soft=True` swaps the fixed-Baumgarte position pass for the TGS soft-step
    path of soft_step.py (opt-in; default off, so the default behaviour is unchanged)."""
    name = "split_impulse_cpu"

    def __init__(s, ramp_deg=0., mu=0.5, vel_iters=10, pos_iters=6, soft=False, soft_substeps=4, soft_hertz=30.0):
        th = np.radians(ramp_deg)
        s.n = np.array([-np.sin(th), 0, np.cos(th)]); s.mu = mu
        s.VIT = vel_iters; s.PIT = pos_iters; s.B = []
        s.soft = soft                                             # TGS soft-step contact solve (opt-in)
        s.soft_substeps = soft_substeps                           # the soft solve buys stack accuracy from substeps, not inner iterations
        s.soft_hertz = soft_hertz                                 # world contact stiffness, clamped to 0.125*inv_h for stability
        s._cimp = {}                                              # accumulated contact impulse per body (for contact_forces)
        s._warm = {}                                              # warm start: contact key -> (jn, jt), carried across steps
        s._last_dt = None

    # ── konstruktion ──
    def add_body(s, w, h, d, c, density=700., kin=False):
        b = _Body(w, h, d, c, density, kin); s.B.append(b); return len(s.B) - 1

    def set_kinematic(s, i, xc):
        s.B[i].kin = True; s.B[i].xc = np.array(xc, float)

    # ── DATA-ut ──
    def get_state(s):
        return BodyState(xc=np.array([b.xc for b in s.B]), Rm=np.array([b.Rm for b in s.B]),
                         vc=np.array([b.vc for b in s.B]), om=np.array([b.om for b in s.B]))

    def contact_forces(s):
        dt = s._last_dt or 1.0
        return np.array([s._cimp.get(id(b), np.zeros(3)) / dt for b in s.B])

    def com(s, i):
        return s.B[i].xc

    # ── intern ──
    def _apply(s, b, J, r):                                       # torque = cross(LEVER=r−com, J) (lever-fixen)
        if b.kin:
            return
        b.vc = b.vc + J / b.M; b.om = b.om + b.Iwi() @ np.cross(r - b.xc, J)
        s._cimp[id(b)] = s._cimp.get(id(b), np.zeros(3)) + J     # accumulate the net contact impulse

    def _vel_at(s, b, pwi):
        return np.zeros(3) if b.kin else b.vc + np.cross(b.om, pwi - b.xc)

    def _collect(s):
        C = []; PW = [b.pw() for b in s.B]
        # A non-finite body position (e.g. NaN xc/Rm cascaded from a corrupt velocity through position integration)
        # is silenced by two independent mechanisms if not checked: (1) the ramp contact filter `sg < 0` is False for
        # a NaN sg, so a NaN corner is silently excluded from ground contact; (2) `astype(np.int64)` on a NaN cell
        # coordinate casts silently to a garbage int. The net effect would be a corrupt body falling through the
        # ground with no contact and no error, with contact_forces() reporting a clean [0,0,0]. Fail closed instead.
        if not np.all(np.isfinite(np.vstack(PW))):
            raise ValueError("SplitImpulseEngine._collect: non-finite body world-point(s) -- "
                             "corrupted body state, refusing to silently drop it from contact detection")
        for bi, b in enumerate(s.B):                              # MARK/ramp
            if b.kin:
                continue
            sg = PW[bi] @ s.n - R
            for i in np.where(sg < 0)[0]:
                C.append([b, None, PW[bi][i], None, s.n, -sg[i]])
        allp = np.vstack(PW); owner = np.concatenate([np.full(len(PW[bi]), bi) for bi in range(len(s.B))])
        cell = np.floor(allp / (2 * R)).astype(np.int64); grid = {}
        for idx in range(len(allp)):
            grid.setdefault((cell[idx, 0], cell[idx, 1], cell[idx, 2]), []).append(idx)
        offs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
        for idx in range(len(allp)):
            bi = owner[idx]; ci = cell[idx]
            for do in offs:
                for jdx in grid.get((ci[0] + do[0], ci[1] + do[1], ci[2] + do[2]), []):
                    if jdx <= idx:
                        continue
                    bj = owner[jdx]
                    if bi == bj or (s.B[bi].kin and s.B[bj].kin):
                        continue
                    d = allp[idx] - allp[jdx]; dist = np.linalg.norm(d) + 1e-12
                    if dist < 2 * R:
                        C.append([s.B[bi], s.B[bj], allp[idx], allp[jdx], d / dist, 2 * R - dist])
        return C

    def _tang(s, c):                                              # slip-direction tangent (not gravity-derived, so not degenerate)
        A, B, rA, rB, nrm, pen = c
        vr = s._vel_at(A, rA) - (s._vel_at(B, rB) if B else 0)
        vt = vr - (vr @ nrm) * nrm; m = np.linalg.norm(vt)
        if m > 5e-3:
            return vt / m
        a = np.array([1., 0, 0]) if abs(nrm[0]) < 0.9 else np.array([0, 1., 0])
        t = a - (a @ nrm) * nrm; return t / (np.linalg.norm(t) + 1e-12)

    def _soft_solve(s, C, sdt):
        """TGS soft-step substep (opt-in, default off): warm start -> one biased soft velocity
        solve -> integrate positions -> one relax solve. The softness comes from soft_step.make_soft;
        bias_velocity supplies (velocity_bias, mass_scale, impulse_scale) per contact, with the
        separation passed as -penetration.

        Two properties are load bearing and both are measured, not assumed:
        - ONE biased plus ONE relax sweep per substep. Inner Gauss-Seidel sweeps re-apply
          impulse_scale and break the mass_scale + impulse_scale == 1 partition the primitive
          guarantees, which makes the result non-monotonic in vel_iters. Accuracy comes from the
          substep count and from the cross-step warm start instead.
        - CROSS-STEP WARM START is what makes load propagation converge at vel_iters=10 instead of
          40: each contact is matched to the previous step by a body-local anchor (Rm^T (r - xc),
          the rest-frame particle position, identical every step for a rigid body) and its converged
          impulse is re-applied before the solve.
        Static and kinematic contacts use a stiffer softness (twice the hertz, half the damping ratio)
        than body-body contacts."""
        inv_h = 1.0 / sdt
        ch = contact_hertz(inv_h, s.soft_hertz)                   # clamped to the 0.125*inv_h stability limit
        soft = make_soft(ch, CONTACT_DAMPING_RATIO, sdt)
        static_soft = make_soft(2.0 * ch, 0.5 * CONTACT_DAMPING_RATIO, sdt)
        jn = [0.] * len(C); jt = [0.] * len(C); tang = [s._tang(c) for c in C]

        def _key(X, r):
            return None if X.kin else (id(X), tuple(np.round(X.Rm.T @ (r - X.xc), 6)))
        keys = []
        for ci, c in enumerate(C):
            A, B, rA, rB, nrm, pen = c
            key = (_key(A, rA), _key(B, rB) if B is not None else None)
            keys.append(key)
            prev = s._warm.get(key)
            if prev is not None:
                jn[ci], jt[ci] = prev
                P = jn[ci] * nrm + jt[ci] * tang[ci]              # re-apply the stored impulse (warm start)
                s._apply(A, P, rA)
                if B:
                    s._apply(B, -P, rB)

        def _meffn(ci):
            A, B, rA, rB, nrm, pen = C[ci]
            return (0 if A.kin else 1 / A.M + np.cross(rA - A.xc, nrm) @ (A.Iwi() @ np.cross(rA - A.xc, nrm))) + \
                   (0 if (B is None or B.kin) else 1 / B.M + np.cross(rB - B.xc, nrm) @ (B.Iwi() @ np.cross(rB - B.xc, nrm)))

        def _norm_solve(ci, meff, use_bias):
            A, B, rA, rB, nrm, pen = C[ci]
            vrel = s._vel_at(A, rA) - (s._vel_at(B, rB) if B else 0); vn = vrel @ nrm
            sf = static_soft if (B is None or B.kin) else soft    # static contact is stiffer
            vb, ms, isc = bias_velocity(-pen, sf, use_bias, inv_h)
            impulse = -(ms * vn + vb) / meff - isc * jn[ci]       # soft normal solve
            if not math.isfinite(impulse):
                raise ValueError("SplitImpulseEngine: non-finite soft normal impulse increment -- "
                                 "corrupted contact/body state, refusing to silently no-op it")
            nw = max(0., jn[ci] + impulse); dj = nw - jn[ci]; jn[ci] = nw
            s._apply(A, dj * nrm, rA)
            if B:
                s._apply(B, -dj * nrm, rB)

        def _fric_solve(ci, meff):
            A, B, rA, rB, nrm, pen = C[ci]
            vrel = s._vel_at(A, rA) - (s._vel_at(B, rB) if B else 0); t = tang[ci]; vt = vrel @ t
            djt = -vt / meff
            if not math.isfinite(djt):
                raise ValueError("SplitImpulseEngine: non-finite soft tangential impulse increment -- "
                                 "corrupted contact/body state, refusing to silently no-op it")
            bud = s.mu * jn[ci]
            nwt = max(-bud, min(bud, jt[ci] + djt)); djt = nwt - jt[ci]; jt[ci] = nwt
            s._apply(A, djt * t, rA)
            if B:
                s._apply(B, -djt * t, rB)

        def solve(use_bias, iters):
            for _ in range(iters):
                for ci in range(len(C)):
                    meff = _meffn(ci)
                    if meff < 1e-12:
                        continue
                    _norm_solve(ci, meff, use_bias)
                    _fric_solve(ci, meff)

        solve(True, 1)                                            # biased soft solve (pushes penetration out)
        for b in s.B:                                             # integrate positions with the real velocities
            if not b.kin:
                b.xc = b.xc + b.vc * sdt; b.Rm = _rodr(b.om, sdt) @ b.Rm
        solve(False, 1)                                           # relax: remove the bias energy
        s._warm = {keys[ci]: (jn[ci], jt[ci]) for ci in range(len(C))}   # store impulses for the next step

    def step(s, dt, substeps=1, control=None, t0=0.):
        s._last_dt = dt / substeps; s._cimp = {}
        # the soft path buys stack accuracy from the substep count, not from inner iterations, so it
        # runs at least soft_substeps internal substeps regardless of the caller
        substeps = max(substeps, s.soft_substeps) if s.soft else substeps
        for ss in range(substeps):
            sdt = dt / substeps
            if control:
                control(s, t0 + ss * sdt)
            for b in s.B:
                if not b.kin:
                    b.vc = b.vc + np.array([0, 0, -G]) * sdt
            C = s._collect()
            if not C:
                for b in s.B:
                    if not b.kin:
                        b.xc = b.xc + b.vc * sdt; b.Rm = _rodr(b.om, sdt) @ b.Rm
                continue
            if s.soft:                                            # TGS soft-step path
                s._soft_solve(C, sdt)
                continue
            jn = [0.] * len(C); jt = [0.] * len(C); tang = [s._tang(c) for c in C]
            for _ in range(s.VIT):                                # VELOCITY-pass
                for ci, c in enumerate(C):
                    A, B, rA, rB, nrm, pen = c; vrel = s._vel_at(A, rA) - (s._vel_at(B, rB) if B else 0)
                    meff = (0 if A.kin else 1 / A.M + np.cross(rA - A.xc, nrm) @ (A.Iwi() @ np.cross(rA - A.xc, nrm))) + \
                           (0 if (B is None or B.kin) else 1 / B.M + np.cross(rB - B.xc, nrm) @ (B.Iwi() @ np.cross(rB - B.xc, nrm)))
                    if meff < 1e-12:
                        continue
                    vn = vrel @ nrm; dj = -vn / meff
                    # A non-finite dj (from an upstream corrupt body velocity) would make `max(0, jn[ci]+dj)`
                    # silently return the previous jn[ci] unchanged (Python's max never picks NaN), so
                    # contact_forces() would report a clean finite no-op impulse for a genuinely corrupt contact.
                    # Fail closed.
                    if not math.isfinite(dj):
                        raise ValueError("SplitImpulseEngine: non-finite normal impulse increment -- "
                                         "corrupted contact/body state, refusing to silently no-op it")
                    nw = max(0, jn[ci] + dj); dj = nw - jn[ci]; jn[ci] = nw
                    s._apply(A, dj * nrm, rA)
                    if B:
                        s._apply(B, -dj * nrm, rB)
                    vrel = s._vel_at(A, rA) - (s._vel_at(B, rB) if B else 0); t = tang[ci]; vt = vrel @ t
                    djt = -vt / meff
                    if not math.isfinite(djt):
                        raise ValueError("SplitImpulseEngine: non-finite tangential impulse increment -- "
                                         "corrupted contact/body state, refusing to silently no-op it")
                    nwt = max(-s.mu * jn[ci], min(s.mu * jn[ci], jt[ci] + djt)); djt = nwt - jt[ci]; jt[ci] = nwt
                    s._apply(A, djt * t, rA)
                    if B:
                        s._apply(B, -djt * t, rB)
            # position pass: separate pseudo-velocity (Catto), resolving penetration without touching the real velocity
            pv = {id(b): np.zeros(3) for b in s.B}; po = {id(b): np.zeros(3) for b in s.B}; jp = [0.] * len(C)
            def pvel(b, r):
                return np.zeros(3) if b.kin else pv[id(b)] + np.cross(po[id(b)], r - b.xc)
            for _ in range(s.PIT):
                for ci, c in enumerate(C):
                    A, B, rA, rB, nrm, pen = c
                    meff = (0 if A.kin else 1 / A.M + np.cross(rA - A.xc, nrm) @ (A.Iwi() @ np.cross(rA - A.xc, nrm))) + \
                           (0 if (B is None or B.kin) else 1 / B.M + np.cross(rB - B.xc, nrm) @ (B.Iwi() @ np.cross(rB - B.xc, nrm)))
                    if meff < 1e-12:
                        continue
                    rel = pvel(A, rA) - (pvel(B, rB) if B is not None else 0); bias = BETA * max(pen - SLOP, 0) / sdt
                    dj = (bias - rel @ nrm) / meff
                    # same guard as the velocity pass above: fail closed on a non-finite position-bias impulse
                    if not math.isfinite(dj):
                        raise ValueError("SplitImpulseEngine: non-finite position-pass impulse increment -- "
                                         "corrupted contact/body state, refusing to silently no-op it")
                    nw = max(0, jp[ci] + dj); dj = nw - jp[ci]; jp[ci] = nw
                    if not A.kin:
                        pv[id(A)] = pv[id(A)] + dj * nrm / A.M; po[id(A)] = po[id(A)] + A.Iwi() @ np.cross(rA - A.xc, dj * nrm)
                    if B is not None and not B.kin:
                        pv[id(B)] = pv[id(B)] - dj * nrm / B.M; po[id(B)] = po[id(B)] - B.Iwi() @ np.cross(rB - B.xc, dj * nrm)
            for b in s.B:
                if not b.kin:
                    b.xc = b.xc + (b.vc + pv[id(b)]) * sdt; b.Rm = _rodr(b.om + po[id(b)], sdt) @ b.Rm


# ───────────────────────── selftest (contract gate) ─────────────────────────
def _selftest():
    assert isinstance(SplitImpulseEngine(), ContactEngine), "does not satisfy the ContactEngine Protocol"
    ok = True
    # 1) ramp stick/slide = atan(μ)
    for deg, exp in [(20, "stick"), (35, "slide")]:
        e = SplitImpulseEngine(ramp_deg=deg, mu=0.5); e.add_body(0.3, 0.2, 0.2, [0, 0, 0.101])
        x0 = e.com(0).copy()
        for _ in range(400):
            e.step(1 / 240, substeps=2)
        nn = e.n; down = np.array([-np.cos(np.radians(deg)), 0, -np.sin(np.radians(deg))])
        slid = (e.com(0) - x0) @ down; got = "slide" if slid > 0.02 else "stick"
        print(f"  [ramp {deg} deg] {got} (expected {exp}) slid={slid:+.3f}  {'ok' if got == exp else 'FAIL'}"); ok = ok and got == exp
    # 2) stack of K=2 stable
    e = SplitImpulseEngine(mu=0.5)
    for k in range(2):
        e.add_body(0.3, 0.3, 0.2, [0, 0, 0.101 + k * 0.205])
    for _ in range(300):
        e.step(1 / 240, substeps=2)
    st = e.get_state(); ke = sum(0.5 * b.M * b.vc @ b.vc for b in e.B)
    z_ok = st.xc[1, 2] > st.xc[0, 2] + 0.15
    print(f"  [stack K=2] KE={ke:.2e} sep={st.xc[1,2]-st.xc[0,2]:.3f}  {'stable' if ke < 1e-2 and z_ok else 'FAIL'}"); ok = ok and ke < 1e-2 and z_ok
    # 3) contact force = Mg on flat ground
    e = SplitImpulseEngine(mu=0.8); e.add_body(0.3, 0.2, 0.2, [0, 0, 0.101])
    for _ in range(300):
        e.step(1 / 240, substeps=1)
    Fz = e.contact_forces()[0, 2]; Mg = e.B[0].M * G; err = abs(Fz - Mg) / Mg * 100
    print(f"  [contact force, flat] Fz={Fz:.2f} Mg={Mg:.2f} error={err:.1f}%  {'ok' if err < 3 else 'FAIL'}"); ok = ok and err < 3
    # 4) a kinematic grip lifts the box (friction carries it)
    e = SplitImpulseEngine(mu=1.0, vel_iters=20)
    box = e.add_body(0.2, 0.2, 0.2, [0, 0, 0.3]); f1 = e.add_body(0.06, 0.2, 0.2, [-0.145, 0, 0.3], kin=True); f2 = e.add_body(0.06, 0.2, 0.2, [0.145, 0, 0.3], kin=True)
    def grip(s, t):
        s.set_kinematic(f1, [-0.118, 0, 0.3 + max(0, t - 0.3) * 0.3]); s.set_kinematic(f2, [0.118, 0, 0.3 + max(0, t - 0.3) * 0.3])
    for st_ in range(400):
        e.step(1 / 240, substeps=4, control=grip, t0=st_ / 240)
    lifted = e.com(box)[2] > 0.33
    print(f"  [grip lift] box_z={e.com(box)[2]:.3f}  {'lifted by friction' if lifted else 'FAIL'}"); ok = ok and lifted
    print(f"  -> {'ContactEngine: all checks pass, contract gate green' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    print("contact-engine selftest (SplitImpulseEngine behind the ContactEngine Protocol):")
    sys.exit(0 if _selftest() else 1)
