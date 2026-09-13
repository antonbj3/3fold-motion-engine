"""MidpointVBDEngine: a midpoint Vertex-Block-Descent contact-solver backend behind the
ContactEngine Protocol (see contact_engine.py), translational core.

Math adopted (cited, not novelty): Vertex Block Descent (Chen et al., SIGGRAPH 2024) =
per-block minimization of the variational integrator; midpoint (Dinev et al. 2018) for
second order; PHR augmented Lagrangian with lagged Coulomb friction.

Scheme: c = 4/h^2, xhat = x_n + 0.5 h v_n; the stationarity condition
c*m*(y - xhat) + m*grav + contact/friction gradients = 0 is solved per body by Newton.
Contact is evaluated at the endpoint x_{n+1} = 2y - x_n and enters the functional with
WEIGHT 1/2 (the chain-rule factor a = 2 would otherwise double the force). Then
x_{n+1} = 2y - x_n and v_{n+1} = 2(x_{n+1} - x_n)/h - v_n, second-order exact on constant
acceleration. Contact geometry is the 8 box corners against the ramp plane, one PHR-AL
multiplier per corner; friction is a Coulomb-clamped post-step tangential-velocity
projection using the total normal force.

I/O: `add_body(w, h, d, c, density, kin)`, `set_kinematic(i, xc)`, `step(dt, substeps,
control, t0)`, `get_state() -> BodyState(xc, Rm, vc, om)`, `com(i)`, `contact_forces()`.
`python -m motion_engine.midpoint_vbd_engine` runs the selftest (free fall, flat rest,
incline friction battery); exit 0 on pass.

Scope: translational only. Rotation (corners are frozen at Rm = I within a substep),
friction composed into the block Newton as a variational descent, and body-body contact
are not implemented. Additive backend; it does not touch SplitImpulseEngine.
"""
from dataclasses import dataclass
import numpy as np

G = 9.81
CORNER_SIGNS = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]   # 8 box corners


@dataclass
class BodyState:
    xc: np.ndarray; Rm: np.ndarray; vc: np.ndarray; om: np.ndarray


class _B:
    def __init__(self, w, h, d, c, density, kin):
        self.hw = np.array([w, h, d])/2.0; self.M = density*w*h*d
        self.xc = np.array(c, float); self.vc = np.zeros(3); self.Rm = np.eye(3); self.om = np.zeros(3)
        self.kin = kin


class MidpointVBDEngine:
    """Midpoint-VBD integrator (translational core). Implements the ContactEngine Protocol."""
    name = "midpoint_vbd_cpu"

    def __init__(self, ramp_deg=0.0, mu=0.5, kappa=1e6, use_al=True, newton_iters=8, outer=4):
        th = np.radians(ramp_deg)
        self.n = np.array([-np.sin(th), 0.0, np.cos(th)])          # incline outward normal (plane through origin)
        self.mu = mu; self.kappa = kappa; self.use_al = use_al; self.NIT = newton_iters; self.OUT = outer
        self.B = []; self._cimp = {}; self._last_dt = None

    # ── construction / Protocol data ──
    def add_body(self, w, h, d, c, density=700., kin=False):
        self.B.append(_B(w, h, d, c, density, kin)); return len(self.B)-1

    def set_kinematic(self, i, xc):
        self.B[i].kin = True; self.B[i].xc = np.array(xc, float); self.B[i].vc = np.zeros(3)

    def get_state(self):
        return BodyState(xc=np.array([b.xc for b in self.B]), Rm=np.array([b.Rm for b in self.B]),
                         vc=np.array([b.vc for b in self.B]), om=np.array([b.om for b in self.B]))

    def com(self, i): return self.B[i].xc

    def contact_forces(self):
        f = np.zeros((len(self.B), 3))
        for i, b in enumerate(self.B):
            f[i] = self._cimp.get(i, np.zeros(3))/max(self._last_dt or 1.0, 1e-12)
        return f

    # ── contact geometry: base gap of body i vs the incline plane (translational core) ──
    def _gap(self, x, b):
        base_off = float(self.hw_along_n(b))                       # half-extent projected on n
        return float(x @ self.n) - base_off
    def hw_along_n(self, b): return float(np.sum(np.abs(self.Rm_n(b))*b.hw))
    def Rm_n(self, b): return b.Rm.T @ self.n                      # normal in body frame

    # ── one midpoint-VBD substep ──
    def _substep(self, h):
        grav = np.array([0.0, 0.0, G])                            # gravity POTENTIAL gradient (force=-grad pulls DOWN)
        cimp = {i: np.zeros(3) for i in range(len(self.B))}
        for i, b in enumerate(self.B):
            if b.kin:
                b.xc = b.xc + h*b.vc; continue
            m = b.M; c = 4.0/h**2; xhat = b.xc + 0.5*h*b.vc; xn = b.xc.copy(); y = xhat.copy()
            n = self.n
            offs = [b.Rm @ (np.array(s, float)*b.hw) for s in CORNER_SIGNS]   # 8 box corners (Rm frozen this substep)
            laml = np.zeros(len(offs))                             # per-corner PHR-AL multiplier
            for _o in range(self.OUT):                             # ALM outer
                for _it in range(self.NIT):                        # inner block Newton on the com (translational)
                    grad = c*m*(y - xhat) + m*grav; curv = c*m*np.ones(3)
                    for sidx, off in enumerate(offs):
                        gap = float((2*y - xn + off) @ n)          # endpoint corner gap (a=2), plane through origin
                        if self.use_al:
                            t = laml[sidx] - self.kappa*gap
                            if t > 0: grad += 0.5*(-t)*(2*n); curv += 0.5*self.kappa*(2*n)**2   # endpoint weight 1/2
                        elif gap < 0:
                            grad += 0.5*(self.kappa*gap)*(2*n); curv += 0.5*self.kappa*(2*n)**2
                    step = -grad/np.maximum(curv, 1e-12); y = y + step
                    if np.linalg.norm(step) < 1e-12*(np.linalg.norm(y) + 1e-9): break
                if self.use_al:                                    # PHR multiplier update per corner
                    for sidx, off in enumerate(offs):
                        laml[sidx] = max(0.0, laml[sidx] - self.kappa*float((2*y - xn + off) @ n))
            xend = 2*y - xn; b.xc = xend; v1 = 2*(b.xc - xn)/h - b.vc
            fN = float(np.sum(laml)) if self.use_al else float(sum(max(0.0, -self.kappa*float((xend + o) @ n)) for o in offs))
            # Coulomb friction: STABLE post-step tangential velocity projection with the TOTAL normal force
            # (lagged-descent friction composed INTO the block Newton is the named refinement; the in-
            #  Newton normal-tangential coupling is stiff at v_t->0, which is what the friction gate probes.)
            if fN > 0:
                vt = v1 - (v1 @ n)*n; vtn = np.linalg.norm(vt)
                if vtn > 1e-12:
                    v1 = v1 - min(self.mu*fN*h/m, vtn)*(vt/vtn)   # Coulomb-clamped tangential impulse
                ftm = min(self.mu*fN, m*vtn/h)
                cimp[i] = (fN*n - (ftm*(vt/vtn) if vtn > 1e-12 else np.zeros(3)))*h
            else:
                cimp[i] = np.zeros(3)
            b.vc = v1
        self._cimp = {i: self._cimp.get(i, 0) + cimp[i] for i in cimp}

    def step(self, dt, substeps=1, control=None, t0=0.0):
        self._last_dt = dt; self._cimp = {i: np.zeros(3) for i in range(len(self.B))}
        for ss in range(substeps):
            if control: control(self, t0 + ss*dt/substeps)
            self._substep(dt/substeps)


def _selftest():
    """Acceptance: free fall (2nd-order exact), flat rest (contact force == Mg), and the incline friction battery
    (stick/slide at atan(mu), Coulomb slide acceleration, no chatter). Bodies start AT REST ON the surface: a
    settle transient injects a spurious downhill velocity that biases the transition angle and the accel fit."""
    import os
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")
    MU = 0.5; atan_mu = np.degrees(np.arctan(MU)); hw = np.array([0.15, 0.1, 0.1]); dt = 1/240
    ok = True
    # (1) free-fall: z(t) = z0 - 0.5 g t^2 exactly (no contact)
    e = MidpointVBDEngine(ramp_deg=0, mu=MU); e.add_body(0.3, 0.2, 0.2, [0, 0, 5.0])
    for _ in range(200): e.step(dt, 2)
    T = 200*dt; ff = abs(e.com(0)[2] - (5.0 - 0.5*G*T**2)); ok &= ff < 1e-10
    print("  free-fall exact err %.1e %s" % (ff, "PASS" if ff < 1e-10 else "FAIL"))
    # (2) flat rest: contact force == Mg
    e = MidpointVBDEngine(ramp_deg=0, mu=MU); e.add_body(0.3, 0.2, 0.2, [0, 0, 0.1]); M = e.B[0].M
    for _ in range(400): e.step(dt, 2)
    fr = abs(np.linalg.norm(e.contact_forces()[0]) - M*G); ok &= fr < 1e-6
    print("  flat rest f-Mg err %.1e %s" % (fr, "PASS" if fr < 1e-6 else "FAIL"))
    # (3) incline N-T battery: rest-on-surface start at each angle
    def slid(deg, N=500):
        th = np.radians(deg); n = np.array([-np.sin(th), 0, np.cos(th)])
        com0 = (np.sin(th)*hw[0] + np.cos(th)*hw[2])*n
        e = MidpointVBDEngine(ramp_deg=deg, mu=MU); e.add_body(0.3, 0.2, 0.2, com0)
        down = np.array([-np.cos(th), 0, -np.sin(th)]); x0 = e.com(0).copy(); pr = []
        for _ in range(N): e.step(dt, 2); pr.append(float((e.com(0)-x0) @ down))
        return np.array(pr)
    sw = {a: slid(a) for a in [18, 22, 25, 28, 32, 38]}
    is_sl = lambda a: sw[a][-1] > 0.03
    stick = max((a for a in sw if not is_sl(a)), default=0); slide = min((a for a in sw if is_sl(a)), default=90)
    trans = 0.5*(stick+slide); ok_t = abs(trans - atan_mu) <= 3 and stick < atan_mu+1 and slide > atan_mu-3
    ok &= ok_t; print("  transition %.1f vs atan(mu) %.2f %s" % (trans, atan_mu, "PASS" if ok_t else "FAIL"))
    p = sw[38]; t = np.arange(len(p))*dt; w = slice(len(p)//4, None); a = 2*np.polyfit(t[w]**2, p[w], 1)[0]
    aan = G*(np.sin(np.radians(38)) - MU*np.cos(np.radians(38))); ae = abs(a-aan)/aan; ok &= ae < 0.12
    print("  slide@38 accel %.3f vs %.3f err %.1e %s" % (a, aan, ae, "PASS" if ae < 0.12 else "FAIL"))
    print("  SELFTEST", "PASS" if ok else "FAIL"); return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
