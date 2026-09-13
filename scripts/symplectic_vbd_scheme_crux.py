#!/usr/bin/env python3
"""Which higher-order / symplectic schemes admit a per-block descent structure at all?

VBD/AVBD does a per-block minimization of backward Euler's incremental potential
E(x) = (1/2h^2)||x - xhat||^2_M + U(x), which is descent-stable, and coloured block Gauss-Seidel converges.
Derived here and verified numerically below:
  (a) BDF2      : M(x_{n+1} - xhat_B) + (4h^2/9) grad U = 0
                  => incremental potential E = (9/8h^2)||x - xhat_B||^2_M + U(x)
                  xhat_B = 4/3 x_n - 1/3 x_{n-1} + 8h/9 v_n - 2h/9 v_{n-1}
                  2nd order, A-stable, dissipative but less so. Not symplectic. A straight VBD swap.
  (b) MIDPOINT  : in x_mid = (x_n + x_{n+1})/2: M(x_mid - xhat_m) + (h^2/4) grad U(x_mid) = 0
                  => incremental potential E = (2/h^2)||x_mid - xhat_m||^2_M + U(x_mid)
                  xhat_m = x_n + (h/2) v_n; reconstruct x_{n+1} = 2 x_mid - x_n,
                  v_{n+1} = (4/h)(x_mid - x_n) - v_n. 2nd order, symplectic (bounded energy), and more
                  diagonally dominant than backward Euler (4/h^2 vs 1/h^2 inertial regularization), so block
                  descent converges faster. The key positive result.
  (c) DISCRETE-GRADIENT / AVF : the force (a discrete gradient depending on BOTH endpoints) is not a gradient
                  field in x_{n+1}, so it is not a plain minimization but a root/block-Newton problem.
                  Exactly energy conserving, 2nd order. Loses the pure-descent guarantee.

Falsifying test: a stiff N-mass chain (linear, exact modal anchor; block-descent iteration scaling plus
mass-ratio blow-up) and a nonlinear pendulum (analytic anchor; convergence order and long-run energy drift).
All schemes run through the SAME red-black block-descent VBD solver where a descent structure exists.

Pre-registered gate: the approach survives iff at least one scheme that is >=2nd order or energy conserving
runs under per-block descent/colouring with iteration counts within ~3x of backward-Euler VBD at stiffness
1e4.

I/O: no input. Writes reports/symplectic_vbd_scheme_crux.json and prints the orders, the drift, the iteration
scaling and the mass-ratio table. Exit 0 always.
"""
import json, os, time, numpy as np

np.set_printoptions(precision=4, suppress=True)
OUT = {}

# ---------------------------------------------------------------------------
# Inertial coefficient c and inertial target xhat per scheme (E = (c/2)||x-xhat||^2_M + U)
#   BE:  c=1/h^2 , xhat = x_n + h v_n            , x_{n+1}=x* , v_{n+1}=(x*-x_n)/h
#   BDF2:c=9/(4h^2), xhat=4/3 x_n -1/3 x_{n-1} +8h/9 v_n -2h/9 v_{n-1}
#                    x_{n+1}=x*, v_{n+1}=(3 x* -4 x_n + x_{n-1})/(2h)
#   MID: c=4/h^2 , xhat = x_n + (h/2) v_n  (solve in x_mid)
#                    x_{n+1}=2 x* - x_n , v_{n+1}=(4/h)(x*-x_n)-v_n
# ---------------------------------------------------------------------------

def scheme_params(name, h):
    if name == "BE":   return 1.0/h**2
    if name == "BDF2": return 9.0/(4.0*h**2)
    if name == "MID":  return 4.0/h**2
    raise ValueError(name)

# ===========================================================================
# PART 1  --  LINEAR STIFF CHAIN : exact modal anchor + block-descent scaling
# ===========================================================================
# 1D chain: masses m_i at x_i, i=0..N-1, springs stiffness k between neighbours,
# rest length L=1, both ends fixed to walls (x_{-1}=0 wall spring, x_N=(N+1)L wall).
# U = sum_{i} (k/2)(x_{i+1}-x_i - L)^2 including two wall springs.
# grad_i U = k(2 x_i - x_{i-1} - x_{i+1})  (walls contribute fixed terms) -> linear K.

def build_chain(N, k, masses):
    # K (NxN) for interior masses with fixed walls at x=0 and x=(N+1)*L
    K = np.zeros((N, N))
    for i in range(N):
        K[i, i] += 2*k
        if i > 0:   K[i, i-1] -= k
        if i < N-1: K[i, i+1] -= k
    M = np.diag(masses)
    # equilibrium x_eq: K x = b, b from wall rest-length forcing
    L = 1.0
    b = np.zeros(N)
    b[0]  += k*(0.0 + L)          # left wall at 0, rest length L -> target x0=L
    b[-1] += k*((N+1)*L - L)      # right wall
    # general interior rest contributions cancel in this uniform setup; solve:
    # Actually assemble b properly: grad_i U = k(2xi - xi-1 - xi+1) - k*(rest terms)=0
    # For uniform L and walls, equilibrium is x_i=(i+1)L. Use that directly.
    x_eq = np.array([(i+1)*L for i in range(N)])
    return K, M, x_eq

def chain_exact(N, k, masses, x0, v0, t):
    """Exact solution of M xddot = -K (x - x_eq) via modal decomposition."""
    K, M, x_eq = build_chain(N, k, masses)
    Minv_sqrt = np.diag(1.0/np.sqrt(masses))
    A = Minv_sqrt @ K @ Minv_sqrt          # symmetric
    w2, V = np.linalg.eigh(A)              # A = V diag(w2) V^T
    w = np.sqrt(np.maximum(w2, 0.0))
    q0 = V.T @ (Minv_sqrt @ np.linalg.solve(Minv_sqrt, (x0 - x_eq)))  # modal disp
    # careful: modal coords p = V^T M^{1/2}(x-x_eq); pdot = V^T M^{1/2} v
    Msqrt = np.diag(np.sqrt(masses))
    p0  = V.T @ (Msqrt @ (x0 - x_eq))
    pd0 = V.T @ (Msqrt @ v0)
    pt  = np.where(w > 1e-12, p0*np.cos(w*t) + (pd0/np.where(w>1e-12,w,1))*np.sin(w*t),
                   p0 + pd0*t)
    x = x_eq + Minv_sqrt @ (V @ pt)
    return x

def _gs_residual(x, xhat, c, masses, k, N):
    res = np.empty(N)
    for i in range(N):
        left  = x[i-1] if i > 0   else 0.0
        right = x[i+1] if i < N-1 else (N+1)*1.0
        res[i] = c*masses[i]*(x[i]-xhat[i]) + (2*k*x[i] - k*(left+right))
    return np.max(np.abs(res))

def _gs_solve(xhat, c, masses, k, N, rel_tol=1e-6, maxit=500):
    """Red-black block Gauss-Seidel on the incremental potential; iters to reduce the
    residual by rel_tol from its initial (warm-started) value. Rate = the geometry."""
    x = xhat.copy()
    colors = [np.arange(0, N, 2), np.arange(1, N, 2)]
    res0 = _gs_residual(x, xhat, c, masses, k, N)
    if res0 == 0.0: return x, 1
    for it in range(maxit):
        for col in colors:
            for i in col:
                left  = x[i-1] if i > 0   else 0.0
                right = x[i+1] if i < N-1 else (N+1)*1.0
                x[i] = (c*masses[i]*xhat[i] + k*(left + right)) / (c*masses[i] + 2*k)
        if _gs_residual(x, xhat, c, masses, k, N) < rel_tol*res0:
            break
    return x, it+1

def block_descent_step(name, h, x_n, v_n, hist, masses, k, N, **kw):
    c = scheme_params(name, h)
    if name == "BE":    xhat = x_n + h*v_n
    elif name == "MID": xhat = x_n + 0.5*h*v_n
    else: raise ValueError(name)
    x, it = _gs_solve(xhat, c, masses, k, N, **kw)
    if name == "BE":  x_next, v_next = x, (x - x_n)/h
    else:             x_next, v_next = 2*x - x_n, (4.0/h)*(x - x_n) - v_n
    return x_next, v_next, it


def block_descent_step_bdf2(h, x_n, v_n, x_nm1, v_nm1, masses, k, N, **kw):
    c = 9.0/(4.0*h**2)
    xhat = (4.0/3.0)*x_n - (1.0/3.0)*x_nm1 + (8.0*h/9.0)*v_n - (2.0*h/9.0)*v_nm1
    x, it = _gs_solve(xhat, c, masses, k, N, **kw)
    v_next = (3*x - 4*x_n + x_nm1)/(2*h)
    return x, v_next, it


def run_chain(name, N, k, masses, x0, v0, T, h, rel_tol=1e-10, maxit=2000):
    nsteps = int(round(T/h))
    x, v = x0.copy(), v0.copy()
    iters = []
    kw = dict(rel_tol=rel_tol, maxit=maxit)
    if name == "BDF2":
        x1, v1, it0 = block_descent_step("MID", h, x, v, None, masses, k, N, **kw)  # 2nd-order bootstrap
        x_nm1, v_nm1 = x, v
        x, v = x1, v1
        iters.append(it0)
        for s in range(1, nsteps):
            xn, vn, it = block_descent_step_bdf2(h, x, v, x_nm1, v_nm1, masses, k, N, **kw)
            x_nm1, v_nm1 = x, v
            x, v = xn, vn
            iters.append(it)
    else:
        for s in range(nsteps):
            xn, vn, it = block_descent_step(name, h, x, v, None, masses, k, N, **kw)
            x, v = xn, vn
            iters.append(it)
    return x, v, np.array(iters)


# ===========================================================================
# PART 2  --  NONLINEAR PENDULUM : analytic anchor, order + energy drift + discrete-grad
# ===========================================================================
# theta'' = -(g/l) sin theta ; unit-mass generalized coord with m_eff = m l^2, U=-m g l cos th
# incremental potential in theta (single block -> Newton is the "block descent")

class Pend:
    def __init__(self, g=9.81, l=1.0, m=1.0):
        self.g, self.l, self.m = g, l, m
        self.meff = m*l**2
    def U(self, th):   return -self.m*self.g*self.l*np.cos(th)
    def dU(self, th):  return  self.m*self.g*self.l*np.sin(th)
    def d2U(self, th): return  self.m*self.g*self.l*np.cos(th)
    def energy(self, th, om): return 0.5*self.meff*om**2 + self.U(th)

def newton_min(c, meff, xhat, dU, d2U, x0, tol=1e-13, maxit=100):
    """solve c*meff*(x-xhat)+dU(x)=0 (min of incremental potential), Newton."""
    x = x0
    for _ in range(maxit):
        f  = c*meff*(x - xhat) + dU(x)
        fp = c*meff + d2U(x)
        dx = -f/fp
        x += dx
        if abs(dx) < tol: break
    return x

def pend_step(name, P, h, th, om, hist=None):
    meff = P.meff
    if name == "BE":
        c = 1.0/h**2; xhat = th + h*om
        thn = newton_min(c, meff, xhat, P.dU, P.d2U, xhat)
        omn = (thn - th)/h
        return thn, omn, hist
    if name == "MID":
        c = 4.0/h**2; xhat = th + 0.5*h*om
        xm = newton_min(c, meff, xhat, P.dU, P.d2U, xhat)
        thn = 2*xm - th; omn = (4.0/h)*(xm - th) - om
        return thn, omn, hist
    if name == "BDF2":
        th_nm1, om_nm1 = hist
        c = 9.0/(4.0*h**2)
        xhat = (4.0/3.0)*th - (1.0/3.0)*th_nm1 + (8.0*h/9.0)*om - (2.0*h/9.0)*om_nm1
        thn = newton_min(c, meff, xhat, P.dU, P.d2U, xhat)
        omn = (3*thn - 4*th + th_nm1)/(2*h)
        return thn, omn, (th, om)
    if name == "DG":   # discrete gradient (energy-conserving), root solve (NON-minimization)
        # 2(th1-th)/h - 2 om + h * gradbar/meff = 0 ; gradbar = (U(th1)-U(th))/(th1-th)
        def gradbar(th1):
            d = th1 - th
            if abs(d) < 1e-9: return P.dU(0.5*(th1+th))    # limit -> midpoint gradient
            return (P.U(th1) - P.U(th))/d
        def F(th1):  return 2*(th1-th)/h - 2*om + h*gradbar(th1)/meff
        # Newton (numerical derivative for the awkward gradbar)
        th1 = th + h*om
        for _ in range(100):
            f = F(th1); eps=1e-8
            fp = (F(th1+eps)-F(th1-eps))/(2*eps)
            step = -f/fp; th1 += step
            if abs(step) < 1e-13: break
        om1 = 2*(th1-th)/h - om
        return th1, om1, hist
    raise ValueError(name)

def pend_ref(P, th0, om0, T, dt=1e-5):
    """RK4 high-accuracy reference."""
    def deriv(s):
        th, om = s
        return np.array([om, -(P.g/P.l)*np.sin(th)])
    n = int(round(T/dt)); s = np.array([th0, om0])
    for _ in range(n):
        k1=deriv(s); k2=deriv(s+0.5*dt*k1); k3=deriv(s+0.5*dt*k2); k4=deriv(s+dt*k3)
        s = s + dt/6*(k1+2*k2+2*k3+k4)
    return s

def run_pend(name, P, th0, om0, T, h):
    n = int(round(T/h)); th, om = th0, om0; hist=None
    if name == "BDF2":
        th1, om1, _ = pend_step("MID", P, h, th, om)
        hist=(th, om); th, om = th1, om1
        for _ in range(1, n):
            th, om, hist = pend_step("BDF2", P, h, th, om, hist)
    else:
        for _ in range(n):
            th, om, hist = pend_step(name, P, h, th, om, hist)
    return th, om


# ===========================================================================
# EXPERIMENTS
# ===========================================================================
def exp_order():
    """Convergence order on nonlinear pendulum vs RK4 reference."""
    P = Pend()
    th0, om0 = 1.0, 0.0        # ~57 deg, genuinely nonlinear
    T = 2.0
    ref = pend_ref(P, th0, om0, T, dt=2e-6)
    hs = [0.02, 0.01, 0.005, 0.0025, 0.00125]
    res = {}
    for name in ["BE", "BDF2", "MID", "DG"]:
        errs = []
        for h in hs:
            thf, omf = run_pend(name, P, th0, om0, T, h)
            errs.append(np.hypot(thf-ref[0], omf-ref[1]))
        errs = np.array(errs)
        # log-log slope (order)
        slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
        res[name] = {"h": hs, "err": errs.tolist(), "order": float(slope)}
    return res

def exp_energy_drift():
    """Long-run energy drift on nonlinear pendulum (secular growth = symplectic test)."""
    P = Pend(); th0, om0 = 1.2, 0.0
    E0 = P.energy(th0, om0)
    # period ~ 2*pi*sqrt(l/g)*(1+th0^2/16) ; use ~ 2.16 s
    Tper = 2.16
    nper = 3000
    h = 0.01
    res = {}
    for name in ["BE", "BDF2", "MID", "DG"]:
        th, om, hist = th0, om0, None
        n = int(round(nper*Tper/h))
        Emax_dev = 0.0; Efinal=None
        # track drift: sample energy every ~1 period
        samp = max(1, int(round(Tper/h)))
        Es=[]
        if name=="BDF2":
            th1,om1,_=pend_step("MID",P,h,th,om); hist=(th,om); th,om=th1,om1
        for s in range(n):
            if name=="BDF2" and s>0:
                th,om,hist=pend_step("BDF2",P,h,th,om,hist)
            elif name=="BDF2" and s==0:
                pass
            else:
                th,om,hist=pend_step(name,P,h,th,om,hist)
            if s % samp == 0:
                Es.append(P.energy(th,om))
        Es=np.array(Es); Efinal=P.energy(th,om)
        # secular drift = slope of energy vs period index (relative)
        idx=np.arange(len(Es))
        drift_slope = np.polyfit(idx, Es, 1)[0] if len(Es)>2 else 0.0
        res[name]={"E0":float(E0),"Efinal":float(Efinal),
                   "rel_final_dev":float((Efinal-E0)/abs(E0)),
                   "rel_drift_per_period":float(drift_slope/abs(E0)),
                   "Emax_abs_dev":float(np.max(np.abs(Es-E0))/abs(E0)),
                   "nper_run":nper}
    return res

def exp_chain_order():
    """Convergence order on stiff linear chain vs EXACT modal anchor."""
    N=8; k=1.0e3; masses=np.ones(N)
    K,M,x_eq=build_chain(N,k,masses)
    x0=x_eq.copy(); x0[0]+=0.02   # small pluck -> linear
    v0=np.zeros(N); T=0.1
    xref=chain_exact(N,k,masses,x0,v0,T)
    hs=[2e-3,1e-3,5e-4,2.5e-4]
    res={}
    for name in ["BE","BDF2","MID"]:
        errs=[]
        for h in hs:
            xf,vf,_=run_chain(name,N,k,masses,x0,v0,T,h)
            errs.append(np.linalg.norm(xf-xref))
        errs=np.array(errs)
        slope=np.polyfit(np.log(hs),np.log(errs),1)[0]
        res[name]={"h":hs,"err":errs.tolist(),"order":float(slope)}
    return res

def exp_iteration_scaling():
    """Block-descent iteration count vs stiffness (the KILL gate)."""
    N=8; masses=np.ones(N); h=1e-3
    stiffs=[1e0,1e1,1e2,1e3,1e4,1e5]
    res={}
    for name in ["BE","BDF2","MID"]:
        row={}
        for k in stiffs:
            K,M,x_eq=build_chain(N,k,masses)
            x0=x_eq.copy(); x0[0]+=0.02; v0=np.zeros(N)
            # run 20 steps, report median iters
            x,v=x0.copy(),v0.copy(); its=[]
            if name=="BDF2":
                x1,v1,i0=block_descent_step("MID",h,x,v,None,masses,k,N)
                xnm1,vnm1=x,v; x,v=x1,v1; its.append(i0)
                for s in range(1,20):
                    xn,vn,it=block_descent_step_bdf2(h,x,v,xnm1,vnm1,masses,k,N)
                    xnm1,vnm1=x,v; x,v=xn,vn; its.append(it)
            else:
                for s in range(20):
                    xn,vn,it=block_descent_step(name,h,x,v,None,masses,k,N)
                    x,v=xn,vn; its.append(it)
            row[f"{k:.0e}"]=int(np.median(its))
        res[name]=row
    # ratios vs BE at 1e4
    be=res["BE"]["1e+04"]
    res["_gate_ratio_vs_BE_at_1e4"]={n:res[n]["1e+04"]/be for n in ["BE","BDF2","MID"]}
    return res

def exp_mass_ratio():
    """Mass-ratio blow-up: two-mass stiff coupling, iterations vs mass ratio."""
    h=1e-3; k=1e4
    ratios=[1e0,1e1,1e2,1e3,1e4,1e6]
    res={}
    for name in ["BE","BDF2","MID"]:
        row={}
        for r in ratios:
            N=2; masses=np.array([1.0, r])
            K,M,x_eq=build_chain(N,k,masses)
            x0=x_eq.copy(); x0[0]+=0.01; v0=np.zeros(N)
            x,v=x0.copy(),v0.copy(); its=[]
            if name=="BDF2":
                x1,v1,i0=block_descent_step("MID",h,x,v,None,masses,k,N)
                xnm1,vnm1=x,v; x,v=x1,v1; its.append(i0)
                for s in range(1,15):
                    xn,vn,it=block_descent_step_bdf2(h,x,v,xnm1,vnm1,masses,k,N)
                    xnm1,vnm1=x,v; x,v=xn,vn; its.append(it)
            else:
                for s in range(15):
                    xn,vn,it=block_descent_step(name,h,x,v,None,masses,k,N)
                    x,v=xn,vn; its.append(it)
            row[f"{r:.0e}"]=int(np.median(its))
        res[name]=row
    return res


if __name__ == "__main__":
    t0=time.time()
    print("[1/5] convergence order (pendulum, analytic RK4 anchor) ...")
    OUT["pendulum_order"]=exp_order()
    print("[2/5] convergence order (stiff linear chain, EXACT modal anchor) ...")
    OUT["chain_order"]=exp_chain_order()
    print("[3/5] long-run energy drift (pendulum, 3000 periods) ...")
    OUT["energy_drift"]=exp_energy_drift()
    print("[4/5] block-descent iteration scaling vs stiffness (KILL gate) ...")
    OUT["iteration_scaling"]=exp_iteration_scaling()
    print("[5/5] mass-ratio blow-up ...")
    OUT["mass_ratio"]=exp_mass_ratio()
    OUT["_meta"]={"runtime_s":time.time()-t0,
        "gate":"the approach survives iff >=1 (>=2nd-order OR energy-conserving) scheme runs under "
               "per-block descent with iters within ~3x of BE at stiffness 1e4"}
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports",
                            "symplectic_vbd_scheme_crux.json")
    with open(out_path, "w") as f:
        json.dump(OUT, f, indent=2, default=float)
    print(f"\nDONE in {OUT['_meta']['runtime_s']:.1f}s -> reports/symplectic_vbd_scheme_crux.json")
    # terse summary
    print("\n=== ORDERS (pendulum) ===")
    for n,d in OUT["pendulum_order"].items(): print(f"  {n:5s} order={d['order']:.2f}")
    print("=== ORDERS (chain) ===")
    for n,d in OUT["chain_order"].items(): print(f"  {n:5s} order={d['order']:.2f}")
    print("=== ENERGY DRIFT (rel/period, 3000 per) ===")
    for n,d in OUT["energy_drift"].items():
        print(f"  {n:5s} drift/period={d['rel_drift_per_period']:.2e}  maxdev={d['Emax_abs_dev']:.2e}")
    print("=== ITER SCALING vs stiffness ===")
    for n in ["BE","BDF2","MID"]:
        print(f"  {n:5s} " + " ".join(f"{kk}:{vv}" for kk,vv in OUT['iteration_scaling'][n].items()))
    print("  gate ratio vs BE@1e4:", OUT["iteration_scaling"]["_gate_ratio_vs_BE_at_1e4"])
    print("=== MASS-RATIO iters ===")
    for n in ["BE","BDF2","MID"]:
        print(f"  {n:5s} " + " ".join(f"{kk}:{vv}" for kk,vv in OUT['mass_ratio'][n].items()))
