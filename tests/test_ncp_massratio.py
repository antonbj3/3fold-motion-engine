"""Mass disparity: physical reconstruction, original units, and explicit failures."""
import pickle
import numpy as np
import pytest
from motion_engine.ncp import ncp_ref as R


def _stack(ratio):
    bodies, _, cfg = R.scene_massratio_stack(ratio=ratio)
    v = np.zeros(18); v[2::6] = -9.81*cfg['dt']
    return R.build_scene(bodies,R.collect_contacts(bodies),cfg['mu'],cfg['dt'],v_free=v)


@pytest.mark.parametrize('ratio',[1e2,1e4,1e6])
@pytest.mark.parametrize('xstep',['legacy','triangular'])
def test_mass_ratio_default_is_finite_and_balances_weight(ratio,xstep):
    sc=_stack(ratio); G=sc.delassus(); b=sc.b()
    out=R.solve_ncp_admm(G,b,sc.mu,xstep=xstep)
    lam,hist,_=out
    assert np.isfinite(lam).all() and np.isfinite(hist).all()
    assert R.natural_residual(lam,G,b,sc.mu)<1e-8
    assert np.max(abs(sc.apply(lam)))<1e-6
    ground=sc.body_pairs[:,1]==-1
    assert np.isclose(lam.reshape(-1,3)[ground,0].sum(),(ratio+2)*9.81*sc.dt,rtol=1e-8)
    status=out.solver_status
    assert status in ('CONVERGED','PRECISION_LIMIT','MAX_ITERATIONS')
    assert (status=='CONVERGED') == (out.solver_diagnostics['residual']<1e-12)


@pytest.mark.parametrize('ratio',[1e2,1e4,1e6])
def test_heterogeneous_contacts_retry_in_original_impulse_units(ratio):
    # Two separate bodies touching the same ground; G has a known six-mode
    # mass contrast. Scaling one scalar per contact must keep mu unchanged.
    J=np.eye(6)
    mi=np.array([1,2,3,1/ratio,2/ratio,3/ratio])
    v=np.array([-.04,.01,-.007,-.04,-.013,.009])
    sc=R.Scene(J,mi,v,np.array([.5,.5]),np.array([[0,-1],[1,-1]]),.004)
    G=sc.delassus(); b=sc.b()
    out=R.solve_ncp_admm(G,b,sc.mu,iters=120,tol=1e-10,xstep='triangular')
    assert out.solver_status=='CONVERGED'
    assert out.force_selection=='ADMM_EQUILIBRATED_RETRY'
    assert np.isfinite(out[0]).all()
    expected=-b/mi
    assert np.allclose(out[0],expected,rtol=1e-8,atol=1e-9)
    assert R.natural_residual(out[0],G,b,sc.mu)<1e-10
    assert out.solver_diagnostics['residual']==out[1][-1]
    assert np.max(abs(sc.apply(out[0])))<1e-8


def test_unattainable_tolerance_has_honest_status():
    sc=_stack(1e6); G=sc.delassus(); b=sc.b()
    out=R.solve_ncp_admm(G,b,sc.mu,tol=1e-30,xstep='triangular')
    assert np.isfinite(out[0]).all() and np.isfinite(out[1]).all()
    assert out.solver_status=='PRECISION_LIMIT'
    assert out.solver_diagnostics['residual']>1e-30
    assert R.natural_residual(out[0],G,b,sc.mu)<1e-8
    restored=pickle.loads(pickle.dumps(out))
    assert restored.solver_status==out.solver_status
    assert restored.solver_diagnostics==out.solver_diagnostics
    assert restored.info['solver_status']==out.solver_status


def test_fixed_penalty_stops_at_precision_instead_of_exhausting_budget():
    sc=_stack(1e6)
    out=R.solve_ncp_admm(sc.delassus(),sc.b(),sc.mu,rho=8e-5,
                       adapt_every=0,tol=1e-30,xstep='triangular')
    assert out.solver_status=='PRECISION_LIMIT'
    assert np.isfinite(out[0]).all() and R.natural_residual(out[0],sc.delassus(),sc.b(),sc.mu)<1e-8


def test_penalty_floor_is_safe_when_progress_observer_stalls(monkeypatch):
    # Fault injection: an observer fails to report progress even at equilibrium.
    # Shrinking rho must stop before the zero modes become numerically singular.
    monkeypatch.setattr(R,'natural_residual',lambda *a,**kw: 1.0)
    out=R.solve_ncp_admm(np.diag([1.,0.,0.]),np.array([-.1,0.,0.]),
                       np.array([0.]),iters=1500,xstep='triangular')
    assert out.solver_status=='PRECISION_LIMIT'
    assert out.solver_diagnostics['attempts'][0]['iterations']<1500
    assert np.isfinite(out[0]).all()


def test_equilibrated_dual_preserves_exact_slip_fixed_point(monkeypatch):
    # Continue an exact Coulomb solution after an observer fault. Changing only
    # the penalty must preserve the physical solution and its unscaled dual.
    G=np.diag([1.,2.,3.]); b=np.array([-.04,1.,0.]); mu=np.array([.5])
    exact=np.array([.04,-.02,0.])
    monkeypatch.setattr(R,'natural_residual',lambda *a,**kw: 1.0)
    diag={}
    out=R._admm_core(G,b,mu,51,1e-12,exact,2.,25,True,'triangular',diag,
                     physical=(G,b,np.ones(3),np.array([1/3])))
    assert diag['n_factor']>1
    assert np.max(abs(out[0]-exact))<1e-12


def test_iteration_budget_is_not_reported_as_convergence():
    sc=_stack(1e6)
    out=R.solve_ncp_admm(sc.delassus(),sc.b(),sc.mu,iters=1)
    assert out.solver_status=='MAX_ITERATIONS'
    assert out.solver_diagnostics['residual']>1e-12
    assert np.isfinite(out[0]).all()


@pytest.mark.parametrize('where',['G','b','mu','warm','rho'])
@pytest.mark.parametrize('value',[np.nan,np.inf,-np.inf])
def test_nonfinite_inputs_are_explicit_errors(where,value):
    G=np.eye(3); b=np.array([-.1,.2,.3]); mu=np.array([.5])
    kw={}
    if where=='G': G[0,0]=value
    if where=='b': b[0]=value
    if where=='mu': mu[0]=value
    if where=='warm': kw['warm']=np.array([value,0.,0.])
    if where=='rho': kw['rho']=value
    with pytest.raises(ValueError):
        R.solve_ncp_admm(G,b,mu,**kw)


def test_failed_factorization_has_finite_tuple_and_status():
    out=R.solve_ncp_admm(-np.eye(3),np.array([-.1,0.,0.]),np.array([.5]))
    assert out.solver_status=='NUMERICAL_FAILURE'
    assert np.isfinite(out[0]).all()
    assert out.solver_diagnostics['attempts'][0]['reason']


def test_injected_nonfinite_xstep_cannot_return_nan(monkeypatch):
    import scipy.linalg.blas as blas
    monkeypatch.setattr(blas,'dtrsv',lambda c,x,**kw: np.full_like(x,np.nan))
    out=R.solve_ncp_admm(np.eye(3),np.array([-.1,0.,0.]),np.array([.5]),xstep='triangular')
    assert out.solver_status=='NUMERICAL_FAILURE'
    assert np.isfinite(out[0]).all() and np.isfinite(out[1]).all()


def test_converged_legacy_bytes_on_diagonal_problem():
    # An independent old iteration with no adaptation: it must execute exactly
    # the same arithmetic, including the natural residual history.
    G=np.diag([1.,2.,3.]); b=np.array([-.1,.02,-.01]); mu=np.array([.5])
    rho=2.; chol=np.linalg.cholesky(G+rho*np.eye(3))
    z=np.zeros(3); gamma=z.copy(); history=[]
    for _ in range(300):
        u=(G@z+b).reshape(1,3)
        g=b+R.desaxce(u,mu).reshape(-1)
        x=-np.linalg.solve(chol.T,np.linalg.solve(chol,g+gamma-rho*z))
        z=R.proj_cone((x+gamma/rho).reshape(1,3),mu).ravel()
        gamma=gamma+rho*(x-z)
        r=R.natural_residual(z,G,b,mu); history.append(r)
        if r<1e-12: break
    out=R.solve_ncp_admm(G,b,mu,iters=300,adapt_every=0)
    assert out[0].tobytes()==z.tobytes()
    assert out[1].tobytes()==np.array(history).tobytes()
    assert out.force_selection=='ADMM_WARM_START'
