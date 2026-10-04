"""Adversarial contracts for redundant rigid contact and observable responses."""
import itertools

import numpy as np
import pytest

from motion_engine.ncp import ncp_ref as R, ncp_router as T, ncp_gpu as GPU
from motion_engine.ncp.contact_rank import rank_factor, guarded_response


BOXES = [(0.5, 0.5, 0.5, 1.0), (0.25, 0.75, 0.4, 2.0),
         (1.0, 0.2, 0.6, 0.5), (0.8, 0.6, 0.3, 3.0)]
PATTERNS = list(itertools.product(('stick', 'slip'), repeat=4))


def box_scene(half=(0.5, 0.5, 0.5), mass=1.0, angle=0.0, moving=True):
    dt = 0.01
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    body = R.Box(np.array(half), mass, rot @ np.array([0., 0., half[2]]),
                 quat=np.array([np.cos(angle/2), 0., np.sin(angle/2), 0.]))
    n = rot @ np.array([0., 0., 1.])
    contacts = [{'p': rot @ np.array([sx*half[0], sy*half[1], 0.]), 'n': n,
                 'gap': 0., 'a': 0, 'b': -1}
                for sx, sy in itertools.product((-1, 1), repeat=2)]
    vf = np.zeros(6)
    vf[:3] = (rot @ np.array([1., 0., 0.]) if moving else 0.) + dt*R.GRAVITY
    return R.build_scene([body], contacts, 0.3, dt, v_free=vf)


def algebraic_branch(sc, labels, spin=False):
    """A residual fixture; arbitrary mixed labels need not be physical v_free."""
    G = sc.delassus()
    lam, u = np.zeros(12), np.zeros(12)
    for c, label in enumerate(labels):
        i=3*c; lam[i]=0. if label=='open' else 1.
        if label=='open':
            u[i]=.5
        if label == 'slip':
            sh = np.array([1., 0.]) if not spin else np.array([(c%2)*2-1., (c//2)*2-1.])/np.sqrt(2.)
            lam[i+1:i+3] = -sc.mu[c]*sh
            u[i+1:i+3] = sh
    return lam, G, u-G@lam


@pytest.mark.parametrize('dims', BOXES)
@pytest.mark.parametrize('labels', PATTERNS)
def test_all_64_closed_corner_instances(dims, labels):
    sc=box_scene(dims[:3], dims[3])
    lam,G,b=algebraic_branch(sc, labels)
    dm,db,info=R.sensitivity(lam,G,b,sc.mu,labels)
    assert info['status']=='REDUNDANT_CONTACTS'
    assert info['nullity']>=1
    assert np.isnan(dm).all() and np.isnan(db).all()
    N=info['nullspace']; A=R._active_system(lam,G,b,sc.mu,labels)[0]
    assert np.linalg.norm(A@N) < 1e-10*np.linalg.norm(A)
    assert np.allclose(N.T@N, np.eye(N.shape[1]), atol=1e-12)
    # Independent geometric left-null witness: alternating corner normals.
    witness=np.zeros(12); witness[::3]=[1., -1., -1., 1.]
    assert np.linalg.norm(witness@A) < 1e-12*np.linalg.norm(A)


@pytest.mark.parametrize('angle', [0., 0.15])
def test_feasible_alternate_split_preserves_motion_energy_and_wrench(angle):
    sc=box_scene(angle=angle); G,b=sc.delassus(),sc.b()
    result=R.solve_ncp_admm(G,b,sc.mu,iters=5000,tol=1e-13)
    lam,hist,labels=result
    assert hist[-1]<1e-11 and list(labels)==['slip']*4
    N=result.info['nullspace']; assert N.shape[1]==1
    alt=lam+0.05*np.min(lam[::3])/np.max(abs(N))*N[:,0]
    assert R.natural_residual(alt,G,b,sc.mu)<1e-10
    assert not np.allclose(lam,alt,atol=1e-10)
    assert np.allclose(sc.J.T@lam,sc.J.T@alt,atol=1e-10)
    assert np.allclose(sc.apply(lam),sc.apply(alt),atol=1e-10)
    M=np.linalg.inv(sc.minv_mat()); v,w=sc.apply(lam),sc.apply(alt)
    assert abs(v@M@v-w@M@w)<1e-10
    assert result.force_selection=='ADMM_WARM_START'
    dm,db,info=R.sensitivity(lam,G,b,sc.mu,labels,observable=sc.J.T,db_dparam=sc.J)
    assert info['observable_invariant'].all() and info['b_compatible'].all()
    assert info['observable_b_valid'].all()
    # Full finite differences through an independent warm PGS solver.
    for j in [0,2]:
        h=1e-6; shift=h*sc.J[:,j]
        lp=R.solve_ncp_pgs(G,b+shift,sc.mu,iters=5000,tol=1e-13,warm=lam)[0]
        lm=R.solve_ncp_pgs(G,b-shift,sc.mu,iters=5000,tol=1e-13,warm=lam)[0]
        fd=sc.J.T@((lp-lm)/(2*h))
        assert np.allclose(fd,info['observable_db'][:,j],atol=1e-6,rtol=1e-6)


def test_force_selection_requires_opt_in_and_inconsistent_changes_stay_nan():
    sc=box_scene(); lam,G,b=algebraic_branch(sc,['slip']*4)
    dm,db,info=R.sensitivity(lam,G,b,sc.mu,['slip']*4,
                            selection='minimum_norm',observable=sc.J.T)
    assert info['selection']=='MINIMUM_NORM_LINEAR_RESPONSE'
    assert np.isfinite(dm[:,info['mu_compatible']]).all()
    assert np.isnan(db[:,~info['b_compatible']]).all()
    assert not info['b_compatible'][::3].any()
    assert np.linalg.norm(info['nullspace'].T@dm[:,info['mu_compatible']])<1e-12


def test_singular_does_not_imply_wrench_is_unique():
    sc=box_scene(); lam,G,b=algebraic_branch(sc,['slip']*4,spin=True)
    _,_,info=R.sensitivity(lam,G,b,sc.mu,['slip']*4,observable=sc.J.T)
    assert info['singular'] and not info['observable_invariant'].all()
    assert np.isnan(info['observable_dmu'][~info['observable_invariant']]).all()


def test_stack_stick_redundancy_preserves_both_body_wrenches():
    bodies,_,kw=R.scene_massratio_stack(ratio=2.)
    ct=R.collect_contacts(bodies); vf=np.tile(kw['dt']*np.r_[R.GRAVITY,np.zeros(3)],len(bodies))
    sc=R.build_scene(bodies,ct,0.6,kw['dt'],v_free=vf)
    G,b=sc.delassus(),sc.b()
    result=R.solve_ncp_admm(G,b,sc.mu,iters=5000,tol=1e-12)
    lam,hist,labels=result; assert hist[-1]<1e-10
    info=R.contact_diagnostics(lam,G,b,sc.mu,labels,observable=sc.J.T)
    assert info['singular'] and info['observable_invariant'].all()
    N=info['nullspace']; assert np.linalg.norm(sc.J.T@N)<1e-10
    alt=lam+1e-6*N[:,0]
    assert np.allclose(sc.apply(alt),sc.apply(lam),atol=1e-10)
    assert R.natural_residual(alt,G,b,sc.mu)<1e-10


def test_nearly_singular_is_guarded_and_zero_matrix_is_not_unique():
    A=np.diag([1.,1e-14,2.]); fac=rank_factor(A)
    X,info=guarded_response(A,np.eye(3),fac,selection='minimum_norm',observable=np.eye(3))
    assert fac[3]['singular']
    assert not info['compatible'][1] and np.isnan(X[:,1]).all()
    assert not info['observable_invariant'][1]
    assert rank_factor(np.zeros((3,3)))[3]['nullity']==3


def test_boundary_disables_even_minimum_norm_derivatives():
    lam=np.array([1.,-.3,0.]); G=np.eye(3); b=-lam
    dm,db,info=R.sensitivity(lam,G,b,[.3],['slip'],selection='minimum_norm',observable=np.eye(3))
    assert info['boundary'] and not info['residual_differentiable']
    assert np.isnan(dm).all() and np.isnan(db).all()
    assert not info['observable_mu_valid'].any()


def test_router_rejects_inconsistent_redundant_bilateral():
    G=np.diag([1.,1.,0.]); b=np.array([-1.,0.,1e-3])
    cert=T.bilateral_certificate(G,b,np.array([.5]))
    assert not cert['is_interior'] and not cert['compatible']
    assert cert['status']=='INCOMPATIBLE_BILATERAL'


def test_router_labels_minimum_norm_on_consistent_redundancy():
    sc=box_scene(moving=False); G,b=sc.delassus(),sc.b()
    cert=T.bilateral_certificate(G,b,sc.mu)
    assert cert['is_interior'] and cert['status']=='REDUNDANT_CONTACTS'
    assert cert['force_selection']=='MINIMUM_NORM_BILATERAL'
    assert np.linalg.norm(G@cert['nullspace'])<1e-10
    assert np.linalg.norm(G@cert['lam']+b)<1e-10


def test_router_all_open_does_not_return_dropped_forces():
    cert=T.bilateral_certificate(np.eye(3),np.array([1.,0.,0.]),np.array([.3]))
    assert cert['is_interior'] and np.array_equal(cert['lam'],np.zeros(3))


def test_gpu_result_host_diagnostic_is_bounded_and_model_aware():
    sc=box_scene(); G,b=sc.delassus(),sc.b(); lam=R.solve_ncp_admm(G,b,sc.mu)[0]
    bs=GPU.Scene(sc.J,sc.Minv,sc.v_free,sc.mu,sc.body_pairs,sc.dt).blocks()
    result=GPU.SolveResult(lam.reshape(-1,3),sc.apply(lam).reshape(-1,6),0.,0,0,False)
    assert result.contact_info['status']=='UNASSESSED_CONTACT_UNIQUENESS'
    assert result.diagnose_contacts(bs)['status']=='REDUNDANT_CONTACTS'
    assert result.contact_info['observable_invariant'].all()
    assert result.diagnose_contacts(bs,max_contacts=3)['status']=='UNASSESSED_SIZE_LIMIT'
    result.contact_model='SOFT_STEP'
    assert result.diagnose_contacts(bs)['status']=='UNASSESSED_CONTACT_MODEL'


def test_empty_contact_sensitivity_and_result():
    sc=R.Scene(np.zeros((0,6)),np.ones(6),np.zeros(6),np.zeros(0),np.zeros((0,2),int),.01)
    dm,db,info=R.sensitivity(np.zeros(0),np.zeros((0,0)),np.zeros(0),np.zeros(0),[],observable=sc.J.T)
    assert info['nullity']==0 and dm.shape==(0,0) and db.shape==(0,0)
    assert R.solve_ncp_admm(np.zeros((0,0)),np.zeros(0),np.zeros(0)).info['nullity']==0


def test_invalid_selection_and_labels_are_rejected():
    _,_,valid=R.sensitivity(np.array([1.,0.,0.]),np.eye(3),np.array([-1.,0.,0.]),[.3],['stick'])
    assert valid['rank']==3 and not valid['singular']
    with pytest.raises(ValueError):
        R.sensitivity(np.ones(3),np.eye(3),np.zeros(3),[.3],['typo'])
    with pytest.raises(ValueError):
        R.sensitivity(np.ones(3),np.eye(3),np.zeros(3),[.3],['stick'],selection='silent_pinv')


def test_forward_result_retains_tuple_and_pickle_contract():
    import pickle
    result=R.solve_ncp_pgs(np.eye(3),np.array([-1.,0.,0.]),np.array([.3]))
    copy=pickle.loads(pickle.dumps(result))
    assert isinstance(copy,tuple) and len(copy)==3
    assert copy.force_selection==result.force_selection
    for a,b in zip(copy,result):
        assert np.array_equal(a,b)
    assert copy.info['status']=='UNIQUE_LINEAR_SYSTEM'


def rotating_physical_scene():
    # Independent reconstruction of an in-model counterexample: a rotating box whose four corners all slip.
    # Unlike the mixed-label census, b is exactly J @ v_free here.
    J=np.zeros((12,6)); lam=np.zeros(12); mu=np.full(4,.5)
    v=np.array([10.5,0.,0.,0.,0.,1.])
    for c,(sx,sy) in enumerate(itertools.product((1,-1),repeat=2)):
        x,y,z=12*sx,5.5*sy,-1.
        J[3*c:3*c+3]=[[0,0,1,y,-x,0],[1,0,0,0,z,-y],[0,1,0,-z,0,x]]
        ut=(J@v)[3*c+1:3*c+3]
        lam[3*c]=1.; lam[3*c+1:3*c+3]=-.5*ut/np.linalg.norm(ut)
    Mi=np.diag([1.,1.,1.,12/125,3/145,12/697])
    vf=v-Mi@J.T@lam
    return R.Scene(J,Mi,vf,mu,np.array([[0,-1]]*4),.01),lam,v


def test_physical_rotation_can_change_wrench_velocity_and_energy():
    from scipy.optimize import root
    sc,lam,v=rotating_physical_scene(); G,b=sc.delassus(),sc.b()
    Q=np.vstack([sc.J.T,sc.minv_mat()@sc.J.T,v@sc.J.T])
    info=R.contact_diagnostics(lam,G,b,sc.mu,['slip']*4,observable=Q)
    assert info['rank']==11
    assert info['response_scope']=='LINEARIZED_RESPONSE_ONLY'
    assert not info['observable_invariant'][:6].all()
    assert not info['observable_invariant'][6:12].all()
    assert not info['observable_invariant'][-1]
    assert R.natural_residual(lam,G,b,sc.mu)<1e-12
    # Remove the redundant fourth normal equation and fix the final component.
    rows=[i for i in range(12) if i!=9]
    def residual(x):
        u=(G@x+b).reshape(4,3); out=np.zeros((4,3)); L=x.reshape(4,3)
        out[:,0]=u[:,0]
        out[:,1:]=L[:,1:]+.5*L[:,0,None]*u[:,1:]/np.linalg.norm(u[:,1:],axis=1)[:,None]
        return out.ravel()
    eps=1e-4; N=info['nullspace'][:,0]; z=N/N[-1]
    def lift(y):
        return np.r_[y,lam[-1]+eps]
    rr=root(lambda y:residual(lift(y))[rows],(lam+eps*z)[:-1],tol=1e-11)
    alternate=lift(rr.x)
    assert rr.success and np.max(abs(residual(alternate)))<1e-10
    assert R.natural_residual(alternate,G,b,sc.mu)<1e-10
    assert min(alternate[::3])>0
    assert np.linalg.norm(sc.J.T@(alternate-lam))>1e-5
    assert np.linalg.norm(sc.apply(alternate)-sc.apply(lam))>1e-5
    M=np.linalg.inv(sc.minv_mat()); va=sc.apply(alternate)
    assert abs((va@M@va-v@M@v)/2)>1e-5


def test_open_contact_transition_is_not_differentiable():
    dm,db,info=R.sensitivity(np.zeros(3),np.eye(3),np.zeros(3),[.3],['open'])
    assert not info['residual_differentiable'] and np.isnan(db).all()


def test_gpu_relaxed_diagnostic_removes_biased_rhs():
    sc=box_scene(); lam=R.solve_ncp_admm(sc.delassus(),sc.b(),sc.mu)[0]
    bs=GPU.Scene(sc.J,sc.Minv,sc.v_free,sc.mu,sc.body_pairs,sc.dt,bias=np.tile([0.,-.5,0.],4)).blocks()
    result=GPU.SolveResult(lam.reshape(-1,3),sc.apply(lam).reshape(-1,6),0.,0,0,False,bias_applied=False)
    info=result.diagnose_contacts(bs)
    expected=R.contact_diagnostics(lam,sc.delassus(),sc.b(),sc.mu,['slip']*4,observable=sc.J.T)
    assert np.allclose(info['singular_values'],expected['singular_values'])


def test_router_does_not_certify_an_unfinished_drop_sequence():
    cert=T.bilateral_certificate(np.eye(3),np.array([1.,0.,0.]),np.array([.3]),max_rounds=1)
    assert not cert['is_interior']


@pytest.mark.parametrize('labels,expected_rank', [
    (['open','slip','slip','slip'],12),
    (['open','stick','stick','stick'],9),
    (['open','open','stick','stick'],11),
    (['open','open','open','stick'],12),
    (['open']*4,12),
])
def test_open_patterns_can_remove_or_retain_ambiguity(labels,expected_rank):
    sc=box_scene(); lam,G,b=algebraic_branch(sc,labels)
    dm,db,info=R.sensitivity(lam,G,b,sc.mu,labels)
    assert info['rank']==expected_rank
    assert info['singular']==(expected_rank<12)
    assert not info['boundary']
    if expected_rank==12:
        assert np.isfinite(dm).all() and np.isfinite(db).all()
    else:
        assert np.isnan(dm).all() and np.isnan(db).all()


def test_mu_response_is_conditional_on_the_supplied_force_representative():
    sc=box_scene(); G,b=sc.delassus(),sc.b()
    lam,_,labels=R.solve_ncp_admm(G,b,sc.mu)
    N=R.contact_diagnostics(lam,G,b,sc.mu,labels)['nullspace']
    alt=lam+.01*N[:,0]
    assert R.natural_residual(alt,G,b,sc.mu)<1e-10
    a=R.sensitivity(lam,G,b,sc.mu,labels,observable=sc.J.T)[2]
    z=R.sensitivity(alt,G,b,sc.mu,labels,observable=sc.J.T)[2]
    assert np.allclose(sc.J.T@lam,sc.J.T@alt,atol=1e-12)
    assert np.max(abs(a['observable_dmu']-z['observable_dmu']))>1e-4
    assert a['scope']=='NUMERICAL_FIXED_BRANCH_AT_SUPPLIED_IMPULSE'


@pytest.mark.parametrize('scale',[1e-160,1.,1e160])
def test_relative_guard_does_not_accept_overflow_or_underflow(scale):
    A=np.diag([1.,0.,1.]); B=np.array([[0.],[scale],[0.]])
    X,info=guarded_response(A,B,rank_factor(A),selection='minimum_norm',
                            observable=np.array([[0.,scale,0.]]))
    assert not info['compatible'][0]
    assert not info['observable_invariant'][0]
    assert np.isnan(X).all() and np.isnan(info['observable']).all()
