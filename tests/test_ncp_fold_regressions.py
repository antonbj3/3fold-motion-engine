"""Regression witnesses for the nonconvex discrete Coulomb solution set.

These do not claim that Newton converges to every root or select a physical branch.
Exact G/b witness and fold analysis are reproduced by the tests below.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq
from motion_engine.ncp import ncp_newton as N


def problem(mu=.5, copies=1):
    G=np.array([[1.,3.,1.],[3.,20.,6.],[1.,6.,2.]])
    J=sp.csr_matrix(np.c_[np.linalg.cholesky(G),np.zeros((3,3))])
    J=sp.block_diag([J]*copies,format='csr')
    return N.Problem(J,sp.eye(6*copies),sp.eye(6*copies),np.zeros(6*copies),
                     np.tile([.5,8.,2.],copies),np.full(copies,mu))


def parameterized(t):
    D=3*t**4+20*t**3-18*t*t-20*t+3
    P=3*t**4+26*t**3-26*t*t-26*t+3
    p=P/D;k=-(1+t*t)*(3*t*t+26*t-3)/(2*D)
    return np.array([p,-k*(1-t*t)/(1+t*t),-k*2*t/(1+t*t)]),k/p



def require_live_natural_map(pb, root):
    """A true root alone cannot distinguish a residual from a constant-zero stub."""
    bad = np.asarray(root, dtype=float).copy()
    bad[0] += .125
    scalar = pb.natres(bad)
    vector = N._natmap_parts(pb, bad)[0]
    assert scalar > 1e-6, "invalid impulse accepted by scalar natural residual"
    assert np.max(np.abs(vector)) > 1e-6, "invalid impulse accepted by vector natural map"
    np.testing.assert_allclose(np.max(np.abs(vector)), scalar, rtol=1e-12, atol=1e-14)


def test_exact_positive_load_sliding_fold_is_an_ncp_root():
    pb=problem();lam=np.array([1.,-.5,0.])
    require_live_natural_map(pb, lam)
    u=pb.G@lam+pb.b
    np.testing.assert_allclose(u,[0.,1.,0.],atol=2e-14)
    assert pb.natres(lam)<2e-14


def test_two_sliding_roots_and_open_root_coexist_above_fold():
    pb=problem(.505)
    require_live_natural_map(pb, np.zeros(3))
    roots=[]
    for lo,hi in [(-.1,-1e-6),(1e-6,.08)]:
        t=brentq(lambda t:parameterized(t)[1]-.505,lo,hi,xtol=1e-15)
        x,_=parameterized(t);roots.append(x)
        assert x[0]>0 and pb.natres(x)<1e-11
    assert pb.natres(np.zeros(3))==0
    assert np.linalg.norm(pb.Minv@(pb.J.T@(roots[0]-roots[1])))>1e-2


def test_direct_sum_has_two_singular_natural_map_directions():
    pb=problem(copies=2);x=np.tile([1.,-.5,0.],2);h=1e-6
    require_live_natural_map(pb, x)
    def residual(y):return N._natmap_parts(pb,y)[0].ravel()
    eye=np.eye(6)
    A=np.column_stack([(residual(x+h*e)-residual(x-h*e))/(2*h) for e in eye])
    sv=np.linalg.svd(A,compute_uv=False)
    assert sv[-1]<1e-8 and sv[-2]<1e-8
    assert sv[-3]>1e-3


def test_fold_curve_turns_in_mu_with_valid_positive_slip():
    xs=[]
    for t in [-.001,0.,.001]:
        lam,mu=parameterized(t);pb=problem(mu)
        require_live_natural_map(pb, lam)
        assert pb.natres(lam)<1e-12
        assert lam[0]>0 and np.linalg.norm((pb.G@lam+pb.b)[1:])>0
        xs.append(mu)
    assert xs[0]>xs[1] and xs[2]>xs[1]
