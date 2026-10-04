import math
from fractions import Fraction as F
import numpy as np
import pytest
from motion_engine.ncp.compliant_step_risk import compliant_step_risk


def risk(mu,t,response):
    return compliant_step_risk(mu_max=mu,tangent_gram_upper=t,normal_response_upper=response)


def test_published_rod_two_coarse_step_roots_are_never_accepted():
    # Genot-Brogliato / Hogan-Kristiansen uniform rod sin=4/5 cos=3/5, mu=2.
    # G is SPD; frozen sliding normal coefficient is 52/25 - 2*36/25 = -4/5.
    G=np.array([[52/25,36/25,0],[36/25,73/25,0],[0,0,1.]])
    assert min(np.linalg.eigvalsh(G))>0
    h=F(1,100);k=F(10000);c=F(100);b=F(11,5)
    response=k*h*h+c*h;eps=1/response;p=F(-4,5)
    normal=-(h*b)/(p+eps)
    roots=[np.zeros(3),np.array([float(normal),-2*float(normal),0.])]
    for lam in roots:
        u=G@lam+np.array([float(h*b),1.,0.]);u[0]+=float(eps)*lam[0]
        assert lam[0]>=0 and u[0]>=-1e-15 and abs(lam[0]*u[0])<1e-15
        assert u[1]>0 and abs(lam[1]+2*lam[0])<1e-15
    assert not np.array_equal(roots[0],roots[1])
    assert risk(2.,73/25,float(response))['verdict']=='UNKNOWN'
    # Below the exact pole: only the open root is admissible, bound accepts at 1ms.
    hh=F(1,1000);rr=k*hh*hh+c*hh
    assert p+1/rr>0
    assert risk(2.,73/25,float(rr))['verdict']=='UNIQUE_MOTION_IF_BOUNDS_VALID'


def test_unknown_does_not_assert_multiple_roots():
    # The same exact rod has one BE root at 5ms; sufficient bound is conservative.
    h=F(1,200);response=10000*h*h+100*h
    assert F(-4,5)+1/response>0
    result=risk(2.,73/25,float(response))
    assert result['verdict']=='UNKNOWN' and result['refine_or_verify']


def test_equality_and_outward_rounding_against_exact_arithmetic():
    assert risk(1.,4.,1.)['verdict']=='UNKNOWN'
    for vals in [(0.46,59.6044763505843,1/33.70074952819428),
                 (.5,59.6044763505843,1/2442.85073595705),
                 (1.,4.,math.nextafter(1.,0.)),(1e-100,1e-100,1e-100)]:
        exact=F(vals[0])**2*F(vals[1])*F(vals[2])/4
        result=risk(*vals)
        assert math.isinf(result['risk_ratio_upper']) or F(result['risk_ratio_upper'])>=exact
        if result['verdict']=='UNIQUE_MOTION_IF_BOUNDS_VALID':assert exact<1


@pytest.mark.parametrize('vals',[(.5,1.,math.inf),(.5,math.inf,1.),(1e200,1e200,1e200)])
def test_unbounded_or_overflow_returns_unknown(vals):
    assert risk(*vals)['verdict']=='UNKNOWN'


@pytest.mark.parametrize('vals',[(0.,1.,math.inf),(.5,0.,1.),(.5,1.,0.)])
def test_zero_friction_or_zero_response_or_tangent(vals):
    result=risk(*vals)
    assert result['verdict']=='UNIQUE_MOTION_IF_BOUNDS_VALID'
    assert not result['existence_certified']
    assert not result['force_uniqueness_certified']
    assert not result['time_accuracy_certified']


@pytest.mark.parametrize('vals',[(-1.,1.,1.),(.5,-1.,1.),(.5,1.,-1.),(.5,math.nan,1.),(math.inf,1.,1.)])
def test_invalid_bounds_fail_explicitly(vals):
    with pytest.raises(ValueError):risk(*vals)


def test_initial_hertz_recipe_bounds_all_active_reference_gaps():
    from motion_engine.ncp.compliant_step_risk import initial_hertz_step_risk
    h=4e-5;an=1.5;va=140.;kh=5e10;eta=20.
    result=initial_hertz_step_risk(h=h,mu_max=.5,tangent_gram_upper=100.,
        normal_row_norm_upper=an,gravity_mass_norm_upper=va,
        hertz_coefficient_upper=kh,damping_eta_upper=eta)
    # Exact bound, independently written as a polynomial of sqrt(D0).
    d=F(h)**2*F(an)*F(va)
    # Bound uses an upward square root; every active g and delta is <=D0.
    r=F(result['normal_response_upper_kg'])
    exact_bracket=F(3,2)*F(h)+4*F(eta)*d
    assert r*r >= (F(h)*F(kh)*exact_bracket)**2*d
    assert result['verdict']=='UNIQUE_MOTION_IF_BOUNDS_VALID'
    assert result['scope']=='initial_rest_zero_prestrain_static_geometry'


def test_initial_hertz_dyadic_route_preserves_fixed_material():
    from motion_engine.ncp.compliant_step_risk import initial_hertz_step_risk
    args=dict(mu_max=.5,tangent_gram_upper=100.,normal_row_norm_upper=1.5,
              gravity_mass_norm_upper=140.,hertz_coefficient_upper=5e10,damping_eta_upper=20.)
    h=.004;history=[]
    for _ in range(20):
        r=initial_hertz_step_risk(h=h,**args);history.append(r['risk_ratio_upper'])
        if r['verdict']=='UNIQUE_MOTION_IF_BOUNDS_VALID':break
        h*=.5
    assert h<.004 and h>0 and history[-1]<1
    assert all(b<=a for a,b in zip(history,history[1:]))
    assert not r['time_accuracy_certified']


@pytest.mark.parametrize('h',[0.,-1.,math.inf,math.nan])
def test_initial_hertz_invalid_step(h):
    from motion_engine.ncp.compliant_step_risk import initial_hertz_step_risk
    with pytest.raises(ValueError):
        initial_hertz_step_risk(h=h,mu_max=.5,tangent_gram_upper=1.,
            normal_row_norm_upper=1.,gravity_mass_norm_upper=1.,
            hertz_coefficient_upper=1.,damping_eta_upper=1.)
