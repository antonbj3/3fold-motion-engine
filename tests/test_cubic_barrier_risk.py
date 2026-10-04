"""Regressions distinguish updated-force curvature and whole-domain scope."""
from fractions import Fraction as F
import pytest
from motion_engine.ncp.cubic_barrier_risk import cubic_barrier_step_risk as gate

def args(**changes):
    out=dict(gap_lower=F(9,10),barrier_width_upper=1,gap_mass_energy_upper=1,
             elastic_stiffness_upper=0,timestep=1,mu_upper=2,tangent_gram_upper=1)
    out.update(changes)
    return out

def test_updated_force_derivative_prevents_false_accept_at_small_gap():
    # A source-style frozen block would produce gamma=4*(1/g²)*(1-g).
    # At g=2/5,h=1/5 that is 3/5<1; true updated derivative gives 3/2>1.
    r=gate(**args(gap_lower=F(2,5),timestep=F(1,5)))
    assert r['curvature_upper_exact']=='75/2'
    assert r['risk_ratio_exact']=='3/2'
    assert r['status']=='UNKNOWN'

def test_witness_all_root_box_rejected_and_single_root_box_stays_conditional():
    # Three exact source-law roots are ~.4477,.5,.8944. Cover all of them.
    assert gate(**args(gap_lower=F(2,5)))['status']=='UNKNOWN'
    r=gate(**args())
    assert r['status']=='AT_MOST_ONE_ON_DECLARED_GAP_DOMAIN'
    assert r['scope']=='DECLARED_GAP_DOMAIN'
    assert not r['all_solution_gap_floor_certified']
    assert not r['existence_certified']

def test_strict_boundary_and_width_rule():
    # g=1/2,w=1,eta=1 gives k=16; mu=1,h=1/2 gives gamma=1 exactly.
    r=gate(**args(gap_lower=F(1,2),mu_upper=1,timestep=F(1,2)))
    assert r['risk_ratio_exact']=='1'
    assert r['status']=='UNKNOWN'
    assert gate(**args(gap_lower=F(1,2),mu_upper=1,timestep=F(49,100)))['status'].startswith('AT_MOST_ONE')
    assert gate(**args(gap_lower=F(1,2),mu_upper=1,timestep=F(1,2),barrier_width_upper=F(99,100)))['status'].startswith('AT_MOST_ONE')

def test_zero_floor_and_no_friction():
    assert gate(**args(gap_lower=0))['status']=='UNKNOWN'
    assert gate(**args(gap_lower=0,mu_upper=0))['status'].startswith('AT_MOST_ONE')

@pytest.mark.parametrize('changes', [dict(gap_lower=-1),dict(timestep=0),
    dict(gap_mass_energy_upper=float('inf')),dict(mu_upper=True),dict(barrier_width_upper=0)])
def test_bad_bound_rejected(changes):
    with pytest.raises(ValueError): gate(**args(**changes))

def test_elastic_force_alone_can_prevent_uniqueness_certificate():
    # With eta=0, -f'(g)=4*e*(1-g/w). At g=1/2,w=e=1,
    # curvature is 2; mu=2,h=1 makes gamma=2, so certification is invalid.
    r=gate(**args(gap_lower=F(1,2),gap_mass_energy_upper=0,
                  elastic_stiffness_upper=1))
    assert r['curvature_upper_exact']=='2'
    assert r['risk_ratio_exact']=='2'
    assert r['status']=='UNKNOWN'

def test_tangent_gram_amplification_crosses_the_bound():
    # Doubling the tangential inverse-mass bound doubles gamma. The original
    # default fixture passes at 400/729; the doubled bound fails at 800/729.
    assert gate(**args())['status']=='AT_MOST_ONE_ON_DECLARED_GAP_DOMAIN'
    r=gate(**args(tangent_gram_upper=2))
    assert r['risk_ratio_exact']=='800/729'
    assert r['status']=='UNKNOWN'
