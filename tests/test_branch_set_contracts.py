import math
from fractions import Fraction as Q
import pytest
from motion_engine.ncp.branch_set import Child,Step,propagate,observe,rank_measurements

def test_witness_aliases_keep_edges_without_double_counting_physical_paths():
    def oracle(s):
        return Step((Child((s[0]+1,),('mode_one',)),Child((s[0]+1,),('mode_two',))),True)
    result=propagate([(0,)],oracle,3,verify=lambda p,c:c.state==(p[0]+1,))
    assert result.status=='COMPLETE'
    assert result.layers[-1]=={(3,):1}
    assert all(len(edges)==2 for edges in result.edges)
    assert all(s['histories']==1 and s['split_excess']==0 for s in result.stats)

@pytest.mark.parametrize('invalid',[math.nan,math.inf,-math.inf])
@pytest.mark.parametrize('field',['value','error','readout'])
def test_invalid_observation_numbers_never_delete_admissible_states(invalid,field):
    args=dict(states=((Q(0),),(Q(1),)),readout=lambda s:s[0],value=Q(0),error=Q(1))
    args[field]=(lambda s:invalid) if field=='readout' else invalid
    with pytest.raises(ValueError):observe(**args)

@pytest.mark.parametrize('invalid',[math.nan,math.inf,-math.inf])
@pytest.mark.parametrize('field',['error','resolution','cost','readout'])
def test_invalid_sensor_cannot_report_removal_of_all_states(invalid,field):
    m=dict(id='x',readout=lambda s:s[0],error=Q(1,10),resolution=Q(1),cost=Q(1),unit='m',scope='same endpoint')
    m[field]=(lambda s:invalid) if field=='readout' else invalid
    with pytest.raises(ValueError):rank_measurements(((Q(0),),(Q(1),)),[m])

def test_arbitrarily_large_exact_observations_keep_boundary():
    large=Q(10**500)
    assert observe(((large,),),lambda s:s[0],large,Q(0))==((large,),)
    m=dict(id='large',readout=lambda s:s[0],error=large,resolution=Q(1),cost=Q(1),unit='m',scope='same endpoint')
    assert rank_measurements(((large,),),[m])[0]['worst_survivors']==1

def test_one_bad_center_does_not_leave_a_valid_looking_partial_rank():
    m=dict(id='x',readout=lambda s:math.nan if s[0] else 0.,error=.1,resolution=1.,cost=1.,unit='m',scope='same endpoint')
    with pytest.raises(ValueError):rank_measurements(((0,),(1,)),[m])

def test_arithmetic_overflow_of_sensor_bands_is_invalid():
    m=dict(id='x',readout=lambda s:1.7e308,error=1.7e308,resolution=1.,cost=1.,unit='m',scope='same endpoint')
    with pytest.raises(ValueError):rank_measurements(((0,),),[m])

def test_invalid_witness_keeps_last_complete_frontier():
    def oracle(s):return Step((Child((s[0]+1,),('certificate',)),),True)
    result=propagate([(0,)],oracle,3,verify=lambda p,c:p[0]<1)
    assert result.status=='UNKNOWN' and result.reason=='INVALID_WITNESS'
    assert result.layers==[{(0,):1},{(1,):1}]

def test_all_exact_siblings_survive_and_unknown_coverage_is_sticky():
    result=propagate([(0,)],lambda s:Step(tuple(Child((j,),('mode',j)) for j in (0,1,2)),True),1,verify=lambda p,c:True)
    assert result.status=='COMPLETE' and set(result.layers[-1])=={(0,),(1,),(2,)}
    unknown=propagate([(0,)],lambda s:Step((Child((1,),('root',)),)),1,verify=lambda p,c:True)
    assert unknown.status=='UNKNOWN' and len(unknown.layers)==1

def test_different_memory_and_nearby_exact_states_do_not_merge():
    states=[(Q(0),'elastic'),(Q(0),'plastic'),(Q(1,10**30),'elastic')]
    result=propagate(states,lambda s:Step((Child(s,('retain',)),),True),3,verify=lambda p,c:c.state==p)
    assert result.status=='COMPLETE' and len(result.layers[-1])==3

def test_frontier_limit_returns_completed_frontier():
    result=propagate([(0,)],lambda s:Step(tuple(Child((j,),('mode',j)) for j in (1,2)),True),1,verify=lambda p,c:True,max_states=1)
    assert result.status=='UNKNOWN' and result.reason=='FRONTIER_BUDGET' and result.layers==[{(0,):1}]
