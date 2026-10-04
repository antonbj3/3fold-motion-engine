from fractions import Fraction as Q
from motion_engine.ncp.branch_set import Child,Step,propagate,observe,rank_measurements

def test_unknown_is_not_empty_or_success():
    out=propagate([(0,)],lambda s:Step((),False,'missing coverage'),3,verify=lambda s,c:True)
    assert out.status=='UNKNOWN' and len(out.layers)==1
    empty=propagate([(0,)],lambda s:Step((),True),3,verify=lambda s,c:True)
    assert empty.status=='COMPLETE' and not empty.layers[-1]

def test_budget_does_not_certify_truncated_frontier():
    oracle=lambda s:Step((Child((0,),('a',)),Child((1,),('b',))),True)
    a=propagate([(0,)],oracle,2,verify=lambda s,c:True,max_states=1)
    assert a.status=='UNKNOWN' and len(a.layers)==1 and a.reason=='FRONTIER_BUDGET'

def test_exact_merge_preserves_two_predecessor_histories():
    a=propagate([(0,),(1,)],lambda s:Step((Child((2,),('reset',s)),),True),1,verify=lambda s,c:True)
    assert a.stats[0]['merges']==1 and a.layers[-1][(2,)]==2 and len(a.edges[0])==2

def test_equal_kinematics_distinct_memory_never_merge():
    a=propagate([(0,'elastic'),(0,'plastic')],lambda s:Step((Child(s,('retain',)),),True),2,verify=lambda s,c:True)
    assert len(a.layers[-1])==2 and a.stats[-1]['merges']==0

def test_damping_contraction_is_not_finite_time_equality():
    a=propagate([(Q(0),),(Q(1),)],lambda s:Step((Child((s[0]/2,),('damp',)),),True),100,verify=lambda s,c:c.state[0]==s[0]/2)
    assert len(a.layers[-1])==2 and max(s[0] for s in a.layers[-1])==Q(1,2**100)

def test_bounded_observation_keeps_boundary_and_has_no_prior_collapse():
    states=((Q(0),),(Q(1),),(Q(2),))
    assert observe(states,lambda s:s[0],Q(1),Q(1))==states
    assert observe(states,lambda s:s[0],Q(1),Q(0))==((Q(1),),)
    assert not observe(states,lambda s:s[0],Q(8),Q(0))

def test_partial_measurement_split_without_disjoint_all_bands():
    states=((Q(0),Q(0)),(Q(0),Q(1)),(Q(1),Q(0)))
    sensors=[dict(id='x',readout=lambda s:s[0],error=Q(1,10),resolution=Q(1,10),cost=Q(1),unit='m',scope='same endpoint')]
    r=rank_measurements(states,sensors)[0]
    assert r['worst_survivors']==2 and r['farkas_score']==0 and r['guaranteed_removed']==1

def test_closed_bands_touching_have_two_survivors():
    m=dict(id='x',readout=lambda s:s[0],error=Q(1),resolution=Q(1),cost=Q(1),unit='m',scope='same endpoint')
    assert rank_measurements([(Q(0),),(Q(2),)],[m])[0]['worst_survivors']==2

def test_witness_list_does_not_default_to_complete():
    out=propagate([(0,)],lambda s:Step((Child((1,),('sample',)),)),1,verify=lambda s,c:True)
    assert out.status=='UNKNOWN' and out.reason=='ORACLE_COVERAGE_UNVERIFIED'
