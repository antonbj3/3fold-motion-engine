"""Independent exact coupled-compliance solutions and model grip endpoints."""
from fractions import Fraction as Q
from itertools import product
import pytest
from motion_engine.ncp.coulomb_outer import SparseCoulombBox, energy_enclosure
from motion_engine.ncp.virtual_budget import virtual_budget_enclosure, verify_virtual_budget

@pytest.mark.parametrize('g1,g2',list(product([Q(-1,10),Q(0),Q(1,10)],repeat=2)))
def test_cluster_cap_covers_independent_coupled_solution(g1,g2):
    # Direct inverse of [[2,-1],[-1,3]]; vf=(-1,-1), both active.
    p1=(3*(1-g1)-g2)/5
    p2=((1-g1)-2*g2)/5
    velocity=(-1+p1-p2,-1+p2)
    assert p1>0 and p2>=0
    assert velocity[0]+p1+g1==0
    assert -velocity[0]+velocity[1]+p2+g2==0
    pb=SparseCoulombBox.make([[(0,1)],[],[],[(0,-1),(1,1)],[],[]],
        [1,1],[-1,-1],[['-1/10','1/10']]*2,1,[0,0],compliance=[1,1])
    seed=energy_enclosure(pb,[1,2])
    a=virtual_budget_enclosure(pb,seed,energy_fields=[['1/10','1/5']],
        budget_fields=[[1,1]],clusters=[('body',(0,1))])
    assert a['cluster_force_bounds']['body']['hi']>=p1+p2
    assert verify_virtual_budget(pb,a)
    for i,value in enumerate(velocity):
        assert a['bounds'][f'velocity:{i}']['lo']<=value<=a['bounds'][f'velocity:{i}']['hi']

@pytest.mark.parametrize('gap,normal_N',[('-1/500',20),('-3/2500',12)])
def test_grip_N_is_model_force_with_step_conversion(gap,normal_N):
    h=Q(1,100);g=Q(gap)
    pb=SparseCoulombBox.make([[(0,1)],[(1,1)],[(2,1)],[(0,-1)],[(1,1)],[(2,1)]],
        [1,1,1],[0,'-981/10000','-3/100'],[[g,g]]*2,h,['1/2']*2,compliance=[1,1])
    seed=energy_enclosure(pb,[0,0,0])
    # Opposing normals have no strict global separator in translation space.
    assert seed['status']=='OSÄKER'
    # The claim is the separate exact SPD normal model, not an energy certificate.
    p=-g/h
    assert p/h==normal_N
    assert (p)**2>Q(981,10000)**2+Q(3,100)**2
    assert 2*p-p+g/h==0
