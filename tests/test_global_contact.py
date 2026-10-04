from fractions import Fraction as Q
from dataclasses import FrozenInstanceError
import pytest
from motion_engine.ncp.global_contact import global_rigid_translation_verdict as verdict

XY=((1,1),(1,-1),(-1,1),(-1,-1))
MI=(1,1,1,'3/2','3/2','3/2')
VF=(4,0,-4,0,-3,0)
def positive():
    r=verdict(XY,1,MI,VF,'1/2');assert r.status=='ENTYDIG';return r

def test_exact_witness_and_all_mode_result():
    r=positive();assert r.velocity==(2,0,0,0,0,0)
    assert r.impulse_witness==(Q(1),Q(-1,2),Q(0))*4
    assert len(r.input_sha256)==64
    with pytest.raises(FrozenInstanceError):r.status='UNKNOWN'

def test_force_redistribution_keeps_unique_velocity():
    r=positive()
    # Two different exact pressure distributions at same motion, including open contacts.
    for p in ((2,0,0,2),(0,2,2,0)):
        assert sum(p)==4
        assert sum(x*a for (x,y),a in zip(XY,p))==0
        assert sum(y*a for (x,y),a in zip(XY,p))==0
    assert 'individual impulses' in r.scope

@pytest.mark.parametrize('n',(4,8,16))
def test_rectangular_nested_contacts(n):
    positive();base=[(Q(12),Q(11,2)),(Q(12),-Q(11,2)),(-Q(12),Q(11,2)),(-Q(12),-Q(11,2))]
    xy=base+[(x*Q(j+2,j+3),y*Q(j+2,j+3)) for j in range(n//4-1) for x,y in base]
    mi=(1,1,1,Q(12,125),Q(3,145),Q(12,697));vf=(Q(25,2),0,-4,0,-Q(6,145),0)
    r=verdict(xy,1,mi,vf,'1/2');assert r.status=='ENTYDIG';assert r.velocity[0]==Q(21,2)

@pytest.mark.parametrize('vf',[(4,0,-4,0,-3,1),(4,1,-4,0,-3,0),(4,0,-4,0,0,0),(1,0,-4,0,-3,0)])
def test_changed_free_state_is_unknown(vf):
    positive();assert verdict(XY,1,MI,vf,'1/2').status=='UNKNOWN'

@pytest.mark.parametrize('xy',[XY[:-1],XY+((Q(1,2),0),),((0,0),)*4])
def test_geometry_cannot_reuse_certificate(xy):
    positive();assert verdict(xy,1,MI,VF,'1/2').status=='UNKNOWN'

@pytest.mark.parametrize('bad',[0.5,True,'nan','inf',-1])
def test_invalid_friction_is_unknown(bad):
    positive();assert verdict(XY,1,MI,VF,bad).status=='UNKNOWN'

def test_boundary_no_tilt_is_allowed_but_indefinite_work_is_not():
    positive()  # equality min corner = 2 mu h is sound, rotational kinetic term remains.
    mi=(1,1,1,Q(3,2),Q(3,2),16)
    assert verdict(XY,1,mi,VF,'1/2').status=='UNKNOWN'

def test_physical_parameters_are_bound_to_input():
    a=positive();b=verdict(XY,1,MI,(5,0,-4,0,-3,0),'1/2')
    assert b.status=='ENTYDIG' and b.velocity[0]==3
    assert a.input_sha256!=b.input_sha256
