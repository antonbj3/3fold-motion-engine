"""Exact routing boundaries for the sufficient global contact certificate."""
from fractions import Fraction as Q
import pytest
from motion_engine.ncp.global_contact import global_rigid_translation_verdict as verdict

XY=((1,1),(1,-1),(-1,1),(-1,-1))
MI=(1,1,1,Q(3,2),Q(3,2),Q(3,2))


@pytest.mark.parametrize('mu',(Q(1,2)-Q(1,2**80),Q(1,2),Q(1,2)+Q(1,2**80)))
def test_no_tilt_boundary_uses_exact_geometry(mu):
    vf=(2+4*mu,0,-4,0,-MI[4]*mu*4,0)
    result=verdict(XY,1,MI,vf,mu)
    assert result.status==('ENTYDIG' if mu<=Q(1,2) else 'UNKNOWN')
    if result.status=='ENTYDIG': assert result.velocity==(2,0,0,0,0,0)


@pytest.mark.parametrize('speed',(Q(1,2**80),Q(0),-Q(1,2**80)))
def test_positive_slip_is_distinguished_from_rest_exactly(speed):
    result=verdict(XY,1,MI,(2+speed,0,-4,0,-3,0),Q(1,2))
    assert result.status==('ENTYDIG' if speed>0 else 'UNKNOWN')
    if speed>0: assert result.velocity[0]==speed


def test_positive_tangential_work_form_requires_strict_inequality():
    for Iz in (Q(1,16)-Q(1,2**80),Q(1,16),Q(1,16)+Q(1,2**80)):
        mi=MI[:5]+(1/Iz,)
        result=verdict(XY,1,mi,(4,0,-4,0,-3,0),Q(1,2))
        assert result.status==('ENTYDIG' if Iz>Q(1,16) else 'UNKNOWN')
