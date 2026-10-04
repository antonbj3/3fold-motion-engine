from fractions import Fraction as Q
import pytest
from motion_engine.ncp import equilibrium_ellipsoid as e


def test_coupled_equilibrium_bound_contains_external_primal():
    # Direct Cramer's rule gives x*=(1/11,7/11), ell*x*=9/11.
    b = e.equilibrium_readout([[4,1],[1,3]],[1,2],[0,0],[[2,0],[0,2]],[2,1],unit='N')
    assert b.lower < Q(9,11)
    assert b.upper > Q(9,11)
    assert b.centre == Q(1)
    assert b.support_squared == Q(25,16)
    assert b.status == 'CERTIFIED_LINEAR_MODEL'


def test_exact_equilibrium_zero_radius():
    b = e.equilibrium_readout([[4,1],[1,3]],[1,2],['1/11','7/11'],[[4,1],[1,3]],[2,1],unit='N')
    assert b.lower == Q(9,11)
    assert b.upper == Q(9,11)
    assert b.support_squared == 0


def test_tight_lower_matrix_in_one_dimension():
    b = e.equilibrium_readout([[3]],[1],[0],[[3]],[1],unit='m',bits=16)
    assert b.lower < Q(1,3)
    assert b.upper > Q(1,3)
    assert b.upper-b.lower < Q(1,3)+Q(1,2**15)


def test_exact_psd_zero_pivot_rule():
    assert e._psd([[Q(0),Q(1)],[Q(1),Q(1)]]) == False
    assert e._psd([[Q(0),Q(0)],[Q(0),Q(1)]]) == True


def test_sqrt_enclosure_is_outward():
    u = e._sqrt_upper(Q(2),20)
    assert u*u > 2
    assert (u-Q(1,2**20))**2 < 2
    assert e._sqrt_upper(Q(1,4),20) == Q(1,2)


@pytest.mark.parametrize('A,M', [([[0]],[[0]]), ([[1]],[[2]]), ([[1,2],[2,1]],[[1,0],[0,1]]), ([[2,1],[0,2]],[[1,0],[0,1]])])
def test_invalid_curvature_is_rejected(A,M):
    n = len(A)
    with pytest.raises(ValueError):
        e.equilibrium_readout(A,[1]*n,[0]*n,M,[1]*n,unit='N')


def test_float_input_is_not_trusted():
    with pytest.raises(ValueError):
        e.equilibrium_readout([[1.]],[1],[0],[[1]],[1],unit='N')
