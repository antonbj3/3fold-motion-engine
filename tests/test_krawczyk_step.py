from fractions import Fraction as Q
import pytest
import numpy as np
from motion_engine.ncp.krawczyk_step import certify_root_box


def scalar(root, jlo, jhi, inverse):
    value=root**3-root
    return certify_root_box(residual_id="cubic:x^3-x", radius=[Q(1,100)],
        force_lower=[value], force_upper=[value], jacobian_lower=[[jlo]],
        jacobian_upper=[[jhi]], preconditioner=[[inverse]])


def test_three_disjoint_boxes_do_not_become_one_global_root():
    # F=x^3-x has roots -1,0,1. The exact derivative bounds cover each box.
    middle = scalar(0, -1, Q(-9997,10000), -1)
    sides = [scalar(r, Q(19403,10000), Q(20603,10000), Q(1,2)) for r in [-1,1]]
    assert all(c.verdict == "UNIQUE_IN_BOX_IF_ENCLOSURES_VALID" for c in [middle]+sides)
    assert all(c.global_root_count == "UNKNOWN" for c in [middle]+sides)


def test_singular_zero_preconditioner_never_certifies():
    assert scalar(0, -1, -1, 0).verdict == "UNKNOWN"


def test_outward_uncertainty_can_break_inclusion_even_with_invertible_j():
    c = certify_root_box(residual_id="uncertain centre", radius=[Q(1,100)],
        force_lower=[Q(-1,10)], force_upper=[Q(1,10)],
        jacobian_lower=[[1]], jacobian_upper=[[1]], preconditioner=[[1]])
    assert c.verdict == "UNKNOWN" and c.inclusion_upper == 10


def test_weighted_norm_handles_unequal_coordinate_scales():
    c = certify_root_box(residual_id="coupled affine", radius=[Q(1,1000000),Q(1,1000)],
        force_lower=[0,0], force_upper=[0,0],
        jacobian_lower=[[2,1],[1,2]], jacobian_upper=[[2,1],[1,2]],
        preconditioner=[[Q(2,3),Q(-1,3)],[Q(-1,3),Q(2,3)]])
    assert c.verdict == "UNIQUE_IN_BOX_IF_ENCLOSURES_VALID" and c.contraction_upper == 0


@pytest.mark.parametrize("bad", [float("nan"),float("inf"),True])
def test_invalid_endpoints_are_unknown(bad):
    assert scalar(0,bad,1,1).verdict == "UNKNOWN"


def test_newton_majorizer_is_explicitly_refused():
    c = certify_root_box(residual_id="Ando",radius=[1],force_lower=[0],force_upper=[0],
        jacobian_lower=[[1]],jacobian_upper=[[1]],preconditioner=[[1]],
        jacobian_role="PSD_NEWTON_SURROGATE")
    assert c.verdict == "UNKNOWN"


def test_boundary_equality_and_reversed_interval_are_refused():
    assert scalar(0,0,2,1).verdict == "UNKNOWN"
    assert scalar(0,2,1,1).verdict == "UNKNOWN"


def test_nonzero_centre_force_moves_the_local_inclusion():
    # Shifted centre .005 with radius .01: the actual derivative box spans
    # [-.005,.015], so its upper endpoint is 3*(.015)^2-1.
    c=scalar(Q(1,200),-1,Q(-39973,40000),-1)
    assert c.verdict=="UNIQUE_IN_BOX_IF_ENCLOSURES_VALID"
    assert c.inclusion_upper==Q(40053,80000)


def test_signed_interval_matrix_with_nonzero_contraction():
    c=certify_root_box(residual_id="signed 2D",radius=[1,2],
        force_lower=[Q(-1,128),Q(-1,128)],force_upper=[Q(1,128),Q(1,128)],
        jacobian_lower=[[Q(31,16),Q(-17,16)],[Q(15,16),Q(47,16)]],
        jacobian_upper=[[Q(33,16),Q(-15,16)],[Q(17,16),Q(49,16)]],
        preconditioner=[[Q(3,7),Q(1,7)],[Q(-1,7),Q(2,7)]])
    assert c.verdict=="UNIQUE_IN_BOX_IF_ENCLOSURES_VALID"
    assert c.contraction_upper==Q(3,28)
    assert c.inclusion_upper==Q(25,224)
    assert c.global_root_count=="UNKNOWN"


def test_strict_force_boundary_with_exact_sub_float_difference():
    def linear(value):
        return certify_root_box(residual_id="x+f",radius=[1],force_lower=[value],
            force_upper=[value],jacobian_lower=[[1]],jacobian_upper=[[1]],preconditioner=[[1]])
    assert linear(1).verdict=="UNKNOWN"
    assert linear(1-Q(1,2**100)).verdict=="UNIQUE_IN_BOX_IF_ENCLOSURES_VALID"
    assert linear(1+Q(1,2**100)).verdict=="UNKNOWN"


@pytest.mark.parametrize('wrap',[np.int64,lambda x:Q(np.int64(x))])
def test_fixed_width_rational_does_not_certify_an_outside_root(wrap):
    # F=x/4+2^62 has its only root at -2^64, outside centre0/radius1.
    # Unnormalized Fraction(np.int64) multiplies 4*2^62 into zero.
    c=certify_root_box(residual_id="outside root",radius=[1],
        force_lower=[wrap(2**62)],force_upper=[wrap(2**62)],
        jacobian_lower=[[Q(1,4)]],jacobian_upper=[[Q(1,4)]],preconditioner=[[4]])
    assert c.verdict=="UNKNOWN"
    assert c.contraction_upper==0
    assert c.inclusion_upper==Q(2**64)
