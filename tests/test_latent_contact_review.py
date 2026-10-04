"""A shared cause does not imply symmetry when the response is asymmetric."""
from fractions import Fraction as Q
from motion_engine.ncp.latent_contact import make_family, certify_family, verify_family


def test_shared_cause_through_asymmetric_coupling_has_nonconstant_difference():
    problem = make_family([[3, 1], [1, 2]], [-1, -1], [Q(1, 2)]*2,
        [-1, 1], step=1, source_id='review-asymmetric', source_sha256='f'*64)
    answer = certify_family(problem)
    assert verify_family(problem, answer)
    # Independent 2x2 elimination: p=(1/5,2/5)*(1-z/2).
    assert len(answer['cells']) == 1
    assert answer['cells'][0]['p0'] == [Q(1, 5), Q(2, 5)]
    assert answer['cells'][0]['p1'] == [Q(-1, 10), Q(-1, 5)]
    assert answer['bounds']['force_difference:0:1']['lo'] == Q(-3, 10)
    assert answer['bounds']['force_difference:0:1']['hi'] == Q(-1, 10)
