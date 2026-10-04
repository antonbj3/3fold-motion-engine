"""Exact isolating witnesses can certify non-rational point roots."""
from fractions import Fraction as Q
from motion_engine.ncp.step_root_bundle import certify_step_roots


def test_two_irrational_roots_have_exact_rational_existence_boxes():
    pid = 'c' * 64
    boxes = []
    for center in (Q(-7, 5), Q(7, 5)):
        radius = Q(1, 20)
        boxes.append(dict(problem_id=pid, center=[center], radius=[radius],
            force_lower=[center**2-2], force_upper=[center**2-2],
            jacobian_lower=[[2*(center-radius)]],
            jacobian_upper=[[2*(center+radius)]],
            preconditioner=[[1/(2*center)]], jacobian_role='TRUE_RESIDUAL'))
    # Independent exact sign change: sqrt(2) is strictly within this box.
    assert Q(27, 20)**2 < 2 < Q(29, 20)**2
    result = certify_step_roots(problem_id=pid, domain_lower=[-2],
        domain_upper=[2], initial=[0], boxes=boxes)
    assert result['root_count_lower'] == 2
    assert result['verdict'] == 'PROVEN_MULTIPLE_IF_ENCLOSURES_VALID'
    assert result['collision_free_root_count_lower'] == 0
    assert result['accepted_step'] is False
    assert len(result['box_results']) == 2
    assert all(row['accepted'] for row in result['box_results'])
