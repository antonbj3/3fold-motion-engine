"""Public solver, zero-friction and circular/polygon distinction regressions."""
from fractions import Fraction as Q
import pytest
from motion_engine.ncp.polyhedral_gap import PolyhedralGapProblem, DIAMOND, SQUARE, solve_polyhedral_gap
from motion_engine.ncp.gap_box import enclose_gap_box, verify_gap_enclosure
from motion_engine.ncp.coulomb_outer import SparseCoulombBox, energy_enclosure, verify_energy_enclosure

J = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


@pytest.mark.parametrize('polygon', [DIAMOND, SQUARE])
@pytest.mark.parametrize('normal_free,gap,expected_normal', [(-1, 0, 0), (0, 0, 0), (1, Q(1, 10), 1)])
def test_public_wrapper_covers_zero_friction_and_tangency(polygon, normal_free, gap, expected_normal):
    free = [normal_free, Q(-4, 5), Q(-3, 5)]
    pb = PolyhedralGapProblem.make(J, [1]*3, free, [[gap, gap]], 1, mu=[0], polygon=polygon)
    answer = solve_polyhedral_gap(J, [1]*3, free, [[gap, gap]], 1, mu=[0], polygon=polygon)
    assert answer['complete']
    assert verify_gap_enclosure(pb, answer)
    for index, expected in enumerate([expected_normal, Q(-4, 5), Q(-3, 5)]):
        bound = answer['bounds'][f'velocity:{index}']
        assert bound['lo'] == expected
        assert bound['hi'] == expected


def test_circular_slip_is_in_energy_ball_and_outside_polygon_solutions():
    free = [-1, Q(-4, 5), Q(-3, 5)]
    # Exact circle solution: pn=1, pt=(2/5,3/10), ut=(-2/5,-3/10).
    impulse = [Q(1), Q(2, 5), Q(3, 10)]
    velocity = [Q(0), Q(-2, 5), Q(-3, 10)]
    assert impulse[1]**2 + impulse[2]**2 == (impulse[0]/2)**2
    assert impulse[1]*velocity[2] == impulse[2]*velocity[1]
    assert impulse[1]*velocity[1] + impulse[2]*velocity[2] < 0
    outer = SparseCoulombBox.make([[(0,1)],[(1,1)],[(2,1)]], [1]*3, free, [[0,0]], 1, [Q(1,2)])
    answer = energy_enclosure(outer, [1,0,0])
    assert verify_energy_enclosure(outer, answer)
    for index, exact in enumerate(velocity):
        bound = answer['bounds'][f'velocity:{index}']
        assert bound['lo'] <= exact <= bound['hi']
    assert answer['existence'] == 'UNPROVED'
    for polygon in (DIAMOND, SQUARE):
        pb = PolyhedralGapProblem.make(J, [1]*3, free, [[0,0]], 1, mu=[Q(1,2)], polygon=polygon)
        polygon_answer = enclose_gap_box(pb)
        assert verify_gap_enclosure(pb, polygon_answer)
        bound = polygon_answer['bounds']['velocity:1']
        assert not bound['lo'] <= velocity[1] <= bound['hi']
