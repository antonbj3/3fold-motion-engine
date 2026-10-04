"""EE coverage and an exact interior-time crossing with coupled impulses."""
from fractions import Fraction as Q
from motion_engine.ncp.latent_contact import make_family, certify_family
from motion_engine.ncp.family_sweep import certify_sweep, verify_sweep


def scene(clearance):
    problem = make_family([[2, 1], [1, 2]], [-5, -6], [7, -7], [-1, 1],
        step=Q(1, 100), source_id='review-EE', source_sha256='e'*64)
    family = certify_family(problem)
    cells = []
    for cell in family['cells']:
        vertices = [[0, 0, clearance], [1, 0, clearance], [0, -1, 0], [0, 1, 0]]
        base = [[list(map(Q, v)) for v in vertices] for _ in range(2)]
        slope = [[[Q(0), Q(0), Q(1, 10)] for _ in range(4)] for _ in range(2)]
        for vertex, impulse in ((0, 0), (1, 0), (2, 1), (3, 1)):
            base[1][vertex][2] += Q(1, 100)*cell['p0'][impulse]
            slope[1][vertex][2] += Q(1, 100)*cell['p1'][impulse]
        cells.append(dict(base=base, slope=slope))
    return problem, family, [dict(id='EE', kind='EE', skin=0, cells=cells)]


def test_edge_edge_proof_covers_every_latent_cell():
    problem, family, geometry = scene(Q(1, 5))
    answer = certify_sweep(problem, family, geometry, complete=True)
    assert verify_sweep(problem, family, geometry, answer, complete=True)
    assert len(answer['certificates']) == len(family['cells'])
    assert min(c['margin'] for c in answer['certificates']) == Q(27, 200)


def test_edge_edge_interior_time_intersection_is_refused():
    problem, family, geometry = scene(Q(1, 100))
    answer = certify_sweep(problem, family, geometry, complete=True)
    assert answer['status'] == 'UNKNOWN'
    cell = geometry[0]['cells'][-1]
    start = cell['base'][0][0][2] - cell['base'][0][2][2]
    finish = (cell['base'][1][0][2] + cell['slope'][1][0][2]
              - cell['base'][1][2][2] - cell['slope'][1][2][2])
    assert start > 0
    assert finish < 0
    t = start/(start-finish)
    assert 0 < t < 1
    assert (1-t)*start + t*finish == 0
