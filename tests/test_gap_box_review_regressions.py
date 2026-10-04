"""Independent review regressions for existence and the enumeration budget."""
from copy import deepcopy
from motion_engine.ncp import gap_box as G


def test_empty_model_certificate_cannot_be_relabeled_unique():
    problem = G.GapProblem.make([[0]], [1], [0], [[-1, -1]], 1)
    answer = G.enclose_gap_box(problem)
    assert answer['status'] == 'OSÄKER'
    assert answer['records'][0]['kind'] == 'pruned'
    forged = deepcopy(answer)
    forged.update(complete=True, status='ENTYDIG')
    assert not G.verify_gap_enclosure(problem, forged)


def test_exhaustive_product_obeys_zero_node_budget(monkeypatch):
    real_product = G.product
    consumed = []

    def counted_product(*args, **kwargs):
        for item in real_product(*args, **kwargs):
            consumed.append(item)
            yield item

    monkeypatch.setattr(G, 'product', counted_product)
    problem = G.GapProblem.make([[1]] * 12, [1], [-1], [[0, 0]] * 12, 1)
    answer = G.enclose_gap_box(problem, exhaustive=True, max_nodes=0)
    assert answer['status'] == 'OSÄKER'
    assert answer['reason'] == 'PREFIX_BUDGET'
    assert answer['stats']['lp_calls'] == 0
    assert len(consumed) <= 1
