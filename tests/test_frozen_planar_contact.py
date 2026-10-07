"""Consumer contract: exact inventory supplements, never upgrades, local checker."""
from fractions import Fraction as Q
import importlib.util
from pathlib import Path
import pytest

pytest.importorskip('sympy')


def test_true_singular_residual_keeps_native_unknown_and_two_physical_states():
    path = Path(__file__).resolve().parents[1] / 'examples/frozen_planar_contact_demo.py'
    spec = importlib.util.spec_from_file_location('contact_demo', path)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    local, inventory = demo.singular_fixture_check()
    assert local.verdict == 'UNKNOWN'
    assert local.global_root_count == 'UNKNOWN'
    assert local.contraction_upper >= 1
    assert inventory['status'] == 'COMPLETE'
    assert inventory['distinct_physical_states'] == 2
    assert inventory['step_acceptance'] == 'NOT_ASSESSED'
    repeated = [b for b in inventory['branches'] if
                any(a['algebraic_multiplicity'] == 2 for a in b['mode_aliases'])]
    assert len(repeated) == 1
    assert list(map(Q, repeated[0]['gap1_interval'])) == [Q(3,4), Q(3,4)]


def test_per_mode_counts_include_inactive_without_dropping_active_branches():
    from motion_engine.ncp.frozen_planar_contact import inventory_single
    result = inventory_single(1, eta=2)
    assert result['mode_admissible_counts'] == {'P':1, 'T':1, 'I':1}
    assert result['distinct_physical_states'] == 3
