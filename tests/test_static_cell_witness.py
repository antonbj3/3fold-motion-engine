"""Boundary and input-contract regressions for the opt-in static witness."""
from fractions import Fraction as F
from pathlib import Path
import json

import pytest

from motion_engine.ncp.static_cell_witness import certify_static_cell


def square_case():
    return dict(base_contacts_xy=[(-1,-1),(-1,1),(1,-1),(1,1)]*2,
                moving_ids=[4,5,6,7], yaw_deg=0, com_xy=[0,0],
                cell_center_xy=[0,0], cell_side=0, pair=[0,3],
                weight='1/4', level='981/200', height_max=1,
                pelvis_offset_max=0)


def test_closed_containment_boundary_and_exact_perturbation():
    case = square_case()
    result = certify_static_cell(**case)
    assert result['status'] == 'STATIC_FEASIBLE_AT_LEVEL'
    assert F(result['minimum_squared_edge_slack']) == 0
    case['level'] = F(981,200) + F(1,10**20)
    assert certify_static_cell(**case)['status'] == 'UNKNOWN'


def test_signed_containment_and_capacity_are_separate_obligations():
    case = square_case()
    case['com_xy'] = [100,100]
    result = certify_static_cell(**case)
    assert result['status'] == 'UNKNOWN'
    assert result['reason'] == 'ball-containment'
    case = square_case()
    case['pelvis_offset_max'] = 100
    result = certify_static_cell(**case)
    assert result['status'] == 'UNKNOWN'
    assert result['reason'] == 'friction-capacity'


@pytest.mark.parametrize('key,value', [
    ('level',0.05), ('level',True), ('level','1e999999999'),
    ('level','1/0'), ('level','nan'), ('level',2**257),
    ('level','1'*161), ('level',-1), ('height_max',0),
    ('pelvis_offset_max',-1), ('cell_side',-1), ('weight',0),
    ('weight','1/2'), ('moving_ids',[4,4,6,7]),
    ('moving_ids',[4,5,6,8]), ('moving_ids',[4,5,6]),
    ('moving_ids',[True,5,6,7]), ('pair',[1,1]), ('pair',[1,8]),
    ('pair',[1,True]), ('yaw_deg',20.5), ('yaw_deg',20.0),
    ('yaw_deg',True), ('yaw_deg',181), ('com_xy',[0,0,0]),
    ('cell_center_xy',[0.,0]), ('base_contacts_xy',[[0,0]]*7),
    ('base_contacts_xy',[[0,0,0]]*8),
])
def test_malformed_or_unsupported_input_fails_closed(key,value):
    case = square_case()
    case[key] = value
    with pytest.raises((TypeError,ValueError)):
        certify_static_cell(**case)


def test_degenerate_geometry_is_unknown_not_infeasible():
    case = square_case()
    case['base_contacts_xy'] = [(0,0)]*8
    assert certify_static_cell(**case)['status'] == 'UNKNOWN'


@pytest.mark.parametrize('geometry,pair', [('g2',[2,5]),('g8',[5,7]),('g9',[2,4])])
def test_actual_p4_geometry_with_new_envelope(geometry,pair):
    source = Path(__file__).resolve().parents[1]/'reproducibility/p4/raw/certified_regions.json'
    block = json.loads(source.read_text())['direction_cover'][geometry]
    # Explicit model decision: binary64 JSON constants become exact rationals.
    case = dict(base_contacts_xy=[[F(v) for v in p] for p in block['base_contacts_xy']],
                moving_ids=block['moving_ids'], yaw_deg=int(block['yaw_deg']),
                com_xy=[F(v) for v in block['com_xy']], cell_center_xy=['1/100','1/100'],
                cell_side='1/50',pair=pair,weight='1/26',level='1/20',
                height_max='11/10',pelvis_offset_max='1/10')
    assert certify_static_cell(**case)['status'] == 'STATIC_FEASIBLE_AT_LEVEL'
    case['level'] = 100
    assert certify_static_cell(**case)['status'] == 'UNKNOWN'


def test_zero_load_and_lower_load_monotonicity():
    case = square_case()
    for level in (0,1,F(981,200)):
        case['level'] = level
        assert certify_static_cell(**case)['status'] == 'STATIC_FEASIBLE_AT_LEVEL'
