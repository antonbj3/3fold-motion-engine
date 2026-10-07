"""Run with PYTHONPATH=src python examples/frozen_planar_contact_demo.py.

This checks a declared reduced contact solver fixture, not a full Scene.
SymPy is optional for Motion but required for COMPLETE inventories here.
"""
from fractions import Fraction as Q
import json
from motion_engine.ncp.frozen_planar_contact import (
    ExactEvent, ParameterInterval, inventory_single, inventory_coupled,
)
from motion_engine.ncp.krawczyk_step import certify_root_box


def singular_fixture_check():
    # R(g)=g-q+2eta(1-g)^2/g^2, the positive-slip true residual.
    eta, q, centre, radius = Q(27,64), Q(27,32), Q(3,4), Q(1,1000)
    derivative = lambda g: 1 + 4*eta*(g-1)/g**3
    # R''=4eta(3-2g)/g^4>0 on this box: these endpoints enclose R'.
    assert centre-q+2*eta*(1-centre)**2/centre**2 == 0
    local = certify_root_box(
        residual_id='frozen-planar-positive-slip-eta27/64-q27/32',
        radius=[radius], force_lower=[0], force_upper=[0],
        jacobian_lower=[[derivative(centre-radius)]],
        jacobian_upper=[[derivative(centre+radius)]], preconditioner=[[1]],
    )
    complete = inventory_single(q, eta=eta)
    return local, complete


def summary(result):
    return {key: result.get(key) for key in (
        'status', 'distinct_physical_states', 'mode_admissible_counts',
        'step_acceptance', 'reason',
    )}


def main():
    local, complete = singular_fixture_check()
    print(json.dumps({
        'use': 'Regression oracle for an explicitly matching reduced contact solver',
        'ordinary_exact_input': summary(inventory_single(Q(3,4))),
        'singular_true_residual_local_checker': {
            'verdict': local.verdict, 'global_root_count': local.global_root_count,
            'contraction_upper': str(local.contraction_upper),
        },
        'singular_complete_inventory': summary(complete),
        'singular_branches': [b['mode_aliases'] for b in complete['branches']],
        'exact_coupled_seam': summary(inventory_coupled(ExactEvent('pair_PT_TT_seam'))),
        'uncertain_input': summary(inventory_single(ParameterInterval(Q(84,100),Q(85,100)),eta=Q(27,64))),
        'interpretation': 'UNKNOWN is correct at the fold; the complete model has two states. A solver returning one state need not violate its contract. No physical branch is selected.',
    }, indent=2))


if __name__ == '__main__':
    main()
