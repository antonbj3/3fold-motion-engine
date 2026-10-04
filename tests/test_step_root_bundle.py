from fractions import Fraction as Q
from copy import deepcopy
from motion_engine.ncp.step_root_bundle import certify_step_roots, translated_rectangle_path

PID='a'*64

def box(center, radius=Q(1,100)):
    c=Q(center);r=Q(radius)
    # Exact rational interval enclosure for F=x^3-x on the selected boxes.
    lo,hi=c-r,c+r
    jlo=3*min(lo*lo,hi*hi)-1 if lo*hi>0 else Q(-1)
    jhi=3*max(lo*lo,hi*hi)-1
    return dict(problem_id=PID,center=[c],radius=[r],force_lower=[c**3-c],force_upper=[c**3-c],
        jacobian_lower=[[jlo]],jacobian_upper=[[jhi]],preconditioner=[[1/(3*c*c-1)]],jacobian_role='TRUE_RESIDUAL')

def panels():
    return [dict(origin=[0,0,0],size=[1,1],translation=[[0],[0],[0]]),
            dict(origin=[0,0,3],size=[1,1],translation=[[0],[0],[1]])]

def run(boxes,**kw):
    args=dict(problem_id=PID,domain_lower=[-2],domain_upper=[2],initial=[0],boxes=boxes,panels=panels())
    args.update(kw)
    return certify_step_roots(**args)

def test_three_existential_roots_with_free_paths():
    r=run([box(-1),box(0),box(1)])
    assert r['root_count_lower']==3
    assert r['collision_free_root_count_lower']==3
    assert r['verdict']=='PROVEN_MULTIPLE_WITH_FREE_PATHS_IF_ENCLOSURES_VALID'
    assert r['global_root_count']=='AT_LEAST_TWO_IF_ENCLOSURES_VALID'
    assert r['accepted_step'] is False

def test_single_box_never_global_unique():
    r=run([box(1)])
    assert r['root_count_lower']==1 and r['global_root_count']=='UNKNOWN'
    assert r['accepted_step'] is False

def test_instability_at_single_root_never_multiple():
    b=box(0);b.update(jacobian_lower=[[-1]],jacobian_upper=[[-1]],preconditioner=[[-1]])
    r=run([b])
    assert r['root_count_lower']==1 and r['global_root_count']=='UNKNOWN'

def test_empty_boxes_not_vacuously_accepted():
    r=run([])
    assert r['verdict']=='UNKNOWN' and r['root_count_lower']==0
    assert not r['accepted_step']

def test_duplicate_boxes_count_once():
    r=run([box(1),box(1)])
    assert r['root_count_lower']==1 and r['global_root_count']=='UNKNOWN'

def test_different_problem_not_composed():
    b=box(-1);b['problem_id']='b'*64
    r=run([box(1),b])
    assert r['root_count_lower']==1 and r['global_root_count']=='UNKNOWN'

def test_unbound_problem_rejected():
    r=run([box(1)],problem_id='unbound')
    assert r['verdict']=='UNKNOWN' and r['root_count_lower']==0

def test_outside_domain_existence_not_counted():
    r=run([box(-1),box(1)],domain_lower=[0])
    assert r['root_count_lower']==1 and r['global_root_count']=='UNKNOWN'

def test_surrogate_jacobian_refused():
    b=box(-1);b['jacobian_role']='PSD_NEWTON'
    r=run([b,box(1)])
    assert r['root_count_lower']==1

def test_nonzero_force_does_not_pass_by_root_label():
    b=box(-1);b['force_lower']=b['force_upper']=[10]
    r=run([b,box(1)])
    assert r['root_count_lower']==1

def test_exact_contraction_boundary_refused():
    b=box(0);b.update(jacobian_lower=[[-2]],jacobian_upper=[[0]])
    r=run([b])
    assert r['root_count_lower']==0

def test_missing_geometry_keeps_mathematical_multiplicity():
    r=run([box(-1),box(1)],panels=None)
    assert r['root_count_lower']==2 and r['collision_free_root_count_lower']==0
    assert r['verdict']=='PROVEN_MULTIPLE_IF_ENCLOSURES_VALID'
    assert not r['accepted_step']

def test_box_uncertainty_contact_cannot_use_safe_center():
    p=panels();p[1]['origin'][2]=Q(1,200)
    r=run([box(0)],panels=p)
    assert r['root_count_lower']==1 and r['collision_free_root_count_lower']==0

def test_affine_path_can_collide_between_free_endpoints():
    r=translated_rectangle_path(initial=[0],lower=[-4],upper=[-4],panels=panels())
    assert not r['safe']

def test_touching_endpoint_refused():
    r=translated_rectangle_path(initial=[0],lower=[-3],upper=[-3],panels=panels())
    assert not r['safe']

def test_negative_translation_map():
    p=panels();p[1]['translation'][2][0]=-1
    r=translated_rectangle_path(initial=[0],lower=[1],upper=[2],panels=p)
    assert r['safe']

def test_degenerate_rectangle_refused():
    p=panels();p[1]['size'][0]=0
    r=run([box(-1),box(1)],panels=p)
    assert r['collision_free_root_count_lower']==0

def test_malformed_bundle_resets_compound_verdict():
    r=run([box(1),{}])
    assert r['verdict']=='UNKNOWN' and r['global_root_count']=='UNKNOWN'
    assert r['root_count_lower']==0

def test_nonfinite_and_boolean_input_refused():
    for invalid in [float('inf'),float('nan'),True]:
        b=box(1);b['center']=[invalid]
        r=run([b])
        assert r['root_count_lower']==0 and r['verdict']=='UNKNOWN'

def test_local_boxes_at_three_roots_do_not_accept_step():
    r=run([box(-1),box(0),box(1)])
    assert not r['accepted_step']
