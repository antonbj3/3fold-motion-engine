"""Independent acceptance of the declared physical equations, not source mirrors."""
from fractions import Fraction as Q
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest
try:
    import sympy as s
except ImportError:
    s=None

requires_sympy=pytest.mark.skipif(s is None,reason='optional SymPy required for exact physical-root checks')

from motion_engine.ncp.frozen_planar_contact import (
    ExactEvent, ParameterInterval, inventory_single, inventory_coupled,
)
from motion_engine.ncp.krawczyk_step import certify_root_box


def _parameter_field(answer):
    p=answer['parameter']
    if p['kind']=='EXACT_RATIONAL':
        return s.QQ,s.QQ.convert(s.Rational(p['value'])),(Q(p['value']),Q(p['value']))
    t=s.Symbol('t');poly=s.Poly.from_list(list(map(s.Rational,p['polynomial'])),t)
    a,b=map(s.Rational,p['isolating_interval'])
    if a==b:return s.QQ,s.QQ.convert(a),(Q(a),Q(a))
    assert poly.count_roots(a,b)==1
    index=poly.count_roots(-s.oo,a)
    K=s.QQ.algebraic_field(s.CRootOf(poly,int(index)))
    a,b=poly.refine_root(a,b,eps=s.Rational(1,2**1024))
    return K,K.unit,(Q(a),Q(b))


def _decode_poly(rows,K,parameter,g):
    coefficients=[]
    for row in rows:
        coefficient=K.zero
        for q in row:coefficient=coefficient*parameter+K.convert(s.Rational(q))
        coefficients.append(coefficient)
    return s.Poly.from_list(coefficients,g,domain=K)


def _coefficient_interval(v,K,parameter_box):
    """Independent coefficient power intervals; no candidate field-sign calls."""
    if not v:return Q(0),Q(0)
    coefficients=v.to_list() if K.is_AlgebraicField else [v]
    lo=hi=Q(0);a,b=parameter_box
    # Every named event lies in a positive interval, so powers are ordered.
    assert a>0 and b>0
    degree=len(coefficients)-1
    for i,c in enumerate(coefficients):
        coefficient=Q(int(c.numerator),int(c.denominator));power=degree-i
        terms=(coefficient*a**power,coefficient*b**power)
        lo+=min(terms);hi+=max(terms)
    return lo,hi


def _sign_in_embedding(v,K,parameter_box):
    if not v:return 0
    lo,hi=_coefficient_interval(v,K,parameter_box)
    assert lo>0 or hi<0,'independent sign precision insufficient'
    return 1 if lo>0 else -1


def _root_count_in_receipt(poly,K,parameter_box,a,b):
    if a==b:return int(not poly.rep.eval(K.convert(s.Rational(a))))
    sequence=poly.sturm()
    def changes(x):
        signs=[_sign_in_embedding(z.rep.eval(K.convert(s.Rational(x))),K,parameter_box)
               for z in sequence]
        signs=[v for v in signs if v]
        return sum(v!=w for v,w in zip(signs,signs[1:]))
    return changes(a)-changes(b)-int(not poly.rep.eval(K.convert(s.Rational(b))))


def _verify_original_residuals(answer):
    """Root factors must satisfy the ORIGINAL rational force equations."""
    assert answer['status']=='COMPLETE' and answer['complete'] is True
    g=s.Symbol('g');K,q,parameter_box=_parameter_field(answer)
    const=lambda v:K.convert(s.Rational(v))
    ground=lambda v:s.Poly.from_dict({(0,):v},g,domain=K)
    x=s.Poly(g,g,domain=K);one=s.Poly(1,g,domain=K)
    seen=[]
    for branch in answer['branches']:
        if branch.get('gap1_exact_parameter'):
            assert branch['mode_aliases']==[dict(mode='I',algebraic_multiplicity=1)]
            assert _sign_in_embedding(q-const(1),K,parameter_box)>=0
            continue
        factor=_decode_poly(branch['gap1_polynomial'],K,q,g)
        a,b=map(Q,branch['gap1_interval'])
        assert 0<a<=b<1 and _root_count_in_receipt(factor,K,parameter_box,a,b)==1
        assert branch['distinct_state_multiplicity']==1
        seen.append((a,b))
        def vanishes_at_selected_root(residual):
            if (residual%factor).is_zero:return True
            common=s.gcd(residual,factor)
            return common.degree()>0 and _root_count_in_receipt(common,K,parameter_box,a,b)==1
        def physical_sign(poly):
            if vanishes_at_selected_root(poly):return 0
            low=high=Q(0);degree=poly.degree()
            for i,c in enumerate(poly.rep.to_list()):
                cl,ch=_coefficient_interval(c,K,parameter_box);power=degree-i
                values=[cl*a**power,cl*b**power,ch*a**power,ch*b**power]
                low+=min(values);high+=max(values)
            assert low>0 or high<0,'independent guard enclosure insufficient'
            return 1 if low>0 else -1
        if answer['model_id'].startswith('frozen-planar-pair'):
            N=_decode_poly(branch['reconstruction_numerator'],K,q,g)
            D=_decode_poly(branch['reconstruction_denominator'],K,q,g)
            assert D==(x*x).mul_ground(const(Q(2,5)))
            assert physical_sign(N)>0 and physical_sign(N-D)<0 and physical_sign(D)>0
            ylo,yhi=map(Q,branch['gap2_interval']);assert 0<ylo<=yhi<1
            for alias in branch['mode_aliases']:
                modes=alias['mode'];aa={'P':const(-1),'T':const(Q(1,2))}
                oo={'P':const(0),'T':const(Q(-3,8))}
                a1,a2=aa[modes[0]],aa[modes[1]];u1=q+oo[modes[0]];u2=const(Q(7,10))+oo[modes[1]];c=const(Q(1,10))
                # Multiply original normal residuals by4*x^2*N^2*D.
                r1=(x-ground(u1))*x*x*N*N*D*4-((one-x)**2*N*N*D).mul_ground(a1)-((D-N)**2*x*x*D).mul_ground(c)
                r2=N*x*x*N*N*4-(x*x*N*N*D).mul_ground(u2)*4-((one-x)**2*N*N*D).mul_ground(c)-((D-N)**2*x*x*D).mul_ground(a2)
                assert vanishes_at_selected_root(r1) and vanishes_at_selected_root(r2)
                first=physical_sign(x-ground(const(Q(1,2))))
                second=physical_sign(N-D.mul_ground(const(Q(1,2))))
                assert first>=0 if modes[0]=='P' else first<=0
                assert second>=0 if modes[1]=='P' else second<=0
                assert alias['algebraic_multiplicity']>=1
            if len(branch['mode_aliases'])>1:
                # Shared first-contact mode seams occur EXACTLY atg1=.5.
                assert a==b==Q(1,2)
        else:
            eta=const(answer['eta']);force_numerator=((one-x)**2).mul_ground(2*eta)
            force_guard=x*x-force_numerator*4
            for alias in branch['mode_aliases']:
                if alias['mode']=='P':residual=(x-ground(q))*x*x+force_numerator
                else:residual=(x-ground(q)+ground(const(Q(3,8))))*x*x-force_numerator.mul_ground(const(Q(1,2)))
                assert vanishes_at_selected_root(residual)
                sign=physical_sign(force_guard)
                assert sign>=0 if alias['mode']=='P' else sign<=0
    assert len(answer['branches'])==answer['distinct_physical_states']
    for left,right in zip(sorted(seen),sorted(seen)[1:]):assert left[1]<right[0]


@pytest.mark.parametrize('q,expected',[(Q(667,800),1),(Q(27,32),2),(Q(683,800),3)])
@requires_sympy
def test_documented_true_force_rational_fold_and_sides(q,expected):
    answer=inventory_single(q,eta=Q(27,64))
    assert answer['distinct_physical_states']==expected
    _verify_original_residuals(answer)
    if q==Q(27,32):
        double=next(b for b in answer['branches'] if b['mode_aliases'][0]['mode']=='P')
        assert double['gap1_interval']==['3/4','3/4']
        assert double['mode_aliases'][0]['algebraic_multiplicity']==2
        # Actual derivative zero means the regular checker should remainUNKNOWN.
        # f'= -4eta(1-g)/g^3 is increasing on this entire positive gap box.
        eta=Q(27,64);left=Q(3,4)-Q(1,100);right=Q(3,4)+Q(1,100)
        true_derivative=lambda x:1-4*eta*(1-x)/x**3
        assert -1<true_derivative(left)<0<true_derivative(right)<1
        cert=certify_root_box(residual_id='declared-reduced-positive-slip-fold',radius=[Q(1,100)],
          force_lower=[0],force_upper=[0],jacobian_lower=[[-1]],jacobian_upper=[[1]],preconditioner=[[1]])
        assert cert.verdict=='UNKNOWN' and cert.global_root_count=='UNKNOWN'


@requires_sympy
def test_exact_original_irrational_fold_has_two_distinct_states():
    answer=inventory_single(ExactEvent('one_fold'))
    assert answer['distinct_physical_states']==2
    assert sorted(a['algebraic_multiplicity']for b in answer['branches']for a in b['mode_aliases'])==[1,2]
    _verify_original_residuals(answer)


@pytest.mark.parametrize('q,expected',[(Q(3,4),2),(Q(1),1),(Q(2),1)])
@requires_sympy
def test_single_closed_seam_and_inactive_activation(q,expected):
    answer=inventory_single(q)
    assert answer['distinct_physical_states']==expected
    _verify_original_residuals(answer)
    if q==Q(3,4):assert sum(len(b['mode_aliases'])==2 for b in answer['branches'])==1
    else:assert len(answer['branches'])==1 and answer['branches'][0]['mode_aliases'][0]['mode']=='I'


@requires_sympy
def test_inactive_activation_does_not_erase_other_eta_dependent_active_states():
    # Unlike canonical86, the admitted eta=2 law has active states EVEN atq=1.
    # g=1 is inactive, while g=-2+2sqrt(2) is a distinct licensed slip state.
    answer=inventory_single(Q(1),eta=Q(2))
    assert answer['distinct_physical_states']==3
    assert sorted(a['mode']for b in answer['branches']for a in b['mode_aliases'])==['I','P','T']
    _verify_original_residuals(answer)


@pytest.mark.parametrize('name,expected',[
 ('pair_inactive_fold',7),('pair_PT_TT_seam',6),
 ('pair_PP_TP_seam_low',4),('pair_PP_TP_seam_high',2),('pair_inactive_seam',1)])
@requires_sympy
def test_all_five_previous_guardbands_at_exact_events(name,expected):
    answer=inventory_coupled(ExactEvent(name))
    assert answer['distinct_physical_states']==expected
    _verify_original_residuals(answer)
    assert not any(a['mode'].startswith('I') or a['mode'].endswith('I')
                   for b in answer['branches']for a in b['mode_aliases'])


@pytest.mark.parametrize('value',[True,0.75,float('nan'),complex(1),ParameterInterval(Q(7,10),Q(4,5)),ExactEvent('unknown')])
def test_unsupported_or_uncertain_input_never_snaps_to_event(value):
    for fn in [inventory_single,inventory_coupled]:
        result=fn(value)
        assert result['status']=='UNKNOWN' and result['complete'] is False
        assert result['distinct_physical_states'] is None and result['branches']==[]


def test_budget_and_wrong_physical_binding_decline_without_partial_admission():
    for result in [inventory_single(Q(3,4),max_work=0),inventory_coupled(Q(7,10),max_work=1),
                   inventory_coupled(Q(7,10),q2=Q(71,100)),
                   inventory_single(ExactEvent('one_fold'),eta=Q(27,64))]:
        assert result['status']=='UNKNOWN' and result['complete'] is False and result['branches']==[]


def test_fixed_width_fraction_components_are_normalized_or_declined():
    wrapped=Q(np.int64(3),np.int64(4))
    answer=inventory_single(wrapped)
    assert answer['status'] in ['UNKNOWN','COMPLETE']
    if answer['status']=='COMPLETE':assert answer['distinct_physical_states']==2


def test_optional_sympy_does_not_break_import_or_base_checker():
    project=Path(__file__).resolve().parents[1]
    code='''import builtins
old=builtins.__import__
def guarded(name,*args,**kwargs):
    if name=="sympy" or name.startswith("sympy."):raise ImportError("deliberately absent optional dependency")
    return old(name,*args,**kwargs)
builtins.__import__=guarded
from motion_engine.ncp.krawczyk_step import certify_root_box
from motion_engine.ncp.frozen_planar_contact import inventory_single
answer=inventory_single(0)
assert answer["status"]=="UNKNOWN" and answer["reason"]=="OPTIONAL_SYMPY_UNAVAILABLE"
'''
    env=dict(os.environ,PYTHONPATH=str(project/'src'),PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable,'-c',code],env=env,check=True,timeout=10)
