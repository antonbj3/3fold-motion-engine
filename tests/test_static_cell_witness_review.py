"""Independent arithmetic, geometric-boundary and full-wrench regressions."""
from fractions import Fraction as F
from math import factorial, isqrt

import pytest

from motion_engine.ncp import static_cell_witness as witness


def square_with_pair():
    return dict(base_contacts_xy=[(2,0),(-2,0),(-1,-1),(1,-1),
                                  (1,1),(-1,1),(0,0),(0,F(1,2))],
                moving_ids=[2,3,4,5],yaw_deg=0,com_xy=[0,0],
                cell_center_xy=[0,0],cell_side=0,pair=[0,1],
                weight=F(1,4),level=F(1,2),height_max=F(981,100),
                pelvis_offset_max=0)


def root96(x):
    den=1<<96
    n=isqrt((x.numerator*den*den)//x.denominator)
    lower=F(n,den)
    return lower,lower if lower*lower==x else F(n+1,den)


def test_whole_cell_erosion_closed_boundary_and_sign():
    case=square_with_pair()
    case['cell_side']=F(1,5)
    rho=root96(F(1,50))[1]
    assert rho*rho>=F(1,50)
    # E is [-1/2,1/2]^2. The certifier erodes by its proved rho upper bound.
    case['level']=F(1,2)-rho
    result=witness.certify_static_cell(**case)
    assert result['status']=='STATIC_FEASIBLE_AT_LEVEL'
    assert F(result['rho_upper'])==rho
    assert F(result['minimum_squared_edge_slack'])==0
    case['level']+=F(1,1<<40)
    assert witness.certify_static_cell(**case)['reason']=='ball-containment'
    case['level']=F(1,100)
    case['com_xy']=[1,0]
    result=witness.certify_static_cell(**case)
    assert result['status']=='UNKNOWN' and result['reason']=='ball-containment'


def test_capacity_exact_bound_equality_and_adjacent_levels():
    case=square_with_pair()
    r=root96(F(2))[0];r=root96(2+r)[0];c=root96(2+r)[0]/2
    tau=2*c/(1+c)
    case['height_max']=F(1,100)
    case['level']=tau*F(981,100)
    result=witness.certify_static_cell(**case)
    assert result['status']=='STATIC_FEASIBLE_AT_LEVEL'
    assert F(result['capacity_slack'])==0
    eps=F(1,1<<40)
    case['level']+=eps
    result=witness.certify_static_cell(**case)
    assert result['status']=='UNKNOWN' and result['reason']=='friction-capacity'
    case['level']-=2*eps
    assert witness.certify_static_cell(**case)['status']=='STATIC_FEASIBLE_AT_LEVEL'


def test_pair_singular_and_zero_load_on_hull_boundary_are_unknown():
    case=square_with_pair()
    case['base_contacts_xy'][1]=(2,0)
    result=witness.certify_static_cell(**case)
    assert result['status']=='UNKNOWN' and result['reason']=='pair-distance'
    case=square_with_pair();case['level']=0;case['com_xy']=[F(1,2),0]
    assert witness.certify_static_cell(**case)['status']=='UNKNOWN'


def test_rational_force_witness_balances_all_six_components():
    # A nonzero pelvis offset requires actual pair yaw correction. Build the
    # lemma's primal witness independently and check forces and every moment.
    case=square_with_pair();L=F(1,10);g=F(981,100);h=g;w=F(1,4)
    direction=(F(3,5),F(4,5));rp=(F(3,50),F(-2,25))
    case.update(level=L,pelvis_offset_max=F(1,10))
    assert witness.certify_static_cell(**case)['status']=='STATIC_FEASIBLE_AT_LEVEL'
    q=tuple(L*x for x in direction)
    normals=[w*g,w*g]
    for x,y in case['base_contacts_xy'][2:6]:
        normals.append((1-2*w)*g*(1+x*2*q[0])*(1+y*2*q[1])/4)
    normals.extend((F(0),F(0)))
    tangents=[[-N/g*L*v for v in direction] for N in normals]
    delta=-L*(rp[0]*direction[1]-rp[1]*direction[0])
    beta=-delta/4
    # perp(p_k-p_i)/ell = (0,-1).
    tangents[0][1]-=beta;tangents[1][1]+=beta
    assert all(N>=0 for N in normals)
    assert sum(normals)==g
    assert sum(t[0] for t in tangents)==-L*direction[0]
    assert sum(t[1] for t in tangents)==-L*direction[1]
    px=case['base_contacts_xy']
    moment=(sum(p[1]*N for p,N in zip(px,normals)),
            -sum(p[0]*N for p,N in zip(px,normals)),
            sum(p[0]*t[1]-p[1]*t[0] for p,t in zip(px,tangents)))
    external=(-h*L*direction[1],h*L*direction[0],-delta)
    assert all(a+b==0 for a,b in zip(moment,external))
    c=root96(2+root96(2+root96(F(2))[0])[0])[0]/2
    tau=2*c/(1+c)
    assert all(sum(v*v for v in t)<=tau*tau*N*N for t,N in zip(tangents,normals))


def test_fraction_with_numpy_components_cannot_trigger_fixed_width_arithmetic():
    np=pytest.importorskip('numpy')
    case=square_with_pair()
    case['weight']=F(np.int64(1),np.int64(4))
    result=witness.certify_static_cell(**case)
    assert result['status']=='STATIC_FEASIBLE_AT_LEVEL'
    assert result['weight']=='1/4'


def test_fraction_denominator_bound_checked_and_custom_subclasses_rejected():
    case=square_with_pair();case['level']=F(1,1<<256)
    with pytest.raises(ValueError):
        witness.certify_static_cell(**case)
    class IntegerSubclass(int):
        pass
    case=square_with_pair();case['cell_side']=IntegerSubclass(0)
    with pytest.raises(TypeError):
        witness.certify_static_cell(**case)


@pytest.mark.parametrize('deg',[-180,-70,0,20,60,120,180])
def test_independent_direct_taylor_enclosure_of_integral_yaw(deg):
    # Higher precision direct polynomial summation is independent of the
    # implementation's 96-bit rounded term recurrence.
    den=1<<128
    def add(a,b):
        return a[0]+b[0],a[1]+b[1]
    def mul(a,b):
        values=[x*y for x in a for y in b]
        return min(values),max(values)
    def scale(a,b):
        return mul(a,(b,b))
    def atan(n):
        low=sum((F((-1)**k,(2*k+1)*n**(2*k+1)) for k in range(80)),F(0))
        return low,low+F(1,161*n**161)
    raw=add(scale(atan(5),16),scale(atan(239),-4))
    pi=(F((raw[0].numerator*den)//raw[0].denominator,den),
        F(-((-raw[1].numerator*den)//raw[1].denominator),den))
    x=scale(pi,F(deg,180));powers=[(F(1),F(1))]
    for _ in range(65):
        powers.append(mul(powers[-1],x))
    s=(F(0),F(0));c=(F(0),F(0))
    for k in range(33):
        s=add(s,scale(powers[2*k+1],F((-1)**k,factorial(2*k+1))))
        c=add(c,scale(powers[2*k],F((-1)**k,factorial(2*k))))
    radius=max(abs(v) for v in x)
    se=radius**66/factorial(66);ce=radius**65/factorial(65)
    actual_s,actual_c=witness._sincos_deg(deg)
    assert actual_s.lo<=s[0]-se<=s[1]+se<=actual_s.hi
    assert actual_c.lo<=c[0]-ce<=c[1]+ce<=actual_c.hi
