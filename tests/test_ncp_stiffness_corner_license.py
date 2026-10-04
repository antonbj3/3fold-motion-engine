"""Fixed slip: column-affine Cramer maps license velocity/impulse corners.

These are conditional algebra regressions, separate from refreshed Coulomb
directions. Diagonal parameters are relative compliance. Kinetic energy is
convex in velocity, so only its maximum inherits the corner license.
"""
from fractions import Fraction as Q
from itertools import product
import numpy as np
import pytest
from motion_engine.ncp.ncp_ref import natural_residual


def det(a):
    a=[list(row) for row in a]; value=Q(1)
    for i in range(len(a)):
        k=next((k for k in range(i,len(a)) if a[k][i]),None)
        if k is None: return Q(0)
        if k!=i: a[i],a[k]=a[k],a[i];value=-value
        pivot=a[i][i];value*=pivot
        for k in range(i+1,len(a)):
            c=a[k][i]/pivot
            for j in range(i+1,len(a)): a[k][j]-=c*a[i][j]
    return value


def solve(a,b):
    q=det(a)
    return [det([[b[i] if j==k else row[j] for j in range(len(a))]
                 for i,row in enumerate(a)])/q for k in range(len(a))]


def fixture(h=Q(1)):
    xy=((Q(12),Q(11,2)),(Q(12),-Q(11,2)),(-Q(12),Q(11,2)),(-Q(12),-Q(11,2)))
    mi=(Q(1),Q(1),Q(1),Q(12,125),Q(3,145),Q(12,697))
    v0=(Q(21,2),Q(0),Q(0),Q(0),Q(0),Q(1))
    J=[];W=[]
    for x,y in xy:
        block=((0,0,1,y,-x,0),(1,0,0,0,-h,-y),(0,1,0,h,0,x))
        J.extend(block);ux=v0[0]-y;uy=x
        speed=Q(13) if y>0 else Q(20)
        W.append(tuple(row[0]-ux*row[1]/(2*speed)-uy*row[2]/(2*speed) for row in zip(*block)))
    W=list(map(list,zip(*W)));N=J[::3]
    vf=[v0[k]-mi[k]*sum(W[k]) for k in range(6)]
    A=[[sum(N[i][k]*mi[k]*W[k][j] for k in range(6)) for j in range(4)] for i in range(4)]
    c=[-sum(row[k]*vf[k] for k in range(6)) for row in N]
    return J,mi,W,vf,A,c


def system(d,epsilon,h=Q(1)):
    J,mi,W,vf,A,c=fixture(h)
    if epsilon:
        K=[[A[i][j]+(epsilon*d[j] if i==j else 0) for j in range(4)] for i in range(4)];rhs=c
    else:
        K=A[:3]+[[s*x for s,x in zip((1,-1,-1,1),d)]];rhs=c[:3]+[Q(0)]
    p=solve(K,rhs);w=[sum(a*b for a,b in zip(row,p)) for row in W]
    v=[vf[k]+mi[k]*w[k] for k in range(6)]
    return K,rhs,p,v,w,sum(v[k]**2/(2*mi[k]) for k in range(6))


def corner_weights(d,box,determinants):
    if not determinants or any(q==0 for q in determinants) or any(q*determinants[0]<=0 for q in determinants):
        raise ValueError('corner determinants need one nonzero sign')
    theta=[]
    for bits in product((0,1),repeat=len(d)):
        weight=Q(1)
        for x,(lo,hi),b in zip(d,box,bits): weight*=((x-lo) if b else (hi-x))/(hi-lo)
        theta.append(weight)
    q=sum(t*v for t,v in zip(theta,determinants))
    return theta,[t*v/q for t,v in zip(theta,determinants)]


@pytest.mark.parametrize('epsilon',(Q(0),Q(1,100)))
def test_multiaffine_cramer_license_and_velocity_wrench_hull(epsilon):
    box=[(Q(9,10),Q(11,10))]*4
    corners=[system(tuple(box[i][bit] for i,bit in enumerate(bits)),epsilon) for bits in product((0,1),repeat=4)]
    # Determinants and Cramer numerators are multiaffine because parameter i
    # enters only column i affinely. This checks the actual physical matrices.
    for d in ((Q(93,100),Q(101,100),Q(106,100),Q(98,100)),(Q(1),)*4):
        K,rhs,p,v,w,E=system(d,epsilon)
        theta,alpha=corner_weights(d,box,[det(row[0]) for row in corners])
        assert det(K)==sum(t*det(row[0]) for t,row in zip(theta,corners))
        assert sum(alpha)==1 and min(alpha)>=0
        for j in range(4):
            assert p[j]*det(K)==sum(t*row[2][j]*det(row[0]) for t,row in zip(theta,corners))
            assert p[j]>0
        for result,index in ((v,3),(w,4)):
            for j,value in enumerate(result):
                assert value==sum(a*row[index][j] for a,row in zip(alpha,corners))
                assert min(row[index][j] for row in corners)<=value<=max(row[index][j] for row in corners)
        assert E<=max(row[5] for row in corners)
        _,mi,_,vf,_,_=fixture()
        assert w==[(v[k]-vf[k])/mi[k] for k in range(6)]


def test_corner_gate_rejects_zero_or_changing_determinant_sign():
    for bad in ((Q(0),Q(1)),(Q(-1),Q(1))):
        with pytest.raises(ValueError): corner_weights((Q(1,2),),[(Q(0),Q(1))],bad)
    # Same-sign denominators alone do not license a nonmultiaffine numerator.
    t=Q(1,2);theta,alpha=corner_weights((t,),[(Q(0),Q(1))],(Q(1),Q(1)))
    nonmultiaffine=lambda x:1+4*x*(1-x)
    assert nonmultiaffine(t)>sum(a*nonmultiaffine(x) for a,x in zip(alpha,(Q(0),Q(1))))


def test_all_sixteen_corner_energies_miss_full_box_interior_minimum():
    box=[(Q(1,2),Q(2))]+[(Q(99,100),Q(101,100))]*3
    center=system((Q(1),)*4,Q(0),Q(0))[5]
    energies=[system(tuple(box[i][b] for i,b in enumerate(bits)),Q(0),Q(0))[5] for bits in product((0,1),repeat=4)]
    assert center==Q(505,6)
    assert min(energies)-center==Q(974169,1029218450)


def test_engine_residual_distinguishes_true_witness_from_perturbed_impulse():
    J,mi,_,_,_,_=fixture(Q(0));J=np.asarray(J,float);mi=np.asarray(mi,float)
    v=np.array((10.5,0,0,0,0,1));slip=(J@v).reshape(4,3)[:,1:]
    lam=np.column_stack((np.ones(4),-.5*slip/np.linalg.norm(slip,axis=1)[:,None])).ravel()
    vf=v-mi*(J.T@lam);G=(J*mi)@J.T;b=J@vf
    assert natural_residual(lam,G,b,np.full(4,.5))<1e-12
    bad=lam.copy();bad[0]+=.1
    assert natural_residual(bad,G,b,np.full(4,.5))>1e-4
