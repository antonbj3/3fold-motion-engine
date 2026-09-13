"""Multi-body split-impulse contact engine: the validated single-body split impulse carried to N bodies with
body-body contact. Voxel particles (parallel collision detection) -> sequential split impulses (velocity jn/jt
accumulated and clamped at mu*jn, with a separate position pass) -> rigid-body Newton-Euler. Importable;
kinematic bodies are supported for gripping."""
import numpy as np
G=9.81; R=0.05; BETA=0.2; SLOP=2e-4; VIT=10; PIT=6
def voxbox(w,h,d,res=R*2):
    a=lambda L: np.arange(-L/2+res/2,L/2,res); return np.array([[x,y,z] for z in a(d) for y in a(h) for x in a(w)])
def skew(w): return np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
def rodr(w,dt):
    th=np.linalg.norm(w)*dt
    if th<1e-9: return np.eye(3)
    k=w/(np.linalg.norm(w)+1e-12); K=skew(k); return np.eye(3)+np.sin(th)*K+(1-np.cos(th))*K@K

class Body:
    def __init__(s,w,h,d,c,density=700.,kin=False):
        s.rest=voxbox(w,h,d); n=len(s.rest); s.mp=density*w*h*d/n; s.M=s.mp*n
        s.Ib=sum(s.mp*((r@r)*np.eye(3)-np.outer(r,r)) for r in s.rest)
        s.xc=np.array(c,float); s.Rm=np.eye(3); s.vc=np.zeros(3); s.om=np.zeros(3); s.kin=kin
    def pw(s): return s.rest@s.Rm.T+s.xc
    def Iwi(s): return np.linalg.inv(s.Rm@s.Ib@s.Rm.T)

class World:
    def __init__(s,ramp=0.,mu=0.5):
        th=np.radians(ramp); s.n=np.array([-np.sin(th),0,np.cos(th)]); s.mu=mu; s.B=[]
    def add(s,*a,**k): b=Body(*a,**k); s.B.append(b); return b
    def set_kin(s,i,c): s.B[i].xc=np.array(c,float)
    def _apply(s,b,J,r):                                          # impulse J at world point r applied to a body; torque = cross(LEVER = r - com, J), NOT cross(r, J)
        if b.kin: return
        b.vc=b.vc+J/b.M; b.om=b.om+b.Iwi()@np.cross(r-b.xc,J)
    def _vel_at(s,b,pwi): return np.zeros(3) if b.kin else b.vc+np.cross(b.om,pwi-b.xc)
    def step(s,dt,sub,control=None,t0=0.):
        for ss in range(sub):
            sdt=dt/sub
            if control: control(s,t0+ss*sdt)
            for b in s.B:
                if not b.kin: b.vc=b.vc+np.array([0,0,-G])*sdt
            # samla kontakter: (bodyA,bodyB el None=mark, rA, rB, normal, pen)
            C=[]
            PW=[b.pw() for b in s.B]                              # world-frame particles, computed once
            for bi,b in enumerate(s.B):                            # MARK
                if b.kin: continue
                sg=PW[bi]@s.n - R
                for i in np.where(sg<0)[0]: C.append([b,None,PW[bi][i],None,s.n,-sg[i]])
            # BODY-BODY via GRID-HASH broad-phase (cell-ID, kache GPU Gems ch.32): O(n) ist.f. O(K²n²)
            allp=np.vstack(PW); owner=np.concatenate([np.full(len(PW[bi]),bi) for bi in range(len(s.B))])
            cell=np.floor(allp/(2*R)).astype(np.int64); grid={}
            for idx in range(len(allp)): grid.setdefault((cell[idx,0],cell[idx,1],cell[idx,2]),[]).append(idx)
            offs=[(a,b,c) for a in(-1,0,1) for b in(-1,0,1) for c in(-1,0,1)]
            for idx in range(len(allp)):
                bi=owner[idx]; ci=cell[idx]
                for do in offs:
                    for jdx in grid.get((ci[0]+do[0],ci[1]+do[1],ci[2]+do[2]),[]):
                        if jdx<=idx: continue
                        bj=owner[jdx]
                        if bi==bj or (s.B[bi].kin and s.B[bj].kin): continue
                        d=allp[idx]-allp[jdx]; dist=np.linalg.norm(d)+1e-12
                        if dist<2*R: C.append([s.B[bi],s.B[bj],allp[idx],allp[jdx],d/dist,2*R-dist])
            if not C:
                for b in s.B:
                    if not b.kin: b.xc=b.xc+b.vc*sdt; b.Rm=rodr(b.om,sdt)@b.Rm
                continue
            jn=[0.]*len(C); jt=[0.]*len(C)
            # the friction tangent comes from the SLIP direction, not from gravity: a gravity tangent DEGENERATES for horizontal contact
            # (normal parallel to gravity -> projection about 0 -> the friction direction is meaningless, a lateral-kick bug). Falls back to a direction perpendicular to n when slip < 5 mm/s.
            def _tang(c):
                A,B,rA,rB,nrm,pen=c
                vr=s._vel_at(A,rA)-(s._vel_at(B,rB) if B else 0); vt=vr-(vr@nrm)*nrm; m=np.linalg.norm(vt)
                if m>5e-3: return vt/m
                a=np.array([1.,0,0]) if abs(nrm[0])<0.9 else np.array([0,1.,0]); t=a-(a@nrm)*nrm; return t/(np.linalg.norm(t)+1e-12)
            tang=[_tang(c) for c in C]
            for _ in range(VIT):                                  # VELOCITY-pass
                for ci,c in enumerate(C):
                    A,B,rA,rB,nrm,pen=c; vrel=s._vel_at(A,rA)-(s._vel_at(B,rB) if B else 0)
                    meff=(0 if A.kin else 1/A.M+np.cross(rA-A.xc,nrm)@(A.Iwi()@np.cross(rA-A.xc,nrm)))+(0 if (B is None or B.kin) else 1/B.M+np.cross(rB-B.xc,nrm)@(B.Iwi()@np.cross(rB-B.xc,nrm)))
                    if meff<1e-12: continue
                    vn=vrel@nrm; dj=-vn/meff; nw=max(0,jn[ci]+dj); dj=nw-jn[ci]; jn[ci]=nw
                    s._apply(A,dj*nrm,rA);
                    if B: s._apply(B,-dj*nrm,rB)
                    vrel=s._vel_at(A,rA)-(s._vel_at(B,rB) if B else 0); t=tang[ci]; vt=vrel@t
                    djt=-vt/meff; nwt=max(-s.mu*jn[ci],min(s.mu*jn[ci],jt[ci]+djt)); djt=nwt-jt[ci]; jt[ci]=nwt
                    s._apply(A,djt*t,rA)
                    if B: s._apply(B,-djt*t,rB)
            # POSITION pass: a proper iterative pseudo-velocity (Catto split impulse), which converges for COUPLED contacts (K >= 3);
            # pushing xc directly was the bug: it over-corrects when coupled, injects energy and explodes.
            pv={id(b):np.zeros(3) for b in s.B}; po={id(b):np.zeros(3) for b in s.B}; jp=[0.]*len(C)
            def pvel(b,r): return np.zeros(3) if b.kin else pv[id(b)]+np.cross(po[id(b)],r-b.xc)
            for _ in range(PIT):
                for ci,c in enumerate(C):
                    A,B,rA,rB,nrm,pen=c
                    meff=(0 if A.kin else 1/A.M+np.cross(rA-A.xc,nrm)@(A.Iwi()@np.cross(rA-A.xc,nrm)))+(0 if (B is None or B.kin) else 1/B.M+np.cross(rB-B.xc,nrm)@(B.Iwi()@np.cross(rB-B.xc,nrm)))
                    if meff<1e-12: continue
                    rel=pvel(A,rA)-(pvel(B,rB) if B is not None else 0); bias=BETA*max(pen-SLOP,0)/sdt
                    dj=(bias-rel@nrm)/meff; nw=max(0,jp[ci]+dj); dj=nw-jp[ci]; jp[ci]=nw
                    if not A.kin: pv[id(A)]=pv[id(A)]+dj*nrm/A.M; po[id(A)]=po[id(A)]+A.Iwi()@np.cross(rA-A.xc,dj*nrm)
                    if B is not None and not B.kin: pv[id(B)]=pv[id(B)]-dj*nrm/B.M; po[id(B)]=po[id(B)]-B.Iwi()@np.cross(rB-B.xc,dj*nrm)
            for b in s.B:
                if not b.kin: b.xc=b.xc+(b.vc+pv[id(b)])*sdt; b.Rm=rodr(b.om+po[id(b)],sdt)@b.Rm
    def com(s,i): return s.B[i].xc
