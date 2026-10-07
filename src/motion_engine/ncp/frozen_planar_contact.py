"""Opt-in exact inventory for a declared frozen planar force model.

This is a diagnostic oracle, not a replacement for generic Krawczyk, a full-scene
residual exporter, or permission to accept a physical time step. SymPy is optional
and imported only when an inventory is requested. See docs/frozen_planar_contact.md.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from types import MappingProxyType

MODEL_SINGLE = 'frozen-planar-G11-12-mu2-t3/4-v1'
MODEL_COUPLED = 'frozen-planar-pair-c1/10-eta1/8-q2=7/10-v1'

@dataclass(frozen=True)
class ExactEvent:
    """A name in the immutable internal event catalogue; never a float snap."""
    name: str

@dataclass(frozen=True)
class ParameterInterval:
    lower: object
    upper: object

# Polynomials are highest degree first. Rational bounds isolate exactly one root.
_EVENTS = MappingProxyType({
 'one_fold': ((16,-8,37,-28), ('7107/10000','7108/10000')),
 'pair_inactive_fold': ((16,-8,37,-28), ('7107/10000','7108/10000')),
 'pair_PT_TT_seam': ((100000,-236500,187140,-49469), ('7224/10000','7225/10000')),
 'pair_PP_TP_seam_low': ((6400000,-13312000,9192840,-2106509), ('7307/10000','7309/10000')),
 'pair_PP_TP_seam_high': ((6400000,-13312000,9192840,-2106509), ('7431/10000','7433/10000')),
 'pair_inactive_seam': ((4,-3), ('3/4','3/4')),
})

class _Budget(Exception): pass

def _rational(v):
    if isinstance(v,(bool,float)) or not isinstance(v,(int,Q)):
        raise ValueError('exact int/Fraction required; floats are not event evidence')
    v=Q(v)
    v=Q(int(v.numerator),int(v.denominator))
    if max(abs(v.numerator).bit_length(),v.denominator.bit_length())>128:
        raise ValueError('exact input exceeds128-bit admission budget')
    return v

def _unknown(model,reason):
    return dict(status='UNKNOWN',complete=False,model_id=model,reason=str(reason),
                branches=[],distinct_physical_states=None,step_acceptance='NOT_ASSESSED')

def _mul(a,b):
    vals=[x*y for x in a for y in b];return min(vals),max(vals)

def _add(a,b):return a[0]+b[0],a[1]+b[1]

class _Field:
    def __init__(self,value,allowed,max_work):
        if isinstance(max_work,bool) or not isinstance(max_work,int) or not 1<=max_work<=100000:
            raise ValueError('work budget must be an integer in1..100000')
        import sympy as s
        self.s=s;self.x=s.Symbol('g');self.t=s.Symbol('parameter');self.work=0;self.max_work=max_work
        if isinstance(value,ExactEvent):
            if value.name not in allowed:raise ValueError('event is not admitted by this model')
            coeff,bounds=_EVENTS[value.name]
            self.parameter_poly=s.Poly.from_list(list(coeff),self.t,domain=s.QQ)
            self.lo,self.hi=map(Q,bounds)
            if self.lo==self.hi:
                assert self.parameter_poly.eval(s.Rational(self.lo))==0
                self.K=s.QQ;self.value=self.K.convert(s.Rational(self.lo))
            else:
                assert self.parameter_poly.count_roots(s.Rational(self.lo),s.Rational(self.hi))==1
                index=int(self.parameter_poly.count_roots(-s.oo,s.Rational(self.lo)))
                alpha=s.CRootOf(self.parameter_poly,index)
                self.K=s.QQ.algebraic_field(alpha);self.value=self.K.unit
                self._refine_parameter(64)
            self.parameter=dict(kind='EXACT_NAMED_ALGEBRAIC_EVENT',name=value.name,polynomial=list(map(str,coeff)),isolating_interval=list(bounds))
        else:
            q=_rational(value);self.K=s.QQ;self.value=self.K.convert(s.Rational(q));self.lo=self.hi=q
            self.parameter=dict(kind='EXACT_RATIONAL',value=str(q))
        self.sign_cache={}
    def tick(self):
        self.work+=1
        if self.work>self.max_work:raise _Budget('exact work budget exhausted')
    def _refine_parameter(self,bits):
        self.tick()
        a,b=self.parameter_poly.refine_root(self.s.Rational(self.lo),self.s.Rational(self.hi),eps=self.s.Rational(1,2**bits))
        self.lo,self.hi=Q(a),Q(b)
    def q(self,v):return self.K.convert(self.s.Rational(v))
    def coeffs(self,v):
        values=v.to_list() if self.K.is_AlgebraicField else [v]
        return [Q(int(z.numerator),int(z.denominator)) for z in values]
    def enclosure(self,v):
        r=(Q(0),Q(0))
        for z in self.coeffs(v):r=_add(_mul(r,(self.lo,self.hi)),(z,z))
        return r
    def sign(self,v):
        if not v:return 0
        key=tuple(self.coeffs(v))
        if key in self.sign_cache:return self.sign_cache[key]
        for bits in (64,128,256,512,1024):
            lo,hi=self.enclosure(v)
            if lo>0 or hi<0:
                ans=1 if lo>0 else -1;self.sign_cache[key]=ans;return ans
            self._refine_parameter(bits*2)
        raise _Budget('algebraic sign separation budget exhausted')
    def poly(self,coeff):
        return self.s.Poly.from_dict({(i,):v for i,v in enumerate(coeff) if v},self.x,domain=self.K)
    def eval(self,p,v):return p.rep.eval(self.q(v))
    def pbox(self,p,box):
        r=(Q(0),Q(0))
        for v in p.rep.to_list():r=_add(_mul(r,box),self.enclosure(v))
        return r
    def sequence(self,p):
        self.tick();return p.sturm()
    def variations(self,seq,x):
        signs=[self.sign(self.eval(p,x)) for p in seq];signs=[v for v in signs if v]
        return sum(a!=b for a,b in zip(signs,signs[1:]))
    def count(self,p,seq,a,b):
        self.tick()
        if a==b:return int(not self.eval(p,a))
        # Sturm V(a)-V(b) counts(a,b]; remove a root atb for open(a,b).
        return self.variations(seq,a)-self.variations(seq,b)-int(not self.eval(p,b))
    def roots(self,p):
        """Distinct roots in the physical open gap domain(0,1), with multiplicity."""
        roots=[]
        for factor,multiplicity in p.sqf_list()[1]:
            seq=self.sequence(factor);todo=[(Q(0),Q(1))]
            while todo:
                a,b=todo.pop();n=self.count(factor,seq,a,b)
                if not n:continue
                mid=(a+b)/2
                if not self.eval(factor,mid):
                    roots.append(dict(p=factor,seq=seq,lo=mid,hi=mid,multiplicity=int(multiplicity)))
                    todo.extend([(a,mid),(mid,b)]);continue
                if n==1 and b-a<=Q(1,2**32):
                    roots.append(dict(p=factor,seq=seq,lo=a,hi=b,multiplicity=int(multiplicity)))
                else:todo.extend([(a,mid),(mid,b)])
        return sorted(roots,key=lambda r:r['lo'])
    def refine(self,r):
        self.tick();a,b=r['lo'],r['hi']
        if a==b:return
        mid=(a+b)/2
        if not self.eval(r['p'],mid):r['lo']=r['hi']=mid
        elif self.count(r['p'],r['seq'],a,mid):r['hi']=mid
        else:r['lo']=mid
    def sign_at(self,p,r):
        if p.is_zero:return 0
        if r['lo']==r['hi']:return self.sign(self.eval(p,r['lo']))
        common=self.s.gcd(p,r['p'])
        if common.degree()>0:
            seq=self.sequence(common)
            if self.count(common,seq,r['lo'],r['hi']):return 0
        for _ in range(256):
            lo,hi=self.pbox(p,(r['lo'],r['hi']))
            if lo>0:return 1
            if hi<0:return -1
            self.refine(r)
        raise _Budget('physical guard separation budget exhausted')
    def encode_poly(self,p):
        return [[str(q) for q in self.coeffs(v)] for v in p.rep.to_list()]

# Small coefficient arithmetic, shared by rational and degree-three event fields.
def _plus(a,b):
    n=max(len(a),len(b));return [(a[i] if i<len(a) else 0)+(b[i] if i<len(b) else 0) for i in range(n)]
def _scale(a,c):return [c*x for x in a]
def _conv(a,b):
    r=[0]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):r[i+j]+=x*y
    return r

def _pair_polynomials(F,mode):
    data={'P':(F.q(-1),F.q(0)),'T':(F.q(Q(1,2)),F.q(Q(-3,8)))}
    a1,o1=data[mode[0]];a2,o2=data[mode[1]];c=F.q(Q(1,10));u1=F.value+o1;u2=F.q(Q(7,10))+o2
    det=a1*a2-c*c;n=[-det,2*det,-4*a2*u1+4*c*u2-det,4*a2];d=[F.q(0),F.q(0),4*c]
    n2=_conv(n,n);p=_scale(_conv(n2,n),4*a1)
    p=_plus(p,_scale(_conv(_conv([a1*u2-c*u1,c],n2),d),-4))
    dn=_plus(d,_scale(n,-1));p=_plus(p,_scale(_conv(_conv(dn,dn),d),-det))
    return F.poly(p),F.poly(n),F.poly(d)

def _same_root(F,a,b):
    ra,rb=a['root'],b['root'];lo=max(ra['lo'],rb['lo']);hi=min(ra['hi'],rb['hi'])
    if lo>hi:return False
    common=F.s.gcd(ra['p'],rb['p'])
    if common.degree()<=0:return False
    seq=F.sequence(common)
    if lo==hi:match=not F.eval(common,lo)
    else:match=bool(F.count(common,seq,lo,hi))
    if not match:return False
    return F.sign_at(a['n']*b['d']-b['n']*a['d'],ra)==0

def _finish(F,model,raw,mode_counts,proof,eta=None):
    branches=[]
    for row in raw:
        previous=next((v for v in branches if _same_root(F,v,row)),None)
        alias=dict(mode=row['mode'],algebraic_multiplicity=row['root']['multiplicity'])
        if previous is None:row['aliases']=[alias];branches.append(row)
        else:previous['aliases'].append(alias)
    rows=[]
    for i,v in enumerate(sorted(branches,key=lambda z:z['root']['lo'])):
        r=v['root'];yb=F.pbox(v['n'],(r['lo'],r['hi']));db=F.pbox(v['d'],(r['lo'],r['hi']))
        if db[0]<=0:raise _Budget('positive reconstruction denominator not separated')
        gap2=_mul(yb,(1/db[1],1/db[0]))
        rows.append(dict(id=str(i),gap1_interval=[str(r['lo']),str(r['hi'])],gap2_interval=[str(x) for x in gap2] if model==MODEL_COUPLED else None,
          gap1_polynomial=F.encode_poly(r['p']),coefficient_format='descending gap powers; each coefficient is descending powers of the named parameter root',
          reconstruction_numerator=F.encode_poly(v['n']) if model==MODEL_COUPLED else None,reconstruction_denominator=F.encode_poly(v['d']) if model==MODEL_COUPLED else None,
          mode_aliases=v['aliases'],distinct_state_multiplicity=1))
    return dict(status='COMPLETE',complete=True,model_id=model,parameter=F.parameter,eta=str(eta) if eta is not None else '1/8',
       branches=rows,distinct_physical_states=len(rows),mode_admissible_counts=mode_counts,proof_obligations=proof,
       exact_work=F.work,step_acceptance='NOT_ASSESSED',meaning='Complete inventory only for the bound reduced frozen equilibrium model; multiplicity is not a number of physical states.')

def inventory_single(q,*,eta=Q(1,8),max_work=20000):
    """Exact one-contact inventory; q is int/Fraction or ExactEvent('one_fold')."""
    try:
        if isinstance(q,ParameterInterval):return _unknown(MODEL_SINGLE,'PARAMETER_INTERVAL_NOT_SUPPORTED: no event snapping')
        eta=_rational(eta)
        if eta<=0:raise ValueError('strictly positive eta required')
        if isinstance(q,ExactEvent) and eta!=Q(1,8):raise ValueError('named fold binds eta=1/8')
        F=_Field(q,{'one_fold'},max_work);e=F.q(eta);one=F.q(1);zero=F.q(0);x=F.poly([zero,one]);raw=[];counts={}
        # f=2eta(1-g)^2/g^2; positive slip g=q-f, stick g=q-3/8+f/2.
        guard=F.poly([-8*e,16*e,one-8*e]) # g^2(1-4f); P>=0,T<=0.
        for mode,p in [('P',F.poly([2*e,-4*e,2*e-F.value,one])),('T',F.poly([-e,2*e,-e-F.value+F.q(Q(3,8)),one]))]:
            rows=[]
            for r in F.roots(p):
                sign=F.sign_at(guard,r)
                if (mode=='P' and sign>=0) or (mode=='T' and sign<=0):rows.append(dict(root=r,mode=mode,n=x,d=F.poly([one])))
            raw+=rows;counts[mode]=len(rows)
        result=_finish(F,MODEL_SINGLE,raw,counts,['All cubic roots in0<g<1 isolated by ordered-field Sturm; force denominators nonzero.','P requiresf<=1/4; T requiresf>=1/4; shared seam states deduplicated algebraically.','Negative slip impossible becausevt=3/4+5f>0.','Inactive stateg=q exists iff q>=1 and is counted once.'],eta)
        inactive = F.sign(F.value-one)>=0
        result['mode_admissible_counts']['I'] = int(inactive)
        if inactive:
            result['branches'].append(dict(id=str(len(result['branches'])),gap1_exact_parameter=True,mode_aliases=[dict(mode='I',algebraic_multiplicity=1)],distinct_state_multiplicity=1))
            result['distinct_physical_states']+=1
        return result
    except ImportError:return _unknown(MODEL_SINGLE,'OPTIONAL_SYMPY_UNAVAILABLE')
    except (ValueError,TypeError,_Budget) as exc:return _unknown(MODEL_SINGLE,exc)

def inventory_coupled(q1,*,q2=Q(7,10),max_work=50000):
    """Exact coupled inventory for q1 in[7/10,4/5], fixed q2=7/10.

    Named pair events include all five points left open by the earlier event-free
    certificate. The inactive-fold event is deliberately not called a physical fold.
    """
    try:
        if isinstance(q1,ParameterInterval):return _unknown(MODEL_COUPLED,'PARAMETER_INTERVAL_NOT_SUPPORTED: no event snapping')
        if _rational(q2)!=Q(7,10):raise ValueError('coupled family binds q2=7/10')
        F=_Field(q1,{k for k in _EVENTS if k.startswith('pair_')},max_work)
        if F.sign(F.value-F.q(Q(7,10)))<0 or F.sign(F.value-F.q(Q(4,5)))>0:raise ValueError('q1 outside admitted[7/10,4/5]')
        half=F.q(Q(1,2));xhalf=F.poly([-half,F.q(1)]);raw=[];counts={}
        for mode in ('PP','PT','TP','TT'):
            p,n,d=_pair_polynomials(F,mode);assert p.degree()==9
            rows=[]
            for r in F.roots(p):
                side=F.sign_at(xhalf,r)
                if mode[0]=='P' and side<0 or mode[0]=='T' and side>0:continue
                if F.sign_at(n,r)<=0 or F.sign_at(n-d,r)>=0:continue
                side=F.sign_at(n-d.mul_ground(half),r)
                if mode[1]=='P' and side<0 or mode[1]=='T' and side>0:continue
                rows.append(dict(root=r,mode=mode,n=n,d=d))
            counts[mode]=len(rows);raw+=rows
        return _finish(F,MODEL_COUPLED,raw,counts,['Four exhaustive active modes, exact degree9 eliminants and unique y=N/(4cx^2) reconstruction.','Sturm inventory over0<x<1; exact signs impose each original P/T gap domain and0<y<1.','All inactive modes excluded throughout domain: active P force<=1/4; active T force<=7/20; inactive gap<=4/5+(1/10)(7/20)=167/200<1.','Negative slip impossible; G leading minors1,1,49/50,24/25 establishSPD.','Mode seams deduplicated by polynomial gcd and exact equality of reconstructed y.'])
    except ImportError:return _unknown(MODEL_COUPLED,'OPTIONAL_SYMPY_UNAVAILABLE')
    except (ValueError,TypeError,_Budget) as exc:return _unknown(MODEL_COUPLED,exc)
