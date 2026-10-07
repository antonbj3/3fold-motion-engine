"""Exact static feasibility witness for eight coplanar point contacts.

Opt-in sufficient certificate for a declared mathematical model, derived from
P4 WITNESS_LEMMA.md. No dynamics, joint limits, reached branch or unsafe verdict.
All public geometry and loads are rational; rotations are enclosed internally.
The implementation uses only the Python standard library.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
from math import factorial, isqrt

__all__ = ["certify_static_cell"]

_BITS = 96
_DEN = 1 << _BITS


def _q(x):
    """Internal conversion; public inputs were validated before this call."""
    return F(x)


def _floor_dyadic(x):
    return F((x.numerator * _DEN) // x.denominator, _DEN)


def _ceil_dyadic(x):
    return -_floor_dyadic(-x)


def _sqrt_bounds(x):
    if x < 0:
        raise ValueError('negative square root')
    n = isqrt((x.numerator * _DEN * _DEN) // x.denominator)
    lo = F(n, _DEN)
    hi = lo if lo * lo == x else F(n + 1, _DEN)
    assert lo * lo <= x <= hi * hi
    return lo, hi


@dataclass(frozen=True)
class _Iv:
    lo: F
    hi: F

    def __post_init__(self):
        if self.lo > self.hi:
            raise ValueError('reversed interval')

    def __add__(self, other):
        other = _iv(other)
        return _Iv(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self):
        return _Iv(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -_iv(other)

    def __rsub__(self, other):
        return _iv(other) + -self

    def __mul__(self, other):
        other = _iv(other)
        p = [a*b for a in (self.lo,self.hi) for b in (other.lo,other.hi)]
        return _Iv(min(p), max(p))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = _iv(other)
        if other.lo <= 0 <= other.hi:
            raise ZeroDivisionError('interval contains zero')
        return self * _Iv(1/other.hi, 1/other.lo)

    def rounded(self):
        return _Iv(_floor_dyadic(self.lo), _ceil_dyadic(self.hi))


def _iv(x):
    return x if isinstance(x, _Iv) else _Iv(_q(x),_q(x))


def _atan_recip(n, terms=48):
    # Alternating decreasing series: two consecutive partial sums enclose atan.
    a = sum((F((-1)**k, (2*k+1)*n**(2*k+1)) for k in range(terms)), F(0))
    b = a + F((-1)**terms, (2*terms+1)*n**(2*terms+1))
    return _Iv(min(a,b),max(a,b)).rounded()


@lru_cache(None)
def _pi_bounds():
    # Machin identity pi=16 atan(1/5)-4 atan(1/239).
    return (16*_atan_recip(5)-4*_atan_recip(239)).rounded()


@lru_cache(None)
def _sincos_deg(deg):
    x = (_pi_bounds()*F(deg,180)).rounded()
    # Taylor remainder at zero |R_n| <= max|x|^(n+1)/(n+1)!.
    # Exact interval operations include argument uncertainty. Round EACH step
    # outward onto a dyadic grid to bound integer growth, not a fixed epsilon.
    xx = (x*x).rounded()
    sine, cosine, st, ct = _iv(0), _iv(0), x, _iv(1)
    n = 32
    for k in range(n):
        sine = (sine+st).rounded()
        cosine = (cosine+ct).rounded()
        st = (-st*xx/F((2*k+2)*(2*k+3))).rounded()
        ct = (-ct*xx/F((2*k+1)*(2*k+2))).rounded()
    radius = max(abs(x.lo), abs(x.hi))
    se = radius**(2*n)/factorial(2*n)
    ce = radius**(2*n-1)/factorial(2*n-1)
    return (sine+_Iv(-se,se)).rounded(), (cosine+_Iv(-ce,ce)).rounded()


def _sub(a,b):
    return (a[0]-b[0],a[1]-b[1])


def _cross(a,b):
    return a[0]*b[1]-a[1]*b[0]


def _norm2(a):
    return a[0]*a[0]+a[1]*a[1]


def _hull(points):
    """Exact monotone-chain _hull, CCW; no geometry tolerance."""
    points = sorted(set(points))
    def half(seq):
        h = []
        for p in seq:
            while len(h)>1 and _cross(_sub(h[-1],h[-2]),_sub(p,h[-1])) <= 0:
                h.pop()
            h.append(p)
        return h
    return half(points)[:-1]+half(reversed(points))[:-1]


def _rotate_geometry(block):
    base = [tuple(_q(x) for x in p) for p in block['base_contacts_xy']]
    moving = set(block['moving_ids'])
    centre = tuple(sum(base[i][k] for i in moving)/len(moving) for k in range(2))
    sine,cosine = _sincos_deg(int(block['yaw_deg']))
    approx, radius = [], F(0)
    for i,p in enumerate(base):
        if i not in moving:
            approx.append(p)
            continue
        dx,dy = _sub(p,centre)
        box = [_iv(centre[0])+cosine*dx-sine*dy,
               _iv(centre[1])+sine*dx+cosine*dy]
        mid = tuple(_floor_dyadic((a.lo+a.hi)/2) for a in box)
        err2 = sum(max(abs(a.lo-m),abs(a.hi-m))**2 for a,m in zip(box,mid))
        radius = max(radius,_sqrt_bounds(err2)[1])
        approx.append(mid)
    return approx, radius


@lru_cache(None)
def _tau_lower():
    # cos(pi/16) = sqrt(2+sqrt(2+sqrt(2)))/2, monotone all the way.
    r = _sqrt_bounds(F(2))[0]
    r = _sqrt_bounds(2+r)[0]
    c = _sqrt_bounds(2+r)[0]/2
    return 2*c/(1+c)


def _placed(base,moving,centre):
    return [tuple(p[k]+(centre[k] if i in moving else 0) for k in range(2))
            for i,p in enumerate(base)]


def _certify(base,err,moving,com,centre,side,pair,weight,level,height_max,rp_max):
    """SAFE means full 6D static polygon feasibility for ALL cell/d/envelope.

    UNKNOWN never means infeasible. Reject invalid contracts before arithmetic.
    Inputs here are exact rationals; real-rotation enclosure is made upstream.
    """
    if (len(base)!=8 or len(set(pair))!=2 or any(i not in range(8) for i in pair)
            or not 0 < weight < F(1,2) or side < 0 or level < 0
            or height_max <= 0 or rp_max < 0 or err < 0):
        raise ValueError('invalid witness contract')
    p = _placed(base,moving,centre)
    i,k = pair
    rho = _sqrt_bounds(side*side/2)[1]
    ell = _sqrt_bounds(_norm2(_sub(p[i],p[k])))[0]-rho-2*err
    if ell <= 0:
        return {'status':'UNKNOWN','reason':'pair-distance'}
    rest = 1-2*weight  # exact, unlike the audited historical interval attempt
    vertices = [tuple(weight*(p[i][d]+p[k][d])+rest*p[j][d] for d in range(2))
                for j in range(8) if j not in pair]
    hh = _hull(vertices)
    if len(hh)<3:
        return {'status':'UNKNOWN','reason':'degenerate-_hull'}
    # Convex coefficients sum to one: each exact E vertex is within err of
    # surrogate E vertex. Each placement vertex moves <= rho. Thus support
    # margins decrease by at most rho+err in EVERY direction, no angle cover.
    required = height_max*level/F('9.81')+rho+err
    edge_slacks = []
    for a,b in zip(hh,hh[1:]+hh[:1]):
        e = _sub(b,a)
        area = _cross(e,_sub(com,a))
        slack = area*area-required*required*_norm2(e)
        if area <= 0 or slack < 0:
            return {'status':'UNKNOWN','reason':'ball-containment'}
        edge_slacks.append(slack)
    capslack = _tau_lower()*weight*F('9.81')-level*(weight+rp_max/ell)
    if capslack < 0:
        return {'status':'UNKNOWN','reason':'friction-capacity'}
    return {'status':'SAFE','pair':list(pair),'weight':str(weight),
            'minimum_squared_edge_slack':str(min(edge_slacks)),
            'capacity_slack':str(capslack),'pair_distance_lower':str(ell),
            'rho_upper':str(rho),'rotation_error_upper':str(err),
            'hull_vertices':[[str(x) for x in a] for a in hh]}


def _rational(value, name):
    # A bounded exact input contract: implicit binary64/NumPy conversion is
    # deliberately forbidden. Callers must explicitly choose rational data.
    if type(value) not in (int, str, F):
        raise TypeError(f'{name}: expected int, rational string or Fraction')
    if isinstance(value, str) and len(value) > 160:
        raise ValueError(f'{name}: rational string exceeds 160 characters')
    if isinstance(value, str):
        # Exponent notation could request an enormous integer before the
        # bit-length guard runs. Accept only signed integers or ratios.
        parts = value.split('/')
        if len(parts) not in (1, 2) or any(
                not p or not p.lstrip('+-').isascii()
                or not p.lstrip('+-').isdigit()
                or p[:1] in '+-' and len(p) == 1 for p in parts):
            raise ValueError(f'{name}: use an integer or numerator/denominator')
    try:
        value = F(value)
        # Fraction can preserve fixed-width integer components supplied by a
        # caller (for example NumPy int64); all accepting arithmetic must use
        # unbounded Python integers, including cross products and squaring.
        value = F(int(value.numerator), int(value.denominator))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f'{name}: invalid rational') from exc
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 256:
        raise ValueError(f'{name}: numerator and denominator must fit 256 bits')
    return value


def _sequence(value, length, name):
    if type(value) not in (tuple, list) or len(value) != length:
        raise ValueError(f'{name}: expected a list or tuple of length {length}')
    return value


def _point(value, name):
    return tuple(_rational(v, name) for v in _sequence(value, 2, name))


def _ids(value, length, name):
    value = _sequence(value, length, name)
    if any(type(i) is not int or not 0 <= i < 8 for i in value) or len(set(value)) != length:
        raise ValueError(f'{name}: expected {length} distinct Python int indices in [0, 7]')
    return tuple(value)


def certify_static_cell(*, base_contacts_xy, moving_ids, yaw_deg, com_xy,
                        cell_center_xy, cell_side, pair, weight, level,
                        height_max, pelvis_offset_max):
    """Prove static feasibility at ``level`` for a whole placement square.

    The eight input points lie on z=0. Exactly four rotate about their mean by
    integral yaw_deg in [-180,180], then translate by u anywhere in the closed
    square centred at cell_center_xy with side cell_side. The COM projection
    remains fixed. Gravity is EXACTLY 981/100; mu=1; the friction model is the
    ideal regular16-ray minimax polygon. The pelvis has 0<h<=height_max and
    |pelvis_xy-com_xy|<=pelvis_offset_max. Every horizontal unit load direction
    and every nonnegative load up to level is covered. Mass may be any m>0.

    Geometry/load scalars accept only Python int, Fraction, or integer/ratio
    strings, with numerator/denominator at most256 bits; floats and bools are
    rejected. Collections must be lists/tuples. All candidate pairs are allowed;
    this function checks ONE proposed pair and 0<weight<1/2, without searching.

    STATIC_FEASIBLE_AT_LEVEL means the sufficient inequalities hold, including
    equality. UNKNOWN means no witness from this candidate, never infeasible.
    Malformed/out-of-contract inputs raise TypeError/ValueError before proof.
    No scene exporter, measured geometry uncertainty, branch/path, dynamic,
    joint-torque or physical-safety admission is implied. See the accompanying
    documentation for the proof and source-model distinctions.
    """
    base = [_point(p, 'base_contacts_xy') for p in
            _sequence(base_contacts_xy, 8, 'base_contacts_xy')]
    moving = _ids(moving_ids, 4, 'moving_ids')
    pair = _ids(pair, 2, 'pair')
    if type(yaw_deg) is not int or not -180 <= yaw_deg <= 180:
        raise ValueError('yaw_deg: expected Python int in [-180, 180]')
    com = _point(com_xy, 'com_xy')
    centre = _point(cell_center_xy, 'cell_center_xy')
    side = _rational(cell_side, 'cell_side')
    weight = _rational(weight, 'weight')
    level = _rational(level, 'level')
    height = _rational(height_max, 'height_max')
    offset = _rational(pelvis_offset_max, 'pelvis_offset_max')
    if side < 0 or level < 0 or height <= 0 or offset < 0 or not 0 < weight < F(1, 2):
        raise ValueError('require side, level, offset >=0, height>0, and 0<weight<1/2')
    approx, err = _rotate_geometry({'base_contacts_xy': base,
                                    'moving_ids': moving, 'yaw_deg': yaw_deg})
    result = _certify(approx, err, set(moving), com, centre, side, pair,
                      weight, level, height, offset)
    if result['status'] == 'SAFE':
        result['status'] = 'STATIC_FEASIBLE_AT_LEVEL'
    result['model'] = 'coplanar8_mu1_minimax16_g981over100_v1'
    result['level'] = str(level)
    return result
