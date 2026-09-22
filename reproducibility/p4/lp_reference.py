#!/usr/bin/env python3
"""Independent static wrench-feasibility LP and subset-enumeration reference.

Two independent computations of the same discrete radial load limit:

  * solve_lp: a scipy/HiGHS LP over the nonnegative ray magnitudes of the
    16-ray polygon at every contact;
  * enumerate: exhaustive enumeration of all nonempty contact subsets, each
    solved as a separate LP over that subset's rays (only feasible for small
    contact counts).

The full-contact feasible set contains every subset cone (by zero-padding), so
the two radial limits must agree up to floating-point round-off.  This is a
numerical parity check on the same discretized friction law, not a statement
about the continuous circular cone.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import linprog

from friction_polygon import K


def solve_lp(W, f0, F, phi, caps=None, active=None):
    """Radial load limit for direction phi (radians)."""
    n_contacts = W.shape[1] // K
    active = list(range(n_contacts)) if active is None else list(active)
    ix = np.array([K * i + j for i in active for j in range(K)], int)
    d = np.array([math.cos(phi), math.sin(phi)])
    A_eq = np.column_stack((W[:, ix], -F @ d))
    c = np.r_[np.zeros(len(ix)), -1.0]
    A_ub = b_ub = None
    if caps is not None:
        A_ub = np.zeros((len(active), len(ix) + 1))
        b_ub = np.array([caps[i] for i in active], float)
        for q in range(len(active)):
            A_ub[q, q * K:(q + 1) * K] = 1.0
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=f0,
                bounds=[(0, None)] * len(ix) + [(0, None)], method="highs")
    return float(r.x[-1]) if r.success else 0.0


def enumerate_radius(W, f0, F, phi, caps=None, max_subsets=None):
    """Maximum radial limit over all nonempty contact subsets."""
    n_contacts = W.shape[1] // K
    best = 0.0
    done = 0
    for mask in range(1, 1 << n_contacts):
        active = [i for i in range(n_contacts) if mask >> i & 1]
        best = max(best, solve_lp(W, f0, F, phi, caps, active))
        done += 1
        if max_subsets is not None and done >= max_subsets:
            break
    return best, done


def radial_error(analytic, measured):
    return abs(measured - analytic) / abs(analytic)
