#!/usr/bin/env python3
"""Regular K-ray Coulomb friction cone used throughout the release.

A contact wrench is expressed as a nonnegative combination of a normal ray
(1, 0, 0) and K tangential rays.  The tangential rays lie on a regular K-gon of
radius POLY_SCALE*mu in the tangential plane.  POLY_SCALE = 2/(1+cos(pi/K)) is
the minimax radial scale: the polygon then STADDLES the circular cone of radius
mu (circumradius POLY_SCALE*mu > mu, inradius POLY_SCALE*cos(pi/K)*mu < mu).  It
is therefore neither inscribed nor circumscribed; the maximal radial deviation
from the circle is balanced on both sides and equals
(1-cos(pi/K))/(1+cos(pi/K)).
"""
from __future__ import annotations
import math

K = 16
POLY_SCALE = 2.0 / (1.0 + math.cos(math.pi / K))
POLY_RADIAL_ERROR_REL = (1.0 - math.cos(math.pi / K)) / (1.0 + math.cos(math.pi / K))
POLY_INRADIUS_REL = POLY_SCALE * math.cos(math.pi / K)   # < 1
POLY_CIRCUMRADIUS_REL = POLY_SCALE                        # > 1


def contact_rays(mu, scale=POLY_SCALE, k=K):
    """K (normal, tangent_x, tangent_y) rows for one contact."""
    return [(1.0, scale * mu * math.cos(2.0 * math.pi * j / k),
             scale * mu * math.sin(2.0 * math.pi * j / k)) for j in range(k)]
