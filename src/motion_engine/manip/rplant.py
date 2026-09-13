"""Plant randomisation over a replicated world: per-replica draws of the physical parameters that the
manipulation cells randomise. Pure numpy.
"""
from __future__ import annotations

import numpy as np


def apply_r_plant(b, *, W: int, nb_rob: int, nd: int, nb_t: int,
                  plant_seed: int):
    """Apply plant randomisation to the replicated builder `b` (mutates b in place).

    b: replikerad newton.ModelBuilder (W kopior).
    nb_rob: robot bodies per world, nd: arm DOFs per world,
    nb_t: total bodies per world (nb_rob + free objects).
    Returns (fm, fr, fobj) for logging.
    """
    rr = np.random.default_rng(plant_seed)
    fm = rr.uniform(0.85, 1.15, nb_rob); fr = rr.uniform(1.0, 6.0, nd)
    fcube = rr.uniform(0.7, 1.3, W)
    for w in range(W):
        for k in range(nb_rob):
            b.body_mass[w * nb_t + k] *= fm[k]
        b.body_mass[w * nb_t + nb_rob] *= fcube[w]
        for k in range(nd):
            b.joint_friction[w * nd + k] = float(fr[k])
    return fm, fr, fcube
