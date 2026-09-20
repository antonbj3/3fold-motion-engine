"""Material-point sand: the constitutive core, the exact-cone contact coupling, and the
closed-form bearing-capacity references the results are measured against.

`sand_core`     Drucker-Prager with Lode-matched cones; the rate, spline and tension
                branches are parameter-gated and off by default.
`sand_contact`  grid-node contact against the same exact cone as the rigid solver.
`geotech`       Terzaghi / Meyerhof / Vesic / Hansen bearing capacity, and the local
                failure reduction phi* = atan(2/3 tan phi).
"""
