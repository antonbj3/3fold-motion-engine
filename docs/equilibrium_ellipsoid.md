# Exact reduced linear-equilibrium ellipsoid

`motion_engine.ncp.equilibrium_ellipsoid.equilibrium_readout` accepts exact
rational A, b, an approximate state x, a symmetric lower matrix M and a linear
readout ell. It verifies 0<M<=A by exact Schur/PSD elimination. No eigentolerance
or floating inverse is trusted. Singular rigid modes, nonsymmetry and invalid
lower curvature are rejected. Dimension is bounded at 32.

For Ax*=b, r=Ax-b and e=x*-x, e'Me<=-r'e. Completing the square gives centre
`-M^-1 r/2`, metric radius squared `r'M^-1 r/4`, and readout support squared
`(r'M^-1 r)(ell'M^-1 ell)/4`. An integer square root supplies an outward dyadic
radius. The result bounds ell*x*, including residual and readout uncertainty
from this exact algebraic model. The output unit is explicit; upstream scaling
must make force/translation/rotation coordinates and matrices consistent.

This transfers the residual-ellipsoid mechanism only. It does not import a
nonlinear inverse-joint law, a floating nullspace, tolerance probabilities,
material calibration or a clinical safety screen. A nonlinear extension needs
a verified monotonicity/curvature license on every actual contact regime. A
detached-contact regime may lose positive curvature, requiring UNKNOWN or an
explicit restraint. This module does not replace the existing NCP solver.
