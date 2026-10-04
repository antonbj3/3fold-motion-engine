# Conditional uniqueness for the updated cubic force

This API certifies at most one motion on a declared gap domain. It requires
a gap floor that covers every admissible solution, frozen affine contact
Jacobians, SPD mass, and monotone bounded friction. It supplies neither
existence nor a global gap floor. UNKNOWN does not imply several solutions.

For `0 < g < w`, with constant `eta > 0` and `e >= 0`,

    f(g) = 2 * (eta/g**2 + e) * (w-g)**2 / w
    -f'(g) = 4*eta*(w-g)/g**3 + 4*e*(w-g)/w.

For one planar ideal-Coulomb contact, let the frozen contact Gram matrix
be `[[a,b],[b,d]]`, SPD, and reverse the tangent orientation if necessary
so `b >= 0`. If `alpha = mu*b-a > 0` and the free tangent velocity `t > 0`,
set `L = mu*d-b`, and define `g*` by `h*f(g*) = t/L`.
Exactly one solution **for every free normal state q, at fixed t** holds
if and only if `h**2*alpha*(-f'(g*)) <= 1`. Above that threshold there are
states with three solutions; individual states can still have one.
The API uses a sufficient whole-domain bound, so its strict inequality
does not implement that sharp one-contact iff boundary.

Ando's method deliberately holds the stiffness coefficient constant while
forming a local force/Hessian pair, then refreshes it in the next Newton
assembly. This is stated in the [paper, sections 1, 3.1 and 3.4](https://zozo.box.com/s/ckss1ejz0lbw848pg1qo7eqtvujsogy5).
The source block `4*(eta/g**2+e)*(1-g/w)` is the derivative with that
coefficient frozen. The mass term in the derivative of the updated force
is `w/g` times larger. That difference does not establish a source bug.
The formulas above describe a reduced frozen-geometry force system; they
do not certify the full external solver's geometry, stopping rule or path.
