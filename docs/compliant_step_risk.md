# Compliant-step motion bound

An implicit Coulomb residual can have several valid motion answers even when
normal stiffness is finite. `compliant_step_risk` provides an opt-in sufficient
uniqueness test. The rigid solver has no constitutive bound and cannot use it
without extra inputs. Existing solver defaults remain as defined by their API.

For two solutions put E=delta_lambda^T G delta_lambda, a=||delta_lambda_n||,
and T² >= ||J_t M^-1/2||². Normal complementarity and an inverse normal slope
s_min > 0 give delta_lambda_n dot delta_u_n <= -s_min*a². Maximal dissipation
of Coulomb friction gives delta_lambda_t dot delta_u_t <= mu_max*a*T*sqrt(E),
even at zero slip. Therefore

    E+s_min*a² <= mu_max*T*a*sqrt(E).

If s_min > mu_max²*T²/4, the quadratic is strictly positive for E or a nonzero;
all solutions have the same motion and normal impulses. Tangential redundant
forces can remain nonunique. Existence is a separate question.

The scalar helper uses normal_response_upper >= 1/s_min (kg), a tangential Gram
upper bound (1/kg), and mu_max. Its ratio is monotone increasing separately in
all three nonnegative bound inputs; this is the routing license. It uses outward
rounded scalar arithmetic. Input bounds must cover every admissible root; the
Jacobian at one computed root is insufficient. Matrix assembly and constitutive
bound certification are caller obligations. `UNKNOWN` is never evidence that
multiple roots exist. Refining a material-model time step can tighten the bound;
refinement alone does not certify temporal accuracy or resolve a rigid paradox.

For a zero-prestrain Hertz/Hunt-Crossley first step with static frozen J, reference
gap g>=0, h>0 and v_free, passivity gives ||v||_M<=||v_free||_M. Thus

    delta_max_i=max(h*||J_ni M^-1/2||*||v_free||_M-g_i,0)
    lambda_i=k_i*delta_i^(3/2)*(h+eta*(delta_i+g_i))
    response_upper=max_i h*k_i*sqrt(delta_max_i)*
                   (1.5*(h+eta*g_i)+2.5*eta*delta_max_i).

Use globally valid upper bounds in these expressions. For fixed initial rest
and gravity, v_free=h*a, so the response bound is monotone in h,k,eta and each
friction bound. This permits a local dyadic step refinement route. Positive
prestrain, moving surfaces, geometry changes, restitution, or tangential history
require a new passivity/reachability bound. No corner license for kinetic energy
or the number of branches is implied. Only the scalar ratio has this license.

Work: O(1) per scalar query after the bound setup/update. Sparse Gram row sums
cost O(nnz) if not already accumulated during assembly. These costs must be
charged; the function is not a claim of zero total compute cost.

`initial_hertz_step_risk` provides the constant-work route for the initial rest
case, from cached geometry and force/material bounds. With
D0=h²*normal_row_norm_upper*gravity_mass_norm_upper, every loaded contact has
g<=D0 and delta<=D0. Hence an additional safe, gap-independent bound is

    response_upper=h*K_H_max*sqrt(D0)*(1.5*h+4*eta_max*D0).

The bound is monotone in h and all supplied nonnegative upper bounds. A caller
can halve h until `refine_or_verify` is false while preserving K_H and eta.
The helper returns its scope explicitly. It is not a controller for following
history-bearing steps; those require the general helper with a new valid bound.
