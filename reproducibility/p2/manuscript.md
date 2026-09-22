# Deterministic sensitivities of Coulomb contact on GPUs

Anton Björkegren

## Abstract

Differentiating a frictional contact step requires agreement between the forward law and its local branch equations. We combine proximal ADMM for Coulomb contact with fixed-regime implicit differentiation and integer scatter accumulation on GPUs. A symmetric reduced operator is supplemented by an affine correction for non-associated slip. The measured implementation computes parameter directions; the complete mathematical adjoint requires transposing the coupled correction as well.

A scene-matched accumulation study reports derivative error, repeated bytes and execution cost from separate executions. A second matched study records the forward residual, the derivative error against a separately implemented CPU branch derivative, the repeated digest and the elapsed time from a single captured forward-and-gradient graph, one launch per direction. Eight of twenty-seven configurations meet both an absolute residual and a derivative error below 1e-8 in that graph; keeping the residual gate at 1e-8 N·s and admitting a derivative error below 1e-6 raises the count to thirteen. Each table row reports the maximum residual, the maximum derivative error and the median elapsed time over the three directions. A cube direction, an asymmetric tower and single-precision accumulation remain red at the tighter tolerance. Robot stance comparisons check the shared equations with analytical bilateral dynamics. Simulated identification through GPU sensitivities recovers the CPU information gain after each derivative chunk receives its own operator. The mass gain is 3.3426×. The friction gain is 1.7298×. A missing operator upload and an affine-origin error each produced repeatable but incorrect derivatives. CPU-reference and finite-difference checks expose why byte identity cannot validate a derivative. The repaired estimator is validated at the parameter state it accepts, and a separate residual gate removes the accepted states above tolerance while changing the estimate. The excitation still violates actuator limits and misses the friction-error target.

## 1 Introduction

A contact derivative used in estimation must describe how the solved contact law changes with the unknown parameter. Replacing Coulomb friction by a convex relaxation changes this map even when both numerical solves converge. Conversely, retaining the Coulomb law does not make a finite-budget iterate or its derivative accurate. The distinction becomes consequential near separation, where small numerical changes can alter the contact regime.

Complementarity formulations connect rigid contact to optimization without prescribing a contact sequence in advance, as demonstrated by Posa, Cantu and Tedrake [1]. Analytical differentiation of rigid body dynamics supplies another part of this computation [2]. Proximal methods address compliant and rigid contact within a common formulation [3]. Our contribution concerns their numerical implementation: a contact operator with deterministic accumulation, its fixed-regime sensitivity, and experiments that assess derivative accuracy independently of execution repeatability. We make no priority claim for the individual solver ingredients. Recently, Song, Fan, Ascher and Pai [5] adopt the same de Saxcé reduced-cone law and the cone-QP/scalar-coupling decomposition, but replace a Picard outer iteration by a Tseng forward–backward–forward splitting and replace the mixed-unit natural-map residual by a dimensionless three-part residual. Their outer correction is a single explicit forward step inside one iteration of a splitting scheme; it is not a parameter sensitivity, and that work reports no derivative path. The object measured here is therefore different: a fixed-regime parameter-direction sensitivity whose transposed adjoint we state but do not execute. Their inner problem is default block Gauss–Seidel over per-contact blocks, capped by a fixed sweep count in the quoted cost study, whereas the GPU batch path here uses a preconditioned conjugate-gradient inner solve; the two inner solvers and the two outer maps are not interchangeable. For deformable contact, Ménager and Carpentier [6] couple the Delassus operator to the global elastic tangent through G = H_c A⁻¹ H_cᵀ, so its inverse is a global solve rather than independent rigid-body inverse masses. Implicit-MPM contact therefore exists in the literature; no novelty is claimed for it, and the experiments below are rigid-body only. No priority verdict is drawn from the presence or absence of either work.

A sliding box distinguishes the contact laws through their normal separation. Robot stance tests then check the sensitivity where unilateral and bilateral models coincide, while slipping tests exercise the non-associated correction. These comparisons separate formulation error from derivative error. Object identification asks a further question: whether contact observations constrain the unknown parameters. Its GPU sensitivity is checked against a CPU branch derivative using the same observations. Reloading the operator before differentiating each chunk restores agreement with the CPU branch derivative on the tested observations. The remaining questions concern estimator error, forward convergence and excitation feasibility, rather than a demonstrated defect in the slip derivative.

## 2 Contact equations, sensitivity and stopping conditions

### 2.1 Forward problem

Let J map generalized velocities to contact velocities, M be the generalized mass matrix, and λ collect contact impulses in normal–tangential order. Define the Delassus operator G and the free contact velocity b by

\[
G=JM^{-1}J^\top,\qquad b=Jv_{\rm free},\qquad
v^+=v_{\rm free}+M^{-1}J^\top\lambda.
\]

A speculative offset, when present, is included in b. The regularized contact velocity is u=Hλ+b with H=G+E, where E contains isotropic compliance blocks. Compliance has inverse-mass units. Positive compliance changes the contact problem and can select an impulse in a hyperstatic configuration; it is not merely a preconditioner. With compliance, u includes Eλ and must be distinguished from the physical contact velocity Jv⁺. The rigid equations correspond to vanishing E.

For positive friction coefficient μ, define

\[
K_\mu=\{(\lambda_n,\lambda_t):\lambda_n\geq0,\ 
\|\lambda_t\|\leq\mu\lambda_n\},\qquad
\Gamma_c(u)=(\mu_c\|u_{t,c}\|,0,0).
\]

The de Saxcé formulation [4] is

\[
K_{\mu_c}\ni\lambda_c\perp u_c+\Gamma_c(u)\in K_{\mu_c}^*,
\qquad K_{\mu_c}^*=K_{1/\mu_c}.
\]

The frictionless limit uses λ_t=0 with scalar normal complementarity, rather than the reciprocal-friction cone. “Exact cone” refers to this friction formulation and its projection, not to an exact solution after a finite iteration budget.

**Lemma (closed sliding contact).** Suppose the normal impulse is positive and the tangential velocity is nonzero. Write d=u_t/‖u_t‖. The branch relations are

\[
\lambda_t=-\mu\lambda_n d,\qquad u_n=0\quad\text{(Coulomb)},\qquad u_n=\mu\|u_t\|\quad\text{(without the correction)}.
\]

*Proof.* Substitution into λᵀ(u+Γ)=0 cancels tangential dissipation against the normal correction. Division by the positive normal impulse gives the Coulomb relation. Without Γ, the tangential contribution remains, giving the relaxed normal velocity. For rigid contact without a speculative offset, multiplying the normal velocity by the step size gives the local separation increment. With compliance or an offset, the regularized normal velocity is not itself the physical separation rate. This argument concerns a fixed contact branch, not a rocking trajectory with changing contacts.

For frozen Γ, ADMM solves the cone-constrained quadratic

\[
\min_{z\in K_\mu}\frac12z^\top Hz+(b+\Gamma)^\top z.
\]

With unscaled dual variable γ, the ADMM updates are

\[
\begin{aligned}
(H+\rho I)x&=-(b+\Gamma+\gamma-\rho z),\\
z&=\Pi_{K_\mu}(x+\gamma/\rho),\\
\gamma&\leftarrow\gamma+\rho(x-z).
\end{aligned}
\]

Refreshing Γ supplies the outer nonlinear correction. Convergence of the frozen quadratic alone does not establish convergence of the Coulomb problem. The large-scene path applies G through a body scatter and contact gather; the articulated batch path instead stores dense operators for its small contact systems. A block Jacobi preconditioner supports the inner conjugate-gradient solve. A fixed inner budget and a refresh schedule are part of the algorithm being measured.

We assess contact convergence with the natural map, rather than the ADMM splitting residual:

\[
\begin{aligned}
\alpha_c&=\|H_{cc}\|_2^{-1},\\
r_{\rm nat}(\lambda)&=\max_c\left\|
\lambda_c-\Pi_{K_{\mu_c}}\bigl(\lambda_c-\alpha_c[u_c+\Gamma_c(u)]\bigr)
\right\|_\infty .
\end{aligned}
\]

Here α is distinct from the ADMM penalty ρ. The residual is an absolute impulse discrepancy, expressed in N·s for the translational contact coordinates used here. It is not divided by impulse magnitude. The implementation substitutes a unit scale if a diagonal block has zero norm. For the relaxed model Γ is suppressed in both the solve and this diagnostic. On the residual-gated batch path, an environment is frozen only when its checked residual is below the tolerance; exhausting the iteration budget is not a successful stop. The solve-cost and batch-rate tables distinguish the stopping threshold from the executed iteration count. Reference residuals in the slipping-reference table describe the states supplied to the derivative test, not a separate guarantee that every forward batch reaches those states.

A kernel estimate supplies the scale, so the tolerance is metric dependent. Replacing that estimate by a dense spectral norm changes the reported residual. At the 300-iteration series state, whose checked residual is 2.001447e-2 N·s, the relative change is 6.52e-4 and the absolute change is 1.30e-5 N·s; at a separate smoke state with residual 2.255035e-4 N·s the relative change reaches 8.06%. Neither state is converged to the 1e-5 gate, and neither is the converged env27 state used by the finite-difference study below, whose residual is 5.380e-13 N·s. A smaller relative metric disagreement is not a convergence certificate. The solution is unaffected because the regularized operator matches its exact form to 1e-16 and the ADMM penalty is built from Jacobi eigenvalues rather than from this estimate. The residual is refreshed every fourth iteration, so an iteration budget that is not a multiple of four leaves the last logged value stale. The matched budgets in this paper are multiples of four.

### 2.2 Branch differentiation and the affine correction

The derivative is local to the contact regime. Open contacts impose zero impulse. Sticking contacts impose zero regularized contact velocity, and slipping contacts impose

\[
u_{n,c}=0,\qquad \lambda_{t,c}+\mu_c\lambda_{n,c}d_c=0,
\qquad d_c=u_{t,c}/\|u_{t,c}\|.
\]

These equations use the same H as the forward solve. For a parameter perturbation, write h=δHλ+δb, so that δu=Hδλ+h. Changes in mass, geometry and compliance belong in h whenever they depend on that parameter. The post-step velocity sensitivity also differentiates the free velocity and the impulse-to-velocity map; δλ alone is insufficient for a position or mass derivative.

**Proposition (implicit sensitivity).** Collect the branch equations in R(λ,θ). If the regime persists, R is continuously differentiable in a neighborhood of the state, slip speeds are nonzero, and the impulse Jacobian is nonsingular, then

\[
R_\lambda\lambda_\theta=-R_\theta.
\]

For a differentiable scalar objective ℓ, its mathematical adjoint satisfies

\[
R_\lambda^\top p=\ell_\lambda,\qquad
\frac{d\ell}{d\theta}=\ell_\theta-p^\top R_\theta.
\]

*Proof.* Differentiate R(λ(θ),θ)=0 and solve for the impulse variation. Substitute it into the chain rule for ℓ and eliminate the variation using the transposed system. The assumptions exclude regime transitions and singular impulse distributions. In particular, a pseudo-inverse at a singular configuration does not establish a unique physical impulse derivative.

The measured implementation computes parameter-direction sensitivities. Its reuse of symmetric linear algebra can be derived without claiming that the complete Coulomb Jacobian is symmetric. At a slipping contact with positive normal impulse and positive friction coefficient, let d_perp be a unit tangent orthogonal to d. Write δλ=Zy+p₀, where p₀ accounts for the movement of the cone boundary under δμ. The local blocks are

\[
Z_s=\begin{bmatrix}1&0&0\\-\mu d&d_\perp&0\end{bmatrix},\qquad
D_s=\operatorname{diag}\!\left(0,\frac{\|u_t\|}{\mu\lambda_n},1\right),
\qquad p_{0,s}=(0,-\lambda_n d\,\delta\mu).
\]

Open contacts use a zero basis and identity D. Sticking contacts use an identity basis and zero D. The boundary-motion term vanishes in both cases. Let W select dᵀδu_t at slipping contacts, and let C inject μσ into the first reduced coordinate of each such contact. Define

\[
A=Z^\top HZ+D,\qquad g=Hp_0+h.
\]

The coupled equations executed by the sensitivity method are

\[
Ay=-Z^\top g-C\sigma,\qquad
\sigma=W(HZy+g),\qquad \delta\lambda=Zy+p_0.
\]

To verify the reduction, differentiate the slip direction as δd=(I−ddᵀ)δu_t/‖u_t‖. The tangential component orthogonal to d gives the curvature term in D. The first reduced component is δu_n−μdᵀδu_t; adding μσ recovers δu_n=0. Thus the correction is needed even though A is symmetric.

Eliminating y gives an affine correction map:

\[
\begin{aligned}
\sigma&=a+B\sigma,\\
a&=Wg-WHZA^{-1}Z^\top g,\\
B&=-WHZA^{-1}C.
\end{aligned}
\]

The implementation evaluates the zero input and basis inputs, solves (I−B)σ=a, and recovers δλ. This derivation assumes nonsingular A and I−B and exact inner solves; truncated conjugate gradients approximate the map. A poorly conditioned correction system can amplify inner-solve error. Applying A with conjugate gradients therefore neither solves the complete adjoint equation nor verifies a reverse-mode implementation. The reported direction rates measure the sensitivity computation just described.

**Corollary (friction information in strict stick).** If the other model quantities are independent of a contact's friction coefficient, that coefficient has zero local sensitivity while the contact remains strictly sticking and the branch Jacobian is nonsingular.

*Proof.* The sticking equations contain no friction coefficient. Their parameter right-hand side therefore vanishes, and nonsingularity gives zero impulse variation. Cone feasibility still gives an inequality constraint on the coefficient; the corollary concerns a local derivative rather than the absence of all information.

### 2.3 Transposition, conditioning and the numerical stop

The block structure makes the distinction between a sensitivity and an adjoint explicit. Before elimination, the reduced equations have the form

\[
\begin{bmatrix} A&C\\-WHZ&I\end{bmatrix}
\begin{bmatrix}y\\\sigma\end{bmatrix}
=\begin{bmatrix}-Z^\top g\\Wg\end{bmatrix}.
\]

Transposing this block system changes the direction of the coupling between the slip correction and the reduced impulse coordinates. The reduced operator A is symmetric; the off-diagonal coupling blocks are not transposes of one another. For a scalar objective depending on the impulse variation, let its reduced load be Zᵀℓ_λ. The transposed reduced equations are

\[
A^\top a-Z^\top H^\top W^\top b=Z^\top\ell_\lambda,
\qquad C^\top a+b=0.
\]

Eliminating b shows why transposing the entire coupled system is necessary. The same symmetric solve can be used within that elimination, but the correction factors must appear in reverse order. The transpose of a product is not obtained by applying its symmetric middle factor alone. Direct parameter dependence of the objective and of the particular solution also contributes to the final scalar derivative. These equations specify the mathematical adjoint; the numerical results below measure parameter directions through the untransposed affine solve. They do not constitute an execution test of this reverse calculation.

The unused coordinate in the slip basis makes the local blocks uniform across contact types. Its diagonal entry in D removes an artificial null direction and is not contact compliance.

A shared friction coefficient requires p₀ contributions from every affected slipping contact. A mass perturbation also changes the contact operator, so its right-hand side must include δHλ as well as δb.

Subtracting the origin evaluation from each basis evaluation isolates the linear part of the affine map. An incorrect origin therefore changes the derivative even when the forward state is unchanged; the validation below exhibits this failure.

There are consequently distinct sources of derivative error. The forward iterate can differ from the reference root. Its active labels can select a different branch. The inner solves can approximate the correct reduced equations insufficiently, and the affine reconstruction can magnify that error. Finally, near vanishing tangential speed, the branch derivative itself can be poorly conditioned. Increasing the gradient budget tests the inner-solve explanation while holding the supplied state fixed. Increasing the forward budget tests a different explanation and can also change the labels. A residual norm alone does not distinguish these mechanisms.

The natural-map residual cannot bound derivative error without control of the inverse branch Jacobian. Compliance changes both that inverse and the contact problem; reducing compliance is therefore a model change, not an unconditional accuracy improvement.

The matched experiment records a sampled first crossing in a separate forward probe. That probe refreshes the nonlinear correction every iteration, whereas the accuracy solve uses the cadence in Table K. Its crossing is therefore not a stopping certificate for the state used in the derivative comparison. The gradient-budget sweep compares a selected velocity direction with the largest tested budget; agreement there is a stability diagnostic against a finite solve, not an exact derivative certificate.

## 3 Repeatability and derivative validation

Integer accumulation addresses the order of the contact scatter. For scale S, contributions are converted to integers, summed, and converted back before the contact gather. Other reductions use a prescribed reduction layout. The scale is recomputed as a power of two from the maximum component magnitude and a host-computed bound on the scatter coefficients. This reduces the conversion quantum as the Krylov residual shrinks. This remains a quantized operator; no finite-termination theorem for exact-arithmetic conjugate gradients follows from it.

**Lemma (order independence).** For fixed converted contributions, if every intermediate integer sum is representable, the scatter result is independent of thread order.

*Proof.* Integer addition agrees with addition in the integers under the stated range condition, so associativity and commutativity apply to every ordering. Conversion back from the same final integer yields the same represented result. A bound on the sum of absolute converted contributions suffices to check representability of every ordering. The lemma does not establish that bound for arbitrary scenes or accuracy of the converted contributions.

The tested compiler configuration disables floating point fusion and fast-math transformations. Host-built inputs and device outputs must nevertheless be distinguished: a change in host linear algebra can change the input bytes before the deterministic scatter is reached. Table F reports cross-device equality for the recorded batch outputs, including sensitivities. Those runs are separate from the corrected-kernel accuracy tests in Table C. Equal hashes therefore establish repetition of the recorded computation, without establishing derivative accuracy or a guarantee across compilers and architectures.

Table A tests a different property. The exact and relaxed sliding cases use the same geometry, time step and iteration budget, so the normal-gap difference isolates the friction formulation within that code. A cross-engine comparison would also change contact generation and time integration. The slip lemma explains the local separation mechanism, while the trajectory maximum remains a measured quantity.

Table B compares with Pinocchio in a bilateral sticking stance at rest. With stabilization disabled and compatible contact frames, impulse and acceleration formulations obey λ=Δt f and v⁺=Δt a. The same scaling applies to torque derivatives. The position derivatives compose the contact sensitivity with finite differences of the reference library’s kinematic and inertial quantities; no finite difference passes through the contact solve. Away from a compatible initial velocity, imposing zero contact acceleration and imposing zero post-step contact velocity are different maps. We therefore do not treat their velocity derivatives as interchangeable. Slipping contacts require the independent Coulomb reference rather than a bilateral force that may violate the cone.

The relative error in Table B is the maximum absolute entrywise difference divided by the maximum absolute reference entry. Table C uses the maximum of the reference and computed entry magnitudes in the denominator, with zero reported for equal zero arrays. The latter test spans every impulse right-hand-side and friction-coefficient direction, comparing the batch implementation against a separately implemented CPU differentiation of the branch equations at the same regularized state. The reference states and contact equations are shared; this is an implementation check, not validation against an independent physical model.

The faulty sensitivity kernel reproduced its own bytes while returning incorrect slipping derivatives. It evaluated the nominal zero input of the affine correction map with a nonzero correction at a slipping contact. The resulting error contaminated the reconstructed map even though repeated execution returned the same bytes. Correctly zeroing that evaluation restores the agreement shown in Table C. Partial-slip cases expose the mechanism: moving the sticking contact from the first position to the last changes whether the erroneous write affects a slipping contact. Neither the contact residual nor repetition alone detects this defect. Increasing the inner-solve budget reduces the tower error before it reaches a plateau. This supports a truncation explanation for the tabulated tower result, without attributing every remaining error to the inner solve.

The missing chunk upload in identification provides a second example. CPU-reference comparisons and finite-difference checks, rather than byte identity, exposed the derivative discrepancies in these cases. In the chunk case, isolating a saved row and then restoring its own operator separates the driver error from the contact kernel. The slipping single-contact tests in Table W agree with the CPU branch derivative to rounding precision; finite differences provide a separate check with a larger numerical error. The single-contact tests validate the slipping branch at the prescribed roots; they do not cover contact transitions or arbitrary finite-budget states.

### 3.1 Quantization and host inputs

Adaptive scaling is needed because a Krylov iteration generates vectors of decreasing magnitude. With a fixed conversion quantum, a sufficiently small vector can be mapped to zero even though its direction remains relevant to the linear solve. Recomputing the scale from a deterministic magnitude reduction preserves information over a wider dynamic range. The bound used in that computation must cover the scatter coefficients, and integer representability remains a separate obligation. An experiment with a looser coefficient bound changes the quantization error, so the bound is part of the numerical configuration rather than an invisible implementation choice.

Updating a quantized body total by removing the old contact contribution and adding the new one makes it depend on current impulses. Rounding successive velocity increments can instead accumulate drift. Thread-order independence does not distinguish these storage rules.

Disabling fusion fixes whether the tested kernels contract multiplication and addition. Input identity remains a separate requirement: a changed host spectral estimate can alter every device iteration.

Host and device digests are therefore evaluated separately. In the source experiments, selecting the same OpenBLAS microkernel family reproduced the tested dot-product and inverse digests on the cloud host. The spectral eigensolver retained discrepancies in some cases. This intervention identifies a host channel for divergence without assigning every mismatch to the GPU. The cloud batch digests in Table F are complete output-array digests under their recorded budgets. They do not show that the corrected sensitivity kernel has been validated across those devices, because the corrected-kernel comparison is a separate experiment.

## 4 Execution cost and convergence

ADMM and projected Gauss–Seidel perform different local operations. Iteration counts consequently cannot substitute for elapsed time. Table D compares executed solves on the same cloud device, recording the budget for each method. PGS uses its iteration-loop timer; synthetic ADMM uses graph replay, whereas lattice ADMM includes residual checking in its forward-loop timer. The timings exclude upload, coloring and spectral setup, so their boundaries are not identical. The synthetic case uses a natural-residual gate; the lattice row compares fixed executed budgets. The lattice residuals come from the corresponding accuracy experiment, not from residual fields in the timing log. Its lower PGS residual means that the ratio of those times is not a matched-accuracy speedup. PGS wins the tabulated synthetic case; ADMM executes the lattice budgets faster. The synthetic ADMM timing excludes the search over penalty candidates, and both timings exclude setup. These rows do not establish a general crossover at matched accuracy.

Spectral penalties can expose a global mass-ratio mode that contact diagonal blocks miss. They do not remove the distinction between sticking, cone binding and near separation. The tested routing rules fail on some packing and nearly open configurations, and selection-set success does not establish generalization. Compliance can improve conditioning while changing the force distribution. We therefore make no universal penalty or compliance-error claim from these experiments.

Table E counts contact-kernel work on pre-exported scenes. A forward environment-step excludes robot dynamics export, collision detection, graph construction and parameter estimation. A sensitivity direction is one prescribed parameter perturbation per environment, not a full Jacobian or a complete simulation step with gradients. The number of directions matters when comparing the robot rows. Rates for the sensitivity phase exclude the forward solve and must not be read as combined forward-and-derivative throughput. The nominal sticking population contains slipping contacts, so the sensitivity workload includes a friction direction as well as a torque direction. The reported rates are execution measurements; no derivative-error gate accompanies each timed environment.

The cone-binding accuracy row uses a longer budget than its timing row. Throughput at that accuracy point remains unmeasured. The Talos rate measures the contact solve on an L4 using inputs exported beforehand. Dedicated-device and shared-device timings remain distinct; the latter are observed elapsed times on a shared GPU, affected by concurrent work, with the timing boundary stated for each table. None of these rates is a matched gradient comparison with another simulator.

### 4.1 Penalty selection and batch routing

A bilateral candidate can certify a sticking solution of the regularized contact equations at the supplied geometry. With the operator used by the forward problem, solve Hλ_bil=−b and test that the resulting impulse lies strictly inside every friction cone. The velocity equation then holds and the cone boundary is inactive. If the operator is nonsingular, this supplies the candidate impulse directly. Failure of the test does not prove that every contact must slip; it only rejects this all-sticking candidate. Singular systems need additional treatment and are outside this simple certificate.

The inner solve also changes the meaning of a penalty comparison. A penalty that minimizes outer iterations with an exact factorization can make the PCG system harder to solve. Holding the PCG budget fixed then changes the accuracy of each outer update. Increasing that budget may recover the expected outer behavior while losing elapsed time. Reporting outer iterations without the inner budget would hide this tradeoff. The matched accumulation experiment consequently holds its penalty and compliance fixed across accumulation modes instead of combining that ablation with a new routing decision.

Batch execution is valuable when the same kernel schedule serves many independent small systems. Every state, reduction scalar and accumulation buffer must retain an environment index. The dense articulated path stores the contact operator because those systems are small, while the large-scene path evaluates the body scatter and contact gather without storing the entire Delassus matrix. These are different memory regimes. The packing failures below concern the dense batch representation and cannot be generalized to every matrix-free implementation of the contact law.

### 4.2 Crossover and timer boundaries

The synthetic timing sweep first shows an ADMM advantage at 4096 contacts. Table I preserves the executed iteration budgets and the measured times at that point. This is a crossover in the sampled configurations, not a universal threshold in contact count. Graph topology, active regime, inner accuracy and setup cost all change the comparison. In particular, a lattice with a favorable global mode and a dense packing of similar contact count need not have similar behavior.

## 5 Accuracy, repeatability and cost on the same scenes

Table J crosses scene, batch size and accumulation method. Its scene pack supplies identical contact geometry, inverse mass, free velocity, friction and compliance to the accumulation variants. The reference solves the same regularized contact equations on the CPU and differentiates their branches separately. The GPU accuracy state comes from a reset forward solve, rather than being copied from the CPU solution. Thus the error includes the effect of the computed forward state as well as the sensitivity implementation. This is the relevant distinction from the saved-state tests in Table C.

The penalty is the geometric mean of the largest eigenvalue and the smallest eigenvalue retained by the numerical positivity cutoff for the regularized operator. The compliance uses the same contact-diagonal rule on both devices. The settings are collected in Table K. They are held fixed across accumulation variants, including modes that do not reach the forward threshold. In particular, the asymmetric tower does not use the regime-based penalty from the robot throughput experiment. Holding that failing penalty would conflate accumulation with a separate convergence failure.

The relative derivative error is

\[
e_{\rm rel}=\frac{\|V_{\rm GPU}-V_{\rm CPU}\|_F}{\|V_{\rm CPU}\|_F},
\qquad V=\frac{\partial v^+}{\partial\tau}.
\]

The accuracy loop reads the first environment after the reset forward solve. All environments in a cell share the same input scene, but this sampling is not an error check on every environment. The packing row samples only its saved subset of torque directions; the CSV records the direction count. Repeatability uses the concatenated impulse, post-step velocity and first torque-direction velocity derivative across the batch. Table L gives the full repeated integer digests. Equality refers to repeated graph launches within the measured configuration, not to equality between different batch sizes.

Cost and accuracy are matched by scene, input parameters and kernel, but are obtained from separate executions with different correction-refresh schedules in the crossing probe and accuracy solve. The timing loop replays a captured block and reports cost per ADMM iteration. The accuracy loop resets the solver and rebuilds its active state. The gradient timer and repeatability graph have their own captured execution paths. The direction timer measures a direction across the entire batch. This design measures an accumulation ablation on common configurations; it does not establish a rate of independently validated derivatives for every timed environment. That stronger conclusion would require an error gate on the exact outputs of each timed replay.

A second study measures accuracy, cost, residual and digest on a single execution. One forward-and-gradient graph per direction is captured, warmed, reset and launched once. The elapsed time is measured around that launch, and the forward residual, the derivative error against a separately implemented CPU branch derivative and the output digest are read from the same launch. Table Y reports the result on four small scenes at two batch sizes and three accumulation modes. Each row reports the maximum residual, the maximum relative derivative error and the median elapsed time over the three directions (and over the repeated processes where present), so those aggregates need not describe one launch. A host recomputation of the natural residual from the device arrays equals the device residual in every recorded launch. The joint gate is green in eight of twenty-seven cells when the residual and the derivative error are both below 1e-8. It is green in thirteen cells when the residual remains below 1e-8 and only the derivative tolerance is relaxed to 1e-6. The red cells stay red at the tighter tolerance: a cube direction, the asymmetric-tower forward residual, single-precision accumulation and the reduced-budget packing case. Repetition over three processes agrees for the integer digests and disagrees for single precision. This is a matched record on the tested small scenes, not a general error gate on every timed environment of Table J.

Integer accumulation gives identical recorded bytes in every cell with repeated integer launches. The packing record contains only a single launch, so its reported equality flag cannot establish repeatability. Floating-point atomics usually differ between repetitions, including double-precision accumulation. The single-environment cube is an exception in both floating modes, demonstrating that a repeated small example is insufficient to establish order independence. The double-precision results are often closer to the CPU derivative than the quantized integer results. Determinism and numerical accuracy therefore do not form a single ranking of accumulation methods.

On the sticking tower and lattice configurations whose derivatives are well resolved, the single-precision error is on the order of 1e-6. The matched table gives the observed accumulation-limited range. This derivative-error floor is distinct from the contact accumulation scale and from the natural-map threshold. Single precision can also prevent a threshold crossing within the forward budget. The missing crossings in Table J remain part of the result even when a row has a measured time. “Measured” in that table denotes completion of the measurement, not successful convergence.

The asymmetric tower reaches a relative derivative error of 6.953e-02 with integer accumulation. The same error in double precision rules out integer quantization as its sole explanation. The source budget check reduces the error when the forward solve is extended, while the gradient-budget sweep shows a much smaller truncation contribution at the tabulated state. The budget sensitivity implicates the forward state, but the separately measured threshold crossing does not certify the residual at the derivative state. This is a measured limit of the finite-budget sensitivity, not a footnote to the accurate sticking cases.

The packing is a second limit. Its measured integer row fails the forward tolerance and retains a substantial derivative error. A gradient direction exceeds the allowed run budget, so the remaining single-environment floating variants are budget exclusions rather than measured failures of those arithmetic modes. Larger batches exceed the declared memory allowance of the dense implementation. Table M states the memory estimates and the exclusion categories. No blank cell is filled by extrapolating a derivative or copying a hash from a smaller batch.

The cost of determinism is configuration dependent. Table J measures elapsed time per ADMM iteration and per batch direction, including work beyond accumulation. The integer overhead is smaller in the large-batch tower rows than in the cube rows. These measurements do not isolate a matrix-vector product or compare fixed point with another deterministic reduction.

## 6 Identification and information

The object experiment estimates box mass, yaw inertia and table friction from a simulated robot pushing a box. The standard and selected trajectories are fitted using the GPU affine sensitivity with the corrected origin evaluation and the current chunk operator. The separate CPU branch derivative supplies the reference comparison and the noise study in Tables G and H. Host code still assembles observation terms and the parameter update; GPU sensitivity does not imply that the entire estimator is captured on the device. Dynamics and hand kinematics are recomputed at the integrated state. The estimator conditions on measured joint velocities and uses a compliant contact model. It does not feed noisy torque into the free hand velocity, which would change the observation model. Contact candidates are prescribed for this trajectory, and joint positions and velocities are treated as noise-free. These assumptions limit the identification claim independently of the derivative implementation.

Let J_obs be the observation Jacobian and σ_τ the torque-noise standard deviation. Under independent Gaussian torque errors with common variance, the local information and relative standard-deviation bounds for a locally unbiased estimator with nonsingular information are

\[
F=J_{\rm obs}^\top J_{\rm obs}/\sigma_\tau^2,\qquad
c_j=\frac{\sqrt{[(J_{\rm obs}^\top J_{\rm obs})^{-1}]_{jj}}}{|\theta_j|}.
\]

Thus σ_τ c_j is a relative standard-deviation bound, not a variance bound. Interpreting these bounds as predictions of observed errors additionally requires the estimator to attain the local covariance bound. The information calculation alone establishes neither unbiasedness nor efficiency. Table G reports absolute parameter errors divided by the corresponding true parameter magnitude, then takes percentiles across environments. It applies the same acceptance threshold to mass and friction; yaw inertia is estimated jointly but is not covered by that acceptance test. Lower noise meets the target, while a longer observation window at the higher noise still fails it.

Stronger excitation changes this information, but the resulting maneuver must remain consistent with the model and the robot's limits. Table H compares the standard trajectory with the selected excitation under the same population and noise realization. The mass error decreases, whereas the friction target remains unmet. The selected trajectory loses hand contact frequently and exceeds the joint torque limit. Its true-parameter cost and estimator contact residual also rise, so its information score cannot be interpreted as a validated physical experiment. The cost is the sum of squared torque discrepancies across joints, steps and environments with observation noise removed; it is not a mean or a normalized likelihood. The standard trajectory itself exceeds a torque limit; it is not a feasible hardware baseline.

The local sticking corollary explains why friction probing needs a regime change. It does not ensure that the induced slip is observable above sensor noise. Nor does an information score certify contact preservation or actuator feasibility. No physical force or torque sensor validates the identification experiments, and the noise realization has not been replicated across independent noise seeds. A feasible probing policy and material calibration on hardware remain untested.

### 6.1 Identification through directional sensitivities

The estimator uses the contact sensitivity for mass, yaw inertia and the shared floor-friction coefficient. For inertia and mass, the parameter direction includes the variation of the contact operator acting on the current impulse. For friction, the particular impulse variation sums the moving-boundary terms over the slipping floor contacts. The latter is required because a shared coefficient affects more than an arbitrarily selected contact. The implementation supplies that sum explicitly to the affine sensitivity. Before the active-set construction, the driver uploads the current contact Jacobian, inverse mass matrix and associated operator data. The impulse, branch labels, preconditioner and direction right-hand side must all refer to that same observation chunk.

The experiment reuses the simulated observations. The population contains 512 environments. A defect in the driver left the final forward chunk resident during all derivative calls. Uploading the current chunk before constructing its active set corrects this state mismatch without changing the sensitivity kernel. Table N records the population and solver chunk separately. The single-chunk controls already agree with the CPU calculation, while the unrepaired multiple-chunk evaluations do not. Table X records this transition and the effect of the upload. Changing the observation window also changes the underlying data, so these controls do not replace a chunk-size sweep on one fixed trajectory; the repaired estimator supplies that sweep in the state-validation subsection.

On the standard trajectory, the integer path gives a median mass-error ratio to the local Gaussian CRB prediction of 0.9992. The corresponding friction ratio is 1.0263. Table O reports the errors and bounds rather than replacing them by a claim of universal statistical efficiency. The conversion from a Gaussian standard deviation to median absolute error uses the half-normal factor in Table N. Taking population medians before forming this ratio is an aggregate comparison; it does not prove that each environment is unbiased or attains its own information bound. The standard-trajectory tail differs from the CPU fit even though the reported information coefficients agree, so derivative parity is not stated as identity of complete optimization paths. The jointly estimated yaw inertia also has a relative-error tail exceeding unity, reported in Table O. Agreement of aggregate median-to-bound ratios does not make that tail acceptable.

Separate processes reproduce the numerical payload of the corrected short-window derivative check. Table P gives its canonical serialization digest and identifies the saved corrected parameter arrays separately. A parameter-array digest identifies a fit; it is not by itself a repetition experiment. The repeated full-fit integer and single-precision arrays in the control part of Table P were produced with the missing upload. Their spread in Table Q remains an observation of that defective driver. Corrected paired repetitions exist for the repaired driver, but they use a different population and budget: the corrected integer pair is byte-identical and the corrected single-precision pair differs in its accepted parameter array, at 128 environments and four outer iterations rather than the 512-environment control. No matched recomputation of the control spread statistic under the repaired driver is available, so these controls cannot establish the parameter spread of the repaired estimator.

In that defective-driver control, the mass tail reaches a relative absolute deviation of 2.26e-01. The friction tail reaches 4.36e-01. The stale operator prevents attributing either tail to accumulation alone or to a slip-regime derivative. Repetition matters when a consumer must reproduce a parameter update, but it does not improve the information in the observations or resolve a singular derivative. Yaw inertia remains in the spread table because omitting its tail would hide a failure in the joint fit. A matched repeated-fit comparison on this control population and window is needed before transferring these spread measurements to the corrected estimator.

The application timings cannot isolate an arithmetic speed penalty. They were collected with other work using the local GPU and are observed elapsed times on that shared device, affected by contention. Table R separates the corrected fits from the defective-driver timing controls. Neither an arithmetic speedup nor zero-cost determinism follows from these times. The dedicated-device study supplies the accumulation-cost comparison at its stated execution boundaries. It does not supply cost per derivative accepted by a common accuracy gate.

### 6.2 Excitation through the corrected sensitivity

The selected excitation improves the local mass information through the corrected GPU sensitivity by 3.3426×. The friction improvement is 1.7298×. Table S reports the underlying CRB coefficients alongside the CPU reference. These are ratios of population-median local bounds, not ratios of realized estimation errors. The observations and parameter population are held fixed between derivative paths. The agreement recovers the information benefit through the measured GPU implementation, rather than substituting a CPU result for a GPU measurement.

Table T compares the assembled observation derivatives after the operator upload. The forward impulses and observation costs continue to agree with the reference. The selected trajectory's mass-direction relative error is 2.568e-12. The standard trajectory retains a larger relative direction difference. A separate absolute-error scan places its largest component discrepancy on a nearly vanishing-slip row. The saved yaw-inertia directions have a maximum absolute difference of 1.502e-9 (m·s)⁻¹. These norms measure different quantities; a small absolute discrepancy can coexist with a larger relative error. Increasing the gradient solve budget does not resolve the remaining standard-trajectory difference. The records do not isolate regularization differences from all finite-solve effects, so no unique cause is assigned to that remainder.

The CPU branch formulation differentiates a normalized slip direction, whereas the reduced formulation places the corresponding curvature in D. The single-contact controls in Table W test both formulations at prescribed slipping roots. Their GPU–CPU agreement persists throughout the tested speed range. Finite differences agree less closely because they also include the nonlinear solve and differencing errors. Together with the chunk controls, these results locate the application defect in the driver’s operator state. They do not extend the implicit-function argument to a contact transition or the zero-speed limit.

Table U reports the corrected selected-excitation fit and retains the failed driver as a control. The corrected mass tail follows the CPU reference at the reported precision. The corrected friction tail also follows the reference but still exceeds the prescribed error threshold. The selected fit's maximum logged last-trial residual is 2.90e-2 N·s. Each log entry retains the final evaluated line-search trial, which can contain rejected candidates. It does not measure the accepted fit's residual or the maximum over every trial. The local information score is evaluated at the true parameters, whereas the estimator visits other parameter states; the two residual checks must remain separate.

Correcting the derivative driver does not remove the selected motion's contact-loss and actuator-limit violations in Table H. The recovered information gain is therefore a result for these simulated observations, not an admissible hardware probing policy. A design objective would need to enforce contact and actuator constraints while checking the observation model. Neither constrained design nor hardware validation is measured here.

### 6.3 State, provenance and stopping for the repaired estimator

The repaired identification driver is validated at the parameter state it actually accepts. One verification pass evaluates the forward residual, the cost and the assembled gradient at that state and returns a digest over the impulse, the residual, the cost and the assembled system together with the iteration and conjugate-gradient budgets and the elapsed times. The pass runs for the initial state and, at each outer iteration, for both the state where the gradient is assembled and the state that is accepted. The two states are logged separately and differ in every logged iteration except the first. This records the residual at the accepted parameter array, which the earlier estimator did not.

The accepted residual stays above the tolerance in a fraction of the accepted states under the cost-only rule. Table Z reports those fractions, the maximum accepted residual and the maximum last-trial residual for both excitations. A separate residual-gated variant accepts only states whose checked residual is below the tolerance, and it accepts no state above it. The gate changes the estimate. Table Z gives the gate and cost-only estimates side by side; for the selected excitation the yaw-inertia upper tail more than doubles under the gate, while for the standard push it is unchanged. The gate is reported as a separate variant with its own estimate, not as an improvement of the cost-only result.

Recomputing the failing states with nine times the forward budget reproduces every predetermined case at the original budget and moves most of them below the tolerance. The remainder lie on a residual plateau in which the cost continues to change. The ninefold budget is a work budget; the corresponding wall time was not measured separately.

The accepted parameter array is also used to sweep the solver chunk size. Five chunk sizes, including an incomplete final chunk, return a bit-identical digest at the accepted state, and two processes produce byte-identical records. The dynamic call count differs between the largest chunk and the smaller ones while the returned arrays do not. This is a chunk-size sweep on one fixed accepted trajectory, which the earlier window change did not provide.

Run provenance is captured in two phases. Source and input hashes are written before compilation and before the first launch. The loaded module files, their hashes and the module options are written after the loop. Loaded and frozen hashes agree in every checked binding, and floating-point fusion and fast-math transformations are disabled in every run. Two features limit the provenance claim. The pre-launch hashes are taken after the modules are imported, so they establish the code before compilation and launch, not before import. Some series and benchmark runs loaded a solver module whose only difference from the final module is the environment-variable selector line; both hashes are recorded, and the numerical path is unchanged. The driver scripts are hashed only in a post-hoc manifest, not in the per-run record.

The natural residual is measured with the kernel's own scale, the reciprocal of an analytical estimate of the largest regularized block eigenvalue. That estimate carries a small relative error against a dense eigenvalue computation, and replacing the kernel scale with a dense spectral norm changes the reported residual while leaving the solution unchanged. Table AA records the metric comparison. A CPU recomputation of the natural residual from the device impulse with the kernel scale agrees with the device residual at the 300-iteration series state (residual 2.001447e-2 N·s) to 8.199e-14 relative. That state is not converged to the 1e-5 gate, so the agreement is a metric check and not a convergence certificate. A separate CPU transcription of the branch sensitivity agrees with the device sensitivity at the same state.

The verification pass is not free. Table AA records its summed time against the total wall time of a series run. The extra accepted-state evaluation is part of that cost. All wall times are observed elapsed times on a shared device and are affected by contention.

The reverse calculation is not measured; the results above concern parameter directions through the affine solve. At the tested converged standard-push state, finite-difference relative errors range from 2.46e-8 to 2.32e-7 over the three perturbation sizes. At the tested converged selected-excitation state, they range from 5.13e-4 to 1.35e-2. Forward convergence therefore does not establish uniform finite-difference agreement at a common tolerance. The unconverged states have much larger discrepancies; this check does not separately quantify finite-forward-solve error, regime changes and differencing error. These measurements are reported separately from the device-against-CPU comparison. The largest scene is not run at its original batch size, and no result is extrapolated to it.

## 7 Limits of the integrated simulation

The contact and derivative checks do not establish successful pile dynamics. Correcting a cache that changed contact geometry even at fixed body poses allows the tested tower to satisfy its drift criterion. Under the corrected contact path, however, piles run from their initial states still fail to settle: PGS leaves unresolved contact residuals, and the tested ADMM configuration diverges. Cleaning the contact set improves conditioning without meeting the residual target. This rules out the identified cache defect as a complete explanation, but does not isolate every remaining geometric or integration error.

Alternative soft-step and barrier-inspired updates also fail to establish rest for the tested piles. Those ports omit or alter parts of the corresponding algorithms, so their failures are not evaluations of the published methods in full. A larger pile was only probed from a saved checkpoint; that experiment measures neither time to rest nor a successful trajectory from the initial state. The contact sensitivities validated here therefore support local calculations within measured regimes, with integrated stability and practical identification requiring separate evidence.

Finally, the experiments do not compare gradients with reverse-mode differentiation of unrolled ADMM, nor with another GPU simulator on an identical model and workload. The bilateral and branch references test specific equations; they are not a throughput competition between simulators. No physical force or torque sensor appears in the validation. The measured contribution is repeatable execution and directional differentiation on the tested regularized Coulomb states. Regime transitions, the asymmetric-tower error, packing failures and the missing common accuracy-and-cost record for the larger configurations still bound the result.

## 8 How to verify

The repository tests distinguish recomputed quantities from checks of stored observations. The CPU reference suite exercises cone projection, complementarity, regularized solves, finite-difference sensitivities and explicit handling of singular impulse distributions. The GPU cone suite executes CUDA kernels when the required device is available and checks repeatability, input-order behavior and comparison with the CPU reference. The batched-adjoint suite executes coupled all-slip examples against the CPU branch derivative; its broader saved-scene table is checked as data. Running that suite is not equivalent to rerunning every experiment in this paper.

The Pinocchio parity and multi-step GPU-scene tests read locked JSON values in the accompanying data. They compare those measurements with their recorded bounds. They do not export the robot models again, run Pinocchio, or simulate all trajectories. The identification tests recompute CRB arithmetic from saved Fisher matrices and compare saved parameter errors with their acceptance gates. That is a recomputation of the information calculation, not a fresh fit of the full simulated data. The router tests similarly combine calculations on constructed operators with assertions on saved sweep results.

The accompanying evidence contains the saved scene pack, matched-table CSV, raw cell records and identification arrays. Its manifest identifies supplied solver sources and run plans.

The public package regenerates cells whose raw records are supplied. Thirty printed numeric measurement cells have no bundled original per-run source; eighteen em-dash cells are placeholders for unavailable values. The package identifies each such cell and its affected claim in `SOURCE_AVAILABILITY.json`. These values are retained as reported observations or unavailable entries, and cannot all be independently recomputed from the delivered evidence. Some cell records omit kernel-version hashes, so file identity can be checked without certifying the executed binary in every run. The scene pack supplies the contact operators and CPU branch derivatives; the CSV retains forward residuals, direction counts and exclusions. The corrected identification records add the kernel digests, information coefficients and fit outputs; the driver scripts are hashed in the post-hoc manifest. These permit an audit of the correction and its reported numbers; retrospective hashes do not establish which binary a prior process loaded.

A common execution record is now available for the small-scene accumulation study: one timed graph per direction emits the final state whose residual, reference error, digest and time are recorded, with the same stopping rule and refresh schedule. A table row aggregates the directions by the maximum residual and derivative error and the median elapsed time. Table Y reports it, including the red cells. The larger configuration ablation still uses a separate probe, accuracy reset and timing replay, so it does not provide cost per accepted derivative, and no error measured on a separate state is attached to its timed states after the fact. Within the identification loop, the accepted state, its residual and the assembled system are recorded together (Table Z), but the cost of the extra verification pass is reported as a summed elapsed time rather than as a per-derivative rate.

Execution provenance is recorded at run time for the small-scene study and in two phases for the identification series, including the loaded module and reference digests, the serialized input digests, the graph boundaries and the numerical budgets. Two limits remain. The identification hashes are taken before compilation and launch but after import, and its driver scripts are hashed only in a post-hoc manifest. Cross-device equality of older outputs does not certify the corrected derivative on those devices; repeating the corrected kernel with the same serialized inputs and a common accuracy gate would test that claim. For identification, the chunk sweep is now run on the fixed accepted trajectory, which tests the driver-state invariant without changing the statistical experiment.

## Appendix A. Sliding contact

Table A shows the trajectory maximum normal gap for the same sliding-box configuration under the exact and relaxed contact laws. The relaxed law produces separation despite the matched geometry and iteration budget.

| Law | Box side [m] | Mass [kg] | Initial speed [m/s] | μ [–] | Δt [s] | Steps [–] | Sweeps/step [–] | Maximum gap [m] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coulomb | 0.20 | 1 | 1 | 0.3 | 1/240 | 200 | 400 | 1.552204e-11 |
| Convex relaxation | 0.20 | 1 | 1 | 0.3 | 1/240 | 200 | 400 | 4.352523e-3 |

## Appendix B. Bilateral derivative reference

Table B gives relative entrywise errors against analytical bilateral dynamics in sticking stance configurations, with the torque derivatives scaled from acceleration to velocity. Agreement tests the shared sticking equations; it does not test Coulomb slip.

| Robot | Velocity DOF [–] | Contacts [–] | δv⁺/δτ error [–] | δλ/δτ error [–] | δv⁺/δq error [–] | δλ/δq error [–] |
| --- | --- | --- | --- | --- | --- | --- |
| Unitree A1 | 18 | 4 | 1.22e-13 | 3.86e-12 | 2.99e-11 | 1.62e-10 |
| ANYmal C | 18 | 4 | 1.27e-12 | 3.51e-13 | 7.56e-10 | 4.75e-10 |
| Talos | 38 | 2 | 1.75e-14 | 7.20e-12 | 1.09e-09 | 5.23e-10 |

## Appendix C. Slipping derivative reference

Table C reports relative derivative errors for the faulty and corrected kernels at the same reference state. The correction changes the slipping derivatives, including the mixed case whose last contact slips. The compliance multiplier α_η sets η_c=α_η‖G_cc‖₂; it is distinct from the natural-map step scale. The reference residual is evaluated with the resulting regularized operator.

| Scene | Contacts [–] | Slip [–] | Compliance multiplier [–] | CG iterations [–] | Reference r_nat [N·s] | δλ/δb error, faulty [–] | δλ/δb error, corrected [–] | δλ/δμ error, faulty [–] | δλ/δμ error, corrected [–] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pushed cube | 4 | 4 | 0.01 | 60 | 9.853e-16 | 2.179e-01 | 1.326e-13 | 9.920e-01 | 1.427e-13 |
| Mass-ratio column | 3 | 0 | 0.01 | 60 | 2.132e-14 | 3.401e-13 | 3.401e-13 | 0.000e+00 | 0.000e+00 |
| Tower | 32 | 0 | 0.01 | 60 | 1.263e-15 | 1.751e-09 | 1.751e-09 | 0.000e+00 | 0.000e+00 |
| Sliding chain, one contact | 1 | 1 | 0.0 | 60 | 2.498e-16 | 3.610e-02 | 2.220e-16 | 5.758e-02 | 1.406e-16 |
| Sliding chain, two contacts | 2 | 2 | 0.0 | 60 | 4.441e-16 | 3.498e-02 | 1.243e-15 | 5.413e-02 | 2.044e-16 |
| Sliding chain, four contacts | 4 | 4 | 0.0 | 60 | 3.553e-15 | 3.315e-02 | 8.120e-15 | 5.121e-02 | 1.548e-14 |
| Sliding chain, eight contacts | 8 | 8 | 0.0 | 60 | 7.105e-15 | 3.020e-02 | 6.532e-14 | 5.018e-02 | 2.401e-12 |
| Mixed chain, stick first | 4 | 3 | 0.0 | 60 | 1.332e-15 | 3.166e-02 | 5.329e-15 | 4.576e-02 | 1.747e-14 |
| Mixed chain, stick last | 4 | 3 | 0.0 | 60 | 1.776e-15 | 1.104e-14 | 1.104e-14 | 2.044e-14 | 2.044e-14 |

## Appendix D. Executed solve costs

Table D compares PGS and ADMM on an L4. PGS takes less time for the synthetic case, while ADMM executes the stated lattice budget faster. The lattice residuals are from a separate accuracy experiment, so that row does not pair elapsed time with an observed stopping residual.

| Scene | Contacts [–] | Stopping specification | PGS sweeps [–] | ADMM iterations [–] | PGS [ms] | ADMM [ms] | PGS r_nat [N·s] | ADMM r_nat [N·s] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Synthetic | 256 | r_nat < 1e-8 N·s | 200 | 275 | 72.44 | 188.66 | 2.83e-9 | <1e-8 |
| Lattice | 40000 | Fixed budgets | 1000 | 25 | 445.61 | 84.09 | 7.3e-17 | 2.4e-15 |

## Appendix E. Batch budgets and rates

Table E separates contact-kernel throughput from the fraction below the natural-residual tolerance, with a dash denoting an unreported quantity at that budget. Increasing the cone-binding budget still leaves part of the population above tolerance. Rates are counts of executed directions, not counts of independently validated derivatives.

| Population | Device | Environments [–] | ADMM budget [–] | r_nat gate [N·s] | Fraction below gate [–] | Forward [env-steps/s] | Directions/env [–] | Sensitivity [env-directions/s] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1, nominal stick | RTX 5070 (shared GPU) | 4096 | 200 | 1e-8 | 0.9985 | 18 320 | 2 | 128 350 |
| A1, cone binding | RTX 5070 (shared GPU) | 4096 | 200 | 1e-8 | — | 9 559 | 2 | 68 488 |
| A1, cone binding | RTX 5070 (shared GPU) | 4096 | 2000 | 1e-8 | 0.7766 | — | — | — |
| ANYmal C, nominal stick | L4 | 4096 | 200 | 1e-8 | 0.9971 | 16 703.5 | 2 | 79 506.6 |
| Talos, stick | L4 | 4096 | 200 | 1e-8 | 1.0000 | 4 919.6 | 1 | 72 009.8 |

## Appendix F. Repeated batch bytes

Table F gives full SHA-256 digests of the concatenation (λ, v⁺, δλ/δτ, δv⁺/δτ, δλ/δμ), in that order. Each array is converted to contiguous float64 bytes before hashing. Equality across the listed devices certifies these recorded outputs, including any derivative error. The identity runs use their own forward budgets and are not the timed runs in Table E.

| Batch case | Environments [–] | ADMM budget [–] | L4 = A10G = H100: SHA-256 |
| --- | --- | --- | --- |
| A1, nominal stick | 4096 | 200 | `4d5d4747983a3de20aed4d49b12f453ced005275bce8edd94e956d4aa9c3c7d3` |
| A1, cone binding | 4096 | 500 | `24a370fedefcc7a7b2a3dddb7ebe1bc2432a0259bc2c2265304cb4db1bfd9bf6` |

## Appendix G. Identification under observation noise

Table G reports CPU-reference relative parameter errors over the simulated population and applies the acceptance test that mass and friction each meet the listed percentile threshold. The longer observation window reduces errors but still fails the target at the higher noise level.

| Trajectory | Environments [–] | Window [steps] | σ_τ [N·m] | Mass error p50 [–] | Mass error p95 [–] | Friction error p50 [–] | Friction error p95 [–] | p95 threshold [–] | Both pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Standard push | 512 | 50 | 0.1 | 4.0476e-03 | 1.5272e-02 | 4.3778e-03 | 1.6340e-02 | 0.05 | pass |
| Standard push | 512 | 50 | 1.0 | 3.8590e-02 | 1.5173e-01 | 4.1072e-02 | 1.6086e-01 | 0.05 | fail |
| Standard push | 512 | 100 | 1.0 | 2.2260e-02 | 8.9702e-02 | 1.9507e-02 | 8.3003e-02 | 0.05 | fail |

## Appendix H. Excitation and model consistency

Table H compares CPU-reference fits for the standard push and selected excitation on the same population, distinguishing the generator's contact residual from the estimator's residual at the true parameters. The residual columns report maxima over the evaluated contacts, environments and time steps. Lower mass error accompanies increased model discrepancy, lost contact and a torque-limit violation; the friction target remains unmet.

| Trajectory | Environments [–] | Window [steps] | σ_τ [N·m] | Mass error p95 [–] | Friction error p95 [–] | True-parameter squared torque cost [(N·m)²] | Estimator r_nat [N·s] | Generator r_nat [N·s] | No hand contact [fraction] | Peak joint torque / limit [–] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Standard push | 512 | 50 | 1.0 | 1.517289e-01 | 1.608568e-01 | 2.347430e-06 | 1.282599e-06 | 4.199e-12 | 0.0000 | 1.154 |
| Selected excitation | 512 | 50 | 1.0 | 3.818469e-02 | 8.413782e-02 | 1.503296e+01 | 1.481804e-02 | 2.161e-08 | 0.7213 | 5.284 |

## Appendix I. Synthetic crossover

Table I reports the uncontended cloud-L4 execution sweep. Contact count is the problem size, not the number of batched environments. These rows locate a sampled execution crossover with the stated budgets. An earlier archived timing run for the same configurations reports 40.41 ms / 93.30 ms at 32 contacts and 1137.06 ms / 735.02 ms at 4096 contacts; the difference is a version difference between two runs of the same sweep, and both sets are stated.

| Contacts [–] | PGS sweeps [–] | ADMM iterations [–] | PGS [ms] | ADMM [ms] | PGS / ADMM [–] |
| --- | --- | --- | --- | --- | --- |
| 32 | 200 | 200 | 40.45 | 93.39 | 0.43 |
| 256 | 200 | 275 | 72.58 | 188.86 | 0.38 |
| 4096 | 1000 | 825 | 1139.29 | 735.70 | 1.55 |

## Appendix J. Matched accumulation study

Table J preserves all planned cells. A dash denotes an unavailable result; budget and memory exclusions are not accuracy measurements. All times in this table are dedicated L4 measurements. A batch-direction time covers one prescribed perturbation in every environment. Probe crossings use the separate refresh schedule in Table K. The relative derivative error samples the first environment; measured status does not imply an accuracy pass.

| Scene | B [–] | Accumulation | Status | Probe crossing [iter] | rel_l2 [–] | SHA equal | Batch forward [ms/iter] | Batch direction [ms] | PCG stability [iter] |
| --- | ---: | --- | --- | ---: | ---: | :--: | ---: | ---: | ---: |
| Pushed cube | 1 | int64 | measured | 45 | 1.407e-07 | yes | 0.24 | 10.4 | 15 |
| Pushed cube | 1 | float64 | measured | 45 | 1.407e-07 | yes | 0.19 | 8.6 | 15 |
| Pushed cube | 1 | float32 | measured | 45 | 3.088e-07 | yes | 0.19 | 8.6 | 15 |
| Pushed cube | 256 | int64 | measured | 45 | 1.407e-07 | yes | 0.41 | 17.9 | 15 |
| Pushed cube | 256 | float64 | measured | 45 | 1.407e-07 | no | 0.28 | 13.1 | 15 |
| Pushed cube | 256 | float32 | measured | — | 3.088e-07 | no | 0.28 | 13.0 | 15 |
| Pushed cube | 4096 | int64 | measured | 45 | 1.407e-07 | yes | 0.59 | 26.0 | 15 |
| Pushed cube | 4096 | float64 | measured | 45 | 1.407e-07 | no | 0.43 | 20.3 | 15 |
| Pushed cube | 4096 | float32 | measured | — | 3.088e-07 | no | 0.43 | 20.3 | 15 |
| Sticking tower | 1 | int64 | measured | 129 | 2.132e-12 | yes | 0.53 | 7.5 | 60 |
| Sticking tower | 1 | float64 | measured | 113 | 2.418e-13 | no | 0.44 | 6.4 | 60 |
| Sticking tower | 1 | float32 | measured | 197 | 2.126e-06 | no | 0.44 | 6.3 | — |
| Sticking tower | 256 | int64 | measured | 129 | 2.132e-12 | yes | 3.05 | 41.9 | 60 |
| Sticking tower | 256 | float64 | measured | 113 | 2.411e-13 | no | 2.75 | 37.9 | 60 |
| Sticking tower | 256 | float32 | measured | — | 1.957e-06 | no | 2.65 | 36.6 | — |
| Sticking tower | 4096 | int64 | measured | 129 | 2.132e-12 | yes | 56.09 | 762.1 | 60 |
| Sticking tower | 4096 | float64 | measured | 113 | 2.416e-13 | no | 55.11 | 749.9 | 60 |
| Sticking tower | 4096 | float32 | measured | — | 2.174e-06 | no | 53.50 | 727.5 | — |
| Asymmetric tower | 1 | int64 | measured | 317 | 6.953e-02 | yes | 0.53 | 44.6 | 120 |
| Asymmetric tower | 1 | float64 | measured | 317 | 6.953e-02 | no | 0.45 | 38.1 | 120 |
| Asymmetric tower | 1 | float32 | measured | — | 4.264e+01 | no | 0.44 | 37.6 | — |
| Asymmetric tower | 256 | int64 | measured | 317 | 6.953e-02 | yes | 3.05 | 249.5 | 120 |
| Asymmetric tower | 256 | float64 | measured | 317 | 6.953e-02 | no | 2.75 | 225.9 | 120 |
| Asymmetric tower | 256 | float32 | measured | — | 4.282e+01 | no | 2.65 | 218.3 | — |
| Asymmetric tower | 4096 | int64 | measured | 317 | 6.953e-02 | yes | 56.24 | 4526.2 | 120 |
| Asymmetric tower | 4096 | float64 | measured | 317 | 6.953e-02 | no | 54.93 | 4449.1 | 120 |
| Asymmetric tower | 4096 | float32 | measured | — | 2.624e+01 | no | 53.66 | 4321.3 | — |
| Lattice | 1 | int64 | measured | 25 | 8.481e-12 | yes | 0.66 | 9.4 | 15 |
| Lattice | 1 | float64 | measured | 25 | 1.064e-12 | no | 0.54 | 7.8 | 15 |
| Lattice | 1 | float32 | measured | 25 | 3.667e-06 | no | 0.54 | 7.7 | — |
| Lattice | 256 | int64 | measured | 25 | 8.481e-12 | yes | 5.70 | 78.3 | 15 |
| Lattice | 256 | float64 | measured | 25 | 1.064e-12 | no | 5.23 | 71.9 | 15 |
| Lattice | 256 | float32 | measured | — | 3.364e-06 | no | 5.03 | 69.2 | — |
| Lattice | 4096 | int64 | measured | 25 | 8.481e-12 | yes | 110.34 | 1499.4 | 15 |
| Lattice | 4096 | float64 | measured | 25 | 1.064e-12 | no | 108.49 | 1470.3 | 15 |
| Lattice | 4096 | float32 | measured | — | 3.930e-06 | no | 105.25 | 1430.2 | — |
| Packing | 1 | int64 | measured | — | 3.426e-01 | untested | 119.46 | 104272.6 | — |
| Packing | 1 | float64 | infeasible_budget | — | — | — | — | — | — |
| Packing | 1 | float32 | infeasible_budget | — | — | — | — | — | — |
| Packing | 256 | int64 | infeasible_memory | — | — | — | — | — | — |
| Packing | 256 | float64 | infeasible_memory | — | — | — | — | — | — |
| Packing | 256 | float32 | infeasible_memory | — | — | — | — | — | — |
| Packing | 4096 | int64 | infeasible_memory | — | — | — | — | — | — |
| Packing | 4096 | float64 | infeasible_memory | — | — | — | — | — | — |
| Packing | 4096 | float32 | infeasible_memory | — | — | — | — | — | — |

## Appendix K. Matched protocol

Table K gives the numerical budgets used in the non-packing cells of the matched study. The packing run used a single timing sample and a single repeatability launch; its captured block length was ten iterations. Its gradient comparison used a reduced budget sweep and only the first torque direction. PCG stability compares a sampled direction with the largest tested budget and does not imply exact solution of the branch equations.

| Quantity | Value | Unit |
| --- | --- | --- |
| Batch sizes | 1, 256, 4096 | environments |
| Compliance | η_c = 1e-2 eigmax(G_cc) | kg⁻¹ |
| Penalty | √(λ_min⁺(H) λ_max(H)) | kg⁻¹ |
| Forward tolerance | 1e-8 | N·s |
| Forward PCG budget | 8 | iterations |
| Accuracy/captured correction refresh | 4 | ADMM iterations |
| Crossing-probe correction refresh | 1 | ADMM iteration |
| Retained eigenvalue cutoff | λ > 1e-12 max(λ_max, 1e-300) | kg⁻¹ |
| Forward budget, asymmetric tower | 400 | iterations |
| Forward budget, other scenes | 200 | iterations |
| Sensitivity PCG budget | 60 | iterations |
| PCG sweep | 15, 30, 60, 120, 240, 480 | iterations |
| PCG comparison threshold | 1e-10 | relative direction difference |
| Repeatability launches | 3 | launches |
| Timing repetitions | 5 | launches |
| Captured forward block | 25 | iterations |

## Appendix L. Matched integer digests

Table L records digests for the integer cells. All non-packing digests agree over repeated launches; the packing digest comes from a single launch and provides no repetition test. Digests serialize the concatenated arrays as contiguous bytes in the listed order.

| Scene | B [–] | SHA-256 of (λ, v⁺, ∂v⁺/∂τ₀) |
|---|---:|---|
| Pushed cube | 1 | `6ca701249c58ca50f947ee9d9b662da6622088b293d95cd75c023dd6eb94c4d1` |
| Sticking tower | 1 | `81005e51a433351cf881de3e2d4c20ba54e3bb0590983f952652319ca5396a7a` |
| Asymmetric tower | 1 | `e655a22d2c5716ff51e0a23828930d14d263398e296d13050dc0ec126eca3933` |
| Lattice | 1 | `6962a101ea3f91300e480d67b7d6c55486ecd9fe3ea7b0ab305ae416a9ea3ab6` |
| Pushed cube | 256 | `f2f8cea543d788dfcc01e89f117f6cad51e99588ae37aac29b76146ed105291d` |
| Sticking tower | 256 | `bc2681c2faef6337f5c207e28ae2961c1354b8f0ae48f42d2251bb9e7239ebce` |
| Asymmetric tower | 256 | `e116ac58d7d65be3ba0fc561ad7a115dcdda2087f005501f4b629a12f02cf18b` |
| Lattice | 256 | `44b30344e881917381a15d1807d7b39c44ce204d89b8edbcec0011357befee85` |
| Pushed cube | 4096 | `4122ff6790ad228603e9632f9e7bc245ebde6e57088917e1ccf69b2158eec1e9` |
| Sticking tower | 4096 | `014389bf8f1749726c977edb92820b227e078f90773ce7c7927564c537f3e032` |
| Asymmetric tower | 4096 | `f12dea6965cd942c4bf90829c0c0c67db390bb9b4e96e90f9befe82b97f71cb0` |
| Lattice | 4096 | `3c628b19c999fed5e1088c7ee97dc8b6ebaea6d99f24cdf1d3b6438fba58af7e` |
| Packing | 1 | `0cea5f6a31df384cc900c9d92683c8d195914fb76116ada8f58ad50444bbb372` |

## Appendix M. Packing exclusions

Table M separates a measured accuracy failure from unmeasured budget and memory exclusions. The memory estimates concern the full dense batch allocation, not only its Delassus array.

| Configuration | Count [cells] | Reason or measured value | Unit |
| --- | --- | --- | --- |
| Single-environment integer run | 1 | 3.426e-01 derivative error | relative |
| Single-environment integer run | 1 | 2.47e-5 forward residual | N·s |
| Single-environment integer run | 1 | 104272.6 per direction | ms |
| Single-environment floating modes | 2 | projected direction exceeds 90 | s budget |
| Batch 256, all modes | 3 | 45.3 estimated allocation | GB |
| Batch 4096, all modes | 3 | 725 estimated allocation | GB |
| Allocation allowance | — | 9 | GB |

## Appendix N. Identification protocol

Table N distinguishes trajectory population from solver chunking. The median-error prediction is conditional on centered Gaussian estimation errors attaining the local covariance bound.

| Quantity | Value | Unit |
| --- | --- | --- |
| Trajectory population | 512 | environments |
| Observation window | 50 | steps |
| GPU chunk | 1024 | environment-steps |
| Parameter iterations | 8 | iterations |
| Forward budget | 300 | iterations |
| Torque noise σ_τ | 1 | N·m |
| Gaussian median absolute-error factor | 0.6745 | – |
| Defective-driver integer repetitions | 3 | runs |
| Defective-driver floating repetitions | 5 | runs |
| Corrected derivative-check processes | 2 | processes |
| Corrected derivative-check window | 10 | steps |

## Appendix O. Standard-trajectory estimation

Table O reports the corrected GPU-driven estimator and the CPU branch reference under the same observation model. Parameter errors and CRB coefficients are relative to the true parameter magnitude; CRB coefficients are per unit torque-noise standard deviation. The CPU fit is retained as measured, including its different upper-tail errors.

| Path | Mass error p50 [–] | Mass error p95 [–] | Friction error p50 [–] | Friction error p95 [–] | Mass CRB p50 [(N·m)⁻¹] | Friction CRB p50 [(N·m)⁻¹] | Mass ratio [–] | Friction ratio [–] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU integer, corrected | 3.859002e-2 | 1.575692e-1 | 4.112602e-2 | 1.632154e-1 | 5.725943e-2 | 5.941064e-2 | 0.9992 | 1.0263 |
| CPU reference | 3.8590e-02 | 1.5173e-01 | 4.1072e-02 | 1.6086e-01 | 5.725943e-02 | 5.941064e-02 | — | — |

The remaining jointly estimated parameter and the residual log bound the interpretation of the standard fit. The yaw-inertia tail is not controlled by the mass-and-friction acceptance test. Logged residuals describe final line-search candidate batches, which may include rejected candidates; the residual of the accepted parameter array was not recorded for this earlier standard fit. The repaired estimator records it, and Table Z reports the accepted-state residual and the residual-gated variant.

| Standard corrected GPU diagnostic | Value | Unit |
| --- | --- | --- |
| Yaw inertia error p50 | 8.378583e-1 | relative |
| Yaw inertia error p95 | 1.352927e+1 | relative |
| Yaw inertia CRB p50 | 1.268415 | (N·m)⁻¹ |
| Yaw inertia median-error / CRB prediction | 0.9793 | ratio |
| Logged last-trial residual maximum | 8.857108e-1 | N·s |
| Logged last-trial residual, final iteration | 5.729944e-2 | N·s |
| Contact residual at true parameters | 1.282599e-6 | N·s |

## Appendix P. Identification parameter digests

Table P first identifies corrected parameter arrays and the repeated derivative-check payload. The printed derivative-check digest is the producer's canonical SHA-256 of the original run record, before public renaming of its `sim` input identifier. The cleaned public record has canonical digest `83f646c9d4334bbd96d423965d13c74179d4c2e299f3a29fb97710d2bf830b20`; the two digests identify different serializations of the same measured record. Check payloads omit only the run tag and use sorted JSON keys with compact separators and a final newline, encoded as UTF-8. The control digests below hash contiguous parameter-array bytes from the driver with the missing upload; their floating-point variation is quantified in Table Q.

| Record | SHA-256 |
| --- | --- |
| Corrected standard fit | `5e1fac9d8e9ff0658b28587d084fa78a650def50b3f216b33b4db4e417734057` |
| Corrected selected fit | `c86539873c98e1068c08a7fc87b3a55cb49df41b78c662a61f466f3d19967bc9` |
| Corrected derivative check, both processes | `658ffe86a40bb8587e46f3acbb2240b75e2ba0f7b537e328e79722b7437a909d` |

The following repeated-fit control uses the defective driver.

| Accumulation | Run [–] | SHA-256 |
| --- | --- | --- |
| int64 | 1 | `2b763253b03491ad33448fae6bdff386ddfe6a52cc3148f3199ae3f489a9b6e4` |
| int64 | 2 | `2b763253b03491ad33448fae6bdff386ddfe6a52cc3148f3199ae3f489a9b6e4` |
| int64 | 3 | `2b763253b03491ad33448fae6bdff386ddfe6a52cc3148f3199ae3f489a9b6e4` |
| float32 | 1 | `39dbda32ffbe2f9ab6245e980cab5575dbc577f8c00b432dc0273cb6ce452465` |
| float32 | 2 | `652611a7cbd928038cb7f5fc2db53f54aa760354483788613cc82dc6d3949d2c` |
| float32 | 3 | `d20638eebfd7197f47c451a765ab77d5a67c3c9e61dd4a59a56c6be3f8329b41` |
| float32 | 4 | `4fa4e7e108dd8c55f94cd1415b49f01ba8814156918e8d063c105ede6186b744` |
| float32 | 5 | `242f1fc734b5b866763ea52e9c74625d4affab529db4c8daf7b454356f23a82a` |

## Appendix Q. Parameter spread between repetitions

Table Q gives absolute deviations from each environment’s run-mean parameter, divided by the magnitude of that mean, for the defective-driver control only. Percentiles pool environment and repetition entries. These are not per-environment standard deviations. The integer arrays are byte equal, so their exact spread is zero; rounded mean arithmetic can leave a floating-point remainder. The repaired driver has paired single-precision repetitions, but at a different population and window; no matched recomputation of this spread statistic under the repaired driver is available.

| Accumulation | Parameter | Median [–] | p95 [–] | Maximum [–] |
| --- | --- | --- | --- | --- |
| int64 | mass | 0 | 0 | 0 |
| int64 | yaw inertia | 0 | 0 | 0 |
| int64 | friction | 0 | 0 | 0 |
| float32 | mass | 2.66e-06 | 1.06e-03 | 2.26e-01 |
| float32 | yaw inertia | 3.79e-05 | 1.98e-02 | 3.64e+00 |
| float32 | friction | 3.12e-06 | 8.23e-04 | 4.36e-01 |

## Appendix R. Identification elapsed time

Table R reports observed elapsed times on the local RTX GPU, which is shared with other work; contention prevents attribution of the difference to accumulation alone. The boundary is the per-iteration timer and the total wall time stated in the columns below. Corrected fits are listed first; the repeated accumulation comparison below used the driver with the missing upload.

| Corrected integer fit | Total [s] |
| --- | --- |
| Standard | 187.8 |
| Selected, noisy | 184.9 |
| Selected, noise-free | 172.2 |

Defective-driver timing controls:

| Accumulation | Run [–] | Median parameter iteration [s] | Total [s] |
| --- | --- | --- | --- |
| int64 | 1 | 19.75 | 179.6 |
| int64 | 2 | 26.52 | 234.4 |
| int64 | 3 | 41.04 | 349.3 |
| float32 | 1 | 39.59 | 340.1 |
| float32 | 2 | 39.71 | 341.1 |
| float32 | 3 | 35.67 | 283.7 |
| float32 | 4 | 21.61 | 264.8 |
| float32 | 5 | 18.92 | 163.6 |

## Appendix S. Excitation and local information

Table S holds the observations fixed while changing the derivative path. Corrected GPU coefficients agree with the CPU reference at the displayed precision. Improvement is the standard-to-selected ratio of population-median CRB coefficients.

| Quantity | Standard CPU | Standard GPU, corrected | Selected CPU | Selected GPU, corrected | Unit |
| --- | --- | --- | --- | --- | --- |
| Mass CRB p50 | 5.725943e-2 | 5.725943e-2 | 1.713034e-2 | 1.713034e-2 | (N·m)⁻¹ |
| Friction CRB p50 | 5.941064e-2 | 5.941064e-2 | 3.434491e-2 | 3.434491e-2 | (N·m)⁻¹ |
| Yaw inertia CRB p50 | — | 1.268415 | — | 4.869282e-2 | (N·m)⁻¹ |
| Mass improvement | — | — | 3.343 | 3.3426 | ratio |
| Friction improvement | — | — | 1.730 | 1.7298 | ratio |

## Appendix T. Observation-derivative diagnostic

Table T compares the assembled Gauss–Newton system and impulse directions with the CPU implementation. Errors are relative and dimensionless. Forward impulse and observation-cost differences are zero in all listed cases. The selected-trajectory check uses noise-free observations; the standard check uses the stated torque-noise model. Repaired and defective rows within each trajectory share that check's observations.

| Trajectory | Driver | Window [steps] | Hessian error [–] | Gradient error [–] | Mass direction error [–] | Inertia direction error [–] | Friction direction error [–] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Standard | Missing upload | 50 | 3.061e-2 | 3.084e-2 | 2.689e-2 | 3.386e-2 | 3.137e-2 |
| Standard | Corrected | 50 | 6.959e-12 | 5.155e-11 | 1.034e-8 | 1.634e-8 | 1.107e-9 |
| Selected | Missing upload | 50 | 13.46 | 3.745 | 1.518 | 2.401 | 0.6335 |
| Selected | Corrected | 50 | 4.263e-12 | 2.084e-13 | 2.568e-12 | 3.006e-11 | 8.272e-13 |

## Appendix U. Selected-excitation estimation and failure control

Table U separates the corrected noisy fit from the defective-driver failure. The corrected residual is the maximum over logged last-trial residuals. Each entry describes the final evaluated candidate batch for that parameter iteration, including rejected candidates; it is neither an accepted-state residual nor a maximum over all trials. The CPU column retains the measured reference percentiles.

| Quantity | Corrected GPU | Missing-upload control | CPU reference | Unit |
| --- | --- | --- | --- | --- |
| Mass error p50 | 1.126366e-2 | 1.6705e-02 | — | relative |
| Mass error p95 | 3.818469e-2 | 4.5614e-01 | 3.818469e-02 | relative |
| Friction error p50 | 2.318681e-2 | 3.1448e-02 | — | relative |
| Friction error p95 | 8.413783e-2 | 9.3655e-01 | 8.413782e-02 | relative |
| Yaw inertia error p50 | 3.2942e-2 | — | — | relative |
| Yaw inertia error p95 | 1.0716e-1 | — | — | relative |
| Mass median-error / CRB prediction | 0.9748 | — | — | ratio |
| Friction median-error / CRB prediction | 1.0009 | — | — | ratio |
| Yaw inertia median-error / CRB prediction | 1.0030 | — | — | ratio |
| Logged last-trial contact residual maximum | 2.90e-2 | — | — | N·s |
| Logged last-trial residual, final iteration | — | 4.092455e+5 | — | N·s |
| Logged last-trial residual maximum | — | 3.326372e+29 | — | N·s |

## Appendix V. Integrated pile failures

Table V reports runs from the initial state under the corrected geometry cache. A dash denotes an unavailable endpoint after divergence. The pile did not reach rest in the PGS runs.

| Bodies [–] | Method | Substeps [–] | Sweeps [–] | Observation end [s] | Final energy [J] | Maximum r_nat [N·s] | Divergence time [s] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 300 | PGS | 4 | 400 | 1 | 0.0328229386 | 0.0034598158 | — |
| 300 | ADMM | 4 | — | — | — | — | 0.345833333 |
| 100 | PGS | 4 | 400 | 1 | 0.0030221998 | — | — |
| 100 | ADMM | 4 | — | — | — | — | 0.3375 |

## Appendix W. Single-contact slip validation

Table W evaluates prescribed slipping roots with isotropic and positive-definite Delassus operators. The entries are maximum absolute component differences for the impulse derivative with respect to free contact velocity, with mass units. The isotropic rows report the GPU–CPU and CPU–finite-difference comparisons separately. The finite-difference error cannot be replaced by the smaller GPU–CPU discrepancy.

| Operator | Slip speed [m/s] | GPU–CPU [kg] | CPU–finite difference [kg] |
| --- | --- | --- | --- |
| Identity in stated units | 1e-6 | 1.11e-16 | 3.89e-11 |
| Identity in stated units | 1e-4 | 1.11e-16 | 1.10e-10 |
| Identity in stated units | 1e-2 | 1.11e-16 | 2.68e-11 |
| Identity in stated units | 1e-1 | 2.22e-16 | 1.03e-10 |
| Random positive definite, bound over sweep | 1e-6 … 1e-1 | ≤ 6.4e-16 | ≤ 2.3e-10 |

Suppressing the affine correction changes the isotropic derivative by 0.400 kg. Thus the agreement tests the corrected non-associated branch equations rather than an uncorrected symmetric solve. The sweep stays within the slipping branch and does not evaluate a derivative at its transition to stick.

## Appendix X. Chunk-state control

Table X changes the observation window at fixed solver chunk size. It distinguishes a missing-upload defect from a sensitivity-kernel error, but is not a chunk-size invariance sweep at fixed observations. Errors are relative to the CPU derivative.

| Window [steps] | Environment-steps [–] | Chunks [–] | Missing-upload Hessian error [–] | Missing-upload mass direction error [–] | Corrected Hessian error [–] | Corrected mass direction error [–] |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 512 | 1 | 5.101e-12 | 3.024e-12 | 5.101e-12 | 3.024e-12 |
| 2 | 1024 | 1 | 5.941e-12 | 4.355e-12 | 5.941e-12 | 4.355e-12 |
| 4 | 2048 | 2 | 1.769e-7 | 1.634e-4 | 2.123e-13 | 3.724e-12 |
| 10 | 5120 | 5 | 5.352e-5 | 4.971e-4 | 2.333e-13 | 1.396e-12 |
| 50 | 25600 | 25 | 3.061e-2 | 2.689e-2 | 6.959e-12 | 1.034e-8 |


## Appendix Y. Single-graph matched record

Table Y reports one timed forward-and-gradient graph per direction. The forward residual, the derivative error against the separately implemented CPU branch derivative and the output digest are read from that graph, and the time is measured around its launch. Each row reports the maximum forward residual, the maximum relative derivative error and the median elapsed time over the three directions (and over the repeated processes where present); those aggregates need not describe one launch. The joint gate with both the residual and the derivative error below 1e-8 keeps the residual gate at 1e-8 N·s while relaxing only the derivative tolerance to 1e-6 in the second gate column. The packing row uses a reduced budget. The median time is an observed elapsed time on a shared device, affected by contention.

| Scene | B [–] | Accumulation | Max forward residual [N·s] | Max derivative error [–] | Joint gate: res<1e-8, err<1e-8 | Joint gate: res<1e-8, err<1e-6 | Median time per direction launch [ms] |
| --- | ---: | --- | ---: | ---: | :--: | :--: | ---: |
| Pushed cube | 1 | float32 | 7.112e-09 | 4.583e-07 | no | yes | 51.51 |
| Pushed cube | 1 | float64 | 8.773e-09 | 4.228e-07 | no | yes | 52.64 |
| Pushed cube | 1 | int64 | 8.773e-09 | 4.228e-07 | no | yes | 65.93 |
| Pushed cube | 256 | float32 | 2.167e-08 | 4.583e-07 | no | no | 66.09 |
| Pushed cube | 256 | float64 | 8.773e-09 | 4.228e-07 | no | yes | 64.68 |
| Pushed cube | 256 | int64 | 8.773e-09 | 4.228e-07 | no | yes | 92.43 |
| Sticking tower | 1 | float32 | 1.042e-08 | 7.946e-06 | no | no | 207.6 |
| Sticking tower | 1 | float64 | 9.752e-09 | 1.207e-12 | yes | yes | 205.7 |
| Sticking tower | 1 | int64 | 9.500e-09 | 3.002e-11 | yes | yes | 263.5 |
| Sticking tower | 256 | float32 | 1.464e-08 | 7.952e-06 | no | no | 1091 |
| Sticking tower | 256 | float64 | 9.995e-09 | 1.231e-12 | yes | yes | 1118 |
| Sticking tower | 256 | int64 | 9.500e-09 | 3.002e-11 | yes | yes | 1255 |
| Asymmetric tower | 1 | float32 | 5.365e-03 | 5.225e+01 | no | no | 319.1 |
| Asymmetric tower | 1 | float64 | 2.774e-05 | 7.327e-04 | no | no | 309.2 |
| Asymmetric tower | 1 | int64 | 2.774e-05 | 7.326e-04 | no | no | 394.9 |
| Asymmetric tower | 256 | float32 | 1.050e-01 | 7.344e-04 | no | no | 1683 |
| Asymmetric tower | 256 | float64 | 2.774e-05 | 7.327e-04 | no | no | 1710 |
| Asymmetric tower | 256 | int64 | 2.774e-05 | 7.326e-04 | no | no | 1912 |
| Lattice | 1 | float32 | 4.345e-09 | 1.280e-05 | no | no | 417.6 |
| Lattice | 1 | float64 | 2.161e-09 | 4.071e-12 | yes | yes | 405.4 |
| Lattice | 1 | int64 | 2.162e-09 | 1.562e-11 | yes | yes | 536.7 |
| Lattice | 256 | float32 | 7.984e-09 | 1.299e-05 | no | no | 3310 |
| Lattice | 256 | float64 | 2.161e-09 | 4.055e-12 | yes | yes | 3391 |
| Lattice | 256 | int64 | 2.162e-09 | 1.562e-11 | yes | yes | 3669 |
| Packing | 1 | float32 | 5.732e-06 | 1.544e+00 | no | no | 2.203e+04 |
| Packing | 1 | float64 | 5.732e-06 | 1.544e+00 | no | no | 3.62e+04 |
| Packing | 1 | int64 | 5.732e-06 | 1.544e+00 | no | no | 2.203e+04 |

The repeated integer digests below are the sha256 of the concatenated impulse, post-step velocity and first torque-direction velocity derivative, read from the same graph. Three separate processes agree for the integer cases; the single-precision case disagrees and is reported as measured.

| Batch case | Processes [–] | Digest equal | SHA-256 of (λ, v⁺, ∂v⁺/∂τ₀) |
| --- | ---: | :--: | --- |
| Pushed cube, B=1, int64 | 3 | yes | `14b5816cbb5d3e1081b3d911807f48c4190ff97e2280a6cc7d61c00b1d27d0af` |
| Sticking tower, B=256, int64 | 3 | yes | `62ce9edbb5fee0e7b9f8d93f8aa1612e3cc485e55efad4899050fd355ac55ba7` |
| Asymmetric tower, B=256, int64 | 3 | yes | `898ab699b37238eb79d46f0025a8ddd01cbb855d74516845497513ed52fe2d65` |
| Pushed cube, B=256, float32 | 3 | no | `586ec0ef3baae8e5129957607c5121d3ef7fba4874a9efdde759fc8f99f506bd` |


## Appendix Z. Accepted-state and gate diagnostics for the repaired estimator

Table Z reports the record at the accepted parameter array. The residual tolerance is 1e-5 N·s. Counts are accepted states over the outer iterations of one series. The last-trial column is the maximum over the accepted states of the final evaluated line-search candidate, which may include rejected candidates and is a different state from the accepted array.

| Series | Acceptance | Accepted states [–] | Above tolerance [–] | Maximum accepted residual [N·s] | Maximum last-trial residual [N·s] |
| --- | --- | ---: | ---: | ---: | ---: |
| Standard push | cost-only | 458 | 20 | 2.785e-02 | 3.978e-02 |
| Selected excitation | cost-only | 493 | 76 | 1.828e-02 | 1.828e-02 |
| Standard push | residual gate | 463 | 0 | 2.415e-06 | 4.430e-02 |
| Selected excitation | residual gate | 475 | 0 | 9.931e-06 | 1.252e-02 |
| Standard push | cost-only, single precision | 453 | 15 | 2.132e-01 | 8.453e-02 |

The residual gate and the cost-only rule give different estimates from the same observations. Four series are listed; the digest identifies the accepted parameter array of each.

| Excitation | Acceptance | Mass p50 [–] | Yaw inertia p50 [–] | Yaw inertia p95 [–] | Friction p50 [–] | Accepted-array SHA-256 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Standard push | cost-only | 4.859e-02 | 7.754e-01 | 6.762e+00 | 4.412e-02 | `4e746550280c46a29e2fe1f87b4cb166ced2924c2efa66d38f4a023e6ee72403` |
| Standard push | residual gate | 4.728e-02 | 7.407e-01 | 6.762e+00 | 4.310e-02 | `e0f6be370effa7039033bfe70193015364c145f6be9f1d9edd8e74ab763e6c97` |
| Selected excitation | cost-only | 1.331e-02 | 3.489e-02 | 1.207e-01 | 2.561e-02 | `b7eb521442b479c37e7acefb5f86cc5a26211bc7dd1386b79f3fc0c78aa30812` |
| Selected excitation | residual gate | 1.505e-02 | 3.890e-02 | 2.817e-01 | 2.768e-02 | `c2b60e022e6d2eed5bdcb94260ddba3cc0674b0f1a72c572c87f6c0a49815829` |

The failing states were recomputed with nine times the forward budget on the same observations. Every predetermined case reproduces at the original budget; the recomputation moves a subset below the tolerance and leaves the rest on a plateau.

| Excitation | Predetermined cases [–] | Reproduced at budget [–] | Below tolerance after ninefold budget [–] |
| --- | ---: | ---: | ---: |
| Standard push | 6 | 6 | 4 |
| Selected excitation | 6 | 6 | 3 |

The accepted parameter array is reused for a chunk-size sweep. Five chunk sizes, including an incomplete final chunk, give a bit-identical digest at the accepted state; the dynamic call count changes with the largest chunk while the arrays do not.

| Trajectory | Iteration [–] | Chunk sizes [env-steps] | Final chunk [env-steps] | Calls [–] | Accepted digest matches | Maximum difference [–] |
| --- | ---: | --- | ---: | --- | :--: | ---: |
| Standard push | 0 | 6400, 2048, 1024, 512, 320 | 256 | 7/7/7/7/7 | yes | 0 |
| Standard push | 3 | 6400, 2048, 1024, 512, 320 | 256 | 7/7/7/7/7 | yes | 0 |
| Selected excitation | 0 | 6400, 2048, 1024, 512, 320 | 256 | 7/6/6/6/6 | yes | 0 |
| Selected excitation | 3 | 6400, 2048, 1024, 512, 320 | 256 | 7/6/6/6/6 | yes | 0 |



## Appendix AA. Provenance, residual metric and verification cost

Provenance counts are read from the run records. The pre-launch hashes are written before compilation and launch; the loaded-module hashes are written after the loop and agree with the frozen hashes for every checked binding. Two solver-module source hashes appear in the records; the earlier file differs from the final file by the environment-variable selector line only, and the numerical path is unchanged.

| Check | Count / value | Unit |
| --- | ---: | --- |
| Runs with pre-launch provenance before post-run provenance | 16 | runs |
| Loaded module hashes equal to frozen hashes | 80 | bindings |
| Canonical run records equal to per-run records | 13 | records |
| Saved parameter arrays equal to the recorded digest | 16 | runs |
| Floating-point fusion and fast math | disabled | – |
| Series and benchmark runs loading the earlier solver module | 9 | runs |
| Driver scripts in the per-run module provenance | 0 | scripts |

The natural residual is evaluated with the kernel scale. The kernel estimate of the largest regularized block eigenvalue carries a relative error against a dense eigenvalue computation; replacing it by a dense spectral norm changes the reported residual, while the solution is unchanged because the operator agrees with its exact form and the ADMM penalty is built from Jacobi eigenvalues.

| Metric | Value | Unit |
| --- | ---: | --- |
| Kernel largest-eigenvalue estimate, relative error against dense eigenvalues | 6.947e-04 | relative |
| Regularized operator agreement with the exact form | 1.356e-16 | relative |
| Kernel-scale natural residual, 300-iteration series state | 2.001447e-2 | N·s |
| Reported residual change, dense scale vs kernel scale, 300-iteration series state | 6.52e-4 | relative |
| Absolute residual change, dense scale vs kernel scale, 300-iteration series state | 1.30e-5 | N·s |
| Reported residual change, dense scale vs kernel scale, smoke state (residual 2.255035e-4 N·s) | 8.06 | % |
| CPU natural residual with the kernel scale vs device residual, 300-iteration series state | 8.199e-14 | relative |
| CPU branch sensitivity vs device branch sensitivity at the same state | 4.29e-13 – 4.68e-13 | relative |

The verification pass is not free. Its measured blocks are summed over the nine evaluations of a series and compared with the total wall time. All wall times are observed elapsed times on a shared device and are affected by contention.

| Quantity | Value | Unit |
| --- | ---: | --- |
| Verification pass, summed per series | 20.4–23.5 | s |
| Total wall per series | 32.2–37.2 | s |
| Earlier solver-module source hash | `3e9fdb13998f9a4c02d70318053d531f1489c4092ba3f76c06e7ccd533d05499` | – |
| Final solver-module source hash | `eb1f9a51434ac6d023cf4e765fdf02dbca4ca5e07a2d190ea46ad0be6a499632` | – |

Finite differences are reported separately from the device-against-CPU comparison. The table gives the minimum and maximum relative errors over three perturbation sizes for each state. The two converged states have different error ranges, and the unconverged states have much larger discrepancies. The reverse calculation is not measured.

| State | Environment [–] | Residual [N·s] | Finite-difference relative error [–] |
| --- | ---: | ---: | ---: |
| Standard push, converged | 27 | 5.380e-13 | 2.46e-08 – 2.32e-07 |
| Standard push, not converged | 0 | 2.001e-02 | 8.20e+03 – 9.46e+05 |
| Selected excitation, converged | 27 | 4.622e-12 | 5.13e-04 – 1.35e-02 |
| Selected excitation, not converged | 116 | 1.611e-02 | 2.35e+03 – 2.37e+03 |


## Appendix AB. Full parameter tail for the repaired estimator

Table AB reports the full estimated parameter vector, including the yaw-inertia tail. The cost-only and residual-gated variants share the observation model but accept different states. Single precision is included to show the accumulation effect on the tail.

| Series | Acceptance | Mass p50 / p95 / max [–] | Yaw inertia p50 / p95 / max [–] | Friction p50 / p95 / max [–] |
| --- | --- | --- | --- | --- |
| Standard push | cost-only | 4.859e-02 / 2.103e-01 / 8.059e-01 | 7.754e-01 / 6.762e+00 / 1.179e+02 | 4.412e-02 / 1.947e-01 / 5.761e-01 |
| Standard push, gate | residual gate | 4.728e-02 / 1.855e-01 / 4.720e-01 | 7.407e-01 / 6.762e+00 / 1.179e+02 | 4.310e-02 / 1.832e-01 / 5.761e-01 |
| Selected excitation | cost-only | 1.331e-02 / 4.205e-02 / 7.465e-02 | 3.489e-02 / 1.207e-01 / 2.668e-01 | 2.561e-02 / 8.967e-02 / 1.113e-01 |
| Selected excitation, gate | residual gate | 1.505e-02 / 6.296e-02 / 1.567e-01 | 3.890e-02 / 2.817e-01 / 5.497e-01 | 2.768e-02 / 1.841e-01 / 2.610e-01 |
| Standard push, single precision | cost-only | 4.974e-02 / 2.302e-01 / 6.308e-01 | 7.661e-01 / 6.762e+00 / 1.179e+02 | 4.598e-02 / 2.169e-01 / 9.874e-01 |


## References

[1] M. Posa, C. Cantu and R. Tedrake. A direct method for trajectory optimization of rigid bodies through contact. *International Journal of Robotics Research*, 33(1), 69–81, 2014. [DOI](https://doi.org/10.1177/0278364913506757).

[2] J. Carpentier and N. Mansard. Analytical Derivatives of Rigid Body Dynamics Algorithms. *Robotics: Science and Systems*, 2018. [DOI](https://doi.org/10.15607/RSS.2018.XIV.038).

[3] J. Carpentier, Q. Le Lidec and L. Montaut. From Compliant to Rigid Contact Simulation: a Unified and Efficient Approach. *Robotics: Science and Systems*, 2024. [DOI](https://doi.org/10.15607/RSS.2024.XX.108).

[4] G. de Saxcé and Z.-Q. Feng. The bipotential method: A constructive approach to design the complete contact law with friction and improved numerical algorithms. *Mathematical and Computer Modelling*, 1998. [DOI](https://doi.org/10.1016/S0895-7177(98)00119-8).

[5] H. Song, Y. Fan, U. M. Ascher and D. K. Pai. A splitting architecture for exact reduced Coulomb friction. *Computer Graphics Forum*, 45(8), 2026 (SCA). [arXiv](https://arxiv.org/abs/2607.19599).

[6] E. Ménager and J. Carpentier. Frictional contact solving for material point method. arXiv:2602.02038, 2026. [arXiv](https://arxiv.org/abs/2602.02038).
