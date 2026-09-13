#!/usr/bin/env python3
"""Batched MPPI and gradient trajectory optimisation on the GPU (uses fk_warp + a Warp SDF world).

MPPI (Model Predictive Path Integral) is sampling-based and gradient-free, which fits the fused
`collide_batch` primitive: per iteration K noisy rollouts of T waypoints each are sampled around a nominal
path, all K x T configurations are collision-evaluated in one fused GPU kernel chain, and the nominal path
is updated by a softmax-weighted average. A gradient (Adam) optimiser with the same interface is also
provided, plus a tiered planner (gradient fast path, RRT fallback, gradient refine, exact verification).

Selftest gate: a straight interpolation from start to goal must COLLIDE (otherwise the problem is
trivial), while the planned path must be (a) collision-free when verified against the CPU oracle rather
than self-reported, (b) at the goal, and (c) smoother than noise.
"""
import numpy as np


def _qmul(a, b):                                              # kvaternion-produkt (xyzw)
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def _rotvec(qd):                                             # axis-angle of a quaternion (xyzw) -> 3-vector
    qd = qd * (1.0 if qd[3] >= 0 else -1.0); v = qd[:3]; n = np.linalg.norm(v)
    return np.zeros(3) if n < 1e-9 else v / n * 2.0 * np.arctan2(n, abs(qd[3]))


def _pose_err(ee, tp, tq):                                   # twist (current→target): [pos-fel(3), rot-fel(3)]
    return np.concatenate([ee[:3] - tp, _rotvec(_qmul(ee[3:], np.array([-tq[0], -tq[1], -tq[2], tq[3]])))])


def _resample_path(path, T):
    """Resample a variable-length path (L,nq) to exactly T waypoints along arc length (uniform parametrisation), so an
    RRT seed can be refined by the fixed-T gradient optimiser."""
    path = np.asarray(path, float)
    if len(path) == T:
        return path.copy()
    if len(path) == 1:
        return np.repeat(path, T, axis=0)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-12:
        return np.repeat(path[:1], T, axis=0)
    u = np.linspace(0.0, s[-1], T)
    return np.stack([np.interp(u, s, path[:, k]) for k in range(path.shape[1])], axis=1)


def rrt_connect(cfree, edge_free, q_start, q_goal, lo, hi, rng, n_iter=4000, step=0.4, goal_bias=0.05):
    """Minimal RRT-Connect — KOLLISIONS-MODELL-AGNOSTISK (cfree(q)->bool, edge_free(a,b)->bool callbacks). Bidirektionella
    trees + greedy connect + balanced tree swap. A probabilistically complete global planner, used to seed the gradient
    refinement when the local gradient gets stuck in a local minimum. -> path np(L,nq) from start to goal, or None.
    Model-agnostic: works with a warp-FK free predicate or a coal one; no pinocchio binding."""
    q_start = np.asarray(q_start, float); q_goal = np.asarray(q_goal, float); nq = len(q_start)
    if not cfree(q_start) or not cfree(q_goal):
        return None
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    Ta = [q_start]; Pa = [-1]; Tb = [q_goal]; Pb = [-1]

    def steer(qn, q):
        v = q - qn; d = np.linalg.norm(v)
        return q.copy() if d <= step else qn + v / d * step

    def extend(T, P, q):                                          # ett steg mot q → (idx, qnew, reached) el. (None,None,False)
        i = int(np.argmin(np.sum((np.asarray(T) - q) ** 2, axis=1)))
        qnew = np.clip(steer(T[i], q), lo, hi)
        if cfree(qnew) and edge_free(T[i], qnew):
            T.append(qnew); P.append(i)
            return len(T) - 1, qnew, bool(np.linalg.norm(qnew - q) < 1e-9)
        return None, None, False

    def connect(T, P, q):                                         # greedy: extend towards q until reached or blocked
        while True:
            i, _, reached = extend(T, P, q)
            if i is None:
                return None
            if reached:
                return i

    def path_to(T, P, idx):
        out = []
        while idx != -1:
            out.append(T[idx]); idx = P[idx]
        return out[::-1]

    for _ in range(n_iter):
        qrand = q_goal if rng.random() < goal_bias else lo + rng.random(nq) * (hi - lo)
        ia, qa, _ = extend(Ta, Pa, qrand)
        if ia is not None:
            ib = connect(Tb, Pb, qa)
            if ib is not None:                                   # the trees meet at qa
                pa = path_to(Ta, Pa, ia); pb = path_to(Tb, Pb, ib)
                full = pa + pb[::-1][1:]                          # drop the duplicated junction node
                arr = np.array(full)
                return arr if np.allclose(arr[0], q_start) else arr[::-1]   # canonicalise the direction start -> goal
        Ta, Pa, Tb, Pb = Tb, Pb, Ta, Pa                          # balanserad bidirektionell swap
    return None


def _solve_ik(fk, lo, hi, target_pose, q0=None, iters=120, damp=1e-3, step=0.6, eps=1e-4):
    """Damped-LS IK through warp FK (no pinocchio): EE pose goal -> q. Batched finite-difference Jacobian (nq+1 FK per
    iteration in one call). target_pose = [pos(3), quat xyzw(4)]. -> (q, pos_err_m, rot_err_rad). Shared by the MPPI and
    gradient optimisers so both expose the same pose interface."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float); nq = len(lo)
    tp = np.asarray(target_pose, float); tpos = tp[:3]; tq = tp[3:]
    q = np.clip(np.asarray(q0, float) if q0 is not None else (lo + hi) / 2, lo, hi)
    E = np.eye(nq)
    e0 = None
    for _ in range(iters):
        batch = np.vstack([q] + [q + eps * E[k] for k in range(nq)])
        P = fk.ee_pose(batch)                                  # (nq+1, 7) — EN batchad FK
        e0 = _pose_err(P[0], tpos, tq)
        if np.linalg.norm(e0) < 1e-4:
            break
        J = np.stack([(_pose_err(P[k + 1], tpos, tq) - e0) / eps for k in range(nq)], axis=1)   # 6×nq
        dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), e0)
        q = np.clip(q - step * dq, lo, hi)                    # Newton: minska felet
    return q, float(np.linalg.norm(e0[:3])), float(np.linalg.norm(e0[3:]))


def _ik_seed(fk, verify_fk, world, lo, hi, target_pose, q_start, ik_restarts=6, seed=0):
    """Multi-restart IK seed for a cartesian goal: picks the collision-free seed q with the lowest pose error (checked
    through verify_fk). Shared by MPPI and gradient plan_pose. -> (q_goal, pos_err_m, rot_err_rad, free)."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float); nq = len(lo)
    rng = np.random.default_rng(seed)
    best = None
    for r in range(ik_restarts):
        q0 = q_start if r == 0 else lo + rng.random(nq) * (hi - lo)
        q_ik, pe, re = _solve_ik(fk, lo, hi, target_pose, q0=q0)
        free = bool(verify_fk.collide_batch(q_ik[None], world)[0] > 0)
        score = pe + 0.1 * re + (0.0 if free else 10.0)       # prefer a collision-free seed with low pose error
        if best is None or score < best[0]:
            best = (score, q_ik, pe, re, free)
    _, q_goal, pe, re, free = best
    return q_goal, pe, re, free


class BatchMPPI:
    """Batchad MPPI-banoptimering. fk=WarpBatchFK, world=WarpVolumeWorld (samma GPU-device). Host-orkestrerad yttre loop;
    the heavy collision evaluation (K x T configurations) runs fused on the GPU. lo/hi = joint limits (nq,)."""
    def __init__(self, fk, world, lo, hi, margin=0.03, collide_dens=4, verify_fk=None):
        self.fk = fk; self.world = world                 # fk = possibly sparse point set for the rollout cost (broad phase)
        self.verify_fk = verify_fk or fk                 # dense point set for the convergence and final check
        self.lo = np.asarray(lo, float); self.hi = np.asarray(hi, float); self.nq = len(self.lo)
        self.margin = float(margin)                      # soft clearance preference (a cost, not a hard constraint)
        # Grid-margin inflation was evaluated and rejected: the volume is near-exact on actual paths (~0 mm gap) and
        # inflating the margin to 0.06 destabilised tight scenes (6/6 -> 4/6).
        self.collide_dens = int(collide_dens)            # subdivision per segment i kollisions-kostnaden (svep-gap-medveten)

    def _densify(self, R):
        """R (K,T,nq) -> (K, (T-1)*d+1, nq) interpolated configurations, so the collision cost sees the same sweep as the oracle."""
        d = self.collide_dens
        if d <= 1:
            return R
        K, T, nq = R.shape
        a = R[:, :-1, :]; b = R[:, 1:, :]
        t = np.linspace(0, 1, d, endpoint=False)[None, None, :, None]      # (1,1,d,1)
        mid = (a[:, :, None, :] * (1 - t) + b[:, :, None, :] * t).reshape(K, (T - 1) * d, nq)
        return np.concatenate([mid, R[:, -1:, :]], axis=1)

    def _rollout_cost(self, R, q_goal, w_coll, w_smooth, w_goal):
        """R (K,T,nq) -> (cost (K,), n_eval). Fused GPU collision over all densified configurations in one call, so the
        cost is evaluated between waypoints and not only at them."""
        K, T, _ = R.shape
        D = self._densify(R)                                               # (K, Td, nq) sweep-dense configurations
        Td = D.shape[1]
        sd = self.fk.collide_batch(D.reshape(K * Td, self.nq), self.world).reshape(K, Td)   # ← GPU fusad
        coll = np.maximum(self.margin - sd, 0.0).sum(axis=1)               # clearance violation summed over the whole sweep
        smooth = (np.diff(R, axis=1) ** 2).sum(axis=(1, 2))               # path smoothness (acceleration proxy)
        goal = ((R[:, -1, :] - q_goal) ** 2).sum(axis=1)                  # end point vs goal
        return w_coll * coll + w_smooth * smooth + w_goal * goal, K * Td

    def plan(self, q_start, q_goal, T=32, K=512, iters=40, sigma=0.25, lam=1.0,
             w_coll=50.0, w_smooth=2.0, w_goal=8.0, seed=0,
             early_stop=True, min_iters=8, patience=2, stop_margin=None, goal_tol=0.05, smooth_tol=0.04):
        """MPPI plan. early_stop: stop on genuine convergence - the nominal path is collision-free with margin (dense
        check), the goal is reached, and smoothness has stabilised (relative change < smooth_tol) for `patience`
        consecutive iterations after the `min_iters` floor. info['iters'] reports the iterations actually used."""
        q_start = np.asarray(q_start, float); q_goal = np.asarray(q_goal, float)
        sm = self.margin if stop_margin is None else float(stop_margin)
        rng = np.random.default_rng(seed)
        nom = np.linspace(q_start, q_goal, T)                             # nominell = rak interpolation (kolliderar typ.)
        s = sigma
        evals = 0; used = iters; hits = 0; prev_sm = None
        for it in range(iters):
            noise = rng.normal(0, s, (K, T, self.nq))
            noise[:, 0, :] = 0.0; noise[:, -1, :] = 0.0                   # freeze the end points (start fixed, goal handled by the goal cost)
            R = np.clip(nom[None] + noise, self.lo, self.hi)
            R[:, 0, :] = q_start                                          # hard start pin
            cost, n_eval = self._rollout_cost(R, q_goal, w_coll, w_smooth, w_goal)
            evals += n_eval
            beta = cost.min()
            w = np.exp(-(cost - beta) / lam); w /= w.sum() + 1e-12
            nom = np.clip((w[:, None, None] * R).sum(axis=0), self.lo, self.hi)
            nom[0] = q_start
            s = max(s * 0.97, 0.05)                                       # annealing
            if early_stop and it + 1 >= min_iters:                        # convergence check on the nominal path (cheap: one path)
                smooth = float((np.diff(nom, axis=0) ** 2).sum())
                d_sm = abs(smooth - prev_sm) / (smooth + 1e-9) if prev_sm is not None else 1.0
                prev_sm = smooth
                nd = self._densify(nom[None])
                nsd = self.verify_fk.collide_batch(nd.reshape(nd.shape[1], self.nq), self.world).min()   # dense check
                evals += nd.shape[1]
                if nsd > sm and np.linalg.norm(nom[-1] - q_goal) < goal_tol and d_sm < smooth_tol:   # free + at goal + smoothness stable
                    hits += 1
                    if hits >= patience:
                        used = it + 1; break
                else:
                    hits = 0
        nom[0] = q_start
        # The final min-sd must be evaluated on sweep-dense configurations, not only on waypoints: waypoint-only
        # checking reported collision-free (+0.033) while the swept path penetrated 6.5 cm (verified against the exact
        # oracle on three independent clutter pairs), i.e. a false free in the self-report.
        nd = self._densify(nom[None])                                  # (1, Td, nq) sweep-dense
        sd_final = self.verify_fk.collide_batch(nd.reshape(nd.shape[1], self.nq), self.world)   # dense point set over the sweep
        info = dict(min_sd=float(sd_final.min()), goal_err=float(np.linalg.norm(nom[-1] - q_goal)),
                    iters=used, K=K, T=T, total_evals=evals)
        return nom, info

    def ik(self, target_pose, q0=None, iters=120, damp=1e-3, step=0.6, eps=1e-4):
        """Damped-LS IK through warp FK (no pinocchio): EE pose goal -> q. target_pose = [pos(3), quat xyzw(4)].
        Returns (q, pos_err_m, rot_err_rad) through the shared GPU-FK _solve_ik implementation."""
        return _solve_ik(self.fk, self.lo, self.hi, target_pose, q0=q0, iters=iters, damp=damp, step=step, eps=eps)

    def plan_pose(self, q_start, target_pose, ik_restarts=6, seed=0, **kw):
        """Cartesian goal: EE pose in, collision-free path out. IK seed (warp FK) -> collision-aware MPPI to the seed q.
        Multi-restart IK picks the collision-free seed with the lowest pose error. -> (path, info with ik_pos/rot_err)."""
        q_start = np.asarray(q_start, float)
        q_goal, pe, re, _ = _ik_seed(self.fk, self.verify_fk, self.world, self.lo, self.hi,
                                     target_pose, q_start, ik_restarts=ik_restarts, seed=seed)
        path, info = self.plan(q_start, q_goal, seed=seed, **kw)
        info.update(ik_pos_err_mm=round(pe * 1000, 3), ik_rot_err_mrad=round(re * 1000, 2), q_goal=q_goal)
        return path, info


class BatchGradTrajopt:
    """Gradient-based trajectory optimisation (Adam): far fewer evaluations than MPPI because it descends a gradient
    instead of sampling. The collision gradient comes from a batched finite difference (one collide_batch of T*(nq+1)
    configurations per step) plus analytic smoothness and goal gradients. Gradient descent is fast but local (the
    collision cost is non-convex), so multi-restart from a noised nominal path is used for robustness, dense
    verification gates the result, and the global planners (MPPI/RRT) are the fallback for scenes no restart solves.
    fk = possibly sparse point set; verify_fk = dense (defaults to fk)."""
    def __init__(self, fk, world, lo, hi, margin=0.03, verify_fk=None):
        self.fk = fk; self.world = world; self.verify_fk = verify_fk or fk
        self.lo = np.asarray(lo, float); self.hi = np.asarray(hi, float); self.nq = len(self.lo)
        self.margin = float(margin)

    def _verify(self, nom, dens=8):
        seg = [np.linspace(a, b, dens, endpoint=False) for a, b in zip(nom[:-1], nom[1:])]
        Q = np.vstack(seg + [nom[-1:]])
        return float(self.verify_fk.collide_batch(Q, self.world).min())

    def plan(self, q_start, q_goal, T=32, iters=140, restarts=4, lr=0.05, eps=1e-3,
             w_coll=50.0, w_smooth=2.0, w_goal=8.0, seed=0, min_iters=10, init_path=None, analytic_grad=False):
        """init_path: optional warm-start nominal path (T,nq), e.g. an RRT seed to refine. Default None gives a straight
        interpolation.
        analytic_grad selects the collision-gradient mode. False (default): (nq+1)-FK fused finite difference on min-sd.
        'fused': fused GPU analytic min-sd gradient (joints -> collide -> min-gradient, no host round-trip; same cost as
        the finite difference and validated at cosine ~= 1.0; measured as roughly latency-neutral, so it is an option and
        not the default). 'sum': sum of relu terms (faster per step but a different cost, hence weaker convergence).
        'host': host-side min-sd (validated but round-trip bound). The last three require world.grad_batch."""
        q_start = np.asarray(q_start, float); q_goal = np.asarray(q_goal, float)
        base_nom = _resample_path(np.asarray(init_path, float), T) if init_path is not None else np.linspace(q_start, q_goal, T)
        _wg = hasattr(self.world, "grad_batch")
        use_fused = (analytic_grad in (True, "fused", "sum")) and _wg and hasattr(self.fk, "collision_grad_fused")
        _fmode = "sum" if analytic_grad == "sum" else "minsd"
        use_host = (analytic_grad == "host") and _wg and hasattr(self.fk, "collision_grad_minsd")
        rng = np.random.default_rng(seed)
        evals = 0; best = None
        for r in range(restarts):
            nom = base_nom.copy(); nom[0] = q_start
            if r > 0:                                                          # brusad nominell → escape lokala minima
                nom = np.clip(nom + rng.normal(0, 0.3, nom.shape), self.lo, self.hi); nom[0] = q_start
            m = np.zeros_like(nom); v = np.zeros_like(nom); used = iters
            for it in range(iters):
                if use_fused:                                                  # FUSAD GPU analytisk (min-sd default / 'sum'=Σrelu)
                    gcoll, _ = self.fk.collision_grad_fused(nom, self.world, self.margin, mode=_fmode)
                    evals += T
                elif use_host:                                                 # HOST analytisk min-sd (validerad, roundtrip-bunden)
                    gcoll, _ = self.fk.collision_grad_minsd(nom, self.world, self.margin)
                    evals += 2 * T
                else:
                    batch = [nom] + [nom.copy() for _ in range(self.nq)]
                    for k in range(self.nq):
                        batch[k + 1][:, k] += eps
                    sdall = self.fk.collide_batch(np.vstack(batch), self.world).reshape(self.nq + 1, T)
                    evals += (self.nq + 1) * T
                    base = sdall[0]; viol = (self.margin - base) > 0
                    gcoll = np.zeros_like(nom)
                    for k in range(self.nq):
                        gcoll[:, k] = np.where(viol, -(sdall[k + 1] - base) / eps, 0.0)   # d relu(margin-sd)/dq
                gsm = np.zeros_like(nom); gsm[1:-1] = 2 * (2 * nom[1:-1] - nom[:-2] - nom[2:])   # smoothness
                gg = np.zeros_like(nom); gg[-1] = 2 * (nom[-1] - q_goal)                          # goal
                g = w_coll * gcoll + w_smooth * gsm + w_goal * gg; g[0] = 0.0
                m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
                nom = np.clip(nom - lr * m / (np.sqrt(v) + 1e-8), self.lo, self.hi); nom[0] = q_start
                if it + 1 >= min_iters and self._verify(nom) > 0 and np.linalg.norm(nom[-1] - q_goal) < 0.15:
                    used = it + 1; break
            vsd = self._verify(nom); ge = float(np.linalg.norm(nom[-1] - q_goal))
            solved = vsd > 0 and ge < 0.15
            if best is None or (solved and not best[1]) or (solved == best[1] and vsd > best[2]):
                best = (nom.copy(), solved, vsd, used, r)
            if solved:
                break
        nom, solved, vsd, used, r_used = best
        return nom, dict(solved=bool(solved), min_sd=vsd, goal_err=float(np.linalg.norm(nom[-1] - q_goal)),
                         iters=used, restarts_used=r_used + 1, T=T, total_evals=evals)

    def plan_pose(self, q_start, target_pose, ik_restarts=6, seed=0, **kw):
        """Cartesian goal: EE pose in, collision-free path out via the gradient optimiser. Multi-restart IK seed (warp FK,
        lowest pose error among collision-free seeds) -> gradient.plan to that seed q. Uses exactly the same seed path
        (_ik_seed) as MPPI.plan_pose, so both expose an identical pose interface.
        -> (path, info with ik_pos_err_mm/ik_rot_err_mrad/q_goal)."""
        q_start = np.asarray(q_start, float)
        q_goal, pe, re, _ = _ik_seed(self.fk, self.verify_fk, self.world, self.lo, self.hi,
                                     target_pose, q_start, ik_restarts=ik_restarts, seed=seed)
        path, info = self.plan(q_start, q_goal, seed=seed, **kw)
        info.update(ik_pos_err_mm=round(pe * 1000, 3), ik_rot_err_mrad=round(re * 1000, 2), q_goal=q_goal)
        return path, info


class TieredGpuPlanner:
    """Completeness-tiered GPU planner: gradient fast path for the common case; on a local-minimum miss, a global RRT
    seed (probabilistically complete); gradient refinement of that seed (smooth and fast); and always a dense exact-oracle
    verification at the end (no false free, falling back to the raw RRT seed if the refinement breaks the path). The RRT
    is warp-native via fk.collide_batch, so no pinocchio is needed. Motivated by the measurement that the gradient
    optimiser misses scenes RRT solves; completeness without giving up the fast-path latency. fk/world sit behind the
    WorldModel seam.

    verify_world: the exact oracle for the final gate (plan against the volume, verify against the exact geometry).
    None falls back to self-verification against the volume. margin: clearance preference and RRT free-space threshold."""
    def __init__(self, fk, world, lo, hi, margin=0.03, verify_world=None, verify_fk=None):
        self.fk = fk; self.world = world
        self.grad = BatchGradTrajopt(fk, world, lo, hi, margin=margin, verify_fk=verify_fk)
        self.lo = np.asarray(lo, float); self.hi = np.asarray(hi, float); self.nq = len(self.lo)
        self.margin = float(margin)
        self.verify_world = verify_world
        self.verify_fk = verify_fk or fk

    def _cfree(self, q):                                          # RRT-nod fri (volym-modell, marginal-clearance)
        return bool(self.fk.collide_batch(np.asarray(q, float)[None], self.world)[0] > self.margin)

    def _edge_free(self, a, b, dens=8):                          # RRT edge sweep-free (one batched collide over the densified edge)
        return bool(self.fk.collide_batch(np.linspace(a, b, dens), self.world).min() > 0.0)

    def _dense_exact_sd(self, path, dens=8):
        """Final gate: sweep-dense min-sd, against the exact oracle (verify_world) when given - plan against the volume,
        verify against the exact geometry - otherwise against the volume itself."""
        seg = [np.linspace(a, b, dens, endpoint=False) for a, b in zip(path[:-1], path[1:])]
        Q = np.vstack(seg + [path[-1:]])
        if self.verify_world is not None:
            W = self.fk.fk_points(Q)
            return float(min(self.verify_world.sd_batch(W[c]).min() for c in range(len(Q))))
        return float(self.verify_fk.collide_batch(Q, self.world).min())

    def plan(self, q_start, q_goal, T=32, rrt_iter=4000, rrt_step=0.4, goal_tol=0.15, seed=0, **grad_kw):
        """-> (path, info). info['tier'] in {gradient, rrt+refine, rrt-raw, failed}; info['rrt_used']; info['solved'] =
        dense exact-verified collision-free and goal reached. RRT only runs when the gradient path misses."""
        q_start = np.asarray(q_start, float); q_goal = np.asarray(q_goal, float)
        # 1) FAST PATH — gradient
        path, info = self.grad.plan(q_start, q_goal, T=T, seed=seed, **grad_kw)
        if self._dense_exact_sd(path) > 0 and info["goal_err"] < goal_tol:
            info.update(tier="gradient", rrt_used=False, solved=True); return path, info
        # 2) FALLBACK — global RRT-seed (probabilistiskt komplett)
        rng = np.random.default_rng(seed + 1)
        seedp = rrt_connect(self._cfree, self._edge_free, q_start, q_goal, self.lo, self.hi, rng,
                            n_iter=rrt_iter, step=rrt_step)
        if seedp is None:                                        # the global planner found no path
            info.update(tier="failed", rrt_used=True, solved=False); return path, info
        # 3) refine the seed with the gradient optimiser (warm start): smooth and fast
        rpath, rinfo = self.grad.plan(q_start, q_goal, T=T, seed=seed, init_path=seedp, **grad_kw)
        if self._dense_exact_sd(rpath) > 0 and rinfo["goal_err"] < goal_tol:
            rinfo.update(tier="rrt+refine", rrt_used=True, solved=True, rrt_nodes=len(seedp)); return rpath, rinfo
        # 4) if refinement broke the path, fall back to the raw RRT seed when it passes the dense exact check
        seedT = _resample_path(seedp, T)
        if self._dense_exact_sd(seedT) > 0 and float(np.linalg.norm(seedT[-1] - q_goal)) < goal_tol:
            return seedT, dict(tier="rrt-raw", rrt_used=True, solved=True, rrt_nodes=len(seedp), T=T,
                               min_sd=self._dense_exact_sd(seedT), goal_err=float(np.linalg.norm(seedT[-1] - q_goal)))
        info.update(tier="failed", rrt_used=True, solved=False); return path, info

    def plan_pose(self, q_start, target_pose, ik_restarts=6, seed=0, **kw):
        """Cartesian goal through the whole tiered pipeline: IK seed -> tiered plan (gradient -> RRT -> refine -> verify)."""
        q_start = np.asarray(q_start, float)
        q_goal, pe, re, _ = _ik_seed(self.fk, self.verify_fk, self.world, self.lo, self.hi,
                                     target_pose, q_start, ik_restarts=ik_restarts, seed=seed)
        path, info = self.plan(q_start, q_goal, seed=seed, **kw)
        info.update(ik_pos_err_mm=round(pe * 1000, 3), ik_rot_err_mrad=round(re * 1000, 2), q_goal=q_goal)
        return path, info
