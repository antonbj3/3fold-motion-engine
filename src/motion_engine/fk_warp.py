#!/usr/bin/env python3
"""Batched forward kinematics on the GPU (Warp), fused with SDF collision sampling.

FK, not the collision query, is the throughput bottleneck: a GPU collision query handles ~1.36M
configurations/s while CPU pinocchio FK tops out around 41k/s, so offloading only the collision query caps
on FK. This layer moves FK to the GPU: one @wp.kernel computes oMi for the whole chain (parent-ordered,
topological) over B configurations in parallel, and a second kernel transforms the robot surface points,
feeding straight into WarpVolumeWorld.sd_batch as a fused batched collision pipeline.

The chain spec arrives as plain numpy arrays (see the kinematics-spec extractor on the pinocchio side), so
there is no pinocchio import here. Supports 1-DOF revolute and prismatic joints; parity against pinocchio
is checked in the selftest. The NanoVDB volume path is CUDA-only.
"""
import numpy as np

import warp as wp

wp.init()


def _assert_finite_q(Q, where):
    """The warp/CUDA FK and collision kernels do not propagate NaN reliably: a NaN joint angle produced
    neither a NaN nor an exception, but silently gave a
    ANNAT, trovardigt-utseende FINIT kollisionsavstand (-0.0527 vs sann -0.0852, samma scen). Det slar ut
    varje nedstroms isfinite-baserad sakerhetskontroll i hela banplaneringen. Fixen hor hemma vid kallan:
    vakta Q innan den nar en karn-launch, vid varje publik ingang."""
    if not np.all(np.isfinite(Q)):
        raise ValueError(f"WarpBatchFK.{where}: Q carries non-finite (NaN/inf) joint values -- the "
                         f"underlying GPU kernel does not reliably propagate NaN (verified: it can silently "
                         f"return a different, plausible-looking finite result instead), so this must fail "
                         f"closed before ever reaching the kernel")


def _qrot(q, v):
    """Rotate vector(s) v (...,3) by quaternion(s) q (...,4 xyzw), pure vectorised numpy."""
    q = np.asarray(q, float); v = np.asarray(v, float)
    qv = q[..., :3]; w = q[..., 3:4]
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def subsample_spec(spec, stride):
    """Subsample the robot surface points (broad phase: stride times fewer points, so stride times faster collision
    evaluation). The kinematic chain is unchanged. Use a sparse spec for optimisation and a dense one for final
    verification."""
    s = dict(spec); idx = np.arange(0, len(spec["pt_joint"]), int(stride))
    s["pt_joint"] = np.asarray(spec["pt_joint"])[idx]; s["pt_parent"] = np.asarray(spec["pt_parent"])[idx]
    return s


@wp.kernel
def _fk_joints(parent: wp.array(dtype=wp.int32), jpl: wp.array(dtype=wp.transform),
               axis: wp.array(dtype=wp.vec3), jtype: wp.array(dtype=wp.int32),
               qidx: wp.array(dtype=wp.int32), Q: wp.array2d(dtype=wp.float32),
               nj: int, oMi: wp.array2d(dtype=wp.transform)):
    c = wp.tid()                                              # one thread per configuration; sequential joint loop (parent < j topologically)
    oMi[c, 0] = wp.transform_identity()
    for j in range(1, nj):
        qj = float(0.0)
        qi = qidx[j]
        if qi >= 0:
            qj = Q[c, qi]
        mot = wp.transform_identity()
        if jtype[j] == 1:                                     # revolut: rotation kring axeln
            mot = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_from_axis_angle(axis[j], qj))
        if jtype[j] == 2:                                     # prismatic: translation along the axis
            mot = wp.transform(axis[j] * qj, wp.quat_identity())
        local = wp.transform_multiply(jpl[j], mot)            # jointPlacement ∘ jointMotion
        oMi[c, j] = wp.transform_multiply(oMi[c, parent[j]], local)   # oMi[parent] ∘ local


@wp.kernel
def _fk_points(oMi: wp.array2d(dtype=wp.transform), pt_joint: wp.array(dtype=wp.vec3),
               pt_parent: wp.array(dtype=wp.int32), out: wp.array2d(dtype=wp.vec3)):
    c, k = wp.tid()                                           # 2D: (konfig, punkt)
    out[c, k] = wp.transform_point(oMi[c, pt_parent[k]], pt_joint[k])


@wp.kernel
def _fk_ee(oMi: wp.array2d(dtype=wp.transform), ee_parent: int, ee_pl: wp.transform,
          out: wp.array(dtype=wp.transform)):
    c = wp.tid()                                             # EE world pose = oMi[ee_parent] * ee_placement
    out[c] = wp.transform_multiply(oMi[c, ee_parent], ee_pl)


@wp.kernel
def _fk_collide(oMi: wp.array2d(dtype=wp.transform), pt_joint: wp.array(dtype=wp.vec3),
                pt_parent: wp.array(dtype=wp.int32), vol: wp.uint64, min_sd: wp.array(dtype=wp.float32)):
    c, k = wp.tid()                                           # fused: transform the point, sample the SDF world, reduce to a minimum
    p = wp.transform_point(oMi[c, pt_parent[k]], pt_joint[k])
    uvw = wp.volume_world_to_index(vol, p)
    sd = wp.volume_sample_f(vol, uvw, wp.Volume.LINEAR)
    wp.atomic_min(min_sd, c, sd)                              # minimum sd over the robot points = distance to the nearest obstacle


@wp.kernel
def _fk_collide_grad(oMi: wp.array2d(dtype=wp.transform), jpl: wp.array(dtype=wp.transform),
                     parent: wp.array(dtype=wp.int32), axis: wp.array(dtype=wp.vec3), jtype: wp.array(dtype=wp.int32),
                     qidx: wp.array(dtype=wp.int32), pt_joint: wp.array(dtype=wp.vec3), pt_parent: wp.array(dtype=wp.int32),
                     vol: wp.uint64, inv_vs: float, margin: float,
                     grad: wp.array2d(dtype=wp.float32), min_sd: wp.array(dtype=wp.float32)):
    # Fused analytic collision gradient (sum of relu terms): per (configuration, point) sample sd and grad sd; if the
    # point is colliding, walk the ancestor chain (link -> root) and atomic-add d(-sd)/dq_a = -grad sd . (z_a x (p - o_a))
    # per joint. One GPU kernel, no host round-trip.
    c, k = wp.tid()
    L = pt_parent[k]
    p = wp.transform_point(oMi[c, L], pt_joint[k])
    uvw = wp.volume_world_to_index(vol, p)
    g3 = wp.vec3(0.0, 0.0, 0.0)
    sd = wp.volume_sample_grad_f(vol, uvw, wp.Volume.LINEAR, g3)
    wp.atomic_min(min_sd, c, sd)
    if sd < margin:                                              # only colliding/near points contribute (relu condition)
        gsd = g3 * inv_vs                                        # index-space gradient -> world gradient
        x = L
        while x > 0:                                             # ancestor chain link -> root (only joints that move the point)
            a = qidx[x]
            if a >= 0:
                Fp = oMi[c, parent[x]]                           # frame_x = oMi[parent]∘jpl[x]
                fq = wp.transform_get_rotation(Fp); ft = wp.transform_get_translation(Fp)
                z = wp.quat_rotate(fq, wp.quat_rotate(wp.transform_get_rotation(jpl[x]), axis[x]))   # world axis
                dp = z
                if jtype[x] == 1:                               # revolut: z×(p−o); prismatisk: z
                    o = ft + wp.quat_rotate(fq, wp.transform_get_translation(jpl[x]))
                    dp = wp.cross(z, p - o)
                wp.atomic_add(grad, c, a, -wp.dot(gsd, dp))     # d(-sd)/dq_a, summed over colliding points
            x = parent[x]


# Determinism: the float32 wp.atomic_add above is order-dependent (parallel add order changes the rounding;
# measured 7.3e-4 cross-run jitter in sum mode). This variant accumulates in fixed-point int64 (integer
# atomic_add is associative, hence bit-deterministic) and a finalize kernel scales back to float. Sum mode
# only; the default min-sd path is already deterministic. GRAD_SCALE=1e9 gives ~1 nm equivalent resolution.
@wp.kernel
def _fk_collide_grad_i64(oMi: wp.array2d(dtype=wp.transform), jpl: wp.array(dtype=wp.transform),
                         parent: wp.array(dtype=wp.int32), axis: wp.array(dtype=wp.vec3), jtype: wp.array(dtype=wp.int32),
                         qidx: wp.array(dtype=wp.int32), pt_joint: wp.array(dtype=wp.vec3), pt_parent: wp.array(dtype=wp.int32),
                         vol: wp.uint64, inv_vs: float, margin: float, scale: wp.float64,
                         grad_i64: wp.array2d(dtype=wp.int64), min_sd: wp.array(dtype=wp.float32)):
    c, k = wp.tid()
    L = pt_parent[k]
    p = wp.transform_point(oMi[c, L], pt_joint[k])
    uvw = wp.volume_world_to_index(vol, p)
    g3 = wp.vec3(0.0, 0.0, 0.0)
    sd = wp.volume_sample_grad_f(vol, uvw, wp.Volume.LINEAR, g3)
    wp.atomic_min(min_sd, c, sd)                                 # min is associative and exact, hence already deterministic
    if sd < margin:
        gsd = g3 * inv_vs
        x = L
        while x > 0:
            a = qidx[x]
            if a >= 0:
                Fp = oMi[c, parent[x]]
                fq = wp.transform_get_rotation(Fp); ft = wp.transform_get_translation(Fp)
                z = wp.quat_rotate(fq, wp.quat_rotate(wp.transform_get_rotation(jpl[x]), axis[x]))
                dp = z
                if jtype[x] == 1:
                    o = ft + wp.quat_rotate(fq, wp.transform_get_translation(jpl[x]))
                    dp = wp.cross(z, p - o)
                # fixed point: f64 product (exact from f32 inputs) -> truncated int64 -> associative atomic_add
                wp.atomic_add(grad_i64, c, a, wp.int64(wp.float64(-wp.dot(gsd, dp)) * scale))
            x = parent[x]


@wp.kernel
def _i64_to_f32_grad(grad_i64: wp.array2d(dtype=wp.int64), inv_scale: wp.float64,
                     grad: wp.array2d(dtype=wp.float32)):
    c, a = wp.tid()
    grad[c, a] = wp.float32(wp.float64(grad_i64[c, a]) * inv_scale)


@wp.kernel
def _fk_mingrad(oMi: wp.array2d(dtype=wp.transform), jpl: wp.array(dtype=wp.transform),
                parent: wp.array(dtype=wp.int32), axis: wp.array(dtype=wp.vec3), jtype: wp.array(dtype=wp.int32),
                qidx: wp.array(dtype=wp.int32), pt_joint: wp.array(dtype=wp.vec3), pt_parent: wp.array(dtype=wp.int32),
                vol: wp.uint64, inv_vs: float, margin: float,
                min_sd: wp.array(dtype=wp.float32), grad: wp.array2d(dtype=wp.float32)):
    # Fused min-sd analytic gradient (same cost as finite differences, hence identical convergence, but fused):
    # pass 2 after _fk_collide has produced min_sd[c]; only the nearest point (sd ~= min_sd) and only when colliding
    # contributes to d(-min_sd)/dq.
    c, k = wp.tid()
    L = pt_parent[k]
    p = wp.transform_point(oMi[c, L], pt_joint[k])
    uvw = wp.volume_world_to_index(vol, p)
    g3 = wp.vec3(0.0, 0.0, 0.0)
    sd = wp.volume_sample_grad_f(vol, uvw, wp.Volume.LINEAR, g3)
    if min_sd[c] < margin and sd <= min_sd[c] + 1.0e-5:         # nearest point and colliding (min-sd gradient)
        gsd = g3 * inv_vs
        x = L
        while x > 0:
            a = qidx[x]
            if a >= 0:
                Fp = oMi[c, parent[x]]
                fq = wp.transform_get_rotation(Fp); ft = wp.transform_get_translation(Fp)
                z = wp.quat_rotate(fq, wp.quat_rotate(wp.transform_get_rotation(jpl[x]), axis[x]))
                dp = z
                if jtype[x] == 1:
                    o = ft + wp.quat_rotate(fq, wp.transform_get_translation(jpl[x]))
                    dp = wp.cross(z, p - o)
                wp.atomic_add(grad, c, a, -wp.dot(gsd, dp))
            x = parent[x]


class WarpBatchFK:
    """Batched GPU FK from a kinematics spec (npz). fk_points(Q) -> (B,M,3) world points. numpy in and out."""
    def __init__(self, spec, device=None):
        if isinstance(spec, str):
            spec = dict(np.load(spec))
        if device is None:
            device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
        self.device = device
        self.nj = int(spec["njoints"]); self.nq = int(spec["nq"]); self.M = len(spec["pt_joint"])
        jpl = np.concatenate([np.asarray(spec["jpl_t"], np.float32),
                              np.asarray(spec["jpl_q"], np.float32)], axis=1)   # (nj,7)=[p(3),q xyzw(4)]
        with wp.ScopedDevice(device):
            self.parent = wp.array(np.asarray(spec["parent"], np.int32), dtype=wp.int32)
            self.jpl = wp.array(jpl, dtype=wp.transform)
            self.axis = wp.array(np.asarray(spec["axis"], np.float32), dtype=wp.vec3)
            self.jtype = wp.array(np.asarray(spec["jtype"], np.int32), dtype=wp.int32)
            self.qidx = wp.array(np.asarray(spec["qidx"], np.int32), dtype=wp.int32)
            self.pt_joint = wp.array(np.asarray(spec["pt_joint"], np.float32), dtype=wp.vec3)
            self.pt_parent = wp.array(np.asarray(spec["pt_parent"], np.int32), dtype=wp.int32)
        # numpy spec for the analytic collision gradient (geometric Jacobian, host side)
        self._parent = np.asarray(spec["parent"], int); self._jpl_t = np.asarray(spec["jpl_t"], float)
        self._jpl_q = np.asarray(spec["jpl_q"], float); self._axis = np.asarray(spec["axis"], float)
        self._jtype = np.asarray(spec["jtype"], int); self._qidx = np.asarray(spec["qidx"], int)
        self._pt_parent_np = np.asarray(spec["pt_parent"], int)
        self._anc_mat = np.zeros((self.nj, self.nj), bool)    # anc_mat[L,j] = joint j moves a point on link L (j on the path root -> L)
        for L in range(self.nj):
            x = L
            while x > 0:
                self._anc_mat[L, x] = True; x = self._parent[x]
        self.has_ee = "ee_parent" in spec                    # EE pose support (an older spec without ee still works for fk_points)
        if self.has_ee:
            self.ee_parent = int(spec["ee_parent"])
            self.ee_pl = wp.transform(tuple(np.asarray(spec["ee_t"], np.float32)),
                                      tuple(np.asarray(spec["ee_q"], np.float32)))

    def ee_pose(self, Q):
        """Q (B,nq) -> EE world pose (B,7) = [pos(3), quat xyzw(4)]. For cartesian-goal MPPI (pose cost, no separate IK)."""
        if not self.has_ee:
            raise RuntimeError("spec has no ee_parent — regenerate the kinematics spec")
        Q = np.atleast_2d(np.asarray(Q, np.float32)); B = len(Q)
        _assert_finite_q(Q, "ee_pose")
        with wp.ScopedDevice(self.device):
            Qw = wp.array(Q, dtype=wp.float32)
            oMi = wp.empty((B, self.nj), dtype=wp.transform)
            out = wp.empty(B, dtype=wp.transform)
            wp.launch(_fk_joints, dim=B, inputs=[self.parent, self.jpl, self.axis, self.jtype, self.qidx, Qw, self.nj],
                      outputs=[oMi])
            wp.launch(_fk_ee, dim=B, inputs=[oMi, self.ee_parent, self.ee_pl], outputs=[out])
            return out.numpy().astype(float)                 # (B,7) [px,py,pz,qx,qy,qz,qw]

    def fk_frames(self, Q):
        """Q (B,nq) -> joint world transforms oMi (B, nj, 7) = [pos(3), quat xyzw(4)]. Needed by the analytic collision
        gradient (the geometric Jacobian dp/dq needs joint world axes and origins). Same _fk_joints kernel as fk_points."""
        Q = np.atleast_2d(np.asarray(Q, np.float32)); B = len(Q)
        _assert_finite_q(Q, "fk_frames")
        with wp.ScopedDevice(self.device):
            Qw = wp.array(Q, dtype=wp.float32)
            oMi = wp.empty((B, self.nj), dtype=wp.transform)
            wp.launch(_fk_joints, dim=B, inputs=[self.parent, self.jpl, self.axis, self.jtype,
                                                 self.qidx, Qw, self.nj], outputs=[oMi])
            return oMi.numpy().astype(float)                  # (B, nj, 7)

    def fk_points(self, Q, flat=False):
        """Q (B,nq) → world-punkter (B,M,3); flat=True → (B*M,3) (rakt in i WorldModel.sd_batch)."""
        Q = np.atleast_2d(np.asarray(Q, np.float32)); B = len(Q)
        _assert_finite_q(Q, "fk_points")
        with wp.ScopedDevice(self.device):
            Qw = wp.array(Q, dtype=wp.float32)
            oMi = wp.empty((B, self.nj), dtype=wp.transform)
            out = wp.empty((B, self.M), dtype=wp.vec3)
            wp.launch(_fk_joints, dim=B, inputs=[self.parent, self.jpl, self.axis, self.jtype,
                                                 self.qidx, Qw, self.nj], outputs=[oMi])
            wp.launch(_fk_points, dim=(B, self.M), inputs=[oMi, self.pt_joint, self.pt_parent], outputs=[out])
            W = out.numpy().astype(float)
        return W.reshape(B * self.M, 3) if flat else W

    def alloc(self, B):
        """Pre-allocated GPU buffers for a fixed batch size B (for CUDA graph capture / a full-GPU loop)."""
        with wp.ScopedDevice(self.device):
            return dict(Q=wp.zeros((B, self.nq), dtype=wp.float32), oMi=wp.empty((B, self.nj), dtype=wp.transform),
                        min_sd=wp.full(B, 1.0e10, dtype=wp.float32))

    def collide_launch(self, buf, world):
        """GPU-only two-kernel sequence (FK -> fused sampling + min) on pre-allocated buffers, with no host sync, so it
        can be captured in a CUDA graph. buf['Q'] must be filled; the result lands in buf['min_sd']."""
        B = buf["Q"].shape[0]
        buf["min_sd"].fill_(1.0e10)
        wp.launch(_fk_joints, dim=B, inputs=[self.parent, self.jpl, self.axis, self.jtype, self.qidx, buf["Q"], self.nj],
                  outputs=[buf["oMi"]])
        wp.launch(_fk_collide, dim=(B, self.M), inputs=[buf["oMi"], self.pt_joint, self.pt_parent, world._vid],
                  outputs=[buf["min_sd"]])

    def collide_batch(self, Q, world):
        """Fused GPU pipeline: FK + SDF sampling + per-configuration min-sd in one kernel chain (no host round-trip).
        world = WarpVolumeWorld on the same device. -> min_sd (B,): nearest obstacle distance per configuration; free = min_sd > 0."""
        Q = np.atleast_2d(np.asarray(Q, np.float32)); B = len(Q)
        _assert_finite_q(Q, "collide_batch")
        with wp.ScopedDevice(self.device):
            Qw = wp.array(Q, dtype=wp.float32)
            oMi = wp.empty((B, self.nj), dtype=wp.transform)
            min_sd = wp.full(B, 1.0e10, dtype=wp.float32)
            wp.launch(_fk_joints, dim=B, inputs=[self.parent, self.jpl, self.axis, self.jtype,
                                                 self.qidx, Qw, self.nj], outputs=[oMi])
            wp.launch(_fk_collide, dim=(B, self.M),
                      inputs=[oMi, self.pt_joint, self.pt_parent, world._vid], outputs=[min_sd])
            return min_sd.numpy().astype(float)

    def collision_grad_minsd(self, Q, world, margin):
        """Analytic collision gradient d(-min_sd)/dq per configuration, replacing an nq+1 finite-difference FK sweep with
        an exact expression and fewer FK passes. min_sd is at the nearest robot point p*; d(min_sd)/dq = grad sd(p*) . J_p*(q),
        with J from the geometric Jacobian (joint world axis x (p* - joint origin) for revolute, the axis for prismatic).
        `world` must provide grad_batch.
        -> (grad (B,nq) = -d min_sd/dq where colliding/near (margin - base > 0), 0 otherwise; base (B,) = min_sd).
        Validated at cosine ~= 1.0 against finite differences."""
        Q = np.atleast_2d(np.asarray(Q, float)); B = len(Q)
        W = self.fk_points(Q)                                 # (B,M,3)
        oMi = self.fk_frames(Q)                               # (B,nj,7)
        sd = world.sd_batch(W.reshape(B * self.M, 3)).reshape(B, self.M)
        # A masked argmin (masking a corrupt point to +inf and taking the argmin of the rest) would silently
        # substitute the second-nearest point when the corrupt point is the true nearest, i.e. an optimistic
        # fabrication. Since a masked point cannot be ruled out as the true minimum, any non-finite entry in the row
        # fails the whole row closed to -inf; the masked argmin point is only used for the direction.
        sd_safe = np.where(np.isfinite(sd), sd, np.inf)
        istar = sd_safe.argmin(1); ar = np.arange(B)
        row_bad = ~np.isfinite(sd).all(axis=1)
        base = np.where(row_bad, -np.inf, sd[ar, istar])
        pstar = W[ar, istar]                                  # (B,3) nearest point
        gsd = world.grad_batch(pstar)                         # (B,3) grad sd(p*) in world coordinates
        viol = (margin - base) > 0
        Lstar = self._pt_parent_np[istar]                     # (B,) link of the nearest point
        g = np.zeros((B, self.nq))
        for j in range(1, self.nj):
            a = self._qidx[j]
            if a < 0:
                continue
            m = viol & self._anc_mat[Lstar, j]                # configurations where joint j moves p* and is colliding
            if not m.any():
                continue
            pj = self._parent[j]; qp = oMi[m, pj, 3:]; tp = oMi[m, pj, :3]   # frame_parent
            o = tp + _qrot(qp, self._jpl_t[j])                # frame_j-origo (oMi[parent]∘jpl)
            z = _qrot(qp, _qrot(self._jpl_q[j], self._axis[j]))             # world axis of frame j
            dp = np.cross(z, pstar[m] - o) if self._jtype[j] == 1 else z    # ∂p*/∂q_a
            g[m, a] = -(gsd[m] * dp).sum(1)                   # ∂(-min_sd)/∂q_a
        return g, base

    def collision_grad_fused(self, Q, world, margin, mode="minsd"):
        """Fused GPU analytic collision gradient: the kernel chain stays on the GPU with no host round-trip.
        mode='minsd' (default): d(-min_sd)/dq (same cost as finite differences, hence identical convergence, but fused;
        three kernels: joints -> collide(min_sd) -> min-gradient). mode='sum': sum of relu terms over colliding points
        (faster per step but a different cost function, hence different convergence; one kernel).
        world = WarpVolumeWorld. -> (grad (B,nq), min_sd (B,))."""
        Q = np.atleast_2d(np.asarray(Q, np.float32)); B = len(Q)
        _assert_finite_q(Q, "collision_grad_fused")
        with wp.ScopedDevice(self.device):
            Qw = wp.array(Q, dtype=wp.float32)
            oMi = wp.empty((B, self.nj), dtype=wp.transform)
            grad = wp.zeros((B, self.nq), dtype=wp.float32)
            min_sd = wp.full(B, 1.0e10, dtype=wp.float32)
            inv_vs = 1.0 / world.voxel_size
            wp.launch(_fk_joints, dim=B, inputs=[self.parent, self.jpl, self.axis, self.jtype, self.qidx, Qw, self.nj],
                      outputs=[oMi])
            if mode == "sum":
                # BRYGGA 4: BIT-deterministisk fixpunkts-int64-ackumulering (float32-atomic_add gav ~0.73mm jitter)
                GRAD_SCALE = 1.0e9
                grad_i64 = wp.zeros((B, self.nq), dtype=wp.int64)
                wp.launch(_fk_collide_grad_i64, dim=(B, self.M),
                          inputs=[oMi, self.jpl, self.parent, self.axis, self.jtype, self.qidx, self.pt_joint,
                                  self.pt_parent, world._vid, inv_vs, float(margin), wp.float64(GRAD_SCALE)],
                          outputs=[grad_i64, min_sd])
                wp.launch(_i64_to_f32_grad, dim=(B, self.nq),
                          inputs=[grad_i64, wp.float64(1.0 / GRAD_SCALE)], outputs=[grad])
            else:                                                # min-sd (default): pass 1 min_sd, pass 2 nearest-point gradient
                wp.launch(_fk_collide, dim=(B, self.M),
                          inputs=[oMi, self.pt_joint, self.pt_parent, world._vid], outputs=[min_sd])
                wp.launch(_fk_mingrad, dim=(B, self.M),
                          inputs=[oMi, self.jpl, self.parent, self.axis, self.jtype, self.qidx, self.pt_joint,
                                  self.pt_parent, world._vid, inv_vs, float(margin), min_sd], outputs=[grad])
            return grad.numpy().astype(float), min_sd.numpy().astype(float)
