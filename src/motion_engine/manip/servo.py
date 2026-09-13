"""Servo loop for the manipulation cells (torch + warp kernels): joint targets, gains, integration
and the per-step readout the cells record.
"""
from __future__ import annotations

import time

import numpy as np
import torch

import newton
import warp as wp


class JTServo:
    """Task-space J^T servo across W replicated worlds.

    Owns the simulation stepping (apply_tau runs SUB substeps with collide + solver.step
    and swaps s0/s1), so `state["s0"]` is always "now", the same contract as
    legacy-skriptets state-dict.
    """

    def __init__(self, *, model, solver, view, control, state, dyn,
                 W: int, sub: int, dt: float, profile, nj: int,
                 device: str = "cuda"):
        self.model = model
        self.solver = solver
        self.view = view
        self.control = control
        self.state = state                  # {"s0": State, "s1": State}
        self.dyn = dyn                      # TorchDynamics (FK + gravmodell)
        self.W = W
        self.SUB = sub
        self.dt = dt
        self.dev = device
        # Effektiva konstanter ur profilen; HOME/TAU_MAX trunkeras [:NJ]
        # truncated with [:NJ] on the array literals.
        self.NJ = nj
        self.HOME = profile.home[:nj]
        self.TAU_MAX = profile.tau_max[:nj]
        # Same class as the FMAX/MMAX guard below:
        # a negative TAU_MAX element turns `np.clip(tau, -TAU_MAX, TAU_MAX)` into a min>max clamp that PINS
        # that joint at -|TAU_MAX| (full reversed torque) whatever the commanded tau is. Fail closed at
        # konfig-laddning.
        if not (np.all(np.isfinite(self.TAU_MAX)) and np.all(self.TAU_MAX > 0)):
            raise ValueError(f"profile.tau_max must be all finite and positive, got {self.TAU_MAX!r}")
        self.KP_P, self.KD_P, self.KI_P = profile.kp_p, profile.kd_p, profile.ki_p
        self.KP_O, self.KD_O = profile.kp_o, profile.kd_o
        self.FMAX, self.MMAX = profile.fmax, profile.mmax
        # A degenerate (negative / zero / NaN)
        # FMAX/MMAX makes `F * min(1, FMAX/Fn)` REVERSE the sign of the commanded force/moment instead of
        # clamping it (measured: FMAX=-5 on F=[10,0,0] -> F=[-5,0,0], a reversed command straight into
        # apply_tau). Unlike effortLimit=+inf (a legitimately unbounded joint), a force/moment
        # safety limit has no honest degenerate reading, so fail closed at configuration load.
        if not (np.isfinite(self.FMAX) and self.FMAX > 0):
            raise ValueError(f"profile.fmax must be finite and positive, got {self.FMAX!r}")
        if not (np.isfinite(self.MMAX) and self.MMAX > 0):
            raise ValueError(f"profile.mmax must be finite and positive, got {self.MMAX!r}")
        self.IERR_CLIP = profile.ierr_clip
        self.ierr = np.zeros((W, 3))
        self.APPROACH = np.tile([0.0, 0.0, -1.0], (W, 1))
        self.fctrl = view.get_attribute("joint_f", control)
        self.cpipe = newton.CollisionPipeline(model)
        # optional hooks, default OFF:
        self.sense_noise = None     # callable(q, qd) -> (q, qd)
        self.timing = None          # dict(t_step=, n_sub=) enables timing measurement

    def read_state(self):
        lt = self.view.get_link_transforms(self.state["s0"]).numpy()
        ee = lt[:, 0, -1, :3]; eq = lt[:, 0, -1, 3:7]
        q = self.view.get_attribute("joint_q", self.state["s0"]).numpy()[:, 0, :]
        qd = self.view.get_attribute("joint_qd", self.state["s0"]).numpy()[:, 0, :]
        if self.sense_noise is not None:
            q, qd = self.sense_noise(q, qd)
        ez = np.column_stack([2 * (eq[:, 0] * eq[:, 2] + eq[:, 1] * eq[:, 3]),
                              2 * (eq[:, 1] * eq[:, 2] - eq[:, 0] * eq[:, 3]),
                              1 - 2 * (eq[:, 0] ** 2 + eq[:, 1] ** 2)])
        return ee, ez, q, qd

    def apply_tau(self, tau):
        c = self.fctrl.numpy(); c[:, 0, :] = tau; self.fctrl.assign(c)
        self.view.set_attribute("joint_f", self.control, self.fctrl)
        if self.timing is not None:
            wp.synchronize(); t0 = time.perf_counter()
        for _ in range(self.SUB):
            self.state["s0"].clear_forces()
            ct = self.model.collide(self.state["s0"], collision_pipeline=self.cpipe)
            self.solver.step(self.state["s0"], self.state["s1"], self.control, ct, self.dt)
            self.state["s0"], self.state["s1"] = self.state["s1"], self.state["s0"]
        if self.timing is not None:
            wp.synchronize(); self.timing["t_step"] += time.perf_counter() - t0
            self.timing["n_sub"] += self.SUB

    def servo_step(self, goal, *, sensed=None, ee_force=None, tau_post=None):
        ee, ez, q, qd = sensed if sensed is not None else self.read_state()
        qt = torch.as_tensor(q, dtype=torch.float32, device=self.dev)
        Rw, p, z, _ = self.dyn._fk(qt)
        grav = self.dyn(qt).cpu().numpy()
        p = p.cpu().numpy(); zax = z.cpu().numpy()
        rvec = ee[:, None, :] - p
        Jp = np.cross(zax, rvec); Jr = zax
        perr = goal - ee
        self.ierr = np.clip(self.ierr + perr * (self.SUB * self.dt),
                            -self.IERR_CLIP, self.IERR_CLIP)
        v_ee = np.einsum("wjd,wj->wd", Jp, qd)
        F = self.KP_P * perr + self.KI_P * self.ierr - self.KD_P * v_ee
        Fn = np.linalg.norm(F, axis=1, keepdims=True)
        F = F * np.minimum(1.0, self.FMAX / np.maximum(Fn, 1e-9))
        e_rot = np.cross(ez, self.APPROACH)
        w_ee = np.einsum("wjd,wj->wd", Jr, qd)
        M = self.KP_O * e_rot - self.KD_O * w_ee
        Mn = np.linalg.norm(M, axis=1, keepdims=True)
        M = M * np.minimum(1.0, self.MMAX / np.maximum(Mn, 1e-9))
        tau = (np.einsum("wjd,wd->wj", Jp, F)
               + np.einsum("wjd,wd->wj", Jr, M) + grav)
        if ee_force is not None:
            # payload compensation after the grasp, added BEFORE the clamp,
            # ordagrant legacy-uttryck (`tau += einsum(Jp, kraft)`)
            tau += np.einsum("wjd,wd->wj", Jp, ee_force)
        tau = np.clip(tau, -self.TAU_MAX, self.TAU_MAX)
        if tau_post is not None:
            tau = tau_post(tau)     # contact_pick: momentripple EFTER klampen
        # H208 (portad H->I av S4 2026-08-22, BRYGGA 4): ett NaN i F/M (ur korrupt sensed/goal/ee_force
        # or a caller-supplied tau_post) propagates through perr -> F/M -> tau and SURVIVES
        # np.clip(tau,-TAU_MAX,TAU_MAX) unchanged (clip does not sanitise NaN: no comparison branch is True
        # for NaN). A NaN control torque must never be silently handed to any physics solver. Guard at the
        # last point before apply_tau (after ee_force/clip/tau_post) so every route to the solver is covered.
        if not np.all(np.isfinite(tau)):
            raise ValueError("servo_step: computed tau is non-finite -- refusing to send a NaN/inf control "
                             "torque to the physics solver (a corrupted goal/sensed/ee_force/tau_post input "
                             "would otherwise propagate silently through np.clip, which does not sanitize NaN)")
        self.apply_tau(tau)
        return ee
