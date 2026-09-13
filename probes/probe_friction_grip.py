#!/usr/bin/env python
"""Friction clamp grip probe: a two-finger gripper holding a cube through a lift, with friction contacts only.

Question: can a two-finger gripper (prismatic joints on wrist_3, box plates) clamp a 5 cm cube
(0.5 kg +/- 30%) and hold it through a 0.15 m lift on a randomised plant, using only friction contacts?

Setup: 8 worlds, UR10e, randomised plant (robot mass +/- 15%, joint friction 1-6 Nm, cube mass +/- 30%),
constant clamp force F_GRIP[w] = [3,5,8,8,12,12,16,20] N in effort mode on the prismatic finger joints,
then a +0.15 m lift; measure whether the cube follows (body_q z) and how far it slips.

Pre-registered criteria:
  P1 (structure): the articulation view exposes 8 DOF (6 arm + 2 fingers).
  P2 (grip):      every world with F >= 8 N lifts the cube >= 0.12 m with slip <= 15 mm.
  P3 (physics):   at least one world with F < 8 N slips more than the held high-force worlds, so the force
                  dependence is real and not an artefact.

Exit: 0 = P1+P2, 1 = partial (data in the JSON), 2 = blocked (build or API error).
Output: reports/friction_grip_probe.json. Requires the Newton simulator, torch and warp with CUDA.
"""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

W = 8
PLANT_SEED = int(os.environ.get("PLANT_SEED", "777"))
OBJ = np.array([0.6, 0.0, 0.375])       # cube centre (pedestal top 0.35 + 0.025)
CUBE_H = 0.025
LIFT = 0.15
F_GRIP = np.array([3.0, 5.0, 8.0, 8.0, 12.0, 12.0, 16.0, 20.0])
# 30 N produced a contact blow-up; 20 N is stable, so 20 N is the ceiling
MU_FINGER = float(os.environ.get("MU_FINGER", "1.0"))

# gripper geometry (wrist_3 local frame; local +z points DOWN once the servo has aligned ez with the
# approach direction -z)
F_OFF = 0.05            # y offset of the plate centre from the tool axis at q=0
F_DROP = 0.12           # plate centre 12 cm "down" along local +z
PHX, PHY, PHZ = 0.02, 0.006, 0.022   # plate: 4 cm wide, 1.2 cm thick, 4.4 cm tall
Z_GRASP = OBJ[2] + 0.001 + F_DROP    # plattcentrum ~kubcentrum (run1: +7mm gav
                                     # gravitations-pitchmoment → 95° tilt i v4;
                                     # plattbotten 0.354 > piedestaltopp 0.35)


def build():
    import newton
    rob = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(rob)
    urdf = ROOT / "assets/robots/ur_description/ur10e.urdf"
    rob.add_urdf(str(urdf), collapse_fixed_joints=True, enable_self_collisions=False)
    nb_arm = len(rob.body_mass)          # 6, wrist_3 = nb_arm-1
    nd_arm = len(rob.joint_target_ke)    # 6
    n_jor = len(rob.joint_type)          # URDF joints (for the articulation check)

    # disable the robot's URDF collision geometry, BEFORE adding the finger shapes so the plates keep
    # collision enabled
    for i in range(len(rob.shape_flags)):
        rob.shape_flags[i] = int(rob.shape_flags[i]) & ~int(newton.ShapeFlags.COLLIDE_SHAPES)

    # === two-finger gripper: prismatic joints on wrist_3 ===
    fcfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=MU_FINGER)
    fingers = []
    fjoints = []
    for sgn, name in ((+1.0, "finger_l"), (-1.0, "finger_r")):
        # OBS: add_link, INTE add_body — add_body autoskapar FREE-joint +
        # EGEN artikulation (verifierat: ValueError "most recent is #2")
        fb = rob.add_link(label=name)
        fj = rob.add_joint_prismatic(
            parent=nb_arm - 1, child=fb,
            parent_xform=((0.0, sgn * F_OFF, F_DROP), (0.0, 0.0, 0.0, 1.0)),
            axis=(0.0, 1.0, 0.0),
            limit_lower=-0.040, limit_upper=0.040,
            target_ke=0.0, target_kd=0.0, armature=0.1,
            effort_limit=200.0, friction=0.0,
            actuator_mode=newton.JointTargetMode.EFFORT,
            label=f"{name}_slide")
        rob.add_shape_box(fb, hx=PHX, hy=PHY, hz=PHZ, cfg=fcfg, label=f"{name}_pad")
        fingers.append(fb)
        fjoints.append(fj)
    # KRITISKT: assimilera fingerlederna i URDF-artikulationen (annars
    # otherwise finalize raises an orphan error and ArticulationView does not see them)
    rob._finalize_imported_articulation(fjoints, parent_body=nb_arm - 1)
    assert rob.joint_articulation[fjoints[0]] == rob.joint_articulation[0], \
        "fingerled hamnade inte i URDF-artikulationen"

    # arm + fingers: effort mode, zero cascade stiffness
    for i in range(len(rob.joint_target_ke)):
        rob.joint_target_ke[i] = 0.0; rob.joint_target_kd[i] = 0.0
        rob.joint_target_mode[i] = int(newton.JointTargetMode.EFFORT)
    for i in range(nd_arm):
        rob.joint_armature[i] = 0.5
    for i in range(len(rob.joint_effort_limit)):
        if rob.joint_effort_limit[i] <= 0:
            rob.joint_effort_limit[i] = 1e4

    # piedestal + kub (µ default 1.0 — parfriktion = max(mu_i) ⇒ 1.0)
    rob.add_shape_box(-1, xform=((OBJ[0], OBJ[1], 0.175), (0, 0, 0, 1.0)),
                      hx=0.06, hy=0.06, hz=0.175, label="pedestal")
    cube_density = 0.5 / (2 * CUBE_H) ** 3
    cube = rob.add_body(xform=((OBJ[0], OBJ[1], OBJ[2] + 0.0005), (0, 0, 0, 1.0)),
                        label="cube")
    rob.add_shape_box(cube, hx=CUBE_H, hy=CUBE_H, hz=CUBE_H,
                      cfg=newton.ModelBuilder.ShapeConfig(density=cube_density),
                      label="cube_shape")

    nb_t = len(rob.body_mass)            # 6 arm + 2 finger + 1 kub = 9
    b = newton.ModelBuilder(); b.replicate(rob, W, spacing=(0, 0, 0))
    # R-PLANT
    rr = np.random.default_rng(PLANT_SEED)
    fm = rr.uniform(0.85, 1.15, nb_arm); fr = rr.uniform(1.0, 6.0, nd_arm)
    fcube = rr.uniform(0.7, 1.3, W)
    nd_t = nd_arm + 2
    cube_local = nb_arm + 2              # kubens lokala body-index
    for w in range(W):
        for k in range(nb_arm):
            b.body_mass[w * nb_t + k] *= fm[k]
        b.body_mass[w * nb_t + cube_local] *= fcube[w]
        for k in range(nd_arm):
            b.joint_friction[w * nd_t + k] = float(fr[k])
    model = b.finalize()
    # njmax=300: a smaller value overflows the constraint buffer and silently DROPS contact constraints
    # (the cube is released despite 20-27 N of measured clamp force).
    # cone=elliptic + impratio=10: with the default pyramidal cone the slip was force-INDEPENDENT (~40 mm at
    # 8/12/20 N alike) and heavy cubes crept down during the lift, the known soft-friction creep; the
    # documented fix for gripping is an elliptic cone with a raised impratio.
    solver = newton.solvers.SolverMuJoCo(model, disable_contacts=False,
                                         nconmax=4096, njmax=300,
                                         cone="elliptic", impratio=10.0,
                                         use_mujoco_contacts=False)
    return model, solver, nb_arm, nd_arm, nb_t, cube_local, fcube


def main():
    import warp as wp
    import torch
    import newton
    from newton.selection import ArticulationView
    from newton.sensors import SensorContact
    from newton import Contacts
    from motion_engine.torch_dynamics import TorchDynamics
    wp.init()
    dev = "cuda"

    try:
        model, solver, nb_arm, nd_arm, nb_t, cube_local, fcube = build()
    except Exception as e:
        print(f"BLOCKED (build): {type(e).__name__}: {e}")
        return 2

    view = ArticulationView(model, "*ur10e*",
                            exclude_joint_types=[newton.JointType.FREE])
    NJ = view.joint_dof_count
    # === P1: STRUKTUR ===
    print(f"P1 structure: view dofs={NJ} (required 8), links={view.link_count}")
    p1 = NJ == nd_arm + 2
    if not p1:
        print("BLOCKED: the fingers did not enter the articulation view")
        return 2

    sensorL = SensorContact(model, sensing_obj_shapes="*finger_l_pad*",
                            counterpart_shapes="*cube_shape*")
    sensorR = SensorContact(model, sensing_obj_shapes="*finger_r_pad*",
                            counterpart_shapes="*cube_shape*")
    contacts = Contacts(solver.get_max_contact_count(), 0,
                        requested_attributes=model.get_requested_contact_attributes())

    s0, s1 = model.state(), model.state()
    control = model.control()
    SUB, dt = 4, 1.0 / 30.0 / 4
    fctrl = view.get_attribute("joint_f", control)
    HOME = np.zeros(NJ); HOME[:6] = [0.0, -1.2, 1.5, -1.8, -1.57, 0.0]
    TAU_MAX = np.concatenate([[150.0, 150.0, 150.0, 28.0, 28.0, 28.0],
                              [200.0, 200.0]])

    rng = np.random.default_rng(7)
    qa = view.get_attribute("joint_q", s0); arr = qa.numpy()
    arr[:, 0, :6] = HOME[:6] + rng.normal(0, 0.05, (W, 6))
    arr[:, 0, 6:] = 0.0
    qa.assign(arr); view.set_attribute("joint_q", s0, qa)
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    state = {"s0": s0, "s1": s1}

    gp = ROOT / "data/ur10e_gravparams.npz"
    if not gp.exists():
        from motion_engine.torch_gravity import extract_gravparams
        extract_gravparams(ROOT / "assets/robots/ur_description/ur10e.urdf", gp)
    dyn = TorchDynamics(gp, device=dev)

    cpipe = newton.CollisionPipeline(model)
    t_step = [0.0]; n_sub = [0]

    def apply_tau(tau8):
        c = fctrl.numpy(); c[:, 0, :] = tau8; fctrl.assign(c)
        view.set_attribute("joint_f", control, fctrl)
        wp.synchronize(); t0 = time.perf_counter()
        for _ in range(SUB):
            state["s0"].clear_forces()
            ct = model.collide(state["s0"], collision_pipeline=cpipe)
            solver.step(state["s0"], state["s1"], control, ct, dt)
            state["s0"], state["s1"] = state["s1"], state["s0"]
        wp.synchronize(); t_step[0] += time.perf_counter() - t0
        n_sub[0] += SUB

    def body_x(idx_local):
        bq = state["s0"].body_q.numpy()
        return bq[np.arange(W) * nb_t + idx_local]

    def clamp_force():
        solver.update_contacts(contacts, state["s0"])
        sensorL.update(state["s0"], contacts)
        sensorR.update(state["s0"], contacts)
        fL = np.linalg.norm(sensorL.force_matrix.numpy().reshape(W, 3), axis=1)
        fR = np.linalg.norm(sensorR.force_matrix.numpy().reshape(W, 3), axis=1)
        return fL, fR

    def read_state():
        lt = view.get_link_transforms(state["s0"]).numpy()
        ee = lt[:, 0, nb_arm - 1, :3]; eq = lt[:, 0, nb_arm - 1, 3:7]
        q = view.get_attribute("joint_q", state["s0"]).numpy()[:, 0, :]
        qd = view.get_attribute("joint_qd", state["s0"]).numpy()[:, 0, :]
        ez = np.column_stack([2 * (eq[:, 0] * eq[:, 2] + eq[:, 1] * eq[:, 3]),
                              2 * (eq[:, 1] * eq[:, 2] - eq[:, 0] * eq[:, 3]),
                              1 - 2 * (eq[:, 0] ** 2 + eq[:, 1] ** 2)])
        return ee, ez, q, qd

    cube_z0 = body_x(cube_local)[:, 2].copy()
    cube_xy0 = body_x(cube_local)[:, :2].copy()

    KP_P, KD_P, KI_P = 900.0, 90.0, 600.0
    KP_O, KD_O = 18.0, 1.8
    FMAX, MMAX = 60.0, 15.0
    APPROACH = np.tile([0.0, 0.0, -1.0], (W, 1))
    ierr = np.zeros((W, 3))

    N_WP, N_DESC, N_CLOSE, N_LIFT, N_SET = 110, 240, 70, 120, 80
    Z_WP = 0.80
    goal_z = np.full(W, Z_WP)
    grip_on = np.zeros(W, bool)
    grip_step = np.full(W, -1)
    fingerz_at_grip = np.zeros(W)
    cubez_at_grip = np.zeros(W)
    clampN_lift = np.zeros((W, 2))
    pre_drift = np.full(W, -1.0)
    lift_started = np.zeros(W, bool)

    for k in range(N_WP + N_DESC + N_CLOSE + N_LIFT + N_SET):
        ee, ez, q, qd = read_state()
        xy_ok = np.linalg.norm(ee[:, :2] - OBJ[None, :2], axis=1) < 0.012
        z_ok = np.abs(ee[:, 2] - goal_z) < 0.04
        # grindad descend mot Z_GRASP
        active = ~grip_on & (k >= N_WP) & xy_ok & z_ok
        goal_z[active] = np.maximum(Z_GRASP, goal_z[active] - 0.0028)
        # grip latch: once the EE is in place, apply a constant clamp force
        new_grip = (~grip_on) & (k >= N_WP) & xy_ok \
                   & (np.abs(ee[:, 2] - Z_GRASP) < 0.008) \
                   & (np.linalg.norm(qd[:, :6], axis=1) < 0.6)
        if new_grip.any():
            xf = 0.5 * (body_x(nb_arm)[:, 2] + body_x(nb_arm + 1)[:, 2])
            xc = body_x(cube_local)
            for wdx in np.where(new_grip)[0]:
                grip_on[wdx] = True; grip_step[wdx] = k
                pre_drift[wdx] = np.linalg.norm(xc[wdx, :2] - cube_xy0[wdx]) * 1e3
            print(f"  k={k}: GRIP worlds {list(np.where(new_grip)[0])}")
        # lift after the closing window
        lw = grip_on & (k - grip_step > N_CLOSE)
        if lw.any():
            first = lw & ~lift_started
            if first.any():
                fz = 0.5 * (body_x(nb_arm)[:, 2] + body_x(nb_arm + 1)[:, 2])
                cz = body_x(cube_local)[:, 2]
                fL, fR = clamp_force()
                for wdx in np.where(first)[0]:
                    fingerz_at_grip[wdx] = fz[wdx]
                    cubez_at_grip[wdx] = cz[wdx]
                    clampN_lift[wdx] = (fL[wdx], fR[wdx])
                    lift_started[wdx] = True
                print(f"  k={k}: LIFT worlds {list(np.where(first)[0])}, "
                      f"clamp force L/R {np.round(fL[first],1)}/{np.round(fR[first],1)} N")
            goal_z[lw] = np.minimum(Z_GRASP + LIFT, goal_z[lw] + 0.004)

        # === arm-servo (J^T, 6 dof) ===
        qt = torch.as_tensor(q[:, :6], dtype=torch.float32, device=dev)
        Rw, p, z, _ = dyn._fk(qt)
        grav = dyn(qt).cpu().numpy()
        p = p.cpu().numpy(); zax = z.cpu().numpy()
        rvec = ee[:, None, :] - p
        Jp = np.cross(zax, rvec); Jr = zax
        goal = np.column_stack([np.tile(OBJ[:2], (W, 1)), goal_z])
        perr = goal - ee
        ierr = np.clip(ierr + perr * (SUB * dt), -0.05, 0.05)
        v_ee = np.einsum("wjd,wj->wd", Jp, qd[:, :6])
        F = KP_P * perr + KI_P * ierr - KD_P * v_ee
        Fn = np.linalg.norm(F, axis=1, keepdims=True)
        F = F * np.minimum(1.0, FMAX / np.maximum(Fn, 1e-9))
        e_rot = np.cross(ez, APPROACH)
        w_ee = np.einsum("wjd,wj->wd", Jr, qd[:, :6])
        M = KP_O * e_rot - KD_O * w_ee
        Mn = np.linalg.norm(M, axis=1, keepdims=True)
        M = M * np.minimum(1.0, MMAX / np.maximum(Mn, 1e-9))
        tau6 = (np.einsum("wjd,wd->wj", Jp, F)
                + np.einsum("wjd,wd->wj", Jr, M) + grav)
        tau6 += np.einsum("wjd,wd->wj", Jp,
                          np.where(grip_on[:, None], [0, 0, 0.5 * 9.81], 0.0))
        # === finger effort: open slightly before the grip, then clamp with a ramp ===
        # (a constant 16 N from step 0 gives a ~2 m/s impact and a contact blow-up in 1 of 8 worlds; a ramp of
        # 2 N + 0.5 N per step gives a soft approach and reaches full force in under 40 steps)
        f_now = np.where(grip_on,
                         np.minimum(F_GRIP, 2.0 + 0.5 * np.maximum(k - grip_step, 0)),
                         0.0)
        # finger_l sits at +y and closes towards -y; finger_r sits at -y and closes towards +y
        f_l = np.where(grip_on, -f_now, +1.0)
        f_r = np.where(grip_on, +f_now, -1.0)
        tau8 = np.concatenate([tau6, f_l[:, None], f_r[:, None]], axis=1)
        apply_tau(np.clip(tau8, -TAU_MAX, TAU_MAX))

    # === utfall ===
    xc = body_x(cube_local)
    fz_end = 0.5 * (body_x(nb_arm)[:, 2] + body_x(nb_arm + 1)[:, 2])
    fL_end, fR_end = clamp_force()
    cube_rise = xc[:, 2] - cube_z0
    finger_rise = fz_end - fingerz_at_grip
    slip_mm = (finger_rise - (xc[:, 2] - cubez_at_grip)) * 1e3
    qc = xc[:, 3:7]
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (qc[:, 0]**2 + qc[:, 1]**2), -1, 1)))
    nan_free = bool(np.isfinite(state["s0"].joint_q.numpy()).all())
    held = (cube_rise >= 0.12) & (slip_mm <= 15.0) & lift_started

    print(f"\n  world |  F_grip | cube kg | clamp L/R N (at lift) | cube rise m | slip mm | tilt deg | holds")
    for w in range(W):
        print(f"   {w}   | {F_GRIP[w]:5.1f} N | {0.5*fcube[w]:.2f} | "
              f"{clampN_lift[w,0]:5.1f}/{clampN_lift[w,1]:5.1f} (end {fL_end[w]:.1f}/{fR_end[w]:.1f}) | "
              f"{cube_rise[w]:7.3f} | {slip_mm[w]:6.1f} | {tilt[w]:4.1f} | {bool(held[w])}")
    print(f"  pre-grip drift mm: {np.round(pre_drift, 1)}")
    print(f"  theory: m_max=0.65 kg -> required normal force = 6.38/(2*{MU_FINGER}) = "
          f"{6.38/(2*MU_FINGER):.1f} N per finger")
    print(f"  cost: {n_sub[0]/t_step[0]:.0f} substeps/s at W={W}")

    hi = F_GRIP >= 8.0
    lo = F_GRIP < 8.0
    p2 = bool(held[hi].all()) and nan_free
    held_slip = slip_mm[hi & held]
    p3 = bool((slip_mm[lo].min() > 15.0) and (held[hi].sum() >= 4)
              and (len(held_slip) > 0)
              and (np.median(slip_mm[lo]) > np.median(held_slip)))
    print(f"P1 structure (8 dof in the view): {'ok' if p1 else 'FAIL'}")
    print(f"P2 grip (F>=8N: rise>=0.12 m and slip<=15 mm): {held[hi].sum()}/{hi.sum()} "
          f"{'ok' if p2 else 'FAIL'} (NaN-free: {nan_free})")
    print(f"P3 force dependence (low force slips more): {'ok' if p3 else 'FAIL'} "
          f"[slip F<8N: {np.round(slip_mm[lo],1)} mm, held F>=8N: "
          f"{np.round(held_slip,1)} mm]")

    (ROOT / "reports/friction_grip_probe.json").write_text(json.dumps(dict(
        P1=bool(p1), P2=bool(p2), P3=bool(p3), mu_finger=MU_FINGER,
        F_grip_N=F_GRIP.tolist(), cube_mass_kg=(0.5 * fcube).round(3).tolist(),
        clampN_lift=clampN_lift.round(1).tolist(),
        clampN_end=np.column_stack([fL_end, fR_end]).round(1).tolist(),
        cube_rise_m=cube_rise.round(4).tolist(), slip_mm=slip_mm.round(1).tolist(),
        tilt_deg=tilt.round(1).tolist(), held=held.tolist(),
        pre_drift_mm=pre_drift.round(1).tolist(), nan_free=nan_free,
        sub_steps_per_s=n_sub[0] / t_step[0]), indent=1))
    if p1 and p2:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
