#!/usr/bin/env python
"""Friction pick across the robot fleet: a two-finger gripper picking and lifting a cube with friction only.

Same probe as probe_friction_grip.py, parametrised over robots (ROBOT env var) and plant seeds. A
two-finger gripper (prismatic joints on the wrist, box plates) clamps a 5 cm cube (0.5 kg +/- 30%) and
holds it through a 0.15 m lift on a randomised plant, using friction contacts only.

Setup: W worlds, randomised plant (robot mass, joint friction, cube mass), a per-world constant clamp
force in effort mode on the prismatic finger joints, a servo descent to the grasp pose, a grip latch, and
a lift; measure whether the cube follows and how far it slips.

Pre-registered criteria:
  F1: at least 90% of worlds hold (cube rise >= 0.12 m and slip <= 15 mm).
  F2: median slip among held worlds <= 10 mm.
  F3: median pre-grip drift < 5 mm.

Output: reports/friction_pick_<robot>_s<seed>.json. Requires the Newton simulator, torch and warp with CUDA.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from motion_engine.manip.report import run_meta  # noqa: E402

W = int(__import__("os").environ.get("FG_W", "256"))
ROBOT = os.environ.get("FG_ROBOT", "ur10e")
PLANT_SEED = int(os.environ.get("PLANT_SEED", "777"))
OBJ = np.array([0.6, 0.0, 0.375])       # kubcentrum (piedestaltopp 0.35 + 0.025)
CUBE_H = 0.025
LIFT = 0.15
F_GRIP = np.full(W, 10.0)   # rekommenderade bandet 8-12 N (probens dos-svar)
# run2: 30 N gav kontaktexplosion (kub -862 m); 20 N stabil → tak 20 N
MU_FINGER = float(os.environ.get("MU_FINGER", "1.0"))

# linjerat ez mot APPROACH=-z, samma konvention som newton_stack.py)
F_OFF = 0.05            # y offset of the plate centre from the tool axis at q=0
F_DROP = 0.12           # plate centre 12 cm "down" along local +z
PHX, PHY, PHZ = 0.02, 0.006, 0.022   # plate: 4 cm wide, 1.2 cm thick, 4.4 cm tall
Z_GRASP = OBJ[2] + 0.001 + F_DROP    # plattcentrum ~kubcentrum (run1: +7mm gav
                                     # gravitations-pitchmoment → 95° tilt i v4;
                                     # plattbotten 0.354 > piedestaltopp 0.35)


def resolve_urdf(robot):
    """Path of a vendored description for `robot`, whichever layout it ships in.

    Searched in order: assets/robots/<robot>.urdf, assets/robots/<robot>/<robot>.urdf,
    assets/robots/<vendor>/<robot>.urdf (the vendored descriptions that ship their meshes, e.g.
    ur_description/ur10e.urdf), assets/robots/fleet_urdf/<robot>.urdf. Raises if none exists, instead
    of handing a missing path to the URDF parser.
    """
    cands = [ROOT / f"assets/robots/{robot}.urdf",
             ROOT / f"assets/robots/{robot}/{robot}.urdf"]
    cands += sorted((ROOT / "assets/robots").glob(f"*/{robot}.urdf"))
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"no vendored URDF for robot {robot!r}; looked in "
                            + ", ".join(str(c.relative_to(ROOT)) for c in cands))


def build():
    import newton
    rob = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(rob)
    urdf = resolve_urdf(ROBOT)
    rob.add_urdf(str(urdf), collapse_fixed_joints=True, enable_self_collisions=False)
    nb_arm = len(rob.body_mass)          # 6, wrist_3 = nb_arm-1
    nd_arm = len(rob.joint_target_ke)    # 6
    n_jor = len(rob.joint_type)          # URDF joints (for the articulation check)

    for i in range(len(rob.shape_flags)):
        rob.shape_flags[i] = int(rob.shape_flags[i]) & ~int(newton.ShapeFlags.COLLIDE_SHAPES)

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
    # orphan-fel i finalize och ArticulationView ser dem inte)
    rob._finalize_imported_articulation(fjoints, parent_body=nb_arm - 1)
    assert rob.joint_articulation[fjoints[0]] == rob.joint_articulation[0], \
        "fingerled hamnade inte i URDF-artikulationen"

    # arm + fingrar: EFFORT-mode, noll kaskadstyvhet (som mallen)
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
    # njmax=300: run1 spammade "nefc overflow - please increase njmax to 78"
    # cone=elliptic + impratio=10: run2 visade KRAFTOBEROENDE ~40mm-glid
    solver = newton.solvers.SolverMuJoCo(model, disable_contacts=False,
                                         nconmax=4096, njmax=300,
                                         cone="elliptic",
                                         impratio=float(os.environ.get("IMPRATIO", "10")),
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

    arm_pat = ("*ur10e*" if ROBOT.startswith("ur10e")
               else "*irb2400*" if ROBOT.startswith("abb_irb2400")
               else "*m10ia*" if ROBOT.startswith("fanuc_m10ia")
               else "*gp12*" if ROBOT.startswith("motoman_gp12")
               else "*iiwa*")  # model.articulation_label probad: 'abb_irb2400',
                               # 'fanuc_m10ia', 'motoman_gp12' (natt2 vendor 4+5)
    view = ArticulationView(model, arm_pat,
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
    HOME = np.zeros(NJ)
    if ROBOT.startswith("ur10e"):
        HOME[:6] = [0.0, -1.2, 1.5, -1.8, -1.57, 0.0]
        ARM_TAU = [150.0, 150.0, 150.0, 28.0, 28.0, 28.0]
    elif ROBOT.startswith("abb_irb2400"):
        # frame-probad (nattpass 2): EE=(0.587,0,0.557) r=0.587 ez=[0,0,-1];
        HOME[:6] = [0.0, 0.4, 0.95, 0.0, 1.79, 0.0]
        # grav vid HOME j2 -287 / j3 -78 Nm (probat) + J^T-servo: FMAX 400 N
        # gravkompensationen (run1: kollaps); wrist som ur10e-klassen
        ARM_TAU = [400.0, 800.0, 400.0, 60.0, 40.0, 28.0]
    elif ROBOT.startswith("fanuc_m10ia"):
        # frame-probad (natt2 vendor 4+5): EE=(0.579,0,0.563) ez_z=-0.999;
        # piedestalen vid start → kontaktexplosion knuffade av kuben i
        HOME[:6] = [0.0, 0.1, -1.0, 0.0, -2.0, 0.0]
        # profil; URDF effort=0 (som abb)
        ARM_TAU = [250.0, 300.0, 200.0, 40.0, 40.0, 28.0]
    elif ROBOT.startswith("motoman_gp12"):
        # frame-probad: EE=(0.624,0,0.584) ez_z=-0.999 (kandidaten z 0.540
        # avvisad: paddbotten 0.398 = 2 mm under kubtopp — marginalkravet);
        HOME[:6] = [0.0, 0.0, -1.1, 0.0, -2.0, 0.0]
        # grav@HOME j2 -105/j3 +107 Nm (probat); URDF-effort
        ARM_TAU = [250.0, 300.0, 200.0, 40.0, 40.0, 28.0]
    else:   # iiwa: EE pointing down over the work zone
        HOME[:7] = [0.0, 0.6, 0.0, -1.4, 0.0, 0.8, 0.0]
        ARM_TAU = [150.0, 150.0, 80.0, 60.0, 40.0, 20.0, 20.0]
    TAU_MAX = np.concatenate([ARM_TAU,
                              [200.0, 200.0]])

    rng = np.random.default_rng(7)
    qa = view.get_attribute("joint_q", s0); arr = qa.numpy()
    arr[:, 0, :nd_arm] = HOME[:nd_arm] + rng.normal(0, 0.05, (W, nd_arm))
    arr[:, 0, nd_arm:] = 0.0
    qa.assign(arr); view.set_attribute("joint_q", s0, qa)
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    state = {"s0": s0, "s1": s1}

    gp = ROOT / f"data/{ROBOT}_gravparams.npz"
    assert gp.exists(), (f"gravity parameters missing: {gp} — pre-generate them with "
                         f"src/motion_engine/torch_gravity.py (autogen-fallbacken "
                         f"hade NameError-bugg, borttagen)")
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
    IERR_CLIP = 0.05
    if not ROBOT.startswith("ur10e"):
        KP_P, KD_P, KI_P = 1400.0, 120.0, 900.0
        IERR_CLIP = 0.10
    KP_O, KD_O = 18.0, 1.8
    FMAX, MMAX = 60.0, 15.0
    DEPTH_WIN = 0.025   # in-jaw depth window
    if ROBOT.startswith("abb_irb2400"):
        # (b·T/I = 900·|Jp_j3|²·0,033/5 ≈ 3,8 > 2) → Nyquist-chatter j3/j5
        # (tau-swing 254 Nm/steg) som blockerade |qd|<0,6-grinden i 217
        # KD < 2·I_j3/(T·|Jp_j3|²) ≈ 470.
        KD_P = 300.0
        FMAX = 400.0
        IERR_CLIP = 0.15
        # run3-forensik: 80 ej-grepp hade PERFEKT EE-xy (0,0 mm) men ez_z
        # -0,991 (7,7° pitch) → kub 17 mm "fel" i EE-ram-x (0,12·sin) →
        # diskret stabil (8·0,033/0,5=0,53<2; 30 verifierat INSTABILT i
        KP_O, KD_O = 120.0, 8.0
        MMAX = 40.0
        # run4-census (59%): kvarvarande klass = glid 12-22 mm. Forensiken
        DEPTH_WIN = 0.010
    if ROBOT.startswith(("fanuc_m10ia", "motoman_gp12")):
        # natt2 vendor 4+5: mellanklass (arm 130-150 kg) → ur10e-klassens
        KP_P, KD_P, KI_P = 900.0, 90.0, 600.0
        IERR_CLIP = 0.05
        DEPTH_WIN = 0.025
        # natt2-DIAG2 m10ia: 124 ej-greppade stod PERFEKT (xy-fel 0,6 mm,
        # (steady ~1,3° ≈ 3 mm), KD_O 8 (diskret stabil: 0,033·8/0,5=0,53<2,
        # samma armature), MMAX 40 (transient 120·0,155=18,6 Nm > 15)
        KP_O, KD_O = 120.0, 8.0
        MMAX = 40.0
    if ROBOT.startswith("motoman_gp12"):
        FMAX = 150.0
        KP_P, KD_P, KI_P = 1400.0, 300.0, 900.0
        IERR_CLIP = 0.15
    APPROACH = np.tile([0.0, 0.0, -1.0], (W, 1))
    ierr = np.zeros((W, 3))

    N_WP, N_DESC, N_CLOSE, N_LIFT, N_SET = 110, 240, 70, 120, 80
    if not ROBOT.startswith("ur10e"):
        N_DESC = 1200  # iiwa: slow alignment
    if ROBOT.startswith("abb_irb2400"):
        # glidrot FALSIFIERAD (run5: 59→54%); kantlatch var inte boven.
        N_DESC = 240
    if ROBOT.startswith(("fanuc_m10ia", "motoman_gp12")):
        # natt2-census m10ia @240: 138/256 greppade, 9 fel bland lyftade.
        N_DESC = 240
    Z_WP = 0.80
    goal_z = np.full(W, Z_WP)
    grip_on = np.zeros(W, bool)
    iiwa_forensic = []
    eez_at_grip = np.full(W, Z_GRASP)   # lift target relative to the grasp height
                                        # geometriska near-misses rise ~115mm)
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
        lt_g = view.get_link_transforms(state["s0"]).numpy()
        eqg = lt_g[:, 0, nb_arm - 1, 3:7]
        xcg = body_x(cube_local)
        dpos = xcg[:, :3] - ee
        qc_ = eqg.copy(); qc_[:, :3] *= -1
        uv_ = np.cross(qc_[:, :3], dpos) * 2.0
        rel_g = dpos + qc_[:, 3:4] * uv_ + np.cross(qc_[:, :3], uv_)
        in_jaw = (np.abs(rel_g[:, 0]) < 0.015) & (np.abs(rel_g[:, 1]) < 0.020) \
                 & (np.abs(rel_g[:, 2] - F_DROP) < DEPTH_WIN)
        new_grip = (~grip_on) & (k >= N_WP) & in_jaw \
                   & (np.linalg.norm(qd[:, :nd_arm], axis=1) < 0.6)
        if new_grip.any():
            eez_at_grip[new_grip] = ee[new_grip, 2]
            xf = 0.5 * (body_x(nb_arm)[:, 2] + body_x(nb_arm + 1)[:, 2])
            xc = body_x(cube_local)
            for wdx in np.where(new_grip)[0]:
                grip_on[wdx] = True; grip_step[wdx] = k
                pre_drift[wdx] = np.linalg.norm(xc[wdx, :2] - cube_xy0[wdx]) * 1e3
            print(f"  k={k}: GRIP worlds {list(np.where(new_grip)[0])}")
        if k % 2 == 0 and grip_on.any():
            lt_f = view.get_link_transforms(state["s0"]).numpy()
            eqf = lt_f[:, 0, nb_arm - 1, 3:7]
            xcf = body_x(cube_local)
            fLf, fRf = clamp_force()
            iiwa_forensic.append((k, ee.copy(), eqf.copy(), xcf[:, :3].copy(),
                                  q[:, nd_arm:nd_arm + 2].copy(), fLf, fRf,
                                  grip_on.copy()))
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
            goal_z[lw] = np.minimum(eez_at_grip[lw] + LIFT, goal_z[lw] + 0.004)

        # === arm-servo (J^T, 6 dof) ===
        qt = torch.as_tensor(q[:, :nd_arm], dtype=torch.float32, device=dev)
        Rw, p, z, _ = dyn._fk(qt)
        grav = dyn(qt).cpu().numpy()
        p = p.cpu().numpy(); zax = z.cpu().numpy()
        rvec = ee[:, None, :] - p
        Jp = np.cross(zax, rvec); Jr = zax
        goal = np.column_stack([np.tile(OBJ[:2], (W, 1)), goal_z])
        perr = goal - ee
        ierr = np.clip(ierr + perr * (SUB * dt), -IERR_CLIP, IERR_CLIP)
        v_ee = np.einsum("wjd,wj->wd", Jp, qd[:, :nd_arm])
        F = KP_P * perr + KI_P * ierr - KD_P * v_ee
        Fn = np.linalg.norm(F, axis=1, keepdims=True)
        F = F * np.minimum(1.0, FMAX / np.maximum(Fn, 1e-9))
        e_rot = np.cross(ez, APPROACH)
        w_ee = np.einsum("wjd,wj->wd", Jr, qd[:, :nd_arm])
        M = KP_O * e_rot - KD_O * w_ee
        Mn = np.linalg.norm(M, axis=1, keepdims=True)
        M = M * np.minimum(1.0, MMAX / np.maximum(Mn, 1e-9))
        tau6 = (np.einsum("wjd,wd->wj", Jp, F)
                + np.einsum("wjd,wd->wj", Jr, M) + grav)
        tau6 += np.einsum("wjd,wd->wj", Jp,
                          np.where(grip_on[:, None], [0, 0, 0.5 * 9.81], 0.0))
        # i 1/8; ramp 2 N + 0,5 N/steg ger mjukt anslag, full kraft < 40 steg)
        f_now = np.where(grip_on,
                         np.minimum(F_GRIP, 2.0 + 0.5 * np.maximum(k - grip_step, 0)),
                         0.0)
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
    if iiwa_forensic:
        (ROOT / "data/forensics").mkdir(parents=True, exist_ok=True)
        np.savez(str(ROOT / "data/forensics/iiwa_grip_forensic.npz"),
                 ks=np.array([f[0] for f in iiwa_forensic]),
                 ee=np.array([f[1] for f in iiwa_forensic]),
                 eq=np.array([f[2] for f in iiwa_forensic]),
                 xc=np.array([f[3] for f in iiwa_forensic]),
                 qf=np.array([f[4] for f in iiwa_forensic]),
                 fL=np.array([f[5] for f in iiwa_forensic]),
                 fR=np.array([f[6] for f in iiwa_forensic]),
                 grip=np.array([f[7] for f in iiwa_forensic]))
    ee_f, ez_f, q_f, qd_f = read_state()
    zerr_f = (Z_GRASP + LIFT) - ee_f[:, 2]
    bad_lift = lift_started & (cube_rise < 0.12)
    if bad_lift.any():
        print(f"  diagnostics, incomplete lifts ({int(bad_lift.sum())}): median EE z error "
              f"{np.median(zerr_f[bad_lift])*1e3:.0f} mm (ok worlds "
              f"{np.median(zerr_f[lift_started & (cube_rise>=0.12)])*1e3:.0f}); "
              f"final median |qd| {np.median(np.linalg.norm(qd_f[bad_lift, :nd_arm], axis=1)):.3f} "
              f"(stuck<0.05 / oscillating>0.2); median grip step "
              f"{np.median(grip_step[bad_lift]):.0f} (descent ends at {N_WP+N_DESC})")
    ng = ~grip_on
    if ng.any():
        exy_f = np.linalg.norm(ee_f[:, :2] - OBJ[None, :2], axis=1) * 1e3
        print(f"  diagnostics, worlds without a grip ({int(ng.sum())}): median EE xy error "
              f"{np.median(exy_f[ng]):.1f} mm (gripped {np.median(exy_f[~ng]) if (~ng).any() else float('nan'):.1f}); "
              f"median EE z {np.median(ee_f[ng, 2]):.3f} (target {Z_GRASP:.3f}); "
              f"median ez_z {np.median(ez_f[ng, 2]):.3f}; median |qd| "
              f"{np.median(np.linalg.norm(qd_f[ng, :nd_arm], axis=1)):.3f} "
              f"(stuck<0.05 / oscillating>0.2)")
        lt_d = view.get_link_transforms(state["s0"]).numpy()
        eq_d = lt_d[:, 0, nb_arm - 1, 3:7]
        xc_d = body_x(cube_local)
        dp_d = xc_d[:, :3] - ee_f
        qc_d = eq_d.copy(); qc_d[:, :3] *= -1
        uv_d = np.cross(qc_d[:, :3], dp_d) * 2.0
        rg_d = dp_d + qc_d[:, 3:4] * uv_d + np.cross(qc_d[:, :3], uv_d)
        cd_d = np.linalg.norm(xc_d[:, :2] - cube_xy0, axis=1) * 1e3
        qm = np.median(q_f[ng, :nd_arm], axis=0)
        qdm = np.median(np.abs(qd_f[ng, :nd_arm]), axis=0)
        print(f"  diagnostics per joint (no grip): median q {np.round(qm, 2)}, "
              f"median |qd| {np.round(qdm, 3)}")
        print(f"  diagnostics, cube in the EE frame for worlds without a grip (window x+-15 / y+-20 / "
              f"depth+-{DEPTH_WIN*1e3:.0f}): median |x| {np.median(np.abs(rg_d[ng,0]))*1e3:.1f} mm, "
              f"|y| {np.median(np.abs(rg_d[ng,1]))*1e3:.1f}, "
              f"depth deviation {np.median(rg_d[ng,2]-F_DROP)*1e3:.1f}; "
              f"median cube xy drift {np.median(cd_d[ng]):.1f} mm "
              f"(gripped {np.median(cd_d[~ng]) if (~ng).any() else float('nan'):.1f}); "
              f"median cube z {np.median(xc_d[ng,2]):.3f} (start {np.median(cube_z0[ng]):.3f})")
    held = (cube_rise >= 0.12) & (slip_mm <= 15.0) & lift_started
    print(f"  census: gripped {int(grip_on.sum())}/{W}, lift started "
          f"{int(lift_started.sum())}, rise<0.12 among lifted "
          f"{int(((cube_rise < 0.12) & lift_started).sum())}, "
          f"slip>15 among lifted {int(((slip_mm > 15) & lift_started).sum())}")

    print(f"\n  world |  F_grip | cube kg | clamp L/R N (at lift) | cube rise m | slip mm | tilt deg | holds")
    for w in range(min(W, 8)):
        print(f"   {w}   | {F_GRIP[w]:5.1f} N | {0.5*fcube[w]:.2f} | "
              f"{clampN_lift[w,0]:5.1f}/{clampN_lift[w,1]:5.1f} (end {fL_end[w]:.1f}/{fR_end[w]:.1f}) | "
              f"{cube_rise[w]:7.3f} | {slip_mm[w]:6.1f} | {tilt[w]:4.1f} | {bool(held[w])}")
    print(f"  pre-grip drift mm: {np.round(pre_drift, 1)}")
    print(f"  theory: m_max=0.65 kg -> required normal force = 6.38/(2*{MU_FINGER}) = "
          f"{6.38/(2*MU_FINGER):.1f} N per finger")
    print(f"  cost: {n_sub[0]/t_step[0]:.0f} substeps/s at W={W}")

    f1 = held.mean() >= 0.90
    f2 = float(np.median(slip_mm[held])) <= 10.0 if held.any() else False
    pv = pre_drift[pre_drift >= 0]
    f3 = float(np.median(pv)) < 5.0 if len(pv) else False
    print(f"F1 holds (rise>=0.12 m and slip<=15 mm): {held.mean()*100:.0f}% (required >=90) "
          f"{'ok' if f1 else 'FAIL'}")
    print(f"F2 slip (held worlds): median {np.median(slip_mm[held]) if held.any() else float('nan'):.1f} mm "
          f"(<=10) {'ok' if f2 else 'FAIL'}")
    print(f"F3 pre-grip drift: median {np.median(pv) if len(pv) else float('nan'):.1f} mm (<5) "
          f"{'ok' if f3 else 'FAIL'}")
    allok = f1 and f2 and f3 and nan_free and p1
    print(f"FRICTION GRIP: {'all criteria pass' if allok else 'falsified / partial'} "
          f"(NaN-free: {nan_free}, 8-dof view: {p1})")

    (ROOT / f"reports/friction_pick_{ROBOT}_s{PLANT_SEED}.json").write_text(json.dumps(dict(
        meta=run_meta(robot=ROBOT, W=W, plant_seed=PLANT_SEED),
        F1=bool(f1), F2=bool(f2), F3=bool(f3), all_pass=bool(allok),
        mu_finger=MU_FINGER,
        F_grip_N=F_GRIP.tolist(), cube_mass_kg=(0.5 * fcube).round(3).tolist(),
        clampN_lift=clampN_lift.round(1).tolist(),
        clampN_end=np.column_stack([fL_end, fR_end]).round(1).tolist(),
        cube_rise_m=cube_rise.round(4).tolist(), slip_mm=slip_mm.round(1).tolist(),
        tilt_deg=tilt.round(1).tolist(), held=held.tolist(),
        pre_drift_mm=pre_drift.round(1).tolist(), nan_free=nan_free,
        sub_steps_per_s=n_sub[0] / t_step[0]), indent=1))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
