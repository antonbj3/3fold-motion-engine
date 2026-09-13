"""Scene builder for the manipulation cells.

Builds arm + end-effector sphere, pedestal + free cube, then replicate and plant randomisation (via
rplant.py). `add_grasp_weld` and `build_pick_scene` add the grasp weld used by the contact-pick cell.
The build order URDF -> EE sphere -> pedestal -> cube -> weld -> replicate -> plant randomisation is
load bearing: the replica indices, and therefore the randomisation draws, depend on it.

Imported inside the measurement scripts' main(), after the torch/warp/newton environment is up.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import newton

from .rplant import apply_r_plant


@dataclass
class PushScene:
    """A replicated, plant-randomised world plus the indices the scene build computes."""
    builder: "newton.ModelBuilder"   # replicated (W worlds)
    nb_rob: int                      # robot bodies per world
    nd: int                          # arm DOFs per world
    nb_t: int                        # bodies per world (nb_rob + the cube)


def build_arm(urdf: Path, *, ee_r: float):
    """Arm from URDF + EFFORT mode + armature + end-effector sphere.

    The robot's own collision shapes are disabled (contact happens only through
    the end-effector sphere `ee_tip`). Returns (rob, nb_rob, nd).
    """
    rob = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(rob)
    rob.add_urdf(str(urdf),
                 collapse_fixed_joints=True, enable_self_collisions=False)
    for i in range(len(rob.joint_target_ke)):
        rob.joint_target_ke[i] = 0.0; rob.joint_target_kd[i] = 0.0
        rob.joint_target_mode[i] = int(newton.JointTargetMode.EFFORT)
    for i in range(len(rob.joint_armature)):
        rob.joint_armature[i] = 0.5
    for i in range(len(rob.joint_effort_limit)):
        if rob.joint_effort_limit[i] <= 0:
            rob.joint_effort_limit[i] = 1e4
    nb_rob = len(rob.body_mass)
    nd = len(rob.joint_target_ke)
    for i in range(len(rob.shape_flags)):
        rob.shape_flags[i] = int(rob.shape_flags[i]) & ~int(newton.ShapeFlags.COLLIDE_SHAPES)
    cfg0 = newton.ModelBuilder.ShapeConfig(density=0.0)
    rob.add_shape_sphere(nb_rob - 1, radius=ee_r, cfg=cfg0, label="ee_tip")
    return rob, nb_rob, nd


def add_pedestal_and_cube(rob, *, cube_pos: np.ndarray, cube_h: float,
                          pedestal_hx: float, pedestal_hy: float,
                          pedestal_hz: float):
    """Statisk piedestal + fri kub (massa 0,5 kg nominellt) — ordagrant ur push.

    The cube spawns 0.5 mm above the pedestal top (the settling margin).
    Returnerar kubens body-index.
    """
    # bred piedestal (pushyta) + fri kub
    rob.add_shape_box(-1, xform=((cube_pos[0], 0.0, pedestal_hz), (0, 0, 0, 1.0)),
                      hx=pedestal_hx, hy=pedestal_hy, hz=pedestal_hz, label="pedestal")
    cube_density = 0.5 / (2 * cube_h) ** 3
    cube = rob.add_body(xform=((cube_pos[0], cube_pos[1], cube_pos[2] + 0.0005), (0, 0, 0, 1.0)),
                        label="cube")
    rob.add_shape_box(cube, hx=cube_h, hy=cube_h, hz=cube_h,
                      cfg=newton.ModelBuilder.ShapeConfig(density=cube_density),
                      label="cube_shape")
    return cube


def replicate_with_rplant(rob, *, W: int, plant_seed: int,
                          nb_rob: int, nd: int, n_free_bodies: int = 1):
    """Replicate the single-world builder W times and apply plant randomisation."""
    b = newton.ModelBuilder(); b.replicate(rob, W, spacing=(0, 0, 0))
    nb_t = nb_rob + n_free_bodies
    apply_r_plant(b, W=W, nb_rob=nb_rob, nd=nd, nb_t=nb_t,
                  plant_seed=plant_seed)
    return b, nb_t


def add_grasp_weld(rob, *, ee_body: int, cube_body: int):
    """Grepp-weld EE<->kub, inaktiv vid bygge — ordagrant ur contact_pick.

    Activated per world at runtime by WeldLatch (grip.py) when the contact force
    crosses the threshold."""
    rob.add_equality_constraint_weld(body1=ee_body, body2=cube_body,
                                     anchor=(0.0, 0.0, 0.0), torquescale=1.0,
                                     enabled=False, label="grasp")


def build_pick_scene(urdf: Path, *, W: int, plant_seed: int,
                     cube_pos: np.ndarray, cube_h: float,
                     ee_r: float) -> PushScene:
    """The whole contact-pick scene chain: arm + end-effector sphere + narrow pedestal (0.06 x 0.06,
    topp z=0,35) + fri kub + grepp-weld + replicate + R-plant.

    The pedestal dimensions are the contact-pick literals (hx=hy=0.06, hz=0.175).
    OBS piedestal-y: legacy skrev (OBJ[0], OBJ[1], 0.175) med OBJ[1]=0.0 —
    The hard-coded y=0.0 in add_pedestal_and_cube is numerically identical (the chain's
    cube always stands at y=0; moving the cube requires a scene addition, not silent
    reuse)."""
    rob, nb_rob, nd = build_arm(urdf, ee_r=ee_r)
    cube = add_pedestal_and_cube(rob, cube_pos=cube_pos, cube_h=cube_h,
                                 pedestal_hx=0.06, pedestal_hy=0.06,
                                 pedestal_hz=0.175)
    add_grasp_weld(rob, ee_body=nb_rob - 1, cube_body=cube)
    b, nb_t = replicate_with_rplant(rob, W=W, plant_seed=plant_seed,
                                    nb_rob=nb_rob, nd=nd)
    return PushScene(builder=b, nb_rob=nb_rob, nd=nd, nb_t=nb_t)


def build_push_scene(urdf: Path, *, W: int, plant_seed: int,
                     cube_pos: np.ndarray, cube_h: float, ee_r: float,
                     pedestal_hx: float = 0.12, pedestal_hy: float = 0.16,
                     pedestal_hz: float = 0.175) -> PushScene:
    """The whole push scene chain: arm + end-effector sphere + pedestal + cube + replicate + plant randomisation."""
    rob, nb_rob, nd = build_arm(urdf, ee_r=ee_r)
    add_pedestal_and_cube(rob, cube_pos=cube_pos, cube_h=cube_h,
                          pedestal_hx=pedestal_hx, pedestal_hy=pedestal_hy,
                          pedestal_hz=pedestal_hz)
    b, nb_t = replicate_with_rplant(rob, W=W, plant_seed=plant_seed,
                                    nb_rob=nb_rob, nd=nd)
    return PushScene(builder=b, nb_rob=nb_rob, nd=nd, nb_t=nb_t)
