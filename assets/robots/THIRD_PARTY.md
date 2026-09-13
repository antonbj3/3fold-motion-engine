# Third-party robot descriptions

This repository vendors robot descriptions from third parties. The URDF/XML files are redistributed under the
licences below, with the upstream attribution retained. **No vendor meshes are redistributed for the fleet**:
the fleet descriptions reference `package://<vendor>_support/meshes/...`, which is not present here, so the
fleet XMLs are usable for kinematics, inertials, joint limits and effort limits, but not for mesh collision
without fetching the upstream mesh packages under their own terms.

## `assets/robots/fleet_urdf/` — 165 descriptions

| upstream | count | licence |
|---|---|---|
| ROS-Industrial `abb_*_support` | 28 | Apache License 2.0 |
| ROS-Industrial `fanuc_*_support` | 40 | Apache License 2.0 |
| ROS-Industrial `kuka_*_support` | 37 | Apache License 2.0 |
| ROS-Industrial `motoman_*_support` | 48 | Apache License 2.0 |
| Universal Robots `ur_description` | 12 | BSD 3-Clause |

* ROS-Industrial: https://github.com/ros-industrial — the `*_support` packages are Apache-2.0.
* `ur_description`: https://github.com/UniversalRobots/Universal_Robots_ROS2_Description — BSD 3-Clause.

Not included, and why:
* `ar4_mk1/mk2/mk3.urdf` — carry absolute local mesh paths and no `*_support` provenance; upstream licence not
  confirmed.
* `motoman_motopos_d500.urdf`, `motoman_motopos_mh1655.urdf` — rewritten with absolute local mesh paths; they
  must be regenerated relative before they can ship.
* `ur10e_abs.urdf` — the absolute-path variant of `ur10e.urdf`, superseded by the relative
  `assets/robots/ur_description/ur10e.urdf`.
* `melfa` and `so_arm100` descriptions — upstream licence not confirmed.

## `assets/robots/ur_description/` — UR10e with collision meshes

`ur10e.urdf` and `meshes/ur10e/collision/*.stl` from Universal Robots `ur_description`, BSD 3-Clause; see
`assets/robots/ur_description/LICENSE`. This is the robot the cell scripts default to, and the only one shipped
with meshes.

## `assets/robots/franka_description/` — Franka Emika Panda

`franka_panda.urdf` and meshes from `franka_description`; see `assets/robots/franka_description/LICENSE`.
