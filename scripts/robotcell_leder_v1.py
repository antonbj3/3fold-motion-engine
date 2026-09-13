#!/usr/bin/env python3
"""Serial-chain kinematics for a 6R robot cell, loaded from a URDF.

The chain (joint origins, joint axes and axis limits) is read from a URDF instead of being
hard-coded, so the same `fk()` serves any 6R arm. The default is the vendored UR10e
(`assets/robots/ur_description/ur10e.urdf`, BSD-3, see assets/robots/THIRD_PARTY.md).

Public API (unchanged names, used by robotcell_bansvep_v1 / robotlaster_v1 / robotcell_pickplace_v1):
  JOINTS           [((tx,ty,tz) mm, (rx,ry,rz) rpy rad, (ax,ay,az) unit axis)] * 6
  AXIS_LIMITS_DEG  [(lower_deg, upper_deg)] * 6
  BASE_XY, PED_TOP_Z   base placement of link 0 in world (mm)
  fk(q_rad)        -> [(R, p_mm)] for link_0 (base) .. link_6 (flange)
  joint_world(q_rad) -> per joint: world unit axis + origin (mm) at pose q
  load_chain(urdf)  -> (joints, limits_deg, names)

I/O:
  --urdf PATH      URDF to load (default: the vendored UR10e)
  --pose D D D D D D   joint angles in degrees (default: all zero)
  --json PATH      write the joint table as JSON
  --check          cross-check fk() against pinocchio's forward kinematics on the same URDF
                   (max position error printed; exit 1 above 1e-9 m)
"""
import argparse
import json
import math
import os
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_URDF = os.path.join(ROOT, "assets", "robots", "ur_description", "ur10e.urdf")


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


def load_chain(urdf_path):
    """Serial revolute chain from a URDF: fixed joints are folded into the next movable joint.

    Returns (joints, limits_deg, names) where joints[i] = ((tx,ty,tz) in mm, (r,p,y) in rad,
    (ax,ay,az) unit axis in the parent frame)."""
    root = ET.parse(urdf_path).getroot()
    joints = {}
    child_of = {}
    for j in root.findall("joint"):
        par = j.find("parent").get("link")
        chi = j.find("child").get("link")
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0").split() if o is not None else "0 0 0".split())]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0").split() if o is not None else "0 0 0".split())]
        a = j.find("axis")
        ax = [float(v) for v in a.get("xyz").split()] if a is not None else [0.0, 0.0, 1.0]
        lim = j.find("limit")
        joints[j.get("name")] = {
            "parent": par, "child": chi, "type": j.get("type"), "xyz": xyz, "rpy": rpy, "axis": ax,
            "lower": float(lim.get("lower")) if lim is not None and lim.get("lower") else None,
            "upper": float(lim.get("upper")) if lim is not None and lim.get("upper") else None,
            "effort": float(lim.get("effort")) if lim is not None and lim.get("effort") else None,
            "velocity": float(lim.get("velocity")) if lim is not None and lim.get("velocity") else None,
        }
        child_of.setdefault(par, []).append(j.get("name"))

    children = {j["child"] for j in joints.values()}
    roots = [j["parent"] for j in joints.values() if j["parent"] not in children]
    link = roots[0]

    out, limits, names = [], [], []
    pend_t = [0.0, 0.0, 0.0]
    pend_R = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]

    def _compose(t, R, xyz, rpy):
        Rn = _rpy(*rpy)
        tn = [t[i] + sum(R[i][k] * xyz[k] for k in range(3)) for i in range(3)]
        Rc = [[sum(R[i][k] * Rn[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        return tn, Rc

    while True:
        nxt = [n for n in child_of.get(link, []) if joints[n]["type"] in ("revolute", "continuous", "fixed")]
        if not nxt:
            break
        # depth-first along the branch that still carries movable joints
        pick = None
        for n in nxt:
            if joints[n]["type"] != "fixed":
                pick = n
                break
        if pick is None:
            pick = nxt[0]
        j = joints[pick]
        pend_t, pend_R = _compose(pend_t, pend_R, j["xyz"], j["rpy"])
        if j["type"] != "fixed":
            # express the accumulated fixed transform as (translation mm, rpy) relative to the previous joint
            sy = -pend_R[2][0]
            sy = max(-1.0, min(1.0, sy))
            pitch = math.asin(sy)
            if abs(math.cos(pitch)) > 1e-9:
                roll = math.atan2(pend_R[2][1], pend_R[2][2])
                yaw = math.atan2(pend_R[1][0], pend_R[0][0])
            else:
                roll = math.atan2(-pend_R[1][2], pend_R[1][1])
                yaw = 0.0
            n = math.sqrt(sum(c * c for c in j["axis"])) or 1.0
            out.append(([c * 1000.0 for c in pend_t], [roll, pitch, yaw], [c / n for c in j["axis"]]))
            lo = j["lower"] if j["lower"] is not None else -math.pi
            up = j["upper"] if j["upper"] is not None else math.pi
            limits.append((math.degrees(lo), math.degrees(up)))
            names.append(pick)
            pend_t = [0.0, 0.0, 0.0]
            pend_R = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
        link = j["child"]
        if len(out) == 6 and all(joints[n]["type"] == "fixed" for n in child_of.get(link, [])):
            break
    return out, limits, names


JOINTS, AXIS_LIMITS_DEG, JOINT_NAMES = load_chain(DEFAULT_URDF)
BASE_XY = (0.0, 0.0)
PED_TOP_Z = 0.0

# Effort/velocity limits of the default chain, read from the same URDF (public datasheet values).
_ROOT_XML = ET.parse(DEFAULT_URDF).getroot()
_LIM = {j.get("name"): j.find("limit") for j in _ROOT_XML.findall("joint") if j.find("limit") is not None}
AXIS_EFFORT_NM = [float(_LIM[n].get("effort")) for n in JOINT_NAMES]
AXIS_SPEED_DPS = [math.degrees(float(_LIM[n].get("velocity"))) for n in JOINT_NAMES]


def _mm(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _mv(a, v):
    return [sum(a[i][k] * v[k] for k in range(3)) for i in range(3)]


def _rot(axis, q):
    x, y, z = axis
    c, s, C = math.cos(q), math.sin(q), 1 - math.cos(q)
    return [[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C]]


def fk(q_rad, joints=None):
    """[(R, p)] for link_0 (base) .. link_6 (flange); p in mm."""
    J = joints if joints is not None else JOINTS
    R = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
    p = [BASE_XY[0], BASE_XY[1], PED_TOP_Z]
    out = [(R, tuple(p))]
    for i, (t, rpy, ax) in enumerate(J):
        p = [p[k] + v for k, v in enumerate(_mv(R, t))]
        R = _mm(R, _rpy(*rpy))
        R = _mm(R, _rot(ax, q_rad[i]))
        out.append((R, tuple(p)))
    return out


def joint_world(q_rad, joints=None):
    """Per joint: world unit axis and origin (mm) at pose q. The axis is expressed in the frame of
    link i-1 after the fixed part of the transform, so world axis = R_{i-1} @ rpy_i @ axis_i."""
    J = joints if joints is not None else JOINTS
    R = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
    p = [BASE_XY[0], BASE_XY[1], PED_TOP_Z]
    res = []
    for i, (t, rpy, ax) in enumerate(J):
        p = [p[k] + v for k, v in enumerate(_mv(R, t))]
        R = _mm(R, _rpy(*rpy))
        world_ax = _mv(R, ax)
        n = math.sqrt(sum(c * c for c in world_ax))
        world_ax = [c / n for c in world_ax]
        res.append({"origin": [round(c, 6) for c in p], "axis": [round(c, 9) for c in world_ax]})
        R = _mm(R, _rot(ax, q_rad[i]))
    return res


def _check(urdf, q_deg):
    """fk() vs pinocchio forward kinematics on the same URDF."""
    import numpy as np
    import pinocchio as pin
    model = pin.buildModelFromUrdf(urdf)
    data = model.createData()
    joints, _, names = load_chain(urdf)
    worst = 0.0
    rng = np.random.default_rng(0)
    poses = [np.zeros(6), np.radians(np.array(q_deg, dtype=float))] + \
            [rng.uniform(-1.5, 1.5, 6) for _ in range(20)]
    for q in poses:
        qp = np.zeros(model.nq)
        for i, n in enumerate(names):
            qp[model.idx_qs[model.getJointId(n)]] = q[i]
        pin.forwardKinematics(model, data, qp)
        p_pin = np.array(data.oMi[model.getJointId(names[-1])].translation)
        p_own = np.array(fk(list(q), joints)[6]) if False else np.array(fk(list(q), joints)[6][1]) * 1e-3
        worst = max(worst, float(np.linalg.norm(p_pin - p_own)))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--pose", nargs=6, type=float, default=[0.0] * 6, help="joint angles in degrees")
    ap.add_argument("--json", help="write the joint table here")
    ap.add_argument("--check", action="store_true", help="cross-check fk() against pinocchio")
    args = ap.parse_args()

    joints, limits, names = load_chain(args.urdf)
    q = [math.radians(a) for a in args.pose]
    jw = joint_world(q, joints)
    print(f"chain from {os.path.relpath(args.urdf, ROOT)}: {len(joints)} revolute joints")
    leder = []
    for i, n in enumerate(names):
        print(f"  {n:24s} origin_mm={[round(c,3) for c in jw[i]['origin']]} "
              f"axis={[round(c,4) for c in jw[i]['axis']]} limit_deg={tuple(round(x,2) for x in limits[i])}")
        leder.append({"joint": n, "typ": "revolute", "parent_link": f"link_{i}", "child_link": f"link_{i+1}",
                      "origin_mm": jw[i]["origin"], "axis": jw[i]["axis"],
                      "limit_deg": {"lower": limits[i][0], "upper": limits[i][1]}})
    R6, p6 = fk(q, joints)[6]
    print(f"  flange at pose {list(args.pose)} deg: {[round(c,4) for c in p6]} mm")

    if args.check:
        worst = _check(args.urdf, args.pose)
        print(f"CHECK fk vs pinocchio over 22 poses: worst position error {worst:.3e} m -> "
              f"{'PASS' if worst < 1e-9 else 'FAIL'}")
        return 0 if worst < 1e-9 else 1

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"schema": "leder_v1", "urdf": os.path.relpath(args.urdf, ROOT),
                       "world_frame": "mm", "pose_deg": list(args.pose),
                       "n_leder": len(leder), "leder": leder}, f, indent=1)
        print(f"  -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
