#!/usr/bin/env python3
"""Write the kinematics spec `data/kin_spec_<robot>.npz` that `motion_engine.fk_warp.WarpBatchFK` reads.

The spec is the single input of the GPU planner cells (`fleet_grad_trajopt`, `fleet_mppi_benchmark`,
`fleet_plannability_metric`, `hard_benchmark_plannability`, `mppi_completeness_audit`): the serial joint
chain plus a point cloud of the robot surface, both taken from a URDF that ships its collision meshes.
Only such a URDF can be used; a description without meshes has no surface to sample.

Arrays written (the keys WarpBatchFK reads):
  njoints, nq, parent(int32), jpl_t(nj,3 m), jpl_q(nj,4 xyzw), axis(nj,3), jtype(nj: 1 revolute),
  qidx(nj), pt_joint(M,3 link frame), pt_parent(M), ee_parent, ee_t, ee_q, plus q_lower/q_upper (rad).

Self-check: the spec chain is re-evaluated on the CPU and compared with `robotcell_leder_v1.fk` on the
same URDF over random poses; the maximum link-origin error is printed and gated at 1e-9 m.

  python scripts/extract_kin_spec.py [--urdf PATH] [--robot NAME] [--points N] [--out PATH]
"""
import argparse
import math
import os
import struct
import xml.etree.ElementTree as ET

import numpy as np

from robotcell_leder_v1 import DEFAULT_URDF, ROOT, _rpy, fk, load_chain
from urdf_parts_manifest import link_collisions


def stl_points(path):
    """All vertices of an STL (binary or ASCII) in its own frame, metres, (N,3)."""
    raw = open(path, "rb").read()
    n = struct.unpack("<I", raw[80:84])[0] if len(raw) >= 84 else 0
    if len(raw) == 84 + 50 * n and n > 0:
        v = np.frombuffer(raw[84:], dtype=np.uint8).reshape(n, 50)[:, 12:48].copy()
        return v.view("<f4").reshape(-1, 3).astype(float)
    return np.array([[float(x) for x in ln.split()[1:4]]
                     for ln in raw.decode("utf-8", "replace").splitlines()
                     if ln.strip().startswith("vertex")], dtype=float)


def _quat_from_R(R):
    """Rotation matrix -> quaternion (x, y, z, w)."""
    R = np.asarray(R, float)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] >= R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w], float)
    return q / np.linalg.norm(q)


def chain_link_names(urdf_path, names):
    """Movable-link names in chain order + {fixed child link: parent link}."""
    root = ET.parse(urdf_path).getroot()
    jd = {j.get("name"): (j.find("parent").get("link"), j.find("child").get("link"))
          for j in root.findall("joint")}
    chain = [jd[names[0]][0]] + [jd[n][1] for n in names]
    fixed_parent = {j.find("child").get("link"): j.find("parent").get("link")
                    for j in root.findall("joint") if j.get("type") == "fixed"}
    return chain, fixed_parent


def fixed_transform(urdf_path, from_link, to_link):
    """(t, R) of `to_link` in the frame of `from_link` over fixed joints, or None if not connected."""
    root = ET.parse(urdf_path).getroot()
    step = {}
    for j in root.findall("joint"):
        if j.get("type") != "fixed":
            continue
        o = j.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0").split() if o is not None else "0 0 0".split())])
        rpy = [float(v) for v in (o.get("rpy", "0 0 0").split() if o is not None else "0 0 0".split())]
        step[j.find("child").get("link")] = (j.find("parent").get("link"), xyz, np.array(_rpy(*rpy)))
    t = np.zeros(3); R = np.eye(3); link = to_link
    for _ in range(10):
        if link == from_link:
            return t, R
        if link not in step:
            return None
        par, xyz, Rj = step[link]
        t = xyz + Rj @ t
        R = Rj @ R
        link = par
    return None


def build_spec(urdf_path, n_points, seed=0):
    joints, limits_deg, names = load_chain(urdf_path)
    nj = len(joints) + 1                                   # index 0 = base link
    parent = np.arange(-1, nj - 1, dtype=np.int32); parent[0] = 0
    jpl_t = np.zeros((nj, 3)); jpl_q = np.zeros((nj, 4)); jpl_q[:, 3] = 1.0
    axis = np.zeros((nj, 3)); axis[:, 2] = 1.0
    jtype = np.zeros(nj, np.int32); qidx = np.full(nj, -1, np.int32)
    for i, (t_mm, rpy, ax) in enumerate(joints):
        j = i + 1
        jpl_t[j] = np.asarray(t_mm) / 1000.0
        jpl_q[j] = _quat_from_R(_rpy(*rpy))
        axis[j] = np.asarray(ax) / np.linalg.norm(ax)
        jtype[j] = 1                                       # load_chain returns revolute/continuous joints only
        qidx[j] = i

    chain, fixed_parent = chain_link_names(urdf_path, names)
    cols = link_collisions(urdf_path)
    rng = np.random.default_rng(seed)
    pts, pt_parent = [], []
    for lname, items in cols.items():
        L, seen = lname, 0
        while L is not None and L not in chain and seen < 10:
            L = fixed_parent.get(L); seen += 1
        if L not in chain:
            continue
        idx = chain.index(L)
        for path, xyz, rpy in items:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{lname}: collision mesh missing: {path}")
            v = stl_points(path) @ np.asarray(_rpy(*rpy)).T + np.asarray(xyz)
            take = min(n_points, len(v))
            sel = rng.choice(len(v), size=take, replace=False)
            pts.append(v[sel]); pt_parent.append(np.full(take, idx, np.int32))
    if not pts:
        raise SystemExit(f"{urdf_path}: no collision meshes found; this description cannot produce a spec")
    ee_parent = nj - 1
    ee_t = np.zeros(3); ee_q = np.array([0.0, 0.0, 0.0, 1.0])
    for cand in ("tool0", "flange", "ee_link"):
        tr = fixed_transform(urdf_path, chain[-1], cand)
        if tr is not None:
            ee_t, ee_q = tr[0], _quat_from_R(tr[1])
            break
    lo = np.array([math.radians(a) for a, _ in limits_deg])
    hi = np.array([math.radians(b) for _, b in limits_deg])
    return dict(njoints=nj, nq=len(joints), parent=parent, jpl_t=jpl_t, jpl_q=jpl_q, axis=axis,
                jtype=jtype, qidx=qidx, pt_joint=np.concatenate(pts).astype(np.float64),
                pt_parent=np.concatenate(pt_parent), ee_parent=ee_parent, ee_t=ee_t, ee_q=ee_q,
                q_lower=lo, q_upper=hi), joints


def _qrot(q, v):
    qv = q[:3]; w = q[3]
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def _spec_fk_origins(spec, q):
    """Link origins (m) from the spec alone, CPU reference for the parity check."""
    nj = spec["njoints"]
    T = [(np.zeros(3), np.eye(3))]
    for j in range(1, nj):
        tp, Rp = T[spec["parent"][j]]
        qq = spec["jpl_q"][j]
        Rj = np.column_stack([_qrot(qq, e) for e in np.eye(3)])
        ang = q[spec["qidx"][j]] if spec["qidx"][j] >= 0 else 0.0
        a = spec["axis"][j]; c, s, C = math.cos(ang), math.sin(ang), 1 - math.cos(ang)
        x, y, z = a
        Rm = np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                       [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                       [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])
        T.append((tp + Rp @ spec["jpl_t"][j], Rp @ Rj @ Rm))
    return np.array([t for t, _ in T])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--robot")
    ap.add_argument("--points", type=int, default=400, help="surface points sampled per collision mesh")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    robot = args.robot or os.path.splitext(os.path.basename(args.urdf))[0]
    out = args.out or os.path.join(ROOT, "data", f"kin_spec_{robot}.npz")

    spec, joints = build_spec(args.urdf, args.points)
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(64):
        q = rng.uniform(-2.5, 2.5, spec["nq"])
        ref = np.array([np.asarray(p) / 1000.0 for _, p in fk(list(q), joints)])
        worst = max(worst, float(np.abs(_spec_fk_origins(spec, q) - ref).max()))
    print(f"kin spec {robot}: {spec['nq']} joints, {len(spec['pt_joint'])} surface points")
    print(f"  chain parity vs robotcell_leder_v1.fk over 64 random poses: max |d origin| = {worst:.2e} m")
    if worst > 1e-9:
        raise SystemExit(f"chain parity failed: {worst:.3e} m > 1e-9 m")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, **spec)
    print(f"  wrote {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
