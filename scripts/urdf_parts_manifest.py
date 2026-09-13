#!/usr/bin/env python3
"""Build a parts manifest for a robot cell from a URDF, with no vendor-specific input.

The manifest is the single input that the cell scripts (`robotcell_bansvep_v1`,
`robotlaster_v1`, `robotcell_pickplace_v1`) consume: one entry per rigid part, with the part's
world-frame AABB at a declared reference pose, plus mass / centre of mass / inertia tensor.
Link membership is expressed as `subassembly` = `robot/link{n}` (movable) or `cell/<name>`
(static), which is what `link_of()` parses.

Geometry comes from the collision meshes referenced by the URDF (binary or ASCII STL); the AABB
is the exact mesh AABB transformed by the collision origin and by fk() at the reference pose.
Inertial data comes from the URDF `<inertial>` blocks, so no CAD/STEP input is needed. When a
part entry carries a `step` key instead, `robotlaster_v1` integrates the BRep exactly.

I/O:
  --urdf PATH        URDF to read (default: the vendored UR10e)
  --pose D*6         reference pose in degrees (the pose the AABBs are expressed at)
  --static JSON      optional file of static cell bodies [{name, bbox:[x0,y0,z0,x1,y1,z1] mm}]
  --out PATH         manifest JSON to write (default: data/cells/<robot>_parts_manifest_v1.json)
"""
import argparse
import json
import math
import os
import struct
import xml.etree.ElementTree as ET

import numpy as np

from robotcell_leder_v1 import DEFAULT_URDF, ROOT, _rpy, fk, load_chain


def stl_bounds(path):
    """(min[3], max[3]) of an STL in its own frame, metres. Handles binary and ASCII."""
    raw = open(path, "rb").read()
    n = struct.unpack("<I", raw[80:84])[0] if len(raw) >= 84 else 0
    if len(raw) == 84 + 50 * n and n > 0:
        v = np.frombuffer(raw[84:], dtype=np.uint8).reshape(n, 50)[:, 12:48].copy()
        pts = v.view("<f4").reshape(-1, 3).astype(float)
    else:
        pts = np.array([[float(x) for x in ln.split()[1:4]]
                        for ln in raw.decode("utf-8", "replace").splitlines()
                        if ln.strip().startswith("vertex")], dtype=float)
    return pts.min(axis=0), pts.max(axis=0)


def link_inertials(urdf_path):
    """{link_name: {mass, com(m), I_com(kg m^2, 3x3)}} from the URDF <inertial> blocks."""
    root = ET.parse(urdf_path).getroot()
    out = {}
    for l in root.findall("link"):
        ine = l.find("inertial")
        if ine is None:
            continue
        m = float(ine.find("mass").get("value"))
        o = ine.find("origin")
        c = np.array([float(x) for x in (o.get("xyz", "0 0 0").split() if o is not None else "0 0 0".split())])
        rpy = [float(x) for x in (o.get("rpy", "0 0 0").split() if o is not None else "0 0 0".split())]
        t = ine.find("inertia")
        I = np.array([[float(t.get("ixx")), float(t.get("ixy")), float(t.get("ixz"))],
                      [float(t.get("ixy")), float(t.get("iyy")), float(t.get("iyz"))],
                      [float(t.get("ixz")), float(t.get("iyz")), float(t.get("izz"))]])
        R = np.array(_rpy(*rpy))
        out[l.get("name")] = {"mass": m, "com": c, "I": R @ I @ R.T}
    return out


def link_collisions(urdf_path):
    """{link_name: [(mesh_path, origin_xyz(m), origin_rpy(rad))]} for collision geometry."""
    root = ET.parse(urdf_path).getroot()
    base = os.path.dirname(os.path.abspath(urdf_path))
    out = {}
    for l in root.findall("link"):
        items = []
        for col in l.findall("collision"):
            g = col.find("geometry")
            mesh = g.find("mesh") if g is not None else None
            if mesh is None:
                continue
            fn = mesh.get("filename").replace("package://", "")
            p = fn if os.path.isabs(fn) else os.path.join(base, os.path.basename(os.path.dirname(os.path.dirname(fn))) and fn)
            if not os.path.exists(p):
                p = os.path.join(base, fn.split("/", 1)[-1] if fn.startswith("meshes/") is False else fn)
            o = col.find("origin")
            xyz = [float(x) for x in (o.get("xyz", "0 0 0").split() if o is not None else "0 0 0".split())]
            rpy = [float(x) for x in (o.get("rpy", "0 0 0").split() if o is not None else "0 0 0".split())]
            items.append((p, np.array(xyz), rpy))
        if items:
            out[l.get("name")] = items
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--pose", nargs=6, type=float, default=[0.0, -90.0, 90.0, -90.0, -90.0, 0.0])
    ap.add_argument("--static", help="JSON list of static cell bodies")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "cells", "ur10e_parts_manifest_v1.json"))
    args = ap.parse_args()

    joints, limits, names = load_chain(args.urdf)
    q = [math.radians(a) for a in args.pose]
    F = fk(q, joints)
    inert = link_inertials(args.urdf)
    cols = link_collisions(args.urdf)

    # movable links in chain order: parent link of joint i is link i
    root = ET.parse(args.urdf).getroot()
    jd = {j.get("name"): (j.find("parent").get("link"), j.find("child").get("link"))
          for j in root.findall("joint")}
    chain_links = [jd[names[0]][0]] + [jd[n][1] for n in names]
    # fold links joined by fixed joints into their movable ancestor
    fixed_parent = {}
    for j in root.findall("joint"):
        if j.get("type") == "fixed":
            fixed_parent[j.find("child").get("link")] = j.find("parent").get("link")

    def link_index(name):
        seen = 0
        while name is not None and seen < 10:
            if name in chain_links:
                return chain_links.index(name)
            name = fixed_parent.get(name)
            seen += 1
        return -1

    parts = []
    for lname, items in cols.items():
        L = link_index(lname)
        if L < 0:
            continue
        R = np.array(F[L][0])
        p0 = np.array(F[L][1])
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        vol = 0.0
        for path, xyz, rpy in items:
            a, b = stl_bounds(path)
            Rc = np.array(_rpy(*rpy))
            corners = np.array([[x, y, z] for x in (a[0], b[0]) for y in (a[1], b[1]) for z in (a[2], b[2])])
            wc = (R @ (Rc @ corners.T + xyz[:, None]) * 1000.0).T + p0
            lo = np.minimum(lo, wc.min(axis=0))
            hi = np.maximum(hi, wc.max(axis=0))
            e = (b - a) * 1000.0
            vol += float(e[0] * e[1] * e[2])
        ine = inert.get(lname, {"mass": 0.0, "com": np.zeros(3), "I": np.zeros((3, 3))})
        parts.append({
            "part": lname,
            "subassembly": f"robot/link{L}",
            "bbox": [round(float(x), 4) for x in list(lo) + list(hi)],
            "volume_mm3": round(vol, 3),
            "mass_kg": ine["mass"],
            "com_link_m": [float(x) for x in ine["com"]],
            "inertia_com_kgm2": [[float(x) for x in r] for r in ine["I"]],
        })

    if args.static:
        for b in json.load(open(args.static)):
            parts.append({"part": b["name"], "subassembly": f"cell/{b['name']}",
                          "bbox": [float(x) for x in b["bbox"]],
                          "volume_mm3": 0.0, "mass_kg": 0.0,
                          "com_link_m": [0.0, 0.0, 0.0],
                          "inertia_com_kgm2": [[0.0] * 3] * 3})

    man = {"schema": "parts_manifest_v1",
           "urdf": os.path.relpath(os.path.abspath(args.urdf), ROOT),
           "world_frame": "mm, AABBs at the reference pose",
           "render_pose_deg": list(args.pose),
           "joint_names": names,
           "axis_limits_deg": [list(x) for x in limits],
           "n_parts": len(parts), "parts": parts}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(man, f, indent=1)
    print(f"parts manifest: {len(parts)} parts "
          f"({sum(1 for p in parts if p['subassembly'].startswith('robot/'))} on the arm), "
          f"total mass {sum(p['mass_kg'] for p in parts):.3f} kg")
    print(f"  -> {os.path.relpath(os.path.abspath(args.out), ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
