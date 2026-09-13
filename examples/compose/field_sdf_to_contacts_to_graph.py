#!/usr/bin/env python3
"""Cross-repo example: field SDF -> motion contacts -> graph node, three repos as siblings.

Stage 1 (field engine, sibling repo): build the holed plate, run faltkarna_v1_mesh_to_sdf on the CPU
backend and export the dense signed field in the plain array form (sdf, origin, pitch) as an .npz.
Stage 2 (motion engine, this repo): drop a box on that field with SDFContactEngine behind the
ContactEngine Protocol, run to rest, measure contact force against Mg, penetration against the box
half-extent and the bit-identity of two runs.
Stage 3 (graph engine, sibling repo): write the measurement as one node with two producer edges in the
{nodes, edges} schema of graph_store.py, load it and query it back.

The sibling repositories are found under the directory named by THREEFOLD_ROOT (default: the parent of
this repository). Nothing is imported from a repository that is not on that path, and no file of either
sibling repository is modified.

Three adapters live here (and only here) because the seams are not yet flush; each one is printed in
the report so the composition cost is visible rather than hidden:
  A1  units: the field engine works in mm, the motion engine in SI metres. The exported field is
      converted once, at the boundary.
  A2  contact surface: contact_engine_sdf resolves contact against an analytic half-space (plane or
      ramp), not against a sampled grid, so it cannot read the .npz directly. The adapter reads the
      support plane out of the grid instead: it samples the exported field trilinearly over the drop
      footprint, locates the zero crossing per column and checks that the crossing is flat and that
      the field is a half-space there (no hole under the footprint) before handing the plane height to
      the engine. The engine then runs in the plane frame, and the adapter maps its state back to
      part coordinates.
  A3  callable: the field engine returns a result dict (window origin gmin, shape, dense field, block
      counts); the plain (sdf, origin, pitch) arrays are assembled from it here.

  python examples/compose/field_sdf_to_contacts_to_graph.py [--out DIR] [--pitch MM]

Exit code 0 when every stage passes, 1 when any gate fails.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

MOTION_ROOT = Path(__file__).resolve().parents[2]
THREEFOLD_ROOT = Path(os.environ.get("THREEFOLD_ROOT", MOTION_ROOT.parent)).resolve()
FIELD_ROOT = THREEFOLD_ROOT / "3fold-field-engine"
GRAPH_ROOT = THREEFOLD_ROOT / "3fold-graph-engine"

# gates
VOL_REL_BOUND = 0.05          # the field repository's own rasterisation band (mesh_to_sdf selftest)
FZ_REL_BOUND = 0.01           # contact force against Mg
PEN_OVER_R_BOUND = 0.5        # penetration over the box half-extent
FLATNESS_FRAC_OF_PITCH = 1.0  # support-plane flatness over the footprint, in pitches
SUPPORT_DEPTH_PITCHES = 2.0   # material required below the crossing for the half-space model to hold

BOX_HALF_M = 0.004            # 8 mm cube
DROP_M = 0.004                # release height above the support plane
DENSITY = 700.0
DT = 1.0 / 240.0
N_STEPS = 240
FOOTPRINT_CENTRE_MM = (40.0, 20.0)   # solid material: no hole reaches |y| = 20 mm on this plate


def _need(path: Path, what: str):
    if not path.is_dir():
        raise SystemExit(f"sibling repository missing: {what} expected at {path} "
                         f"(set THREEFOLD_ROOT to the directory holding the 3fold repositories)")


# ───────────────────────────── stage 1: field ─────────────────────────────
def stage_field(out_dir: Path, pitch_mm: float) -> dict:
    _need(FIELD_ROOT, "3fold-field-engine")
    for p in (FIELD_ROOT / "src", FIELD_ROOT / "examples" / "parts"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import warp as wp
    wp.init()
    import holed_plate_v1 as part
    from field_engine.faltkarna_v1_mesh_to_sdf import mesh_to_sdf_del

    V, T = part.mesh()
    lo = V.min(axis=0) - 3.0 * pitch_mm
    res = mesh_to_sdf_del(wp, V, T, pitch_mm, 0.0, lo, "cpu", returnera_falt=True)

    pitch_eff = float(res["pitch_effektiv"])
    sdf_mm = np.asarray(res["sd"], dtype=np.float64)                 # A3: plain arrays out of the result dict
    origin_mm = np.asarray(lo, dtype=np.float64) + np.asarray(res["gmin"], dtype=np.float64) * pitch_eff

    length, width = part.plate_size_mm()
    vol_analytic = (length * width * part.THICKNESS_MM
                    - sum(part.brep_hole_volume_mm3(r) for r in part.HOLE_RADII_MM))
    vol_voxel = float(np.sum(res["solid_final"])) * pitch_eff ** 3
    rel = abs(vol_voxel - vol_analytic) / vol_analytic

    npz = out_dir / "holed_plate_sdf.npz"
    np.savez(npz, sdf=sdf_mm.astype(np.float32), origin=origin_mm, pitch=np.float64(pitch_eff))

    ok = rel < VOL_REL_BOUND and res["vattentathet"]["status"] == "OK"
    print(f"[stage 1 field] part=holed_plate_v1 pitch={pitch_eff} mm shape={tuple(sdf_mm.shape)} "
          f"blocks_active={res['n_active']}/{res['n_blocks']}")
    print(f"[stage 1 field] volume voxel={vol_voxel:.1f} mm3 analytic={vol_analytic:.1f} mm3 "
          f"rel={rel:.2e} (bound {VOL_REL_BOUND}) watertightness={res['vattentathet']['status']}")
    print(f"[stage 1 field] exported {npz.name} keys=sdf,origin,pitch  -> {'PASS' if ok else 'FAIL'}")
    return {"pass": bool(ok), "npz": str(npz), "pitch_mm": pitch_eff, "vol_voxel_mm3": vol_voxel,
            "vol_analytic_mm3": vol_analytic, "vol_rel": rel,
            "watertightness": res["vattentathet"]["status"], "shape": list(map(int, sdf_mm.shape))}


# ───────────────────── adapter A2: grid field -> support plane ─────────────────────
def _trilinear(sdf, origin, pitch, pts_mm):
    from scipy.ndimage import map_coordinates
    c = ((np.asarray(pts_mm, float) - origin) / pitch).T
    return map_coordinates(sdf, c, order=1, mode="nearest")


def support_plane_from_grid(npz_path: Path, centre_mm, half_mm, n=9):
    """Reads the support plane out of the exported grid: zero crossing per column over the footprint.

    Returns (z_plane_mm, flatness_mm, is_halfspace). The footprint is scanned from above; a column
    without a sign change, or one whose material does not continue for SUPPORT_DEPTH_PITCHES below the
    crossing, means the half-space model does not hold there (a hole or a step under the footprint).
    The part is finite, so the test is local: it asks for a support depth, not for an infinite solid.
    """
    d = np.load(npz_path)
    sdf, origin, pitch = d["sdf"].astype(np.float64), d["origin"], float(d["pitch"])
    xs = np.linspace(centre_mm[0] - half_mm, centre_mm[0] + half_mm, n)
    ys = np.linspace(centre_mm[1] - half_mm, centre_mm[1] + half_mm, n)
    zs = np.arange(origin[2], origin[2] + (sdf.shape[2] - 1) * pitch, 0.1 * pitch)[::-1]  # top down
    crossings, halfspace = [], True
    for x in xs:
        for y in ys:
            col = _trilinear(sdf, origin, pitch, np.stack([np.full_like(zs, x), np.full_like(zs, y), zs], -1))
            neg = np.nonzero(col < 0.0)[0]
            if neg.size == 0:
                halfspace = False
                continue
            k = neg[0]
            if k == 0:
                halfspace = False
                continue
            z0, z1, s0, s1 = zs[k - 1], zs[k], col[k - 1], col[k]
            crossings.append(z0 + (z1 - z0) * s0 / (s0 - s1))
            depth = SUPPORT_DEPTH_PITCHES * pitch
            band = (zs <= crossings[-1]) & (zs >= crossings[-1] - depth)
            if np.any(col[band] >= 0.0):                      # material must continue for the support depth
                halfspace = False
    if not crossings:
        return None, None, False
    cr = np.asarray(crossings)
    return float(cr.mean()), float(cr.max() - cr.min()), bool(halfspace)


def _analytic_top_face_mm() -> float:
    """Top face of the plate in part coordinates, from the part module's own constants."""
    import holed_plate_v1 as part
    return 0.5 * part.THICKNESS_MM


# ───────────────────────────── stage 2: motion ─────────────────────────────
def _run_drop(z_plane_m):
    sys.path.insert(0, str(MOTION_ROOT / "src")) if str(MOTION_ROOT / "src") not in sys.path else None
    from motion_engine.contact_engine import ContactEngine
    from motion_engine.contact_engine_sdf import SDFContactEngine

    e = SDFContactEngine(mu=0.5, ramp_deg=0.0)
    assert isinstance(e, ContactEngine), "SDFContactEngine does not satisfy the ContactEngine Protocol"
    # A2: the engine solves in the plane frame; z = 0 of the engine is the support plane of the field.
    e.add_body("box", (BOX_HALF_M, BOX_HALF_M, BOX_HALF_M),
               [0.0, 0.0, BOX_HALF_M + DROP_M], density=DENSITY)
    for _ in range(N_STEPS):
        e.step(DT)
    st = e.get_state()
    f = e.contact_forces()
    M = float(e._M[0])
    return st, f, M


def stage_motion(field_res: dict) -> dict:
    z_plane_mm, flat_mm, halfspace = support_plane_from_grid(
        Path(field_res["npz"]), FOOTPRINT_CENTRE_MM, BOX_HALF_M * 1000.0)
    pitch = field_res["pitch_mm"]
    z_face_mm = _analytic_top_face_mm()
    plane_ok = (z_plane_mm is not None and halfspace
                and flat_mm <= FLATNESS_FRAC_OF_PITCH * pitch)
    print(f"[stage 2 motion] adapter A2 support plane z={z_plane_mm:.3f} mm flatness={flat_mm:.3f} mm "
          f"(bound {FLATNESS_FRAC_OF_PITCH * pitch:.3f}) half-space={halfspace}")
    print(f"[stage 2 motion] adapter A2 plane vs analytic top face {z_face_mm:.3f} mm: "
          f"offset {(z_plane_mm - z_face_mm):+.3f} mm = {(z_plane_mm - z_face_mm) / pitch:+.2f} pitch "
          f"(the grid's zero crossing, not a modelling choice here)")
    if not plane_ok:
        print("[stage 2 motion] -> FAIL (no usable support plane under the footprint)")
        return {"pass": False, "reason": "support plane"}

    z_plane_m = z_plane_mm / 1000.0                                  # A1: mm -> m, once, at the boundary
    st1, f1, M = _run_drop(z_plane_m)
    st2, f2, _ = _run_drop(z_plane_m)

    Mg = M * 9.81
    fz = float(f1[0, 2])
    fz_rel = abs(fz - Mg) / Mg
    z_body_m = float(st1.xc[0, 2])
    pen_m = max(0.0, BOX_HALF_M - z_body_m)                          # deepest corner below the plane
    pen_over_R = pen_m / BOX_HALF_M
    bit = all(a.tobytes() == b.tobytes() for a, b in
              ((st1.xc, st2.xc), (st1.Rm, st2.Rm), (st1.vc, st2.vc), (st1.om, st2.om)))
    z_world_mm = z_plane_mm + z_body_m * 1000.0                      # A2 inverse: back to part coordinates
    v_rest = float(np.linalg.norm(st1.vc[0]))

    ok = fz_rel < FZ_REL_BOUND and pen_over_R < PEN_OVER_R_BOUND and bit
    print(f"[stage 2 motion] engine={'sdf_contact_core'} body=box 2x{BOX_HALF_M} m M={M:.6f} kg "
          f"steps={N_STEPS} dt={DT:.6f} s |v|_rest={v_rest:.2e} m/s")
    print(f"[stage 2 motion] Fz={fz:.6f} N Mg={Mg:.6f} N rel={fz_rel:.2e} (bound {FZ_REL_BOUND})")
    print(f"[stage 2 motion] penetration={pen_m * 1000:.4f} mm /R={pen_over_R:.3f} "
          f"(bound {PEN_OVER_R_BOUND})  two runs bit-identical={bit}")
    print(f"[stage 2 motion] body centre in part coordinates z={z_world_mm:.3f} mm "
          f"-> {'PASS' if ok else 'FAIL'}")
    return {"pass": bool(ok), "z_plane_mm": z_plane_mm, "z_analytic_face_mm": z_face_mm,
            "plane_offset_pitches": (z_plane_mm - z_face_mm) / pitch, "flatness_mm": flat_mm, "M_kg": M,
            "Fz_N": fz, "Mg_N": Mg, "Fz_rel": fz_rel, "penetration_mm": pen_m * 1000.0,
            "pen_over_R": pen_over_R, "bit_identical": bit, "z_body_part_mm": z_world_mm,
            "v_rest_mps": v_rest}


# ───────────────────────────── stage 3: graph ─────────────────────────────
NODE_ID = "n.compose.holed_plate_sdf_box_rest"
FIELD_NODE = "n.field.faltkarna_v1_mesh_to_sdf"
MOTION_NODE = "n.motion.contact_engine_sdf"


def stage_graph(out_dir: Path, field_res: dict, motion_res: dict) -> dict:
    _need(GRAPH_ROOT, "3fold-graph-engine")
    if str(GRAPH_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(GRAPH_ROOT / "src"))
    from graph_engine.tools.graph_store import GraphStore

    status = ("VERIFIED-FRESH" if field_res["pass"] and motion_res["pass"] else "FAILED")
    graph = {
        "nodes": [
            {"id": FIELD_NODE, "type": "build", "title": "mesh to block-sparse SDF (field engine)"},
            {"id": MOTION_NODE, "type": "build", "title": "analytic-SDF contact engine (motion engine)"},
            {"id": NODE_ID, "type": "measurement",
             "title": "box at rest on the holed-plate field: contact force and penetration",
             "part": "holed_plate_v1", "engine": "sdf_contact_core", "status": status,
             "numbers": {"pitch_mm": field_res["pitch_mm"], "vol_rel": field_res["vol_rel"],
                         "Fz_N": motion_res.get("Fz_N"), "Mg_N": motion_res.get("Mg_N"),
                         "Fz_rel": motion_res.get("Fz_rel"), "pen_over_R": motion_res.get("pen_over_R"),
                         "bit_identical": motion_res.get("bit_identical")}},
        ],
        "edges": [{"src": FIELD_NODE, "dst": NODE_ID, "type": "produces"},
                  {"src": MOTION_NODE, "dst": NODE_ID, "type": "produces"}],
    }
    path = out_dir / "GRAPH_STORE.json"
    path.write_text(json.dumps(graph, indent=2), encoding="utf-8")

    gs = GraphStore(str(path))
    node = gs.load(NODE_ID)
    nb = gs.neighbors(NODE_ID, depth=1)
    producers = sorted(e["src"] for e in gs.in_edges(NODE_ID, "produces"))
    ok = (node is not None and node["numbers"]["Fz_N"] == motion_res.get("Fz_N")
          and producers == sorted([FIELD_NODE, MOTION_NODE])
          and set(nb.get(1, [])) == {FIELD_NODE, MOTION_NODE})
    print(f"[stage 3 graph] wrote {path.name}; GraphStore.load({NODE_ID}) -> "
          f"{'node' if node else 'None'} status={node and node['status']}")
    print(f"[stage 3 graph] neighbors(depth=1)={nb.get(1, [])} producers={producers} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return {"pass": bool(ok), "graph": str(path), "node_id": NODE_ID, "producers": producers}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None, help="output directory for the .npz and the graph (default: a temp dir)")
    ap.add_argument("--pitch", type=float, default=1.0, help="rasterisation pitch in mm")
    a = ap.parse_args()
    out = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="compose_"))
    out.mkdir(parents=True, exist_ok=True)
    print(f"compose: field -> motion -> graph   THREEFOLD_ROOT={THREEFOLD_ROOT}  out={out}")

    field_res = stage_field(out, a.pitch)
    motion_res = stage_motion(field_res)
    graph_res = stage_graph(out, field_res, motion_res)

    rows = [("field", field_res["pass"]), ("motion", motion_res["pass"]), ("graph", graph_res["pass"])]
    print("\nstage results: " + "  ".join(f"{n}={'PASS' if p else 'FAIL'}" for n, p in rows))
    (out / "compose_result.json").write_text(
        json.dumps({"field": field_res, "motion": motion_res, "graph": graph_res}, indent=2), encoding="utf-8")
    allpass = all(p for _, p in rows)
    print("ALL_PASS" if allpass else "FAIL")
    return 0 if allpass else 1


if __name__ == "__main__":
    raise SystemExit(main())
