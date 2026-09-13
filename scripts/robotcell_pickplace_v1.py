#!/usr/bin/env python3
"""A pick-and-place work cell that does a job: sequence, retiming and the gate battery.

Builds one complete, playable pick-and-place sequence for a robot described by a URDF and a cell
described by a JSON file, and writes it as a trajectory ({t, q, faser, gods}) that
`robotcell_bansvep_v1 --bana` and `robotlaster_v1 --bana` consume.

  * KINEMATICS      robotcell_leder_v1.fk() / ik_robust() (Newton IK on the same fk, limit-checked
                    over several seeds, wrapped into (-pi, pi])
  * APPROACH HEIGHT measured, not chosen: the highest candidate that is reachable (IK residual below
                    1e-6 mm and inside the joint limits) for ALL stations
  * MOTION          straight-line TCP moves (LIN, industry standard), not joint interpolation; each
                    segment is sampled at <= LIN_STEG_MM with an adaptive step that halves when a
                    step would demand more than LIN_MAX_DQ_RAD of joint change (a measure of
                    proximity to a singularity, reported per segment)
  * TIME LAW        quintic (zero velocity and acceleration at each waypoint), scaled so the peak
                    speed hits exactly V_FRAC of the URDF velocity limit
  * GATE 1          IK residual and joint-limit margin over every waypoint
  * GATE 2          OBB/SAT collision sweep over the whole sequence, with a POSITIVE CONTROL (the
                    sequence is displaced by FALLBEVIS_MM and the gate must then report a collision)
  * GATE 3          speed/acceleration against the URDF datasheet limits

Every quantity is declared MEASURED / DERIVED / ASSUMED in the report's `sanningsmarkning`.

I/O:
  --parts-manifest PATH  collision geometry (default: data/cells/ur10e_parts_manifest_v1.json)
  --cell PATH            cell definition (default: data/cells/synthetic_pickplace_cell_v1.json)
  --verify               recompute, write no files
  --out PATH             report JSON
  --bana-out PATH        trajectory JSON
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robotcell_leder_v1 import ROOT, AXIS_SPEED_DPS, fk, load_chain, DEFAULT_URDF  # noqa: E402
from robotcell_bansvep_v1 import (aabb_of, aabb_overlap, corners, link_of, obb_overlap,  # noqa: E402
                                  se3_apply, se3_inv_apply)

HZ = 50.0                     # trajectory sampling rate
V_FRAC = 0.30                 # share of the datasheet joint speed (typical programmed speed)
APPROACH_KANDIDATER = [150.0, 120.0, 100.0, 80.0, 60.0]
DWELL_S = 0.40                # gripper on/off dwell
LIN_STEG_MM = 20.0
LIN_MAX_DQ_RAD = math.radians(12.0)
LIN_MIN_STEG_MM = 0.5
FALLBEVIS_MM = 200.0          # displacement used by the positive control (must exceed the tool length)
GAP_NOM = 0.05                # nominal grasp gap (mm)

DEFAULT_MANIFEST = os.path.join(ROOT, "data", "cells", "ur10e_parts_manifest_v1.json")
DEFAULT_CELL = os.path.join(ROOT, "data", "cells", "synthetic_pickplace_cell_v1.json")
_STATE = {"joints": None, "nj": 6, "limits": None}


def tcp_frame(q, l_tool):
    F = fk(list(q), _STATE["joints"])[_STATE["nj"]]
    R = np.array(F[0], dtype=float)
    p = np.array(F[1], dtype=float)
    return R, p + l_tool * R[:, 2]


def _pose_err(q, Rt, pt, l_tool):
    R, p = tcp_frame(q, l_tool)
    dp = pt - p
    Rr = Rt @ R.T
    w = np.array([Rr[2, 1] - Rr[1, 2], Rr[0, 2] - Rr[2, 0], Rr[1, 0] - Rr[0, 1]])
    s = np.linalg.norm(w)
    c = (np.trace(Rr) - 1.0) / 2.0
    ang = math.atan2(s / 2.0, c)
    dw = (w / s) * ang if s > 1e-12 else np.zeros(3)
    return np.concatenate([dp, dw * 180.0])      # the rotation part scaled to millimetre magnitude


def ik(Rt, pt, q0, l_tool, iters=200):
    """Gauss-Newton IK on the existing fk(). Returns (q, residual_mm, residual_deg)."""
    n = _STATE["nj"]
    q = np.array(q0, dtype=float)
    for _ in range(iters):
        e = _pose_err(q, Rt, pt, l_tool)
        if np.linalg.norm(e) < 1e-12:
            break
        J = np.zeros((6, n))
        h = 1e-7
        for j in range(n):
            qp = q.copy()
            qp[j] += h
            J[:, j] = (_pose_err(qp, Rt, pt, l_tool) - e) / h
        try:
            dq = np.linalg.solve(J.T @ J + 1e-9 * np.eye(n), J.T @ e)
        except np.linalg.LinAlgError:
            return None, float("inf"), float("inf")
        q = q - dq
    R, p = tcp_frame(q, l_tool)
    Rr = Rt @ R.T
    c = max(-1.0, min(1.0, (np.trace(Rr) - 1.0) / 2.0))
    return q, float(np.linalg.norm(pt - p)), float(math.degrees(math.acos(c)))


def limit_margins(q):
    d = np.degrees(q)
    lim = _STATE["limits"]
    return [min(d[j] - lim[j][0], lim[j][1] - d[j]) for j in range(_STATE["nj"])]


def wrap_q(q):
    """Revolute joints: the angle is defined only modulo 2*pi. Folding into (-pi, pi] keeps a Newton
    solution that went around a full turn from sounding like a limit violation."""
    return np.array([((float(x) + math.pi) % (2.0 * math.pi)) - math.pi for x in q])


def ik_robust(Rt, pt, l_tool, seeds, forsta_duger=False):
    """Tries several seeds, folds the solution, and admits only solutions that (a) hit the pose at
    machine precision and (b) lie INSIDE the joint limits. Picks the one with the largest limit
    margin. Returns (q, res_mm, res_deg, margin_deg) or None."""
    bast = None
    for s0 in seeds:
        q, _, _ = ik(Rt, pt, s0, l_tool)
        if q is None:
            continue
        q = wrap_q(q)
        R2, p2 = tcp_frame(q, l_tool)
        rmm = float(np.linalg.norm(np.asarray(pt) - p2))
        Rr = np.asarray(Rt) @ R2.T
        c = max(-1.0, min(1.0, (np.trace(Rr) - 1.0) / 2.0))
        rdeg = float(math.degrees(math.acos(c)))
        if rmm > 1e-6 or rdeg > 1e-6:
            continue
        m = min(limit_margins(q))
        if m <= 0.0:
            continue
        if bast is None or m > bast[3]:
            bast = (q, rmm, rdeg, m)
        if forsta_duger:
            return bast
    return bast


def quintic(u):
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


def segment_duration(q_a, q_b, vmax_rad):
    """A quintic profile peaks at 1.875x the mean speed, so T = 1.875*max|dq|/vmax."""
    dq = np.abs(np.array(q_b) - np.array(q_a))
    if np.all(dq < 1e-12):
        return 0.0
    return float(np.max(1.875 * dq / vmax_rad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts-manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--cell", default=DEFAULT_CELL)
    ap.add_argument("--verify", action="store_true", help="recompute, write no files")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "robotcell_pickplace_v1.json"))
    ap.add_argument("--bana-out", default=os.path.join(ROOT, "data", "banor",
                                                       "robotcell_pickplace_v1.json"))
    args = ap.parse_args()
    t_start = time.time()

    man = json.load(open(args.parts_manifest))
    cell = json.load(open(args.cell))
    urdf = os.path.join(ROOT, man.get("urdf", os.path.relpath(DEFAULT_URDF, ROOT)))
    joints, limits, jnames = load_chain(urdf)
    _STATE["joints"] = joints
    _STATE["nj"] = nj = len(joints)
    _STATE["limits"] = man.get("axis_limits_deg", limits)

    L_TOOL = float(cell["verktygslangd_mm"])
    VMAX_DPS = np.array(AXIS_SPEED_DPS[:nj], dtype=float)
    VMAX_RAD = np.radians(VMAX_DPS)
    STATIONER = cell["stationer"]
    AMNE = cell["arbetsstycke"]
    LT = float(AMNE["storlek_mm"][2])
    Q_HOME = np.radians(cell["hem_pose_deg"])
    R_TOOLDOWN = np.array(tcp_frame(Q_HOME, L_TOOL)[0])

    rep = {"cell": "robotcell_pickplace_v1", "urdf": man.get("urdf"),
           "cellfil": os.path.relpath(os.path.abspath(args.cell), ROOT),
           "robot": {"verktygslangd_mm": L_TOOL, "axelgranser_deg": _STATE["limits"],
                     "axis_speed_dps": [float(x) for x in VMAX_DPS],
                     "programhastighet_andel_av_datablad": V_FRAC}}

    SEEDS = [Q_HOME] + [wrap_q(Q_HOME + np.radians(d)) for d in
                        ([20, 0, 0, 0, 0, 0], [-20, 0, 0, 0, 0, 0], [0, 15, -15, 0, 0, 0],
                         [0, -15, 15, 0, 0, 0], [0, 0, 0, 25, 0, 0], [0, 0, 0, -25, 0, 0])]

    # ---- approach height: the highest candidate reachable for ALL stations (measured, not chosen)
    APPROACH_MM = None
    approach_prov = []
    for h in APPROACH_KANDIDATER:
        marginaler = []
        for kind in ("in", "ut"):
            for s in STATIONER[kind]:
                tgt = np.array([s["centrum_xy"][0], s["centrum_xy"][1], s["topp_z"] + h])
                b = ik_robust(R_TOOLDOWN, tgt, L_TOOL, SEEDS)
                marginaler.append(None if b is None else round(b[3], 4))
        nabar = all(m is not None for m in marginaler)
        approach_prov.append({"hojd_mm": h, "nabar": bool(nabar),
                              "ledmarginal_per_station_deg": marginaler})
        if nabar:
            APPROACH_MM = h
            break
    if APPROACH_MM is None:
        raise SystemExit("no approach height is reachable for all stations")
    rep["ansatshojd"] = {"vald_mm": APPROACH_MM, "prov": approach_prov,
                         "motiv": "highest candidate with IK residual < 1e-6 mm for all stations"}

    def solve(tcp_xyz, q_seed, namn, typ):
        b = ik_robust(R_TOOLDOWN, np.array(tcp_xyz, dtype=float), L_TOOL,
                      [np.asarray(q_seed)] + SEEDS)
        if b is None:
            raise SystemExit(f"IK: no solution inside the joint limits for {namn} @ {tcp_xyz}")
        q, rmm, rdeg, marg = b
        return {"namn": namn, "q": q, "tcp": tcp_frame(q, L_TOOL)[1], "typ": typ,
                "ik_res_mm": rmm, "ik_res_deg": rdeg, "ledmarginal_deg": marg}

    wp = [{"namn": "hem_start", "q": Q_HOME, "tcp": tcp_frame(Q_HOME, L_TOOL)[1], "typ": "pose",
           "ik_res_mm": 0.0, "ik_res_deg": 0.0, "ledmarginal_deg": min(limit_margins(Q_HOME))}]
    q_seed = Q_HOME
    cykler = []
    for i in range(3):
        sin, sut = STATIONER["in"][i], STATIONER["ut"][i]
        p_pick = [sin["centrum_xy"][0], sin["centrum_xy"][1], sin["topp_z"]]
        p_place = [sut["centrum_xy"][0], sut["centrum_xy"][1], sut["topp_z"]]
        a_pick = [p_pick[0], p_pick[1], p_pick[2] + APPROACH_MM]
        a_place = [p_place[0], p_place[1], p_place[2] + APPROACH_MM]
        w = [solve(a_pick, q_seed, f"ansats_plock_{i}", "ansats"),
             solve(p_pick, q_seed, f"plock_{i}", "kontakt"),
             solve(a_pick, q_seed, f"lyft_{i}", "ansats"),
             solve(a_place, q_seed, f"ansats_lagg_{i}", "ansats"),
             solve(p_place, q_seed, f"lagg_{i}", "kontakt"),
             solve(a_place, q_seed, f"retur_{i}", "ansats")]
        wp += w
        cykler.append({"i": i, "in": sin, "ut": sut, "plock_tcp": p_pick, "lagg_tcp": p_place})
        q_seed = w[0]["q"]
    wp.append({"namn": "hem_slut", "q": Q_HOME, "tcp": tcp_frame(Q_HOME, L_TOOL)[1], "typ": "pose",
               "ik_res_mm": 0.0, "ik_res_deg": 0.0, "ledmarginal_deg": min(limit_margins(Q_HOME))})

    def lin_path(a, b):
        """Dense joint sequence along the straight TCP line a->b with an ADAPTIVE step. Near full
        extension the Jacobian is ill conditioned, so a 20 mm step can demand more than 12 degrees of
        joint change; the step is then halved down to LIN_MIN_STEG_MM. The number of halvings and of
        floor steps is reported: it measures proximity to a singularity."""
        pa, pb = np.asarray(a["tcp"], float), np.asarray(b["tcp"], float)
        d = float(np.linalg.norm(pb - pa))
        qs = [np.asarray(a["q"], float)]
        if d < 1e-9:
            return np.array(qs), d, {"n_steg": 0, "n_halveringar": 0, "n_golvsteg": 0}
        pos, steg, n_halv, n_golv = 0.0, LIN_STEG_MM, 0, 0
        while pos < d - 1e-9:
            h = min(steg, d - pos)
            pt = pa + (pb - pa) * ((pos + h) / d)
            r = None
            for kand in [qs[-1]] + SEEDS:
                c = ik_robust(R_TOOLDOWN, pt, L_TOOL, [kand], forsta_duger=True)
                if c is not None:
                    r = c
                    if float(np.max(np.abs(c[0] - qs[-1]))) <= LIN_MAX_DQ_RAD:
                        break
                    r = None
            if r is not None:
                qs.append(r[0])
                pos += h
                steg = min(LIN_STEG_MM, steg * 2.0)
                continue
            if h <= LIN_MIN_STEG_MM + 1e-9:
                c = ik_robust(R_TOOLDOWN, pt, L_TOOL, [qs[-1]] + SEEDS, forsta_duger=True)
                if c is None:
                    raise SystemExit(f"LIN: no IK solution {a['namn']}->{b['namn']} @ {pt}")
                qs.append(c[0])
                pos += h
                n_golv += 1
                continue
            steg = h / 2.0
            n_halv += 1
        qb = qs[-1]
        Rb, pb2 = tcp_frame(qb, L_TOOL)
        b["q"] = qb
        b["tcp"] = pb2
        b["ik_res_mm"] = float(np.linalg.norm(pb - pb2))
        Rr = R_TOOLDOWN @ Rb.T
        c = max(-1.0, min(1.0, (np.trace(Rr) - 1.0) / 2.0))
        b["ik_res_deg"] = float(math.degrees(math.acos(c)))
        b["ledmarginal_deg"] = min(limit_margins(qb))
        return np.array(qs), d, {"n_steg": len(qs) - 1, "n_halveringar": n_halv, "n_golvsteg": n_golv}

    segs = []
    for a, b in zip(wp[:-1], wp[1:]):
        Qp, dist, lin_stat = lin_path(a, b)
        dQ_step = np.abs(np.diff(Qp, axis=0)) if len(Qp) > 1 else np.zeros((1, nj))
        dt_step = (dQ_step / (VMAX_RAD * V_FRAC)).max(axis=1)
        cum = np.concatenate([[0.0], np.cumsum(dt_step)])
        T_lin = float(cum[-1])
        segs.append({"fran": a["namn"], "till": b["namn"], "T": max(1.875 * T_lin, 0.20),
                     "a": a, "b": b, "path": Qp, "cum": cum, "T_lin": T_lin,
                     "stracka_mm": round(dist, 3), "lin": lin_stat})
        if b["typ"] == "kontakt":
            segs.append({"fran": b["namn"], "till": b["namn"], "T": DWELL_S, "a": b, "b": b,
                         "path": np.array([np.asarray(b["q"], float)]), "dwell": True,
                         "cum": np.array([0.0]), "T_lin": 0.0, "stracka_mm": 0.0})

    def path_at(Qp, cum, tau):
        """Joint value at PATH TIME tau (same unit as cum), linear between the LIN points."""
        if len(Qp) == 1 or cum[-1] <= 0:
            return Qp[-1]
        i = int(np.searchsorted(cum, tau, side="right") - 1)
        i = max(0, min(i, len(Qp) - 2))
        d = cum[i + 1] - cum[i]
        f = 0.0 if d <= 0 else (tau - cum[i]) / d
        return Qp[i] + (Qp[i + 1] - Qp[i]) * min(max(f, 0.0), 1.0)

    t_acc = 0.0
    T_LIST, Q_LIST, FAS_IDX = [], [], []
    fas_tabell = []
    for si, s_ in enumerate(segs):
        n = max(int(round(s_["T"] * HZ)), 1)
        t0 = t_acc
        for k in range(n):
            frac = (k + 1) / n
            T_LIST.append(t_acc + s_["T"] * frac)
            Q_LIST.append(path_at(s_["path"], s_["cum"], quintic(frac) * s_["T_lin"]))
            FAS_IDX.append(si)
        t_acc += s_["T"]
        fas_tabell.append({"index": si, "fran": s_["fran"], "till": s_["till"],
                           "typ": "dwell" if s_.get("dwell") else "LIN",
                           "stracka_mm": s_["stracka_mm"], "lin": s_.get("lin"),
                           "t0_s": round(t0, 6), "t1_s": round(t_acc, 6),
                           "varaktighet_s": round(s_["T"], 6)})
    T_LIST = [0.0] + T_LIST
    Q_LIST = [np.array(wp[0]["q"])] + Q_LIST
    FAS_IDX = [0] + FAS_IDX
    Tarr = np.array(T_LIST)
    Qarr = np.array(Q_LIST)
    N = len(Tarr)

    dQ = np.diff(Qarr, axis=0) / np.diff(Tarr)[:, None]
    ddQ = np.diff(dQ, axis=0) / np.diff(Tarr[:-1])[:, None] if N > 2 else np.zeros((1, nj))
    vmax_uppmatt = np.degrees(np.abs(dQ).max(axis=0))
    v_kvot = [float(vmax_uppmatt[j] / VMAX_DPS[j]) for j in range(nj)]

    # ---- payload windows: grasp at the pick dwell, release at the place dwell
    grepp_index = {s["fran"]: si for si, s in enumerate(segs) if s.get("dwell")}
    gods = []
    for c in cykler:
        i = c["i"]
        gods.append({"instans": f"amne_{i}", "del": AMNE["namn"], "massa_kg": AMNE["massa_kg"],
                     "storlek_mm": AMNE["storlek_mm"],
                     "t_grepp_s": fas_tabell[grepp_index[f"plock_{i}"]]["t0_s"],
                     "t_slapp_s": fas_tabell[grepp_index[f"lagg_{i}"]]["t1_s"],
                     "foralder_fore": c["in"]["nest"], "foralder_efter": c["ut"]["nest"],
                     "vilo_topp_centrum_fore_mm": c["plock_tcp"],
                     "vilo_topp_centrum_efter_mm": c["lagg_tcp"]})

    # ---- GATE 1: IK residual and limit margin over every waypoint
    max_ik_mm = max(w["ik_res_mm"] for w in wp)
    max_ik_deg = max(w["ik_res_deg"] for w in wp)
    marg = [limit_margins(w["q"]) for w in wp]
    min_marg = min(min(m) for m in marg)
    varsta_axel = int(np.argmin([min(m[j] for m in marg) for j in range(nj)]))

    # ---- GATE 2: OBB/SAT collision sweep over the whole sequence, plus a positive control
    q_ref = [math.radians(a) for a in man["render_pose_deg"]]
    F_ref = fk(q_ref, joints)
    parts = []
    for p in man["parts"]:
        L = link_of(p["subassembly"])
        wc = corners(p["bbox"])
        parts.append({"namn": p["part"], "link": L,
                      "local": None if L < 0 else [se3_inv_apply(F_ref[L], c) for c in wc],
                      "aabb_static": aabb_of(wc)})
    movable = [p for p in parts if p["link"] >= 1]
    static = [p for p in parts if p["link"] < 0]

    def part_wc(p, F):
        return [se3_apply(F[p["link"]], c) for c in p["local"]] if p["link"] >= 1 \
            else corners(p["aabb_static"])

    Fr = fk(list(wp[0]["q"]), joints)
    disable = set()
    for i, pa in enumerate(movable):
        for pb in movable[i + 1:] + static:
            if pb["link"] >= 1 and abs(pa["link"] - pb["link"]) < 2:
                continue
            if obb_overlap(part_wc(pa, Fr), part_wc(pb, Fr)):
                disable.add(tuple(sorted((pa["namn"], pb["namn"])))) 

    def svep(offset_mm):
        traffar = []
        for k in range(0, N, 2):
            q = Qarr[k]
            F = [(np.array(R), tuple(np.array(p) + np.array([0.0, 0.0, offset_mm])))
                 for R, p in fk(list(q), joints)]
            wcs = {p["namn"]: part_wc(p, F) for p in movable}
            aabbs = {n: aabb_of(w) for n, w in wcs.items()}
            for i, pa in enumerate(movable):
                for pb in movable[i + 1:] + static:
                    if pb["link"] >= 1 and abs(pa["link"] - pb["link"]) < 2:
                        continue
                    key = tuple(sorted((pa["namn"], pb["namn"])))
                    if key in disable:
                        continue
                    wb = wcs[pb["namn"]] if pb["link"] >= 1 else corners(pb["aabb_static"])
                    bb = aabbs[pb["namn"]] if pb["link"] >= 1 else pb["aabb_static"]
                    if aabb_overlap(aabbs[pa["namn"]], bb) and obb_overlap(wcs[pa["namn"]], wb):
                        traffar.append({"par": list(key), "sampel": k,
                                        "q_deg": [round(math.degrees(x), 3) for x in q]})
        return traffar

    kollision = svep(0.0)
    positiv = svep(-FALLBEVIS_MM)
    grind2 = {"n_kollisioner": len(kollision), "kollisioner": kollision[:20],
              "positiv_kontroll_forskjutning_mm": -FALLBEVIS_MM,
              "positiv_kontroll_n_kollisioner": len(positiv),
              "diskriminerande": bool(len(kollision) == 0 and len(positiv) > 0)}

    grindar = {
        "grind1_ik": {"max_ik_residual_mm": max_ik_mm, "max_ik_residual_deg": max_ik_deg,
                      "min_ledmarginal_deg": min_marg, "varsta_axel": varsta_axel,
                      "gron": bool(max_ik_mm < 1e-6 and max_ik_deg < 1e-6 and min_marg > 0.0)},
        "grind2_kollision": dict(grind2, gron=bool(grind2["diskriminerande"])),
        "grind3_hastighet": {"vmax_uppmatt_dps": [float(x) for x in vmax_uppmatt],
                             "vmax_datablad_dps": [float(x) for x in VMAX_DPS],
                             "kvot": v_kvot, "tak": V_FRAC,
                             "amax_uppmatt_dps2": [float(x) for x in np.degrees(np.abs(ddQ).max(axis=0))],
                             "gron": bool(max(v_kvot) <= V_FRAC + 1e-6)},
    }
    alla_grona = all(g["gron"] for g in grindar.values())

    rep.update({
        "sekvens": {"n_faser": len(fas_tabell), "n_vagpunkter": len(wp), "n_cykler": len(cykler),
                    "n_sampel": N, "langd_s": round(float(Tarr[-1]), 3), "hz": HZ,
                    "faser_sammanfattning": [{"index": f["index"], "fran": f["fran"], "till": f["till"],
                                              "typ": f["typ"], "t0_s": f["t0_s"], "t1_s": f["t1_s"]}
                                             for f in fas_tabell],
                    "lin_singularitetsmatt": {
                        "n_halveringar_totalt": sum((f["lin"] or {}).get("n_halveringar", 0) for f in fas_tabell),
                        "n_golvsteg_totalt": sum((f["lin"] or {}).get("n_golvsteg", 0) for f in fas_tabell)}},
        "vagpunkter": [{"namn": w["namn"], "typ": w["typ"],
                        "q_deg": [round(float(math.degrees(x)), 6) for x in w["q"]],
                        "tcp_mm": [round(float(x), 6) for x in w["tcp"]],
                        "ik_res_mm": w["ik_res_mm"], "ik_res_deg": w["ik_res_deg"],
                        "ledmarginal_deg": w["ledmarginal_deg"]} for w in wp],
        "gods": gods,
        "grindar": grindar,
        "alla_grindar_grona": alla_grona,
        "sanningsmarkning": {
            "MATT": ["joint origins, axes, limits and velocity limits (URDF)",
                     "the approach height (the highest reachable candidate for all stations)",
                     "collision geometry (URDF collision meshes, per-part AABB)"],
            "HARLEDD": ["every waypoint (Newton IK on the same fk)",
                        "the LIN paths and the quintic time law",
                        "the singularity measure (step halvings and floor steps)"],
            "KONSTRUERAT": [f"V_FRAC={V_FRAC}", f"DWELL_S={DWELL_S}", f"HZ={HZ}",
                            f"LIN_STEG_MM={LIN_STEG_MM}", f"LIN_MAX_DQ_RAD={LIN_MAX_DQ_RAD:.6f}",
                            f"FALLBEVIS_MM={FALLBEVIS_MM}", "the cell definition (synthetic)"]},
        "sekunder": round(time.time() - t_start, 2),
    })

    bana = {"namn": "robotcell_pickplace_v1", "urdf": man.get("urdf"), "hz": HZ,
            "n_rutor": N, "led_namn": jnames,
            "t": [round(float(x), 6) for x in Tarr],
            "q": [[round(float(math.degrees(x)), 6) for x in row] for row in Qarr],
            "fas_index": FAS_IDX, "gods": gods}

    if not args.verify:
        for path, obj in ((args.out, rep), (args.bana_out, bana)):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w") as f:
                json.dump(obj, f, indent=1)

    print(f"PICKPLACE {len(fas_tabell)} phases, {len(wp)} waypoints, {N} samples, "
          f"{float(Tarr[-1]):.2f} s, approach {APPROACH_MM} mm")
    print(f"  gate 1 IK: residual {max_ik_mm:.3e} mm / {max_ik_deg:.3e} deg, "
          f"min limit margin {min_marg:.3f} deg -> {'GREEN' if grindar['grind1_ik']['gron'] else 'RED'}")
    print(f"  gate 2 collision: {len(kollision)} along the path, positive control "
          f"{len(positiv)} -> {'GREEN' if grindar['grind2_kollision']['gron'] else 'RED'}")
    print(f"  gate 3 speed: max ratio {max(v_kvot):.4f} of the datasheet limit (cap {V_FRAC}) -> "
          f"{'GREEN' if grindar['grind3_hastighet']['gron'] else 'RED'}")
    if not args.verify:
        print(f"  -> {os.path.relpath(os.path.abspath(args.out), ROOT)}")
        print(f"  -> {os.path.relpath(os.path.abspath(args.bana_out), ROOT)}")
    return 0 if alla_grona else 1


if __name__ == "__main__":
    raise SystemExit(main())
