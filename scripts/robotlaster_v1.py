#!/usr/bin/env python3
"""Inverse dynamics, friction and load margin per joint over a robot path.

What the cell does (no new geometry, no new path — load only):

  1. LINK INERTIA. Each part is read from the parts manifest. Mass, centre of mass and inertia
     tensor come from the manifest entry; when an entry carries a `step` key the exact BRep volume
     integral is used instead (`occ_inertials`, OCC, optional). Parts are aggregated per link via
     `link_of()` and expressed in each link's own frame at the reference pose.
  2. INVERSE DYNAMICS (RNEA) PER JOINT over the whole path, in TWO INDEPENDENT IMPLEMENTATIONS:
       A) pinocchio (pin.rnea) on a model built from the same joint chain,
       B) a world-frame Newton-Euler forward/backward sweep written in this file.
     A payload is attached to the last link inside the grasp windows.
  3. FRICTION: Coulomb (no-load term + gear efficiency) plus viscous, per joint, as a share of the
     total torque.
  4. CROSS-CHECKS: (a) statics by three routes (pinocchio RNEA, own Newton-Euler, virtual-work
     gravity), (b) energy balance over the path, (c) the datasheet effort limits from the URDF.
  5. FALSIFICATION: planted faults that must make named quantities fail.

Every quantity is declared MEASURED / DERIVED / ASSUMED in the report's `sanningsmarkning`.

I/O:
  --parts-manifest PATH   manifest JSON (default: data/cells/ur10e_parts_manifest_v1.json)
  --bana PATH             trajectory JSON {t: [s], q: [[deg]*n], gods: [{t_grepp_s, t_slapp_s, massa_kg}]}
                          (default: a synthetic quintic path generated in-process)
  --payload-kg F          payload mass when the trajectory declares no `gods` track
  --verify                recompute, write no files
  --fallbevis             run the planted faults only
  --out PATH              report JSON
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robotcell_leder_v1 import (ROOT, AXIS_EFFORT_NM, AXIS_SPEED_DPS, fk, joint_world,  # noqa: E402
                                load_chain, DEFAULT_URDF)
from robotcell_bansvep_v1 import link_of  # noqa: E402

CELL = "robotlaster_v1"
DEFAULT_MANIFEST = os.path.join(ROOT, "data", "cells", "ur10e_parts_manifest_v1.json")
G = 9.80665
GVEC = np.array([0.0, 0.0, -G])

# ---- ASSUMED model parameters (class-typical, not measured on this robot) ----
K_DYN = 2.5               # servo peak torque over continuous holding torque (industrial class)
K_DYN_BAND = (2.0, 3.0)   # two-sided band -> sensitivity band on the derived torque envelope
T_ACC_S = 0.20            # ramp time to maximum joint speed; used only for joint 1's dynamic envelope
ETA_VAXEL = 0.80          # gear efficiency (cycloidal/harmonic class) -> load-dependent Coulomb term
K_C0 = 0.02               # no-load Coulomb as a share of the envelope torque
K_V = 0.02                # viscous friction at maximum joint speed, as a share of the envelope torque
EPS_W = 0.05              # rad/s, tanh smoothing of the sign function (numerical, not physics)
SG_WIN, SG_ORD = 9, 3     # Savitzky-Golay window for derivatives of a 50 Hz path

_DRY = [False]
_STATE = {"joints": None, "nj": 6}


def wj(path, obj):
    if _DRY[0]:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: float(o) if isinstance(o, np.floating)
                  else (int(o) if isinstance(o, np.integer) else list(o)))
        f.write("\n")


def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def sg_coeffs(win, order, deriv, dt):
    """Savitzky-Golay filter coefficients (centre tap): smoothed derivative of order `deriv`."""
    half = win // 2
    z = np.arange(-half, half + 1, dtype=float)
    A = np.vander(z, order + 1, increasing=True)
    C = np.linalg.pinv(A)
    return C[deriv] * math.factorial(deriv) / (dt ** deriv)


def sg_filter(y, win, order, deriv, dt):
    """y: (N, m). Edges: polynomial fit on the edge window (no mirroring, the edges are dwells)."""
    N, m = y.shape
    half = win // 2
    c = sg_coeffs(win, order, deriv, dt)
    out = np.zeros_like(y)
    for i in range(N):
        lo = min(max(i - half, 0), N - win)
        seg = y[lo:lo + win]
        if i - half >= 0 and i + half < N:
            out[i] = c @ seg
        else:
            z = np.arange(win, dtype=float) - (i - lo)
            A = np.vander(z, order + 1, increasing=True)
            coef = np.linalg.pinv(A) @ seg
            out[i] = coef[deriv] * math.factorial(deriv) / (dt ** deriv) if deriv <= order else 0.0
    return out


def central_diff(y, dt):
    d = np.zeros_like(y)
    d[1:-1] = (y[2:] - y[:-2]) / (2 * dt)
    d[0] = (y[1] - y[0]) / dt
    d[-1] = (y[-1] - y[-2]) / dt
    return d


def occ_inertials(step_path):
    """(volume_mm3, com_mm[3], I_origin_mm5[3,3]) at unit density, exactly from the BRep integral.
    Only used when a manifest entry carries a `step` key; requires OCC."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.STEPControl import STEPControl_Reader
    r = STEPControl_Reader()
    if r.ReadFile(step_path) != 1:
        raise SystemExit(f"could not read STEP: {step_path}")
    r.TransferRoots()
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(r.OneShape(), g)
    c = g.CentreOfMass()
    M = g.MatrixOfInertia()
    I = np.array([[M.Value(i + 1, j + 1) for j in range(3)] for i in range(3)])
    return g.Mass(), np.array([c.X(), c.Y(), c.Z()]), I


def konventionsprov():
    """Falsification of the inertia convention: an analytically known box far from the origin decides
    whether OCC's MatrixOfInertia is about the ORIGIN or about the CENTRE OF MASS — the difference
    between a correct and a catastrophically wrong inertia. Skipped when OCC is absent."""
    try:
        from OCP.BRepGProp import BRepGProp
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCP.GProp import GProp_GProps
        from OCP.gp import gp_Pnt
    except ImportError:
        return {"konvention": None, "konvention_bevisad": None, "skipped": "OCC not installed"}
    a, b, c = 100.0, 40.0, 20.0
    off = (1000.0, 2000.0, 3000.0)
    s = BRepPrimAPI_MakeBox(gp_Pnt(*off), a, b, c).Shape()
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(s, g)
    V = g.Mass()
    M = g.MatrixOfInertia()
    I = np.array([[M.Value(i + 1, j + 1) for j in range(3)] for i in range(3)])
    ana_com = np.array([V * (b * b + c * c) / 12.0, V * (a * a + c * c) / 12.0, V * (a * a + b * b) / 12.0])
    com = np.array([off[0] + a / 2, off[1] + b / 2, off[2] + c / 2])
    ana_org = ana_com + V * np.array([com[1] ** 2 + com[2] ** 2, com[0] ** 2 + com[2] ** 2,
                                      com[0] ** 2 + com[1] ** 2])
    d_com = float(np.max(np.abs(np.diag(I) - ana_com) / ana_com))
    d_org = float(np.max(np.abs(np.diag(I) - ana_org) / ana_org))
    return {"lada_mm": [a, b, c], "offset_mm": list(off), "volym_mm3": V,
            "rel_avvikelse_om_masscentrum": d_com, "rel_avvikelse_om_origo": d_org,
            "konvention": "MASSCENTRUM" if d_com < d_org else "ORIGO",
            "konvention_bevisad": bool(d_com < 1e-9 and d_org > 1e-3)}


def bygg_lanktroghet(manifest, log):
    """Aggregated {mass, com, I_com} per link 0..n, in the WORLD frame at the reference pose."""
    nj = _STATE["nj"]
    per_lank = {i: {"m": 0.0, "mc": np.zeros(3), "I0": np.zeros((3, 3)), "delar": [], "n": 0,
                    "vol_mm3": 0.0} for i in range(nj + 1)}
    perpart = []
    vol_avvik = []
    t0 = time.time()
    F_ref = fk([math.radians(a) for a in manifest["render_pose_deg"]], _STATE["joints"])
    n_step = 0
    for p in manifest["parts"]:
        L = link_of(p.get("subassembly"))
        if L < 0:
            continue
        step = p.get("step")
        if step:                                   # exact BRep route
            rho = p["rho_kg_m3"]
            vol, com, I = occ_inertials(step if os.path.isabs(step) else os.path.join(ROOT, step))
            n_step += 1
            vman = p.get("volume_mm3")
            if vman:
                vol_avvik.append((abs(vol - vman) / max(vman, 1e-9), p["part"]))
            m = rho * vol * 1e-9
            Isi = rho * I * 1e-15                  # about the world origin, at the reference pose
            cm = com * 1e-3
            Icom = Isi - m * (float(cm @ cm) * np.eye(3) - np.outer(cm, cm))
        else:                                      # URDF inertial route
            m = float(p["mass_kg"])
            R = np.array(F_ref[L][0])
            o = np.array(F_ref[L][1]) * 1e-3
            cm = o + R @ np.array(p["com_link_m"], dtype=float)
            Icom = R @ np.array(p["inertia_com_kgm2"], dtype=float) @ R.T
        d = per_lank[L]
        d["m"] += m
        d["mc"] += m * cm
        d["I0"] += Icom + m * (float(cm @ cm) * np.eye(3) - np.outer(cm, cm))
        d["n"] += 1
        d["vol_mm3"] += float(p.get("volume_mm3") or 0.0)
        d["delar"].append(p["part"])
        perpart.append({"part": p["part"], "lank": L, "m": m, "c": [float(x) for x in cm],
                        "I": [[float(x) for x in r] for r in Icom]})
    log(f"    link inertia: {len(perpart)} parts ({n_step} via exact BRep integral) in "
        f"{time.time() - t0:.2f} s")
    return per_lank, {"max_rel_volymavvikelse_mot_manifest": max(vol_avvik)[0] if vol_avvik else 0.0,
                      "varsta_delen": max(vol_avvik)[1] if vol_avvik else None,
                      "n_delar_provade": len(vol_avvik)}, perpart


def aggregera(perpart, omtilldelning=None):
    """per-part integrals -> per-link aggregate (world frame). omtilldelning: {part_name: new_link}."""
    nj = _STATE["nj"]
    per_lank = {i: {"m": 0.0, "mc": np.zeros(3), "I0": np.zeros((3, 3)), "delar": [], "n": 0,
                    "vol_mm3": 0.0} for i in range(nj + 1)}
    for e in perpart:
        L = (omtilldelning or {}).get(e["part"], e["lank"])
        m = e["m"]
        cm = np.array(e["c"])
        d = per_lank[L]
        d["m"] += m
        d["mc"] += m * cm
        d["I0"] += np.array(e["I"]) + m * (float(cm @ cm) * np.eye(3) - np.outer(cm, cm))
        d["n"] += 1
        d["delar"].append(e["part"])
    return per_lank


def till_lankramar(per_lank, Fref):
    """World aggregate -> the link's own frame: origin = the joint origin at the reference pose,
    orientation = R_i."""
    ut = []
    for i in range(_STATE["nj"] + 1):
        d = per_lank[i]
        m = d["m"]
        if m <= 0:
            ut.append({"m": 0.0, "c": np.zeros(3), "I": np.zeros((3, 3)), "n": 0})
            continue
        c_w = d["mc"] / m
        I_com_w = d["I0"] - m * (float(c_w @ c_w) * np.eye(3) - np.outer(c_w, c_w))
        R = np.array(Fref[i][0])
        p = np.array(Fref[i][1]) * 1e-3
        ut.append({"m": m, "c": R.T @ (c_w - p), "I": R.T @ I_com_w @ R, "n": d["n"],
                   "c_varld": c_w, "I_com_varld": I_com_w, "vol_mm3": d["vol_mm3"]})
    return ut


def kedja(q):
    """(R[i], o[i]) i=0..n in metres, and the world joint axis z[i] i=1..n at joint angles q (rad)."""
    F = fk(list(q), _STATE["joints"])
    R = [np.array(f[0]) for f in F]
    o = [np.array(f[1]) * 1e-3 for f in F]
    # the joint axis is expressed in the parent frame AFTER the fixed rpy part of the joint origin
    from robotcell_leder_v1 import _mm, _rpy as _rpyf
    z = []
    Rc = np.eye(3)
    for i, (_t, rpy, ax) in enumerate(_STATE["joints"]):
        Rp = R[i] @ np.array(_rpyf(*rpy))
        zi = Rp @ np.array(ax, dtype=float)
        z.append(zi / np.linalg.norm(zi))
    return R, o, z


def newton_euler(q, dq, ddq, links, extra=None, g=GVEC):
    """tau[n] (Nm) plus the base reaction. links: [{m,c,I}] in link frames (index 0 = fixed base).
    extra: (m, c_local_in_last_link, I_local) added to the last link (the payload)."""
    nj = _STATE["nj"]
    R, o, z = kedja(q)
    L = [dict(x) for x in links]
    if extra is not None:
        m_e, c_e, I_e = extra
        d = L[nj]
        m_tot = d["m"] + m_e
        c_tot = (d["m"] * d["c"] + m_e * c_e) / m_tot
        I_tot = (d["I"] + d["m"] * (float((d["c"] - c_tot) @ (d["c"] - c_tot)) * np.eye(3)
                                    - np.outer(d["c"] - c_tot, d["c"] - c_tot))
                 + I_e + m_e * (float((c_e - c_tot) @ (c_e - c_tot)) * np.eye(3)
                                - np.outer(c_e - c_tot, c_e - c_tot)))
        L[nj] = {"m": m_tot, "c": c_tot, "I": I_tot}

    w = [np.zeros(3)] * (nj + 1)
    al = [np.zeros(3)] * (nj + 1)
    a_o = [-g] + [np.zeros(3)] * nj      # the base accelerates at -g, so gravity falls out
    a_c = [np.zeros(3)] * (nj + 1)
    c_w = [np.zeros(3)] * (nj + 1)
    Iw = [np.zeros((3, 3))] * (nj + 1)
    for i in range(1, nj + 1):
        zi = z[i - 1]
        w[i] = w[i - 1] + zi * dq[i - 1]
        al[i] = al[i - 1] + zi * ddq[i - 1] + np.cross(w[i - 1], zi * dq[i - 1])
        r_pj = o[i] - o[i - 1]
        a_o[i] = a_o[i - 1] + np.cross(al[i - 1], r_pj) + np.cross(w[i - 1], np.cross(w[i - 1], r_pj))
        c_w[i] = o[i] + R[i] @ L[i]["c"]
        r = c_w[i] - o[i]
        a_c[i] = a_o[i] + np.cross(al[i], r) + np.cross(w[i], np.cross(w[i], r))
        Iw[i] = R[i] @ L[i]["I"] @ R[i].T

    f = np.zeros(3)
    n = np.zeros(3)
    tau = np.zeros(nj)
    n_led = [None] * nj
    o_next = None
    for i in range(nj, 0, -1):
        Fi = L[i]["m"] * a_c[i]
        Ni = Iw[i] @ al[i] + np.cross(w[i], Iw[i] @ w[i])
        n_new = n + np.cross(c_w[i] - o[i], Fi) + Ni
        if o_next is not None:
            n_new = n_new + np.cross(o_next - o[i], f)
        f = f + Fi
        n = n_new
        tau[i - 1] = float(n @ z[i - 1])
        n_led[i - 1] = n.copy()
        o_next = o[i]
    f_arm = f.copy()
    f_bas = f_arm + L[0]["m"] * (-g)
    n_bas = (n + np.cross(o[1] - o[0], f_arm) + np.cross(R[0] @ L[0]["c"], L[0]["m"] * (-g)))
    return tau, {"f_N": f_bas, "n_Nm": n_bas, "n_led": n_led, "w": w, "al": al, "a_c": a_c,
                 "c_w": c_w, "Iw": Iw, "R": R, "o": o, "z": z, "L": L}


def gravitationsmoment_direkt(q, links, extra=None, g=GVEC):
    """INDEPENDENT ROUTE 3: purely static joint torque from virtual work, with no Newton-Euler
    recursion. tau_i = -sum_{j>=i} m_j * g . ( z_i x (c_j - o_i) )."""
    nj = _STATE["nj"]
    R, o, z = kedja(q)
    L = [dict(x) for x in links]
    if extra is not None:
        m_e, c_e, _ = extra
        d = L[nj]
        mt = d["m"] + m_e
        L[nj] = {"m": mt, "c": (d["m"] * d["c"] + m_e * c_e) / mt, "I": d["I"]}
    c_w = [None] + [o[i] + R[i] @ L[i]["c"] for i in range(1, nj + 1)]
    tau = np.zeros(nj)
    for i in range(1, nj + 1):
        s = 0.0
        for j in range(i, nj + 1):
            s += L[j]["m"] * float(g @ np.cross(z[i - 1], c_w[j] - o[i]))
        tau[i - 1] = -s
    return tau


def bygg_pin(links, extra=None, g=GVEC):
    """INDEPENDENT IMPLEMENTATION 1: a pinocchio model built from the same joint chain."""
    import pinocchio as pin
    from robotcell_leder_v1 import _rpy as _rpyf
    nj = _STATE["nj"]
    m = pin.Model()
    m.name = "robotlaster"
    m.gravity.linear = np.array(g)
    jid = 0
    for i, (t, rpy, ax) in enumerate(_STATE["joints"]):
        pl = pin.SE3(np.array(_rpyf(*rpy)), np.array(t) * 1e-3)
        jid = m.addJoint(jid, pin.JointModelRevoluteUnaligned(*[float(a) for a in ax]), pl, f"a{i+1}")
        d = links[i + 1]
        mm, cc, II = d["m"], d["c"], d["I"]
        if extra is not None and i == nj - 1:
            m_e, c_e, I_e = extra
            mt = mm + m_e
            ct = (mm * cc + m_e * c_e) / mt
            II = (II + mm * (float((cc - ct) @ (cc - ct)) * np.eye(3) - np.outer(cc - ct, cc - ct))
                  + I_e + m_e * (float((c_e - ct) @ (c_e - ct)) * np.eye(3) - np.outer(c_e - ct, c_e - ct)))
            mm, cc = mt, ct
        m.appendBodyToJoint(jid, pin.Inertia(mm, cc, II), pin.SE3.Identity())
    return m, m.createData()


def markmoment(links, k_dyn, limits_deg, payload_max_kg, c_last=None, t_acc=None, log=None):
    """tau_mark_i — a DERIVED capacity envelope, not a manufacturer table.

    Joints 2..n: tau_stat_i = max over the permitted workspace of the STATIC gravity torque with the
    maximum payload at the tool tip; tau_mark_i = k_dyn * tau_stat_i. The only assumed number is
    k_dyn (servo peak over continuous holding torque), reported as a two-sided band.
    Joint 1: a vertical first axis carries ZERO static gravity torque, so no static envelope exists.
    There, and only there, a dynamic envelope is used: the datasheet joint speed ramped over t_acc.
    It is the weaker anchor and is marked as such in the report.
    """
    nj = _STATE["nj"]
    if c_last is None:
        c_last = links[nj]["c"]
    extra = (payload_max_kg, c_last, np.zeros((3, 3)))
    grid = [np.radians(np.linspace(max(limits_deg[i][0], -190.0), min(limits_deg[i][1], 190.0),
                                   n)) for i, n in zip(range(1, 5), (13, 13, 5, 9))]
    stat = np.zeros(nj)
    lager = np.zeros(nj)
    pose = [None] * nj
    n = 0
    for q2 in grid[0]:
        for q3 in grid[1]:
            for q4 in grid[2]:
                for q5 in grid[3]:
                    q = np.zeros(nj)
                    q[1:5] = (q2, q3, q4, q5)
                    t = np.abs(gravitationsmoment_direkt(q, links, extra=extra))
                    _, aux_s = newton_euler(q, np.zeros(nj), np.zeros(nj), links, extra=extra)
                    n += 1
                    for i in range(nj):
                        if t[i] > stat[i]:
                            stat[i] = t[i]
                            pose[i] = [round(float(x), 3) for x in np.degrees(q)]
                        nm = float(np.linalg.norm(aux_s["n_led"][i]))
                        if nm > lager[i]:
                            lager[i] = nm
    tau = k_dyn * stat
    dyn1 = 0.0
    pose1 = None
    if t_acc:
        w_max = np.radians(AXIS_SPEED_DPS[:nj])
        a_max = w_max / t_acc
        for q2 in np.radians(np.linspace(max(limits_deg[1][0], -190.0), min(limits_deg[1][1], 190.0), 7)):
            for q3 in np.radians(np.linspace(max(limits_deg[2][0], -190.0), min(limits_deg[2][1], 190.0), 7)):
                q = np.zeros(nj)
                q[1], q[2] = q2, q3
                for sgn in (+1.0, -1.0):
                    t_, _ = newton_euler(q, sgn * w_max, sgn * a_max, links, extra=extra)
                    if abs(t_[0]) > dyn1:
                        dyn1 = abs(t_[0])
                        pose1 = [round(float(x), 3) for x in np.degrees(q)]
    if dyn1 > tau[0]:
        tau[0] = dyn1
        pose[0] = pose1
    if log:
        log(f"    capacity envelope: {n} static poses, k_dyn={k_dyn}, joint-1 dynamic {dyn1:.1f} Nm")
    return tau, pose, n, stat, lager


def syntetisk_bana(limits_deg, hz=50.0, n_seg=6, seed=3):
    """A quintic point-to-point path inside the joint limits: the synthetic stand-in for a recorded
    production path. Returns {t, q(deg), gods}."""
    nj = _STATE["nj"]
    rng = np.random.default_rng(seed)
    lo = np.array([max(l, -150.0) for l, _ in limits_deg[:nj]])
    hi = np.array([min(h, 150.0) for _, h in limits_deg[:nj]])
    knots = [lo + (hi - lo) * rng.uniform(0.3, 0.7, nj) for _ in range(n_seg + 1)]
    t, q = [], []
    tt = 0.0
    for a, b in zip(knots[:-1], knots[1:]):
        T = max(0.8, float(np.max(np.abs(b - a)) / 60.0))
        n = max(2, int(round(T * hz)))
        for k in range(n):
            u = k / n
            s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
            q.append(list(a + (b - a) * s))
            t.append(tt + k / hz)
        tt += n / hz
    q.append(list(knots[-1]))
    t.append(tt)
    return {"hz": hz, "n_rutor": len(t), "t": t, "q": q,
            "gods": [{"t_grepp_s": t[len(t) // 4], "t_slapp_s": t[3 * len(t) // 4], "massa_kg": 5.0}]}


def kor(args):
    t_start = time.time()
    _DRY[0] = bool(args.verify)
    LOG = (lambda s: print(s, flush=True)) if not args.tyst else (lambda s: None)

    manifest = json.load(open(args.parts_manifest))
    urdf = os.path.join(ROOT, manifest.get("urdf", os.path.relpath(DEFAULT_URDF, ROOT)))
    joints, limits, jnames = load_chain(urdf)
    _STATE["joints"] = joints
    _STATE["nj"] = nj = len(joints)
    limits_deg = manifest.get("axis_limits_deg", limits)

    LOG(f"[{CELL}] 1/8 link inertia ...")
    konv = konventionsprov()
    if konv.get("konvention_bevisad") is False:
        raise SystemExit(f"INERTIA CONVENTION PROBE FAILS: {konv}")
    per_lank, volprov, perpart = bygg_lanktroghet(manifest, LOG)
    Q_REF = np.radians(manifest["render_pose_deg"])
    Fref = fk(list(Q_REF), joints)
    links = till_lankramar(per_lank, Fref)
    robot_massa = sum(links[i]["m"] for i in range(nj + 1))
    arm_massa = sum(links[i]["m"] for i in range(1, nj + 1))

    bana = json.load(open(args.bana)) if args.bana else syntetisk_bana(limits_deg)
    t_arr = np.array(bana["t"], dtype=float)
    q_arr = np.radians(np.array(bana["q"], dtype=float))
    N = len(t_arr)
    dt = float(np.median(np.diff(t_arr)))
    gods = bana.get("gods") or [{"t_grepp_s": t_arr[0], "t_slapp_s": t_arr[-1],
                                 "massa_kg": args.payload_kg}]
    m_gods = float(gods[0].get("massa_kg", args.payload_kg))
    c_gods_l6 = np.array(gods[0].get("com_link_m", links[nj]["c"]), dtype=float)
    I_gods_l6 = np.array(gods[0].get("inertia_kgm2", (m_gods * 0.05 ** 2) * np.eye(3)), dtype=float)
    extra = (m_gods, c_gods_l6, I_gods_l6)
    LOG(f"    robot {robot_massa:.3f} kg (arm without base {arm_massa:.3f}) | payload {m_gods:.5f} kg")

    LOG(f"[{CELL}] 2/8 derivatives from the path ({N} samples @ {1/dt:.1f} Hz) ...")
    dq = sg_filter(q_arr, SG_WIN, SG_ORD, 1, dt)
    ddq = sg_filter(q_arr, SG_WIN, SG_ORD, 2, dt)
    dq_ra = central_diff(q_arr, dt)
    ddq_ra = central_diff(dq_ra, dt)
    brusfaktor = float(np.max(np.abs(ddq_ra)) / max(np.max(np.abs(ddq)), 1e-12))

    greppad = np.zeros(N, dtype=bool)
    for gd in gods:
        greppad |= (t_arr >= gd["t_grepp_s"]) & (t_arr <= gd["t_slapp_s"])

    LOG(f"[{CELL}] 3/8 inverse dynamics in TWO independent implementations ...")
    import pinocchio as pin
    m_fri, d_fri = bygg_pin(links)
    m_last, d_last = bygg_pin(links, extra=extra)
    tau_pin = np.zeros((N, nj))
    tau_ne = np.zeros((N, nj))
    bas_f = np.zeros((N, 3))
    bas_n = np.zeros((N, 3))
    Ek = np.zeros(N)
    Ep = np.zeros(N)
    fk_par = 0.0
    for k in range(N):
        ex = extra if greppad[k] else None
        mm, dd = (m_last, d_last) if greppad[k] else (m_fri, d_fri)
        tau_pin[k] = pin.rnea(mm, dd, q_arr[k], dq[k], ddq[k])
        tau_ne[k], aux = newton_euler(q_arr[k], dq[k], ddq[k], links, extra=ex)
        bas_f[k] = aux["f_N"]
        bas_n[k] = aux["n_Nm"]
        L = aux["L"]
        ek = ep = 0.0
        v_o = [np.zeros(3)] * (nj + 1)
        for i in range(1, nj + 1):
            v_o[i] = v_o[i - 1] + np.cross(aux["w"][i - 1], aux["o"][i] - aux["o"][i - 1])
            r = aux["c_w"][i] - aux["o"][i]
            v_c = v_o[i] + np.cross(aux["w"][i], r)
            ek += 0.5 * L[i]["m"] * float(v_c @ v_c) + 0.5 * float(aux["w"][i] @ (aux["Iw"][i] @ aux["w"][i]))
            ep += -L[i]["m"] * float(GVEC @ aux["c_w"][i])
        Ek[k], Ep[k] = ek, ep
        if k % 71 == 0:
            pin.forwardKinematics(mm, dd, q_arr[k])
            _, o6, _ = kedja(q_arr[k])
            p_pin = np.array(dd.oMi[nj].translation)
            fk_par = max(fk_par, float(np.linalg.norm(p_pin - o6[nj]) * 1e3))

    d_impl = np.abs(tau_pin - tau_ne)
    skala = np.maximum(np.abs(tau_pin).max(axis=0), 1e-9)
    impl_rel = float(np.max(d_impl / skala))
    LOG(f"    pinocchio vs own Newton-Euler: max |diff| {float(d_impl.max()):.3e} Nm "
        f"(rel {impl_rel:.3e}) | FK parity {fk_par:.3e} mm")

    LOG(f"[{CELL}] 4/8 capacity envelope (DERIVED) ...")
    payload_max = float(args.payload_max_kg)
    tau_mark, tau_mark_pose, n_svep, tau_stat, tau_lager = markmoment(
        links, K_DYN, limits_deg, payload_max, c_gods_l6, T_ACC_S, LOG)
    band = {}
    for tag, kd in (("lag", K_DYN_BAND[0]), ("hog", K_DYN_BAND[1])):
        b = kd * tau_stat
        b[0] = tau_mark[0]
        band[tag] = b

    LOG(f"[{CELL}] 5/8 friction ...")
    k_eff = 1.0 / ETA_VAXEL - 1.0
    w_max = np.radians(AXIS_SPEED_DPS[:nj])
    tau_c0 = K_C0 * tau_lager
    b_visk = K_V * tau_lager / w_max
    tau_fr = np.zeros((N, nj))
    for i in range(nj):
        coul = (tau_c0[i] + k_eff * np.abs(tau_pin[:, i])) * np.tanh(dq[:, i] / EPS_W)
        tau_fr[:, i] = coul + b_visk[i] * dq[:, i]
    tau_tot = tau_pin + tau_fr
    frikt_andel, frikt_topp = [], []
    for i in range(nj):
        num = float(np.mean(np.abs(tau_fr[:, i])))
        den = float(np.mean(np.abs(tau_tot[:, i])))
        frikt_andel.append(100.0 * num / den if den > 1e-12 else 0.0)
        kmax = int(np.argmax(np.abs(tau_tot[:, i])))
        dtop = abs(tau_tot[kmax, i])
        frikt_topp.append(100.0 * abs(tau_fr[kmax, i]) / dtop if dtop > 1e-12 else 0.0)

    LOG(f"[{CELL}] 6/8 cross-check (a): statics by three routes ...")
    stilla = (np.max(np.abs(dq), axis=1) < 1e-6) & (np.max(np.abs(ddq), axis=1) < 1e-4)
    idx_st = np.where(stilla)[0]
    if len(idx_st) == 0:
        idx_st = np.argsort(np.max(np.abs(dq), axis=1))[:20]
    st = []
    for k in idx_st:
        ex = extra if greppad[k] else None
        mm, dd = (m_last, d_last) if greppad[k] else (m_fri, d_fri)
        t_pin = pin.rnea(mm, dd, q_arr[k], np.zeros(nj), np.zeros(nj))
        t_ne, _ = newton_euler(q_arr[k], np.zeros(nj), np.zeros(nj), links, extra=ex)
        t_gr = gravitationsmoment_direkt(q_arr[k], links, extra=ex)
        sc = max(float(np.max(np.abs(t_pin))), 1e-9)
        st.append((float(np.max(np.abs(t_pin - t_gr)) / sc), float(np.max(np.abs(t_ne - t_gr)) / sc)))
    stat_pin_grav = 100.0 * max(x[0] for x in st)
    stat_ne_grav = 100.0 * max(x[1] for x in st)

    LOG(f"[{CELL}] 7/8 cross-check (b): energy balance ...")
    # The balance is evaluated per interval of CONSTANT payload state: attaching or releasing a
    # payload injects energy that no joint torque did work for, so an interval that straddles a grasp
    # event cannot close. Reported: the worst relative residual over the intervals.
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    P = np.sum(tau_pin * dq, axis=1)
    W = float(trap(P, t_arr))
    dE = float((Ek[-1] + Ep[-1]) - (Ek[0] + Ep[0]))
    W_abs = float(trap(np.abs(P), t_arr))
    W_fr = float(trap(np.sum(tau_fr * dq, axis=1), t_arr))
    brytpunkt = [0] + [k for k in range(1, N) if greppad[k] != greppad[k - 1]] + [N]
    e_rel = 0.0
    e_res = 0.0
    intervall = []
    for a, b in zip(brytpunkt[:-1], brytpunkt[1:]):
        if b - a < 5:
            continue
        Wi = float(trap(P[a:b], t_arr[a:b]))
        dEi = float((Ek[b - 1] + Ep[b - 1]) - (Ek[a] + Ep[a]))
        Wai = float(trap(np.abs(P[a:b]), t_arr[a:b]))
        ri = 100.0 * abs(Wi - dEi) / max(Wai, 1e-12)
        intervall.append({"i0": a, "i1": b, "greppad": bool(greppad[a]), "arbete_J": Wi,
                          "delta_energi_J": dEi, "residual_procent": ri})
        if ri > e_rel:
            e_rel, e_res = ri, Wi - dEi

    LOG(f"[{CELL}] 8/8 effort limits + planted faults ...")
    effort = np.array(AXIS_EFFORT_NM[:nj], dtype=float)
    tau_max = np.max(np.abs(tau_tot), axis=0)
    kvot_datablad = tau_max / effort
    kvot = tau_max / np.maximum(tau_mark, 1e-12)
    stat_max = float(np.max(tau_stat))
    kvot_def = [bool((i == 0 and tau_mark[0] > 0.0) or tau_stat[i] >= 0.01 * stat_max) for i in range(nj)]
    kvot_def_v = np.array([kvot[i] for i in range(nj) if kvot_def[i]])
    kvot_max_def = float(kvot_def_v.max()) if kvot_def_v.size else 0.0
    kvot_band = {t: [float(x) for x in (tau_max / np.maximum(band[t], 1e-12))] for t in band}

    fall = {}
    def plantera_lank(idx, faktor):
        lp = [dict(x) for x in links]
        lp[idx] = dict(lp[idx])
        lp[idx]["m"] = lp[idx]["m"] * faktor
        lp[idx]["I"] = lp[idx]["I"] * faktor
        tp = np.zeros((N, nj))
        for k in range(N):
            ex = extra if greppad[k] else None
            tp[k], _ = newton_euler(q_arr[k], dq[k], ddq[k], lp, extra=ex)
        return np.max(np.abs(tp), axis=0) / effort

    # F1: 10x link mass on link 3 must drive the effort ratio over 1
    svep = []
    faktor_faller = None
    kvot_faller = 0.0
    for fkt in (2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0):
        kv = float(np.max(plantera_lank(min(3, nj), fkt)))
        svep.append({"faktor": fkt, "max_kvot": kv, "faller": bool(kv > 1.0)})
        if faktor_faller is None and kv > 1.0:
            faktor_faller, kvot_faller = fkt, kv
    fall["f1_planterad_lankmassa"] = {"led": min(3, nj), "svep": svep,
                                      "faktor_som_faller": faktor_faller if faktor_faller else -1.0,
                                      "kvot_vid_fall": kvot_faller,
                                      "faller": bool(faktor_faller is not None)}
    # F2: zeroed gravity must break the three-way statics agreement
    t_g0 = gravitationsmoment_direkt(q_arr[idx_st[0]], links, extra=None, g=np.zeros(3))
    t_pin0 = pin.rnea(m_fri, d_fri, q_arr[idx_st[0]], np.zeros(nj), np.zeros(nj))
    sc0 = max(float(np.max(np.abs(t_pin0))), 1e-9)
    fall["f2_nollad_gravitation"] = {
        "statik_avvikelse_procent_nominell": stat_pin_grav,
        "statik_avvikelse_procent_planterad": float(100.0 * np.max(np.abs(t_pin0 - t_g0)) / sc0),
        "faller": bool(100.0 * np.max(np.abs(t_pin0 - t_g0)) / sc0 > 10.0)}
    # F3: the payload is swept until the effort ratio exceeds 1; the sweep must find such a factor,
    # and the ratio must be monotone in the factor
    svep3 = []
    faktor3 = None
    for fkt in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0):
        ex_f = (fkt * m_gods, c_gods_l6, fkt * I_gods_l6)
        tp = np.zeros((N, nj))
        for k in range(N):
            tp[k], _ = newton_euler(q_arr[k], dq[k], ddq[k], links, extra=ex_f if greppad[k] else None)
        kv = float(np.max(np.max(np.abs(tp), axis=0) / effort))
        svep3.append({"lastfaktor": fkt, "last_kg": fkt * m_gods, "max_kvot": kv, "faller": bool(kv > 1.0)})
        if faktor3 is None and kv > 1.0:
            faktor3 = fkt
    monoton = all(b["max_kvot"] >= a["max_kvot"] - 1e-9 for a, b in zip(svep3[:-1], svep3[1:]))
    fall["f3_planterad_last"] = {"svep": svep3, "kvot_nominell": float(np.max(kvot_datablad)),
                                 "lastfaktor_som_faller": faktor3 if faktor3 else -1.0,
                                 "monoton_i_lastfaktor": bool(monoton),
                                 "faller": bool(faktor3 is not None and monoton)}
    # F4: negative control — the two RNEA implementations must still agree after a random perturbation
    lp = [dict(x) for x in links]
    rng = np.random.default_rng(7)
    for i in range(1, nj + 1):
        lp[i]["m"] = lp[i]["m"] * float(rng.uniform(0.5, 1.5))
    m_p, d_p = bygg_pin(lp)
    dmax = 0.0
    for k in range(0, N, 13):
        a = pin.rnea(m_p, d_p, q_arr[k], dq[k], ddq[k])
        b, _ = newton_euler(q_arr[k], dq[k], ddq[k], lp)
        dmax = max(dmax, float(np.max(np.abs(a - b))))
    fall["f4_negativ_kontroll_paritet"] = {"max_diff_Nm": dmax, "paritet_bevarad": bool(dmax < 1e-6)}

    rep = {
        "cell": CELL,
        "urdf": manifest.get("urdf"),
        "n_sampel": N, "hz": round(1.0 / dt, 3), "n_leder": nj,
        "massa": {"robot_kg": robot_massa, "arm_utan_bas_kg": arm_massa, "last_kg": m_gods},
        "troghet": {"volymprov": volprov, "konventionsprov": konv,
                    "per_lank_massa_kg": [links[i]["m"] for i in range(nj + 1)]},
        "krysskoll": {
            "rnea_pinocchio_vs_egen_newton_euler_max_Nm": float(d_impl.max()),
            "rnea_relativ": impl_rel,
            "fk_paritet_mm": fk_par,
            "statik_pinocchio_vs_virtuellt_arbete_procent": stat_pin_grav,
            "statik_newton_euler_vs_virtuellt_arbete_procent": stat_ne_grav,
            "energibalans_residual_J": e_res,
            "energibalans_residual_procent_av_absolut_arbete": e_rel,
            "energibalans_per_intervall": intervall,
            "arbete_J": W, "delta_energi_J": dE, "friktionsarbete_J": W_fr,
            "derivat_brusfaktor_central_vs_savitzky_golay": brusfaktor},
        "moment": {
            "tau_max_Nm": [float(x) for x in tau_max],
            "tau_kapacitetsenvelopp_Nm": [float(x) for x in tau_mark],
            "kvot_mot_envelopp": [float(x) for x in kvot],
            "kvot_definierad": kvot_def, "max_kvot_definierad": kvot_max_def,
            "kvot_band": kvot_band,
            "effort_urdf_Nm": [float(x) for x in effort],
            "kvot_mot_effort_urdf": [float(x) for x in kvot_datablad],
            "friktionsandel_medel_procent": frikt_andel,
            "friktionsandel_vid_toppmoment_procent": frikt_topp},
        "basreaktion": {"Fv_max_N": float(np.abs(bas_f[:, 2]).max()),
                        "Mk_max_Nm": float(np.linalg.norm(bas_n[:, :2], axis=1).max())},
        "fallbevis": fall,
        "sanningsmarkning": {
            "MATT": ["link mass / com / inertia (URDF inertial blocks, or exact BRep integrals when the "
                     "manifest carries STEP paths)", "joint origins, axes, limits, effort and velocity "
                     "limits (URDF)", "the path (given or generated)"],
            "HARLEDD": ["joint torques (RNEA, two implementations)", "base reaction",
                        "capacity envelope tau_mark (static gravity envelope scaled by k_dyn)",
                        "energy balance residual"],
            "KONSTRUERAT": [f"k_dyn={K_DYN} (band {K_DYN_BAND})", f"eta_gear={ETA_VAXEL}",
                            f"K_C0={K_C0}", f"K_V={K_V}", f"eps_w={EPS_W} rad/s",
                            f"t_acc={T_ACC_S} s", f"payload_max={payload_max} kg"]},
        "sekunder": round(time.time() - t_start, 2),
    }
    wj(args.out, rep)
    LOG(f"  RNEA parity {float(d_impl.max()):.3e} Nm | statics 3-way {max(stat_pin_grav, stat_ne_grav):.3e} % | "
        f"energy residual {e_rel:.4f} % | max effort ratio {float(np.max(kvot_datablad)):.3f}")
    LOG("  planted faults: " + ", ".join(
        f"{k}=" + ("FALLS" if v.get("faller") else "HOLDS") if "faller" in v
        else f"{k}=" + ("PARITY-KEPT" if v["paritet_bevarad"] else "PARITY-LOST") for k, v in fall.items()))
    if not _DRY[0]:
        LOG(f"  -> {os.path.relpath(os.path.abspath(args.out), ROOT)}")
    ok = (float(d_impl.max()) < 1e-9 and e_rel < 1.0
          and all(v["faller"] for k, v in fall.items() if "faller" in v)
          and fall["f4_negativ_kontroll_paritet"]["paritet_bevarad"])
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts-manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--bana", help="trajectory JSON {t, q(deg), gods}")
    ap.add_argument("--payload-kg", type=float, default=5.0)
    ap.add_argument("--payload-max-kg", type=float, default=10.0)
    ap.add_argument("--verify", action="store_true", help="recompute, write no files")
    ap.add_argument("--fallbevis", action="store_true", help="run the planted faults only")
    ap.add_argument("--tyst", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "robotlaster_v1.json"))
    args = ap.parse_args()
    return kor(args)


if __name__ == "__main__":
    raise SystemExit(main())
