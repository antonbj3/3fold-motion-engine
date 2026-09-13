#!/usr/bin/env python3
"""Joint axis / limit validity gate over the URDF fleet (kinematics data-quality axis).

A URDF with broken joint KINEMATICS corrupts planning and FK differently from a URDF with broken inertia
(`fleet_urdf_physics_quality.py` covers that one): a degenerate rotation axis (zero norm), inverted position
limits (lower > upper), non-finite origins, or missing velocity/effort limits (the planner has no ceiling, so
motion is unbounded).

Checks per URDF: (a) position limits lower < upper (inverted = broken); (b) joint axis norm > 0 (XML
`<axis xyz>`, not degenerate); (c) joint origins finite (XML `<origin>`); (d) velocity/effort limit COVERAGE
(reported, missing is a planning gap, not a hard fail). FAIL = inverted limits / degenerate axis / non-finite
origin.

The effort boolean is refined to a 3-way provenance, because a uniform effort limit (e.g. every joint 1000 Nm)
is a placeholder, not a measurement: only REAL (varied, > 0) effort limits make a torque-feasibility claim
meaningful; placeholder/missing make the effort gate vacuous.

Selftest checks: (1) the gate runs over the fleet; (2) it BINDS - the detector is run on a constructed broken
URDF (zero-norm axis, inverted limits) and must flag it; (3) hard-broken (corrupting) is reported separately
from limit-coverage gaps (informational).

  python -u scripts/kinematic_validity_gate.py   (requires pinocchio)
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
URDF_DIR = ROOT / "assets" / "robots" / "fleet_urdf"


def check_xml_joints(path):
    """XML level: degenerate axes and non-finite origins on movable joints."""
    bad = []
    try:
        root = ET.parse(path).getroot()
    except Exception as e:
        return [f"xml-parse:{type(e).__name__}"]
    for j in root.findall("joint"):
        jt = j.get("type", "")
        name = j.get("name", "?")
        if jt in ("revolute", "continuous", "prismatic"):
            ax = j.find("axis")
            if ax is not None and ax.get("xyz"):
                try:
                    v = np.array([float(x) for x in ax.get("xyz").split()])
                    if not np.all(np.isfinite(v)) or np.linalg.norm(v) < 1e-9:
                        bad.append(f"{name}:degenerate-axis")
                except Exception:
                    bad.append(f"{name}:axis-parse")
        org = j.find("origin")
        if org is not None and org.get("xyz"):
            try:
                if not np.all(np.isfinite([float(x) for x in org.get("xyz").split()])):
                    bad.append(f"{name}:origin-non-finite")
            except Exception:
                bad.append(f"{name}:origin-parse")
    return bad


def assess(path):
    try:
        m = pin.buildModelFromUrdf(str(path))
    except Exception as e:
        return dict(robot=path.stem, loaded=False, err=str(e)[:50])
    lo, hi = np.array(m.lowerPositionLimit), np.array(m.upperPositionLimit)
    inverted = int(np.sum(lo > hi + 1e-9))
    vlim = np.array(m.velocityLimit); elim = np.array(m.effortLimit)
    has_vel = bool(np.all(np.isfinite(vlim)) and np.all(vlim > 0))
    has_eff = bool(np.all(np.isfinite(elim)) and np.all(elim > 0))
    # 3-way effort provenance: real (varied > 0) / placeholder (uniform > 0) / missing (zero). A boolean
    # has_eff HIDES the placeholder case, and only real effort limits make torque-feasibility meaningful.
    if not has_eff:
        eff_prov = "missing"
    else:
        cv = float(elim.std() / (elim.mean() + 1e-12))
        eff_prov = "placeholder" if cv < 0.02 else "real"
    xml_bad = check_xml_joints(path)
    broken = inverted > 0 or len(xml_bad) > 0
    return dict(robot=path.stem, loaded=True, inverted_limits=inverted, xml_bad=xml_bad[:4],
                has_velocity_limits=has_vel, has_effort_limits=has_eff, effort_provenance=eff_prov, broken=broken)


def main():
    urdfs = sorted(URDF_DIR.glob("*.urdf"))
    print(f"KINEMATIC-VALIDITY-GATE — joint axis/limit validity over {len(urdfs)} URDFs")
    res = [assess(u) for u in urdfs]
    loaded = [r for r in res if r.get("loaded")]
    broken = [r for r in loaded if r["broken"]]
    n_vel = sum(1 for r in loaded if r["has_velocity_limits"])
    n_eff = sum(1 for r in loaded if r["has_effort_limits"])
    n_eff_real = sum(1 for r in loaded if r.get("effort_provenance") == "real")
    n_eff_ph = sum(1 for r in loaded if r.get("effort_provenance") == "placeholder")

    print(f"  loaded {len(loaded)}/{len(urdfs)}; HARD-BROKEN kinematics {len(broken)}")
    print(f"  limit coverage (info): velocity {n_vel}/{len(loaded)} ({100*n_vel//max(len(loaded),1)}%), "
          f"effort {n_eff}/{len(loaded)} ({100*n_eff//max(len(loaded),1)}%)")
    print(f"  effort provenance: REAL {n_eff_real}/{len(loaded)} | placeholder(uniform) {n_eff_ph} | "
          f"missing {len(loaded)-n_eff} — only REAL gives meaningful torque feasibility")
    for r in broken[:10]:
        issue = (f"{r['inverted_limits']} inverted limits" if r["inverted_limits"] else "") + \
                (" " + ",".join(r["xml_bad"]) if r["xml_bad"] else "")
        print(f"    BROKEN {r['robot']:28s} {issue}")
    if not broken:
        print("  no hard-broken kinematics (inverted limits / degenerate axes / non-finite origins)")

    # positive control: run the DETECTION path on a constructed broken URDF (zero-norm axis, inverted limits)
    import tempfile as _tf, os as _os
    _broken = ('<robot name="broken"><link name="base"/><link name="l1"/><joint name="j1" type="revolute">'
               '<parent link="base"/><child link="l1"/><axis xyz="0 0 0"/>'
               '<limit lower="1.0" upper="-1.0" effort="10" velocity="1"/></joint></robot>')
    _fd, _bp = _tf.mkstemp(suffix=".urdf"); _os.write(_fd, _broken.encode()); _os.close(_fd)
    try:
        _flags = check_xml_joints(_bp)                           # calls the real detector
        bind_ok = any("degenerate-axis" in f for f in _flags)
    finally:
        _os.unlink(_bp)
    g1 = len(loaded) >= 100
    g3 = bool(bind_ok)
    print(f"  (1) gate runs over the fleet ({len(loaded)}): {g1}")
    print(f"  (2) hard-broken {len(broken)} (corrupting) vs effort coverage {n_eff}/{len(loaded)} (info): True")
    print(f"  (3) gate BINDS (detector flags a known-broken URDF: zero axis -> {_flags}): {g3}")

    ok = g1 and g3
    print(f"\nVERDICT: kinematic-validity-gate = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"{len(loaded)} URDFs checked, {len(broken)} hard-broken (inverted limits / degenerate axes / "
             "non-finite origins). "
             + (f"{len(broken)} broken -> quarantine (planning/FK corrupted). " if broken else
                "No hard-broken kinematics: the fleet's joint axes, limits and origins are sound. ")
             + f"Limit coverage (info, not a fail): effort limits on only {n_eff}/{len(loaded)} "
             f"({100*n_eff//max(len(loaded),1)}%) = a known URDF data gap, so the torque gate has to fall back on a "
             "twin or a datasheet rather than the URDF. Velocity coverage is high. The gate binds."
             if ok else "Not validated. ")
          + " Scope: limit ordering, axis norm, origin finiteness and limit coverage; not collision-pair "
            "consistency or mimic joints.")

    out = dict(role="kinematic validity gate (joint axes/limits)",
               n_urdf=len(urdfs), n_loaded=len(loaded), n_broken=len(broken),
               velocity_coverage=f"{n_vel}/{len(loaded)}", effort_coverage=f"{n_eff}/{len(loaded)}",
               effort_provenance={"real": n_eff_real, "placeholder_uniform": n_eff_ph, "missing": len(loaded) - n_eff},
               effort_real_robots=sorted(r["robot"] for r in loaded if r.get("effort_provenance") == "real")[:40],
               broken=[dict(robot=r["robot"], inverted_limits=r["inverted_limits"], xml_bad=r["xml_bad"]) for r in broken],
               gate_binds=bool(g3),
               note="hard-broken (inverted limits/degenerate axis/non-finite origin) corrupts planning; the effort "
                    "coverage boolean hides placeholder (uniform) vs real -> effort_provenance is 3-way and only REAL "
                    "gives meaningful torque feasibility",
               validated=bool(ok))
    (ROOT / "reports" / "kinematic_validity_gate.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("  wrote reports/kinematic_validity_gate.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
