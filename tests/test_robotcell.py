"""Pytest wrappers for the robot-cell chain: kinematics, sweep, dynamics, pick-and-place."""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
MANIFEST = ROOT / "data" / "cells" / "ur10e_parts_manifest_v1.json"


def run(script, *args):
    r = subprocess.run([PY, str(ROOT / "scripts" / script), *args], cwd=ROOT,
                       capture_output=True, text=True)
    return r


def test_fk_matches_pinocchio():
    r = run("robotcell_leder_v1.py", "--check")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout


def test_parts_manifest_regenerates():
    r = run("urdf_parts_manifest.py", "--static", "data/cells/synthetic_cell_static_v1.json")
    assert r.returncode == 0, r.stdout + r.stderr
    man = json.loads(MANIFEST.read_text())
    assert man["schema"] == "parts_manifest_v1"
    assert sum(1 for p in man["parts"] if p["subassembly"].startswith("robot/")) == 7
    assert 30.0 < sum(p["mass_kg"] for p in man["parts"]) < 35.0


def test_bansvep_reference_pose_is_clean():
    """The sweep must not false-flag the reference pose."""
    r = run("robotcell_bansvep_v1.py", "--ref-only")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "GREEN_FULLENVELOPP" in r.stdout


def test_bansvep_full_envelope_is_not_collision_free():
    """A 6R arm folds into itself at the joint extremes; the sweep must report that, not hide it."""
    r = run("robotcell_bansvep_v1.py", "--combos", "40", "--steps", "7")
    assert r.returncode == 2, r.stdout + r.stderr
    assert "FULLENVELOPP_EJ_KOLLISIONSFRI_VANTAT_6R" in r.stdout


def test_obb_sat_separating_axis():
    sys.path.insert(0, str(ROOT / "scripts"))
    from robotcell_bansvep_v1 import corners, obb_overlap, aabb_overlap
    a = corners([0, 0, 0, 1, 1, 1])
    b = corners([2, 0, 0, 3, 1, 1])
    c = corners([0.5, 0.5, 0.5, 1.5, 1.5, 1.5])
    assert not obb_overlap(a, b)
    assert obb_overlap(a, c)
    assert not aabb_overlap([0, 0, 0, 1, 1, 1], [2, 0, 0, 3, 1, 1])


def test_pickplace_all_gates_green():
    r = run("robotcell_pickplace_v1.py")
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads((ROOT / "reports" / "robotcell_pickplace_v1.json").read_text())
    assert rep["sekvens"]["n_faser"] == 25
    assert rep["grindar"]["grind1_ik"]["max_ik_residual_mm"] < 1e-6
    assert rep["grindar"]["grind2_kollision"]["n_kollisioner"] == 0
    assert rep["grindar"]["grind2_kollision"]["positiv_kontroll_n_kollisioner"] > 0
    assert rep["alla_grindar_grona"]


def test_bansvep_on_the_planned_path_is_green():
    bana = ROOT / "data" / "banor" / "robotcell_pickplace_v1.json"
    if not bana.exists():
        pytest.skip("run robotcell_pickplace_v1.py first")
    r = run("robotcell_bansvep_v1.py", "--bana", str(bana), "--bana-steps", "1",
            "--out", "reports/robotcell_bansvep_bana_v1.json")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "GREEN_BANA" in r.stdout


def test_robotlaster_three_way_agreement_and_planted_faults():
    r = run("robotlaster_v1.py", "--tyst")
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads((ROOT / "reports" / "robotlaster_v1.json").read_text())
    k = rep["krysskoll"]
    assert k["rnea_pinocchio_vs_egen_newton_euler_max_Nm"] < 1e-9
    assert k["statik_pinocchio_vs_virtuellt_arbete_procent"] < 1e-6
    assert k["statik_newton_euler_vs_virtuellt_arbete_procent"] < 1e-6
    assert k["energibalans_residual_procent_av_absolut_arbete"] < 1.0
    f = rep["fallbevis"]
    assert f["f1_planterad_lankmassa"]["faller"]
    assert f["f2_nollad_gravitation"]["faller"]
    assert f["f3_planterad_last"]["faller"]
    assert f["f4_negativ_kontroll_paritet"]["paritet_bevarad"]
