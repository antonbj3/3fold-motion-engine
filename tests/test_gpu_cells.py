"""GPU cells: the warp/newton modules and the friction-grip scripts, run as they ship.

Every test here needs a CUDA device (and, for the friction cells, `newton` + `torch`); each one runs the
module or script in a subprocess exactly as docs/RUNNING.md documents it and checks the cell's own verdict
line, so the gate stays the cell's, not the test's.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _cuda_available():
    try:
        import warp as wp
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


cuda_only = pytest.mark.skipif(not _cuda_available(), reason="needs a CUDA device")


def _run(args, timeout=600, env=None):
    e = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}:{ROOT / 'scripts'}")
    e.update(env or {})
    return subprocess.run([PY, "-u", *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout, env=e)


@cuda_only
@pytest.mark.parametrize("module", ["motion_engine.torch_dynamics", "motion_engine.torch_gravity",
                                    "motion_engine.fk_warp", "motion_engine.trajopt_warp",
                                    "motion_engine.world_model_warp", "motion_engine.manip.scene",
                                    "motion_engine.manip.grip", "motion_engine.manip.servo",
                                    "motion_engine.manip.sensors"])
def test_gpu_module_imports_clean(module):
    """These are libraries, not selftests: they must import and initialise on the GPU without error.
    `torch_dynamics` only imports as a module (`python -m`), not as a file path."""
    r = _run(["-m", module])
    assert r.returncode == 0, r.stderr[-2000:]


@cuda_only
def test_rnea_warp_selftest_holds_pinocchio_parity():
    r = _run(["-m", "motion_engine.rnea_warp"])
    assert r.returncode == 0, r.stderr[-2000:]
    assert "pinocchio parity holds" in r.stdout


@cuda_only
def test_rnea_warp_parity_script():
    r = _run(["scripts/rnea_warp_parity.py"])
    assert r.returncode == 0, r.stderr[-2000:]
    assert "-> PARITY" in r.stdout


@cuda_only
def test_sdf_contact_engine_selftest():
    r = _run(["-m", "motion_engine.contact_engine_sdf"])
    assert r.returncode == 0, r.stderr[-2000:]
    assert "rolling preserved" in r.stdout


@cuda_only
def test_newton_probe_friction_grip_dose_response():
    """Its own pre-registered P1/P2/P3 criteria (structure, grip at F>=8 N, force dependence)."""
    r = _run(["probes/probe_friction_grip.py"])
    assert r.returncode == 0, r.stderr[-3000:]
    for line in ("P1 structure", "P2 grip", "P3 force dependence"):
        assert f"{line}" in r.stdout
    assert "BLOCKED" not in r.stdout


@cuda_only
def test_newton_friction_pick_criteria():
    """The cell's own F1/F2/F3 criteria on a reduced world count (FG_W=32; the shipped default is 256)."""
    r = _run(["scripts/newton_friction_pick.py"], env={"FG_W": "32"})
    assert r.returncode == 0, r.stderr[-3000:]
    assert "FRICTION GRIP: all criteria pass" in r.stdout
