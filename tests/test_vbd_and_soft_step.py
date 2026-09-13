"""Pytest wrappers for the soft-step primitive, the midpoint-VBD backend and its validation cells."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run_script(script, *args, timeout=300):
    return subprocess.run([PY, str(ROOT / "scripts" / script), *args], cwd=ROOT,
                          capture_output=True, text=True, timeout=timeout)


def run_module(module, timeout=300):
    env = {"PYTHONPATH": str(ROOT / "src")}
    return subprocess.run([PY, "-m", module], cwd=ROOT, capture_output=True, text=True,
                          timeout=timeout, env={**__import__("os").environ, **env})


def test_soft_step_invariants():
    r = run_module("motion_engine.soft_step", timeout=60)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "invariants I-2/I-3 + clamp verified" in r.stdout


def test_soft_step_make_soft_partition_and_clamp():
    sys.path.insert(0, str(ROOT / "src"))
    from motion_engine.soft_step import (CONTACT_DAMPING_RATIO, CONTACT_SPEED, bias_velocity,
                                         contact_hertz, make_soft)
    s = make_soft(30.0, CONTACT_DAMPING_RATIO, 1 / 240)
    assert abs(s.mass_scale + s.impulse_scale - 1.0) < 1e-12
    rigid = make_soft(0.0, CONTACT_DAMPING_RATIO, 1 / 240)
    assert rigid.bias_rate == rigid.mass_scale == rigid.impulse_scale == 0.0
    assert contact_hertz(240.0, 30.0) == 30.0 and contact_hertz(80.0, 30.0) == 10.0
    soft = make_soft(contact_hertz(240.0), CONTACT_DAMPING_RATIO, 1 / 240)
    vb, ms, isc = bias_velocity(0.01, soft, True, 240.0)          # speculative: no soft scaling
    assert abs(vb - 2.4) < 1e-12 and ms == 1.0 and isc == 0.0
    assert abs(bias_velocity(-10.0, soft, True, 240.0)[0] + CONTACT_SPEED) < 1e-12   # clamp
    assert tuple(bias_velocity(-0.05, soft, False, 240.0)) == (0.0, 1.0, 0.0)        # relax


def test_midpoint_vbd_engine_selftest():
    r = run_module("motion_engine.midpoint_vbd_engine", timeout=120)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "SELFTEST PASS" in r.stdout


def test_midpoint_vbd_engine_satisfies_the_contact_engine_protocol():
    sys.path.insert(0, str(ROOT / "src"))
    from motion_engine.contact_engine import ContactEngine
    from motion_engine.midpoint_vbd_engine import MidpointVBDEngine
    assert isinstance(MidpointVBDEngine(), ContactEngine)


def test_midpoint_vbd_core_gates():
    r = run_script("midpoint_vbd_core.py", timeout=120)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "ALL_PASS = True" in r.stdout


def test_energy_drift_is_symplectic_where_the_eulers_drift():
    r = run_script("vbd_energy_drift_symplectic.py", timeout=200)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "ALL_PASS = True" in r.stdout
    assert "[PASS] G3_eulers_DO_drift_null" in r.stdout      # the null is exhibited, not assumed


def test_nt2_within_step_sliding_impact_gate():
    r = run_script("vbd_nt2_within_step_sliding_impact_gate.py", timeout=200)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "NT2_PASS: True" in r.stdout


def test_nt_friction_gate_result():
    r = run_script("vbd_nt_friction_gate_result.py", timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "ALL_PASS = True" in r.stdout
    assert "[PASS] G1_stick_slide_transition_at_atan_mu" in r.stdout
    assert "[PASS] G4_sliding_accel_matches_coulomb_law" in r.stdout


def test_symplectic_scheme_crux_orders_and_iteration_scaling():
    r = run_script("symplectic_vbd_scheme_crux.py", timeout=400)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "MID   order=2.00" in r.stdout


def test_soft_step_wiring_default_is_off_and_opt_in_runs():
    """The soft flag is opt-in: the default engine keeps the Baumgarte position pass, and the soft
    path settles a shallow K=2 stack to the same resting state (the two paths only separate on the
    deep stack measured by the metrics harness)."""
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    from motion_engine.contact_engine import SplitImpulseEngine

    e0 = SplitImpulseEngine(mu=0.5)
    assert e0.soft is False and e0._warm == {}
    pens = {}
    for soft in (False, True):
        e = SplitImpulseEngine(mu=0.5, soft=soft)
        for k in range(2):
            e.add_body(0.3, 0.3, 0.2, [0, 0, 0.101 + k * 0.205])
        for _ in range(150):
            e.step(1 / 240, substeps=1)
        z = np.sort(e.get_state().xc[:, 2])
        ke = sum(0.5 * b.M * b.vc @ b.vc for b in e.B)
        assert ke < 1e-2, (soft, ke)
        assert z[0] > 0.09, (soft, z)
        pens[soft] = 0.205 - (z[1] - z[0])
    assert abs(pens[True] - pens[False]) < 0.2 * pens[False], pens
    assert SplitImpulseEngine(mu=0.5, soft=True)._warm == {}


def test_engine_metrics_harness_quick():
    """Runs the metrics harness on the short scenes and checks the shape of the report."""
    import json
    r = run_script("engine_metrics.py", "--quick", timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "M1 stack_load_K4" in r.stdout and "M5 stack8 pen/R" in r.stdout
    d = json.loads((ROOT / "reports" / "engine_metrics.json").read_text())
    assert set(d["engines"]) >= {"cpu_v0", "cpu_soft"}
    for m in d["engines"].values():
        for key in ("M1_stack_load_K4", "M4_iters_to_tol", "M5_stack8_pen", "M6_determinism"):
            assert key in m
        assert m["M6_determinism"]["pass"] is True
    # the soft path removes the deep-stack penetration the fixed-Baumgarte path leaves
    assert (d["engines"]["cpu_soft"]["M5_stack8_pen"]["max_pen_over_R"]
            < d["engines"]["cpu_v0"]["M5_stack8_pen"]["max_pen_over_R"])
