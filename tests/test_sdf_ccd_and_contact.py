"""Pytest wrappers for the tropical-SDF swept-CCD chain, the split-impulse engine, the identifiability
governor, the occlusion-safe planner and the URDF fleet gate."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(script, *args, timeout=300):
    return subprocess.run([PY, str(ROOT / "scripts" / script), *args], cwd=ROOT,
                          capture_output=True, text=True, timeout=timeout)


@pytest.mark.parametrize("script", [
    "tropical_sdf_backend_min_of_capsules_beats_voxel_3x_bytes_matched_penetration_so_arm100.py",
    "tropical_sdf_so_arm100_geometry_backend_v2_config_driven_posing_plus_contact_normal_gradient.py",
    "tropical_sdf_backend_computes_full_v6_contact_geometry_ladder_2jet_curvature_anisotropy_on_so_arm100.py",
    "v6_sigma_min_identifiability_climb_force_to_mean_to_spectrum_on_so_arm100_curvatures.py",
    "so_arm100_contact_sim_cost_from_tropical_sdf_backend_hertz_stiffness_sets_stable_dt_stability_verified.py",
])
def test_tropical_sdf_chain_runs(script):
    r = run(script)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


def test_swept_ccd_catches_what_endpoint_only_misses():
    r = run("tropical_sdf_swept_volume_continuous_collision_detection_min_over_trajectory.py")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "endpoint-only CCD is UNSOUND" in r.stdout
    assert "PASS (a,b,c,d)" in r.stdout


def test_trajectory_cert_rejects_the_colliding_segment_with_a_witness():
    r = run("collision_free_trajectory_cert_swept_ccd_plus_margin_binding_segment_witness.py")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "certified=True" in r.stdout
    assert "certified=False" in r.stdout
    assert "binding_segment=1" in r.stdout


def test_identifiability_governor_selftest():
    r = subprocess.run([PY, "-m", "motion_engine.identifiability_governor"], cwd=ROOT,
                       capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "PASS: all 6 cases correct" in r.stdout


def test_occlusion_safe_plan_reaches_goal_with_zero_hidden_collisions():
    r = run("motion_occlusion_safe_plan.py")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "ZERO GT-collisions" in r.stdout


def test_split_impulse_engine_validates():
    r = run("prbe_impulse_validate.py", timeout=400)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "FAIL" not in r.stdout
    assert r.stdout.count("PASS") >= 8


def test_urdf_fleet_physics_quality_gate_binds():
    r = run("fleet_urdf_physics_quality.py")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert "the gate BINDS" in r.stdout
    assert "gate runs on the fleet (165 URDF): True" in r.stdout


def test_collision_false_free_audit():
    r = run("collision_false_free_audit.py")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "SURFACE SAMPLING OK" in r.stdout


def test_retime_jerk_artefact_confirmed():
    r = run("retime_jerk_resolution_audit.py")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "ARTEFACT CONFIRMED AND FIXED" in r.stdout


def test_footprint_collision_and_humanoid_sequence_planner():
    for s in ("motion_footprint_collision.py", "hum_seq_planner_v0.py"):
        r = run(s)
        assert r.returncode == 0, s + r.stdout[-1500:] + r.stderr[-1500:]


def test_synthetic_friction_cache_and_band_cells_run():
    r = run("make_synthetic_friction_cache.py", "--n-train", "8000", "--n-test", "2000")
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    for s in ("friction_model_reality_gap_presliding_on_real_ur_robot_data.py",
              "friction_gap_is_dynamic_presliding_superres_on_right_axis_not_static.py",
              "deployable_friction_cert_error_band_validated_out_of_sample_regime_gated.py"):
        rr = run(s)
        assert rr.returncode == 0, s + rr.stdout[-1500:] + rr.stderr[-1500:]
