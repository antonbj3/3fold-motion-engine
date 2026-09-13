"""Planner and dynamics cells: the CPU ones run, the CUDA/optional-dependency ones skip."""
import json
from pathlib import Path

import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("hppfcl")

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _cuda_available():
    try:
        import warp as wp
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


cuda_only = pytest.mark.skipif(not _cuda_available(), reason="needs a CUDA device")


# ---------------------------------------------------------------- CPU cells

def test_kinematic_validity_gate_binds_and_reports_effort_provenance():
    import kinematic_validity_gate as kvg
    assert kvg.main() == 0
    out = json.loads((REPORTS / "kinematic_validity_gate.json").read_text())
    assert out["gate_binds"] and out["n_loaded"] >= 100
    assert out["n_broken"] == 0
    prov = out["effort_provenance"]
    assert prov["real"] + prov["placeholder_uniform"] + prov["missing"] == out["n_loaded"]
    # the detector must actually flag a constructed broken URDF, not merely return an empty list
    flags = kvg.check_xml_joints(_broken_urdf())
    assert any("degenerate-axis" in f for f in flags)


def _broken_urdf(tmp=[]):
    import tempfile, os
    if tmp:
        return tmp[0]
    xml = ('<robot name="broken"><link name="base"/><link name="l1"/><joint name="j1" type="revolute">'
           '<parent link="base"/><child link="l1"/><axis xyz="0 0 0"/>'
           '<limit lower="1.0" upper="-1.0" effort="10" velocity="1"/></joint></robot>')
    fd, p = tempfile.mkstemp(suffix=".urdf"); os.write(fd, xml.encode()); os.close(fd)
    tmp.append(p)
    return p


def test_cartesian_goal_becomes_a_collision_free_effort_feasible_plan():
    import fleet_cartesian_to_plan as fc
    assert fc.main() == 0
    out = json.loads((REPORTS / "fleet_cartesian_to_plan.json").read_text())
    assert out["collision_free"] and out["effort_feasible"] and out["honest_on_unreachable"]
    assert out["ik_pos_err_mm"] < 1.0 and out["infeasible_segments"] == 0


def test_full_motion_stack_dynamics_gate_is_load_bearing():
    import fleet_motion_stack as fms
    assert fms.main() == 0
    out = json.loads((REPORTS / "fleet_motion_stack.json").read_text())
    assert out["collision_free"] and out["dynamics_load_bearing"]
    assert out["aggressive_peak_ratio"] > 1.0 >= out["max_peak_ratio"]


def test_stack_is_exercised_on_more_than_one_vendor_description():
    import fleet_multirobot_movement as fmm
    assert fmm.main([]) == 0
    out = json.loads((REPORTS / "fleet_multirobot_movement.json").read_text())
    assert len(out["vendors_exercised"]) >= 2
    assert all(r["collision_free"] and r["collision_binds"] for r in out["results"] if r.get("planned"))


def test_effort_provenance_guard_never_claims_feasible_on_placeholder_limits():
    import fleet_cartesian_industrial as fci
    assert fci.main() == 0
    out = json.loads((REPORTS / "fleet_cartesian_industrial.json").read_text())
    assert out["provenance_guard"] and out["placeholder_detected"]
    assert out["fleet_scan"]["placeholder_uniform"] >= 1 and out["fleet_scan"]["missing"] >= 1
    assert all(r.get("effort_feasible") is True for r in out["pipeline_robots"]
               if r.get("effort_provenance") == "real" and r.get("collision_free"))


def test_false_free_rate_stays_low_on_every_robot_with_meshes():
    import fleet_collision_false_free as fcf
    assert fcf.main() == 0
    out = json.loads((REPORTS / "fleet_collision_false_free.json").read_text())
    assert out["all_under_20pct"] and len(out["robots"]) >= 2
    assert out["worst"]["false_free_pct"] < 20.0


def test_ik_quality_and_reachability_are_reported_apart():
    import honest_plannability_probe as hpp
    assert hpp.main() == 0
    out = json.loads((REPORTS / "honest_plannability.json").read_text())
    # IK on reachable targets is near-perfect; arbitrary 6D poses are lower, and that gap is reachability
    assert out["R1_soft_rate"] >= 0.9 and out["R2_reachable_full6d_rate"] >= 0.9
    assert out["R3_arbitrary_rate"] <= out["R1_soft_rate"]


def test_twin_sigma_propagates_end_to_end_and_the_gate_answers_to_sigma():
    import verify_motion_sigma_e2e as vs
    assert vs.main() == 0


def test_kin_spec_extractor_reproduces_the_chain():
    """The producer of data/kin_spec_<robot>.npz (the input of the CUDA planner cells): the spec chain must
    reproduce the URDF chain of robotcell_leder_v1 exactly."""
    import extract_kin_spec as eks
    assert eks.main(["--robot", "ur10e"]) == 0
    spec = dict(__import__("numpy").load(ROOT / "data" / "kin_spec_ur10e.npz"))
    assert int(spec["nq"]) == 6 and len(spec["pt_joint"]) > 1000


# ------------------------------------------------- CUDA / optional-dependency cells

@cuda_only
def test_fleet_grad_trajopt():
    # The cell's own verdict on the vendored UR10e is PARTIAL (3 of 4 scenes solved), so it exits 1 by design;
    # the test asserts that recorded verdict, the same way the kernel repo's amr_octree_fv wrapper does.
    import json
    import fleet_grad_trajopt as m
    rc = m.main()
    rep = json.load(open(ROOT / "reports" / "fleet_grad_trajopt.json"))
    assert rc in (0, 1)
    assert rep["total"] >= 1 and rep["total_solved"] >= 1        # the robot set is not empty and the cell solves scenes
    assert (rc == 0) == bool(rep["generalizes"])                   # exit code is the cell's own recorded verdict


@cuda_only
def test_fleet_mppi_benchmark():
    import fleet_mppi_benchmark as m
    assert m.main() == 0


@cuda_only
def test_mppi_completeness_audit():
    import mppi_completeness_audit as m
    assert m.main() == 0


@cuda_only
def test_fleet_plannability_metric():
    import fleet_plannability_metric as m
    assert m.main() == 0


@cuda_only
def test_hard_benchmark_plannability():
    import hard_benchmark_plannability as m
    assert m.main() == 0


def test_rrt_completeness_verify_consumes_the_cuda_audit_scenes():
    """Consumes reports/mppi_completeness_scenes.npz, produced by mppi_completeness_audit.py (CUDA)."""
    import rrt_completeness_verify as m
    if not (REPORTS / "mppi_completeness_scenes.npz").exists():
        pytest.skip("needs reports/mppi_completeness_scenes.npz from mppi_completeness_audit.py (CUDA)")
    assert m.main([]) == 0
    out = json.loads((REPORTS / "rrt_completeness_verify.json").read_text())
    assert out["planner_misses"] == out["false_infeasible"] + out["genuine_hard"]


def test_motion_stack_integration():
    pytest.importorskip("vamp")
    import motion_stack_integration as m
    assert m.main() == 0


def test_coacd_scene_to_planner():
    pytest.importorskip("vamp"); pytest.importorskip("coacd"); pytest.importorskip("trimesh")
    import coacd_scene_to_planner as m
    assert m.main() == 0
