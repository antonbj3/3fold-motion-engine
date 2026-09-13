"""Contact resolution, the solver router and the grasp force-closure certificate (CPU only)."""
import numpy as np

from motion_engine.contact_engine import SplitImpulseEngine, ContactEngine, BodyState
from motion_engine.federation import Federation, SceneSpec
from motion_engine.grasp_wrench_cert import grasp_force_closure_cert, _selftest as grasp_selftest

G = 9.81


def test_contact_engine_selftest_passes():
    from motion_engine.contact_engine import _selftest
    assert _selftest()


def test_reference_backend_satisfies_the_protocol():
    assert isinstance(SplitImpulseEngine(), ContactEngine)


def test_stick_and_slide_at_the_friction_angle():
    verdicts = {}
    for deg in (20, 35):
        e = SplitImpulseEngine(ramp_deg=deg, mu=0.5)
        e.add_body(0.3, 0.2, 0.2, [0, 0, 0.101])
        x0 = e.com(0).copy()
        for _ in range(400):
            e.step(1 / 240, substeps=2)
        down = np.array([-np.cos(np.radians(deg)), 0, -np.sin(np.radians(deg))])
        verdicts[deg] = (e.com(0) - x0) @ down
    assert verdicts[20] < 0.02      # below atan(0.5) = 26.6 deg it sticks
    assert verdicts[35] > 0.02      # above it slides


def test_contact_force_equals_weight_on_flat_ground():
    e = SplitImpulseEngine(mu=0.8)
    e.add_body(0.3, 0.2, 0.2, [0, 0, 0.101])
    for _ in range(300):
        e.step(1 / 240, substeps=1)
    Fz = e.contact_forces()[0, 2]
    Mg = e.B[0].M * G
    assert abs(Fz - Mg) / Mg < 0.03


def test_state_is_returned_as_plain_data():
    e = SplitImpulseEngine()
    e.add_body(0.3, 0.2, 0.2, [0, 0, 0.101])
    e.step(1 / 240)
    st = e.get_state()
    assert isinstance(st, BodyState)
    assert st.xc.shape == (1, 3) and st.Rm.shape == (1, 3, 3)
    assert np.isfinite(st.xc).all() and np.isfinite(st.vc).all()


def test_router_selects_the_expected_specialists():
    from motion_engine.federation import _selftest
    assert _selftest()


def test_router_reports_which_backends_are_wired():
    f = Federation()
    sel = f.route(SceneSpec(n_bodies=8, shapes=("box",), coupling_depth=8))
    assert sel.member.name == "relaxed_jacobi_gpu"
    assert sel.score > 0.0 and sel.rationale
    assert any(m.is_wired() for m in f.members)


def test_grasp_certificate_selftest_passes():
    assert grasp_selftest()


def test_grasp_certificate_discriminates_enclosing_from_one_sided():
    enclosing = [([1, 0, 0], [-1, 0, 0]), ([-1, 0, 0], [1, 0, 0]),
                 ([0, 1, 0], [0, -1, 0]), ([0, -1, 0], [0, 1, 0]),
                 ([0, 0, 1], [0, 0, -1]), ([0, 0, -1], [0, 0, 1])]
    one_sided = [([1, y, z], [-1, 0, 0]) for y in (-0.3, 0.3) for z in (-0.3, 0.3)]
    r_enc = grasp_force_closure_cert(enclosing, mu=0.5)
    r_one = grasp_force_closure_cert(one_sided, mu=0.5)
    assert r_enc["force_closure"] and r_enc["epsilon_quality"] > 0
    assert not r_one["force_closure"] and r_one["epsilon_quality"] == 0.0


def test_grasp_certificate_abstains_below_two_contacts():
    r = grasp_force_closure_cert([([1, 0, 0], [-1, 0, 0])], mu=0.5)
    assert r["verdict"].startswith("ABSTAIN") or not r["force_closure"]
