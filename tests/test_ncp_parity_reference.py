"""Parity against Pinocchio's analytical constrained-dynamics derivatives, locked as data.

    OMP_NUM_THREADS=4 python -m pytest -q tests/test_ncp_parity_reference.py

Reproducing these rows needs Pinocchio, two robot models and a GPU run per platform, so
`data/ncp/pinocchio_parity_reference.json` carries the measured relative errors and the
run-to-run sha256 pairs, and this test holds them to the bounds the comparison was made
against. The same file carries the cross-device byte identity of the batched solver and
the refuted 3.47x ADMM claim, so the refutation travels with the numbers.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REF = json.loads((ROOT / "data/ncp/pinocchio_parity_reference.json").read_text())
PLATFORMS = sorted(REF["platforms"])


@pytest.mark.parametrize("name", PLATFORMS)
@pytest.mark.parametrize("quantity", sorted(REF["bounds"]))
def test_derivative_parity_against_pinocchio(name, quantity):
    p = REF["platforms"][name]
    bound = REF["bounds"][quantity]
    assert p[quantity] < bound, f"{name}.{quantity} = {p[quantity]:.3e}, bound {bound:.0e}"


@pytest.mark.parametrize("name", PLATFORMS)
def test_two_runs_are_bit_identical(name):
    p = REF["platforms"][name]
    assert p["sha256_run1"] == p["sha256_run2"]
    for k, v in p["sha256_run1"].items():
        assert len(v) == 64 and int(v, 16) >= 0, f"{name}.{k} is not a sha256"


@pytest.mark.parametrize("name", PLATFORMS)
def test_talos_covers_every_dof_without_composite_joints(name):
    p = REF["platforms"][name]
    assert p["composite_joints"] == 0, (
        "the q-derivative parity holds over the whole state space only because neither "
        "platform has composite joints; a robot that does needs the subset restriction")


def test_batched_solver_is_byte_identical_across_devices():
    b = REF["batched_cross_device"]
    assert b["envs_identical_alone_vs_in_batch"] == "8/8"
    assert b["arrays_byte_identical_across_cloud_devices"] == "26/26"
    assert b["suites_byte_identical_across_cloud_devices"] == "5/5"
    assert len(b["devices"]) == 4
    assert max(b["repeat_spread_percent"]) < 1.0


def test_the_refuted_claim_travels_with_the_numbers():
    text = " ".join(REF["reservations"])
    assert "0.38x" in text and "refuted" in text, (
        "the 3.47x ADMM win at n_c = 256 was a sweep artefact; the reservation must stay")
    assert "5.30x" in text, "the large-scale row that does stand must stay too"
