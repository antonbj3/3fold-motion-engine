"""Deterministic cost checks for the actual capsule-distance reduction."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

SCRIPT=Path(__file__).resolve().parents[1]/"scripts/tropical_sdf_swept_volume_continuous_collision_detection_min_over_trajectory.py"
spec=importlib.util.spec_from_file_location("swept_cost_review",SCRIPT)
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)

def fixture():
    return core.posed_capsules(np.zeros(5)),np.array([[0.35,0.0,0.30],[0.0,0.0,0.0]])

def test_real_kernel_counts_time_capsules_and_point_vectors():
    caps,points=fixture();rows=core._core_operation_check(caps,points)
    assert rows and all(row['valid'] for row in rows)
    for row in rows:
        assert row['capsule_calls']==row['T']*row['K']
        assert row['point_distances']==row['T']*row['K']*row['M']

@pytest.mark.parametrize('mutation',['skip_time','skip_capsule','ignore_counter'])
def test_count_guard_rejects_missing_work(monkeypatch,mutation):
    actual=core._min_plus_core_timed
    def broken(caps,P,capsule_fn=None):
        if mutation=='skip_time':caps=caps[:1]
        if mutation=='skip_capsule':caps=[row[:1] for row in caps]
        if mutation=='ignore_counter':capsule_fn=None
        return actual(caps,P,capsule_fn)
    monkeypatch.setattr(core,'_min_plus_core_timed',broken)
    caps,points=fixture();rows=core._core_operation_check(caps,points)
    assert rows and any(not row['valid'] for row in rows)
