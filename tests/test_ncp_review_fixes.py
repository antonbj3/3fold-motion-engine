"""Review regressions: finite tolerances are required for a rank guard."""
import numpy as np
import pytest
from motion_engine.ncp.contact_rank import rank_factor, guarded_response
from motion_engine.ncp import ncp_ref as R

@pytest.mark.parametrize("tol", [np.inf, np.nan, -1., 0., 1., 2.])
def test_guard_rejects_tolerances_that_disable_the_certificate(tol):
    A=np.diag([1.,0.,1.])
    with pytest.raises(ValueError, match="tol"):
        guarded_response(A,np.array([[0.],[1.],[0.]]),rank_factor(A),
                         selection="minimum_norm",observable=[[0.,1.,0.]],tol=tol)

def test_unknown_admm_xstep_is_rejected():
    with pytest.raises(ValueError, match="xstep"):
        R.solve_ncp_admm(np.eye(3),np.array([-1.,.1,0.]),[.3],xstep="typo")
