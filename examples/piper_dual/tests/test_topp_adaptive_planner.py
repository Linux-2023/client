import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topp_policy import Planner, limits
from topp_adaptive_planner import AdaptivePlanner


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_refinement_recovers_acceleration_rejection_without_raising_limits(seed):
    a = np.cumsum(np.random.default_rng(seed).normal(0,.015,(50,14)), axis=0)
    a[:,[6,13]] *= .1
    velocity, acceleration = limits(4,.05), limits(1,.2)
    with pytest.raises(RuntimeError, match='derivative 2'):
        Planner(velocity, acceleration).plan(np.zeros(14), np.zeros(14), a)
    p = AdaptivePlanner(velocity, acceleration)
    t = p.plan(np.zeros(14), np.zeros(14), a)
    assert p.attempts > 1
    ts = np.linspace(0,t.duration,50000)
    assert np.max(np.abs(t(ts,1))/velocity) <= 1.01
    assert np.max(np.abs(t(ts,2))/acceleration) <= 1.01
    np.testing.assert_allclose(t(0), 0, atol=1e-7)
    np.testing.assert_allclose(t(t.duration), a[-1], atol=1e-7)
    np.testing.assert_allclose(t(t.duration,1), 0, atol=1e-7)


def test_infeasible_boundary_is_not_silently_rescaled(tmp_path):
    p = AdaptivePlanner(limits(.5,.05), limits(1,.2))
    p.diagnostic_dir = tmp_path
    v = np.zeros(14); v[0] = 2
    a = np.zeros((10,14)); a[:,0] = np.linspace(.1,.5,10)
    with pytest.raises(RuntimeError):
        p.plan(np.zeros(14),v,a)
    saved = np.load(tmp_path/'last_rejected_plan.npz')
    np.testing.assert_array_equal(saved['dq'],v)
    np.testing.assert_array_equal(saved['actions'],a)
