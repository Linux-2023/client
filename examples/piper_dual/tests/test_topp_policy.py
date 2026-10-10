from pathlib import Path
import sys
import time
from concurrent.futures import Future
from unittest.mock import Mock
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topp_policy import Planner, ToppPolicy, limits


def planner():
    return Planner(limits(.5, .05), limits(1, .2))


def chunk(end=.3):
    a = np.zeros((22,14))
    a[:, 0] = np.linspace(.005,end,22)
    a[:, 7] = -a[:, 0]*.5
    a[:, 6] = np.linspace(.001,.02,22)
    return a


def test_constraints_and_synchronized_endpoints():
    p = planner()
    a = chunk()
    t = p.plan(np.zeros(14),np.zeros(14),a)
    ts = np.linspace(0,t.duration,3000)
    assert np.max(np.abs(t(ts,1))/p.velocity) <= 1.01
    assert np.max(np.abs(t(ts,2))/p.acceleration) <= 1.01
    np.testing.assert_allclose(t(t.duration),a[-1],atol=1e-7)
    np.testing.assert_allclose(t(0,1),0,atol=1e-7)
    np.testing.assert_allclose(t(t.duration,1),0,atol=1e-7)


def test_nonzero_velocity_handoff():
    p = planner()
    a = np.zeros((22,14)); a[:,0]=np.linspace(.01,.4,22)
    old = p.plan(np.zeros(14),np.zeros(14),a)
    q,v = old(.2),old(.2,1)
    a[:,0]=np.linspace(q[0]+v[0]/15,.5,22)
    new = p.plan(q,v,a)
    np.testing.assert_allclose(new(0),q,atol=1e-7)
    np.testing.assert_allclose(new(0,1),v,atol=1e-6)


@pytest.mark.parametrize('actions', [np.zeros((3,13)),np.full((3,14),np.nan),np.empty((0,14))])
def test_bad_chunk_rejected(actions):
    with pytest.raises(ValueError):
        planner().plan(np.zeros(14),np.zeros(14),actions)


def test_constant_path_and_duplicate_points():
    p = planner()
    assert p.plan(np.zeros(14),np.zeros(14),np.zeros((22,14))) is None
    a=chunk()
    assert p.plan(np.zeros(14),np.zeros(14),np.repeat(a,2,axis=0)) is not None


def test_delayed_inference_finishes_old_path_without_jump():
    p=planner(); server=Mock()
    wrapper=ToppPolicy(server,p,prefetch_seconds=.4)
    wrapper.q=np.zeros(14)
    wrapper.trajectory=p.plan(wrapper.q,np.zeros(14),chunk())
    wrapper.future=Future(); wrapper.last_requested=True
    outputs=[wrapper.infer({'state':np.zeros(14)})['actions'] for _ in range(80)]
    np.testing.assert_allclose(outputs[-1],chunk()[-1],atol=1e-6)
    np.testing.assert_allclose(outputs[-1],outputs[-2])
    assert np.all(np.abs(np.diff(outputs,axis=0)) <= p.velocity/15*1.02)
    server.infer.assert_not_called()


def test_rejected_plan_keeps_prior_trajectory():
    p=planner(); wrapper=ToppPolicy(Mock(),p)
    wrapper.q=np.zeros(14)
    old=wrapper.trajectory=p.plan(wrapper.q,np.zeros(14),chunk())
    wrapper.future=Future(); wrapper.future.set_result(({'actions':np.full((22,14),np.nan)},20))
    wrapper.last_requested=True
    out=wrapper.infer({'state':np.zeros(14)})
    assert wrapper.trajectory is old
    assert np.isfinite(out['actions']).all()


def test_first_output_anchors_to_measured_state_and_server_errors_propagate():
    server=Mock();server.infer.return_value={'actions':chunk()}
    wrapper=ToppPolicy(server,planner(),asynchronous=False)
    state=np.zeros(14)
    np.testing.assert_array_equal(wrapper.infer({'state':state})['actions'],state)
    assert wrapper.trajectory is not None
    wrapper.future=Future();wrapper.future.set_exception(RuntimeError('offline'))
    with pytest.raises(RuntimeError,match='offline'):
        wrapper.infer({'state':state})


def test_close_and_reset():
    server=Mock(); wrapper=ToppPolicy(server,planner())
    wrapper.q=np.ones(14)
    wrapper.reset()
    assert wrapper.q is None
    server.reset.assert_called_once()
    wrapper.close()
    with pytest.raises(RuntimeError,match='closed'):
        wrapper.infer({'state':np.zeros(14)})


@pytest.mark.parametrize('moving', [False, True])
def test_output_fps_does_not_change_planned_path(moving):
    low = planner()
    high = Planner(low.velocity, low.acceleration, fps=60)
    a = np.zeros((22, 14))
    a[:, 0] = np.linspace(.01, .4, 22)
    q, v = np.zeros(14), np.zeros(14)
    if moving:
        old = low.plan(q, v, a)
        q, v = old(.2), old(.2, 1)
        a[:, 0] = np.linspace(q[0] + v[0]/15, .5, 22)
    first, second = low.plan(q, v, a), high.plan(q, v, a)
    assert first.duration == pytest.approx(second.duration, abs=1e-9)
    ts = np.linspace(0, first.duration, 1000)
    for order in (0, 1, 2):
        np.testing.assert_allclose(first(ts, order), second(ts, order), atol=1e-9)
