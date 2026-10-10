from pathlib import Path
import sys
from concurrent.futures import Future
from unittest.mock import Mock
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topp_rtc_policy import Planner, ToppRtcPolicy, limits
import main_dual_topp_rtc_ros as entry


def make(fps=30):
    p = Planner(limits(.5, .05), limits(1, .2), fps=fps, path_fps=30)
    return ToppRtcPolicy(Mock(), p, rolling=False)


def actions():
    a = np.zeros((50, 14))
    a[:, 0] = np.linspace(.001, .2, 50)
    return a


def test_prefix_is_resampled_plan_not_raw_indices():
    w = make(fps=60)
    w.q = np.zeros(14)
    w.trajectory = w.planner.plan(w.q, w.q, actions()[:22])
    w.elapsed = .1
    w.q = w.trajectory(w.elapsed)
    w.policy.infer.return_value = {'actions': actions()}
    obs = {'state': np.ones(14)*.01}
    w._request(obs)
    w.thread.join(2)
    sent = w.policy.infer.call_args.args[0]
    expected = w.trajectory(np.minimum(.1+np.arange(50)/30, w.trajectory.duration))
    np.testing.assert_allclose(sent['rtc_obs']['prev_action'], expected, atol=1e-7)
    assert sent['rtc_obs']['s'] == 22
    assert sent['rtc_obs']['d'] == 5
    assert 'rtc_obs' not in obs
    w.close()


def test_first_prefix_holds_measured_pose_not_zero():
    w = make()
    w.q = np.ones(14)*.03
    w.policy.infer.return_value = {'actions': actions()}
    w._request({'state': w.q})
    w.thread.join(2)
    np.testing.assert_allclose(w.policy.infer.call_args.args[0]['rtc_obs']['prev_action'], .03)
    w.close()


@pytest.mark.parametrize('fps,ticks', [(30, 6), (60, 12)])
def test_trim_same_model_time_at_different_output_fps(fps, ticks):
    w = make(fps)
    w.q = np.zeros(14)
    w.trajectory = w.planner.plan(w.q, w.q, actions()[:22])
    w.request_has_plan = True
    w.request_tick = 0
    w.tick = ticks-1
    w.last_requested = True
    w.future = Future()
    w.future.set_result(({'actions': actions()}, 200))
    w.planner.plan = Mock(return_value=w.trajectory)
    w.infer({'state': w.q})
    np.testing.assert_array_equal(w.planner.plan.call_args.args[2], actions()[6:28])
    assert w.latency_s == .2


def test_expired_and_wrong_horizon_keep_old_plan():
    for raw, ticks in [(actions(), 60), (actions()[:22], 1)]:
        w = make()
        w.q = np.zeros(14)
        old = w.trajectory = w.planner.plan(w.q, w.q, actions()[:22])
        w.request_has_plan = True
        w.tick = ticks
        w.last_requested = True
        w.future = Future()
        w.future.set_result(({'actions': raw}, 2000))
        out = w.infer({'state': w.q})
        assert w.trajectory is old
        assert np.isfinite(out['actions']).all()


def test_reset_and_required_async():
    w = make()
    w.tick = 10
    w.latency_s = .5
    w.reset()
    assert w.tick == 0 and w.latency_s is None and not w.request_has_plan
    with pytest.raises(ValueError):
        ToppRtcPolicy(Mock(), w.planner, asynchronous=False)


def test_cli_and_integration(monkeypatch, tmp_path):
    from openpi_client import websocket_client_policy
    ns = entry.build_arg_parser().parse_args([])
    assert ns.use_rtc and ns.use_async
    server = Mock()
    server.infer.return_value = {'actions': actions()}
    monkeypatch.setattr(websocket_client_policy, 'WebsocketClientPolicy', Mock(return_value=server))
    env = Mock()
    monkeypatch.setattr(entry.original, '_build_environment', Mock(return_value=env))
    def build(args, environment, policy):
        def run():
            policy.reset()
            policy.infer({'state': np.zeros(14)})
            policy.thread.join(2)
            policy.infer({'state': np.zeros(14)})
            assert policy.trajectory is not None
            assert 'rtc_obs' in server.infer.call_args.args[0]
        return Mock(run=run)
    monkeypatch.setattr(entry.original, '_build_runtime', build)
    assert entry.main(['--topp-rtc-backend', 'legacy', '--out-dir', str(tmp_path), '--control-stack', 'local-ros']) == 0
    env.close.assert_called_once()


@pytest.mark.parametrize('flags', [['--no-use-rtc'], ['--no-use-async'], ['--rtc-model-horizon', '22']])
def test_invalid_modes_fail_before_connect(monkeypatch, flags):
    from openpi_client import websocket_client_policy
    factory = Mock()
    monkeypatch.setattr(websocket_client_policy, 'WebsocketClientPolicy', factory)
    assert entry.main(flags) == 1
    factory.assert_not_called()


def test_rolling_uses_full_prediction_and_requests_before_braking():
    w = make()
    w.rolling = True
    w.q = np.zeros(14)
    raw = actions()
    whole = w._plan_rolling(raw, 0, np.zeros(14))
    short = w.planner.plan(w.q, np.zeros(14), raw[:22])
    np.testing.assert_allclose(whole(whole.duration), raw[-1], atol=1e-7)
    assert whole.duration > short.duration
    w.trajectory = whole
    w._schedule()
    assert w.request_at < whole.duration - w.prefetch_seconds
    # Request while moving, rather than waiting for a zero-speed endpoint.
    assert np.max(np.abs(whole(w.request_at, 1))) > .01


def test_rolling_exact_buffer_and_nonzero_velocity_seam():
    w = make()
    w.rolling = True
    w.q = np.zeros(14)
    old = w.trajectory = w.planner.plan(w.q, w.q, actions())
    w.elapsed = .1
    w.q = old(.1)
    raw = actions()
    # Construct an aligned continuation at the future join.
    join = .2
    raw[3:, 0] = np.linspace(old(join)[0] + old(join, 1)[0]/30, .3, 47)
    new = w._plan_rolling(raw, 0, old(.1, 1))
    for order in (0, 1):
        ts = np.linspace(0, .1, 20)
        np.testing.assert_allclose(new(ts, order), old(.1+ts, order), atol=1e-6)
    assert abs(new(.1, 1)[0]) > .01
    for order, bounds in [(1, w.planner.velocity), (2, w.planner.acceleration)]:
        ts = np.linspace(0, new.duration, 2000)
        assert np.max(np.abs(new(ts, order))/bounds) <= 1.01
    np.testing.assert_allclose(new(new.duration, 1), 0, atol=1e-6)


def test_rolling_late_inference_still_stops():
    w = make()
    w.rolling = True
    w.q = np.zeros(14)
    w.trajectory = w.planner.plan(w.q, w.q, actions())
    w.future = Future()
    w.last_requested = True
    out = [w.infer({'state': np.zeros(14)})['actions'] for _ in range(150)]
    np.testing.assert_allclose(out[-1], actions()[-1], atol=1e-6)
    np.testing.assert_array_equal(out[-1], out[-2])


def test_rejection_at_endpoint_does_not_deadlock_requests():
    w = make()
    w.rolling = True
    w.q = np.zeros(14)
    w.trajectory = w.planner.plan(w.q, w.q, actions())
    w.elapsed = w.trajectory.duration
    w.last_requested = True
    w.future = Future()
    w.future.set_result(({'actions': actions()[:1]}, 200))
    w.infer({'state': w.q})
    w._request = Mock()
    for _ in range(5):
        w.infer({'state': w.q})
    assert w._request.called


def test_infeasible_moving_handoff_waits_for_rest_then_requests():
    w = make()
    w.rolling = True
    w.q = np.zeros(14)
    old = w.trajectory = w.planner.plan(w.q, w.q, actions())
    w.elapsed = .1
    w.last_requested = True
    w.future = Future()
    w.future.set_result(({'actions': actions()}, 200))
    w._plan_rolling = Mock(side_effect=RuntimeError('TOPP path infeasible from current planned velocity'))
    w.infer({'state': w.q})
    assert w.trajectory is old
    assert w.request_at == old.duration
    w._request = Mock()
    while w.elapsed + 1/w.planner.fps < old.duration:
        w.infer({'state': w.q})
    w._request.assert_not_called()
    w.infer({'state': w.q})
    w._request.assert_called_once()
    np.testing.assert_allclose(w.q, old(old.duration), atol=1e-7)


def test_stationary_existing_plan_does_not_mark_request_as_moving():
    w = make()
    w.q = np.zeros(14)
    w.trajectory = w.planner.plan(w.q, w.q, actions())
    w.elapsed = w.trajectory.duration
    w.q = w.trajectory(w.elapsed)
    w.policy.infer.return_value = {'actions': actions()}
    w._request({'state': w.q})
    w.thread.join(2)
    assert not w.request_has_plan
    w.close()
