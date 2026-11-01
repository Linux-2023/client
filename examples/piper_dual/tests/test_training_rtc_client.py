"""No-server/no-robot behavioral tests for committed Training-RTC execution."""
from pathlib import Path
import copy
import sys
import threading
from unittest.mock import Mock

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / 'packages/openpi-client/src'))

from openpi_client.action_chunk_broker import ActionChunkBroker_RTC
from openpi_client import websocket_client_policy
import main_dual_ros as entry
import main_dual_delta_ros as delta_entry
import main_dual_topp_ros as nonrtc_topp_entry
from topp_committed_rtc_policy import CommittedRtcPolicy


CAPABILITY = {'supported': True, 'max_prefix_steps': 15, 'action_horizon': 50, 'action_dim': 14}


def observation():
    return {'state': np.linspace(.1, .8, 14, dtype=np.float32),
            'images': {name: np.zeros((3, 224, 224), np.uint8)
                       for name in ('cam_high', 'cam_left_wrist', 'cam_right_wrist')}}


def chunk(base=0):
    return np.repeat((base + np.arange(50, dtype=np.float32))[:, None], 14, axis=1)


class Server:
    def __init__(self, block_call=None, failure=None):
        self.requests = []
        self.block_call = block_call
        self.failure = failure
        self.entered = threading.Event()
        self.release = threading.Event()
        self.resets = 0

    def get_server_metadata(self):
        return {'training_rtc': dict(CAPABILITY)}

    def reset(self):
        self.resets += 1

    def infer(self, obs):
        call = len(self.requests)
        self.requests.append(copy.deepcopy(obs))
        if call == self.block_call:
            self.entered.set()
            if not self.release.wait(5):
                raise TimeoutError('test did not release the inference worker')
        if call > 0 and self.failure is not None:
            raise self.failure
        actions = chunk(call * 100)
        # Deliberately distinguish the returned prefix from the old plan: even
        # an early reply must not execute it, and adoption must skip it entirely.
        if call and 'training_rtc' in obs:
            actions[:len(obs['training_rtc']['action_prefix'])] = -999
        return {'actions': actions}


def make_broker(server=None, **options):
    return ActionChunkBroker_RTC(server or Server(), action_horizon=3, actions_during_latency=3,
                                 training_rtc=True, trace_enabled=True, **options)


def finish_worker(broker):
    broker._inference_thread.join(2)
    assert not broker._inference_thread.is_alive()


def test_broker_cold_start_and_exact_early_handoff_without_prefix_replay():
    server = Server()
    broker = make_broker(server)
    obs = observation()
    obs.update(rtc_obs={'ignored': True}, prev_action=chunk())
    for expected in range(4):
        np.testing.assert_array_equal(broker.infer(obs)['actions'], expected)
    finish_worker(broker)
    assert server.requests[0]['training_rtc']['action_prefix'].shape == (0, 14)
    request = server.requests[1]
    assert set(request['training_rtc']) == {'action_prefix'}
    assert request['training_rtc']['action_prefix'].dtype == np.float32
    np.testing.assert_array_equal(request['training_rtc']['action_prefix'], chunk()[3:6])
    np.testing.assert_array_equal(request['state'], obs['state'])
    assert all('rtc_obs' not in r and 'prev_action' not in r for r in server.requests)
    assert 'rtc_obs' in obs and 'prev_action' in obs  # Caller observation untouched.
    for expected in (4, 5):
        np.testing.assert_array_equal(broker.infer(obs)['actions'], expected)
    adopted = broker.infer(obs)
    np.testing.assert_array_equal(adopted['actions'], 103)
    assert adopted['_chunk_trace']['chunk_boundary']
    assert adopted['_chunk_trace']['skipped_steps'] == 3
    np.testing.assert_array_equal(broker.infer(obs)['actions'], 104)


def test_broker_late_result_is_discarded_and_fresh_request_uses_current_queue():
    server = Server(block_call=1)
    broker = make_broker(server)
    try:
        for expected in range(4):
            np.testing.assert_array_equal(broker.infer(observation())['actions'], expected)
        assert server.entered.wait(1)
        for expected in (4, 5, 6):
            np.testing.assert_array_equal(broker.infer(observation())['actions'], expected)
        server.release.set()
        finish_worker(broker)
        np.testing.assert_array_equal(broker.infer(observation())['actions'], 7)
        finish_worker(broker)
        np.testing.assert_array_equal(server.requests[2]['training_rtc']['action_prefix'], chunk()[7:10])
        for expected in (8, 9, 203):
            np.testing.assert_array_equal(broker.infer(observation())['actions'], expected)
    finally:
        server.release.set()
        finish_worker(broker)


def test_broker_missed_deadline_retains_endpoint_hold_and_commits_real_hold_targets():
    server = Server(block_call=1)
    broker = make_broker(server)
    try:
        for tick in range(60):
            np.testing.assert_array_equal(broker.infer(observation())['actions'], min(tick, 49))
        assert server.entered.wait(1)
        server.release.set()
        finish_worker(broker)
        np.testing.assert_array_equal(broker.infer(observation())['actions'], 49)
        finish_worker(broker)
        np.testing.assert_array_equal(server.requests[2]['training_rtc']['action_prefix'], np.full((3, 14), 49))
        for expected in (49, 49, 203):
            np.testing.assert_array_equal(broker.infer(observation())['actions'], expected)
    finally:
        server.release.set()
        finish_worker(broker)


def test_broker_reset_detaches_stale_worker_reply_and_restarts_with_empty_prefix():
    server = Server(block_call=1)
    broker = make_broker(server)
    try:
        for _ in range(4):
            broker.infer(observation())
        assert server.entered.wait(1)
        broker.reset()
        server.release.set()
        finish_worker(broker)
        assert broker._last_results is None
        assert broker._training_future is None
        np.testing.assert_array_equal(broker.infer(observation())['actions'], 200)
        assert server.requests[2]['training_rtc']['action_prefix'].shape == (0, 14)
        assert server.resets == 1
    finally:
        server.release.set()
        finish_worker(broker)


def test_broker_new_mode_failures_remain_fatal_until_reset():
    server = Server(failure=ConnectionError('lost connection'))
    broker = make_broker(server)
    for _ in range(4):
        broker.infer(observation())
    finish_worker(broker)
    for _ in range(2):
        with pytest.raises(ConnectionError, match='lost connection'):
            broker.infer(observation())
    assert len(server.requests) == 2


def test_broker_rejects_zero_length_non_cold_commitment():
    with pytest.raises(ValueError):
        ActionChunkBroker_RTC(Server(), 3, actions_during_latency=0, training_rtc=True)


def test_broker_rejects_execution_clock_different_from_training():
    with pytest.raises(ValueError):
        make_broker(fps=15)


@pytest.mark.parametrize('metadata', [{}, {'training_rtc': {}},
    {'training_rtc': {**CAPABILITY, 'supported': False}},
    {'training_rtc': {**CAPABILITY, 'max_prefix_steps': 16}},
    {'training_rtc': {**CAPABILITY, 'action_horizon': 49}},
    {'training_rtc': {**CAPABILITY, 'action_dim': 32}}])
def test_capability_rejected_before_inference(metadata):
    server = Mock()
    server.get_server_metadata.return_value = metadata
    with pytest.raises(ValueError, match='Training-RTC'):
        make_broker(server)
    server.infer.assert_not_called()


def test_broker_rejects_blending_and_overlong_prefix():
    with pytest.raises(ValueError, match='blending'):
        make_broker(handoff_blend_steps=1)
    with pytest.raises(ValueError):
        ActionChunkBroker_RTC(Server(), 3, actions_during_latency=16, training_rtc=True)


class Line:
    duration = 10.

    def __init__(self, q, v):
        self.q, self.v = q.copy(), v.copy()

    def __call__(self, t, order=0):
        t = np.asarray(t)
        if order == 0:
            return self.q + t[..., None] * self.v
        return np.broadcast_to(self.v if order == 1 else np.zeros(14), t.shape + (14,)).copy()


def make_topp(fps=30, model_fps=30, server=None, **options):
    planner = Mock(fps=fps, path_fps=model_fps, velocity=np.ones(14))
    planner.plan.side_effect = lambda q, v, actions: Line(q, v)
    policy = CommittedRtcPolicy(server or Server(), planner, training_rtc=True,
                                clock=lambda: 0., **options)
    return policy


def test_topp_cold_start_is_empty_and_reset_restores_cold_start():
    policy = make_topp()
    try:
        policy.infer(observation())
        policy.thread.join(2)
        assert policy.policy.requests[0]['training_rtc']['action_prefix'].shape == (0, 14)
        policy.reset()
        policy.infer(observation())
        policy.thread.join(2)
        assert policy.policy.requests[1]['training_rtc']['action_prefix'].shape == (0, 14)
    finally:
        policy.close()


@pytest.mark.parametrize('fps,model_fps', [(30, 30), (60, 30), (15, 30)])
def test_topp_prefix_includes_anchor_and_preserves_exact_position_velocity(fps, model_fps):
    policy = make_topp(fps, model_fps)
    old = policy.trajectory = Line(np.linspace(.1, .5, 14), np.linspace(.01, .02, 14))
    policy.q = old.q.copy()
    policy.training_cold = False
    try:
        policy.infer(observation())
        job = policy.job.copy()
        policy.thread.join(2)
        request = policy.policy.requests[0]
        prefix = request['training_rtc']['action_prefix']
        assert 'rtc_obs' not in request and 'prev_action' not in request
        assert prefix.shape == (job['d'] + 1, 14)
        assert len(prefix) <= 15
        assert job['target_tick'] - job['request_tick'] == pytest.approx(job['d'] * fps / model_fps)
        np.testing.assert_allclose(prefix, old(1 / fps + np.arange(len(prefix)) / model_fps), atol=1e-7)
        np.testing.assert_allclose(prefix[-1], job['q'], atol=1e-7)
        np.testing.assert_array_equal(policy.planner.plan.call_args.args[2], chunk()[job['d']+1:job['d']+23])
        while policy.tick < job['target_tick']:
            output = policy.infer(observation())
            np.testing.assert_allclose(output['actions'], old(policy.tick / fps), atol=1e-7)
            if policy.tick < job['target_tick']:
                assert policy.trajectory is old
        assert policy.trajectory is not old
        np.testing.assert_allclose(policy.trajectory(0), job['q'], atol=1e-7)
        np.testing.assert_allclose(policy.trajectory(0, 1), job['v'], atol=1e-7)
    finally:
        policy.close()


def test_topp_trained_prefix_cap_and_common_grid_boundaries():
    policy = make_topp(30, 30, soft_prefix_steps=49)
    policy.latencies.append(100)
    ticks, d, saturated = policy._commit_window()
    assert saturated and ticks == d == 14
    with pytest.raises(ValueError, match='common handoff'):
        make_topp(17, 30)
    with pytest.raises(ValueError, match='model horizon'):
        make_topp(model_horizon=49)
    with pytest.raises(ValueError):
        make_topp(actions_during_latency=15)


@pytest.mark.parametrize('model_fps', [15, 29.97])
def test_topp_rejects_model_grid_different_from_training(model_fps):
    with pytest.raises(ValueError):
        make_topp(model_fps=model_fps)


def test_topp_late_training_reply_is_discarded():
    server = Server(block_call=0)
    policy = make_topp(server=server)
    old = policy.trajectory = Line(np.ones(14), np.full(14, .01))
    policy.q = old.q.copy()
    policy.training_cold = False
    try:
        policy.infer(observation())
        assert server.entered.wait(1)
        deadline = policy.job['target_tick']
        while policy.tick <= deadline:
            policy.infer(observation())
        server.release.set()
        policy.thread.join(2)
        output = policy.infer(observation())
        assert policy.trajectory is old and policy.chunk_id == 0
        np.testing.assert_allclose(output['actions'], old(policy.tick / 30), atol=1e-7)
    finally:
        server.release.set()
        policy.close()


def test_topp_malformed_training_reply_is_fatal_not_planner_recovery():
    server = Mock()
    server.get_server_metadata.return_value = {'training_rtc': CAPABILITY}
    server.infer.return_value = {'actions': np.zeros((49, 14))}
    policy = make_topp(server=server)
    try:
        policy.infer(observation())
        policy.thread.join(2)
        with pytest.raises(ValueError, match='finite'):
            policy.infer(observation())
        policy.planner.plan.assert_not_called()
    finally:
        policy.close()




@pytest.mark.parametrize('extra', [['--no-use-async'], ['--handoff-blend-steps', '1'],
                                  ['--fps', '15'], ['--actions-during-latency', '0']])
def test_incompatible_shared_options_fail_before_server_connection(monkeypatch, extra):
    connect = Mock()
    monkeypatch.setattr(websocket_client_policy, 'WebsocketClientPolicy', connect)
    with pytest.raises(ValueError, match='Training-RTC'):
        entry._build_policy(entry.parse_args(['--training-rtc', '--fps', '30'] + extra))
    connect.assert_not_called()


def test_unsupported_training_server_fails_before_ros_backend(monkeypatch):
    server = Mock()
    server.get_server_metadata.return_value = {}
    monkeypatch.setattr(websocket_client_policy, 'WebsocketClientPolicy', lambda **kw: server)
    environment = Mock()
    monkeypatch.setattr(entry, '_build_environment', environment)
    assert entry.main(entry.Args(training_rtc=True, fps=30)) == 1
    environment.assert_not_called()


def test_delta_wrapper_rejects_client_delta_decode(monkeypatch):
    server = Server()
    monkeypatch.setattr(websocket_client_policy, 'WebsocketClientPolicy', lambda **kw: server)
    with pytest.raises(ValueError, match='absolute server targets'):
        delta_entry._build_policy(delta_entry.parse_args(['--training-rtc', '--server-output', 'delta']))






def test_nonrtc_topp_entrypoint_rejects_training_flag_before_connect(monkeypatch):
    connect = Mock()
    monkeypatch.setattr(websocket_client_policy, 'WebsocketClientPolicy', connect)
    assert nonrtc_topp_entry.main(['--training-rtc']) == 1
    connect.assert_not_called()
