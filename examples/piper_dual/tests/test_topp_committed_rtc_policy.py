from pathlib import Path
import sys
import threading
from unittest.mock import Mock
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topp_committed_rtc_policy import CommittedRtcPolicy
from topp_adaptive_planner import AdaptivePlanner
from topp_policy import limits
import main_dual_topp_rtc_ros as entry


class Line:
    duration = 10.
    def __init__(self, q=None, v=None):
        self.q = np.zeros(14) if q is None else q.copy()
        self.v = np.r_[.03,np.zeros(13)] if v is None else v.copy()
    def __call__(self, t, order=0):
        t = np.asarray(t)
        if order == 0:
            return self.q+t[...,None]*self.v
        return np.broadcast_to(self.v if order == 1 else np.zeros(14),t.shape+(14,)).copy()


def make(fps=15, model_fps=15, planner=None, server=None):
    if planner is None:
        planner = Mock(fps=fps, path_fps=model_fps, velocity=limits(.5,.05))
        planner.plan.side_effect = lambda q,v,a: Line(q,v)
    if server is None:
        server = Mock()
        raw = np.zeros((50,14)); raw[:,0]=np.arange(50)*.002
        server.infer.return_value = {'actions':raw}
    w = CommittedRtcPolicy(server, planner)
    return w


def step(w):
    return w.infer({'state':np.zeros(14)})


@pytest.mark.parametrize('fps,model_fps', [(15,15),(30,15),(15,30),(60,15)])
def test_immutable_prefix_and_exact_future_handoff(fps, model_fps):
    w=make(fps,model_fps)
    old=w.trajectory=Line()
    w.q=np.zeros(14)
    first=step(w)
    job=w.job.copy()
    w.thread.join(2)
    sent=w.policy.infer.call_args.args[0]['rtc_obs']
    assert job['target_tick']-job['request_tick'] == pytest.approx(sent['d']*fps/model_fps)
    assert 50-sent['s'] == sent['d']+3
    np.testing.assert_allclose(sent['prev_action'],old(1/fps+np.arange(50)/model_fps),atol=1e-7)
    assert w.trajectory is old
    for tick in range(2, job['target_tick']+1):
        out=step(w)
        np.testing.assert_allclose(out['actions'],old(tick/fps),atol=1e-7)
        if tick < job['target_tick']:
            assert w.trajectory is old
    assert w.trajectory is not old
    np.testing.assert_allclose(w.trajectory(0,1),old(0,1))
    raw=w.policy.infer.return_value['actions']
    np.testing.assert_array_equal(w.planner.plan.call_args.args[2],raw[sent['d']+1:sent['d']+23])
    assert len(w.planner.plan.call_args.args[2]) <=22
    w.close()


def test_blocked_solver_does_not_block_output_and_late_result_is_discarded():
    entered,release=threading.Event(),threading.Event()
    w=make()
    old=w.trajectory=Line();w.q=np.zeros(14)
    def blocked(q,v,a):
        assert threading.current_thread().name == 'rtc-inference-topp'
        entered.set()
        assert release.wait(3)
        return Line(q,v)
    w.planner.plan.side_effect=blocked
    try:
        step(w)
        assert entered.wait(1)
        deadline=w.job['target_tick']
        while w.tick <= deadline:
            out=step(w)
            np.testing.assert_allclose(out['actions'],old(w.tick/15),atol=1e-7)
        assert w.trajectory is old and not release.is_set()
        release.set();w.thread.join(2)
        step(w)
        assert w.trajectory is old
        assert w.chunk_id ==0
        assert w.job is None
    finally:
        release.set();w.close()


def test_failure_at_rest_does_not_deadlock():
    w=make();w.planner.plan.side_effect=RuntimeError('infeasible')
    step(w);w.thread.join(2)
    step(w)
    assert w.job is None
    deadline=w.request_at_tick
    while w.tick < deadline:
        step(w)
    assert w.policy.infer.call_count >= 2
    w.close()


def test_network_error_propagates():
    w=make();w.policy.infer.side_effect=ConnectionError('server disconnected')
    step(w);w.thread.join(2)
    with pytest.raises(ConnectionError): step(w)
    w.close()


def test_worker_and_reset_do_not_race():
    entered,release=threading.Event(),threading.Event()
    w=make()
    def server(obs):
        entered.set();release.wait(1)
        return {'actions':np.zeros((50,14))}
    w.policy.infer.side_effect=server
    step(w);assert entered.wait(1)
    release.set();w.reset()
    assert w.future is None and w.job is None and w.tick==0
    assert w.trajectory is None
    w.close()


def test_budget_adapts_and_incompatible_frequencies_rejected():
    w=make()
    before=w._commit_window()[0]
    w.latencies.append(.7);w.plan_times.append(.4)
    assert w._commit_window()[0]>before
    w.latencies.append(100)
    assert w._commit_window()[2]
    with pytest.raises(ValueError,match='common handoff'):
        make(fps=17,model_fps=47)


def test_real_planner_initial_handoff_and_cli(monkeypatch,tmp_path):
    from openpi_client import websocket_client_policy
    ns=entry.build_arg_parser().parse_args([])
    assert ns.use_rtc and ns.topp_rtc_backend=='committed'
    server=Mock()
    def predict(obs):
        d=obs['rtc_obs']['d']
        raw=obs['rtc_obs']['prev_action'].copy()
        raw[d+1:,0]=np.linspace(.001,.15,49-d)
        return {'actions':raw}
    server.infer.side_effect=predict
    monkeypatch.setattr(websocket_client_policy,'WebsocketClientPolicy',Mock(return_value=server))
    env=Mock();monkeypatch.setattr(entry.original,'_build_environment',Mock(return_value=env))
    def build(args,environment,policy):
        def run():
            policy.reset();step(policy)
            deadline=policy.job['target_tick']
            policy.thread.join(3)
            assert policy.future.done()
            while policy.tick<deadline: step(policy)
            assert policy.chunk_id==1 and policy.trajectory is not None
            np.testing.assert_allclose(policy.trajectory(0),0,atol=1e-7)
        return Mock(run=run)
    monkeypatch.setattr(entry.original,'_build_runtime',build)
    assert entry.main(['--out-dir',str(tmp_path),'--control-stack','local-ros'])==0
    env.close.assert_called_once()


def test_real_nonzero_handoff_respects_committed_prefix():
    p=AdaptivePlanner(limits(.5,.05),limits(1,.2),fps=30,path_fps=15)
    a=np.zeros((50,14));a[:,0]=np.linspace(.001,.5,50)
    old=p.plan(np.zeros(14),np.zeros(14),a)
    server=Mock()
    w=CommittedRtcPolicy(server,p)
    w.trajectory=old;w.q=np.zeros(14)
    def aligned(obs):
        d=obs['rtc_obs']['d']
        # Offline mock of a prefix-consistent prediction, not a model evaluation.
        join=(1/30+d/15)
        q,v=old(join),old(join,1)
        raw=obs['rtc_obs']['prev_action'].astype(float)
        raw[d+1:]=q+np.arange(1,50-d)[:,None]/15*v
        return {'actions':raw}
    server.infer.side_effect=aligned
    step(w);deadline=w.job['target_tick']
    w.thread.join(3)
    assert w.future.done()
    while w.tick<deadline:
        step(w)
        np.testing.assert_allclose(w.q,old(w.tick/30),atol=1e-7)
    assert w.chunk_id==1
    assert np.max(np.abs(w.trajectory(0,1))) > .01
    np.testing.assert_allclose(w.trajectory(0,1),old(deadline/30,1),atol=1e-5)
    ts=np.linspace(0,w.trajectory.duration,2000)
    assert np.max(np.abs(w.trajectory(ts,2))/p.acceleration)<=1.01
    w.close()


def test_one_moving_retry_preserves_prefix_and_is_bounded():
    w=make();w.q=np.zeros(14);w.trajectory=Line()
    w.tick=3;w.job={'target_tick':9}
    w._recover('No bounded velocity transition satisfies limits',at_rest=True)
    assert w.request_at_tick>=10
    assert w.request_at_tick<150
    assert w.moving_retries==1
    assert w.trajectory is not None
    w.tick=w.request_at_tick;w.job={'target_tick':w.tick+9}
    w._recover('TOPP path infeasible',at_rest=True)
    assert w.request_at_tick>=150
    w.close()


def test_no_moving_retry_without_enough_pipeline_time():
    w=make();w.q=np.zeros(14);w.trajectory=Line()
    w.tick=145;w.job={'target_tick':147}
    w._recover('TOPP path infeasible',at_rest=True)
    assert w.request_at_tick>=150
    assert w.moving_retries==0
    w.close()


def test_failed_transition_input_saved_for_replay(tmp_path):
    w=make()
    w.planner.diagnostic_dir=tmp_path
    w.planner.acceleration=limits(1,.2)
    w.planner.plan.side_effect=RuntimeError('No bounded velocity transition')
    step(w);w.thread.join(2)
    assert (tmp_path/'last_committed_failure.npz').is_file()
    with np.load(tmp_path/'last_committed_failure.npz') as saved:
        np.testing.assert_allclose(saved['q'],0)
        assert saved['actions'].shape==(22,14)
        assert 'transition' in str(saved['reason'])
    step(w)
    w.close()


def test_moving_retry_writes_real_json_log_and_continues(tmp_path):
    import json
    w=make();w.metrics_path=tmp_path/'metrics.jsonl'
    old=w.trajectory=Line();w.q=np.zeros(14)
    w.tick=3;w.job={'target_tick':9}
    w._recover('No bounded velocity transition satisfies limits',at_rest=True)
    event=json.loads(w.metrics_path.read_text())
    assert event['retry_moving'] is True
    assert event['at_rest'] is False
    assert w.job is None and w.pending is None
    out=step(w)
    assert w.trajectory is old
    np.testing.assert_allclose(out['actions'],old(w.tick/15),atol=1e-7)
    w.close()


def test_log_numpy_scalars_keep_json_types(tmp_path):
    import json
    w=make();w.metrics_path=tmp_path/'metrics.jsonl'
    w._log(event='test',flag=np.bool_(True),tick=np.int64(5),value=np.float32(.25))
    event=json.loads(w.metrics_path.read_text())
    assert event['flag'] is True and event['tick']==5 and event['value']==.25
    w.close()
