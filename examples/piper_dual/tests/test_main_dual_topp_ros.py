from pathlib import Path
import sys
from unittest.mock import Mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import main_dual_topp_ros as entry


def test_original_cli_is_accepted():
    ns=entry.build_arg_parser().parse_args([
        '--host','127.0.0.1','--port','8000','--bridge-python','/usr/bin/python3',
        '--ros2-config','examples/piper_dual/ros2_piper_dual_local.yaml',
        '--control-stack','local-ros','--prompt','test','--max-action-delta','20',
        '--no-dry-run','--publish-actions'])
    assert ns.publish_actions and not ns.dry_run
    assert ns.max_action_delta==20
    assert ns.use_async and not ns.use_rtc


def test_main_wires_topp_and_preserves_original_environment(monkeypatch,tmp_path):
    from openpi_client import websocket_client_policy
    server=Mock()
    server.infer.return_value={'actions':np.tile(np.linspace(.001,.03,22)[:,None],(1,14))}
    monkeypatch.setattr(websocket_client_policy,'WebsocketClientPolicy',Mock(return_value=server))
    env=Mock()
    env_builder=Mock(return_value=env)
    monkeypatch.setattr(entry.original,'_build_environment',env_builder)
    def build_runtime(args,environment,policy):
        assert environment is env
        def run():
            policy.reset()
            a=policy.infer({'state':np.zeros(14)})
            assert a['actions'].shape==(14,)
            assert policy.trajectory is not None
        return Mock(run=run)
    monkeypatch.setattr(entry.original,'_build_runtime',build_runtime)
    assert entry.main(['--no-use-async','--out-dir',str(tmp_path),'--control-stack','local-ros'])==0
    assert env_builder.call_args.args[0].dry_run
    env.close.assert_called_once()


def test_rtc_rejected_before_any_environment(monkeypatch):
    factory=Mock()
    monkeypatch.setattr(entry.original,'_build_environment',factory)
    assert entry.main(['--use-rtc'])==1
    factory.assert_not_called()
