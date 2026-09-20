import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import tyro


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "serve_policy.py"
SPEC = importlib.util.spec_from_file_location("serve_policy_orientation_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
serve_policy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = serve_policy
SPEC.loader.exec_module(serve_policy)


@pytest.mark.parametrize("mode", ["legacy", "wrapped-rpy", "so3"])
def test_cli_accepts_server_mode_before_checkpoint_subcommand(mode):
    args = tyro.cli(
        serve_policy.Args,
        args=[
            "--rtc-orientation-mode",
            mode,
            "policy:checkpoint",
            "--policy.config",
            "xyz-config",
            "--policy.dir",
            "/unused/checkpoint",
        ],
    )
    assert args.rtc_orientation_mode == mode
    assert args.policy.config == "xyz-config"


def test_cli_keeps_legacy_as_comparison_default():
    assert tyro.cli(serve_policy.Args, args=[]).rtc_orientation_mode == "legacy"


def test_cli_rejects_unknown_orientation_mode():
    with pytest.raises(SystemExit) as exc:
        tyro.cli(serve_policy.Args, args=["--rtc-orientation-mode", "quaternion"])
    assert exc.value.code != 0


def test_cli_help_explains_server_scope_and_comparison_default(capsys):
    with pytest.raises(SystemExit) as exc:
        tyro.cli(serve_policy.Args, args=["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    for text in ("--rtc-orientation-mode", "server-side", "XYZ", "PyTorch", "comparison"):
        assert text in help_text


@pytest.mark.parametrize("mode", ["legacy", "wrapped-rpy", "so3"])
@pytest.mark.parametrize("default_policy", [False, True])
def test_checkpoint_and_default_policy_forward_orientation_mode(monkeypatch, mode, default_policy):
    checkpoint = serve_policy.Checkpoint(config="xyz-config", dir="/unused/checkpoint")
    config = object()
    load = mock.Mock(return_value=object())
    monkeypatch.setattr(serve_policy._config, "get_config", mock.Mock(return_value=config))
    monkeypatch.setattr(serve_policy._policy_config, "create_trained_policy", load)
    monkeypatch.setitem(serve_policy.DEFAULT_CHECKPOINT, serve_policy.EnvMode.PIPER, checkpoint)
    args = serve_policy.Args(
        env=serve_policy.EnvMode.PIPER,
        default_prompt="pick cup",
        rtc_orientation_mode=mode,
        policy=serve_policy.Default() if default_policy else checkpoint,
    )
    assert serve_policy.create_policy(args) is load.return_value
    load.assert_called_once_with(config, checkpoint.dir, default_prompt="pick cup", rtc_orientation_mode=mode)


def test_server_handshake_preserves_authoritative_policy_provenance(monkeypatch):
    metadata = {
        "existing": "kept",
        "policy_config": "xyz-config",
        "checkpoint_dir": "/actual/checkpoint",
        "rtc_orientation": {
            "mode": "so3",
            "rpy_indices": [[3, 4, 5], [10, 11, 12]],
            "rpy_scale": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "rpy_offset": [[0.0] * 3, [0.0] * 3],
            "gradient_semantics": "negative-physical-rpy-gradient-divided-by-scale",
        },
    }
    policy = SimpleNamespace(metadata=metadata)
    recorder = object()
    server = mock.Mock()
    server_factory = mock.Mock(return_value=server)
    monkeypatch.setattr(serve_policy, "create_policy", lambda args: policy)
    monkeypatch.setattr(serve_policy._policy, "PolicyRecorder", lambda *args: recorder)
    monkeypatch.setattr(serve_policy.socket, "gethostname", lambda: "offline")
    monkeypatch.setattr(serve_policy.socket, "gethostbyname", lambda host: "127.0.0.1")
    monkeypatch.setattr(serve_policy.websocket_policy_server, "WebsocketPolicyServer", server_factory)
    serve_policy.main(serve_policy.Args(record=True))
    assert server_factory.call_args.kwargs["metadata"] is metadata
    assert server_factory.call_args.kwargs["policy"] is recorder
    server.serve_forever.assert_called_once_with()
