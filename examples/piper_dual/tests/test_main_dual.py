'''Focused tests for the dual-arm deployment and backend-selection contract.'''

from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import env_dual
import main_dual


def test_validate_backend_contract_rejects_publish_actions_without_ros2_backend() -> None:
    args = main_dual.Args(backend='sdk', publish_actions=True, dry_run=False)

    with pytest.raises(ValueError, match='--publish-actions requires --backend ros2'):
        main_dual._validate_backend_contract(args)


def test_validate_backend_contract_rejects_publish_actions_while_dry_running() -> None:
    args = main_dual.Args(backend='ros2', publish_actions=True, dry_run=True)

    with pytest.raises(ValueError, match='--publish-actions requires --no-dry-run'):
        main_dual._validate_backend_contract(args)


def test_contract_summary_mentions_safe_defaults_and_gates() -> None:
    summary = main_dual.contract_summary(main_dual.Args())

    assert 'backend=sdk' in summary
    assert 'ROS 2 defaults to dry-run' in summary
    assert 'publish-actions requires --backend ros2 and --no-dry-run' in summary


def test_build_arg_parser_exposes_ros2_safety_flags() -> None:
    help_text = main_dual.build_arg_parser().format_help()

    for flag in (
        '--backend',
        '--bridge-python',
        '--ros2-config',
        '--dry-run',
        '--no-dry-run',
        '--publish-actions',
    ):
        assert flag in help_text


def test_parse_args_maps_ros2_safety_flags_into_args() -> None:
    args = main_dual.parse_args(['--backend', 'ros2', '--no-dry-run', '--publish-actions'])

    assert args.backend == 'ros2'
    assert args.dry_run is False
    assert args.publish_actions is True


def test_parse_args_accepts_documented_underscore_pi05_options() -> None:
    args = main_dual.parse_args([
        '--mode',
        'remote',
        '--host',
        '0.0.0.0',
        '--port',
        '8000',
        '--prompt',
        'pick up the object',
        '--left_can_port',
        'can_left_doc',
        '--right_can_port',
        'can_right_doc',
        '--high_camera_id',
        '148522073709',
        '--left_wrist_camera_id',
        '6',
        '--right_wrist_camera_id',
        '8',
    ])

    assert args.left_can_port == 'can_left_doc'
    assert args.right_can_port == 'can_right_doc'
    assert args.high_camera_id == '148522073709'
    assert args.left_wrist_camera_id == 6
    assert args.right_wrist_camera_id == 8


def test_parse_args_preserves_dashed_pi05_options() -> None:
    args = main_dual.parse_args([
        '--left-can-port',
        'can_left_dash',
        '--right-can-port',
        'can_right_dash',
        '--high-camera-id',
        'high_dash',
        '--left-wrist-camera-id',
        '7',
        '--right-wrist-camera-id',
        '9',
    ])

    assert args.left_can_port == 'can_left_dash'
    assert args.right_can_port == 'can_right_dash'
    assert args.high_camera_id == 'high_dash'
    assert args.left_wrist_camera_id == 7
    assert args.right_wrist_camera_id == 9


def test_build_environment_omits_ros2_flags_for_sdk_backend(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_create_dual_environment(*, backend: str, **kwargs: object) -> object:
        captured['backend'] = backend
        captured['kwargs'] = kwargs
        return object()

    monkeypatch.setattr(main_dual, 'create_dual_environment', fake_create_dual_environment)

    args = main_dual.Args()
    environment = main_dual._build_environment(args)

    assert environment is not None
    assert captured['backend'] == 'sdk'
    kwargs = captured['kwargs']
    assert isinstance(kwargs, dict)
    assert kwargs['left_can_port'] == 'can_left'
    assert kwargs['right_can_port'] == 'can_right'
    assert kwargs['camera_fps'] == 30
    assert kwargs['prompt'] == 'Fold_the_towel'
    assert 'bridge_python' not in kwargs
    assert 'ros2_config' not in kwargs
    assert 'dry_run' not in kwargs
    assert 'publish_actions' not in kwargs


def test_build_environment_passes_ros2_flags_for_ros2_backend(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_create_dual_environment(*, backend: str, **kwargs: object) -> object:
        captured['backend'] = backend
        captured['kwargs'] = kwargs
        return object()

    monkeypatch.setattr(main_dual, 'create_dual_environment', fake_create_dual_environment)

    args = main_dual.Args(
        backend='ros2',
        bridge_python=Path('/tmp/python'),
        ros2_config=Path('/tmp/config'),
        dry_run=False,
        publish_actions=True,
        prompt='Fold the towel',
    )
    environment = main_dual._build_environment(args)

    assert environment is not None
    assert captured['backend'] == 'ros2'
    kwargs = captured['kwargs']
    assert isinstance(kwargs, dict)
    assert kwargs['bridge_python'] == Path('/tmp/python')
    assert kwargs['ros2_config'] == Path('/tmp/config')
    assert kwargs['dry_run'] is False
    assert kwargs['publish_actions'] is True
    assert kwargs['prompt'] == 'Fold the towel'


def test_sdk_backend_path_does_not_import_ros2_bridge(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, 'ros2_environment', raising=False)
    assert 'ros2_environment' not in sys.modules

    records: dict[str, object] = {}

    class FakeSDKEnv:
        def __init__(self, **kwargs: object) -> None:
            records.update(kwargs)

    monkeypatch.setattr(env_dual, 'PiperDualEnvironment', FakeSDKEnv)

    env = env_dual.create_dual_environment(
        backend='sdk',
        left_can_port='can_left',
        bridge_python=Path('/tmp/python'),
        ros2_config=Path('/tmp/config'),
        dry_run=False,
        publish_actions=True,
        max_action_delta=0.25,
    )

    assert isinstance(env, FakeSDKEnv)
    assert 'ros2_environment' not in sys.modules
    assert records == {'left_can_port': 'can_left'}


def test_ros2_backend_path_uses_ros2_environment_without_instantiating_sdk_hardware(monkeypatch) -> None:
    assert 'piper_dual_controller' not in sys.modules
    assert 'cameras' not in sys.modules

    records: dict[str, object] = {}

    class FakeRos2Env:
        def __init__(self, **kwargs: object) -> None:
            records.update(kwargs)

    fake_module = types.ModuleType('ros2_environment')
    fake_module.Ros2DualEnvironment = FakeRos2Env
    monkeypatch.setitem(sys.modules, 'ros2_environment', fake_module)

    env = env_dual.create_dual_environment(
        backend='ros2',
        bridge_python=Path('/tmp/python'),
        ros2_config=Path('/tmp/config'),
        dry_run=True,
        publish_actions=False,
        prompt='Fold the towel',
        max_action_delta=0.25,
    )

    assert isinstance(env, FakeRos2Env)
    assert records['bridge_python'] == Path('/tmp/python')
    assert records['ros2_config'] == Path('/tmp/config')
    assert records['dry_run'] is True
    assert records['publish_actions'] is False
    assert records['prompt'] == 'Fold the towel'
    assert records['max_action_delta'] == 0.25
    assert 'piper_dual_controller' not in sys.modules
    assert 'cameras' not in sys.modules
