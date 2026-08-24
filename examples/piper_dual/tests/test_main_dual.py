'''Focused tests for the dual-arm deployment and backend-selection contract.'''

from __future__ import annotations

from pathlib import Path
import ast
import math
import subprocess
import sys
import types

import numpy as np

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import env_dual
import main_dual

def _indexed_args_defaults() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ['git', 'show', ':examples/piper_dual/main_dual.py'],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    tree = ast.parse(result.stdout)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'Args':
            defaults: dict[str, object] = {}
            for statement in node.body:
                if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name):
                    continue
                try:
                    defaults[statement.target.id] = ast.literal_eval(statement.value)
                except ValueError:
                    continue
            return defaults
    raise AssertionError('Args dataclass not found in indexed main_dual.py')


def test_indexed_task4_defaults_keep_only_stack_and_safety_changes() -> None:
    defaults = _indexed_args_defaults()

    assert defaults['action_horizon'] == 10
    assert defaults['num_steps'] == 800
    assert defaults['port'] == 8000
    assert defaults['prompt'] == 'Fold_the_towel'
    assert defaults['use_async'] is True
    assert defaults['record_mode'] is True
    assert defaults['backend'] == 'ros2'
    assert defaults['dry_run'] is True
    assert defaults['publish_actions'] is False


def test_validate_backend_contract_rejects_publish_actions_without_ros2_backend() -> None:
    args = main_dual.Args(backend='sdk', publish_actions=True, dry_run=False)

    with pytest.raises(ValueError, match='--publish-actions requires --backend ros2'):
        main_dual._validate_backend_contract(args)


def test_validate_backend_contract_rejects_publish_actions_while_dry_running() -> None:
    args = main_dual.Args(backend='ros2', publish_actions=True, dry_run=True)

    with pytest.raises(ValueError, match='--publish-actions requires --no-dry-run'):
        main_dual._validate_backend_contract(args)


def test_validate_backend_contract_rejects_control_stack_for_sdk_backend() -> None:
    args = main_dual.Args(backend='sdk', control_stack='local-ros')

    with pytest.raises(ValueError, match='--control-stack requires --backend ros2'):
        main_dual._validate_backend_contract(args)


def test_validate_backend_contract_rejects_non_positive_or_non_finite_fps_before_environment(monkeypatch) -> None:
    def fail_if_constructed(args: main_dual.Args) -> object:
        raise AssertionError('environment must not be constructed for invalid fps')

    monkeypatch.setattr(main_dual, '_build_environment', fail_if_constructed)

    for invalid_fps in (0, -1, math.nan, math.inf):
        result = main_dual.main(main_dual.Args(fps=invalid_fps))

        assert result == 2


def test_non_empty_run_tag_must_be_known_and_match_resolved_control_stack() -> None:
    assert main_dual._comparison_out_dir(
        main_dual.Args(out_dir=Path('runs'), control_stack='local-ros', run_tag='local-ros')
    ) == Path('runs/local-ros')

    cases = (
        main_dual.Args(out_dir=Path('runs'), control_stack='official-ros', run_tag='local-ros'),
        main_dual.Args(out_dir=Path('runs'), control_stack='local-ros', run_tag='unknown'),
        main_dual.Args(out_dir=Path('runs'), backend='sdk', run_tag='direct-sdk'),
    )
    for args in cases:
        with pytest.raises(ValueError, match='--run-tag must match selected control stack'):
            main_dual._validate_backend_contract(args)


def test_mismatched_run_tag_rejects_before_environment_creation(monkeypatch) -> None:
    def fail_if_constructed(args: main_dual.Args) -> object:
        raise AssertionError('environment must not be constructed for mismatched run tag')

    monkeypatch.setattr(main_dual, '_build_environment', fail_if_constructed)

    result = main_dual.main(main_dual.Args(control_stack='official-ros', run_tag='local-ros'))

    assert result == 2


def test_contract_summary_mentions_safe_defaults_and_selected_stack() -> None:
    summary = main_dual.contract_summary(main_dual.Args())

    assert 'backend=ros2' in summary
    assert 'selected control stack=official-ros' in summary
    assert 'ROS 2 defaults to dry-run: True' in summary
    assert 'publish-actions requires --backend ros2 and --no-dry-run' in summary


def test_build_arg_parser_exposes_ros2_control_stack_and_safety_flags() -> None:
    help_text = main_dual.build_arg_parser().format_help()

    for flag in (
        '--backend',
        '--bridge-python',
        '--ros2-config',
        '--control-stack',
        '--dry-run',
        '--no-dry-run',
        '--publish-actions',
    ):
        assert flag in help_text


def test_parse_args_defaults_to_safe_ros2_behavior() -> None:
    args = main_dual.parse_args([])

    assert args.backend == 'ros2'
    assert args.control_stack is None
    assert args.dry_run is True
    assert args.publish_actions is False


def test_parse_args_accepts_ros2_control_stack_choices() -> None:
    for name in ('local-ros', 'official-ros', 'direct-sdk'):
        args = main_dual.parse_args(['--backend', 'ros2', '--control-stack', name])
        assert args.control_stack == name


def test_resolve_control_stack_defaults_to_official_ros_for_ros2() -> None:
    assert main_dual._resolve_control_stack(main_dual.Args()) == 'official-ros'
    assert main_dual._resolve_control_stack(main_dual.Args(backend='ros2', control_stack='local-ros')) == 'local-ros'
    assert main_dual._resolve_control_stack(main_dual.Args(backend='sdk', control_stack='local-ros')) is None


def test_parse_args_accepts_max_action_delta_for_ros2_backend() -> None:
    args = main_dual.parse_args(['--backend', 'ros2', '--max-action-delta', '0.25'])

    assert args.max_action_delta == pytest.approx(0.25)


def test_environment_kwargs_forwards_max_action_delta_only_for_ros2_backend() -> None:
    ros2_kwargs = main_dual._environment_kwargs(main_dual.Args(backend='ros2', max_action_delta=0.25))
    sdk_kwargs = main_dual._environment_kwargs(main_dual.Args(backend='sdk', max_action_delta=0.25))

    assert ros2_kwargs['max_action_delta'] == pytest.approx(0.25)
    assert 'max_action_delta' not in sdk_kwargs


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

    monkeypatch.setattr(env_dual, 'create_dual_environment', fake_create_dual_environment)

    args = main_dual.Args(backend='sdk', prompt='Fold the towel')
    environment = main_dual._build_environment(args)

    assert environment is not None
    assert captured['backend'] == 'sdk'
    kwargs = captured['kwargs']
    assert isinstance(kwargs, dict)
    assert kwargs['left_can_port'] == 'can_left'
    assert kwargs['right_can_port'] == 'can_right'
    assert kwargs['camera_fps'] == 30
    assert kwargs['prompt'] == 'Fold the towel'
    assert 'bridge_python' not in kwargs
    assert 'ros2_config' not in kwargs
    assert 'dry_run' not in kwargs
    assert 'publish_actions' not in kwargs
    assert 'control_stack' not in kwargs


def test_build_environment_passes_ros2_flags_for_ros2_backend(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_create_dual_environment(*, backend: str, **kwargs: object) -> object:
        captured['backend'] = backend
        captured['kwargs'] = kwargs
        return object()

    monkeypatch.setattr(env_dual, 'create_dual_environment', fake_create_dual_environment)

    args = main_dual.Args(
        backend='ros2',
        bridge_python=Path('/tmp/python'),
        ros2_config=Path('/tmp/config'),
        dry_run=False,
        publish_actions=True,
        control_stack='direct-sdk',
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
    assert kwargs['control_stack'] == 'direct-sdk'
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

def test_comparison_out_dir_separates_run_tag_and_rejects_path_separators() -> None:
    args = main_dual.Args(out_dir=Path('runs'), control_stack='official-ros', run_tag='official-ros')

    assert main_dual._comparison_out_dir(args) == Path('runs/official-ros')

    for bad_tag in ('.', '..', 'bad/tag', 'bad\\tag'):
        with pytest.raises(ValueError, match='--run-tag must match selected control stack'):
            main_dual._comparison_out_dir(main_dual.Args(out_dir=Path('runs'), run_tag=bad_tag))


def test_build_runtime_routes_run_tagged_output_and_runtime_fps(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeVideoSaver:
        def __init__(self, out_dir: Path, subsample: int = 1, fps: float = 50.0) -> None:
            captured['video_out_dir'] = out_dir
            captured['video_subsample'] = subsample
            captured['video_fps'] = fps

    class FakePlotter:
        def __init__(self, out_dir: Path, broker: object = None, run_tag: str = '') -> None:
            captured['plot_out_dir'] = out_dir
            captured['plot_run_tag'] = run_tag

    class FakeRuntime:
        def __init__(self, *, subscribers: list[object], **kwargs: object) -> None:
            captured['runtime_kwargs'] = kwargs
            captured['subscribers'] = subscribers

    monkeypatch.setitem(sys.modules, 'saver', types.SimpleNamespace(VideoSaver=FakeVideoSaver))
    monkeypatch.setitem(sys.modules, 'plot_dynamics', types.SimpleNamespace(RobotStatePlotter=FakePlotter))
    monkeypatch.setitem(sys.modules, 'openpi_client.runtime.runtime', types.SimpleNamespace(Runtime=FakeRuntime))
    monkeypatch.setitem(
        sys.modules,
        'openpi_client.runtime.agents.policy_agent',
        types.SimpleNamespace(PolicyAgent=lambda policy: ('policy-agent', policy)),
    )

    runtime = main_dual._build_runtime(
        main_dual.Args(run_tag='direct-sdk', control_stack='direct-sdk', fps=30), object(), object()
    )

    assert isinstance(runtime, FakeRuntime)
    assert captured['video_out_dir'] == Path('data/piper_dual/videos/direct-sdk')
    assert captured['plot_out_dir'] == Path('data/piper_dual/videos/direct-sdk')
    assert captured['plot_run_tag'] == 'direct-sdk'
    assert captured['video_fps'] == 30


def test_video_saver_validates_fps_and_uses_subsampled_playback_fps(monkeypatch, tmp_path) -> None:
    import importlib

    calls: dict[str, object] = {}
    fake_imageio = types.SimpleNamespace()

    def fake_mimwrite(path: Path, images: list[np.ndarray], fps: float) -> None:
        calls['path'] = path
        calls['images'] = images
        calls['fps'] = fps

    fake_imageio.mimwrite = fake_mimwrite
    monkeypatch.setitem(sys.modules, 'imageio', fake_imageio)
    monkeypatch.setitem(sys.modules, 'cv2', types.SimpleNamespace())
    monkeypatch.delitem(sys.modules, 'saver', raising=False)
    saver_module = importlib.import_module('saver')

    for invalid_fps in (0, -1, math.nan, math.inf):
        with pytest.raises(ValueError, match='fps must be a positive finite value'):
            saver_module.VideoSaver(tmp_path, fps=invalid_fps)
    with pytest.raises(ValueError, match='subsample must be positive'):
        saver_module.VideoSaver(tmp_path, subsample=0, fps=30)

    video_saver = saver_module.VideoSaver(tmp_path, subsample=2, fps=30)
    video_saver._images = [np.zeros((2, 2, 3), dtype=np.uint8), np.ones((2, 2, 3), dtype=np.uint8)]
    video_saver.on_episode_end()

    assert calls['path'] == tmp_path / 'out_0.mp4'
    assert calls['fps'] == 15
    assert len(calls['images']) == 1


def test_main_records_run_metadata_for_keyboard_interrupt_and_exit_error(monkeypatch, tmp_path) -> None:
    events: list[str] = []

    class FakeRecorder:
        def __init__(self, path: Path, metadata: dict[str, object]) -> None:
            events.append(f'init:{path.name}')
            self.path = path
            self.metadata = metadata
            self.finished: tuple[int, str] | None = None

        def start(self) -> None:
            events.append('start')

        def finish(self, exit_code: int, exit_reason: str) -> None:
            self.finished = (exit_code, exit_reason)
            events.append(f'finish:{exit_code}:{exit_reason}')

    class FakeRuntime:
        def __init__(self, **kwargs: object) -> None:
            events.append('runtime-init')

        def run(self) -> None:
            raise KeyboardInterrupt

        def close(self) -> None:
            events.append('runtime-close')

    class FakeEnvironment:
        def close(self) -> None:
            events.append('environment-close')

    monkeypatch.setitem(sys.modules, 'run_metadata', types.SimpleNamespace(RunMetadataRecorder=FakeRecorder))
    monkeypatch.setattr(main_dual, '_build_environment', lambda args: FakeEnvironment())
    monkeypatch.setattr(main_dual, '_build_policy', lambda args: object())
    monkeypatch.setattr(main_dual, '_build_runtime', lambda args, environment, policy: FakeRuntime())
    monkeypatch.setattr(main_dual, '_comparison_out_dir', lambda args: tmp_path / 'direct-sdk')
    monkeypatch.setattr(main_dual, 'print', lambda *a, **k: None, raising=False)

    result = main_dual.main(main_dual.Args(run_tag='direct-sdk', control_stack='direct-sdk', fps=30))

    assert result == 130
    assert 'start' in events
    assert any(item.startswith('finish:130:keyboard_interrupt') for item in events)


def test_main_records_run_metadata_for_runtime_error(monkeypatch, tmp_path) -> None:
    events: list[str] = []

    class FakeRecorder:
        def __init__(self, path: Path, metadata: dict[str, object]) -> None:
            events.append(f'init:{path.name}')

        def start(self) -> None:
            events.append('start')

        def finish(self, exit_code: int, exit_reason: str) -> None:
            events.append(f'finish:{exit_code}:{exit_reason}')

    class FakeRuntime:
        def run(self) -> None:
            raise RuntimeError('boom')

        def close(self) -> None:
            events.append('runtime-close')

    class FakeEnvironment:
        def close(self) -> None:
            events.append('environment-close')

    monkeypatch.setitem(sys.modules, 'run_metadata', types.SimpleNamespace(RunMetadataRecorder=FakeRecorder))
    monkeypatch.setattr(main_dual, '_build_environment', lambda args: FakeEnvironment())
    monkeypatch.setattr(main_dual, '_build_policy', lambda args: object())
    monkeypatch.setattr(main_dual, '_build_runtime', lambda args, environment, policy: FakeRuntime())
    monkeypatch.setattr(main_dual, '_comparison_out_dir', lambda args: tmp_path / 'direct-sdk')
    monkeypatch.setattr(main_dual, 'print', lambda *a, **k: None, raising=False)

    result = main_dual.main(main_dual.Args(run_tag='direct-sdk', control_stack='direct-sdk', fps=30))

    assert result == 1
    assert 'start' in events
    assert any(item.startswith('finish:1:RuntimeError: boom') for item in events)


def test_main_records_run_metadata_for_normal_exit(monkeypatch, tmp_path) -> None:
    events: list[str] = []

    class FakeRecorder:
        def __init__(self, path: Path, metadata: dict[str, object]) -> None:
            events.append(f'init:{path.name}')

        def start(self) -> None:
            events.append('start')

        def finish(self, exit_code: int, exit_reason: str) -> None:
            events.append(f'finish:{exit_code}:{exit_reason}')

    class FakeRuntime:
        def run(self) -> None:
            events.append('runtime-run')

        def close(self) -> None:
            events.append('runtime-close')

    class FakeEnvironment:
        def close(self) -> None:
            events.append('environment-close')

    monkeypatch.setitem(sys.modules, 'run_metadata', types.SimpleNamespace(RunMetadataRecorder=FakeRecorder))
    monkeypatch.setattr(main_dual, '_build_environment', lambda args: FakeEnvironment())
    monkeypatch.setattr(main_dual, '_build_policy', lambda args: object())
    monkeypatch.setattr(main_dual, '_build_runtime', lambda args, environment, policy: FakeRuntime())
    monkeypatch.setattr(main_dual, '_comparison_out_dir', lambda args: tmp_path / 'direct-sdk')
    monkeypatch.setattr(main_dual, 'print', lambda *a, **k: None, raising=False)

    result = main_dual.main(main_dual.Args(run_tag='direct-sdk', control_stack='direct-sdk', fps=30))

    assert result == 0
    assert 'runtime-run' in events
    assert 'finish:0:completed' in events


def test_three_stack_runbook_contains_exact_manual_command_blocks() -> None:
    readme = (Path(__file__).resolve().parents[1] / 'README.md').read_text(encoding='utf-8')

    required_snippets = (
        'bash can_config.sh',
        'source /home/agilex/piper_ros/install/setup.bash',
        "ros2 launch piper start_two_piper.launch.py \\\n  can_left_port:=can_left can_right_port:=can_right auto_enable:=false",
        'ros2 topic echo --once /arm_status_left',
        'ros2 topic echo --once /arm_status_right',
        'ros2 service call /piper_left_ctrl_node/enable_srv piper_msgs/srv/Enable "{enable_request: true}"',
        'ros2 service call /piper_right_ctrl_node/enable_srv piper_msgs/srv/Enable "{enable_request: true}"',
        '--control-stack local-ros',
        '--run-tag local-ros',
        'source /home/agilex/piper_ros/.worktrees/piper-official-humble/install-official/setup.bash',
        "ros2 launch piper_official_bringup start_two_piper_official.launch.py \\\n  can_left_port:=can_left can_right_port:=can_right auto_enable:=false",
        '--control-stack official-ros',
        '--run-tag official-ros',
        'export PYTHONPATH=/home/agilex/lgd/control_your_robot/src:$PYTHONPATH',
        "/usr/bin/python3 examples/piper_dual/piper_direct_sdk_adapter.py \\\n  --left-can can_left --right-can can_right --allow-enable",
        '--control-stack direct-sdk',
        '--run-tag direct-sdk',
        "pgrep -af 'piper_single_ctrl|piper_direct_sdk_adapter|main_dual.py|ros2_bridge_process'",
        'ip -details link show type can',
        'local-ros=30%',
        'official-ros=30%',
        'direct-sdk=100%',
        '不是等速对比实验',
    )
    for snippet in required_snippets:
        assert snippet in readme
