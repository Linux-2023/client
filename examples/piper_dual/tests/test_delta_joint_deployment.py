from pathlib import Path
import importlib.util
import hashlib
import json
import sys
from unittest.mock import Mock

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import main_dual_delta_joint_ros as entry
from joint_deployment import FIELDS, UNITS, CheckedJointPolicy, load_deployment, server_metadata, validate_metadata

@pytest.fixture
def deployment(tmp_path_factory):
    """Bind isolated deployment evidence without private workstation artifacts."""
    root = tmp_path_factory.mktemp('joint-deployment')
    model = root / 'model'
    asset_id = 'fixture/piper_joint'
    config_name = 'fixture_pi05_joint'
    prompt = 'Move the fixture object.'
    checkpoint = str(root / 'experiment' / '49999')
    norm_path = model / 'assets' / asset_id / 'norm_stats.json'
    norm_path.parent.mkdir(parents=True)
    stats = {'mean': [0.] * 14, 'std': [1.] * 14,
             'q01': [-1.] * 14, 'q99': [1.] * 14}
    norm_path.write_text(json.dumps({'norm_stats': {'state': stats, 'actions': stats}}))
    norm_sha = hashlib.sha256(norm_path.read_bytes()).hexdigest()
    # Manifest validation checks file presence; no model is loaded by these tests.
    (model / 'model.safetensors').touch()
    (model / 'conversion_report.json').write_text(json.dumps({
        'config': config_name, 'asset_id': asset_id, 'norm_sha256': norm_sha,
        'action_horizon': 50, 'physical_action_dim': 14,
    }))
    contract = {'fields': FIELDS, 'units': UNITS}

    def evidence(name, value):
        path = root / (name + ('.jsonl' if name == 'tasks' else '.json'))
        text = ('\n'.join(json.dumps(row) for row in value)
                if name == 'tasks' else json.dumps(value)) + '\n'
        path.write_text(text)
        return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    training = evidence('spec', {
        'base_config': config_name, 'repo_id': asset_id, 'dataset_fps': 30,
        'training_action_contract': contract, 'action_representation': 'delta_joint_current_v1',
        'state_conditioning': 'none',
    })
    launch = evidence('launch', {
        'spec_sha256': training['sha256'], 'training_action_contract': contract,
        'command': ['train.py', config_name, '--data.repo-id', asset_id,
                    '--model.action-horizon', '50', '--model.action-dim', '32',
                    '--model.max-token-len', '200', '--model.pi05', '--model.no-discrete-state-input'],
    })
    smoke = evidence('checkpoint_smoke', {
        'spec_sha256': training['sha256'], 'action_shape': [50, 14],
        'norm_stats_sha256': norm_sha, 'checkpoint': checkpoint, 'passed': True,
    })
    spec = {
        'config': config_name, 'asset_id': asset_id, 'model': str(model),
        'checkpoint': checkpoint, 'norm_sha256': norm_sha, 'fields': FIELDS, 'units': UNITS,
        'model_horizon': 50, 'model_fps': 30, 'prompt': prompt,
        'action_representation': 'delta_joint_current_v1', 'server_output': 'absolute_joint',
        'training_evidence': {'spec': training, 'launch': launch, 'checkpoint_smoke': smoke,
                              'clock': evidence('clock', {'fps': 30}),
                              'tasks': evidence('tasks', [{'task': prompt}])},
    }
    path = root / 'deployment.json'
    path.write_text(json.dumps(spec))
    return path


def observation():
    return {'state': np.linspace(.1, .5, 14).astype(np.float32),
            'images': {name: np.zeros((3, 224, 224), np.uint8)
                       for name in ('cam_high', 'cam_left_wrist', 'cam_right_wrist')}}




@pytest.mark.parametrize('key,value', [('server_output', 'absolute_eef'), ('joint_unit', 'deg'),
    ('gripper_unit', 'normalized'), ('action_representation', 'delta_eef_current_rpy_v1'),
    ('norm_sha256', 'wrong'), ('policy_config', 'other_model'), ('model_fps', 15)])
def test_reject_wrong_server_contract(key, value, deployment):
    spec = load_deployment(deployment)
    metadata = server_metadata(spec)
    metadata[key] = value
    with pytest.raises(ValueError, match='Wrong joint server'):
        validate_metadata(metadata, spec)




def test_async_failure_latched_before_next_publish(deployment):
    spec = load_deployment(deployment)
    base = Mock()
    base.get_server_metadata.return_value = server_metadata(spec)
    base.infer.side_effect = ConnectionError('connection lost')
    checked = CheckedJointPolicy(base, spec)
    with pytest.raises(ConnectionError): checked.infer(observation())
    env = Mock()
    env.is_episode_complete.return_value = False
    args = entry.original.Args(num_episodes=1)
    with pytest.raises(RuntimeError, match='connection lost'):
        entry.run_episodes(args, env, Mock(), checked, Mock())
    env.apply_action.assert_not_called()


def test_wrong_metadata_does_not_create_ros_backend(monkeypatch, tmp_path, deployment):
    from openpi_client import websocket_client_policy
    connection = Mock()
    connection.get_server_metadata.return_value = {'server_output':'absolute_eef'}
    monkeypatch.setattr(websocket_client_policy, 'WebsocketClientPolicy', Mock(return_value=connection))
    environment = Mock()
    monkeypatch.setattr(entry.original, '_build_environment', environment)
    result = entry.main(['--deployment-config', str(deployment), '--out-dir', str(tmp_path),
                         '--no-dry-run', '--publish-actions', '--no-record-ros-diagnostics'])
    assert result == 1
    environment.assert_not_called()
    connection._ws.close.assert_called_once()
    assert list(tmp_path.glob('*/client/summary.json'))


@pytest.mark.parametrize('interrupted', [False, True])
def test_sync_episode_records_three_cameras_and_commands(monkeypatch, tmp_path, interrupted, deployment):
    from openpi_client import websocket_client_policy
    spec = load_deployment(deployment)
    connection = Mock()
    connection.get_server_metadata.return_value = server_metadata(spec)
    connection.infer.return_value = {'actions': np.full((50,14), .02, np.float32)}
    monkeypatch.setattr(websocket_client_policy, 'WebsocketClientPolicy', Mock(return_value=connection))
    env = Mock()
    env.is_episode_complete.side_effect = [False, False, True]
    env.get_observation.return_value = observation()
    if interrupted:
        env.apply_action.side_effect = KeyboardInterrupt
    monkeypatch.setattr(entry.original, '_build_environment', Mock(return_value=env))
    result = entry.main(['--deployment-config', str(deployment), '--out-dir', str(tmp_path),
                         '--no-use-async', '--no-use-rtc', '--no-record-ros-diagnostics'])
    assert result == (130 if interrupted else 0)
    assert env.apply_action.call_count == (1 if interrupted else 2)
    run = next(tmp_path.iterdir())
    summary = json.loads((run/'client/summary.json').read_text())
    assert summary['observations'] == (1 if interrupted else 2) and summary['requests'] == 1
    assert summary['reason'] == ('interrupted' if interrupted else 'completed')
    tree = json.loads((run/'client/observations/000000.json').read_text())
    assert set(tree['images']) == set(observation()['images'])
    assert (run/'client/requests/000000_output.npz').exists()
    env.close.assert_called_once()


def test_pyagxarm_js_receives_radians_and_gripper_meters():
    location = HERE.parent/'piper_dual_pyagxarm/arm.py'
    spec = importlib.util.spec_from_file_location('joint_test_pyagxarm', location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    robot, gripper = Mock(), Mock()
    arm = module.Arm(robot, gripper, [], motion_mode='js')
    target = [.1, .2, -.3, .4, .5, -.6, .08]
    arm.send(arm.validate_command(target, [0]*6+[100], [0]*6+[1]))
    robot.move_js.assert_called_once_with(target[:6])
    robot.move_j.assert_not_called()
    gripper.move_gripper_m.assert_called_once_with(value=.08, force=1.)




def test_client_rejects_frequency_different_from_training(deployment):
    spec = load_deployment(deployment)
    ns = entry.build_arg_parser().parse_args(['--deployment-config', str(deployment), '--fps', '15'])
    with pytest.raises(ValueError, match='match training dataset fps=30'):
        entry.validate_args(ns, spec)


@pytest.mark.parametrize('key,value', [('model_fps', 15), ('model_horizon', 22), ('prompt', 'wrong task')])
def test_deployment_cannot_override_training_evidence(tmp_path, key, value, deployment):
    spec = json.loads(deployment.read_text())
    spec[key] = value
    path = tmp_path/'deployment.json'
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='disagrees with training evidence'):
        load_deployment(path)
