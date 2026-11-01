"""Checkpoint identity and SI joint contract shared by the two deployment entrypoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
JOINTS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
FIELDS = [f"{arm}_{name}" for arm in ("left", "right")
          for name in (*[f"joint_{i}" for i in range(1, 7)], "gripper")]
UNITS = ["rad"] * 6 + ["m"] + ["rad"] * 6 + ["m"]


def validate_training_evidence(spec):
    evidence = {}
    for name, record in spec['training_evidence'].items():
        data = Path(record['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError(f'Training evidence hash mismatch: {name}')
        evidence[name] = ([json.loads(line) for line in data.decode().splitlines() if line.strip()]
                          if name == 'tasks' else json.loads(data))
    train, launch, smoke = evidence['spec'], evidence['launch'], evidence['checkpoint_smoke']
    if launch['spec_sha256'] != spec['training_evidence']['spec']['sha256'] or smoke['spec_sha256'] != launch['spec_sha256']:
        raise ValueError('Training spec is not bound to the final checkpoint smoke report')
    command = launch['command']
    def option(name):
        return command[command.index(name)+1]
    contract = train['training_action_contract']
    checks = {
        'config': train['base_config'] == spec['config'] and spec['config'] in command,
        'dataset': train['repo_id'] == spec['asset_id'] == option('--data.repo-id'),
        'clock': train['dataset_fps'] == evidence['clock']['fps'] == spec['model_fps'],
        'horizon': int(option('--model.action-horizon')) == spec['model_horizon'] == smoke['action_shape'][0],
        'fields': contract['fields'] == spec['fields'] and contract['units'] == spec['units'],
        'action_mode': train['action_representation'] == spec['action_representation'],
        'action_contract': contract == launch['training_action_contract'],
        'state_conditioning': train['state_conditioning'] == 'none' and '--model.no-discrete-state-input' in command,
        'norm': smoke['norm_stats_sha256'] == spec['norm_sha256'],
        'physical_dim': smoke['action_shape'][1] == len(spec['fields']) == 14,
        'checkpoint': smoke['passed'] and Path(smoke['checkpoint']).parts[-2:] == Path(spec['checkpoint']).parts[-2:],
        'prompt': spec['prompt'] in [task['task'] for task in evidence['tasks']],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f'Deployment disagrees with training evidence: {failed}')
    return dict(pi05='--model.pi05' in command, discrete_state_input=False,
                action_dim=int(option('--model.action-dim')), action_horizon=spec['model_horizon'],
                max_token_len=int(option('--model.max-token-len')))


def load_deployment(path):
    spec = json.loads(Path(path).read_text())
    spec['training_model'] = validate_training_evidence(spec)
    if spec["action_representation"] != "delta_joint_current_v1" or spec["server_output"] != "absolute_joint":
        raise ValueError("Deployment must describe delta_joint_current_v1 -> absolute_joint")
    if spec["fields"] != FIELDS or spec["units"] != UNITS:
        raise ValueError("Expected left six joints rad + gripper m, then right")
    if spec["model_horizon"] != 50 or spec["model_fps"] != 30:
        raise ValueError("These checkpoints require a 50-step model window at 30 Hz")
    model = Path(spec["model"]).resolve()
    report = json.loads((model / "conversion_report.json").read_text())
    for key in ("config", "asset_id", "norm_sha256"):
        if report[key] != spec[key]:
            raise ValueError(f"Checkpoint conversion report disagrees with {key}")
    if report["action_horizon"] != 50 or report["physical_action_dim"] != 14:
        raise ValueError("Wrong checkpoint action shape")
    norm = model / "assets" / spec["asset_id"] / "norm_stats.json"
    if not norm.resolve().is_relative_to(model):
        raise ValueError("Norm path escapes checkpoint")
    if hashlib.sha256(norm.read_bytes()).hexdigest() != spec["norm_sha256"]:
        raise ValueError("Checkpoint norm hash mismatch")
    if not (model / "model.safetensors").is_file():
        raise FileNotFoundError(model / "model.safetensors")
    return spec


def server_metadata(spec):
    return dict(policy_config=spec["config"], checkpoint_dir=str(Path(spec["model"]).resolve()),
                action_representation="delta_joint_current_v1", server_output="absolute_joint",
                anchor="request_state", joint_unit="rad", gripper_unit="m",
                norm_sha256=spec["norm_sha256"], action_fields=FIELDS, action_units=UNITS,
                action_horizon=spec['model_horizon'], model_fps=spec['model_fps'],
                rtc_orientation={"mode": "legacy", "gradient_semantics": "normalized-euclidean"})


def validate_metadata(metadata, spec):
    for key, expected in server_metadata(spec).items():
        actual = metadata.get(key)
        if key == "checkpoint_dir" and isinstance(actual, str):
            actual = str(Path(actual).resolve())
        if actual != expected:
            raise ValueError(f"Wrong joint server: {key}={actual!r}; expected {expected!r}")


class CheckedJointPolicy:
    """The server already decoded deltas. Never add state to its targets again."""

    def __init__(self, policy, spec):
        validate_metadata(policy.get_server_metadata(), spec)
        self.policy = policy
        self.failure = None

    def get_server_metadata(self):
        return self.policy.get_server_metadata()

    def reset(self):
        return self.policy.reset()

    def infer(self, obs):
        self.raise_if_failed()
        try:
            return self._infer(obs)
        except Exception as exc:
            # The legacy asynchronous broker logs worker errors and continues.
            # Keep a latch so this entrypoint stops before publishing again.
            self.failure = exc
            raise

    def raise_if_failed(self):
        if self.failure is not None:
            raise RuntimeError(f"Joint inference failed: {self.failure}") from self.failure

    def _infer(self, obs):
        state = np.asarray(obs["state"])
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("Expected 14 finite measured joint/gripper values")
        for name in CAMERAS:
            image = np.asarray(obs["images"][name])
            if image.shape != (3, 224, 224) or image.dtype != np.uint8:
                raise ValueError(f"{name} must be RGB uint8 CHW 224x224")
        result = self.policy.infer(obs)
        actions = np.asarray(result["actions"])
        if actions.shape != (50, 14) or not np.isfinite(actions).all():
            raise ValueError("Joint server must return 50x14 finite absolute targets")
        return result
