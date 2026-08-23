# Piper Dual EEF XYZ3D ROS 2 Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent ROS 2 client that sends 14D XYZ+RPY EEF observations to the existing XYZ3D policy server and publishes its validated 14D actions as dual-arm Piper `PosCmd` commands.

**Architecture:** Add dedicated XYZ3D observation and action adapters beside the existing 20D rot6d adapters. Add a standalone `main_dual_eef_xyz3d_ros.py` entrypoint that reuses the existing bridge, environment, WebSocket policy, action broker, runtime, and ROS topics while injecting only the new 14D adapters. Keep the existing rot6d client unchanged.

**Tech Stack:** Python 3.11 client, NumPy, ROS 2 Humble bridge through `/usr/bin/python3`, OpenPI WebSocket client/runtime, pytest.

## Global Constraints

- External state/action layout is 14D `[left_xyz,left_rpy,left_gripper,right_xyz,right_rpy,right_gripper]`.
- XYZ and grippers use meters; roll, pitch, and yaw use radians.
- The checkpoint path is `/home/agilex/checkpoints/pi05_piper_dual_stack_cups_eef_xyz3d_jax_step10000_pytorch`.
- The server config is `pi05_piper_dual_stack_cups_eef_xyz3d`.
- Checkpoint model metadata remains `action_dim=32` and `action_horizon=50`; the server output transform exposes 14D.
- The client defaults to `action_horizon=50`, `fps=30`, `dry_run=True`, and `publish_actions=False`.
- Real publication requires both `--no-dry-run` and `--publish-actions`.
- Existing EEF observation topics remain `/puppet/end_pose_left` and `/puppet/end_pose_right`.
- Existing EEF action topics remain `/pos_left_cmd` and `/pos_right_cmd`.
- Do not modify `main_dual_eef_ros.py`, `eef_action_adapter.py`, `eef_observation_adapter.py`, the checkpoint, normalization assets, server transforms, bridge protocol, or controller behavior.
- Run pytest with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` to avoid unrelated system ROS pytest plugins.

---

### Task 1: Implement dedicated 14D XYZ3D adapters

**Files:**
- Create: `examples/piper_dual/tests/test_eef_xyz3d_adapters.py`
- Create: `examples/piper_dual/eef_xyz3d_action_adapter.py`
- Create: `examples/piper_dual/eef_xyz3d_observation_adapter.py`
- Reference only: `examples/piper_dual/eef_action_adapter.py`
- Reference only: `examples/piper_dual/eef_observation_adapter.py`

**Interfaces:**
- Consumes: synchronized bridge messages containing `eef.puppet_left`, `eef.puppet_right`, 14D `action`, images, timestamps, sync metadata, and task.
- Produces: `EefXyz3dActionAdapter.validate(action) -> np.ndarray`, `EefXyz3dActionAdapter.to_bridge_request(action) -> dict[str, object]`, `EefXyz3dObservationAdapter.adapt(message) -> dict[str, object]`, and `EefXyz3dObservationAdapter.camera_specs`.

- [ ] **Step 1: Write failing action-adapter contract tests**

Create `examples/piper_dual/tests/test_eef_xyz3d_adapters.py` with imports and action tests:

```python
from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eef_xyz3d_action_adapter import EefXyz3dActionAdapter
from eef_xyz3d_observation_adapter import EefXyz3dObservationAdapter


def test_xyz3d_action_adapter_splits_dual_arm_poscmd_values() -> None:
    action = np.array(
        [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.07,
         -0.8, 0.9, 1.0, -1.1, 1.2, -1.3, 0.14],
        dtype=np.float32,
    )

    request = EefXyz3dActionAdapter().to_bridge_request(action)

    assert request["type"] == "publish_eef_action"
    np.testing.assert_allclose(request["left"], action[:7])
    np.testing.assert_allclose(request["right"], action[7:])


@pytest.mark.parametrize(
    "action, message",
    [
        (np.zeros(13), "exactly fourteen"),
        (np.zeros((1, 14)), "exactly fourteen"),
        (["bad"] * 14, "numeric"),
        (np.array([0.0] * 13 + [np.inf]), "finite"),
    ],
)
def test_xyz3d_action_adapter_rejects_invalid_actions(action: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EefXyz3dActionAdapter().to_bridge_request(action)
```

- [ ] **Step 2: Write failing observation-adapter contract tests**

Append helpers and observation tests to the same module:

```python
def _jpeg_bytes(value: int) -> bytes:
    image = np.full((8, 12, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _message() -> dict[str, object]:
    master = np.arange(14, dtype=np.float32) / 100.0
    return {
        "eef": {
            "puppet_left": [0.1, 0.2, 0.3, -0.4, 0.5, -0.6],
            "puppet_right": [-0.7, 0.8, 0.9, 1.0, -1.1, 1.2],
        },
        "action": master,
        "images": {
            "cam_high": _jpeg_bytes(10),
            "cam_left_wrist": _jpeg_bytes(20),
            "cam_right_wrist": _jpeg_bytes(30),
        },
        "timestamps": {
            "cam_high": 1.0,
            "cam_left_wrist": 1.0,
            "cam_right_wrist": 1.0,
        },
        "sync_error": 0.0,
        "task": "Stack the cups",
    }


def test_xyz3d_observation_adapter_builds_exact_14d_state() -> None:
    adapted = EefXyz3dObservationAdapter(task="Stack the cups").adapt(_message())

    expected = np.array(
        [0.1, 0.2, 0.3, -0.4, 0.5, -0.6, 0.06,
         -0.7, 0.8, 0.9, 1.0, -1.1, 1.2, 0.13],
        dtype=np.float32,
    )
    np.testing.assert_allclose(adapted["observation.state"], expected)
    assert adapted["observation.state"].shape == (14,)
    assert adapted["observation.state"].dtype == np.float32
    assert adapted["prompt"] == "Stack the cups"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda item: item["eef"].update(puppet_left=[0.0] * 5), "six numeric values"),
        (lambda item: item["eef"].update(puppet_right=[0.0] * 5 + [np.nan]), "finite"),
        (lambda item: item.update(action=np.zeros(13)), "fourteen numeric values"),
        (lambda item: item.update(action=np.array([0.0] * 13 + [np.inf])), "finite"),
    ],
)
def test_xyz3d_observation_adapter_rejects_invalid_state_inputs(mutate, message: str) -> None:
    item = _message()
    mutate(item)

    with pytest.raises(ValueError, match=message):
        EefXyz3dObservationAdapter().adapt(item)
```

- [ ] **Step 3: Run the adapter tests and verify the missing modules fail**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  examples/piper_dual/tests/test_eef_xyz3d_adapters.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'eef_xyz3d_action_adapter'`.

- [ ] **Step 4: Implement the XYZ3D action adapter**

Create `examples/piper_dual/eef_xyz3d_action_adapter.py`:

```python
"""XYZ+RPY EEF action adapter for the Piper dual ROS 2 contract."""

from __future__ import annotations

from typing import Sequence

import numpy as np

ACTION_DIMENSION = 14
ARM_ACTION_DIMENSION = 7


class EefXyz3dActionAdapter:
    """Validate 14D XYZ+RPY EEF actions and encode Piper PosCmd requests."""

    def validate(self, action: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(action)
        if values.ndim != 1 or values.shape != (ACTION_DIMENSION,):
            raise ValueError("XYZ3D EEF action must contain exactly fourteen values")
        if values.dtype.kind not in "fiu":
            raise ValueError("XYZ3D EEF action values must be numeric")
        values = values.astype(np.float32, copy=False)
        if not np.isfinite(values).all():
            raise ValueError("XYZ3D EEF action values must be finite")
        return values

    def to_bridge_request(self, action: Sequence[float] | np.ndarray) -> dict[str, object]:
        values = self.validate(action)
        return {
            "type": "publish_eef_action",
            "left": values[:ARM_ACTION_DIMENSION].tolist(),
            "right": values[ARM_ACTION_DIMENSION:].tolist(),
        }
```

- [ ] **Step 5: Implement the XYZ3D observation adapter**

Create `examples/piper_dual/eef_xyz3d_observation_adapter.py`:

```python
"""XYZ+RPY EEF observation adapter for the Piper dual ROS 2 contract."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from observation_adapter import CameraSpec
from observation_adapter import DEFAULT_CAMERA_SPECS
from observation_adapter import ObservationAdapter

OBSERVATION_DIMENSION = 14
EEF_POSE_DIMENSION = 6


def _validate_pose(name: str, pose: object) -> np.ndarray:
    values = np.asarray(pose)
    if values.ndim != 1 or values.shape != (EEF_POSE_DIMENSION,) or values.dtype.kind not in "fiu":
        raise ValueError(f"Observation {name} must contain exactly six numeric values")
    values = values.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"Observation {name} values must be finite")
    return values


class EefXyz3dObservationAdapter:
    """Adapt synchronized images, puppet XYZ+RPY poses, and master grippers."""

    def __init__(self, cameras: Sequence[CameraSpec] = DEFAULT_CAMERA_SPECS, task: str = "") -> None:
        self._image_adapter = ObservationAdapter(cameras=cameras, task=task)

    @property
    def camera_specs(self) -> tuple[CameraSpec, ...]:
        return self._image_adapter.camera_specs

    def adapt(self, message: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(message, dict):
            raise ValueError("Observation message must be a JSON object")
        eef = message.get("eef")
        if not isinstance(eef, dict):
            raise ValueError("Observation message must include eef")
        if "puppet_left" not in eef or "puppet_right" not in eef:
            raise ValueError("Observation eef must include puppet_left and puppet_right")
        left = _validate_pose("puppet_left", eef["puppet_left"])
        right = _validate_pose("puppet_right", eef["puppet_right"])

        master = np.asarray(message.get("action"))
        if master.ndim != 1 or master.shape != (14,) or master.dtype.kind not in "fiu":
            raise ValueError("Observation action must contain exactly fourteen numeric values")
        master = master.astype(np.float32, copy=False)
        if not np.isfinite(master).all():
            raise ValueError("Observation action values must be finite")

        state = np.empty((OBSERVATION_DIMENSION,), dtype=np.float32)
        state[:6] = left
        state[6] = master[6]
        state[7:13] = right
        state[13] = master[13]

        image_message = dict(message)
        image_message["state"] = np.zeros(14, dtype=np.float32)
        adapted = self._image_adapter.adapt(image_message)
        adapted["observation.state"] = state
        return adapted
```

- [ ] **Step 6: Run focused adapter tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  examples/piper_dual/tests/test_eef_xyz3d_adapters.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit the adapter slice**

```bash
git add \
  examples/piper_dual/eef_xyz3d_action_adapter.py \
  examples/piper_dual/eef_xyz3d_observation_adapter.py \
  examples/piper_dual/tests/test_eef_xyz3d_adapters.py
git commit -m "feat: add xyz3d eef client adapters"
```

---

### Task 2: Implement the independent XYZ3D deployment entrypoint

**Files:**
- Create: `examples/piper_dual/tests/test_main_dual_eef_xyz3d_ros.py`
- Create: `examples/piper_dual/main_dual_eef_xyz3d_ros.py`
- Reference only: `examples/piper_dual/main_dual_eef_ros.py`

**Interfaces:**
- Consumes: `EefXyz3dActionAdapter`, `EefXyz3dObservationAdapter`, `Ros2BackendClient`, `Ros2DualEnvironment`, `ActionChunkBroker`, `ActionChunkBroker_RTC`, `WebsocketClientPolicy`, `Runtime`, and `VideoSaver`.
- Produces: `Args`, `build_arg_parser()`, `parse_args(argv=None)`, `_validate_xyz3d_contract(args)`, `contract_summary(args)`, `_build_environment(args)`, `_build_policy(args)`, `_build_runtime(args, environment, policy)`, and `main(args=None)` in `main_dual_eef_xyz3d_ros.py`.

- [ ] **Step 1: Write failing parser, safety, and environment-construction tests**

Create `examples/piper_dual/tests/test_main_dual_eef_xyz3d_ros.py`:

```python
"""Tests for the independent XYZ3D EEF ROS 2 deployment entrypoint."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main_dual_eef_xyz3d_ros


def test_parser_defaults_to_safe_xyz3d_contract() -> None:
    args = main_dual_eef_xyz3d_ros.parse_args([])

    assert args.action_horizon == 50
    assert args.fps == 30
    assert args.dry_run is True
    assert args.publish_actions is False
    assert args.eef_left_topic == "/puppet/end_pose_left"
    assert args.eef_right_topic == "/puppet/end_pose_right"
    assert args.eef_left_action_topic == "/pos_left_cmd"
    assert args.eef_right_action_topic == "/pos_right_cmd"
    assert args.bridge_python == Path("/usr/bin/python3")


def test_parser_exposes_xyz3d_topics_and_safety_flags() -> None:
    help_text = main_dual_eef_xyz3d_ros.build_arg_parser().format_help()

    for flag in (
        "--eef-left-topic",
        "--eef-right-topic",
        "--eef-left-action-topic",
        "--eef-right-action-topic",
        "--dry-run",
        "--no-dry-run",
        "--publish-actions",
        "--max-action-delta",
    ):
        assert flag in help_text


def test_publish_actions_requires_no_dry_run() -> None:
    args = main_dual_eef_xyz3d_ros.Args(publish_actions=True, dry_run=True)

    with pytest.raises(ValueError, match="--publish-actions requires --no-dry-run"):
        main_dual_eef_xyz3d_ros._validate_xyz3d_contract(args)


def test_contract_summary_names_14d_xyz3d_layout() -> None:
    summary = main_dual_eef_xyz3d_ros.contract_summary(main_dual_eef_xyz3d_ros.Args())

    assert "14D" in summary
    assert "left_xyzrpy" in summary
    assert "right_xyzrpy" in summary


def test_build_environment_constructs_xyz3d_backend_and_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeBackend:
        def __init__(self, **kwargs: object) -> None:
            captured["backend"] = kwargs

    class FakeEnvironment:
        def __init__(self, **kwargs: object) -> None:
            captured["environment"] = kwargs

    monkeypatch.setattr(main_dual_eef_xyz3d_ros, "Ros2BackendClient", FakeBackend)
    monkeypatch.setattr(main_dual_eef_xyz3d_ros, "Ros2DualEnvironment", FakeEnvironment)

    args = main_dual_eef_xyz3d_ros.Args(
        prompt="Stack the cups",
        eef_left_topic="/left_pose",
        eef_right_topic="/right_pose",
        eef_left_action_topic="/left_cmd",
        eef_right_action_topic="/right_cmd",
        max_action_delta=0.2,
    )
    environment = main_dual_eef_xyz3d_ros._build_environment(args)

    assert isinstance(environment, FakeEnvironment)
    backend_kwargs = captured["backend"]
    assert backend_kwargs["eef_control"] is True
    assert backend_kwargs["eef_left_topic"] == "/left_pose"
    assert backend_kwargs["eef_right_topic"] == "/right_pose"
    assert backend_kwargs["eef_left_action_topic"] == "/left_cmd"
    assert backend_kwargs["eef_right_action_topic"] == "/right_cmd"
    assert backend_kwargs["action_adapter"].__class__.__name__ == "EefXyz3dActionAdapter"
    env_kwargs = captured["environment"]
    assert env_kwargs["prompt"] == "Stack the cups"
    assert env_kwargs["max_action_delta"] == 0.2
    assert env_kwargs["observation_adapter"].__class__.__name__ == "EefXyz3dObservationAdapter"
    assert env_kwargs["action_adapter"].__class__.__name__ == "EefXyz3dActionAdapter"


def test_main_rejects_unsafe_publish_before_environment_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fail_build(_: main_dual_eef_xyz3d_ros.Args) -> object:
        nonlocal called
        called = True
        raise AssertionError("environment must not be created")

    monkeypatch.setattr(main_dual_eef_xyz3d_ros, "_build_environment", fail_build)

    result = main_dual_eef_xyz3d_ros.main(
        main_dual_eef_xyz3d_ros.Args(publish_actions=True, dry_run=True)
    )

    assert result == 2
    assert called is False
```

- [ ] **Step 2: Run the entrypoint tests and verify the missing module fails**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  examples/piper_dual/tests/test_main_dual_eef_xyz3d_ros.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'main_dual_eef_xyz3d_ros'`.

- [ ] **Step 3: Create the independent entrypoint**

Create `examples/piper_dual/main_dual_eef_xyz3d_ros.py` by retaining the imports, runtime lifecycle, signal handling, error handling, policy construction, runtime construction, EEF topics, and environment timeouts from `main_dual_eef_ros.py`, with these exact contract changes:

```python
#!/usr/bin/env python3
"""ROS 2-only dual-arm Piper XYZ+RPY EEF policy deployment entrypoint."""

# Keep the existing standard imports and OPENPI client path setup.
from eef_xyz3d_action_adapter import EefXyz3dActionAdapter
from eef_xyz3d_observation_adapter import EefXyz3dObservationAdapter

@dataclass
class Args:
    out_dir: Path = Path("data/piper_dual_eef_xyz3d/videos")
    action_horizon: int = 50
    fps: int = 30
    actions_during_latency: int = 5
    num_steps: int = 8000
    num_episodes: int = 1
    run_tag: str = ""
    bridge_python: Path = DEFAULT_BRIDGE_PYTHON
    ros2_config: Path = DEFAULT_ROS2_CONFIG
    dry_run: bool = True
    publish_actions: bool = False
    max_action_delta: float | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    prompt: str = "Stack_the_paper_cups_together."
    use_async: bool = True
    use_rtc: bool = True
    eef_left_topic: str = DEFAULT_EEF_LEFT_TOPIC
    eef_right_topic: str = DEFAULT_EEF_RIGHT_TOPIC
    eef_left_action_topic: str = DEFAULT_EEF_LEFT_ACTION_TOPIC
    eef_right_action_topic: str = DEFAULT_EEF_RIGHT_ACTION_TOPIC
```

The parser description must be `Deploy a dual-arm Piper XYZ+RPY EEF policy through ROS 2 PosCmd control.`. Preserve aliases `--action-horizon/--action_horizon` and `--actions-during-latency/--actions_during_latency`. Define `--dry-run` with `argparse.BooleanOptionalAction`, and define `--publish-actions` with `action="store_true"` and default `False`.

Define validation and summary as:

```python
def _validate_xyz3d_contract(args: Args) -> None:
    if args.publish_actions and args.dry_run:
        raise ValueError("--publish-actions requires --no-dry-run")
    if args.max_action_delta is not None and (
        not math.isfinite(args.max_action_delta) or args.max_action_delta < 0
    ):
        raise ValueError("--max-action-delta must be finite and non-negative")
    if not args.eef_left_topic or not args.eef_right_topic:
        raise ValueError("EEF observation topics must both be non-empty")
    if not args.eef_left_action_topic or not args.eef_right_action_topic:
        raise ValueError("EEF action topics must both be non-empty")


def contract_summary(args: Args) -> str:
    return "\n".join(
        [
            "ROS 2 XYZ3D EEF deployment contract:",
            "- policy state/action = 14D [left_xyzrpy,left_gripper,right_xyzrpy,right_gripper]",
            f"- action_horizon={args.action_horizon}",
            f"- dry_run={args.dry_run}",
            f"- publish_actions={args.publish_actions}",
            "- publish-actions requires --no-dry-run",
            f"- eef_observation_topics={args.eef_left_topic}, {args.eef_right_topic}",
            f"- eef_action_topics={args.eef_left_action_topic}, {args.eef_right_action_topic}",
            f"- bridge_python={args.bridge_python}",
            f"- ros2_config={args.ros2_config}",
            f"- policy_server={args.host}:{args.port}",
        ]
    )
```

Build the environment with both dedicated adapters:

```python
def _build_environment(args: Args) -> Ros2DualEnvironment:
    action_adapter = EefXyz3dActionAdapter()
    backend = Ros2BackendClient(
        bridge_python=args.bridge_python,
        config=args.ros2_config,
        dry_run=args.dry_run,
        publish_actions=args.publish_actions,
        action_adapter=action_adapter,
        eef_control=True,
        eef_left_topic=args.eef_left_topic,
        eef_right_topic=args.eef_right_topic,
        eef_left_action_topic=args.eef_left_action_topic,
        eef_right_action_topic=args.eef_right_action_topic,
    )
    return Ros2DualEnvironment(
        backend=backend,
        dry_run=args.dry_run,
        publish_actions=args.publish_actions,
        observation_adapter=EefXyz3dObservationAdapter(task=args.prompt),
        action_adapter=action_adapter,
        prompt=args.prompt,
        max_episode_steps=args.num_steps,
        watchdog_timeout=DEFAULT_WATCHDOG_TIMEOUT,
        frame_timeout=DEFAULT_FRAME_TIMEOUT,
        max_action_delta=args.max_action_delta,
    )
```

Retain `_build_policy`, `_build_runtime`, and `main` behavior from the existing EEF entrypoint, changing user-facing labels from `EEF` to `XYZ3D EEF` and invoking `_validate_xyz3d_contract` before creating the environment.

- [ ] **Step 4: Run focused entrypoint tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  examples/piper_dual/tests/test_main_dual_eef_xyz3d_ros.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Run combined adapter and entrypoint tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  examples/piper_dual/tests/test_eef_xyz3d_adapters.py \
  examples/piper_dual/tests/test_main_dual_eef_xyz3d_ros.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the entrypoint slice**

```bash
git add \
  examples/piper_dual/main_dual_eef_xyz3d_ros.py \
  examples/piper_dual/tests/test_main_dual_eef_xyz3d_ros.py
git commit -m "feat: add xyz3d eef ros2 client"
```

---

### Task 3: Document and verify the complete XYZ3D client

**Files:**
- Modify: `examples/piper_dual/README.md`
- Verify: `examples/piper_dual/eef_xyz3d_action_adapter.py`
- Verify: `examples/piper_dual/eef_xyz3d_observation_adapter.py`
- Verify: `examples/piper_dual/main_dual_eef_xyz3d_ros.py`
- Verify: `examples/piper_dual/tests/test_eef_xyz3d_adapters.py`
- Verify: `examples/piper_dual/tests/test_main_dual_eef_xyz3d_ros.py`

**Interfaces:**
- Consumes: server config `pi05_piper_dual_stack_cups_eef_xyz3d`, the checkpoint path, and the new client CLI.
- Produces: exact operator commands and final verification evidence.

- [ ] **Step 1: Add server and safe client commands to the README**

Add an `EEF XYZ+RPY 模型部署` subsection beside the existing EEF deployment documentation. Include:

```bash
cd /home/agilex/client/.worktrees/piper-dual-ros2-migration
source .venv/bin/activate
python scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_piper_dual_stack_cups_eef_xyz3d \
  --policy.dir=/home/agilex/checkpoints/pi05_piper_dual_stack_cups_eef_xyz3d_jax_step10000_pytorch
```

and the safe client command:

```bash
source /opt/ros/humble/setup.bash
source /home/agilex/camera_ros/install/setup.bash
source /home/agilex/piper_ros/install/setup.bash
cd /home/agilex/client/.worktrees/piper-dual-ros2-migration
source .venv/bin/activate
python examples/piper_dual/main_dual_eef_xyz3d_ros.py \
  --host 127.0.0.1 \
  --port 8000 \
  --bridge-python /usr/bin/python3 \
  --ros2-config examples/piper_dual/ros2_piper_dual.yaml \
  --prompt "Stack_the_paper_cups_together." \
  --action-horizon 50 \
  --dry-run
```

State that the 14D layout is `[left_xyz,left_rpy,left_gripper,right_xyz,right_rpy,right_gripper]` and real publication requires replacing `--dry-run` with `--no-dry-run --publish-actions` only after dry-run inspection.

- [ ] **Step 2: Run focused new tests and existing rot6d regressions**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  examples/piper_dual/tests/test_eef_xyz3d_adapters.py \
  examples/piper_dual/tests/test_main_dual_eef_xyz3d_ros.py \
  examples/piper_dual/tests/test_eef_adapters.py \
  examples/piper_dual/tests/test_main_dual_eef_ros.py -q
```

Expected: all selected tests pass. If an existing rot6d test exposes unrelated pre-existing workspace changes, report the exact failure rather than weakening the new safety contract.

- [ ] **Step 3: Compile all new production modules**

Run:

```bash
.venv/bin/python -m py_compile \
  examples/piper_dual/eef_xyz3d_action_adapter.py \
  examples/piper_dual/eef_xyz3d_observation_adapter.py \
  examples/piper_dual/main_dual_eef_xyz3d_ros.py
```

Expected: exit code 0 with no output.

- [ ] **Step 4: Exercise the actual CLI surface**

Run:

```bash
.venv/bin/python examples/piper_dual/main_dual_eef_xyz3d_ros.py --help
```

Expected: exit code 0; output names the XYZ+RPY EEF deployment and includes `--action-horizon`, `--dry-run`, `--no-dry-run`, `--publish-actions`, all four EEF topic flags, host, port, prompt, ROS config, and bridge Python.

- [ ] **Step 5: Verify the real checkpoint and registered server contract**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  src/openpi/training/config_test.py -k "eef_xyz3d" -q
```

Expected: the XYZ3D config test passes, proving model width 32, horizon 50, 14D normalization assets, and the dedicated input/output transforms remain registered.

- [ ] **Step 6: Check the intended diff and whitespace**

Run:

```bash
git diff --check -- \
  examples/piper_dual/eef_xyz3d_action_adapter.py \
  examples/piper_dual/eef_xyz3d_observation_adapter.py \
  examples/piper_dual/main_dual_eef_xyz3d_ros.py \
  examples/piper_dual/tests/test_eef_xyz3d_adapters.py \
  examples/piper_dual/tests/test_main_dual_eef_xyz3d_ros.py \
  examples/piper_dual/README.md
```

Expected: exit code 0 with no whitespace errors.

- [ ] **Step 7: Commit documentation and verification-complete state**

```bash
git add examples/piper_dual/README.md
git commit -m "docs: add xyz3d eef deployment commands"
```

Expected: commit succeeds; no generated caches, videos, or runtime output are staged.

## Self-review result

- Spec coverage: tasks cover both adapters, the independent entrypoint, safe defaults, failure closure, exact model/client dimensions, deployment commands, regressions, compilation, and CLI behavior.
- Placeholder scan: no deferred implementation or unspecified validation steps remain.
- Type consistency: both environment and backend consume one `EefXyz3dActionAdapter`; the observation adapter emits 14D; the server config exposes 14D; the action adapter accepts and splits 14D.
- Scope: existing rot6d production files remain reference-only and unchanged.
