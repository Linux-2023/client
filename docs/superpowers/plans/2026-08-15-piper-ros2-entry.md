# Piper ROS2 专用入口 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone `main_dual_ros.py` that exposes only the ROS2 bridge + remote PI05 deployment path, without SDK/CAN/camera-device options.

**Architecture:** Keep `main_dual.py` unchanged as the compatibility entrypoint. The new entrypoint constructs `Ros2DualEnvironment` directly, connects `WebsocketClientPolicy` to the PI05 server, optionally wraps it with the existing action broker, and runs the existing `PolicyAgent`/`Runtime` loop. Safety validation happens before environment creation; dry-run remains the default.

**Tech Stack:** Python 3.10/3.12 split runtime, `argparse`, `openpi-client` WebSocket policy and runtime, existing ROS2 bridge sidecar, pytest.

## Global Constraints

- New entrypoint: `examples/piper_dual/main_dual_ros.py`.
- Do not modify or remove SDK behavior from `examples/piper_dual/main_dual.py`.
- Do not expose SDK backend, CAN ports, USB/RealSense IDs, `tele_mode`, `gripper_norm`, or SDK recording options in the new parser.
- `--dry-run` defaults to true; `--publish-actions` requires `--no-dry-run`.
- `--publish-actions` and invalid safety combinations must be rejected before creating the ROS2 environment.
- Runtime failures return exit code 1; safety validation failures return 2; Ctrl-C returns 130.
- No physical ROS2 nodes or real action publishing during verification.

---

### Task 1: Implement the ROS2-only deployment entrypoint

**Files:**
- Create: `examples/piper_dual/main_dual_ros.py`
- Reference: `examples/piper_dual/main_dual.py:14-418`
- Reference: `examples/piper_dual/ros2_environment.py:23-215`
- Reference: `packages/openpi-client/src/openpi_client/runtime/runtime.py:11-153`

**Interfaces:**
- Produces `Args`, `build_arg_parser()`, `parse_args(argv)`, `_validate_ros2_contract(args)`, `contract_summary(args)`, `_build_environment(args)`, `_build_policy(args)`, `_build_runtime(args, environment, policy)`, and `main(args=None) -> int`.
- `_build_environment` must instantiate `Ros2DualEnvironment` directly with `bridge_python`, `ros2_config`, `dry_run`, `publish_actions`, `prompt`, `max_episode_steps`, and optional `max_action_delta`.
- `_build_policy` must create `WebsocketClientPolicy(host=args.host, port=args.port)` and wrap it with `ActionChunkBroker_RTC` when `use_async` is true, otherwise `ActionChunkBroker`.
- `_build_runtime` must create `Runtime(environment=..., agent=PolicyAgent(policy=...), subscribers=[VideoSaver(out_dir), RobotStatePlotter(...)], max_hz=fps, num_episodes=num_episodes)`.

- [ ] **Step 1: Write the minimal failing parser/safety tests first**

Create `examples/piper_dual/tests/test_main_dual_ros.py` with tests asserting:

```python
def test_parser_has_ros2_deployment_options_but_no_sdk_options():
    help_text = main_dual_ros.build_arg_parser().format_help()
    for flag in ("--host", "--port", "--bridge-python", "--ros2-config", "--dry-run", "--no-dry-run", "--publish-actions", "--max-action-delta"):
        assert flag in help_text
    for flag in ("--backend", "--left-can-port", "--right-can-port", "--high-camera-id", "--left-wrist-camera-id", "--right-wrist-camera-id", "--tele-mode", "--gripper-norm"):
        assert flag not in help_text


def test_publish_actions_requires_no_dry_run():
    args = main_dual_ros.Args(publish_actions=True, dry_run=True)
    with pytest.raises(ValueError, match="--publish-actions requires --no-dry-run"):
        main_dual_ros._validate_ros2_contract(args)


def test_parse_args_defaults_to_ros2_dry_run():
    args = main_dual_ros.parse_args([])
    assert args.dry_run is True
    assert args.publish_actions is False
    assert args.bridge_python == Path("/usr/bin/python3")


def test_build_environment_constructs_ros2_environment_only(monkeypatch):
    captured = {}
    class FakeRos2Environment:
        def __init__(self, **kwargs):
            captured.update(kwargs)
    monkeypatch.setattr(main_dual_ros, "Ros2DualEnvironment", FakeRos2Environment)
    environment = main_dual_ros._build_environment(main_dual_ros.Args(prompt="Fold the towel", max_action_delta=0.25))
    assert isinstance(environment, FakeRos2Environment)
    assert captured["prompt"] == "Fold the towel"
    assert captured["dry_run"] is True
    assert captured["publish_actions"] is False
    assert captured["max_action_delta"] == 0.25
```

- [ ] **Step 2: Run the focused tests and verify the expected RED failure**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_main_dual_ros.py -q
```

Expected: collection or import failure because `main_dual_ros.py` does not exist yet; do not treat this as completion.

- [ ] **Step 3: Implement `main_dual_ros.py` with only ROS2 deployment concerns**

Use this concrete shape:

```python
#!/usr/bin/env python3
"""ROS 2-only dual-arm Piper PI05 deployment entrypoint."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import signal
import sys
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OPENPI_CLIENT_SRC = _PROJECT_ROOT / "packages/openpi-client/src"
if str(_OPENPI_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(_OPENPI_CLIENT_SRC))

from env_dual import DEFAULT_ROS2_CONFIG
from ros2_environment import Ros2DualEnvironment

DEFAULT_BRIDGE_PYTHON = Path("/usr/bin/python3")
DEFAULT_WATCHDOG_TIMEOUT = 5.0

@dataclass
class Args:
    out_dir: Path = Path("data/piper_dual/videos")
    action_horizon: int = 10
    fps: int = 30
    actions_during_latency: int = 5
    num_steps: int = 800
    num_episodes: int = 1
    run_tag: str = ""
    bridge_python: Path = DEFAULT_BRIDGE_PYTHON
    ros2_config: Path = DEFAULT_ROS2_CONFIG
    dry_run: bool = True
    publish_actions: bool = False
    max_action_delta: float | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    prompt: str = "Fold_the_towel"
    use_async: bool = True
    use_rtc: bool = False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deploy the dual-arm Piper PI05 policy through the ROS 2 backend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-dir", type=Path, default=Args.out_dir)
    parser.add_argument("--action-horizon", "--action_horizon", dest="action_horizon", type=int, default=Args.action_horizon)
    parser.add_argument("--fps", type=int, default=Args.fps)
    parser.add_argument("--actions-during-latency", "--actions_during_latency", dest="actions_during_latency", type=int, default=Args.actions_during_latency)
    parser.add_argument("--num-steps", type=int, default=Args.num_steps)
    parser.add_argument("--num-episodes", type=int, default=Args.num_episodes)
    parser.add_argument("--run-tag", type=str, default=Args.run_tag)
    parser.add_argument("--bridge-python", type=Path, default=Args.bridge_python)
    parser.add_argument("--ros2-config", type=Path, default=Args.ros2_config)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=Args.dry_run)
    parser.add_argument("--publish-actions", action="store_true", default=Args.publish_actions)
    parser.add_argument("--max-action-delta", type=float, default=Args.max_action_delta)
    parser.add_argument("--host", type=str, default=Args.host)
    parser.add_argument("--port", type=int, default=Args.port)
    parser.add_argument("--prompt", type=str, default=Args.prompt)
    parser.add_argument("--use-async", action=argparse.BooleanOptionalAction, default=Args.use_async)
    parser.add_argument("--use-rtc", action=argparse.BooleanOptionalAction, default=Args.use_rtc)
    return parser


def parse_args(argv: list[str] | None = None) -> Args:
    return Args(**vars(build_arg_parser().parse_args(argv)))


def _validate_ros2_contract(args: Args) -> None:
    if args.publish_actions and args.dry_run:
        raise ValueError("--publish-actions requires --no-dry-run")
    if args.max_action_delta is not None and args.max_action_delta < 0:
        raise ValueError("--max-action-delta must be non-negative")


def _build_environment(args: Args) -> Ros2DualEnvironment:
    kwargs: dict[str, Any] = {
        "bridge_python": args.bridge_python,
        "ros2_config": args.ros2_config,
        "dry_run": args.dry_run,
        "publish_actions": args.publish_actions,
        "prompt": args.prompt,
        "max_episode_steps": args.num_steps,
        "watchdog_timeout": DEFAULT_WATCHDOG_TIMEOUT,
    }
    if args.max_action_delta is not None:
        kwargs["max_action_delta"] = args.max_action_delta
    return Ros2DualEnvironment(**kwargs)


def _build_policy(args: Args) -> Any:
    from openpi_client import action_chunk_broker
    from openpi_client import websocket_client_policy
    base_policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    if args.use_async:
        return action_chunk_broker.ActionChunkBroker_RTC(
            policy=base_policy,
            action_horizon=args.action_horizon,
            fps=args.fps,
            actions_during_latency=args.actions_during_latency,
            use_rtc=args.use_rtc,
        )
    return action_chunk_broker.ActionChunkBroker(policy=base_policy, action_horizon=args.action_horizon, fps=args.fps)


def _build_runtime(args: Args, environment: Ros2DualEnvironment, policy: Any) -> Any:
    from openpi_client.runtime import runtime
    from openpi_client.runtime.agents import policy_agent
    import saver
    from plot_dynamics import RobotStatePlotter
    return runtime.Runtime(
        environment=environment,
        agent=policy_agent.PolicyAgent(policy=policy),
        subscribers=[saver.VideoSaver(args.out_dir), RobotStatePlotter(args.out_dir, broker=policy if args.use_async else None, run_tag=args.run_tag)],
        max_hz=args.fps,
        num_episodes=args.num_episodes,
    )
```

Implement `main()` using the existing signal/cleanup pattern from `main_dual.py`, but remove all SDK branches and device logging. It must validate before `_build_environment`, print ROS2-only configuration, run the runtime, return 0 on normal completion, 130 on `KeyboardInterrupt`, 1 on other exceptions, and close runtime/environment in `finally`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_main_dual_ros.py -q
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit the entrypoint and focused tests**

```bash
git add examples/piper_dual/main_dual_ros.py examples/piper_dual/tests/test_main_dual_ros.py
git commit -m "feat: add ROS2-only Piper deployment entrypoint"
```

---

### Task 2: Document and verify the standalone command

**Files:**
- Modify: `examples/piper_dual/README.md:259-291`
- Test: `examples/piper_dual/tests/test_main_dual_ros.py`

**Interfaces:**
- Documentation must invoke `main_dual_ros.py` and show `--dry-run` first.
- Physical publishing example must explicitly include `--no-dry-run --publish-actions`.

- [ ] **Step 1: Add README commands for the new entrypoint**

Add a ROS2-only deployment section with:

```bash
cd /home/agilex/client
python examples/piper_dual/main_dual_ros.py \
  --host 127.0.0.1 \
  --port 8000 \
  --bridge-python /usr/bin/python3 \
  --ros2-config examples/piper_dual/ros2_piper_dual.yaml \
  --prompt "Fold the towel" \
  --dry-run
```

Then show the explicit physical publishing form:

```bash
python examples/piper_dual/main_dual_ros.py \
  --host 127.0.0.1 \
  --port 8000 \
  --bridge-python /usr/bin/python3 \
  --ros2-config examples/piper_dual/ros2_piper_dual.yaml \
  --prompt "Fold the towel" \
  --no-dry-run \
  --publish-actions
```

State that the client does not start/enable/reset physical arms and that official camera/Piper ROS2 nodes must already be running.

- [ ] **Step 2: Verify CLI surface under client Python**

Run:

```bash
python examples/piper_dual/main_dual_ros.py --help
```

Expected: help lists ROS2/server/runtime flags and contains no `--backend`, CAN, camera-ID, `--tele-mode`, or `--gripper-norm` options.

- [ ] **Step 3: Verify both interpreters compile the new entrypoint**

Run:

```bash
python -m py_compile examples/piper_dual/main_dual_ros.py
/usr/bin/python3 -m py_compile examples/piper_dual/main_dual_ros.py
```

Expected: both commands exit 0 with no output. Do not launch ROS2 nodes or publish actions.

- [ ] **Step 4: Run the existing ROS2/client regression matrix**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_main_dual.py examples/piper_dual/tests/test_ros2_environment.py examples/piper_dual/tests/test_ros2_end_to_end_mock.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest examples/piper_dual/tests/test_ros2_protocol.py examples/piper_dual/tests/test_ros2_bridge_codec.py examples/piper_dual/tests/test_ros2_backend.py -q
```

Expected: existing suites remain green; report any interpreter-specific dependency limitation instead of launching hardware.

- [ ] **Step 5: Commit documentation and final verification changes**

```bash
git add examples/piper_dual/README.md examples/piper_dual/tests/test_main_dual_ros.py
 git commit -m "docs: document ROS2-only Piper deployment command"
```
