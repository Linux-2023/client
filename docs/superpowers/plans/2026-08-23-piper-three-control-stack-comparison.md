# Piper 三控制栈统一真机对比实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox syntax and each task ends with an independently testable deliverable.

**Goal:** Extend `main_dual.py` with `--control-stack local-ros|official-ros|direct-sdk` so one PI05 policy/runtime can drive three mutually exclusive real-hardware control stacks through one normalized 14D ROS 2 contract.

**Architecture:** Keep every control stack externally started and manually selected. `main_dual.py` remains the only policy/runtime entrypoint and always talks to a ROS 2 bridge contract; `local-ros`, `official-ros`, and `direct-sdk` differ only in immutable endpoint profiles. The direct SDK stack runs as an external ROS 2 adapter that imports the unchanged third-party `PiperController`; it never makes `main_dual.py` open CAN.

**Tech Stack:** Python 3.10 ROS 2 Humble/rclpy, Python 3.11 client, `sensor_msgs/msg/JointState`, `piper_msgs/msg/PiperStatusMsg`, PyYAML, NumPy, pytest, `piper_sdk.C_PiperInterface_V2` through the unchanged `control_your_robot` controller.

## Global Constraints

- Three stacks are mutually exclusive and may not concurrently own `can_left` or `can_right`.
- `main_dual.py` must not configure CAN, launch Piper nodes, enable/disable arms, or import direct SDK modules.
- Existing `/home/agilex/piper_ros/src/piper/**` and `/home/agilex/lgd/control_your_robot/**` control implementations remain unchanged.
- Official source remains isolated in `/home/agilex/piper_ros/.worktrees/piper-official-humble`; never reuse or overwrite `/home/agilex/piper_ros/install` for official builds.
- `control_your_robot/src/robot/controller/Piper_controller.py` remains read-only; direct adapter imports and calls its public `set_up`, `get_state`, `set_joint`, and `set_gripper` methods.
- Every profile exposes `observation.state` and `action` as `(14,)`: left J1–J6 plus gripper, then right J1–J6 plus gripper; joints are radians and grippers are meters. The direct adapter converts third-party normalized gripper values with `meters = normalized * 0.07` and `normalized = meters / 0.07`.
- Every live action path requires `--backend ros2 --no-dry-run --publish-actions`; dry-run creates zero action publishers.
- Every live profile validates its selected stack identity and rejects a missing or conflicting ROS graph before creating action publishers.
- Automatic tests and smoke checks must not open CAN, start Piper nodes, enable/disable an arm, or publish a physical action.
- Three real-hardware runs use one PI05 server/checkpoint, one prompt, one camera setup, one set of runtime/action limits, one initial scene, and separate `run-tag` output directories.

---

### Task 1: Add immutable control-stack profiles and YAML identities

**Files:**
- Create: `examples/piper_dual/control_stack.py`
- Create: `examples/piper_dual/ros2_piper_dual_local.yaml`
- Create: `examples/piper_dual/ros2_piper_dual_direct_sdk.yaml`
- Modify: `examples/piper_dual/ros2_piper_dual.yaml`
- Test: `examples/piper_dual/tests/test_control_stack.py`

**Interfaces:**
- `ControlStackId = Literal["local-ros", "official-ros", "direct-sdk"]`.
- `@dataclass(frozen=True, slots=True) class ControlStackProfile` exposes `stack_id`, `config_path`, `required_topics: tuple[str, ...]`, `forbidden_topics: tuple[str, ...]`, `action_topics: tuple[str, ...]`, and `direct_adapter_node: str | None`.
- `profile_for(stack_id: str) -> ControlStackProfile` returns the three shipped profiles and raises `ValueError` for unknown IDs.
- `load_profile_config(stack_id: str, config_path: Path) -> tuple[ControlStackProfile, dict[str, Any]]` loads YAML, requires top-level `control_stack_id` equal to `stack_id`, and returns the profile plus mapping.

- [ ] **Step 1: Write failing profile tests**

```python

def test_profiles_have_distinct_ids_and_identity_files():
    assert [profile_for(name).stack_id for name in ("local-ros", "official-ros", "direct-sdk")] == [
        "local-ros", "official-ros", "direct-sdk"
    ]
    for name in ("local-ros", "official-ros", "direct-sdk"):
        profile, mapping = load_profile_config(name, profile_for(name).config_path)
        assert profile.stack_id == name
        assert mapping["control_stack_id"] == name


def test_profile_rejects_yaml_for_another_stack(tmp_path):
    path = tmp_path / "wrong.yaml"
    path.write_text("control_stack_id: official-ros\n", encoding="utf-8")
    with pytest.raises(ValueError, match="control_stack_id.*local-ros"):
        load_profile_config("local-ros", path)


def test_profiles_have_non_overlapping_identity_topics():
    profiles = [profile_for(name) for name in ("local-ros", "official-ros", "direct-sdk")]
    assert all(set(profile.required_topics).isdisjoint(profile.forbidden_topics) for profile in profiles)
```

Run:

```bash
cd /home/agilex/client/.worktrees/piper-dual-ros2-migration
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q examples/piper_dual/tests/test_control_stack.py
```

Expected: collection fails because `control_stack.py` and the three profile identities do not yet exist.

- [ ] **Step 2: Implement profile parsing and exact endpoint sets**

`control_stack.py` must define these identity topics:

```python
_PROFILES = {
    "local-ros": {
        "required_topics": (
            "/puppet/joint_left", "/puppet/joint_right",
            "/master/joint_left", "/master/joint_right",
            "/piper_left_ctrl_node/arm_status", "/piper_right_ctrl_node/arm_status",
        ),
        "forbidden_topics": (
            "/joint_left", "/joint_right", "/direct_sdk/joint_left", "/direct_sdk/joint_right",
        ),
        "action_topics": ("/joint_left_states", "/joint_right_states"),
        "direct_adapter_node": None,
    },
    "official-ros": {
        "required_topics": (
            "/joint_left", "/joint_right",
            "/joint_states_ctrl_left", "/joint_states_ctrl_right",
            "/arm_status_left", "/arm_status_right",
        ),
        "forbidden_topics": (
            "/puppet/joint_left", "/puppet/joint_right", "/direct_sdk/joint_left", "/direct_sdk/joint_right",
        ),
        "action_topics": ("/joint_ctrl_cmd_left", "/joint_ctrl_cmd_right"),
        "direct_adapter_node": None,
    },
    "direct-sdk": {
        "required_topics": (
            "/direct_sdk/joint_left", "/direct_sdk/joint_right",
            "/direct_sdk/joint_ctrl_left", "/direct_sdk/joint_ctrl_right",
            "/direct_sdk/arm_status_left", "/direct_sdk/arm_status_right",
        ),
        "forbidden_topics": (
            "/puppet/joint_left", "/puppet/joint_right", "/joint_left", "/joint_right",
        ),
        "action_topics": ("/direct_sdk/joint_cmd_left", "/direct_sdk/joint_cmd_right"),
        "direct_adapter_node": "piper_direct_sdk_adapter",
    },
}
```

Use `yaml.safe_load`, reject missing/non-string `control_stack_id`, and never infer identity from filename. Keep the profile endpoint lists immutable tuples.

- [ ] **Step 3: Add the three profile YAML contracts**

Add `control_stack_id` to the existing official YAML. Keep its current official contract unchanged otherwise. Create the local YAML with the existing camera block and:

```yaml
control_stack_id: local-ros
bridge_contract:
  joint_topics:
    /puppet/joint_left: puppet_left
    /puppet/joint_right: puppet_right
    /master/joint_left: master_left
    /master/joint_right: master_right
  joint_action_topics:
    left: /joint_left_states
    right: /joint_right_states
  eef_topics:
    left: /puppet/end_pose_left
    right: /puppet/end_pose_right
  eef_action_topics:
    left: /pos_left_cmd
    right: /pos_right_cmd
  status_topics:
    left: /piper_left_ctrl_node/arm_status
    right: /piper_right_ctrl_node/arm_status
  joint_names: [joint0, joint1, joint2, joint3, joint4, joint5, joint6]
  control:
    speed_percent: 30
    gripper_effort: 0.5
    max_action_delta: 0.05
    status_watchdog_s: 0.5
```

Create the direct SDK YAML with the same camera/control blocks and:

```yaml
control_stack_id: direct-sdk
bridge_contract:
  joint_topics:
    /direct_sdk/joint_left: puppet_left
    /direct_sdk/joint_right: puppet_right
    /direct_sdk/joint_ctrl_left: master_left
    /direct_sdk/joint_ctrl_right: master_right
  joint_action_topics:
    left: /direct_sdk/joint_cmd_left
    right: /direct_sdk/joint_cmd_right
  eef_topics: {}
  eef_action_topics:
    left: /pos_left_cmd
    right: /pos_right_cmd
  status_topics:
    left: /direct_sdk/arm_status_left
    right: /direct_sdk/arm_status_right
  joint_names: [joint1, joint2, joint3, joint4, joint5, joint6, gripper]
  control:
    speed_percent: 100
    gripper_effort: 0.5
    max_action_delta: 0.05
    status_watchdog_s: 0.5
```

- [ ] **Step 4: Run profile tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q examples/piper_dual/tests/test_control_stack.py
```

Expected: all profile tests pass.

---

### Task 2: Make the ROS bridge profile-aware and refuse graph conflicts

**Files:**
- Modify: `examples/piper_dual/ros2_contract.py`
- Modify: `examples/piper_dual/ros2_bridge_process.py`
- Modify: `examples/piper_dual/ros2_backend.py`
- Test: `examples/piper_dual/tests/test_ros2_contract.py`
- Test: `examples/piper_dual/tests/test_ros2_bridge_codec.py`
- Test: `examples/piper_dual/tests/test_ros2_backend.py`

**Interfaces:**
- `validate_profile_graph(node: Any, profile: ControlStackProfile, contract: BridgeContract) -> None` checks `node.get_topic_names_and_types()`, all contract sensor/status topics, required identity topics, expected ROS types, forbidden topic absence, direct adapter node presence when required, and one expected subscriber on each live action topic.
- `Ros2BackendClient(..., control_stack: str = "official-ros")` forwards `--control-stack` to the sidecar and stores it as `self.control_stack`.
- `_parse_args()` in `ros2_bridge_process.py` accepts `--control-stack` with choices `local-ros`, `official-ros`, `direct-sdk`.
- `PiperRos2Bridge(..., control_stack: str = "official-ros")` validates config identity during construction but creates no live action publisher until a valid action request passes graph/status readiness.

- [ ] **Step 1: Write failing graph and forwarding tests**

```python
def test_bridge_backend_forwards_control_stack_to_sidecar(tmp_path, monkeypatch):
    client = Ros2BackendClient(
        bridge_python=Path(sys.executable),
        config=write_config(tmp_path),
        control_stack="local-ros",
        dry_run=True,
    )
    captured = {}
    monkeypatch.setattr(ros2_backend.subprocess, "Popen", fake_popen(captured))
    client.start()
    assert "--control-stack" in captured["args"]
    assert captured["args"][captured["args"].index("--control-stack") + 1] == "local-ros"


def test_profile_graph_rejects_forbidden_stack_topics():
    profile, mapping = load_profile_config("local-ros", profile_for("local-ros").config_path)
    contract = BridgeContract.from_mapping(mapping)
    node = FakeGraphNode(topics={"/joint_left": ["sensor_msgs/msg/JointState"]})
    with pytest.raises(ValueError, match="forbidden.*joint_left"):
        validate_profile_graph(node, profile, contract)


def test_profile_graph_rejects_wrong_message_type():
    profile, mapping = load_profile_config("local-ros", profile_for("local-ros").config_path)
    contract = BridgeContract.from_mapping(mapping)
    node = FakeGraphNode(topics={"/puppet/joint_left": ["std_msgs/msg/Bool"]})
    with pytest.raises(ValueError, match="sensor_msgs/msg/JointState"):
        validate_profile_graph(node, profile, contract)
```

Run the three named tests; expected failure is missing `control_stack` propagation and graph validator.

- [ ] **Step 2: Implement profile-aware config loading and deferred live publishers**

Load `profile, config = load_profile_config(control_stack, config_path)` before `EndpointPlan.for_mode`. Do not demand sensor graph readiness during dry-run construction. In live mode, also defer `create_publisher` until the first valid action request; otherwise the bridge's own action publishers would make `action_topics` appear in the graph before the external stack is proven ready.

On the first live action request, validate graph using topic type names:

```python
actual = {name: tuple(types) for name, types in node.get_topic_names_and_types()}
for topic in profile.required_topics:
    if topic not in actual:
        raise ValueError(f"control stack {profile.stack_id} missing required topic {topic}")
for topic in profile.forbidden_topics:
    if topic in actual:
        raise ValueError(f"control stack {profile.stack_id} found forbidden topic {topic}")
```
Require `sensor_msgs/msg/JointState` for every contract joint stream, command echo, and action; `sensor_msgs/msg/Image` for every contract camera; and `piper_msgs/msg/PiperStatusMsg` for every contract status. For `direct-sdk`, require a node name containing `piper_direct_sdk_adapter`. Require each selected action topic to have exactly one external subscription. Then call `PiperStatusLatch.assert_joint_ready()`. Only after all checks pass may the bridge create both action publishers and publish the validated request. On any failure, permanently latch publishing off for that bridge process.

Dry-run must create zero action publishers and preserve `action_ignored`; it may start even with no hardware graph so stop-only smoke remains possible. Config ID mismatch still fails during construction in every mode.

- [ ] **Step 3: Implement client propagation**

Add `control_stack` to `Ros2BackendClient`, append `--control-stack` to bridge subprocess argv, and pass it through `Ros2DualEnvironment`. Keep `control_stack="official-ros"` as the environment default to preserve the migrated official path. Do not import `rclpy` or SDK modules in the Python 3.11 client.

- [ ] **Step 4: Run bridge/contract/backend tests**

```bash
/usr/bin/python3 -m pytest -q \
  examples/piper_dual/tests/test_ros2_contract.py \
  examples/piper_dual/tests/test_ros2_bridge_codec.py
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  examples/piper_dual/tests/test_ros2_backend.py
```

Expected: both ROS-message tests and client-side backend tests pass under their compatible interpreters; no CAN or ROS node is launched.

### Task 3: Add the direct `piper_sdk` ROS adapter without modifying the third-party tree

**Files:**
- Create: `examples/piper_dual/piper_direct_sdk_adapter.py`
- Test: `examples/piper_dual/tests/test_piper_direct_sdk_adapter.py`
- Reference read-only: `/home/agilex/lgd/control_your_robot/src/robot/controller/Piper_controller.py`

**Interfaces:**
- `DirectSdkArm` wraps one imported `PiperController` and exposes `setup()`, `read_state()`, `apply_action(values: Sequence[float])`, and `shutdown()`.
- `PiperDirectSdkAdapter(Node)` publishes `/direct_sdk/joint_left/right`, `/direct_sdk/joint_ctrl_left/right`, and `/direct_sdk/arm_status_left/right`; subscribes to `/direct_sdk/joint_cmd_left/right`; and owns exactly one left/right `DirectSdkArm`.
- CLI requires `--left-can`, `--right-can`, and `--allow-enable`; without `--allow-enable` it exits before importing or constructing `PiperController`.
- `shutdown()` calls each underlying SDK object’s `DisableArm(7)` when available and then `DisconnectPort()`; cleanup errors are surfaced and return nonzero.
- Direct SDK intentionally retains `PiperController.set_joint()` at its native fixed 100% speed. Local/official ROS remain 30%; document this tested difference in every comparison run.

- [ ] **Step 1: Write failing adapter tests with fake controllers**

```python

def test_adapter_refuses_to_load_controller_without_allow_enable(monkeypatch):
    loaded = []
    monkeypatch.setattr(adapter, "_load_piper_controller", lambda: loaded.append(True))
    assert adapter.main(["--left-can", "can_left", "--right-can", "can_right"]) == 2
    assert loaded == []


def test_apply_action_converts_meter_gripper_to_normalized_controller_input():
    arm = DirectSdkArm("left", "can_left", controller_factory=FakeController)
    arm.setup()
    arm.apply_action([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.035])
    assert arm.controller.set_joint_calls == [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
    assert arm.controller.set_gripper_calls == [pytest.approx(0.5)]


def test_read_state_converts_normalized_gripper_to_meters():
    arm = DirectSdkArm("left", "can_left", controller_factory=FakeController)
    arm.setup()
    arm.controller.state["gripper"] = 0.5
    assert arm.read_state()[-1] == pytest.approx(0.035)


@pytest.mark.parametrize("gripper", [-0.001, 0.071])
def test_apply_action_rejects_gripper_outside_meter_contract(gripper):
    arm = DirectSdkArm("left", "can_left", controller_factory=FakeController)
    arm.setup()
    with pytest.raises(ValueError, match="0.0.*0.07"):
        arm.apply_action([0.0] * 6 + [gripper])


def test_shutdown_disables_and_disconnects_both_arms():
    node = make_fake_adapter()
    node.shutdown()
    assert all(controller.disable_calls == [7] for controller in node.controllers)
    assert all(controller.disconnect_calls == [True] for controller in node.controllers)
```

Run:

```bash
/usr/bin/python3 -m pytest -q examples/piper_dual/tests/test_piper_direct_sdk_adapter.py
```

Expected: failure because the adapter file and lifecycle methods do not exist.

- [ ] **Step 2: Implement lazy third-party import and explicit enable gate**

The module may import ROS message types at module load, but `_load_piper_controller() -> type` imports `robot.controller.Piper_controller.PiperController` only after `main()` validates `--allow-enable`. Set `PYTHONPATH` in the documented command rather than changing the third-party repository. Pass the returned class into `DirectSdkArm(..., controller_factory=controller_type)`. `DirectSdkArm.setup()` calls:

```python
self.controller = self._controller_factory(self.name)
self.controller.set_up(self.can_port)
```

Tests inject a fake factory and never call the real loader or controller setup.

- [ ] **Step 3: Implement state, command-echo, and real status publishers plus action subscriptions**

Use a 20 Hz timer to call `get_state()`. Publish a feedback `JointState` with names `joint1..joint6,gripper`, six actual joint radians, and `get_state()["gripper"] * 0.07` meters. Validate action gripper meters in `[0.0, 0.07]`, call `set_joint(values[:6])`, then call `set_gripper(values[6] / 0.07)`. After both calls succeed, publish the accepted meter-valued action on `/direct_sdk/joint_ctrl_left/right`; before the first command, publish the initial real feedback as command echo. Build `PiperStatusMsg` from `controller.controller.GetArmStatus().arm_status` (`ctrl_mode`, `arm_status`, `mode_feed`, `motion_status`, `err_code`) and inspect all six booleans from `GetArmEnableStatus()`. When all are enabled, publish the raw status fields. If any driver is disabled or a field is missing, preserve actual fields except set outward `ctrl_mode=0` as the documented adapter-not-ready projection, log/latch the exact driver reason, and stop accepting future commands; never synthesize a healthy value. Do not replace actual feedback with command echo.

- [ ] **Step 4: Implement signal-safe shutdown**

On SIGINT/SIGTERM, stop timers/subscriptions, reject new actions, invoke `DisableArm(7)` if present on each raw controller, invoke `DisconnectPort()`, and return nonzero if cleanup fails. Do not rely on `__del__` for hardware cleanup.

- [ ] **Step 5: Run adapter tests**

```bash
/usr/bin/python3 -m pytest -q examples/piper_dual/tests/test_piper_direct_sdk_adapter.py
```

Expected: all fake-controller tests pass; no CAN interface is opened.

---

### Task 4: Add `--control-stack` to the single policy entrypoint

**Files:**
- Modify: `examples/piper_dual/main_dual.py`
- Modify: `examples/piper_dual/env_dual.py`
- Modify: `examples/piper_dual/ros2_environment.py`
- Modify: `examples/piper_dual/tests/test_main_dual.py`
- Modify: `examples/piper_dual/tests/test_ros2_environment.py`

**Interfaces:**
- `Args.control_stack: str | None = None`.
- Parser flag: `--control-stack`, choices `local-ros`, `official-ros`, `direct-sdk`.
- `_resolve_control_stack(args) -> str | None` returns `None` for `backend="sdk"` and `official-ros` for `backend="ros2"` when omitted.
- `_validate_backend_contract` rejects a non-`None` control stack with `backend="sdk"`.
- `_environment_kwargs` forwards `control_stack` only for the ROS 2 backend.
- `Ros2DualEnvironment(..., control_stack: str = "official-ros")` passes it to `Ros2BackendClient`.

- [ ] **Step 1: Write failing parser and propagation tests**

```python

def test_parser_accepts_three_control_stack_ids():
    for name in ("local-ros", "official-ros", "direct-sdk"):
        assert main_dual.parse_args(["--backend", "ros2", "--control-stack", name]).control_stack == name


def test_sdk_backend_rejects_control_stack():
    with pytest.raises(ValueError, match="control-stack.*ros2"):
        main_dual._validate_backend_contract(main_dual.Args(backend="sdk", control_stack="local-ros"))


def test_ros2_environment_kwargs_forwards_control_stack():
    args = main_dual.Args(backend="ros2", control_stack="local-ros")
    kwargs = main_dual._environment_kwargs(args)
    assert kwargs["control_stack"] == "local-ros"
```

- [ ] **Step 2: Implement safe defaults, parser resolution, and runtime propagation**

Set `Args.dry_run=True` and `Args.publish_actions=False`; every ROS live run must explicitly pass `--no-dry-run --publish-actions`. Keep the existing `sdk` backend behavior unchanged when no stack is specified. For ROS 2, default `control_stack` to `official-ros`, print `control_stack=<id>` in `contract_summary`, and pass it through environment construction. The strategy, broker, subscriber list, `run_tag`, and policy server remain unchanged.

Add a test asserting `main_dual.parse_args(["--backend", "ros2"])` yields `dry_run is True` and `publish_actions is False`, while explicit `--no-dry-run --publish-actions` yields the live combination. Both safety checks must run before environment construction.

- [ ] **Step 3: Run main/environment tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  examples/piper_dual/tests/test_main_dual.py \
  examples/piper_dual/tests/test_ros2_environment.py
```

Expected: all existing and new tests pass; no hardware imports occur in parser/propagation tests.

---

### Task 5: Add stack-specific bridge integration tests and preserve legacy behavior

**Files:**
- Modify: `examples/piper_dual/tests/test_ros2_end_to_end_mock.py`
- Modify: `examples/piper_dual/tests/test_ros2_bridge_codec.py`
- Create: `examples/piper_dual/tests/test_control_stack_integration.py`
- Modify: `examples/piper_dual/README.md`

**Interfaces:**
- Every profile is exercised through `BridgeContract`, `EndpointPlan`, action validation, and a fake graph node.
- The local profile must encode `joint0..joint6` and `/joint_left_states`/`/joint_right_states`.
- The official profile must retain `joint1..joint6,gripper` and `/joint_ctrl_cmd_left`/`/joint_ctrl_cmd_right`.
- The direct profile must encode `/direct_sdk/joint_cmd_left/right` and decode `/direct_sdk/joint_left/right`.

- [ ] **Step 1: Add profile contract tests**

```python
@pytest.mark.parametrize("stack, expected_action_topics", [
    ("local-ros", ("/joint_left_states", "/joint_right_states")),
    ("official-ros", ("/joint_ctrl_cmd_left", "/joint_ctrl_cmd_right")),
    ("direct-sdk", ("/direct_sdk/joint_cmd_left", "/direct_sdk/joint_cmd_right")),
])
def test_each_profile_has_expected_action_topics(stack, expected_action_topics):
    profile, mapping = load_profile_config(stack, profile_for(stack).config_path)
    contract = BridgeContract.from_mapping(mapping)
    assert tuple(contract.joint_action_topics.values()) == expected_action_topics
```

- [ ] **Step 2: Add dry-run zero-publisher tests for all profiles**

Construct `PiperRos2Bridge` with fake ROS graph methods for each profile and assert `create_publisher` is never called, even when `publish_actions=True`. Submit a valid 7+7 action request and assert one `action_ignored` status.

- [ ] **Step 3: Add normalized mock frame tests**

Feed each profile’s fake image/joint/status events through `InProcessBridgeBackend`; assert every profile produces the same `(14,)` state/action shape and seven sensor aliases. Do not start rclpy or CAN.

- [ ] **Step 4: Document exact external commands**

Add a Chinese README section with three commands. The local command must use the existing CAN setup followed by `ros2 launch piper start_two_piper.launch.py` with explicit `auto_enable:=false`; the official command must source only `/opt/ros/humble` plus the isolated official `install-official`; the direct command must set `PYTHONPATH=/home/agilex/lgd/control_your_robot/src` and require `--allow-enable`. Each command must show the matching `--control-stack` and unique `--run-tag`.

- [ ] **Step 5: Run profile integration tests**

```bash
/usr/bin/python3 -m pytest -q \
  examples/piper_dual/tests/test_ros2_contract.py \
  examples/piper_dual/tests/test_ros2_bridge_codec.py \
  examples/piper_dual/tests/test_control_stack_integration.py
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  examples/piper_dual/tests/test_ros2_end_to_end_mock.py
```

Expected: all tests pass without live ROS/CAN.

---

### Task 6: Build and verify the isolated official candidate only

**Files:**
- No source modifications expected in `/home/agilex/piper_ros/.worktrees/piper-official-humble`.
- Verify: `/home/agilex/piper_ros/.worktrees/piper-official-humble/src/piper_official_bringup/launch/start_two_piper_official.launch.py`
- Verify: `/home/agilex/piper_ros/.worktrees/piper-official-humble/install-official/**`

**Interfaces:**
- Official worktree remains based on `017ffefa64511bc6325bd77ddc4e16065c152051` plus wrapper commit `ef4b3d4...`.
- Official build uses `build-official`, `install-official`, and `log-official`, never `/home/agilex/piper_ros/install`.

- [ ] **Step 1: Build official candidate in isolated prefixes**

```bash
cd /home/agilex/piper_ros/.worktrees/piper-official-humble
source /opt/ros/humble/setup.bash
colcon --log-base log-official build --symlink-install \
  --build-base build-official \
  --install-base install-official \
  --packages-up-to piper_official_bringup
```

Expected: exit 0; `install-official/piper_official_bringup` and official `piper` package are present.

- [ ] **Step 2: Verify source and fallback identity**

```bash
git rev-parse HEAD
git diff --exit-code 017ffefa64511bc6325bd77ddc4e16065c152051 -- src/piper/piper/piper_ctrl_single_node.py
sha256sum /home/agilex/piper_ros/src/piper/piper/piper_ctrl_single_node.py \
  /home/agilex/piper_ros/src/piper/launch/start_two_piper.launch.py
```

Expected: official control node has no diff from pinned baseline; fallback files are unchanged by the candidate build.

- [ ] **Step 3: Run launch-contract test only**

```bash
/usr/bin/python3 -m pytest -q \
  src/piper_official_bringup/test/test_official_dual_launch.py
```

Expected: focused launch contract passes. Do not launch the real node in this task.

---

### Task 7: Add manual three-stack runbook and comparison artifact boundaries

**Files:**
- Modify: `examples/piper_dual/main_dual.py`
- Modify: `examples/piper_dual/saver.py`
- Modify: `examples/piper_dual/README.md`
- Create: `examples/piper_dual/run_metadata.py`
- Test: `examples/piper_dual/tests/test_main_dual.py`
- Create: `examples/piper_dual/tests/test_run_metadata.py`
**Interfaces:**
- Run tags are exactly `local-ros`, `official-ros`, and `direct-sdk` for the manual comparison.
- `_comparison_out_dir(args: Args) -> Path` returns `args.out_dir / args.run_tag` for non-empty run tags and rejects path separators or `.`/`..`.
- `VideoSaver(out_dir: Path, subsample: int = 1, fps: float = 50.0)` validates positive finite `fps` and writes MP4 at `fps / subsample`; `main_dual.py` passes `args.fps`. `VideoSaver` and `RobotStatePlotter` receive the stack-specific output directory, so videos, plots, and CSV files cannot mix across runs or play at a misleading hard-coded rate.
- `RunMetadataRecorder(path: Path, metadata: Mapping[str, Any])` writes `run_metadata.json` atomically at start and rewrites it at exit with `ended_at`, `exit_code`, and `exit_reason`.
- Metadata records `control_stack`, `run_tag`, prompt, policy host/port, client/runtime parameters, selected config, `video_fps`, and `effective_speed_percent` (`30` local/official, `100` direct).
- The runbook explicitly stops `main_dual.py` first, stops the active control stack, verifies no stale process/CAN owner, then starts the next stack.

- [ ] **Step 1: Write failing output-isolation and metadata tests**

```python
def test_comparison_out_dir_separates_run_tag():
    args = main_dual.Args(out_dir=Path("runs"), run_tag="official-ros")
    assert main_dual._comparison_out_dir(args) == Path("runs/official-ros")


def test_video_saver_uses_runtime_fps(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(saver.imageio, "mimwrite", lambda path, images, fps: calls.update(fps=fps))
    subscriber = saver.VideoSaver(tmp_path, fps=30)
    subscriber._images = [np.zeros((8, 8, 3), dtype=np.uint8)]
    subscriber.on_episode_end()
    assert calls["fps"] == 30


def test_run_metadata_records_stack_speed_and_exit(tmp_path):
    recorder = RunMetadataRecorder(
        tmp_path / "run_metadata.json",
        {"control_stack": "direct-sdk", "effective_speed_percent": 100, "video_fps": 30},
    )
    recorder.start()
    recorder.finish(exit_code=130, exit_reason="keyboard_interrupt")
    data = json.loads((tmp_path / "run_metadata.json").read_text())
    assert data["effective_speed_percent"] == 100
    assert data["video_fps"] == 30
    assert data["exit_code"] == 130
    assert data["exit_reason"] == "keyboard_interrupt"
```

Implement atomic JSON writes through a temporary sibling followed by `Path.replace()`. Update `main_dual._build_runtime()` to pass `_comparison_out_dir(args)` and `args.fps` to `VideoSaver`, and the same output directory to `RobotStatePlotter`; wrap `main()` lifecycle with the metadata recorder. Invalid run tags or non-positive video FPS fail before environment creation.

- [ ] **Step 2: Document local-ROS run**

```bash
cd /home/agilex/piper_ros
bash can_config.sh
source /opt/ros/humble/setup.bash
source /home/agilex/piper_ros/install/setup.bash
ros2 launch piper start_two_piper.launch.py \
  can_left_port:=can_left can_right_port:=can_right auto_enable:=false
```

After both status topics are observed, explicitly enable:

```bash
ros2 service call /piper_left_ctrl_node/enable_srv piper_msgs/srv/Enable "{enable_request: true}"
ros2 service call /piper_right_ctrl_node/enable_srv piper_msgs/srv/Enable "{enable_request: true}"
```

Then explicitly enable only after checking status, and run `main_dual.py --backend ros2 --control-stack local-ros --ros2-config examples/piper_dual/ros2_piper_dual_local.yaml --run-tag local-ros ...`.

- [ ] **Step 3: Document official-ROS run**

```bash
cd /home/agilex/piper_ros/.worktrees/piper-official-humble
source /opt/ros/humble/setup.bash
source install-official/setup.bash
ros2 launch piper_official_bringup start_two_piper_official.launch.py \
  can_left_port:=can_left can_right_port:=can_right auto_enable:=false
```

After `/arm_status_left` and `/arm_status_right` are observed, explicitly enable:

```bash
ros2 service call /piper_left_ctrl_node/enable_srv piper_msgs/srv/Enable "{enable_request: true}"
ros2 service call /piper_right_ctrl_node/enable_srv piper_msgs/srv/Enable "{enable_request: true}"
```

Then run `main_dual.py --backend ros2 --control-stack official-ros --ros2-config examples/piper_dual/ros2_piper_dual.yaml --run-tag official-ros ...`.

- [ ] **Step 4: Document direct-SDK adapter run**

```bash
cd /home/agilex/client/.worktrees/piper-dual-ros2-migration
source /opt/ros/humble/setup.bash
source /home/agilex/piper_ros/.worktrees/piper-official-humble/install-official/setup.bash
export PYTHONPATH=/home/agilex/lgd/control_your_robot/src:$PYTHONPATH
/usr/bin/python3 examples/piper_dual/piper_direct_sdk_adapter.py \
  --left-can can_left --right-can can_right --allow-enable
```

Then run `main_dual.py --backend ros2 --control-stack direct-sdk --ros2-config examples/piper_dual/ros2_piper_dual_direct_sdk.yaml --run-tag direct-sdk ...`.

The runbook must state the fixed tested speed difference beside the commands: `local-ros=30%`, `official-ros=30%`, `direct-sdk=100%`. This is intentional user-selected behavior, not a controlled equal-speed experiment. Each run’s metadata must record the effective speed.

- [ ] **Step 5: State the manual stop/switch protocol**

Require this exact sequence between runs:

```bash
# Stop policy/client first with Ctrl-C.
# Stop the active ROS launch or direct adapter with Ctrl-C.
pgrep -af 'piper_single_ctrl|piper_direct_sdk_adapter|main_dual.py|ros2_bridge_process'
ip -details link show type can
```

No next stack starts until the previous control owner is gone and CAN names/bitrate are verified. No automatic mode switching or automatic retry is allowed.

- [ ] **Step 6: Run artifact and entrypoint tests**

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  examples/piper_dual/tests/test_main_dual.py \
  examples/piper_dual/tests/test_run_metadata.py
```

Expected: run-tag validation, stack-specific output directories, atomic metadata, exit reasons, and existing entrypoint behavior pass without hardware imports.

---

### Task 8: Run complete non-hardware verification and review

**Files:**
- Verify all changed client files and tests from Tasks 1–7, including `test_run_metadata.py`.
- Verify no modifications under `/home/agilex/lgd/control_your_robot` or the original local Piper control files.

- [ ] **Step 1: Run Python client tests**

```bash
cd /home/agilex/client/.worktrees/piper-dual-ros2-migration
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  examples/piper_dual/tests/test_main_dual.py \
  examples/piper_dual/tests/test_ros2_environment.py \
  examples/piper_dual/tests/test_ros2_backend.py \
  examples/piper_dual/tests/test_ros2_end_to_end_mock.py \
  examples/piper_dual/tests/test_control_stack.py \
  examples/piper_dual/tests/test_piper_direct_sdk_adapter.py \
  examples/piper_dual/tests/test_control_stack_integration.py \
  examples/piper_dual/tests/test_run_metadata.py
```

Expected: all client tests pass with no ROS plugin autoload failure and no hardware activity.

- [ ] **Step 2: Run system ROS-message tests**

```bash
/usr/bin/python3 -m pytest -q \
  examples/piper_dual/tests/test_ros2_contract.py \
  examples/piper_dual/tests/test_ros2_bridge_codec.py \
  examples/piper_dual/tests/test_ros2_safety.py
```

Expected: all system-Python ROS tests pass.

- [ ] **Step 3: Run syntax and help smoke checks**

```bash
python -m py_compile \
  examples/piper_dual/control_stack.py \
  examples/piper_dual/main_dual.py \
  examples/piper_dual/run_metadata.py \
  examples/piper_dual/saver.py \
  examples/piper_dual/ros2_environment.py \
  examples/piper_dual/ros2_backend.py \
  examples/piper_dual/ros2_bridge_process.py \
  examples/piper_dual/piper_direct_sdk_adapter.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python examples/piper_dual/main_dual.py --help
```

Expected: no syntax error; help lists all three `--control-stack` choices.

- [ ] **Step 4: Run exact no-action bridge smokes**

Run these three commands; each feeds only stop:

```bash
printf '%s\n' '{"type":"stop"}' | /usr/bin/python3 examples/piper_dual/ros2_bridge_process.py --config examples/piper_dual/ros2_piper_dual_local.yaml --control-stack local-ros --dry-run
printf '%s\n' '{"type":"stop"}' | /usr/bin/python3 examples/piper_dual/ros2_bridge_process.py --config examples/piper_dual/ros2_piper_dual.yaml --control-stack official-ros --dry-run
printf '%s\n' '{"type":"stop"}' | /usr/bin/python3 examples/piper_dual/ros2_bridge_process.py --config examples/piper_dual/ros2_piper_dual_direct_sdk.yaml --control-stack direct-sdk --dry-run
```

Expected per command: `ready` reports `publish_actions=false`, then `stopped`; no graph validation, publisher creation, Piper node, adapter, or CAN activity occurs.

- [ ] **Step 5: Verify fallback isolation**

```bash
git -C /home/agilex/lgd/control_your_robot status --short --branch
git -C /home/agilex/piper_ros status --short --branch
sha256sum /home/agilex/piper_ros/src/piper/piper/piper_ctrl_single_node.py \
  /home/agilex/piper_ros/src/piper/launch/start_two_piper.launch.py
```

Expected: no third-party control source modification and no source/install overwrite caused by this implementation.

- [ ] **Step 6: Review manual hardware gate**

Before any real run, verify the runbook, selected profile, CAN mapping, emergency stop, and explicit `--no-dry-run --publish-actions`. Do not claim real-hardware comparison until all three manually tagged runs have been observed and saved.

---

## Plan Self-Review

- **Spec coverage:** profile identity and endpoint normalization are Task 1; graph conflict protection, deferred live publisher creation, and bridge propagation are Task 2; unchanged third-party direct SDK integration is Task 3; unified `main_dual.py` is Task 4; profile mock coverage is Task 5; isolated official build is Task 6; runbook plus stack-isolated video/metadata output is Task 7; complete no-hardware verification and fallback checks are Task 8.
- **Placeholder scan:** no `TBD`, `TODO`, “implement later”, or unspecified future symbol appears in the task steps. Every referenced new function is defined in the task interface where first consumed.
- **Type consistency:** `ControlStackProfile`, `profile_for`, `load_profile_config`, `validate_profile_graph(node, profile, contract)`, `Ros2BackendClient.control_stack`, `PiperDirectSdkAdapter`, and `Args.control_stack` signatures are repeated consistently across tasks.
- **Safety consistency:** the direct adapter’s explicit `--allow-enable` is separate from the client’s `--no-dry-run --publish-actions`; external stack lifecycle remains manual; no test command opens CAN.
- **Fairness disclosure:** state/action units, strategy, scene, action delta, cameras, and runtime are normalized; speed is deliberately not normalized. Direct SDK retains its native 100%, while local/official ROS use 30%.
- **Known hardware prerequisite:** current machine CAN names must be confirmed before any real run. The existing `control_your_robot` example’s hard-coded `can0/can1` is not reused; the adapter commands use the verified `can_left/can_right` names.
