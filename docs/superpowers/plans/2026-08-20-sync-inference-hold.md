# 同步推理保持目标 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在同步 ROS 2 推理阻塞期间，以推理开始时的当前反馈姿态周期性重发保持目标，避免旧 action chunk 末尾目标继续驱动机械臂并在新 chunk 到达时产生回退。

**Architecture:** 仅在同步模式启用保持机制。`Runtime._step()` 在调用可能阻塞的 agent 推理前调用可选环境钩子，推理完成或异常时停止保持；`Ros2DualEnvironment` 使用后台线程以 20 Hz 重发一次姿态快照，停止线程并 join 后才允许新动作发布，消除旧保持消息覆盖新动作的竞态。异步/RTC 路径不启用该钩子。

**Tech Stack:** Python 3.11 client runtime, ROS 2 Humble bridge, NumPy, pytest.

## Global Constraints

- 不修改模型、server、JAX/PyTorch checkpoint 或 Piper SDK。
- 不改变异步/RTC action broker 行为。
- 保持目标使用进入推理时的 14 维 `observation["state"]` 快照；不把旧 action 目标当作保持目标。
- 保持线程退出后才能发布新动作；所有保持发布失败必须传播为环境错误并停止动作发布。
- 真机发布仍受现有 `dry_run`、`publish_actions`、动作维度和 finite 校验约束。

---

### Task 1: 同步保持失败测试

**Files:**
- Modify: `examples/piper_dual/tests/test_ros2_environment.py`
- Modify: `packages/openpi-client/src/openpi_client/runtime/runtime_test.py` (若不存在则创建)

**Interfaces:**
- `Ros2DualEnvironment.begin_inference_hold(observation: dict[str, Any]) -> None`
- `Ros2DualEnvironment.end_inference_hold() -> None`
- `Runtime._step()` calls these optional environment hooks only when `getattr(environment, "inference_hold_enabled", False)` is true.

- [ ] **Step 1: Write failing tests**
  - Add a fake backend test proving `begin_inference_hold` publishes the observation state and `end_inference_hold` stops further hold publishes.
  - Add a runtime test proving the hold starts before a blocking agent call and ends before `apply_action`.
  - Assert hold uses the 14-dim state, not the agent's returned action.

- [ ] **Step 2: Run focused tests and verify expected failure**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q examples/piper_dual/tests/test_ros2_environment.py packages/openpi-client/src/openpi_client/runtime/runtime_test.py
```

Expected: FAIL because the hold hooks and runtime integration do not exist.

### Task 2: Implement environment hold publisher

**Files:**
- Modify: `examples/piper_dual/ros2_environment.py`

**Interfaces:**
- `inference_hold_enabled: bool` constructor option, default `False`.
- `begin_inference_hold(observation)` snapshots `observation["state"]`, starts one daemon thread, and publishes the snapshot at 20 Hz.
- `end_inference_hold()` signals and joins the thread; it is idempotent.

- [ ] **Step 1: Add validated state snapshot and lifecycle fields**
- [ ] **Step 2: Add 20 Hz hold loop using `backend.publish_action`**
- [ ] **Step 3: Serialize hold shutdown before normal `apply_action`**
- [ ] **Step 4: On hold publish failure, latch the existing fatal environment state and surface the exception**

### Task 3: Integrate hooks into runtime only for synchronous mode

**Files:**
- Modify: `packages/openpi-client/src/openpi_client/runtime/runtime.py`
- Modify: `examples/piper_dual/main_dual_ros.py`

**Interfaces:**
- Runtime uses optional `inference_hold_enabled`, `begin_inference_hold`, and `end_inference_hold` without affecting environments that do not implement them.
- `main_dual_ros._build_environment` passes `inference_hold_enabled=not args.use_async`.

- [ ] **Step 1: Wrap agent inference in `try/finally` hold lifecycle**
- [ ] **Step 2: Stop and join hold before `environment.apply_action(action)`**
- [ ] **Step 3: Ensure async/RTC mode receives `inference_hold_enabled=False`**

### Task 4: Verify simulated synchronous behavior

**Files:**
- Modify: `examples/piper_dual/tests/test_ros2_environment.py` only if a test assertion needs tightening.

- [ ] **Step 1: Run focused environment/runtime tests**
- [ ] **Step 2: Run existing action adapter and frame synchronizer tests**
- [ ] **Step 3: Run a deterministic simulation with a blocking fake agent and assert all hold commands equal the pre-inference state until the new action is applied**
- [ ] **Step 4: Run CLI help and Python syntax checks**

Expected: focused tests pass; no real hardware command is issued during verification.
