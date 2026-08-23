# Task 2 Three-Stack Bridge Guard Report

## Scope
- Implemented profile-aware bridge graph protection and `control_stack` propagation for Task 2 only.
- Target paths changed:
  - `examples/piper_dual/ros2_contract.py`
  - `examples/piper_dual/ros2_bridge_process.py`
  - `examples/piper_dual/ros2_backend.py`
  - `examples/piper_dual/tests/test_ros2_contract.py`
  - `examples/piper_dual/tests/test_ros2_bridge_codec.py`
  - `examples/piper_dual/tests/test_ros2_backend.py`

## RED
- `/usr/bin/python3 -m pytest -q examples/piper_dual/tests/test_ros2_contract.py examples/piper_dual/tests/test_ros2_bridge_codec.py`
  - Observed failure: `ImportError: cannot import name 'validate_profile_graph' from 'ros2_contract'`.
- `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q examples/piper_dual/tests/test_ros2_backend.py::test_bridge_backend_forwards_control_stack_to_sidecar`
  - Observed failure: `TypeError: Ros2BackendClient.__init__() got an unexpected keyword argument 'control_stack'`.

## GREEN
- `/usr/bin/python3 -m pytest -q examples/piper_dual/tests/test_ros2_contract.py examples/piper_dual/tests/test_ros2_bridge_codec.py`
  - Output: `36 passed in 0.39s`.
- `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q examples/piper_dual/tests/test_ros2_backend.py`
  - Output: `26 passed in 3.89s`.

## Implementation notes
- Added `validate_profile_graph(node, profile, contract)` to validate required/forbidden stack topics, contract topic ROS types, direct-SDK adapter node identity, and exactly one external action subscriber.
- Added bridge-side `--control-stack` parsing and construction-time config identity validation.
- Changed live action publisher creation to be deferred until the first valid live action request passes graph/status readiness; any readiness failure latches action publishing off for the bridge process.
- Preserved dry-run behavior: no graph validation and zero action publishers.
- Added `Ros2BackendClient(control_stack="official-ros")`, stored `self.control_stack`, and forwarded it to the sidecar argv.

## Commits
- `5e7c136` Task 2 profile-aware bridge guard

## Self-review
- Checked target diff and staged only Task 2 implementation/test paths.
- Ran `git diff --check --` on Task 2 paths: no output.
- Did not run ROS nodes, CAN, hardware, broad suites, formatters, direct SDK adapter work, main CLI work, or docs/build integration.

## Concerns
- The worktree contains many pre-existing unrelated unstaged/untracked user edits outside this task; they were not staged or committed.
- No live ROS graph or hardware verification was run by design; verification is limited to the focused ROS-message and backend tests above.

## Review fix follow-up

### RED
- `/usr/bin/python3 -m pytest -q examples/piper_dual/tests/test_ros2_contract.py::test_profile_graph_rejects_selected_contract_action_topic_without_subscriber examples/piper_dual/tests/test_ros2_bridge_codec.py::test_bridge_rechecks_status_before_every_live_joint_publish examples/piper_dual/tests/test_ros2_bridge_codec.py::test_bridge_eef_live_publishers_require_profile_graph_and_status_ready`
  - Output: `3 failed in 0.37s`.
  - Failures: selected contract action topic without subscriber did not raise; live joint publish after status fault did not raise; EEF graph guard did not raise.
- `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q examples/piper_dual/tests/test_ros2_environment.py::test_environment_forwards_control_stack_to_backend`
  - Output: `2 failed in 0.18s`.
  - Failure: `KeyError: 'control_stack'` for both local-ros and direct-sdk backend construction captures.

### GREEN
- `/usr/bin/python3 -m pytest -q examples/piper_dual/tests/test_ros2_contract.py::test_profile_graph_rejects_selected_contract_action_topic_without_subscriber examples/piper_dual/tests/test_ros2_bridge_codec.py::test_bridge_rechecks_status_before_every_live_joint_publish examples/piper_dual/tests/test_ros2_bridge_codec.py::test_bridge_eef_live_publishers_require_profile_graph_and_status_ready`
  - Output: `3 passed in 0.32s`.
- `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q examples/piper_dual/tests/test_ros2_environment.py::test_environment_forwards_control_stack_to_backend`
  - Output: `2 passed in 0.13s`.
- `/usr/bin/python3 -m pytest -q examples/piper_dual/tests/test_ros2_contract.py examples/piper_dual/tests/test_ros2_bridge_codec.py`
  - Output: `39 passed in 0.39s`.
- `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q examples/piper_dual/tests/test_ros2_backend.py examples/piper_dual/tests/test_ros2_environment.py::test_environment_forwards_control_stack_to_backend`
  - Output: `28 passed in 3.88s`.

### Fix notes
- Re-checks graph and status readiness before every live publish, including after publishers already exist.
- Applies the same selected-profile graph/status guard to EEF live publisher creation.
- Counts subscribers on the selected contract action topics actually published.
- Added `control_stack` to `Ros2DualEnvironment` and forwards it to `Ros2BackendClient`.
