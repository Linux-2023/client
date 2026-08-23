# Task 4 Three-Stack Selection Report

## Result
DONE

## Scope
Focused only on:
- `examples/piper_dual/main_dual.py`
- `examples/piper_dual/env_dual.py`
- `examples/piper_dual/tests/test_main_dual.py`
- `examples/piper_dual/tests/test_ros2_environment.py`

No runtime artifacts, docs, integration suite, or unrelated files were changed for this task.

## RED
Focused tests initially failed because the entrypoint and tests were not aligned on the new control-stack contract and default selection behavior.

Observed failure during verification:
- `examples/piper_dual/tests/test_main_dual.py::test_build_environment_omits_ros2_flags_for_sdk_backend`
- `examples/piper_dual/tests/test_main_dual.py::test_build_environment_passes_ros2_flags_for_ros2_backend`

The failures showed the new parser/propagation expectations were not fully reflected in the implementation and test assumptions.

## GREEN
After the fixes:
- `--control-stack` accepts `local-ros`, `official-ros`, and `direct-sdk`.
- `--control-stack` is only valid with `--backend ros2`.
- Omitted control stack resolves to `official-ros` for ROS 2 and `None` for SDK.
- ROS 2 kwargs receive `control_stack`; SDK kwargs do not.
- Safe defaults are preserved: `dry_run=True`, `publish_actions=False`.
- The summary prints the selected control stack.
- Parser tests remain import-safe for SDK/ROS separation.

Verification result:
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q examples/piper_dual/tests/test_main_dual.py examples/piper_dual/tests/test_ros2_environment.py`
- Result: `56 passed`

## Changed paths
- `examples/piper_dual/main_dual.py`
- `examples/piper_dual/env_dual.py`
- `examples/piper_dual/tests/test_main_dual.py`
- `examples/piper_dual/tests/test_ros2_environment.py`

## Commit
Pending at report write time.

## Self-review
- Kept the change focused on the entrypoint/control-stack contract.
- Preserved SDK backend compatibility when explicitly selected.
- Preserved safe defaults and avoided importing ROS/hardware modules in parser-only paths.

## Concerns
- None for this task scope.
