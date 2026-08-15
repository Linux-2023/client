# Task 2 Report: Shared Piper Dual Observation and Action Adapters

## Status
Implemented Task 2 in `/home/agilex/client/.worktrees/piper-dual-ros2-migration`.

## Commit
- Implementation commit subject: `feat: define shared Piper dual observation contract`

## Changed Files
- `examples/piper_dual/observation_adapter.py`
  - Added `CameraSpec` with validated name/topic/size/rotation fields.
  - Added ROS-independent JPEG decode to RGB HWC and model-image encode to RGB CHW.
  - Added observation adaptation for `observation.state`, `images`, `timestamps`, `sync_error`, and `task`.
  - Enforced 14-value state vectors, finite scalar validation, and deterministic aspect-ratio-preserving resize with zero padding.
- `examples/piper_dual/action_adapter.py`
  - Added 14-value action validation, left/right split, and Task-1-compatible `publish_action` bridge request generation.
- `examples/piper_dual/ros2_piper_dual.yaml`
  - Added the fixed ALOHA/camera contract: schema version, sync limit, JPEG quality, model image format, camera mappings, and explicit state/action names, ordering, and units.
- `examples/piper_dual/tests/test_observation_adapter.py`
  - Added focused contract tests for camera specs, JPEG decode, resize padding, CHW conversion, observation adaptation, malformed inputs, and YAML contract.
- `examples/piper_dual/tests/test_action_adapter.py`
  - Added focused contract tests for action validation, split behavior, and bridge request shape.

## TDD RED Evidence
First RED run after the tests existed but before implementation:

```text
/usr/bin/python3 -m pytest examples/piper_dual/tests/test_observation_adapter.py examples/piper_dual/tests/test_action_adapter.py -q
E   ModuleNotFoundError: No module named 'observation_adapter'
```

Second RED run after adding the scalar-validation tests, before tightening the adapter:

```text
/usr/bin/python3 -m pytest examples/piper_dual/tests/test_observation_adapter.py examples/piper_dual/tests/test_action_adapter.py -q
FAILED examples/piper_dual/tests/test_observation_adapter.py::test_adapt_rejects_non_numeric_sync_error[12.5]
FAILED examples/piper_dual/tests/test_observation_adapter.py::test_adapt_rejects_non_numeric_sync_error[True]
FAILED examples/piper_dual/tests/test_observation_adapter.py::test_adapt_rejects_non_numeric_timestamp_values[0.5]
FAILED examples/piper_dual/tests/test_observation_adapter.py::test_adapt_rejects_non_numeric_timestamp_values[True]
```

## GREEN / Verification Evidence
Focused Task 2 tests:

```text
/usr/bin/python3 -m pytest examples/piper_dual/tests/test_observation_adapter.py examples/piper_dual/tests/test_action_adapter.py -q
25 passed in 0.20s
```

Python compile check:

```text
/usr/bin/python3 -m py_compile examples/piper_dual/observation_adapter.py examples/piper_dual/action_adapter.py
(no output)
```

## Self-Review
- Adapter modules are ROS-independent and only depend on NumPy/OpenCV.
- The observation adapter preserves the RGB HWC input path separately from the RGB CHW model tensor path.
- The action adapter normalizes to float32, rejects malformed/non-finite inputs, and emits the expected `publish_action` shape.
- The YAML contract matches the fixed camera names, 224x224 model image size, 0° default rotation, 30 ms sync limit, and left-then-right state/action ordering.
- Focused tests cover the required contract points and passed cleanly.
- No formatter, linter, or project-wide test suite was run, per brief.

## Concerns
- No live ROS runtime or hardware smoke test was run; verification is limited to focused unit tests and `py_compile`.
- The report and verification were kept within Task 2 scope; Task 1 implementation and official ROS workspaces were not modified.
