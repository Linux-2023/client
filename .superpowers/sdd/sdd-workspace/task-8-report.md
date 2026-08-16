# Task 8 report

## Status
Implemented compatibility checks, no-hardware end-to-end Task 8 coverage, README compatibility/deployment documentation, and split-interpreter verification for Piper dual ROS 2 dataset/render/deployment compatibility.

## What changed
- Added `examples/piper_dual/tests/test_legacy_compatibility.py` covering structural detection of:
  - source-verified official legacy fixture `/home/agilex/lgd_data/episode0/episode0.hdf5`
  - structurally valid `piper_dual_ros2_v1` episodes independent of filename
  - structurally valid legacy files independent of filename
  - malformed new metadata, ambiguous mixed new/legacy structures, and unrecognized HDF5 files with descriptive rejection.
- Added `examples/piper_dual/tests/test_ros2_end_to_end_mock.py` composing fake bridge sensor events through the public `Ros2BackendClient` decoding/synchronizer boundary, `CollectorController`, `StreamingEpisodeWriter`, `validate_episode()`, and `render_episode()`.
- Added `detect_schema(path: Path) -> str` to `examples/piper_dual/render_dataset.py`, returning only `piper_dual_ros2_v1` or `legacy_official_hdf5` and rejecting malformed/ambiguous files instead of trusting filenames.
- Updated `render_episode()` to dispatch through schema detection and preserve the existing legacy visualizer refusal path.
- Updated `examples/piper_dual/visualize_hdf5.py` to use shared structural detection before dispatching new self-contained HDF5 files to the renderer.
- Updated `examples/piper_dual/README.md` with source-grounded ROS 2 collector/render/PI05 safety and official legacy data_tools commands.

## Original RED evidence
- Prior Task 8 worker observed initial RED collection failure before production changes:
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_legacy_compatibility.py examples/piper_dual/tests/test_ros2_end_to_end_mock.py -q
ERROR collecting examples/piper_dual/tests/test_legacy_compatibility.py
```
- During resume/self-review, client-side full collection exposed an existing-contract regression after adding stricter legacy schema detection:
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_action_adapter.py examples/piper_dual/tests/test_collector_state_machine.py examples/piper_dual/tests/test_dataset_schema.py examples/piper_dual/tests/test_frame_synchronizer.py examples/piper_dual/tests/test_legacy_compatibility.py examples/piper_dual/tests/test_main_dual.py examples/piper_dual/tests/test_observation_adapter.py examples/piper_dual/tests/test_render_dataset.py examples/piper_dual/tests/test_ros2_end_to_end_mock.py examples/piper_dual/tests/test_ros2_environment.py examples/piper_dual/tests/test_streaming_hdf5.py -q
FAILED examples/piper_dual/tests/test_render_dataset.py::test_render_episode_refuses_legacy_schema_with_clear_message
Actual message: unrecognized HDF5 schema ... legacy schema missing ...
pytest: 1 failed, 119 passed in 0.73s
```
- Fix: `render_episode()` now converts HDF5 files matching the pre-existing `visualize_hdf5.py` legacy shape (`observations/qpos` + `observations/images`) into the documented `legacy schema; use existing visualizer` refusal, while `detect_schema()` remains strict for official legacy detection and malformed/ambiguous rejections.

## GREEN focused Task 8 evidence
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_legacy_compatibility.py examples/piper_dual/tests/test_ros2_end_to_end_mock.py -q
..............                                                           [100%]
14 passed in 0.35s
```

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_render_dataset.py -q
....                                                                     [100%]
4 passed in 0.30s
```

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_ros2_end_to_end_mock.py -q
......                                                                   [100%]
6 passed in 0.39s
```

## Split-suite matrix
Single-interpreter collection still cannot cover all tests:

- Client `python` is Python 3.12 with client/HDF5/OpenCV dependencies, but ROS Humble C extensions are Python 3.10 artifacts. Full collection fails at `ros2_bridge_process.py` imports with `ModuleNotFoundError: No module named 'rclpy._rclpy_pybind11'` and NumPy/cv_bridge ABI stderr (`_ARRAY_API not found`).
- `/usr/bin/python3` is the ROS Humble interpreter and imports `rclpy`, `sensor_msgs.msg`, and `cv_bridge`, but it lacks client HDF5 dependency `h5py`; full collection fails on HDF5/client suites with `ModuleNotFoundError: No module named 'h5py'`.

Covered files under client `python`:
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_action_adapter.py examples/piper_dual/tests/test_collector_state_machine.py examples/piper_dual/tests/test_dataset_schema.py examples/piper_dual/tests/test_frame_synchronizer.py examples/piper_dual/tests/test_legacy_compatibility.py examples/piper_dual/tests/test_main_dual.py examples/piper_dual/tests/test_observation_adapter.py examples/piper_dual/tests/test_render_dataset.py examples/piper_dual/tests/test_ros2_environment.py examples/piper_dual/tests/test_streaming_hdf5.py -q
........................................................................ [ 63%]
..........................................                               [100%]
114 passed in 0.60s
```

Covered files under `/usr/bin/python3`:
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest examples/piper_dual/tests/test_ros2_protocol.py examples/piper_dual/tests/test_ros2_bridge_codec.py examples/piper_dual/tests/test_ros2_backend.py -q
..............................................                           [100%]
46 passed in 3.97s
```

All `examples/piper_dual/tests/*.py` files were assigned to one of those commands:
- client/HDF5/render/environment: `test_action_adapter.py`, `test_collector_state_machine.py`, `test_dataset_schema.py`, `test_frame_synchronizer.py`, `test_legacy_compatibility.py`, `test_main_dual.py`, `test_observation_adapter.py`, `test_render_dataset.py`, `test_ros2_environment.py`, `test_streaming_hdf5.py`.
- ROS bridge protocol/codec/backend: `test_ros2_protocol.py`, `test_ros2_bridge_codec.py`, `test_ros2_backend.py`.
- Task 8 focused mock: `test_ros2_end_to_end_mock.py` was run directly in focused/smoke commands; it is client/HDF5/OpenCV based and cannot be collected by `/usr/bin/python3` because that interpreter lacks `h5py`.

## No-hardware smoke and CLI evidence
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -c 'import rclpy, sensor_msgs.msg, cv_bridge; print("ROS2 bridge imports OK")'
ROS2 bridge imports OK
```

Direct CLI help was used because `uv` is unavailable:
```text
python examples/piper_dual/collect_data_ros2.py --help && python examples/piper_dual/render_dataset.py --help && python examples/piper_dual/main_dual.py --help
```
Output included:
```text
usage: collect_data_ros2.py [-h] --output-dir OUTPUT_DIR --prompt PROMPT --config CONFIG [--bridge-python BRIDGE_PYTHON] [--jpeg-quality JPEG_QUALITY] [--max-sync-error-ms MAX_SYNC_ERROR_MS] [--dry-run | --no-dry-run] [--publish-actions] [--render-after-save]
usage: render_dataset.py [-h] --input INPUT --output-dir OUTPUT_DIR [--fps FPS] [--no-plots]
usage: main_dual.py [-h] ... [--backend {sdk,ros2}] [--bridge-python BRIDGE_PYTHON] [--ros2-config ROS2_CONFIG] [--dry-run | --no-dry-run] [--publish-actions] ...
```

Py compile:
```text
python -m py_compile examples/piper_dual/render_dataset.py examples/piper_dual/visualize_hdf5.py examples/piper_dual/tests/test_legacy_compatibility.py examples/piper_dual/tests/test_ros2_end_to_end_mock.py && /usr/bin/python3 -m py_compile examples/piper_dual/render_dataset.py examples/piper_dual/visualize_hdf5.py examples/piper_dual/tests/test_legacy_compatibility.py examples/piper_dual/tests/test_ros2_end_to_end_mock.py
(no output)
```

Real ten-frame mock smoke:
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_ros2_end_to_end_mock.py -q
......                                                                   [100%]
6 passed in 0.39s
```
This test writes a real ten-frame HDF5, renders `cam_high.mp4`, `cam_left_wrist.mp4`, `cam_right_wrist.mp4`, fixed-order `views_3x1.mp4`, and writes/compares `quality.json` through public interfaces.

## Source-grounded documentation evidence
README commands are grounded in:
- `/home/agilex/data_ros/src/data_tools/README.md` official `run_data_capture.launch.py`, `data_sync.py`, `data_to_hdf5.py`, `run_data_publish.launch.py`, and `data_publish.py` command forms.
- `/home/agilex/data_ros/src/data_tools/launch/run_data_capture.launch.py`, `run_data_sync.launch.py`, and `run_data_publish.launch.py` arguments.
- `/home/agilex/data_ros/src/data_tools/scripts/data_sync.py`, `data_to_hdf5.py`, and `data_publish.py` argument parsers.
- `/home/agilex/camera_ros/scripts/start_realsense_3cam_color.sh` for the three-RealSense color node command.
- `/home/agilex/piper_ros/start_multi_piper.sh` for the dual Piper ROS 2 launch path.
- `collect_data_ros2.py`, `render_dataset.py`, and `main_dual.py --help` for the new collector/render/PI05 dry-run and explicit physical flags.

## Limitations and safety statement
- No ROS nodes were launched.
- No hardware was moved.
- No physical action publishing was performed.
- Physical deployment success is not claimed. It still requires the hardware checkpoint from the Task 8 brief: start official camera and Piper ROS 2 nodes, validate dry-run collector and PI05 observation/action shapes, confirm CAN/topic/safety conditions, then explicitly opt into low-risk physical publishing with `--publish-actions --no-dry-run`.

## Self-review
- Schema detection is structural and has tests for official legacy, valid new, valid legacy, malformed new, ambiguous mixed, and unrecognized files.
- Ten-frame flow uses public collection/writer/renderer boundaries and real HDF5/video artifacts.
- Five named no-false-success failures are covered: incomplete frame, malformed JPEG, stale timestamp, bridge EOF, interrupted finalization.
- All test files in `examples/piper_dual/tests` are covered under an interpreter with the required dependencies; single-interpreter impossibility is recorded with actual failures.
- Documentation does not claim physical verification.

## Task 8 visualizer dispatch fix

### RED evidence
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_render_dataset.py::test_visualize_hdf5_schema_gate_distinguishes_valid_new_from_official_legacy examples/piper_dual/tests/test_render_dataset.py::test_visualize_hdf5_main_rejects_invalid_schema_without_constructing_legacy_visualizer -q
FAILED examples/piper_dual/tests/test_render_dataset.py::test_visualize_hdf5_main_rejects_invalid_schema_without_constructing_legacy_visualizer[malformed_new_missing_camera-malformed piper_dual_ros2_v1.*observations/images/cam_right_wrist]
FAILED examples/piper_dual/tests/test_render_dataset.py::test_visualize_hdf5_main_rejects_invalid_schema_without_constructing_legacy_visualizer[ambiguous_mixed-ambiguous HDF5 schema.*piper_dual_ros2_v1.*legacy_official_hdf5]
FAILED examples/piper_dual/tests/test_render_dataset.py::test_visualize_hdf5_main_rejects_invalid_schema_without_constructing_legacy_visualizer[unrecognized-unrecognized HDF5 schema.*missing]
E       AssertionError: HDF5Visualizer should not be constructed for invalid schemas
pytest: 3 failed, 1 passed in 0.23s
```

### GREEN evidence
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_render_dataset.py::test_visualize_hdf5_schema_gate_distinguishes_valid_new_from_official_legacy examples/piper_dual/tests/test_render_dataset.py::test_visualize_hdf5_main_rejects_invalid_schema_without_constructing_legacy_visualizer examples/piper_dual/tests/test_render_dataset.py::test_visualize_hdf5_cli_reports_invalid_schema_and_exits_nonzero -q
.....                                                                    [100%]
5 passed in 0.35s
```

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_render_dataset.py -q
.........                                                                [100%]
9 passed in 0.46s
```

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_legacy_compatibility.py examples/piper_dual/tests/test_ros2_end_to_end_mock.py -q
..............                                                           [100%]
14 passed in 0.35s
```

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_action_adapter.py examples/piper_dual/tests/test_collector_state_machine.py examples/piper_dual/tests/test_dataset_schema.py examples/piper_dual/tests/test_frame_synchronizer.py examples/piper_dual/tests/test_legacy_compatibility.py examples/piper_dual/tests/test_main_dual.py examples/piper_dual/tests/test_observation_adapter.py examples/piper_dual/tests/test_render_dataset.py examples/piper_dual/tests/test_ros2_end_to_end_mock.py examples/piper_dual/tests/test_ros2_environment.py examples/piper_dual/tests/test_streaming_hdf5.py -q
........................................................................ [ 57%]
.....................................................                    [100%]
125 passed in 0.96s
```

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest examples/piper_dual/tests/test_ros2_protocol.py examples/piper_dual/tests/test_ros2_bridge_codec.py examples/piper_dual/tests/test_ros2_backend.py -q
..............................................                           [100%]
46 passed in 3.98s
```

```text
python -m py_compile examples/piper_dual/visualize_hdf5.py examples/piper_dual/tests/test_render_dataset.py && /usr/bin/python3 -m py_compile examples/piper_dual/visualize_hdf5.py examples/piper_dual/tests/test_render_dataset.py
(no output)
```

### Dispatch invariant
- `detect_schema()` is the sole structural discriminator.
- `_is_new_schema_episode()` compares the detector result to `piper_dual_ros2_v1` and no longer converts `ValueError` into a legacy fallback.
- `visualize_hdf5.py` dispatches `piper_dual_ros2_v1` to `render_episode()` and only explicit `legacy_official_hdf5` to `HDF5Visualizer`.
- Malformed declared-new, ambiguous mixed, unrecognized, and unreadable HDF5 errors propagate to CLI failure; they must not instantiate `HDF5Visualizer` or print the success banner.

## Mixed legacy/new schema detector fix

### RED evidence
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_legacy_compatibility.py::test_detect_schema_rejects_legacy_episode_with_new_schema_path_without_schema_version -q
F                                                                        [100%]
=================================== FAILURES ===================================
_ test_detect_schema_rejects_legacy_episode_with_new_schema_path_without_schema_version _

tmp_path = PosixPath('/tmp/pytest-of-agilex/pytest-66/test_detect_schema_rejects_leg0')

    def test_detect_schema_rejects_legacy_episode_with_new_schema_path_without_schema_version(tmp_path: Path) -> None:
        mixed = _write_structural_legacy_episode(tmp_path / "mixed_legacy_new.hdf5")
        with h5py.File(mixed, "a") as episode:
            observations = episode.create_group("observations")
            observations.create_dataset("state", data=np.zeros((1, 14), dtype=np.float32), maxshape=(None, 14))

>       with pytest.raises(ValueError, match="ambiguous HDF5 schema.*mixed legacy/new.*legacy_official_hdf5.*observations/state"):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       Failed: DID NOT RAISE ValueError

examples/piper_dual/tests/test_legacy_compatibility.py:122: Failed
=========================== short test summary info ============================
FAILED examples/piper_dual/tests/test_legacy_compatibility.py::test_detect_schema_rejects_legacy_episode_with_new_schema_path_without_schema_version - Failed: DID NOT RAISE ValueError
1 failed in 0.12s
```

### GREEN evidence
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_legacy_compatibility.py::test_detect_schema_rejects_legacy_episode_with_new_schema_path_without_schema_version -q
.                                                                        [100%]
1 passed in 0.11s
```

### Verification
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_legacy_compatibility.py examples/piper_dual/tests/test_render_dataset.py examples/piper_dual/tests/test_ros2_end_to_end_mock.py -q
........................                                                 [100%]
24 passed in 0.71s
```

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_action_adapter.py examples/piper_dual/tests/test_collector_state_machine.py examples/piper_dual/tests/test_dataset_schema.py examples/piper_dual/tests/test_frame_synchronizer.py examples/piper_dual/tests/test_legacy_compatibility.py examples/piper_dual/tests/test_main_dual.py examples/piper_dual/tests/test_observation_adapter.py examples/piper_dual/tests/test_render_dataset.py examples/piper_dual/tests/test_ros2_end_to_end_mock.py examples/piper_dual/tests/test_ros2_environment.py examples/piper_dual/tests/test_streaming_hdf5.py -q
........................................................................ [ 57%]
......................................................                   [100%]
126 passed in 0.99s
```

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest examples/piper_dual/tests/test_ros2_protocol.py examples/piper_dual/tests/test_ros2_bridge_codec.py examples/piper_dual/tests/test_ros2_backend.py -q
..............................................                           [100%]
46 passed in 3.98s
```

```text
python -m py_compile examples/piper_dual/render_dataset.py examples/piper_dual/tests/test_legacy_compatibility.py && /usr/bin/python3 -m py_compile examples/piper_dual/render_dataset.py examples/piper_dual/tests/test_legacy_compatibility.py
(no output)
```

### Invariant
`detect_schema()` returns `legacy_official_hdf5` only when the legacy structure is unambiguous; if any new-schema required attr/path is present alongside all legacy required paths and `schema_version` is absent, it raises an ambiguous/mixed error instead of falling through to legacy. Complete new metadata/paths still returns `piper_dual_ros2_v1`, and malformed declared-new or unsupported schema versions keep their existing errors.
