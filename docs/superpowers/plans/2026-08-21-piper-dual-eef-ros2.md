# Piper Dual ROS 2 EEF Collector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `collect_data_eef_ros2.py` so every saved ROS 2 frame contains the existing dual-arm joint state/action plus synchronized puppet-left and puppet-right EEF poses.

**Architecture:** Extend the shared bridge/backend/synchronizer/frame/writer pipeline with opt-in EEF support; leave all defaults unchanged for `collect_data_ros2.py`. The EEF collector enables two `PoseStamped` streams, requires them during timestamp synchronization, and writes two frame-aligned `(N, 6)` float32 HDF5 datasets.

**Tech Stack:** Python 3.11, ROS 2 Humble (`rclpy`, `geometry_msgs/PoseStamped`), NumPy, h5py, OpenCV, pytest.

## Global Constraints

- Collect only puppet EEF from `/puppet/end_pose_left` and `/puppet/end_pose_right` by default.
- Store EEF vectors as `[x, y, z, roll, pitch, yaw]`, using meters and radians.
- Require both EEF samples inside `--max-sync-error-ms`; never write NaN placeholders or an independent time series.
- Preserve the existing 14D `/observations/state`, 14D `/action`, `piper_dual_ros2_v1` schema version, and default collector behavior.
- Mark EEF episodes with `metadata_json.eef.enabled = true` and store `/observations/eef/puppet_left` and `/observations/eef/puppet_right`.
- Do not modify `convert_ros2_piper_data_to_lerobot.py`.
- Keep dry-run enabled by default and retain the existing `--dry-run` plus `--publish-actions` safety rejection.

---

### Task 1: Opt-in EEF frame synchronization

**Files:**
- Modify: `examples/piper_dual/frame_synchronizer.py`
- Modify: `examples/piper_dual/tests/test_frame_synchronizer.py`

**Interfaces:**
- Consumes: bridge sensor names `eef_puppet_left` and `eef_puppet_right`; each value is a finite NumPy vector of shape `(6,)`.
- Produces: `EEF_SENSORS`, `EEF_DIMENSION`, `FrameSynchronizer(..., include_eef: bool = False)`, `FrameSynchronizer.required_sensors`, and `SynchronizedFrame.eef: dict[str, np.ndarray] | None`.

- [ ] **Step 1: Write failing opt-in EEF synchronization tests**

Add helpers and tests that prove the default required sensor set is unchanged, EEF mode waits for both pose streams, accepts samples exactly at the error boundary, and returns aligned vectors:

```python
EEF_SENSORS = ("eef_puppet_left", "eef_puppet_right")

def eef_values(start: float) -> np.ndarray:
    return np.arange(start, start + 6, dtype=np.float32)

def test_eef_mode_requires_and_aligns_both_pose_streams():
    sync = FrameSynchronizer(max_error=0.03, include_eef=True)
    push_complete_frame(sync, 12.0)
    sync.push("eef_puppet_left", 11.97, eef_values(40))
    assert sync.try_sync() is None
    sync.push("eef_puppet_right", 12.03, eef_values(50))
    frame = sync.try_sync()
    assert frame is not None
    np.testing.assert_array_equal(frame.eef["puppet_left"], eef_values(40))
    np.testing.assert_array_equal(frame.eef["puppet_right"], eef_values(50))
    assert frame.sensor_timestamps["eef_puppet_left"] == pytest.approx(11.97)
    assert frame.sync_error["eef_puppet_right"] == pytest.approx(0.03)

def test_default_mode_rejects_eef_sensor_and_emits_frame_without_eef():
    sync = FrameSynchronizer(max_error=0.03)
    assert "eef_puppet_left" not in sync.required_sensors
    with pytest.raises(ValueError, match="unknown sensor"):
        sync.push("eef_puppet_left", 1.0, eef_values(0))
    push_complete_frame(sync, 1.0)
    assert sync.try_sync().eef is None
```

Also add invalid EEF shape, non-finite value, missing stream, and stale stream cases.

- [ ] **Step 2: Run the focused test and observe failure**

Run:

```bash
.venv/bin/pytest examples/piper_dual/tests/test_frame_synchronizer.py -v
```

Expected: failures because `include_eef`, `required_sensors`, and `SynchronizedFrame.eef` do not exist.

- [ ] **Step 3: Implement configurable required sensors and EEF frame data**

Add constants and the optional frame field without changing existing constructors:

```python
EEF_SENSORS = ("eef_puppet_left", "eef_puppet_right")
EEF_DIMENSION = 6

@dataclass(frozen=True, slots=True)
class SynchronizedFrame:
    timestamp: float
    images: dict[str, bytes]
    state: np.ndarray
    action: np.ndarray
    sensor_timestamps: dict[str, float]
    sync_error: dict[str, float]
    eef: dict[str, np.ndarray] | None = None
```

Make the synchronizer own its required set:

```python
def __init__(self, max_error: float, max_buffer_seconds: float = 2.0, *, include_eef: bool = False) -> None:
    self.include_eef = bool(include_eef)
    self.required_sensors = REQUIRED_SENSORS + EEF_SENSORS if self.include_eef else REQUIRED_SENSORS
    self._buffers = {sensor: deque() for sensor in self.required_sensors}
```

Use `self.required_sensors` for matching and timing dictionaries. Validate 7D joint vectors and 6D EEF vectors separately. In `_build_frame`, create:

```python
eef = None
if self.include_eef:
    eef = {
        "puppet_left": selected["eef_puppet_left"].value.copy(),
        "puppet_right": selected["eef_puppet_right"].value.copy(),
    }
```

- [ ] **Step 4: Run the synchronization tests**

Run:

```bash
.venv/bin/pytest examples/piper_dual/tests/test_frame_synchronizer.py -v
```

Expected: all synchronization tests pass, including unchanged default-mode tests.

---

### Task 2: PoseStamped bridge events and backend transport

**Files:**
- Modify: `examples/piper_dual/ros2_bridge_process.py`
- Modify: `examples/piper_dual/ros2_backend.py`
- Modify: `examples/piper_dual/tests/test_ros2_bridge_codec.py`
- Modify: `examples/piper_dual/tests/test_ros2_backend.py`

**Interfaces:**
- Consumes: optional backend arguments `eef_left_topic: str | None`, `eef_right_topic: str | None` and ROS `geometry_msgs/msg/PoseStamped` messages.
- Produces: bridge CLI flags `--eef-left-topic` and `--eef-right-topic`; protocol sensor events with six `values`; backend EEF-mode synchronizer and dimension-aware vector decoding.

- [ ] **Step 1: Write failing PoseStamped conversion tests**

Extend fake messages with position/orientation fields and add:

```python
def test_build_eef_sensor_event_converts_quaternion_to_xyz_rpy():
    message = PoseMessage(
        sec=4,
        nanosec=500_000_000,
        position=(0.1, -0.2, 0.3),
        orientation=(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)),
    )
    event = build_eef_sensor_event("eef_puppet_left", message)
    assert event["type"] == "sensor"
    assert event["sensor"] == "eef_puppet_left"
    assert event["timestamp"] == pytest.approx(4.5)
    assert event["values"] == pytest.approx([0.1, -0.2, 0.3, 0.0, 0.0, math.pi / 2])
```

Add rejection tests for zero-norm quaternion and non-finite position/orientation values.

- [ ] **Step 2: Write failing backend launch/decode tests**

Assert ordinary backend launch arguments contain no EEF flags; EEF backend adds both flags, creates `FrameSynchronizer(include_eef=True)`, accepts exactly six values for EEF sensors, and still requires exactly seven for joint sensors.

```python
client = Ros2BackendClient(
    bridge_python=Path("/usr/bin/python3"),
    config=Path("config.yaml"),
    eef_left_topic="/custom/left_pose",
    eef_right_topic="/custom/right_pose",
)
assert client.synchronizer.include_eef is True
```

- [ ] **Step 3: Run bridge/backend tests and observe failure**

Run:

```bash
.venv/bin/pytest \
  examples/piper_dual/tests/test_ros2_bridge_codec.py \
  examples/piper_dual/tests/test_ros2_backend.py -v
```

Expected: failures for the missing EEF event builder and backend constructor arguments.

- [ ] **Step 4: Implement bridge PoseStamped support**

Import `PoseStamped` and add default-disabled topic arguments. Use a local quaternion-to-RPY implementation equivalent to tf2 roll-pitch-yaw conventions, normalizing the quaternion and validating every input with `math.isfinite`:

```python
def build_eef_sensor_event(sensor: str, message: Any) -> JsonObject:
    position = message.pose.position
    orientation = message.pose.orientation
    roll, pitch, yaw = _quaternion_to_rpy(
        float(orientation.x), float(orientation.y), float(orientation.z), float(orientation.w)
    )
    values = [float(position.x), float(position.y), float(position.z), roll, pitch, yaw]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("EEF pose values must be finite")
    return {
        "type": "sensor",
        "sensor": sensor,
        "timestamp": _format_sensor_timestamp(message.header.stamp),
        "values": values,
    }
```

When both EEF topic flags are present, subscribe with sensor names `eef_puppet_left` and `eef_puppet_right`. Reject configurations where only one EEF topic is supplied.

- [ ] **Step 5: Implement dimension-aware backend transport**

Extend `Ros2BackendClient.__init__` with optional paired topic arguments. Validate non-empty strings, construct `FrameSynchronizer(..., include_eef=True)`, append topic flags in `start()`, and decode `values` according to sensor name:

```python
if "values" in message:
    expected_dimension = EEF_DIMENSION if sensor in EEF_SENSORS else JOINT_DIMENSION
    value = self._decode_vector_payload(message["values"], expected_dimension, sensor)
```

Do not alter ordinary backend arguments or its default sensor set.

- [ ] **Step 6: Run bridge/backend tests**

Run:

```bash
.venv/bin/pytest \
  examples/piper_dual/tests/test_ros2_bridge_codec.py \
  examples/piper_dual/tests/test_ros2_backend.py -v
```

Expected: all selected tests pass.

---

### Task 3: EEF-aware streaming HDF5 contract

**Files:**
- Modify: `examples/piper_dual/streaming_hdf5.py`
- Modify: `examples/piper_dual/tests/test_streaming_hdf5.py`

**Interfaces:**
- Consumes: `StreamingEpisodeWriter.open(..., include_eef: bool = False)` and EEF-mode `SynchronizedFrame.eef` mappings.
- Produces: `/observations/eef/puppet_left`, `/observations/eef/puppet_right`, EEF sensor timestamp/error datasets, and metadata-driven EEF validation.

- [ ] **Step 1: Write failing EEF writer round-trip tests**

Add EEF frame helpers and verify the full contract:

```python
def test_eef_writer_round_trips_frame_aligned_pose_datasets(tmp_path):
    final_path = tmp_path / "eef.hdf5"
    metadata = {
        "collector": "collect_data_eef_ros2.py",
        "eef": {"enabled": True, "order": ["x", "y", "z", "roll", "pitch", "yaw"]},
    }
    writer = StreamingEpisodeWriter.open(final_path, metadata=metadata, include_eef=True)
    expected = eef_frame(0)
    writer.append(expected)
    writer.finalize()
    with h5py.File(final_path, "r") as episode:
        np.testing.assert_array_equal(episode["/observations/eef/puppet_left"][0], expected.eef["puppet_left"])
        np.testing.assert_array_equal(episode["/observations/eef/puppet_right"][0], expected.eef["puppet_right"])
        assert episode["/observations/eef/puppet_left"].maxshape == (None, 6)
        assert episode["/observations/sensor_timestamps/eef_puppet_left"].shape == (1,)
    assert validate_episode(final_path)["errors"] == []
```

Add tests that EEF mode rejects `eef=None`, missing one arm, wrong shapes, non-finite values, and a marked file with a deleted/mismatched dataset. Assert ordinary writer files do not contain `/observations/eef`.

- [ ] **Step 2: Run writer tests and observe failure**

Run:

```bash
.venv/bin/pytest examples/piper_dual/tests/test_streaming_hdf5.py -v
```

Expected: failures because `include_eef` and EEF datasets do not exist.

- [ ] **Step 3: Implement opt-in EEF datasets and validation**

Add constants:

```python
EEF_GROUP_PATH = "/observations/eef"
EEF_DATASET_NAMES = ("puppet_left", "puppet_right")
```

Store `_include_eef` and `_required_sensors` on the writer. `open()` validates that `include_eef` equals the metadata marker, initializes EEF datasets only when enabled, and creates timing datasets for the nine required sensors. `append()` validates the EEF mapping before resizing, writes both vectors atomically with the rest of the logical frame, and uses the instance sensor set for timing mappings.

Validation parses `metadata_json`, derives EEF mode from `eef.enabled is True`, and checks both `(frame_count, 6)` datasets, float32 dtype, finite values, and `(None, 6)` maxshape. It checks EEF timing datasets only for marked files. Existing unmarked v1 files keep their current validation path.

- [ ] **Step 4: Run writer tests**

Run:

```bash
.venv/bin/pytest examples/piper_dual/tests/test_streaming_hdf5.py -v
```

Expected: all writer tests pass, including ordinary-file regression cases.

---

### Task 4: New EEF collector entry point

**Files:**
- Create: `examples/piper_dual/collect_data_eef_ros2.py`
- Modify: `examples/piper_dual/collect_data_ros2.py`
- Create: `examples/piper_dual/tests/test_collect_data_eef_ros2.py`
- Modify: `examples/piper_dual/tests/test_collector_state_machine.py`

**Interfaces:**
- Consumes: shared `PreviewWindow`, `CollectorController`, `run_collection_loop`, `_print_validation_report`, `_render_episode`; EEF-enabled backend and writer factory.
- Produces: executable EEF collector CLI with default and override EEF topics, EEF metadata, and the existing multi-episode lifecycle.

- [ ] **Step 1: Write failing collector CLI/controller tests**

Load `collect_data_eef_ros2` and assert its parser mirrors the current flags plus EEF topics:

```python
def test_eef_cli_uses_safe_defaults_and_official_topics(tmp_path):
    args = build_arg_parser().parse_args([
        "--output-dir", str(tmp_path),
        "--prompt", "Fold the towel",
        "--config", "examples/piper_dual/ros2_piper_dual.yaml",
    ])
    assert args.dry_run is True
    assert args.publish_actions is False
    assert args.eef_left_topic == "/puppet/end_pose_left"
    assert args.eef_right_topic == "/puppet/end_pose_right"
```

Use fake backend/writer classes to assert topic overrides reach `Ros2BackendClient`, max sync error is applied, the writer opens with `include_eef=True`, metadata contains collector/topic/order/unit fields, and the shared controller still aborts partial episodes and allocates distinct episode names.

- [ ] **Step 2: Run collector tests and observe failure**

Run:

```bash
.venv/bin/pytest \
  examples/piper_dual/tests/test_collect_data_eef_ros2.py \
  examples/piper_dual/tests/test_collector_state_machine.py -v
```

Expected: import failure for the missing EEF collector.

- [ ] **Step 3: Add narrow extension points to the shared controller**

Extend `CollectorController` with backward-compatible constructor arguments:

```python
collector_name: str = "collect_data_ros2.py"
metadata_extra: dict[str, Any] | None = None
writer_options: dict[str, Any] | None = None
```

At episode start, merge the fixed metadata with a defensive copy of `metadata_extra`, and pass `writer_options` to the writer factory. Existing tests continue to observe exactly the same ordinary metadata and writer call contract.

- [ ] **Step 4: Implement `collect_data_eef_ros2.py`**

Reuse the shared lifecycle instead of copying it. Define official topic defaults, construct the EEF-enabled backend, and construct the controller with:

```python
metadata_extra={
    "eef": {
        "enabled": True,
        "topics": {
            "puppet_left": args.eef_left_topic,
            "puppet_right": args.eef_right_topic,
        },
        "order": ["x", "y", "z", "roll", "pitch", "yaw"],
        "units": ["m", "m", "m", "rad", "rad", "rad"],
    }
},
writer_options={"include_eef": True},
collector_name="collect_data_eef_ros2.py",
```

Keep the current unsafe `--dry-run` plus `--publish-actions` rejection and the same preview/keyboard behavior.

- [ ] **Step 5: Run collector tests**

Run:

```bash
.venv/bin/pytest \
  examples/piper_dual/tests/test_collect_data_eef_ros2.py \
  examples/piper_dual/tests/test_collector_state_machine.py -v
```

Expected: all selected tests pass, including existing collector regression tests.

---

### Task 5: Operator documentation and end-to-end verification

**Files:**
- Modify: `examples/piper_dual/README.md`
- Modify if necessary after integration evidence: focused tests from Tasks 1–4 only.

**Interfaces:**
- Consumes: completed CLI and HDF5 contract.
- Produces: exact operator command, topic prerequisites, output paths/units, and verification evidence.

- [ ] **Step 1: Document the EEF collection command and contract**

Add a concise section after the ordinary ROS 2 collector:

```bash
python examples/piper_dual/collect_data_eef_ros2.py \
  --output-dir /home/agilex/piper_dual_dataset/fold_towel_eef \
  --prompt "Fold the towel" \
  --config examples/piper_dual/ros2_piper_dual.yaml \
  --bridge-python /usr/bin/python3 \
  --max-sync-error-ms 30 \
  --dry-run
```

Document default topics, override flags, `(N, 6)` vector order/units, strict synchronization, and that the current LeRobot converter ignores raw EEF datasets.

- [ ] **Step 2: Run all focused Piper dual tests affected by the change**

Run:

```bash
.venv/bin/pytest \
  examples/piper_dual/tests/test_frame_synchronizer.py \
  examples/piper_dual/tests/test_ros2_bridge_codec.py \
  examples/piper_dual/tests/test_ros2_backend.py \
  examples/piper_dual/tests/test_streaming_hdf5.py \
  examples/piper_dual/tests/test_collector_state_machine.py \
  examples/piper_dual/tests/test_collect_data_eef_ros2.py \
  examples/piper_dual/tests/test_ros2_end_to_end_mock.py -v
```

Expected: all selected tests pass.

- [ ] **Step 3: Run CLI smoke checks**

Run:

```bash
.venv/bin/python examples/piper_dual/collect_data_eef_ros2.py --help
```

Expected: exit 0; help includes `--eef-left-topic`, `--eef-right-topic`, `--max-sync-error-ms`, and safe dry-run flags.

Run a ROS-independent in-process smoke script through pytest that feeds nine synthetic sensor events into the backend handler, obtains one synchronized frame, writes/finalizes it with the EEF writer, and validates the result. Expected: one frame; EEF datasets `(1, 6)`; validation errors `[]`.

- [ ] **Step 4: Inspect language diagnostics**

Run Python diagnostics on:

```text
examples/piper_dual/frame_synchronizer.py
examples/piper_dual/ros2_bridge_process.py
examples/piper_dual/ros2_backend.py
examples/piper_dual/streaming_hdf5.py
examples/piper_dual/collect_data_ros2.py
examples/piper_dual/collect_data_eef_ros2.py
```

Expected: no new errors in changed files.

- [ ] **Step 5: Record hardware verification boundary**

If ROS 2 topics and physical hardware are unavailable, report that verification covered bridge codecs, synchronization, HDF5 behavior, shared state machine, mock end-to-end flow, and CLI parsing—not live camera/arm timing. If hardware is available, run the documented dry-run collector against the live topics and confirm one finalized episode validates without action publication.
