"""Behavioral tests for LIBERO-Mem HDF5 to LeRobot conversion."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "convert_libero_mem_data_to_lerobot.py"
spec = importlib.util.spec_from_file_location("convert_libero_mem_data_to_lerobot", SCRIPT)
assert spec is not None and spec.loader is not None
converter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = converter
spec.loader.exec_module(converter)


class FakeDataset:
    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.episodes: list[list[dict]] = []
        self._episode_start = 0

    def add_frame(self, frame: dict) -> None:
        self.frames.append(frame)

    def save_episode(self) -> None:
        self.episodes.append(self.frames[self._episode_start :])
        self._episode_start = len(self.frames)


def _demo_arrays(frame_count: int = 3) -> dict[str, np.ndarray]:
    agent = np.arange(frame_count * 256 * 256 * 3, dtype=np.uint8).reshape(frame_count, 256, 256, 3)
    wrist = (agent + np.uint8(17)).astype(np.uint8)
    ee = (np.arange(frame_count * 6, dtype=np.float64).reshape(frame_count, 6) / 10.0) - 1.0
    gripper = (np.arange(frame_count * 2, dtype=np.float64).reshape(frame_count, 2) / 20.0) + 0.25
    actions = (np.arange(frame_count * 7, dtype=np.float64).reshape(frame_count, 7) / 100.0) - 0.5
    return {"agent": agent, "wrist": wrist, "ee": ee, "gripper": gripper, "actions": actions}


def _write_demo(group: h5py.Group, arrays: dict[str, np.ndarray]) -> None:
    group.create_dataset("actions", data=arrays["actions"])
    obs = group.create_group("obs")
    obs.create_dataset("ee_states", data=arrays["ee"])
    obs.create_dataset("gripper_states", data=arrays["gripper"])
    obs.create_dataset("agentview_rgb", data=arrays["agent"])
    obs.create_dataset("eye_in_hand_rgb", data=arrays["wrist"])


def _write_task_file(path: Path, *, include_failed_demo: bool = True) -> tuple[dict[str, np.ndarray], dict]:
    arrays = _demo_arrays()
    with h5py.File(path, "w") as hdf5_file:
        data = hdf5_file.create_group("data")
        _write_demo(data.create_group("demo_0"), arrays)
        if include_failed_demo:
            _write_demo(data.create_group("demo_1"), _demo_arrays(frame_count=2))
        _write_demo(data.create_group("demo_2"), _demo_arrays(frame_count=1))

    prompt = converter.TASK_SPECS[0].prompt
    metadata = {
        "demo_0": {"success": True, "task_description": prompt},
        "demo_1": {"success": False, "task_description": prompt},
        "demo_3": {"success": True, "task_description": prompt},
    }
    return arrays, metadata


def test_load_task_metadata_returns_only_exact_targets(tmp_path: Path) -> None:
    payload = {
        converter.TASK_SPECS[0].task_name: {"demo_0": {"success": True}},
        converter.TASK_SPECS[1].task_name: {"demo_0": {"success": True}},
        "unrelated_task": {"demo_0": {"success": True}},
    }
    path = tmp_path / "metainfo.json"
    path.write_text(json.dumps(payload))

    result = converter.load_task_metadata(path)

    assert set(result) == {spec.task_name for spec in converter.TASK_SPECS}
    assert "unrelated_task" not in result


def test_iter_valid_demo_ids_intersects_hdf5_metadata_and_success(tmp_path: Path) -> None:
    path = tmp_path / "task.hdf5"
    _, metadata = _write_task_file(path)

    with h5py.File(path, "r") as hdf5_file:
        ids = list(converter.iter_valid_demo_ids(hdf5_file, metadata))

    assert ids == ["0"]


def test_populate_dataset_preserves_pixels_state_actions_and_prompt(tmp_path: Path) -> None:
    task = converter.TASK_SPECS[0]
    path = tmp_path / task.filename
    arrays, task_metadata = _write_task_file(path)
    dataset = FakeDataset()

    summary = converter.populate_dataset(dataset, {task.task_name: path}, {task.task_name: task_metadata})

    assert summary.episodes == 1
    assert summary.frames == 3
    assert summary.episodes_by_task == {task.task_name: 1}
    assert summary.frames_by_task == {task.task_name: 3}
    assert len(dataset.episodes) == 1
    for index, frame in enumerate(dataset.episodes[0]):
        np.testing.assert_array_equal(frame["image"], arrays["agent"][index])
        np.testing.assert_array_equal(frame["wrist_image"], arrays["wrist"][index])
        np.testing.assert_array_equal(
            frame["state"], np.concatenate((arrays["ee"][index], arrays["gripper"][index])).astype(np.float32)
        )
        np.testing.assert_array_equal(frame["actions"], arrays["actions"][index].astype(np.float32))
        assert frame["state"].dtype == np.float32
        assert frame["actions"].dtype == np.float32
        assert frame["task"] == task.prompt


def test_populate_dataset_rejects_metadata_prompt_mismatch(tmp_path: Path) -> None:
    task = converter.TASK_SPECS[0]
    path = tmp_path / task.filename
    _, metadata = _write_task_file(path)
    metadata["demo_0"]["task_description"] = "wrong prompt"

    with pytest.raises(ValueError, match="task_description"):
        converter.populate_dataset(FakeDataset(), {task.task_name: path}, {task.task_name: metadata})


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("actions", np.zeros((3, 6), dtype=np.float64), "actions.*expected.*7"),
        ("ee", np.zeros((3, 5), dtype=np.float64), "ee_states.*expected.*6"),
        ("gripper", np.zeros((3, 1), dtype=np.float64), "gripper_states.*expected.*2"),
        ("agent", np.zeros((3, 255, 256, 3), dtype=np.uint8), "agentview_rgb.*expected"),
        ("wrist", np.zeros((2, 256, 256, 3), dtype=np.uint8), "same frame count"),
        ("ee", np.full((3, 6), np.nan, dtype=np.float64), "ee_states.*finite"),
        ("actions", np.full((3, 7), np.inf, dtype=np.float64), "actions.*finite"),
    ],
)
def test_read_demo_rejects_invalid_arrays(
    tmp_path: Path, field: str, replacement: np.ndarray, message: str
) -> None:
    task = converter.TASK_SPECS[0]
    path = tmp_path / task.filename
    arrays = _demo_arrays()
    arrays[field] = replacement
    with h5py.File(path, "w") as hdf5_file:
        data = hdf5_file.create_group("data")
        _write_demo(data.create_group("demo_0"), arrays)

    with h5py.File(path, "r") as hdf5_file, pytest.raises(ValueError, match=message):
        converter.read_demo(hdf5_file, "0", task_name=task.task_name)


def test_read_demo_rejects_zero_frames(tmp_path: Path) -> None:
    task = converter.TASK_SPECS[0]
    path = tmp_path / task.filename
    with h5py.File(path, "w") as hdf5_file:
        demo = hdf5_file.create_group("data/demo_0")
        demo.create_dataset("actions", shape=(0, 7), dtype=np.float64)
        obs = demo.create_group("obs")
        obs.create_dataset("ee_states", shape=(0, 6), dtype=np.float64)
        obs.create_dataset("gripper_states", shape=(0, 2), dtype=np.float64)
        obs.create_dataset("agentview_rgb", shape=(0, 256, 256, 3), dtype=np.uint8)
        obs.create_dataset("eye_in_hand_rgb", shape=(0, 256, 256, 3), dtype=np.uint8)

    with h5py.File(path, "r") as hdf5_file, pytest.raises(ValueError, match="positive frame count"):
        converter.read_demo(hdf5_file, "0", task_name=task.task_name)


def test_create_empty_dataset_declares_libero_features(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict = {}

    class FakeLeRobotDataset:
        @classmethod
        def create(cls, **kwargs):
            captured.update(kwargs)
            return cls()

    monkeypatch.setattr(converter, "LeRobotDataset", FakeLeRobotDataset)

    converter.create_empty_dataset("local/libero_mem", tmp_path / "output")

    assert captured["repo_id"] == "local/libero_mem"
    assert captured["root"] == tmp_path / "output"
    assert captured["fps"] == 10
    assert captured["robot_type"] == "panda"
    assert captured["features"]["image"]["shape"] == (256, 256, 3)
    assert captured["features"]["wrist_image"]["shape"] == (256, 256, 3)
    assert captured["features"]["state"]["shape"] == (8,)
    assert captured["features"]["actions"]["shape"] == (7,)


def test_convert_dataset_preserves_completed_output_when_hub_upload_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    metadata: dict[str, dict] = {}
    for task in converter.TASK_SPECS:
        path = raw_dir / task.filename
        _, task_metadata = _write_task_file(path, include_failed_demo=False)
        metadata[task.task_name] = task_metadata
        task_metadata["demo_0"]["task_description"] = task.prompt
    (raw_dir / converter.METADATA_FILENAME).write_text(json.dumps(metadata))
    output = tmp_path / "output"

    class UploadFailingDataset(FakeDataset):
        def push_to_hub(self, **kwargs) -> None:
            raise RuntimeError("hub unavailable")

    def create_dataset(repo_id: str, output_dir: Path) -> UploadFailingDataset:
        output_dir.mkdir(parents=True)
        (output_dir / "conversion-complete").write_text(repo_id)
        return UploadFailingDataset()

    monkeypatch.setattr(converter, "create_empty_dataset", create_dataset)

    with pytest.raises(RuntimeError, match="hub unavailable"):
        converter.convert_dataset(raw_dir, "local/libero_mem", output_dir=output, push_to_hub=True)

    assert (output / "conversion-complete").read_text() == "local/libero_mem"


def test_parse_args_supports_download_upload_and_overwrite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            SCRIPT.name,
            "--raw-dir",
            "/tmp/raw",
            "--repo-id",
            "local/libero_mem",
            "--output-dir",
            "/tmp/output",
            "--no-download",
            "--overwrite",
            "--push-to-hub",
        ],
    )

    args = converter._parse_args()

    assert args.raw_dir == Path("/tmp/raw")
    assert args.repo_id == "local/libero_mem"
    assert args.output_dir == Path("/tmp/output")
    assert args.download is False
    assert args.overwrite is True
    assert args.push_to_hub is True
