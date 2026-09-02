#!/usr/bin/env python3
"""Convert two exact LIBERO-Mem bowl tasks from HDF5 to one LeRobot dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Iterator

if os.getenv("HF_LEROBOT_HOME") is None:
    os.environ["HF_LEROBOT_HOME"] = str(Path.home() / ".cache" / "huggingface" / "lerobot")

import h5py
from huggingface_hub import hf_hub_download
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

SOURCE_REPO_ID = "libero-mem/LIBERO-Mem"
SOURCE_REVISION = "6df130c05b82e09294fa9534e84a069687b72d33"
DEFAULT_REPO_ID = "HITdongdong/libero_mem_bowl_two_tasks"
METADATA_FILENAME = "metainfo.json"
FPS = 10
IMAGE_SHAPE = (256, 256, 3)
STATE_DIMENSION = 8
ACTION_DIMENSION = 7
LEROBOT_HOME = Path(os.environ["HF_LEROBOT_HOME"])


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_name: str
    filename: str
    prompt: str


TASK_SPECS = (
    TaskSpec(
        task_name="KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate",
        filename="KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate_demo.hdf5",
        prompt="pick up the bowl and place it back on the plate",
    ),
    TaskSpec(
        task_name="KITCHEN_SCENE1_3_lift_the_bowl_and_place_it_back_on_the_plate_3_times",
        filename="KITCHEN_SCENE1_3_lift_the_bowl_and_place_it_back_on_the_plate_3_times_demo.hdf5",
        prompt="lift the bowl and place it back on the plate 3 times",
    ),
)


@dataclass(frozen=True, slots=True)
class DemoData:
    images: np.ndarray
    wrist_images: np.ndarray
    states: np.ndarray
    actions: np.ndarray


@dataclass(frozen=True, slots=True)
class ConversionSummary:
    episodes: int
    frames: int
    episodes_by_task: dict[str, int]
    frames_by_task: dict[str, int]


def load_task_metadata(path: Path) -> dict[str, dict]:
    """Load only the two required task entries from LIBERO-Mem metadata."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: could not load valid JSON metadata") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level metadata must be an object")

    selected: dict[str, dict] = {}
    for task in TASK_SPECS:
        task_metadata = payload.get(task.task_name)
        if not isinstance(task_metadata, dict):
            raise ValueError(f"{path}: missing metadata object for task {task.task_name}")
        selected[task.task_name] = task_metadata
    return selected


def _demo_sort_key(demo_id: str) -> tuple[int, str]:
    try:
        return int(demo_id), demo_id
    except ValueError as exc:
        raise ValueError(f"invalid LIBERO-Mem demo identifier: demo_{demo_id}") from exc


def iter_valid_demo_ids(hdf5_file: h5py.File, task_metadata: dict) -> Iterator[str]:
    """Yield numeric demo IDs present in both sources and marked successful."""
    data = hdf5_file.get("data")
    if not isinstance(data, h5py.Group):
        raise ValueError(f"{hdf5_file.filename}: missing required group data")
    hdf5_ids = {name.removeprefix("demo_") for name in data if name.startswith("demo_")}
    metadata_ids = {name.removeprefix("demo_") for name in task_metadata if name.startswith("demo_")}
    for demo_id in sorted(hdf5_ids & metadata_ids, key=_demo_sort_key):
        entry = task_metadata[f"demo_{demo_id}"]
        if not isinstance(entry, dict):
            raise ValueError(f"{hdf5_file.filename}: metadata demo_{demo_id} must be an object")
        if entry.get("success") is True:
            yield demo_id


def _require_dataset(group: h5py.Group, path: str, *, task_name: str, demo_id: str) -> h5py.Dataset:
    value = group.get(path)
    if not isinstance(value, h5py.Dataset):
        raise ValueError(f"{task_name} demo_{demo_id}: missing required dataset {path}")
    return value


def _require_shape(
    dataset: h5py.Dataset,
    trailing_shape: tuple[int, ...],
    *,
    field: str,
    task_name: str,
    demo_id: str,
) -> int:
    expected_rank = len(trailing_shape) + 1
    if dataset.ndim != expected_rank or tuple(dataset.shape[1:]) != trailing_shape:
        raise ValueError(
            f"{task_name} demo_{demo_id}: {field} expected shape (T, {', '.join(map(str, trailing_shape))}), "
            f"found {dataset.shape}"
        )
    return int(dataset.shape[0])


def read_demo(hdf5_file: h5py.File, demo_id: str, *, task_name: str) -> DemoData:
    """Read and validate one demo while preserving current-frame semantics."""
    group = hdf5_file.get(f"data/demo_{demo_id}")
    if not isinstance(group, h5py.Group):
        raise ValueError(f"{task_name} demo_{demo_id}: missing HDF5 demo group")

    actions_ds = _require_dataset(group, "actions", task_name=task_name, demo_id=demo_id)
    ee_ds = _require_dataset(group, "obs/ee_states", task_name=task_name, demo_id=demo_id)
    gripper_ds = _require_dataset(group, "obs/gripper_states", task_name=task_name, demo_id=demo_id)
    images_ds = _require_dataset(group, "obs/agentview_rgb", task_name=task_name, demo_id=demo_id)
    wrist_ds = _require_dataset(group, "obs/eye_in_hand_rgb", task_name=task_name, demo_id=demo_id)

    counts = {
        "actions": _require_shape(
            actions_ds, (ACTION_DIMENSION,), field="actions", task_name=task_name, demo_id=demo_id
        ),
        "ee_states": _require_shape(ee_ds, (6,), field="ee_states", task_name=task_name, demo_id=demo_id),
        "gripper_states": _require_shape(
            gripper_ds, (2,), field="gripper_states", task_name=task_name, demo_id=demo_id
        ),
        "agentview_rgb": _require_shape(
            images_ds, IMAGE_SHAPE, field="agentview_rgb", task_name=task_name, demo_id=demo_id
        ),
        "eye_in_hand_rgb": _require_shape(
            wrist_ds, IMAGE_SHAPE, field="eye_in_hand_rgb", task_name=task_name, demo_id=demo_id
        ),
    }
    if len(set(counts.values())) != 1:
        raise ValueError(f"{task_name} demo_{demo_id}: all datasets must have the same frame count, found {counts}")
    frame_count = next(iter(counts.values()))
    if frame_count <= 0:
        raise ValueError(f"{task_name} demo_{demo_id}: expected a positive frame count")
    if images_ds.dtype != np.dtype("uint8"):
        raise ValueError(f"{task_name} demo_{demo_id}: agentview_rgb expected dtype uint8, found {images_ds.dtype}")
    if wrist_ds.dtype != np.dtype("uint8"):
        raise ValueError(f"{task_name} demo_{demo_id}: eye_in_hand_rgb expected dtype uint8, found {wrist_ds.dtype}")

    ee = np.asarray(ee_ds[:], dtype=np.float32)
    gripper = np.asarray(gripper_ds[:], dtype=np.float32)
    actions = np.asarray(actions_ds[:], dtype=np.float32)
    for field, values in (("ee_states", ee), ("gripper_states", gripper), ("actions", actions)):
        if not np.isfinite(values).all():
            raise ValueError(f"{task_name} demo_{demo_id}: {field} must contain only finite values")

    states = np.concatenate((ee, gripper), axis=1, dtype=np.float32)
    return DemoData(
        images=np.asarray(images_ds[:], dtype=np.uint8),
        wrist_images=np.asarray(wrist_ds[:], dtype=np.uint8),
        states=states,
        actions=actions,
    )


def create_empty_dataset(repo_id: str, output_dir: Path) -> LeRobotDataset:
    features = {
        "image": {"dtype": "image", "shape": IMAGE_SHAPE, "names": ["height", "width", "channel"]},
        "wrist_image": {
            "dtype": "image",
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channel"],
        },
        "state": {"dtype": "float32", "shape": (STATE_DIMENSION,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (ACTION_DIMENSION,), "names": ["actions"]},
    }
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=Path(output_dir),
        fps=FPS,
        robot_type="panda",
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )


def populate_dataset(
    dataset: LeRobotDataset,
    source_paths: dict[str, Path],
    metadata: dict[str, dict],
) -> ConversionSummary:
    episodes_by_task: dict[str, int] = {}
    frames_by_task: dict[str, int] = {}

    for task in TASK_SPECS:
        path = source_paths.get(task.task_name)
        if path is None:
            continue
        task_metadata = metadata.get(task.task_name)
        if not isinstance(task_metadata, dict):
            raise ValueError(f"missing metadata for task {task.task_name}")
        task_episodes = 0
        task_frames = 0
        with h5py.File(path, "r") as hdf5_file:
            for demo_id in iter_valid_demo_ids(hdf5_file, task_metadata):
                entry = task_metadata[f"demo_{demo_id}"]
                prompt = entry.get("task_description")
                if prompt != task.prompt:
                    raise ValueError(
                        f"{task.task_name} demo_{demo_id}: task_description expected {task.prompt!r}, found {prompt!r}"
                    )
                demo = read_demo(hdf5_file, demo_id, task_name=task.task_name)
                for frame_index in range(demo.states.shape[0]):
                    dataset.add_frame(
                        {
                            "image": demo.images[frame_index],
                            "wrist_image": demo.wrist_images[frame_index],
                            "state": demo.states[frame_index],
                            "actions": demo.actions[frame_index],
                            "task": prompt,
                        }
                    )
                dataset.save_episode()
                task_episodes += 1
                task_frames += int(demo.states.shape[0])
        episodes_by_task[task.task_name] = task_episodes
        frames_by_task[task.task_name] = task_frames

    return ConversionSummary(
        episodes=sum(episodes_by_task.values()),
        frames=sum(frames_by_task.values()),
        episodes_by_task=episodes_by_task,
        frames_by_task=frames_by_task,
    )


def download_sources(raw_dir: Path, *, revision: str = SOURCE_REVISION) -> dict[str, Path]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    filenames = [METADATA_FILENAME, *(task.filename for task in TASK_SPECS)]
    downloaded = {
        filename: Path(
            hf_hub_download(
                repo_id=SOURCE_REPO_ID,
                repo_type="dataset",
                filename=filename,
                revision=revision,
                local_dir=raw_dir,
            )
        )
        for filename in filenames
    }
    return downloaded


def _source_paths(raw_dir: Path) -> dict[str, Path]:
    raw_dir = Path(raw_dir)
    result = {task.task_name: raw_dir / task.filename for task in TASK_SPECS}
    missing = [str(path) for path in [raw_dir / METADATA_FILENAME, *result.values()] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing LIBERO-Mem source files: {missing}")
    return result


def _ensure_output_available(output_dir: Path, *, overwrite: bool) -> None:
    if not output_dir.exists():
        return
    if not overwrite:
        raise FileExistsError(f"LeRobot output already exists: {output_dir}; pass --overwrite to replace it")
    if output_dir.is_dir():
        shutil.rmtree(output_dir)
    else:
        output_dir.unlink()


def convert_dataset(
    raw_dir: Path,
    repo_id: str = DEFAULT_REPO_ID,
    *,
    output_dir: Path | None = None,
    overwrite: bool = False,
    push_to_hub: bool = False,
) -> tuple[Path, ConversionSummary]:
    raw_dir = Path(raw_dir)
    source_paths = _source_paths(raw_dir)
    metadata = load_task_metadata(raw_dir / METADATA_FILENAME)
    output = Path(output_dir) if output_dir is not None else LEROBOT_HOME / repo_id
    _ensure_output_available(output, overwrite=overwrite)

    created = False
    try:
        dataset = create_empty_dataset(repo_id, output)
        created = True
        summary = populate_dataset(dataset, source_paths, metadata)
        if any(value == 0 for value in summary.episodes_by_task.values()):
            raise ValueError(f"each target task must contribute at least one successful episode: {summary}")
    except Exception:
        if created:
            shutil.rmtree(output, ignore_errors=True)
        raise
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "libero-mem", "panda"],
            private=False,
            push_videos=True,
            license="mit",
        )
    return output, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory for the three pinned source files")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Output LeRobot/Hugging Face repository ID")
    parser.add_argument("--output-dir", type=Path, help="Local output root; defaults to HF_LEROBOT_HOME/repo-id")
    parser.add_argument("--revision", default=SOURCE_REVISION, help="Pinned LIBERO-Mem source revision")
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download/resume exactly the required source files before conversion",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing local output")
    parser.add_argument("--push-to-hub", action="store_true", help="Publish the verified output as a public dataset")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.download:
        download_sources(args.raw_dir, revision=args.revision)
    output, summary = convert_dataset(
        args.raw_dir,
        args.repo_id,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        push_to_hub=args.push_to_hub,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "episodes": summary.episodes,
                "frames": summary.frames,
                "episodes_by_task": summary.episodes_by_task,
                "frames_by_task": summary.frames_by_task,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
