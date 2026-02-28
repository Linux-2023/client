"""
Script to convert Piper dual-arm hdf5 data to the LeRobot dataset v2.0 format.

Example usage: python examples/piper_dual/utils/convert_piper_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <dataset-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro
import os

if os.getenv("HF_LEROBOT_HOME") is None:
    os.environ["HF_LEROBOT_HOME"] = str(Path.home() / "lgd" / "huggingface" / "lerobot")
LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME"))


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    has_advantage: bool = False,
    cameras: list[str] | None = None,
    num_motors: int = 14,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    all_motors = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
    ]
    motors = all_motors[:num_motors]
    
    if cameras is None:
        cameras = [
            "cam_high",
            "cam_left_wrist",
            "cam_right_wrist",
        ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }
    
    if has_advantage:
        features["advantage"] = {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 224, 224),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep

def has_advantage(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/advantage" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            print(f"Loading compressed images for camera: {camera}")
        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])
        task_prompt = None
        key ="/task"
        num_steps = state.shape[0]
        raw_task = ep[key][:]
        if isinstance(raw_task, str):
            task_bytes = raw_task.encode("utf-8")
        elif isinstance(raw_task, bytes):
            task_bytes = raw_task
        else:
            raw_task ="Insert the straw into the cup"
            task_bytes = raw_task.encode("utf-8")
        task_prompt = np.array([task_bytes] * num_steps)

        if task_prompt is None:
            print("Task prompt not found in HDF5")
            # default_text = "Pick up anything on the table and put it in the basket.".encode("utf-8")
            # task_prompt = np.array([default_text] * num_steps)
        advantage = None
        if "/advantage" in ep:
            advantage = ep["/advantage"][:]

        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        available_cameras = list(ep["/observations/images"].keys())
        available_cameras = [c for c in available_cameras if "depth" not in c]
        
        imgs_per_cam = load_raw_images_per_camera(ep, available_cameras)

    return imgs_per_cam, state, action, velocity, effort, task_prompt, advantage


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        imgs_per_cam, state, action, velocity, effort, task_prompt, advantage = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": str(task_prompt[i],'utf-8'),
            }

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]
            
            if advantage is not None:
                frame["advantage"] = str(advantage[i],'utf-8')

            dataset.add_frame(frame)

        dataset.save_episode()

    return dataset


def port_piper_dual(
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    raw_dir: Path = Path("recorded_data_dual"),
    repo_id: str = "piper_dual_lerobot_data/",
):
    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        else:
            raise ValueError("Downloading raw data is not supported in this script.")
        #download_raw(raw_dir, repo_id=raw_repo_id)

    # 支持两种文件名格式：episode_*.hdf5 和 dual_episode_*.hdf5
    hdf5_files = sorted(list(raw_dir.glob("episode_*.hdf5")) + list(raw_dir.glob("dual_episode_*.hdf5")))
    
    if not hdf5_files:
        raise ValueError(f"No HDF5 files found in {raw_dir}")
    
    print(f"Found {len(hdf5_files)} HDF5 files")
    
    # 从第一个HDF5文件中动态获取相机列表和电机数量
    cameras = get_cameras(hdf5_files)
    print(f"Detected cameras: {cameras}")
    
    with h5py.File(hdf5_files[0], "r") as ep:
        num_motors = ep["/observations/qpos"].shape[1]
        print(f"Detected {num_motors} motors (state dimensions)")

    dataset = create_empty_dataset(
        repo_id,
        robot_type="piper_dual",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        has_advantage=has_advantage(hdf5_files),
        cameras=cameras,
        num_motors=num_motors,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
    )
    #dataset.consolidate()

    # if push_to_hub:
    #     dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_piper_dual)
