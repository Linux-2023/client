"""Image selection must happen before HF's embedded-image decoder runs."""

from types import SimpleNamespace

import datasets
import numpy as np
from PIL import Image
from PIL import UnidentifiedImageError
import pytest

from openpi.models.pi0_config import Pi0Config
from openpi.training import config
from openpi.training import data_loader


@pytest.fixture
def embedded_dataset(monkeypatch):
    features = datasets.Features(
        {
            "observation.images.top": datasets.Image(),
            "observation.images.left_wrist": datasets.Image(),
            "observation.images.right_wrist": datasets.Image(),
            "action": datasets.Sequence(datasets.Value("float32"), length=14),
            "episode_index": datasets.Value("int64"),
        }
    )
    hf_dataset = datasets.Dataset.from_dict(
        {
            "observation.images.top": [Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))],
            "observation.images.left_wrist": [{"bytes": b"corrupt image must not be decoded", "path": None}],
            "observation.images.right_wrist": [Image.fromarray(np.ones((4, 4, 3), dtype=np.uint8))],
            "action": [np.arange(14, dtype=np.float32)],
            "episode_index": [0],
        },
        features=features,
    )
    # A real HF transform must continue to run after columns are removed.
    hf_dataset.set_transform(lambda batch: {**batch, "transformed": [True] * len(batch["action"])})
    meta = SimpleNamespace(
        fps=30,
        tasks={0: "Weigh the mango."},
        video_keys=[],
        camera_keys=[k for k in features if k.startswith("observation.images.")],
        image_keys=[k for k in features if k.startswith("observation.images.")],
        features={k: {"dtype": "image"} for k in features if k.startswith("observation.images.")},
    )

    class LocalDataset:
        def __init__(self):
            self.hf_dataset = hf_dataset
            self.meta = meta

        def __getitem__(self, index):
            return self.hf_dataset[index]

        def __len__(self):
            return len(self.hf_dataset)

    local = LocalDataset()
    monkeypatch.setattr(data_loader.lerobot_dataset, "LeRobotDatasetMetadata", lambda *args, **kwargs: meta)
    monkeypatch.setattr(data_loader.lerobot_dataset, "LeRobotDataset", lambda *args, **kwargs: local)
    return local


def test_image_selection_skips_corrupt_left_image_before_decode(embedded_dataset):
    cfg = config.DataConfig(
        repo_id="local/image-selection",
        image_keys=(
            "observation.images.top",
            "observation.images.right_wrist",
        ),
    )
    selected = data_loader.create_torch_dataset(cfg, 50, Pi0Config(pi05=True))
    row = selected[0]
    assert "observation.images.left_wrist" not in row
    assert row["observation.images.top"].size == (4, 4)
    assert row["observation.images.right_wrist"].size == (4, 4)
    assert row["transformed"] is True
    np.testing.assert_array_equal(row["action"], np.arange(14))
    assert row["episode_index"] == 0


def test_default_image_selection_preserves_existing_decode_behavior(embedded_dataset):
    cfg = config.DataConfig(repo_id="local/image-selection")
    selected = data_loader.create_torch_dataset(cfg, 50, Pi0Config(pi05=True))
    assert "observation.images.left_wrist" in selected.hf_dataset.column_names
    with pytest.raises(UnidentifiedImageError, match="cannot identify image"):
        selected[0]


def test_unknown_image_selection_is_rejected(embedded_dataset):
    cfg = config.DataConfig(repo_id="local/image-selection", image_keys=("observation.images.typo",))
    with pytest.raises(ValueError, match="image"):
        data_loader.create_torch_dataset(cfg, 50, Pi0Config(pi05=True))


def test_video_selection_skips_left_camera_before_decode(monkeypatch):
    import torch

    dataset_type = data_loader.lerobot_dataset.LeRobotDataset
    metadata_type = data_loader.lerobot_dataset.LeRobotDatasetMetadata
    meta = metadata_type.__new__(metadata_type)
    cameras = [f"observation.images.{name}" for name in ("top", "left_wrist", "right_wrist")]
    original_info = {"features": {key: {"dtype": "video"} for key in cameras}, "fps": 30}
    meta.info = original_info
    meta.tasks = {0: "yx button"}
    queried = []

    class VideoDataset(dataset_type):
        def __init__(self):
            self.meta = meta
            self.delta_indices = None
            self.image_transforms = None
            self.hf_dataset = [{"episode_index": torch.tensor(0), "timestamp": torch.tensor(0.0),
                                "task_index": torch.tensor(0), "action": torch.arange(14)}]

        def _query_videos(self, timestamps, ep_idx):
            assert cameras[1] not in timestamps, "Excluded left video reached decoder"
            queried.extend(timestamps)
            return {key: torch.zeros(3, 4, 4) for key in timestamps}

    local = VideoDataset()
    monkeypatch.setattr(data_loader.lerobot_dataset, "LeRobotDatasetMetadata", lambda *a, **kw: meta)
    monkeypatch.setattr(data_loader.lerobot_dataset, "LeRobotDataset", lambda *a, **kw: local)
    cfg = config.DataConfig(repo_id="local/video-selection", image_keys=(cameras[0], cameras[2]))
    row = data_loader.create_torch_dataset(cfg, 50, Pi0Config(pi05=True))[0]
    assert queried == [cameras[0], cameras[2]]
    assert cameras[1] not in row
    assert row["task"] == "yx button"
    np.testing.assert_array_equal(row["action"], np.arange(14))
    assert cameras[1] in original_info["features"]
