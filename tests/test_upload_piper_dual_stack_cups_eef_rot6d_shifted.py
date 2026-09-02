from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "upload_piper_dual_stack_cups_eef_rot6d_shifted.py"


def load_script():
    spec = importlib.util.spec_from_file_location("upload_eef_rot6d_shifted", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_upload_uses_mirror_and_public_dataset_repo(tmp_path):
    module = load_script()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    token_path = tmp_path / "token"
    token_path.write_text("hf_test_token\n")
    calls = {}

    class FakeApi:
        def __init__(self, *, endpoint, token):
            calls["init"] = {"endpoint": endpoint, "token": token}

        def whoami(self):
            return {"name": "HITdongdong"}

        def create_repo(self, **kwargs):
            calls["create_repo"] = kwargs

        def upload_folder(self, **kwargs):
            calls["upload_folder"] = kwargs
            return "uploaded"

    result = module.upload_dataset(
        dataset_dir=dataset_dir,
        token_path=token_path,
        api_factory=FakeApi,
    )

    assert calls["init"] == {
        "endpoint": "https://hf-mirror.com",
        "token": "hf_test_token",
    }
    assert calls["create_repo"] == {
        "repo_id": "HITdongdong/piper_dual_stack_cups_eef_rot6d_shifted",
        "repo_type": "dataset",
        "private": False,
        "exist_ok": True,
    }
    assert calls["upload_folder"]["folder_path"] == str(dataset_dir)
    assert calls["upload_folder"]["ignore_patterns"] == [".cache/**"]
    assert result == "uploaded"


def test_upload_rejects_missing_dataset_directory(tmp_path):
    module = load_script()
    token_path = tmp_path / "token"
    token_path.write_text("hf_test_token")

    with pytest.raises(FileNotFoundError, match="Dataset directory does not exist"):
        module.upload_dataset(
            dataset_dir=tmp_path / "missing",
            token_path=token_path,
        )
