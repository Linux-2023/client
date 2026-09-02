#!/usr/bin/env python3
"""Upload the local shifted EEF Rot6D LeRobot dataset to Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError


DEFAULT_DATASET_DIR = Path(
    "/home/agilex/lgd/lerobot_datasets/local/"
    "piper_dual_stack_cups_eef_rot6d_shifted"
)
DEFAULT_REPO_ID = "HITdongdong/piper_dual_stack_cups_eef_rot6d_shifted"
DEFAULT_ENDPOINT = "https://hf-mirror.com"
DEFAULT_TOKEN_PATH = Path.home() / ".cache/huggingface/token"


def upload_dataset(
    *,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    repo_id: str = DEFAULT_REPO_ID,
    endpoint: str = DEFAULT_ENDPOINT,
    token_path: Path = DEFAULT_TOKEN_PATH,
    api_factory: Callable[..., HfApi] = HfApi,
):
    dataset_dir = dataset_dir.expanduser().resolve()
    token_path = token_path.expanduser()

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    if not token_path.is_file():
        raise FileNotFoundError(
            f"Hugging Face token file does not exist: {token_path}. "
            "Run `hf auth login` first."
        )

    token = token_path.read_text().strip()
    if not token:
        raise ValueError(f"Hugging Face token file is empty: {token_path}")

    api = api_factory(endpoint=endpoint, token=token)
    try:
        account = api.whoami()["name"]
        print(f"Endpoint: {endpoint}")
        print(f"HF account: {account}")
        print(f"Source: {dataset_dir}")
        print(f"Dataset repo: {repo_id}")

        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=False,
            exist_ok=True,
        )
        result = api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(dataset_dir),
            path_in_repo="",
            ignore_patterns=[".cache/**"],
            commit_message="Upload shifted Piper dual-arm EEF Rot6D dataset",
        )
    except HfHubHTTPError as exc:
        if exc.response is not None and exc.response.status_code == 401:
            raise RuntimeError(
                f"Authentication failed at {endpoint}. "
                "hf-mirror.com may reject authenticated uploads even when the "
                "same token works at huggingface.co."
            ) from exc
        raise

    print(result)
    print(f"Dataset URL: https://huggingface.co/datasets/{repo_id}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--token-path", type=Path, default=DEFAULT_TOKEN_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    upload_dataset(
        dataset_dir=args.dataset_dir,
        repo_id=args.repo_id,
        endpoint=args.endpoint,
        token_path=args.token_path,
    )


if __name__ == "__main__":
    main()
