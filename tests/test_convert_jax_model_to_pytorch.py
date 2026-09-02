import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "examples" / "convert_jax_model_to_pytorch.py"
SPEC = importlib.util.spec_from_file_location("convert_jax_model_to_pytorch", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
_copy_checkpoint_assets = MODULE._copy_checkpoint_assets


def test_copy_checkpoint_assets_prefers_assets_inside_checkpoint(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_assets = checkpoint_dir / "assets" / "robot"
    checkpoint_assets.mkdir(parents=True)
    (checkpoint_assets / "norm_stats.json").write_text("checkpoint stats")

    legacy_assets = tmp_path / "assets" / "robot"
    legacy_assets.mkdir(parents=True)
    (legacy_assets / "norm_stats.json").write_text("legacy stats")

    output_path = tmp_path / "pytorch"
    _copy_checkpoint_assets(checkpoint_dir, output_path)

    assert (output_path / "assets" / "robot" / "norm_stats.json").read_text() == "checkpoint stats"
