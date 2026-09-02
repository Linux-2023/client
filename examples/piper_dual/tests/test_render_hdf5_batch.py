"""Behavior tests for the batch HDF5 rendering shell wrapper."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "render_hdf5_batch.sh"


def _write_fake_python(path: Path) -> Path:
    path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s %s\n' "$0" "$*" >> "$BATCH_TEST_LOG"
script="$(basename "$1")"
shift
case "$script" in
    capture_camera_info.py)
        while (( $# )); do
            if [[ "$1" == --output ]]; then
                printf '{"schema_version":"piper_dual_camera_parameters_v1"}\\n' > "$2"
                exit 0
            fi
            shift
        done
        ;;
    export_episode_json.py)
        if [[ "${BATCH_FAIL_EXPORT:-}" == 1 ]]; then
            exit 7
        fi
        while (( $# )); do
            if [[ "$1" == --puppet-output ]]; then
                printf '{"schema_version":"piper_dual_puppet_state_v1"}\\n' > "$2"
                exit 0
            fi
            shift
        done
        ;;
    render_dataset.py)
        output_dir=""
        make_plots=true
        while (( $# )); do
            case "$1" in
                --output-dir)
                    output_dir="$2"
                    shift 2
                    ;;
                --no-plots)
                    make_plots=false
                    shift
                    ;;
                *)
                    shift
                    ;;
            esac
        done
        mkdir -p "$output_dir"
        for name in cam_high.mp4 cam_left_wrist.mp4 cam_right_wrist.mp4 views_3x1.mp4 quality.json; do
            printf 'rendered\\n' > "$output_dir/$name"
        done
        if [[ "$make_plots" == true ]]; then
            printf 'plot\\n' > "$output_dir/state_action.png"
        fi
        ;;
    *)
        echo "unexpected Python script: $script" >&2
        exit 1
        ;;
esac
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _run(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    assert SCRIPT.is_file(), f"batch renderer does not exist: {SCRIPT}"
    fake_python = _write_fake_python(tmp_path / "fake-python")
    fake_ros2_python = _write_fake_python(tmp_path / "fake-ros2-python")
    log_path = tmp_path / "render.log"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON": str(fake_python),
            "ROS2_PYTHON": str(fake_ros2_python),
            "BATCH_TEST_LOG": str(log_path),
        }
    )
    return subprocess.run(
        [str(SCRIPT), *arguments],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )


def test_batch_renderer_converts_sorted_episodes_then_skips_complete_outputs_unless_forced(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    for name in ("episode_000001.hdf5", "episode_000000.hdf5"):
        (input_dir / name).touch()

    first = _run(
        tmp_path,
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--fps",
        "12",
    )

    assert first.returncode == 0, first.stderr
    log_path = tmp_path / "render.log"
    first_calls = log_path.read_text(encoding="utf-8").splitlines()
    assert len(first_calls) == 5
    assert "capture_camera_info.py" in first_calls[0]
    assert str(tmp_path / "fake-ros2-python") in first_calls[0]
    render_calls = [call for call in first_calls if "render_dataset.py" in call]
    export_calls = [call for call in first_calls if "export_episode_json.py" in call]
    assert len(render_calls) == 2
    assert len(export_calls) == 2
    assert "episode_000000.hdf5" in render_calls[0]
    assert "episode_000001.hdf5" in render_calls[1]
    assert all("--fps 12 --no-plots" in call for call in render_calls)
    assert all("--puppet-output" in call for call in export_calls)
    expected_files = {
        "cam_high.mp4",
        "cam_left_wrist.mp4",
        "cam_right_wrist.mp4",
        "views_3x1.mp4",
        "quality.json",
        "puppet_state.json",
        "camera_parameters.json",
    }
    for episode in ("episode_000000", "episode_000001"):
        assert expected_files <= {path.name for path in (output_dir / episode).iterdir() if path.stat().st_size > 0}

    second = _run(
        tmp_path,
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
    )

    assert second.returncode == 0, second.stderr
    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 5
    assert "跳过 2" in second.stdout

    forced = _run(
        tmp_path,
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--force",
    )

    assert forced.returncode == 0, forced.stderr
    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 10


def test_batch_renderer_adds_missing_json_without_rerendering_complete_videos(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    episode_dir = output_dir / "episode_000000"
    input_dir.mkdir()
    episode_dir.mkdir(parents=True)
    (input_dir / "episode_000000.hdf5").touch()
    for name in ("cam_high.mp4", "cam_left_wrist.mp4", "cam_right_wrist.mp4", "views_3x1.mp4", "quality.json"):
        (episode_dir / name).write_text("existing\n", encoding="utf-8")

    completed = _run(
        tmp_path,
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
    )

    assert completed.returncode == 0, completed.stderr
    calls = (tmp_path / "render.log").read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert "capture_camera_info.py" in calls[0]
    assert "export_episode_json.py" in calls[1]
    assert all("render_dataset.py" not in call for call in calls)
    assert (episode_dir / "puppet_state.json").is_file()
    assert (episode_dir / "camera_parameters.json").is_file()


def test_batch_renderer_preserves_existing_camera_json_when_copy_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    episode_dir = output_dir / "episode_000000"
    bin_dir = tmp_path / "bin"
    input_dir.mkdir()
    episode_dir.mkdir(parents=True)
    bin_dir.mkdir()
    (input_dir / "episode_000000.hdf5").touch()
    for name in ("cam_high.mp4", "cam_left_wrist.mp4", "cam_right_wrist.mp4", "views_3x1.mp4", "quality.json"):
        (episode_dir / name).write_text("existing\n", encoding="utf-8")
    camera_path = episode_dir / "camera_parameters.json"
    camera_path.write_text('{"old":true}\n', encoding="utf-8")
    fake_cp = bin_dir / "cp"
    fake_cp.write_text(
        "#!/usr/bin/env bash\nprintf 'partial' > \"${@: -1}\"\nexit 9\n",
        encoding="utf-8",
    )
    fake_cp.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")

    completed = _run(
        tmp_path,
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
    )

    assert completed.returncode != 0
    assert camera_path.read_text(encoding="utf-8") == '{"old":true}\n'
    assert not list(episode_dir.glob(".camera_parameters.json.*"))


def test_batch_renderer_removes_camera_snapshot_when_episode_export_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "episode_000000.hdf5").touch()
    monkeypatch.setenv("BATCH_FAIL_EXPORT", "1")

    completed = _run(
        tmp_path,
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
    )

    assert completed.returncode != 0
    assert not (output_dir / ".camera_parameters.json").exists()


def test_batch_renderer_rejects_missing_input_directory(tmp_path: Path) -> None:
    completed = _run(
        tmp_path,
        "--input-dir",
        str(tmp_path / "missing"),
        "--output-dir",
        str(tmp_path / "output"),
    )

    assert completed.returncode != 0
    assert "输入目录不存在" in completed.stderr


def test_batch_renderer_plots_flag_allows_state_action_plot(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "episode_000000.hdf5").touch()

    completed = _run(
        tmp_path,
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(tmp_path / "output"),
        "--plots",
    )

    assert completed.returncode == 0, completed.stderr
    call = (tmp_path / "render.log").read_text(encoding="utf-8")
    assert "--no-plots" not in call
    assert (tmp_path / "output/episode_000000/state_action.png").is_file()
