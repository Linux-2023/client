from pathlib import Path
import runpy
import sys


def test_original_command_is_forwarded_unchanged(monkeypatch):
    entrypoint = Path(__file__).resolve().parents[1] / "main_dual_ros.py"
    argv = [str(entrypoint), "--host", "127.0.0.1", "--port", "8000",
            "--bridge-python", "/usr/bin/python3",
            "--ros2-config", "examples/piper_dual/ros2_piper_dual_local.yaml",
            "--control-stack", "local-ros",
            "--prompt", "Pass the playing cards and sunflower seeds to the guests.",
            "--max-action-delta", "20", "--no-dry-run", "--publish-actions"]
    monkeypatch.setattr(sys, "argv", argv.copy())
    monkeypatch.setattr(sys, "path", sys.path.copy())
    execute = runpy.run_path
    calls = []
    monkeypatch.setattr(runpy, "run_path", lambda path, **kwargs: calls.append((path, kwargs)))
    execute(str(entrypoint), run_name="__main__")
    assert sys.argv == argv
    assert calls == [(str(entrypoint.parents[1] / "piper_dual/main_dual_ros.py"), {"run_name": "__main__"})]
