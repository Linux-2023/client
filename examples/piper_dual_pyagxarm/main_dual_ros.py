#!/usr/bin/env python3
"""Use the preserved OpenPI runtime with the pyAgxArm local ROS driver."""
from pathlib import Path
import runpy
import sys

LEGACY = Path(__file__).resolve().parents[1] / "piper_dual"

if __name__ == "__main__":
    sys.path.insert(0, str(LEGACY))
    if "--control-stack" not in sys.argv and not any(v.startswith("--control-stack=") for v in sys.argv):
        sys.argv.extend(["--control-stack", "local-ros"])
    if "--ros2-config" not in sys.argv and not any(v.startswith("--ros2-config=") for v in sys.argv):
        sys.argv.extend(["--ros2-config", str(LEGACY / "ros2_piper_dual_local.yaml")])
    runpy.run_path(str(LEGACY / "main_dual_ros.py"), run_name="__main__")
