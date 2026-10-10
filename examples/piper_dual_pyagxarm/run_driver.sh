#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash
source /home/agilex/piper_ros/install/setup.bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Confirmed hardware: ordinary Piper, S-V1.9-1 on both arms.
# Later CLI options can override these defaults for another robot.
exec /usr/bin/python3 "$SCRIPT_DIR/local_driver.py" \
  --model piper --left-firmware v189 --right-firmware v189 "$@"
