# Task 1 three-control-stack report

## Status
Implemented immutable control-stack profiles and focused profile tests for Task 1 only.

## Changed paths
- `examples/piper_dual/control_stack.py`
- `examples/piper_dual/ros2_piper_dual.yaml`
- `examples/piper_dual/ros2_piper_dual_local.yaml`
- `examples/piper_dual/ros2_piper_dual_direct_sdk.yaml`
- `examples/piper_dual/tests/test_control_stack.py`

## RED evidence
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q examples/piper_dual/tests/test_control_stack.py

==================================== ERRORS ====================================
_______ ERROR collecting examples/piper_dual/tests/test_control_stack.py _______
ImportError while importing test module '/home/agilex/client/.worktrees/piper-dual-ros2-migration/examples/piper_dual/tests/test_control_stack.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../../.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
examples/piper_dual/tests/test_control_stack.py:10: in <module>
    from control_stack import ControlStackProfile
E   ModuleNotFoundError: No module named 'control_stack'
=========================== short test summary info ============================
ERROR examples/piper_dual/tests/test_control_stack.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.07s
```

## GREEN evidence
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q examples/piper_dual/tests/test_control_stack.py
.......                                                                  [100%]
7 passed in 0.04s
```

## Self-review
- Profiles are frozen/slotted and expose tuple endpoint lists.
- `profile_for()` returns exactly `local-ros`, `official-ros`, and `direct-sdk`, and rejects unknown IDs.
- `load_profile_config()` uses `yaml.safe_load`, requires a top-level mapping, requires string `control_stack_id`, and rejects identity mismatches without inferring from filenames.
- Official YAML received `control_stack_id: official-ros`; its previously present official contract remains staged with the Task 1 identity.
- Local and official speed is 30; direct-sdk speed is 100.
- Direct profile includes feedback streams (`/direct_sdk/joint_left`, `/direct_sdk/joint_right`) plus command-echo streams (`/direct_sdk/joint_ctrl_left`, `/direct_sdk/joint_ctrl_right`).
- No hardware, ROS process, CAN command, broad suite, or formatter was run.

## Concerns
- The worktree has many unrelated pre-existing user edits. I staged only Task 1 paths.

## Commit
da1f7dd Add Piper dual control stack profiles
