# Task 3 Three-Stack Report

## Status
Focused GREEN for Task 3 direct `piper_sdk` ROS adapter using fake controllers only.

## Changed paths
- `examples/piper_dual/piper_direct_sdk_adapter.py`
- `examples/piper_dual/tests/test_piper_direct_sdk_adapter.py`
- `.superpowers/sdd/task-3-three-stack-report.md`

## RED
Command:

```bash
/usr/bin/python3 -m pytest -q examples/piper_dual/tests/test_piper_direct_sdk_adapter.py
```

Output:

```text
________________________ ERROR collecting test session _________________________
/usr/lib/python3/dist-packages/pluggy/hooks.py:286: in __call__
    return self._hookexec(self, self.get_hookimpls(), kwargs)
/usr/lib/python3/dist-packages/pluggy/manager.py:92: in _hookexec
    return self._inner_hookexec(hook, methods, kwargs)
/usr/lib/python3/dist-packages/pluggy/manager.py:83: in <lambda>
    self._inner_hookexec = lambda hook, methods, kwargs: hook.multicall(
/usr/lib/python3/dist-packages/_pytest/python.py:200: in pytest_collect_file
    module: Module = ihook.pytest_pycollect_makemodule(path=path, parent=parent)
/usr/lib/python3/dist-packages/pluggy/hooks.py:286: in __call__
    return self._hookexec(self, self.get_hookimpls(), kwargs)
/usr/lib/python3/dist-packages/pluggy/manager.py:92: in _hookexec
    return self._inner_hookexec(hook, methods, kwargs)
/usr/lib/python3/dist-packages/pluggy/manager.py:83: in <lambda>
    self._inner_hookexec = lambda hook, methods, kwargs: hook.multicall(
/opt/ros/humble/lib/python3.10/site-packages/launch_testing/pytest/hooks.py:193: in pytest_pycollect_makemodule
    entrypoint = find_launch_test_entrypoint(path)
/opt/ros/humble/lib/python3.10/site-packages/launch_testing/pytest/hooks.py:186: in find_launch_test_entrypoint
    module = path.pyimport()
/usr/lib/python3/dist-packages/py/_path/local.py:704: in pyimport
    __import__(modname)
/usr/lib/python3/dist-packages/_pytest/assertion/rewrite.py:170: in exec_module
    exec(co, module.__dict__)
examples/piper_dual/tests/test_piper_direct_sdk_adapter.py:13: in <module>
    import piper_direct_sdk_adapter as adapter
E   ModuleNotFoundError: No module named 'piper_direct_sdk_adapter'
=========================== short test summary info ============================
ERROR  - ModuleNotFoundError: No module named 'piper_direct_sdk_adapter'
pytest: 1 error in 0.09s

Command exited with code 2
```

## GREEN
Command:

```bash
/usr/bin/python3 -m pytest -q examples/piper_dual/tests/test_piper_direct_sdk_adapter.py
```

Output:

```text
.............                                                            [100%]
13 passed in 0.16s
```

## Implementation notes
- Added lazy `_load_piper_controller()` and CLI `--allow-enable` gate; without the flag `main()` returns 2 before calling the loader.
- `DirectSdkArm` accepts dependency-injected controller factories, so tests never import or construct the real controller and never open CAN.
- Joint commands call unchanged `PiperController.set_joint()` and therefore preserve its native fixed 100% joint speed; the adapter only converts the unified gripper meter contract to/from the third-party normalized `[0, 1]` range by `0.07`.
- The ROS node publishes actual feedback separately from accepted command echoes, builds status from SDK status fields, projects disabled/missing-driver readiness outward as `ctrl_mode=0`, and permanently locks future actions after such a status fault.
- Shutdown cancels the timer, rejects future actions, calls `DisableArm(7)` when available, calls `DisconnectPort()` for both arms, and returns cleanup errors.

## Self-review
- Scope limited to the requested adapter, focused fake-controller test, and this Task 3 report.
- Third-party reference tree was read only, not modified.
- Existing user changes in the worktree were not staged or modified by this task.

## Concerns
None for the focused fake-controller acceptance. No physical hardware/CAN test was run by design.

## Commit
61e06e8c9919b6179cc816447a6bd3bc733c0052
