# Task 4 Three-Stack Selection Report

## Result
DONE

## Scope
Focused only on:
- `examples/piper_dual/main_dual.py`
- `examples/piper_dual/env_dual.py`
- `examples/piper_dual/tests/test_main_dual.py`
- `examples/piper_dual/tests/test_ros2_environment.py`

No runtime artifacts, docs, integration suite, or unrelated files were changed for this task.

## Corrective commit
- `a6d68ac1488f63e03fd0aa41b381e010ded31c5e` (`Fix Task 4 default attribution split`)
- Commit contains only `examples/piper_dual/main_dual.py` and `examples/piper_dual/tests/test_main_dual.py`:
  - `git show --stat --oneline --no-renames HEAD -- examples/piper_dual/main_dual.py examples/piper_dual/tests/test_main_dual.py examples/piper_dual/env_dual.py examples/piper_dual/tests/test_ros2_environment.py .superpowers/sdd/task-4-three-stack-report.md`
  - Evidence: `2 files changed, 68 insertions(+), 9 deletions(-)`.

## RED
Focused verification was run before the correction was complete:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q examples/piper_dual/tests/test_main_dual.py examples/piper_dual/tests/test_ros2_environment.py
5 failed, 54 passed in 0.46s
```

The failures showed the attribution/default split was still wrong: the committed/indexed Task 4 state still had `action_horizon=30`, the default backend was still `sdk`, ROS2 stack resolution defaulted to `None`, and one test import was missing.

A later staged/index check intentionally still failed before staging the correction:

```text
1 failed, 58 passed in 0.37s
FAILED test_indexed_task4_defaults_keep_only_stack_and_safety_changes - assert 30 == 10
```

## GREEN
After re-grounding against `git show a2c3f28:examples/piper_dual/main_dual.py`, the corrective commit restored the pre-existing non-stack defaults in the committed Task 4 state while retaining approved Task 4 behavior:
- committed/default `backend="ros2"`
- committed/default `dry_run=True`
- committed/default `publish_actions=False`
- `--control-stack` choices and ROS2 default `official-ros`
- SDK backend rejects/omits control stack
- ROS2 environment kwargs forward control stack
- `--max-action-delta` parser and ROS2-only forwarding
- focused tests cover safe defaults, stack resolution/propagation, max delta, SDK lazy compatibility, and committed non-stack defaults.

Focused verification after staging the correction:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q examples/piper_dual/tests/test_main_dual.py examples/piper_dual/tests/test_ros2_environment.py
59 passed in 0.35s
```

Focused verification after the corrective commit and after reapplying the six user defaults as unstaged work:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q examples/piper_dual/tests/test_main_dual.py examples/piper_dual/tests/test_ros2_environment.py
59 passed in 0.34s
```

## Attribution and final unstaged state
After commit `a6d68ac1488f63e03fd0aa41b381e010ded31c5e`, the six values from `artifact://1396` were reapplied as unstaged edits in `examples/piper_dual/main_dual.py` only:
- `action_horizon=30`
- `num_steps=8000`
- `port=8001`
- `prompt="Place the red and blue blocks on the wooden board"`
- `use_async=False`
- `record_mode=False`

The final dirty working tree also keeps the Task 4 safety/backend defaults in the working file:
- `backend="ros2"`
- `dry_run=True`
- `publish_actions=False`

Final target status evidence:

```text
a6d68ac
 M .superpowers/sdd/task-4-three-stack-report.md
 M examples/piper_dual/main_dual.py
```

Final unstaged `main_dual.py` diff versus the corrective commit is exactly the six user-default edits:

```text
examples/piper_dual/main_dual.py | 12 ++++++------
1 file changed, 6 insertions(+), 6 deletions(-)
```

No user changes were reset, checked out, or overwritten wholesale.
