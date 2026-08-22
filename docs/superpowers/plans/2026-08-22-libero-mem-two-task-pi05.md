# LIBERO-Mem Two-Task PI05 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert two exact LIBERO-Mem bowl tasks into one verified public LeRobot dataset and start one PI05 fine-tuning run for 20,000 steps with global batch size 64 and checkpoints every 5,000 steps.

**Architecture:** A standalone HDF5-to-LeRobot converter validates and streams successful demos from the two source files, preserving LIBERO-Mem's existing image orientation, 8D end-effector state, 7D delta action, and exact language prompts. The existing OpenPI LIBERO transform is reused through one dedicated training config. Data is verified locally and after server download before task-specific norm statistics and a four-GPU FSDP training launch.

**Tech Stack:** Python 3.11, h5py, NumPy, LeRobotDataset, Hugging Face Hub, pytest, OpenPI/JAX, remote B20Z GPUs.

## Global Constraints

- Include only `KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate` and `KITCHEN_SCENE1_3_lift_the_bowl_and_place_it_back_on_the_plate_3_times`.
- Source artifacts are the two matching `_demo.hdf5` files plus `metainfo.json` from `libero-mem/LIBERO-Mem` revision `6df130c05b82e09294fa9534e84a069687b72d33`.
- Include only demo IDs present in both HDF5 and metadata with `success is true`.
- Preserve source RGB pixels unchanged; do not rotate, mirror, resize, or swap channels during conversion.
- Store state as float32 `(8,) = concat(ee_states[0:6], gripper_states[0:2])`.
- Store action as unchanged float32 `(7,)`; do not add a delta transform or alter the gripper sign.
- Use the exact metadata `task_description` as the LeRobot task string.
- Omit depth, segmentation, reasoning, joint state, and simulator state from model inputs.
- Output dataset repo is `HITdongdong/libero_mem_bowl_two_tasks`, public.
- Training config is `pi05_libero_mem_bowl_two_tasks`: 20,000 steps, global batch 64, save every 5,000, keep every 5,000, FSDP 4.
- Do not overwrite or revert unrelated modified files already present in the worktree.
- Run only focused tests during implementation; run the real conversion and training smoke verification once.

---

### Task 1: Add failing converter contract tests

**Files:**
- Create: `examples/libero/tests/test_convert_libero_mem_data_to_lerobot.py`
- Reference: `examples/libero/convert_libero_data_to_lerobot.py`

**Interfaces:**
- Imports `examples/libero/convert_libero_mem_data_to_lerobot.py` as `converter`.
- Requires `TASK_SPECS`, `load_task_metadata`, `iter_valid_demo_ids`, `read_demo`, `populate_dataset`, `create_empty_dataset`, and `_parse_args`.
- `read_demo(hdf5_file: h5py.File, demo_id: str, *, task_name: str) -> DemoData` returns validated current-frame image/state/action arrays.
- `populate_dataset(dataset: object, source_paths: dict[str, Path], metadata: dict[str, dict]) -> ConversionSummary` writes all accepted episodes.

- [ ] **Step 1: Create a synthetic source fixture.**

Create a temporary HDF5 with `data/demo_0` and `data/demo_1`. Each group contains `(T,7)` actions and `obs/{ee_states,gripper_states,agentview_rgb,eye_in_hand_rgb}` with distinct values. Metadata contains `demo_0` with `success=true`, exact prompt, and `demo_1` with `success=false`. A fake dataset records frames and episode saves.

- [ ] **Step 2: Assert exact conversion behavior.**

Tests must assert:

```python
assert summary.episodes == 1
assert summary.frames == frame_count
np.testing.assert_array_equal(frame["image"], source_agent_image)
np.testing.assert_array_equal(frame["wrist_image"], source_wrist_image)
np.testing.assert_array_equal(frame["state"], np.r_[ee_state, gripper_state].astype(np.float32))
np.testing.assert_array_equal(frame["actions"], source_action.astype(np.float32))
assert frame["task"] == exact_prompt
```

Add a metadata/HDF5 intersection test, a `success=false` exclusion test, and dataset feature assertions for `(256,256,3)`, state `(8,)`, actions `(7,)`, `fps=10`, and `robot_type="panda"`.

- [ ] **Step 3: Assert invalid input rejection.**

Parameterize wrong action width, wrong state/gripper width, wrong image shape, inconsistent `T`, zero frames, NaN state, and Inf action. Match errors containing task name, demo ID, field, expected shape, and actual shape where applicable.

- [ ] **Step 4: Run the focused test and observe RED.**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q examples/libero/tests/test_convert_libero_mem_data_to_lerobot.py
```

Expected: collection fails because `convert_libero_mem_data_to_lerobot.py` does not exist.

---

### Task 2: Implement the streaming LIBERO-Mem converter

**Files:**
- Create: `examples/libero/convert_libero_mem_data_to_lerobot.py`
- Test: `examples/libero/tests/test_convert_libero_mem_data_to_lerobot.py`

**Interfaces:**
- `TaskSpec(task_name: str, filename: str, prompt: str)` is frozen.
- `DemoData(images, wrist_images, states, actions)` stores validated arrays.
- `ConversionSummary(episodes: int, frames: int, episodes_by_task: dict[str,int], frames_by_task: dict[str,int])` reports output.
- `download_sources(raw_dir: Path, *, revision: str) -> dict[str, Path]` downloads only the three required artifacts.
- `convert_dataset(raw_dir: Path, repo_id: str, *, output_dir: Path | None, overwrite: bool, push_to_hub: bool) -> tuple[Path, ConversionSummary]` is the main reusable entry.

- [ ] **Step 1: Define exact target constants and CLI.**

Use two `TaskSpec` entries with exact task keys, filenames, and prompts. CLI flags:

```text
--raw-dir PATH
--repo-id HITdongdong/libero_mem_bowl_two_tasks
--output-dir PATH
--revision 6df130c05b82e09294fa9534e84a069687b72d33
--download/--no-download
--overwrite
--push-to-hub
```

Default output uses `HF_LEROBOT_HOME / repo_id`.

- [ ] **Step 2: Load only target metadata.**

Implement a streaming top-level JSON extractor if `ijson` is available; otherwise use standard `json.load` with a clear log that metadata is loaded once. Return only the two target task dictionaries and fail if either is absent. Do not copy `metainfo.json` into the output dataset.

- [ ] **Step 3: Validate demos before writing frames.**

`iter_valid_demo_ids` sorts IDs numerically, intersects HDF5 groups with metadata keys, and yields only entries whose `success is True`. `read_demo` validates exact ranks and trailing shapes, equal positive `T`, numeric dtypes, finite state/actions, and uint8 RGB. It concatenates state once per demo into a float32 `(T,8)` array and casts actions once to float32 without changing values.

- [ ] **Step 4: Create and populate the LeRobot dataset.**

Declare features named `image`, `wrist_image`, `state`, and `actions`, matching existing `LeRobotLiberoDataConfig` repacking. Process one demo at a time, add its frames, then call `save_episode()`. Never retain decoded arrays after that episode is saved.

- [ ] **Step 5: Implement output and upload safety.**

Reject an existing output unless `--overwrite` is set. Remove only the requested output directory on overwrite. On conversion exception, remove the newly-created partial output. For Hub upload call `dataset.push_to_hub(tags=["libero","libero-mem","panda"], private=False, push_videos=True, license="mit")`.

- [ ] **Step 6: Run focused tests to GREEN.**

Run the Task 1 pytest command. Expected: all tests pass with no warnings or errors.

- [ ] **Step 7: Run CLI help smoke.**

Run:

```bash
.venv/bin/python examples/libero/convert_libero_mem_data_to_lerobot.py --help
```

Expected: exit 0 and all seven CLI controls shown.

---

### Task 3: Add and verify the dedicated PI05 training config

**Files:**
- Modify: `src/openpi/training/config.py`
- Create: `tests/test_libero_mem_config.py`

**Interfaces:**
- `config.get_config("pi05_libero_mem_bowl_two_tasks")` returns the exact training config.
- It reuses `LeRobotLiberoDataConfig`, `libero_policy.LiberoInputs`, and `libero_policy.LiberoOutputs` unchanged.

- [ ] **Step 1: Write the failing config test.**

Assert:

```python
cfg = config.get_config("pi05_libero_mem_bowl_two_tasks")
assert cfg.data.repo_id == "HITdongdong/libero_mem_bowl_two_tasks"
assert cfg.data.base_config.prompt_from_task is True
assert cfg.data.extra_delta_transform is False
assert cfg.model.pi05 is True
assert cfg.model.action_horizon == 10
assert cfg.model.discrete_state_input is False
assert cfg.batch_size == 64
assert cfg.num_train_steps == 20_000
assert cfg.save_interval == 5_000
assert cfg.keep_period == 5_000
assert cfg.fsdp_devices == 4
```

Create the data transforms with temporary assets and verify an 8D state, two RGB images, exact prompt, and `(10,7)` actions become valid model inputs; verify `LiberoOutputs` returns exactly seven action dimensions.

- [ ] **Step 2: Run the test and observe RED.**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_libero_mem_config.py
```

Expected: `ValueError` because the config name is absent.

- [ ] **Step 3: Add the minimal config.**

Copy the optimizer, EMA, learning-rate schedule, and PI05 base weight loader from `pi05_libero`; set only the dedicated name/repo and requested batch/step/save/FSDP fields. Do not add a new policy or data config class.

- [ ] **Step 4: Run focused config tests to GREEN.**

Run `tests/test_libero_mem_config.py` and the existing relevant LIBERO/config tests if present. Expected: all pass.

---

### Task 4: Download, inspect, convert, and publish real data

**Files:**
- Source cache: `/home/agilex/.cache/huggingface/`
- Local LeRobot output: `/home/agilex/lerobot_datasets/local/libero_mem_bowl_two_tasks`

**Interfaces:**
- Input revision and byte sizes must match the approved source contract.
- Hub output must be public and loadable as `HITdongdong/libero_mem_bowl_two_tasks`.

- [ ] **Step 1: Check local free space.**

Require enough room for 28.1 GB source, temporary/output encoding, and cache overhead. If insufficient, place source/output under a larger mounted filesystem rather than deleting unrelated files.

- [ ] **Step 2: Download exactly three artifacts.**

Run the converter with `--download`, pinned revision, and without upload first. Hugging Face cache resumes partial downloads.

- [ ] **Step 3: Record source structure before conversion.**

For each task record source bytes, HDF5 demo count, metadata intersection count, successful episode count, total accepted frames, shapes, dtypes, and prompt. Fail if either task contributes zero successful episodes.

- [ ] **Step 4: Convert locally.**

Stream all accepted episodes into the one output dataset. Capture the final `ConversionSummary` and output disk size.

- [ ] **Step 5: Verify source-to-output behavior.**

Load the resulting LeRobot dataset. Check metadata totals, episode boundaries, task distribution, finite state/action values, and exact first/middle/last state/action values for one episode from each task. Decode corresponding images and compare pixels or lossless source fingerprints according to the LeRobot image codec; verify orientation with corner-coded synthetic tests already proving no converter-side rotation.

- [ ] **Step 6: Publish and verify Hub state.**

Push public dataset. Query `dataset_info(files_metadata=True)` and verify `private=False`, file count, total bytes, required `meta/*`, data parquet files, media files, and both task strings in metadata/data.

---

### Task 5: Synchronize code/data and compute dedicated norm statistics

**Files:**
- Remote repo: `/pfs/pfs-7jnepv/lgd/SANE`
- Remote dataset root: `/pfs/pfs-7jnepv/lgd/datasets/lerobot/HITdongdong/libero_mem_bowl_two_tasks`
- Remote assets: `/pfs/pfs-7jnepv/lgd/SANE/assets/pi05_libero_mem_bowl_two_tasks/HITdongdong/libero_mem_bowl_two_tasks/norm_stats.json`

**Interfaces:**
- Remote `src/openpi/training/config.py` and focused config test must match verified local files.
- `HF_LEROBOT_HOME=/pfs/pfs-7jnepv/lgd/datasets/lerobot` selects the remote dataset.

- [ ] **Step 1: Select an accessible server with shared PFS.**

Prefer 9024 if reachable. Copy only the new converter/test/config changes needed for reproducibility; do not overwrite unrelated remote changes without comparing the target sections.

- [ ] **Step 2: Download dataset from Hub on the server.**

Use `snapshot_download(..., repo_type="dataset", local_dir=...)`. Verify remote file count, total bytes, episodes, frames, tasks, state `(8,)`, action `(7,)`, and finite samples.

- [ ] **Step 3: Run remote focused config tests.**

Expected: the dedicated config test passes in the remote environment.

- [ ] **Step 4: Compute norm stats.**

Run:

```bash
HF_LEROBOT_HOME=/pfs/pfs-7jnepv/lgd/datasets/lerobot \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python scripts/compute_norm_stats.py --config-name pi05_libero_mem_bowl_two_tasks
```

Capture the completed batch count. Verify `norm_stats.json` exists, is nonempty JSON, contains state/action statistics of the expected padded model width, and is selected by `data.create(...)`.

---

### Task 6: Allocate four GPUs, launch training, and prove startup

**Files:**
- Remote checkpoint: `/pfs/pfs-7jnepv/lgd/SANE/checkpoints/pi05_libero_mem_bowl_two_tasks/libero_mem_bowl_two_tasks_20k`
- Remote log: `/pfs/pfs-7jnepv/lgd/SANE/logs/pi05_libero_mem_bowl_two_tasks_20k.log`

**Interfaces:**
- Launch command must include config name, explicit experiment name, checkpoint/assets dirs, 20,000 steps, batch 64, and FSDP 4.
- `CUDA_VISIBLE_DEVICES` must identify four cards with no compute process immediately before launch.

- [ ] **Step 1: Re-scan GPU process ownership and memory.**

Prefer 9024 GPUs `0,1,2,3` only if all four still have no compute process, utilization near zero, and enough free memory. Do not share cards. If occupied, scan other accessible servers for one clean four-card group.

- [ ] **Step 2: Preflight launch inputs.**

Verify config lookup, dataset root, norm stats selection, PI05 base assets, checkpoint output nonexistence, and log directory. Refuse accidental resume/overwrite.

- [ ] **Step 3: Launch detached training.**

Run with the selected four cards:

```bash
CUDA_VISIBLE_DEVICES=<four-cards> \
HF_LEROBOT_HOME=/pfs/pfs-7jnepv/lgd/datasets/lerobot \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
nohup .venv/bin/python scripts/train.py pi05_libero_mem_bowl_two_tasks \
  --exp-name libero_mem_bowl_two_tasks_20k \
  --checkpoint-base-dir /pfs/pfs-7jnepv/lgd/SANE/checkpoints \
  --assets-base-dir /pfs/pfs-7jnepv/lgd/SANE/assets \
  --num-train-steps 20000 \
  --batch-size 64 \
  --fsdp-devices 4 \
  > /pfs/pfs-7jnepv/lgd/SANE/logs/pi05_libero_mem_bowl_two_tasks_20k.log 2>&1 < /dev/null &
```

- [ ] **Step 4: Verify process and exact runtime arguments.**

Capture PID and process command. Verify the four selected GPUs are owned by that training process and no fifth GPU is used.

- [ ] **Step 5: Verify first metrics and checkpoint cadence.**

Wait conditionally for training steps rather than a fixed completion estimate. Confirm steps advance across at least two observations, loss/grad metrics are finite, dataset and dedicated norm stats loaded, W&B run exists if configured, and no Traceback/OOM/fatal line appears. Inspect the effective config or startup log to prove `save_interval=5000` and `keep_period=5000`; expected retained checkpoints are 5k, 10k, 15k, and 20k.

---

## Self-Review

- **Spec coverage:** Both exact source tasks, successful-demo filtering, unchanged image/state/action semantics, one multi-task dataset/model, public Hub publication, dedicated norm stats, 20k/batch64/FSDP4 training, 5k save cadence, and startup evidence are each assigned to a task.
- **Placeholder scan:** No TBD, TODO, deferred implementation, unspecified error handling, or “similar to” implementation step remains.
- **Type consistency:** Converter tests and implementation use `image`, `wrist_image`, `state`, `actions`, and `task`; OpenPI's existing repack maps those exact fields to the inference contract. State is 8D and action is 7D end to end.
- **Operational safety:** Source downloads are pinned/resumable; conversion is streamed; partial output is removed; unrelated worktree changes are untouched; Hub and remote copies are independently verified; GPU sharing and accidental checkpoint overwrite are prohibited.
