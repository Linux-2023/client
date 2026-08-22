# LIBERO-Mem Two-Task PI05 Design

## Goal

Convert exactly two LIBERO-Mem tasks into one LeRobot v2 dataset compatible with OpenPI's existing LIBERO policy contract, then fine-tune one PI05 model for 20,000 steps with global batch size 64 and checkpoints every 5,000 steps.

The selected tasks are:

1. `KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate`
2. `KITCHEN_SCENE1_3_lift_the_bowl_and_place_it_back_on_the_plate_3_times`

The source files are the public Hugging Face dataset artifacts:

- `KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate_demo.hdf5` — 9,668,822,790 bytes
- `KITCHEN_SCENE1_3_lift_the_bowl_and_place_it_back_on_the_plate_3_times_demo.hdf5` — 18,431,683,248 bytes
- `metainfo.json` — shared metadata used to identify task descriptions and successful demonstrations

## Scope

This work produces one multi-task PI05 checkpoint. It does not train one checkpoint per task, add object-reasoning tokens, add depth or segmentation inputs, or change LIBERO control semantics.

The converted dataset will be published as `HITdongdong/libero_mem_bowl_two_tasks` so the training server can fetch the exact verified artifact. The repository will be public, matching the public source license and the existing project data distribution pattern.

## Source Contract

For each HDF5 task file, a usable episode is a `data/demo_<id>` group that also has `demo_<id>` metadata under the matching task key in `metainfo.json`. Metadata entries with `success != true` are excluded. Missing groups, inconsistent sequence lengths, unsupported shapes, or non-finite state/action values are hard conversion errors rather than silently truncated data.

Each accepted demo must contain:

- `actions`: `(T, 7)`
- `obs/ee_states`: `(T, 6)`
- `obs/gripper_states`: `(T, 2)`
- `obs/agentview_rgb`: `(T, 256, 256, 3)`
- `obs/eye_in_hand_rgb`: `(T, 256, 256, 3)`

All five arrays must have the same `T`, and `T` must be positive.

## LeRobot Dataset Contract

One LeRobot episode is emitted for every accepted source demo. Source task and demo identifiers are retained in conversion reports; task identity in training is carried by the per-frame `task` string.

The dataset uses `fps=10`, `robot_type="panda"`, and these stored features:

| LeRobot key | dtype | shape | Source |
| --- | --- | --- | --- |
| `image` | image/uint8 | `(256, 256, 3)` | `obs/agentview_rgb[t]` |
| `wrist_image` | image/uint8 | `(256, 256, 3)` | `obs/eye_in_hand_rgb[t]` |
| `state` | float32 | `(8,)` | `concat(obs/ee_states[t], obs/gripper_states[t])` |
| `actions` | float32 | `(7,)` | `actions[t]` |
| `task` | string | scalar | official `task_description` |

The HDF5 RGB arrays were produced by LIBERO-Mem's replay path after rotating simulator images by 180 degrees. The converter therefore writes their pixels unchanged. OpenPI inference rotates live `agentview_image` and `robot0_eye_in_hand_image` by 180 degrees, preserving train/evaluation orientation parity.

`state` is the official LIBERO-Mem 8D end-effector state: 6D end-effector pose followed by two gripper coordinates. It matches OpenPI's `LiberoInputs` and evaluation state construction (`eef_pos`, quaternion-to-axis-angle, gripper qpos).

`actions` remains the source 7D LIBERO end-effector delta action. No extra delta transform, scaling, clipping, gripper sign change, temporal shift, or action relabeling is applied.

The two exact prompts are:

- `pick up the bowl and place it back on the plate`
- `lift the bowl and place it back on the plate 3 times`

Depth, segmentation, bounding boxes, object-centric reasoning, joint state, done flags, and simulator state are intentionally omitted because PI05's existing LIBERO policy does not consume them.

## Implementation Boundaries

### Converter

Create `examples/libero/convert_libero_mem_data_to_lerobot.py` with a small testable core and a CLI. The core validates one demo and yields frames; the CLI downloads only the two target HDF5 files plus `metainfo.json`, creates the LeRobot dataset, saves each episode, emits a conversion summary, and optionally pushes the result to Hugging Face.

The converter must stream one demo at a time. It must not materialize both 28 GB source files or all decoded episodes in memory. Existing source downloads are reused by the Hugging Face cache.

### Tests

Create `examples/libero/tests/test_convert_libero_mem_data_to_lerobot.py`. Tests use a synthetic temporary HDF5 plus metadata and are written before implementation. They establish these observable contracts:

1. exactly the metadata/HDF5 intersection with `success=true` is converted;
2. image pixels are unchanged, proving no accidental flip or channel swap;
3. `state` equals the ordered 6D+2D concatenation and is float32;
4. 7D actions are unchanged and float32;
5. the exact task prompt is attached to every frame;
6. mismatched lengths and incorrect shapes fail with a precise error;
7. non-finite state/action values fail instead of contaminating norm statistics.

After conversion, a real-data verification opens the resulting LeRobot dataset and checks episode count, frame count, task distribution, feature shapes/dtypes, finite numeric values, episode boundaries, and first/middle/last source-to-output samples for both tasks.

### OpenPI Training Configuration

Add one `TrainConfig` named `pi05_libero_mem_bowl_two_tasks` in `src/openpi/training/config.py`. It reuses `LeRobotLiberoDataConfig` and `libero_policy.LiberoInputs/LiberoOutputs` with:

- `repo_id="HITdongdong/libero_mem_bowl_two_tasks"`
- `prompt_from_task=True`
- `extra_delta_transform=False`
- `Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)`
- PI05 base checkpoint weight loader, matching the existing `pi05_libero` configuration
- `batch_size=64`
- `num_train_steps=20_000`
- `save_interval=5_000`
- `keep_period=5_000`
- `fsdp_devices=4`

The existing PI05 LIBERO optimizer, EMA, and learning-rate schedule remain unchanged unless the current config API requires their values to be copied explicitly. No new policy transform is introduced because the converted schema intentionally matches the established LIBERO contract.

A focused config test verifies repo ID, 8D transformed state, 7D output action, prompt propagation, 20,000 steps, batch 64, FSDP 4, and 5,000-step save/keep periods.

## Data and Training Flow

1. Download the two HDF5 artifacts and `metainfo.json` from `libero-mem/LIBERO-Mem`.
2. Validate task keys and demo intersections.
3. Convert accepted demos into one local LeRobot dataset.
4. Verify real source-to-output samples and complete dataset metadata.
5. Push the verified dataset to `HITdongdong/libero_mem_bowl_two_tasks`.
6. Pull the dataset on the selected training server and repeat file/frame/schema checks.
7. Compute norm statistics with `pi05_libero_mem_bowl_two_tasks`; use only those task-specific statistics.
8. Recheck four GPUs immediately before launch.
9. Launch one FSDP-4 PI05 run for 20,000 steps, global batch 64.
10. Confirm process command line, loaded dataset/norm stats, first metrics, GPU ownership, absence of fatal/OOM errors, and configured checkpoint cadence.

## Resource Selection

The last observed safe allocation was server port 9024 (`instance-de10opsf-03`), GPUs `0,1,2,3`, each with approximately 182,632 MiB free and no compute process. Availability is volatile, so this is not treated as a reservation. The launch must perform a fresh process and memory check. If any of the four cards is occupied, the job is not co-located; another clean four-card group is selected or launch waits.

## Evaluation Compatibility

The official repository registers these tasks in the `libero_mem` suite at task indices 0 and 2. Live evaluation must use the same two views, rotate each live simulator image by 180 degrees, construct the 8D state as position + axis-angle + gripper qpos, and pass predicted 7D actions directly to `env.step`.

Per-task rollout limits follow the official LIBERO-Mem evaluation scripts:

- task 0, one pickup: 200 control steps;
- task 2, three lifts: 320 control steps.

Evaluation implementation or a full rollout campaign is outside this training request. The training artifact is accepted as evaluation-compatible only after its input/output transform is checked against this contract.

## Failure Handling

- Download failure: retry through the configured Hugging Face endpoint/cache; never create a partial output dataset.
- Existing output dataset: require explicit overwrite behavior in the CLI; do not silently merge runs.
- Invalid demo: fail with task name, demo ID, field, expected shape, and actual shape.
- Interrupted conversion: discard the incomplete local output before retrying.
- Hub upload: verify public visibility, revision, file count, and total bytes after upload.
- Norm-stat computation failure: do not start training.
- GPU conflict: do not share cards with an existing compute process.
- Training startup failure: diagnose from the first traceback/OOM and fix the root cause before relaunch.

## Acceptance Criteria

The work is complete when all of the following are observed:

1. The two exact source tasks, and no others, are represented in one LeRobot dataset.
2. Every included episode is a successful HDF5/metainfo intersection with consistent lengths.
3. Images, 8D state, 7D delta actions, and prompts match source samples exactly apart from numeric casting to float32.
4. The converted Hugging Face dataset is public and independently loadable on the training server.
5. Dataset-specific norm statistics exist and are selected by the training config.
6. A PI05 process is running for 20,000 steps with batch size 64 on four clean GPUs.
7. Checkpoints are configured and retained at steps 5,000, 10,000, 15,000, and 20,000.
8. Fresh logs show advancing steps and no traceback, OOM, or fatal error.
