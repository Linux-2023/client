
# Running openpi models remotely

We provide utilities for running openpi models remotely. This is useful for running inference on more powerful GPUs off-robot, and also helps keep the robot and policy environments separate (and e.g. avoid dependency hell with robot software).

## Starting a remote policy server

To start a remote policy server, you can simply run the following command:

```bash
uv run scripts/serve_policy.py --env=[DROID | ALOHA | LIBERO]
```

The `env` argument specifies which $\pi_0$ checkpoint should be loaded. Under the hood, this script will execute a command like the following, which you can use to start a policy server, e.g. for checkpoints you trained yourself (here an example for the DROID environment):

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid
```

This will start a policy server that will serve the policy specified by the `config` and `dir` arguments. The policy will be served on the specified port (default: 8000).

### Piper tidy-desk Training-RTC inference

The `pi05_piper_dual_tidy_desk_ros2_shifted_pytorch` preset supports the
[exported Training-RTC checkpoint](https://huggingface.co/HITdongdong/pi05_piper_dual_tidy_desk_ros2_shifted_training_rtc_step29999_pytorch).
Download the complete export, including `config.json`, `model.safetensors`, and
`assets/**/norm_stats.json`. Model weights and datasets are not part of this repository.
Downloading a checkpoint does not register its preset in a different code checkout.

Use the Python environment belonging to this checkout:

```bash
CKPT=checkpoints/pi05_piper_dual_tidy_desk_ros2_shifted_training_rtc/piper_tidy_1a127a8d_trainingrtc_bs16_20261029_r1/29999_pytorch
.venv/bin/python scripts/serve_policy.py --port=8000 policy:checkpoint \
  --policy.config=pi05_piper_dual_tidy_desk_ros2_shifted_pytorch \
  --policy.dir="$CKPT"
```

Set `CKPT` to the actual download directory. PyTorch loading requires
`transformers==4.53.2` and this repository's replacements. Reapply them in the
**server's environment** after updating the checkout, then restart the server:

```bash
cp -r --remove-destination src/openpi/models_pytorch/transformers_replace/models \
  .venv/lib/python3.11/site-packages/transformers/
```

Training-RTC requires the updated Gemma normalization for per-position diffusion
time conditioning. The preset takes three RGB images (`cam_high`, `cam_left_wrist`,
`cam_right_wrist`) and a 14D state, and returns 50 × 14 absolute joint/gripper targets
using the checkpoint's normalization and joint-delta inverse transform. It retains
`discrete_state_input=False` and the dataset's gripper units. Starting a policy
server does not command the robot; use a trusted network.

#### Client mode and wire contract

The server command is unchanged across ordinary, legacy RTC and Training-RTC
requests. A compatible checkpoint advertises `training_rtc.supported=true`,
`max_prefix_steps=15`, `action_horizon=50`, and `action_dim=14`. This capability is
validated against the checkpoint's training configuration, not its filename.
Unsupported policies reject Training-RTC requests.

Add `--training-rtc` to the joint client's command. This bounded example disables
action publication but still requires the normal observation/ROS infrastructure:

```bash
.venv/bin/python examples/piper_dual/main_dual_ros.py \
  --host 127.0.0.1 --port 8000 \
  --training-rtc --use-async --fps 30 \
  --action-horizon 20 --actions-during-latency 10 \
  --prompt "Tidy up the crumpled papers and pens on the desk." \
  --num-steps 90 --num-episodes 1 --dry-run --no-publish-actions
```

- Training-RTC is opt-in. It supersedes `--use-rtc`; the two algorithms are never
  combined. `--no-training-rtc` restores the existing legacy/non-RTC selection.
- The asynchronous joint broker executes one model row per tick and requires
  `--fps 30`. Its committed window is `--actions-during-latency`, from 1 through
  15 rows. Cold start sends an empty prefix. Synchronous execution and handoff
  blending are incompatible with the commitment contract.
- Early results wait for their committed handoff. Late results are discarded
  rather than replayed at a different time. Reset detaches stale results, and
  inference failures are surfaced instead of falling back to legacy guidance.
- `main_dual_delta_joint_ros.py` also accepts `--training-rtc`; it requires its
  normal verified deployment configuration and absolute server output.
  Pass `--no-record-ros-diagnostics` unless its optional external ROS diagnostics
  recorder is installed at the configured location. That workstation-specific
  recorder is not included in this publication.
- The lightweight `main_dual_delta_ros.py` wrapper also supports this mode with
  `--fps 30 --server-output absolute`. Training-RTC rejects client-side delta
  decoding: committed prefixes and server responses must be absolute targets.
- For TOPP, use `main_dual_topp_rtc_ros.py`: Training-RTC selects the committed
  backend with a 30 Hz model knot clock (`--topp-path-fps 30`). Other model knot
  rates are rejected; output rates need a usable common handoff grid. A TOPP
  anchor at row `d` includes prefix rows `0..d`, so `d <= 14`, and new targets
  start at `d+1`. `main_dual_topp_ros.py` remains the separate non-RTC entrypoint.

Custom clients send `observation["training_rtc"] = {"action_prefix": prefix}`,
where `prefix` is a finite `float32` array of shape `(P,14)`, `0 <= P <= 15`.
These are committed future **absolute** targets, not history or padded leftovers.
Do not mix this request with `rtc_obs` or GPR previous-action fields. The server
converts the 12 joints to deltas from the **current request state**, keeps grippers
absolute, normalizes with checkpoint assets and pads the model width to 32.
Prefix positions stay clean at `t=0` throughout denoising; only the suffix is
generated at the current diffusion time. Ordinary sampling and legacy gradient
guidance remain separate.

#### Known deadline limitation

The plain asynchronous joint broker uses a fixed committed window; it does not
adapt that budget or terminate after repeated late replies. The old plan runs
to its last action and can then remain at that endpoint indefinitely. This
publication does **not** fix that behavior.

At 30 Hz, five rows allow only 166.7 ms. Observed five-row GPU requests took
165.332–169.649 ms. A 120-tick replay through the actual broker at 169.649 ms
discarded 16 late replies and repeated the endpoint for 70 ticks with five rows;
ten rows (333.3 ms) adopted three successor chunks without endpoint repeats.
The example uses ten rows to provide margin for those measurements, not a
guarantee for other hosts or loads. Changing `--action-horizon` alone does not
increase the fixed handoff window. TOPP additionally budgets planning time and
has its own capped-window/trajectory-feasibility constraints.

CPU/offline verification establishes protocol, numerical and queue behavior,
not real-time throughput, robot tracking, collision safety or task success.

## Querying the remote policy server from your robot code

We provide a client utility with minimal dependencies that you can easily embed into any robot codebase.

First, install the `openpi-client` package in your robot environment:

```bash
cd $OPENPI_ROOT/packages/openpi-client
pip install -e .
```

Then, you can use the client to query the remote policy server from your robot code. Here's an example of how to do this:

```python
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

for step in range(num_steps):
    # Inside the episode loop, construct the observation.
    # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
    # We provide utilities for resizing images + uint8 conversion so you match the training routines.
    # The typical resize_size for pre-trained pi0 models is 224.
    # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
    observation = {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        ),
        "observation/state": state,
        "prompt": task_instruction,
    }

    # Call the policy server with the current observation.
    # This returns an action chunk of shape (action_horizon, action_dim).
    # Note that you typically only need to call the policy every N steps and execute steps
    # from the predicted action chunk open-loop in the remaining steps.
    action_chunk = client.infer(observation)["actions"]

    # Execute the actions in the environment.
    ...

```

Here, the `host` and `port` arguments specify the IP address and port of the remote policy server. You can also specify these as command-line arguments to your robot code, or hard-code them in your robot codebase. The `observation` is a dictionary of observations and the prompt, following the specification of the policy inputs for the policy you are serving. We have concrete examples of how to construct this dictionary for different environments in the [simple client example](../examples/simple_client/main.py).
