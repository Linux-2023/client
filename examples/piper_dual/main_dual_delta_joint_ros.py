#!/usr/bin/env python3
"""Verified delta-joint models through the existing ROS joint-control chain."""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import logging
from pathlib import Path
import signal
import subprocess
import tempfile
import time

import main_dual_ros as original
import main_dual_topp_rtc_ros as topp_entry
from joint_deployment import CheckedJointPolicy, load_deployment
from joint_deployment_recording import SessionRecorder, RecordedEnvironment, RecordedPolicy, atomic_json

SERVER_REPO = Path('/home/agilex/client/.worktrees/piper-dual-ros2-migration')


def build_arg_parser():
    parser = topp_entry.build_arg_parser()
    parser.description = __doc__
    parser.set_defaults(control_stack='local-ros',
                        ros2_config=Path(__file__).with_name('ros2_piper_dual_local.yaml'),
                        prompt=None, fps=None, topp_path_fps=None, rtc_model_horizon=None)
    parser.add_argument('--deployment-config', type=Path, required=True)
    parser.add_argument('--use-topp', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--record-ros-diagnostics', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--check-only', action='store_true', help='Validate files/arguments without ROS, CAN or a server')
    return parser


def validate_args(ns, spec):
    ns.prompt = spec['prompt'] if ns.prompt is None else ns.prompt
    ns.fps = spec['model_fps'] if ns.fps is None else ns.fps
    ns.topp_path_fps = spec['model_fps'] if ns.topp_path_fps is None else ns.topp_path_fps
    ns.rtc_model_horizon = spec['model_horizon'] if ns.rtc_model_horizon is None else ns.rtc_model_horizon
    if ns.fps != spec['model_fps']:
        raise ValueError(f"Client fps must match training dataset fps={spec['model_fps']}")
    args = original.Args(**{key: getattr(ns, key) for key in original.Args.__dataclass_fields__})
    original._validate_ros2_contract(args)
    if ns.control_stack != 'local-ros':
        raise ValueError('These deployment manifests use the local-ros joint topics')
    from control_stack import load_profile_config
    _, yaml = load_profile_config(ns.control_stack, ns.ros2_config)
    for kind in ('state', 'action'):
        if yaml[kind]['order'] != 'left_then_right' or yaml[kind]['joint_units'] != ['rad']*6+['m']:
            raise ValueError('ROS config joint order/units mismatch')
    if yaml['bridge_contract']['joint_action_topics'] != {'left': '/joint_left_states', 'right': '/joint_right_states'}:
        raise ValueError('Expected the local joint command topics')
    if ns.fps <= 0 or ns.num_steps <= 0 or ns.num_episodes <= 0:
        raise ValueError('fps, num-steps, num-episodes must be positive')
    if not 0 < ns.action_horizon < 50 or not 0 <= ns.actions_during_latency < 50-ns.action_horizon:
        raise ValueError('Invalid action horizon or latency for a 50-step chunk')
    if ns.training_rtc and not 0 <= ns.actions_during_latency <= 15:
        raise ValueError('Training-RTC prefix length must be between 0 and 15')
    if ns.use_rtc and not ns.use_async:
        raise ValueError('To disable asynchronous execution also pass --no-use-rtc')
    if ns.use_topp and (ns.topp_path_fps != spec['model_fps'] or ns.rtc_model_horizon != 50):
        raise ValueError('TOPP model grid must match training: --topp-path-fps 30 --rtc-model-horizon 50')
    return args


def build_policy(ns, base_policy):
    args = original.Args(**{key: getattr(ns, key) for key in original.Args.__dataclass_fields__})
    original._validate_ros2_contract(args)
    rtc = ns.training_rtc or ns.use_rtc
    backend = 'committed' if ns.training_rtc else ns.topp_rtc_backend
    if not ns.use_topp:
        from openpi_client.action_chunk_broker import ActionChunkBroker, ActionChunkBroker_RTC
        if ns.use_async:
            return ActionChunkBroker_RTC(base_policy, action_horizon=ns.action_horizon, fps=ns.fps,
                                        actions_during_latency=ns.actions_during_latency,
                                        use_rtc=ns.use_rtc, training_rtc=ns.training_rtc, trace_enabled=True,
                                        handoff_blend_steps=ns.handoff_blend_steps, handoff_angle_indices=())
        return ActionChunkBroker(base_policy, action_horizon=ns.action_horizon, fps=ns.fps, trace_enabled=True)
    from topp_policy import Planner, ToppPolicy, limits
    from topp_committed_rtc_policy import CommittedRtcPolicy
    from topp_rtc_policy import ToppRtcPolicy
    from topp_transition_planner import TransitionPlanner
    from topp_adaptive_planner import AdaptivePlanner
    cls = TransitionPlanner if rtc and backend == 'committed' and ns.topp_transition else (
        AdaptivePlanner if rtc else Planner)
    extra = dict(joint_deviation=ns.topp_transition_joint_deviation,
                 gripper_deviation=ns.topp_transition_gripper_deviation,
                 max_blend_seconds=ns.topp_transition_max_seconds) if cls is TransitionPlanner else {}
    planner = cls(limits(ns.topp_joint_velocity, ns.topp_gripper_velocity),
                  limits(ns.topp_joint_acceleration, ns.topp_gripper_acceleration),
                  fps=ns.fps, path_fps=ns.topp_path_fps, gridpoints=ns.topp_gridpoints, **extra)
    planner.diagnostic_dir = ns.out_dir / 'topp_diagnostics'
    options = dict(horizon=ns.action_horizon, prefetch_seconds=ns.topp_prefetch_seconds)
    if not rtc:
        return ToppPolicy(base_policy, planner, asynchronous=ns.use_async,
                          metrics_path=ns.out_dir/'topp_metrics.jsonl', **options)
    options.update(model_horizon=50, actions_during_latency=ns.actions_during_latency,
                   commit_seconds=ns.topp_commit_seconds, planning_margin=ns.topp_planning_margin)
    if backend == 'legacy':
        return ToppRtcPolicy(base_policy, planner, asynchronous=True, rolling=ns.topp_rolling,
                             metrics_path=ns.out_dir/'topp_rtc_metrics.jsonl', **options)
    return CommittedRtcPolicy(base_policy, planner, plan_budget=ns.topp_plan_budget,
                              soft_prefix_steps=ns.rtc_soft_prefix_steps,
                              training_rtc=ns.training_rtc,
                              metrics_path=ns.out_dir/'topp_rtc_committed_metrics.jsonl', **options)


def join_worker(policy, timeout=5):
    worker = getattr(policy, '_inference_thread', None) or getattr(policy, 'thread', None)
    if worker is not None:
        worker.join(timeout=timeout)
        if worker.is_alive():
            raise RuntimeError('Inference still running; refusing to reset into another episode')


def run_episodes(args, environment, policy, checked, recorder):
    for episode in range(args.num_episodes):
        environment.reset()
        policy.reset()
        recorder.event('episode_start', episode=episode)
        while not environment.is_episode_complete():
            start = time.monotonic()
            checked.raise_if_failed()
            observation = environment.get_observation()
            action = policy.infer(observation)
            checked.raise_if_failed()
            environment.apply_action(action)
            time.sleep(max(0, 1/args.fps-(time.monotonic()-start)))
        join_worker(policy)
        checked.raise_if_failed()
        recorder.event('episode_end', episode=episode)


def main(argv=None):
    ns = build_arg_parser().parse_args(argv)
    spec = load_deployment(ns.deployment_config)
    args = validate_args(ns, spec)
    if ns.check_only:
        if ns.use_topp:
            from topp_policy import load_toppra
            load_toppra()
        print(f"Verified {spec['config']}: absolute joint rad/m; fps={ns.fps}; "
              f"horizon={ns.action_horizon}; async={ns.use_async}; training_rtc={ns.training_rtc}; legacy_RTC={ns.use_rtc and not ns.training_rtc}; TOPP={ns.use_topp}")
        print(f"Training model horizon={spec['model_horizon']}; client horizon/latency/RTC are deployment settings, not training parameters")
        return 0
    ns.out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_'), dir=ns.out_dir))
    ns.out_dir = args.out_dir = root
    recorder = SessionRecorder(root/'client', dict(deployment=spec, args=vars(ns)))
    atomic_json(root/'deployment.json', spec)
    handler = logging.FileHandler(root/'client.log')
    logging.getLogger().addHandler(handler)
    print(f'Run recording: {root}', flush=True)
    connection = environment = policy = diagnostics = diagnostics_log = None
    reason, result = 'completed', 0
    def interrupted(*_):
        raise KeyboardInterrupt
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        connection = WebsocketClientPolicy(host=args.host, port=args.port)
        checked = CheckedJointPolicy(connection, spec)  # Fail before creating a ROS backend.
        recorder.event('server_verified', metadata=connection.get_server_metadata())
        policy = build_policy(ns, RecordedPolicy(checked, recorder))
        if ns.record_ros_diagnostics:
            diagnostics_log = (root/'diagnostics.log').open('w')
            diagnostics = subprocess.Popen([str(args.bridge_python), '-u',
                str(SERVER_REPO/'examples/piper_dual/record_ros_diagnostics.py'),
                '--output', str(root/'diagnostics'), '--config', str(args.ros2_config)],
                stdout=diagnostics_log, stderr=subprocess.STDOUT)
        environment = RecordedEnvironment(original._build_environment(args), recorder)
        logging.info('Joint deployment: %s; client fps=%s; training fps=%s; TOPP=%s',
                     spec['config'], args.fps, spec['model_fps'], ns.use_topp)
        run_episodes(args, environment, policy, checked, recorder)
    except KeyboardInterrupt:
        reason, result = 'interrupted', 130
    except Exception as exc:
        reason, result = repr(exc), 1
        logging.exception('Joint deployment stopped')
        recorder.event('fatal_error', error=repr(exc))
    finally:
        # Close the socket first to unblock an outstanding inference before resetting/closing recording.
        if connection is not None:
            try: connection._ws.close()
            except Exception as exc: recorder.event('cleanup_error', component='websocket', error=repr(exc))
        if policy is not None:
            policy._should_stop = True
            try:
                join_worker(policy)
                close = getattr(policy, 'close', None)
                if callable(close): close()
            except Exception as exc:
                recorder.event('cleanup_error', component='policy', error=repr(exc))
        if environment is not None:
            try: environment.close()
            except Exception as exc: recorder.event('cleanup_error', component='environment', error=repr(exc))
        if diagnostics is not None:
            if diagnostics.poll() is None: diagnostics.send_signal(signal.SIGINT)
            try: diagnostics.wait(timeout=8)
            except subprocess.TimeoutExpired: diagnostics.kill(); diagnostics.wait()
            recorder.event('diagnostics_exit', returncode=diagnostics.returncode)
        if diagnostics_log is not None: diagnostics_log.close()
        recorder.close(reason)
        logging.getLogger().removeHandler(handler)
        handler.close()
        for sig, old in previous.items(): signal.signal(sig, old)
    return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
