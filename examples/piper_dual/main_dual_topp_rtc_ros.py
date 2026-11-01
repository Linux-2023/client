#!/usr/bin/env python3
"""Experimental RTC + TOPP-RA entrypoint; existing entrypoints are unchanged."""
from __future__ import annotations
import argparse
import logging
import signal
import sys

import main_dual_ros as original
from topp_rtc_policy import Planner, ToppRtcPolicy, limits
from topp_committed_rtc_policy import CommittedRtcPolicy
from topp_transition_planner import TransitionPlanner


def build_arg_parser():
    parser = original.build_arg_parser()
    parser.description = __doc__
    parser.set_defaults(use_rtc=True)
    parser.add_argument('--topp-rtc-backend', choices=('committed', 'legacy'), default='committed',
                        help='Committed prefix with background planning, or previous implementation')
    parser.add_argument('--topp-transition', action=argparse.BooleanOptionalAction, default=True,
                        help='Committed backend: allow a bounded velocity-matching transition')
    parser.add_argument('--topp-transition-joint-deviation', type=float, default=.03)
    parser.add_argument('--topp-transition-gripper-deviation', type=float, default=.002)
    parser.add_argument('--topp-transition-max-seconds', type=float, default=.6)
    parser.add_argument('--topp-plan-budget', type=float, default=.2,
                        help='Initial planning time reserved in the committed prefix, seconds')
    parser.add_argument('--rtc-soft-prefix-steps', type=int, default=3,
                        help='Soft RTC transition points after the immutable prefix')
    parser.add_argument('--topp-rolling', action=argparse.BooleanOptionalAction, default=True,
                        help='Legacy backend only: use full-chunk rolling planning')
    parser.add_argument('--topp-commit-seconds', type=float, default=.1)
    parser.add_argument('--topp-planning-margin', type=float, default=.05)
    parser.add_argument("--rtc-model-horizon", type=int, default=50, help="Must match server action horizon")
    parser.add_argument('--topp-joint-velocity', type=float, default=4, help='Joint velocity bound, rad/s')
    parser.add_argument('--topp-joint-acceleration', type=float, default=2.0, help='Joint acceleration bound, rad/s^2')
    parser.add_argument('--topp-gripper-velocity', type=float, default=0.05, help='Gripper velocity bound, m/s')
    parser.add_argument('--topp-gripper-acceleration', type=float, default=0.2, help='Gripper acceleration bound, m/s^2')
    parser.add_argument('--topp-prefetch-seconds', type=float, default=0.4,
                        help='Request the next chunk with this much planned motion remaining')
    parser.add_argument('--topp-path-fps', type=float, default=None,
                        help='Model knot frequency: 30 Hz for Training-RTC, otherwise the legacy 15 Hz default')
    parser.add_argument('--topp-gridpoints', type=int, default=200)
    return parser


def main(argv=None):
    ns = build_arg_parser().parse_args(argv)
    if ns.topp_path_fps is None:
        from openpi_client.action_chunk_broker import TRAINING_RTC_MODEL_FPS
        ns.topp_path_fps = TRAINING_RTC_MODEL_FPS if ns.training_rtc else 15
    args = original.Args(**{key: getattr(ns, key) for key in original.Args.__dataclass_fields__})
    environment = policy = runtime = None
    try:
        original._validate_ros2_contract(args)
        if args.training_rtc:
            ns.topp_rtc_backend = 'committed'
        if not (args.training_rtc or args.use_rtc) or not args.use_async:
            raise ValueError('This entrypoint requires RTC and --use-async; use main_dual_topp_ros.py for non-RTC.')
        if args.training_rtc and not 0 <= args.actions_during_latency <= 15:
            raise ValueError('Training-RTC prefix length must be between 0 and 15')
        if not 0 < args.action_horizon < ns.rtc_model_horizon or not 0 <= args.actions_during_latency < ns.rtc_model_horizon-args.action_horizon:
            raise ValueError('Require 0 < action horizon < model horizon and 0 <= delay < H-s')
        if args.fps <= 0 or args.num_steps <= 0 or args.num_episodes <= 0:
            raise ValueError('fps, num-steps and num-episodes must be positive')
        planner_class = TransitionPlanner if ns.topp_rtc_backend == 'committed' and ns.topp_transition else Planner
        transition_options = dict(joint_deviation=ns.topp_transition_joint_deviation,
                                  gripper_deviation=ns.topp_transition_gripper_deviation,
                                  max_blend_seconds=ns.topp_transition_max_seconds) if planner_class is TransitionPlanner else {}
        planner = planner_class(limits(ns.topp_joint_velocity, ns.topp_gripper_velocity),
                          limits(ns.topp_joint_acceleration, ns.topp_gripper_acceleration),
                          fps=args.fps, gridpoints=ns.topp_gridpoints, path_fps=ns.topp_path_fps, **transition_options)
        planner.diagnostic_dir = args.out_dir/'topp_diagnostics'
        if args.action_horizon < 1 or ns.topp_prefetch_seconds < 0:
            raise ValueError('Invalid action horizon or prefetch time')
        print(original.contract_summary(args))
        print('TOPP-RA: continuous planned position/velocity handoff; acceleration/jerk continuity not guaranteed.')
        print(f'Output={args.fps} Hz; joint v/a={ns.topp_joint_velocity}/{ns.topp_joint_acceleration}; '
              f'gripper v/a={ns.topp_gripper_velocity}/{ns.topp_gripper_acceleration}; RTC enabled (time-resampled prefix)')
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        base_policy = WebsocketClientPolicy(host=args.host, port=args.port)
        if ns.topp_rtc_backend == 'committed':
            policy = CommittedRtcPolicy(base_policy, planner, horizon=args.action_horizon,
                            prefetch_seconds=ns.topp_prefetch_seconds,
                            metrics_path=args.out_dir/'topp_rtc_committed_metrics.jsonl',
                            model_horizon=ns.rtc_model_horizon,
                            actions_during_latency=args.actions_during_latency,
                            training_rtc=args.training_rtc,
                            commit_seconds=ns.topp_commit_seconds, planning_margin=ns.topp_planning_margin,
                            plan_budget=ns.topp_plan_budget, soft_prefix_steps=ns.rtc_soft_prefix_steps)
            print('RTC committed prefix: background TOPP; exact scheduled handoff; short planning window')
            print(f'Bounded velocity transition={ns.topp_transition}; '
                  f'extra deviation caps: joint={ns.topp_transition_joint_deviation} rad, '
                  f'gripper={ns.topp_transition_gripper_deviation} m')
        else:
            policy = ToppRtcPolicy(base_policy, planner, horizon=args.action_horizon,
                            prefetch_seconds=ns.topp_prefetch_seconds, asynchronous=args.use_async,
                            metrics_path=args.out_dir/'topp_rtc_metrics.jsonl',
                            model_horizon=ns.rtc_model_horizon,
                            actions_during_latency=args.actions_during_latency, rolling=ns.topp_rolling,
                            commit_seconds=ns.topp_commit_seconds, planning_margin=ns.topp_planning_margin)
        environment = original._build_environment(args)
        runtime = original._build_runtime(args, environment, policy)
        def interrupted(signum, frame):
            raise KeyboardInterrupt
        previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            runtime.run()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logging.exception('TOPP deployment failed: %s', exc)
        return 1
    finally:
        if policy is not None:
            policy.close()
        if environment is not None:
            environment.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
