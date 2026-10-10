#!/usr/bin/env python3
"""Experimental TOPP-RA entrypoint; original main_dual_ros.py is unchanged."""
from __future__ import annotations
import argparse
import logging
import signal
import sys

import main_dual_ros as original
from topp_policy import Planner, ToppPolicy, limits


def build_arg_parser():
    parser = original.build_arg_parser()
    parser.description = __doc__
    parser.set_defaults(use_rtc=False)
    parser.add_argument('--topp-joint-velocity', type=float, default=4, help='Joint velocity bound, rad/s')
    parser.add_argument('--topp-joint-acceleration', type=float, default=1.0, help='Joint acceleration bound, rad/s^2')
    parser.add_argument('--topp-gripper-velocity', type=float, default=0.05, help='Gripper velocity bound, m/s')
    parser.add_argument('--topp-gripper-acceleration', type=float, default=0.2, help='Gripper acceleration bound, m/s^2')
    parser.add_argument('--topp-prefetch-seconds', type=float, default=0.4,
                        help='Request the next chunk with this much planned motion remaining')
    parser.add_argument('--topp-path-fps', type=float, default=30,
                        help='Nominal path knot frequency, independent of output --fps')
    parser.add_argument('--topp-gridpoints', type=int, default=200)
    return parser


def main(argv=None):
    ns = build_arg_parser().parse_args(argv)
    args = original.Args(**{key: getattr(ns, key) for key in original.Args.__dataclass_fields__})
    environment = policy = runtime = None
    try:
        original._validate_ros2_contract(args)
        if args.use_rtc:
            raise ValueError('TOPP changes chunk timing; RTC index semantics are unsupported. Use --no-use-rtc.')
        if args.fps <= 0 or args.num_steps <= 0 or args.num_episodes <= 0:
            raise ValueError('fps, num-steps and num-episodes must be positive')
        planner = Planner(limits(ns.topp_joint_velocity, ns.topp_gripper_velocity),
                          limits(ns.topp_joint_acceleration, ns.topp_gripper_acceleration),
                          fps=args.fps, gridpoints=ns.topp_gridpoints, path_fps=ns.topp_path_fps)
        if args.action_horizon < 1 or ns.topp_prefetch_seconds < 0:
            raise ValueError('Invalid action horizon or prefetch time')
        print(original.contract_summary(args))
        print('TOPP-RA: continuous planned position/velocity handoff; acceleration/jerk continuity not guaranteed.')
        print(f'Output={args.fps} Hz; joint v/a={ns.topp_joint_velocity}/{ns.topp_joint_acceleration}; '
              f'gripper v/a={ns.topp_gripper_velocity}/{ns.topp_gripper_acceleration}; RTC disabled')
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        base_policy = WebsocketClientPolicy(host=args.host, port=args.port)
        policy = ToppPolicy(base_policy, planner, horizon=args.action_horizon,
                            prefetch_seconds=ns.topp_prefetch_seconds, asynchronous=args.use_async,
                            metrics_path=args.out_dir/'topp_metrics.jsonl')
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
