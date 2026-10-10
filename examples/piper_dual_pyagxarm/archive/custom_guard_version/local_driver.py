#!/usr/bin/env python3
"""pyAgxArm ROS 2 driver for the existing Piper local-ros topic contract."""
from __future__ import annotations

import argparse
import math
import signal
import time

from arm import Arm, DualController, JOINT_NAMES, SDK_COMMIT, error_flags


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--left-can-port", default="can_left")
    result.add_argument("--right-can-port", default="can_right")
    result.add_argument("--model", choices=("piper", "piper_h", "piper_l", "piper_x"), default="piper")
    for side in ("left", "right"):
        result.add_argument(f"--{side}-firmware", required=True, choices=("default", "v183", "v188", "v189"))
    result.add_argument("--enable", action="store_true", help="Enable motors and accept joint commands")
    result.add_argument("--reset-stop", action="store_true",
                        help="Explicitly reset a reported electronic stop before enabling; requires --enable")
    result.add_argument("--feedback-timeout", type=float, default=0.5)
    result.add_argument("--command-timeout", type=float, default=5.0)
    result.add_argument("--startup-timeout", type=float, default=10.0)
    result.add_argument("--gripper-max", type=float, default=None,
                        help="Optional extra width limit in meters; default follows official SDK")
    result.add_argument("--publish-hz", type=float, default=200.0)
    return result


def reset_startup_stop(arms, controller, requested, reset_sent):
    """Validate both statuses before any explicit stop recovery transmission."""
    stopped = []
    for side, arm in arms.items():
        status = arm.fresh(arm.robot.get_arm_status(), f"{side} startup status", time.time())
        if status.err_code or status.arm_status not in (0, 1):
            raise ValueError(f"{side}: startup fault arm_status={status.arm_status}, err_code={status.err_code}; reset refused")
        if status.arm_status == 1:
            if not requested:
                raise ValueError(f"{side}: electronic stop (arm_status=1). "
                                 "After checking the robot, restart with --enable --reset-stop to recover explicitly")
            # motion_status=1 means target not reached, not measured velocity.
            # It commonly remains set after stopping an unfinished trajectory.
            # An electronic stop can retain CAN control mode (1).
            # Neither mode 0 nor 1 is a velocity measurement.
            if status.ctrl_mode not in (0, 1) or status.teach_status != 0:
                raise ValueError(f"{side}: reset refused: ctrl_mode={status.ctrl_mode}, "
                                 f"teach_status={status.teach_status}, arm_status={status.arm_status}, "
                                 f"motion_status={status.motion_status}, err_code={status.err_code}; "
                                 "expected electronic stop in standby/CAN mode with teaching inactive")
            if side not in reset_sent:
                stopped.append(side)
    for side in stopped:
        # From this point this process changes controller state, so cleanup must stop it.
        controller.enabled = True
        reset_sent.add(side)
        arms[side].robot.reset()


def wait_stable_feedback(arms, timeout, *, stable_seconds=1.0):
    """Observe a continuous healthy window without sending any commands."""
    deadline = time.monotonic() + timeout
    healthy_since = None
    last_error = "waiting for feedback"
    while time.monotonic() < deadline:
        try:
            for arm in arms.values():
                arm.read()
        except RuntimeError as exc:
            healthy_since = None
            last_error = str(exc)
        else:
            now = time.monotonic()
            if healthy_since is None:
                healthy_since = now
            if now - healthy_since >= stable_seconds:
                return
        time.sleep(0.02)
    raise RuntimeError(f"startup feedback did not stabilize: {last_error}")


def main(argv=None):
    args = parser().parse_args(argv)
    if args.reset_stop and not args.enable:
        raise ValueError("--reset-stop requires --enable")
    for key in ("feedback_timeout", "command_timeout", "startup_timeout", "gripper_max", "publish_hz"):
        if key == "gripper_max" and args.gripper_max is None:
            continue
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if args.left_can_port == args.right_can_port:
        raise ValueError("left and right CAN ports must differ")

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from piper_msgs.msg import PiperStatusMsg
    from std_srvs.srv import Trigger

    rclpy.init(args=[])
    node = Node("pyagxarm_dual_local_driver")
    arms = {}
    controller = DualController(arms, enabled=False, command_timeout=args.command_timeout)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        # Discover existing command consumers before opening CAN or enabling motors.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        for side in ("left", "right"):
            topic = f"/joint_{side}_states"
            if node.count_subscribers(topic):
                raise RuntimeError(f"{topic} already has a driver; stop the old arm driver first")

        node.get_logger().info(f"pyAgxArm checkout {SDK_COMMIT}; enable={args.enable}")
        for side in ("left", "right"):
            arms[side] = Arm.create(
                getattr(args, f"{side}_can_port"), getattr(args, f"{side}_firmware"),
                model=args.model, feedback_timeout=args.feedback_timeout, gripper_max=args.gripper_max,
            )

        # No home/zero/gripper command on startup. Only enable and select MOVE J when requested.
        deadline = time.monotonic() + args.startup_timeout
        reset_sent = set()
        while True:
            try:
                reset_startup_stop(arms, controller, args.reset_stop, reset_sent)
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)
        # Reset acknowledgement can precede motor-controller recovery. Do not
        # enable from a single cached healthy sample during that transition.
        node.get_logger().info("Waiting for 1s of healthy feedback before enabling")
        wait_stable_feedback(arms, args.startup_timeout)
        if args.enable:
            controller.enabled = True
            deadline = time.monotonic() + args.startup_timeout
            last_errors = {}
            while True:
                ready = True
                for side, arm in arms.items():
                    enabled = arm.robot.enable()
                    try:
                        # Official enable() confirms motors, not a motion mode.
                        state, status = arm.read()
                        if not enabled:
                            raise RuntimeError(f"motors={arm.robot.get_joints_enable_status_list()}, status={status}")
                        last_errors.pop(side, None)
                    except RuntimeError as exc:
                        ready = False
                        last_errors[side] = str(exc)
                if ready:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"official enable() did not confirm healthy enabled motors: {last_errors}")
                time.sleep(0.02)

        feedback = {}
        echoes = {}
        statuses = {}
        subscriptions = []

        def receive(side, message):
            if message.name and list(message.name) != list(JOINT_NAMES):
                controller.stop(f"{side}: wrong joint names/order: {message.name}")
                raise RuntimeError(controller.fault)
            controller.apply(side, message.position, message.velocity, message.effort)

        for side in arms:
            feedback[side] = node.create_publisher(JointState, f"/puppet/joint_{side}", 1)
            echoes[side] = node.create_publisher(JointState, f"/master/joint_{side}", 1)
            statuses[side] = node.create_publisher(PiperStatusMsg, f"/piper_{side}_ctrl_node/arm_status", 1)
            if args.enable:
                subscriptions.append(node.create_subscription(
                    JointState, f"/joint_{side}_states", lambda msg, side=side: receive(side, msg), 1,
                ))

        idle_sides = set()

        def publish():
            nonlocal idle_sides
            for side in arms:
                if node.count_subscribers(f"/joint_{side}_states") > int(args.enable):
                    controller.stop(f"{side}: another command consumer appeared")
                    raise RuntimeError(controller.fault)
            snapshots = controller.poll()
            standby = {side for side, (_, fields) in snapshots.items()
                       if args.enable and fields["ctrl_mode"] == 0}
            if standby != idle_sides:
                if standby:
                    node.get_logger().info(
                        f"Waiting for first action: {sorted(standby)} in standby; "
                        "MOVE J will be requested and confirmed before sending the first target"
                    )
                idle_sides = standby
            stamp = node.get_clock().now().to_msg()
            for side, (state, fields) in snapshots.items():
                for publisher, values in ((feedback[side], state),
                                          (echoes[side], arms[side].command or state)):
                    msg = JointState()
                    msg.header.stamp = stamp
                    msg.name = list(JOINT_NAMES)
                    msg.position = values
                    publisher.publish(msg)
                status = PiperStatusMsg()
                for key, value in fields.items():
                    setattr(status, key, value)
                for key, value in error_flags(fields["err_code"]).items():
                    setattr(status, key, value)
                statuses[side].publish(status)

        def finish_zero(request, response):
            try:
                controller.finish_zero()
                response.success = True
                response.message = "Zero complete; driver remains enabled and monitored, waiting for next task"
                node.get_logger().info(response.message)
            except RuntimeError as exc:
                response.success = False
                response.message = str(exc)
            return response

        completion_service = node.create_service(Trigger, "/pyagxarm_dual/finish_zero", finish_zero)
        timer = node.create_timer(1.0 / args.publish_hz, publish)
        node.get_logger().info("Ready: local-ros topics available; cameras remain in their existing ROS nodes")
        rclpy.spin(node)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        controller.stop(str(exc))
        node.get_logger().error(str(exc))
        return 1
    finally:
        controller.stop("driver shutdown")
        if controller.enabled and arms:
            node.get_logger().warning(f"Both-arm stop requested: {controller.fault}")
        for side, arm in arms.items():
            try:
                arm.close()
            except Exception as exc:
                node.get_logger().error(f"{side} disconnect failed: {exc}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
