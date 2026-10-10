#!/usr/bin/env python3
"""pyAgxArm ROS 2 driver for the existing Piper local-ros topic contract."""
from __future__ import annotations

import argparse
import math
import signal
import time

from arm import Arm, DualController, JOINT_NAMES, SDK_COMMIT, error_flags
from eef_feedback import pose_message


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--left-can-port", default="can_left")
    result.add_argument("--right-can-port", default="can_right")
    result.add_argument("--model", choices=("piper", "piper_h", "piper_l", "piper_x"), default="piper")
    for side in ("left", "right"):
        result.add_argument(f"--{side}-firmware", required=True, choices=("default", "v183", "v188", "v189"))
    result.add_argument("--enable", action="store_true", help="Enable motors and accept joint commands")
    result.add_argument("--motion-mode", choices=("j", "js"), default="j",
                        help="Official SDK joint API: j=move_j, js=move_js (MIT fast response)")
    result.add_argument("--reset-stop", action="store_true",
                        help="Explicitly reset a reported electronic stop before enabling; requires --enable")
    result.add_argument("--feedback-timeout", type=float, default=0.5)
    result.add_argument("--startup-timeout", type=float, default=10.0)
    result.add_argument("--publish-hz", type=float, default=200.0)
    return result


def ensure_eef_publishers(node, arms, publishers, message_type):
    """Take over a missing feedback source, including after a relay exits."""
    for side in arms:
        topic = f"/puppet/end_pose_{side}"
        if side not in publishers and node.count_publishers(topic) == 0:
            publishers[side] = node.create_publisher(message_type, topic, 1)
            node.get_logger().info(f"Publishing EEF feedback: {topic}")


def enable_arms(arms, reset_stop, timeout):
    """Sequence asynchronous recovery using new motor feedback, not cached ACKs."""
    def wait_for(predicate, label):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{label}: timed out waiting for new motor feedback")
            time.sleep(0.01)

    for side, arm in arms.items():
        wait_for(lambda: arm.robot.get_arm_status() is not None, f"{side} status")
        if reset_stop and arm.robot.get_arm_status().msg.arm_status == 1:
            sent = time.time()
            arm.robot.reset()

            def recovered():
                status = arm.robot.get_arm_status()
                drivers = [arm.robot.get_driver_states(i) for i in range(1, 7)]
                return (status is not None and status.timestamp > sent
                        and status.msg.arm_status == 0 and status.msg.err_code == 0
                        and all(d is not None and d.timestamp > sent
                                and not d.msg.foc_status.driver_enable_status for d in drivers))

            wait_for(recovered, f"{side} reset completion")

        sent = time.time()

        def enabled():
            official_result = arm.robot.enable()
            drivers = [arm.robot.get_driver_states(i) for i in range(1, 7)]
            return official_result and all(
                d is not None and d.timestamp > sent and d.msg.foc_status.driver_enable_status
                for d in drivers)

        wait_for(enabled, f"{side} enable")


def main(argv=None):
    args = parser().parse_args(argv)
    if args.reset_stop and not args.enable:
        raise ValueError("--reset-stop requires --enable")
    for key in ("feedback_timeout", "startup_timeout", "publish_hz"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if args.left_can_port == args.right_can_port:
        raise ValueError("left and right CAN ports must differ")

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from piper_msgs.msg import PiperStatusMsg
    from std_srvs.srv import Trigger
    from geometry_msgs.msg import PoseStamped

    rclpy.init(args=[])
    node = Node("pyagxarm_dual_local_driver")
    arms = {}
    controller = DualController(arms, enabled=False)

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

        node.get_logger().info(f"pyAgxArm checkout {SDK_COMMIT}; enable={args.enable}; motion_mode={args.motion_mode}")
        for side in ("left", "right"):
            arms[side] = Arm.create(
                getattr(args, f"{side}_can_port"), getattr(args, f"{side}_firmware"),
                model=args.model, feedback_timeout=args.feedback_timeout, motion_mode=args.motion_mode,
            )

        if args.enable:
            enable_arms(arms, args.reset_stop, args.startup_timeout)
            controller.enabled = True

        feedback = {}
        echoes = {}
        statuses = {}
        eef_publishers = {}
        subscriptions = []

        def receive(side, message):
            try:
                if message.name and list(message.name) != list(JOINT_NAMES):
                    raise ValueError(f"wrong joint names/order: {message.name}")
                controller.apply(side, message.position, message.velocity, message.effort)
            except Exception as exc:
                node.get_logger().error(f"{side}: command rejected: {exc}; positions={list(message.position)}")

        for side in arms:
            feedback[side] = node.create_publisher(JointState, f"/puppet/joint_{side}", 1)
            echoes[side] = node.create_publisher(JointState, f"/master/joint_{side}", 1)
            statuses[side] = node.create_publisher(PiperStatusMsg, f"/piper_{side}_ctrl_node/arm_status", 1)
            topic = f"/puppet/end_pose_{side}"
            if not node.count_publishers(topic):
                eef_publishers[side] = node.create_publisher(PoseStamped, topic, 1)
            else:
                node.get_logger().info(f"{topic}: using existing EEF publisher")
            if args.enable:
                subscriptions.append(node.create_subscription(
                    JointState, f"/joint_{side}_states", lambda msg, side=side: receive(side, msg), 1,
                ))

        last_feedback_errors = {}
        next_eef_discovery = 0.0
        eef_errors = {}
        def publish():
            nonlocal last_feedback_errors, next_eef_discovery
            if time.monotonic() >= next_eef_discovery:
                ensure_eef_publishers(node, arms, eef_publishers, PoseStamped)
                next_eef_discovery = time.monotonic() + 1.0
            snapshots = controller.poll()
            if controller.feedback_errors != last_feedback_errors:
                for side, reason in controller.feedback_errors.items():
                    node.get_logger().error(f"{side}: feedback unavailable: {reason}")
                for side in last_feedback_errors.keys() - controller.feedback_errors.keys():
                    node.get_logger().info(f"{side}: feedback restored")
                last_feedback_errors = dict(controller.feedback_errors)
            stamp = node.get_clock().now().to_msg()
            for side, publisher in eef_publishers.items():
                try:
                    publisher.publish(pose_message(arms[side], stamp))
                    if side in eef_errors:
                        node.get_logger().info(f"{side}: EEF feedback restored")
                        eef_errors.pop(side)
                except RuntimeError as exc:
                    if side not in eef_errors:
                        node.get_logger().warning(f"{side}: EEF feedback unavailable: {exc}")
                    eef_errors[side] = str(exc)
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
        node.get_logger().error(str(exc))
        return 1
    finally:
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
