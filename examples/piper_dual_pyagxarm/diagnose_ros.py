#!/usr/bin/python3
"""Read-only Piper ROS status/feedback/parameter snapshot; never publishes commands."""
import json
import time

import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters
from sensor_msgs.msg import JointState
from piper_msgs.msg import PiperStatusMsg


def main():
    rclpy.init()
    node = Node("piper_readonly_diagnostic")
    samples = {}

    def receive(topic, msg):
        row = samples.setdefault(topic, {"count": 0})
        row["count"] += 1
        row["received_at"] = time.monotonic()
        if isinstance(msg, JointState):
            row.setdefault("first_position", list(msg.position))
            row["last_position"] = list(msg.position)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            row.setdefault("first_stamp", stamp)
            row["last_stamp"] = stamp
        else:
            row["status"] = {key: getattr(msg, key) for key in (
                "ctrl_mode", "arm_status", "mode_feedback", "teach_status", "motion_status", "err_code")}

    try:
        for side in ("left", "right"):
            for topic, kind in ((f"/puppet/joint_{side}", JointState),
                                (f"/master/joint_{side}", JointState),
                                (f"/joint_{side}_states", JointState),
                                (f"/piper_{side}_ctrl_node/arm_status", PiperStatusMsg)):
                node.create_subscription(kind, topic, lambda msg, topic=topic: receive(topic, msg), 10)
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        parameters = {}
        for side in ("left", "right"):
            client = node.create_client(GetParameters, f"/piper_{side}_ctrl_node/get_parameters")
            try:
                if not client.wait_for_service(timeout_sec=1):
                    parameters[side] = "parameter service missing"
                    continue
                request = GetParameters.Request()
                request.names = ["can_port", "auto_enable", "control_mode", "joint_move_mode",
                                 "joint_is_mit_mode", "joint_speed_override"]
                future = client.call_async(request)
                rclpy.spin_until_future_complete(node, future, timeout_sec=2)
                if not future.done() or future.result() is None:
                    parameters[side] = "parameter query timeout"
                    continue
                parameters[side] = {}
                for key, value in zip(request.names, future.result().values):
                    parameters[side][key] = (value.bool_value if value.type == 1 else
                                             value.integer_value if value.type == 2 else value.string_value)
            finally:
                node.destroy_client(client)
        print(json.dumps({"parameters": parameters, "samples": samples}, indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
