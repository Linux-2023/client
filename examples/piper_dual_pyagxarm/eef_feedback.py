#!/usr/bin/python3
"""Official flange feedback -> legacy ROS PoseStamped; optional read-only relay."""
import math
import time
from arm import Arm, finite_vector


def pose_message(arm, stamp):
    from geometry_msgs.msg import PoseStamped
    now = time.time()
    for pair in ('end_pose_xy', 'end_pose_zrx', 'end_pose_ryrz'):
        arm.fresh(getattr(arm.robot._parser, pair, None), pair, now)
    values = finite_vector(arm.fresh(arm.robot.get_flange_pose(), 'flange feedback', now), 6)
    x, y, z, roll, pitch, yaw = values
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = 'base_link'
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = x, y, z
    q = msg.pose.orientation
    q.x, q.y = sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy
    q.z, q.w = cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy
    return msg


def main():
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    rclpy.init()
    node = Node('pyagxarm_eef_readonly_relay')
    arms = {}
    try:
        until = time.monotonic()+1
        while time.monotonic()<until:
            rclpy.spin_once(node, timeout_sec=0.05)
        for side in ('left', 'right'):
            topic = f'/puppet/end_pose_{side}'
            if node.count_publishers(topic):
                raise RuntimeError(f'{topic} already has a publisher')
            arms[side] = Arm.create(f'can_{side}', 'v189')
        publishers = {side: node.create_publisher(PoseStamped, f'/puppet/end_pose_{side}', 1) for side in arms}
        errors = {}
        def publish():
            stamp = node.get_clock().now().to_msg()
            for side, arm in arms.items():
                try:
                    publishers[side].publish(pose_message(arm, stamp))
                    errors.pop(side, None)
                except RuntimeError as exc:
                    if side not in errors:
                        node.get_logger().warning(f'{side}: {exc}')
                    errors[side] = str(exc)
        timer = node.create_timer(0.005, publish)
        node.get_logger().info('Read-only EEF relay active; no enable, reset or motion commands')
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for arm in arms.values():
            arm.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
