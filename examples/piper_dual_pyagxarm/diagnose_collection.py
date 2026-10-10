#!/usr/bin/python3
"""Read-only collector input counts and timestamp alignment check."""
import json
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped

rclpy.init()
node = Node('piper_collection_readonly_check')
topics = {f'/camera_{name}/color/image_raw': Image for name in ('f_l','l','r')}
topics.update({f'/{kind}/joint_{side}': JointState for kind in ('puppet','master') for side in ('left','right')})
topics.update({f'/puppet/end_pose_{side}': PoseStamped for side in ('left','right')})
stamps = {topic: [] for topic in topics}
try:
    for topic, kind in topics.items():
        def receive(msg, topic=topic):
            stamps[topic].append(msg.header.stamp.sec+msg.header.stamp.nanosec/1e9)
        node.create_subscription(kind, topic, receive, qos_profile_sensor_data)
    deadline = time.monotonic()+4
    while time.monotonic()<deadline:
        rclpy.spin_once(node, timeout_sec=.05)
    print(json.dumps({topic: {'messages': len(values), 'publishers': node.count_publishers(topic)} for topic, values in stamps.items()}, indent=2))
    errors=[]
    if all(stamps.values()):
        for t in stamps['/camera_f_l/color/image_raw']:
            near=[min(values,key=lambda s:abs(s-t)) for values in stamps.values()]
            errors.append((max(near)-min(near))*1000)
    print('candidate frames within 30ms:', sum(e<=30 for e in errors), '/', len(errors))
finally:
    node.destroy_node()
    rclpy.shutdown()
