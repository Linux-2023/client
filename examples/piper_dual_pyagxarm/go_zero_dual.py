#!/usr/bin/python3
"""ROS graph based return-to-zero for legacy Piper and pyAgxArm drivers."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

ARMS = ("left", "right")
CTRL_NODE = {"left": "piper_left_ctrl_node", "right": "piper_right_ctrl_node"}
JOINT_TOL = 0.02      # rad
GRIPPER_TOL = 0.002   # m


class GoZero(Node):
    def __init__(self, arms, rate_hz: float) -> None:
        super().__init__("go_zero_dual")
        self.arms = arms
        # 注意: 不能叫 self.publishers（rclpy Node 的只读属性）也不能叫 self._publishers
        # （那是 rclpy Node 内部的注册表，覆盖它会破坏节点销毁）
        self._arm_state: dict[str, list[float]] = {}
        self._received_at = {}
        self._arm_publishers = {
            arm: self.create_publisher(JointState, f"/joint_{arm}_states", 1) for arm in arms
        }
        for arm in arms:
            self.create_subscription(
                JointState,
                f"/puppet/joint_{arm}",
                lambda msg, a=arm: self.receive_state(a, msg),
                1,
            )
        self.rate_hz = rate_hz

    def receive_state(self, arm, msg):
        values = [float(v) for v in msg.position]
        if len(values) != 7 or not all(math.isfinite(v) for v in values):
            self._arm_state.pop(arm, None)
            self._received_at.pop(arm, None)
            return
        self._arm_state[arm] = values
        self._received_at[arm] = time.monotonic()

    def fresh(self):
        now = time.monotonic()
        return all(arm in self._arm_state and now - self._received_at[arm] <= 0.5 for arm in self.arms)

    def publish_targets(self, targets: dict[str, list[float]], speed_percent: int, gripper_effort: float) -> None:
        if not self.fresh():
            raise RuntimeError("关节反馈缺失或超过 0.5s，停止发布")
        assert_command_sources(self, self.arms)
        for arm in self.arms:
            values = targets.get(arm)
            if values is None:
                continue
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
            msg.position = list(values)
            msg.velocity = [0.0] * 6 + [float(speed_percent)]
            msg.effort = [0.0] * 6 + [float(gripper_effort)]
            self._arm_publishers[arm].publish(msg)

    def publish_zero(self, speed_percent: int, gripper_effort: float) -> None:
        self.publish_targets({arm: [0.0] * 7 for arm in self.arms}, speed_percent, gripper_effort)

    def ramp_to_zero(
        self,
        starts: dict[str, list[float]],
        seconds: float,
        speed_percent: int,
        gripper_effort: float,
        period: float,
    ) -> None:
        """从 starts（当前位姿）用余弦起停平滑走到零位，按 period 节拍发布。

        每步增量 = 总行程 × Δα，因此机械臂看到的是连续小目标，而不是一次大跳变。
        """
        steps = max(1, int(round(seconds / period)))
        for arm, values in starts.items():
            travel = max(abs(v) for v in values)
            print(
                f"  斜坡 {arm}: max|q|={travel:.3f} → 0 | {seconds:g}s × {1 / period:g}Hz = {steps} 步，"
                f"每步最大 ≈ {travel / steps:.4f} rad"
            )
        for k in range(1, steps + 1):
            u = k / steps
            alpha = 0.5 - 0.5 * math.cos(math.pi * u)  # 余弦起停：两端慢、中间快
            targets = {arm: [v * (1.0 - alpha) for v in values] for arm, values in starts.items()}
            self.publish_targets(targets, speed_percent, gripper_effort)
            spin_period(self, period)
            if k % max(1, steps // 6) == 0:
                self.report(f"斜坡 {u * 100:3.0f}%")

    def converged(self) -> bool:
        if not self.fresh():
            return False
        for arm in self.arms:
            values = self._arm_state.get(arm)
            if values is None or len(values) < 7:
                return False
            if max(abs(v) for v in values[:6]) > JOINT_TOL or abs(values[6]) > GRIPPER_TOL:
                return False
        return True

    def report(self, prefix: str) -> None:
        for arm in self.arms:
            values = self._arm_state.get(arm)
            if values is None:
                print(f"  [{prefix}] {arm}: 未收到 /puppet/joint_{arm} 反馈")
                continue
            worst = max(abs(v) for v in values[:6])
            print(
                f"  [{prefix}] {arm}: " + " ".join(f"{v:+.3f}" for v in values[:6])
                + f" | 夹爪 {values[6]:.3f} m | max|q| {worst:.4f} rad"
            )


def spin_period(node, period):
    # spin_once can return immediately on a feedback callback; keep wall-clock pacing.
    deadline = time.monotonic() + period
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=max(0.0, deadline - time.monotonic()))


def identify_driver(arm, endpoints):
    if len(endpoints) != 1:
        raise RuntimeError(f"{arm}: /joint_{arm}_states 需要唯一订阅者，实际 {len(endpoints)}；"
                           "请启动驱动（pyAgxArm 需 --enable），并确认 ROS_DOMAIN_ID 一致")
    endpoint = endpoints[0]
    if endpoint.topic_type != "sensor_msgs/msg/JointState":
        raise RuntimeError(f"{arm}: 命令话题类型错误: {endpoint.topic_type}")
    if endpoint.node_name == "pyagxarm_dual_local_driver":
        return "pyagxarm", endpoint
    if endpoint.node_name == CTRL_NODE[arm]:
        return "legacy", endpoint
    raise RuntimeError(f"{arm}: 不支持的控制节点 {endpoint.node_name}")


def legacy_mode(node, endpoint):
    from rcl_interfaces.srv import GetParameters
    namespace = endpoint.node_namespace.rstrip("/")
    client = node.create_client(GetParameters, f"{namespace}/{endpoint.node_name}/get_parameters")
    try:
        if not client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError("无法查询旧驱动 control_mode 参数")
        request = GetParameters.Request()
        request.names = ["control_mode", "joint_is_mit_mode", "joint_move_mode", "joint_speed_override"]
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=2.0)
        if not future.done() or future.result() is None or not future.result().values:
            raise RuntimeError("旧驱动 control_mode 查询超时或返回空结果")
        values = future.result().values
        if len(values) != len(request.names) or values[0].type != 4 or any(v.type != 2 for v in values[1:]):
            raise RuntimeError("旧驱动缺少完整关节模式参数，不能确认回零配置")
        settings = {"control_mode": values[0].string_value}
        settings.update({key: value.integer_value for key, value in zip(request.names[1:], values[1:])})
        validate_legacy_settings(endpoint, settings)
        return settings["control_mode"]
    finally:
        node.destroy_client(client)


def validate_legacy_settings(endpoint, settings):
    name = f"{endpoint.node_namespace.rstrip('/')}/{endpoint.node_name}"
    problems = []
    if settings["joint_is_mit_mode"] != 0:
        problems.append(f"joint_is_mit_mode={settings['joint_is_mit_mode']}，需要位置速度模式 0；"
                        f"修正：ros2 param set {name} joint_is_mit_mode 0")
    if settings["joint_move_mode"] != 1:
        problems.append(f"joint_move_mode={settings['joint_move_mode']}，需要 MOVE J 1；"
                        f"修正：ros2 param set {name} joint_move_mode 1")
    speed = settings["joint_speed_override"]
    if speed == 0 or speed < -1 or speed > 100:
        problems.append(f"joint_speed_override={speed}，会阻止正常回零；"
                        f"修正：ros2 param set {name} joint_speed_override -1")
    if problems:
        raise RuntimeError(f"{name}: " + "；".join(problems))


def assert_command_sources(node, arms):
    """Allow the installed event-driven IK only while all PosCmd inputs are idle."""
    for arm in arms:
        publishers = node.get_publishers_info_by_topic(f"/joint_{arm}_states")
        own = [entry for entry in publishers if entry.node_name == node.get_name()
               and entry.node_namespace == node.get_namespace()]
        if len(own) != 1:
            raise RuntimeError(f"{arm}: 本工具发布节点尚未发现或存在同名回零进程")
        for entry in publishers:
            if entry is own[0]:
                continue
            name = f"{entry.node_namespace.rstrip('/')}/{entry.node_name}"
            if entry.node_name == "piper_pinocchio_ik":
                subscriptions = node.get_subscriber_names_and_types_by_node(entry.node_name, entry.node_namespace)
                inputs = [topic for topic, types in subscriptions if "piper_msgs/msg/PosCmd" in types]
                if inputs and all(node.count_publishers(topic) == 0 for topic in inputs):
                    continue
                raise RuntimeError(f"{arm}: IK 节点 {name} 的末端输入有发布者或无法确认输入；"
                                   f"请停止末端控制，输入话题={inputs}")
            raise RuntimeError(f"{arm}: 存在其他动作发布者 {name}，请先停止推理/遥操作")


def check_drivers(node, arms, allow_non_joint=False, discovered=None):
    kinds = []
    for arm in arms:
        endpoints = (discovered[arm] if discovered is not None
                     else node.get_subscriptions_info_by_topic(f"/joint_{arm}_states"))
        kind, endpoint = identify_driver(arm, endpoints)
        mode = "joint" if kind == "pyagxarm" else legacy_mode(node, endpoint)
        print(f"  {arm}: /joint_{arm}_states 订阅者=1 节点={endpoint.node_name} control_mode={mode}")
        if mode != "joint" and not allow_non_joint:
            raise RuntimeError(f"{arm}: control_mode={mode!r}，请先切回 joint")
        kinds.append(kind)
    assert_command_sources(node, arms)
    if len(set(kinds)) != 1:
        raise RuntimeError("两臂控制驱动类型不一致")
    return kinds[0]


def endpoint_metadata_ready(entry):
    return (bool(entry.node_name) and entry.node_name != "_NODE_NAME_UNKNOWN_"
            and bool(entry.node_namespace) and entry.node_namespace.startswith("/")
            and bool(entry.topic_type) and entry.topic_type != "_TOPIC_TYPE_UNKNOWN_")


def finish_zero_task(node):
    from std_srvs.srv import Trigger
    client = node.create_client(Trigger, "/pyagxarm_dual/finish_zero")
    try:
        if not client.wait_for_service(timeout_sec=1.0):
            raise RuntimeError("驱动不支持回零完成通知；请更新并重启驱动")
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=1.0)
        if not future.done() or future.result() is None:
            raise RuntimeError("回零完成通知超时")
        response = future.result()
        if not response.success:
            raise RuntimeError(f"驱动拒绝结束回零任务：{response.message}")
    finally:
        node.destroy_client(client)


def wait_for_drivers(node, arms, timeout):
    """Wait for DDS discovery instead of treating an initial empty graph as failure."""
    deadline = time.monotonic() + timeout
    while True:
        ready = True
        discovered = {}
        for arm in arms:
            topic = f"/joint_{arm}_states"
            endpoints = node.get_subscriptions_info_by_topic(topic)
            discovered[arm] = endpoints
            if len(endpoints) > 1:
                identify_driver(arm, endpoints)  # Never wait away an ambiguous driver.
            if not endpoints or not all(endpoint_metadata_ready(entry) for entry in endpoints):
                ready = False
            else:
                identify_driver(arm, endpoints)
            publishers = node.get_publishers_info_by_topic(topic)
            if not all(endpoint_metadata_ready(entry) for entry in publishers):
                ready = False
            if not any(entry.node_name == node.get_name() and entry.node_namespace == node.get_namespace()
                       for entry in publishers):
                ready = False
        if ready:
            return discovered
        if time.monotonic() >= deadline:
            counts = {f"/joint_{arm}_states": len(node.get_subscriptions_info_by_topic(f"/joint_{arm}_states"))
                      for arm in arms}
            names = [f"{namespace.rstrip('/')}/{name}" for name, namespace in node.get_node_names_and_namespaces()]
            topics = [name for name, _ in node.get_topic_names_and_types() if "joint" in name]
            identities = {arm: [(entry.node_name, entry.node_namespace, entry.topic_type)
                                for entry in endpoints] for arm, endpoints in discovered.items()}
            raise RuntimeError(f"等待 ROS 发现 {timeout:g}s 后仍未找到完整控制链路：{counts}；"
                               f"订阅端点={identities}；可见节点={names}；关节话题={topics}。请核对旧驱动终端是否仍运行、"
                               "话题 remap 和两个终端的 ROS_DOMAIN_ID/ROS_LOCALHOST_ONLY/RMW 配置")
        rclpy.spin_once(node, timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Move both Piper arms to their zero pose through the running ctrl nodes")
    parser.add_argument("--arm", choices=("left", "right", "both"), default="both")
    parser.add_argument("--speed-percent", type=int, default=30, help="关节速度百分比（节点夹到 30~100）")
    parser.add_argument("--gripper-effort", type=float, default=1.0, help="夹爪力 N·m（节点夹到 0.5~3）")
    parser.add_argument("--rate", type=float, default=20.0, help="命令发布频率 Hz")
    parser.add_argument(
        "--seconds", type=float, default=6.0,
        help="从当前位姿斜坡走到零位的秒数（0=直接跳到零位，即原行为）",
    )
    parser.add_argument("--timeout", type=float, default=20.0, help="最长等待到位的秒数")
    parser.add_argument("--discovery-timeout", type=float, default=10.0, help="等待 ROS 控制节点发现的秒数")
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印将下发的计划与等价命令，不发布任何关节命令"
    )
    parser.add_argument("--hold", action="store_true", help="到位后继续按 --timeout 持续发布（做实验时可当载体流）")
    parser.add_argument("--allow-non-joint", action="store_true", help="允许在 control_mode != joint 时执行")
    args = parser.parse_args(argv)
    arms = ARMS if args.arm == "both" else (args.arm,)

    for key in ("rate", "timeout", "seconds", "gripper_effort", "discovery_timeout"):
        value = getattr(args, key)
        if not math.isfinite(value) or (value < 0 if key == "seconds" else value <= 0):
            parser.error(f"--{key} 数值无效")
    rclpy.init()
    node = GoZero(arms, args.rate)
    try:
        return run(node, args, arms)
    except KeyboardInterrupt:
        print("已中断，停止发布。")
        return 130
    except Exception as exc:
        print(f"[拒绝] {exc}", file=sys.stderr)
        return 3
    finally:
        node.destroy_node()
        rclpy.shutdown()


def run(node, args, arms):
    print("回零位计划：")
    print(f"  ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '0')} "
          f"ROS_LOCALHOST_ONLY={os.environ.get('ROS_LOCALHOST_ONLY', '0')} "
          f"RMW_IMPLEMENTATION={os.environ.get('RMW_IMPLEMENTATION', '(default)')}")
    discovered = wait_for_drivers(node, arms, args.discovery_timeout)
    kind = check_drivers(node, arms, args.allow_non_joint, discovered=discovered)
    print(f"  目标: 7 关节全 0，夹爪 0；速度 {args.speed_percent}%，夹爪力 {args.gripper_effort} N·m，{args.rate} Hz，最长 {args.timeout} s")
    if args.seconds > 0:
        print(f"  斜坡: 从当前位姿用 {args.seconds:g}s 平滑走到零位（余弦起停）")
    else:
        print("  斜坡: 关闭（直接跳到零位）")

    if args.dry_run:
        print("\n[dry-run] 不发任何命令。等价命令示例：")
        print("  ros2 topic pub --rate 20 /joint_left_states sensor_msgs/msg/JointState \\")
        print(f"    \"{{name: ['joint0','joint1','joint2','joint3','joint4','joint5','joint6'], position: [0,0,0,0,0,0,0], velocity: [0,0,0,0,0,0,{args.speed_percent}], effort: [0,0,0,0,0,0,{args.gripper_effort}]}}\"")
        return 0

    period = 1.0 / args.rate
    last_report = 0.0
    reached = False
    completed = False
    try:
        # 先拿到当前位姿，作为斜坡起点
        wait_until = time.monotonic() + 3.0
        while not node.fresh() and time.monotonic() < wait_until:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not node.fresh():
            raise RuntimeError("未收到所有目标臂的新鲜反馈；不会直接跳到零位")
        starts = {arm: list(node._arm_state[arm]) for arm in arms}

        if args.seconds > 0:
            if starts:
                node.ramp_to_zero(starts, args.seconds, args.speed_percent, args.gripper_effort, period)
                node.report("斜坡结束")

        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            now = time.monotonic()
            if node._arm_state and now - last_report >= 0.5:
                node.report("当前")
                last_report = now
                if node.converged() and not args.hold:
                    reached = True
                    break
            node.publish_zero(args.speed_percent, args.gripper_effort)
            spin_period(node, period)
        time.sleep(0.3)
        rclpy.spin_once(node, timeout_sec=0.2)
        node.report("结束")
        if args.hold:
            print(f"按 --hold 持续发布了 {args.timeout}s，已停止发布。")
        elif reached:
            if kind == "pyagxarm":
                # Ensure an exact zero target precedes the completion request.
                node.publish_zero(args.speed_percent, args.gripper_effort)
                spin_period(node, 0.1)
                finish_zero_task(node)
                completed = True
            print("已到零位（误差在容差内）。")
            if completed:
                print("pyAgxArm: 回零任务正常结束，驱动继续使能待命，可直接启动下一项任务。")
        else:
            print(f"未在 {args.timeout}s 内收敛到容差，已停止发布。")
    except KeyboardInterrupt:
        raise
    finally:
        if kind == "pyagxarm" and not completed:
            print("pyAgxArm: 已停止发布；驱动没有额外的断流自动急停逻辑。")
    return 0 if reached or args.hold else 1


if __name__ == "__main__":
    raise SystemExit(main())
