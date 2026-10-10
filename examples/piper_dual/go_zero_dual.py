#!/usr/bin/python3
"""双臂回零位（关节零点角 + 夹爪闭合）—— 经 ROS 关节命令话题下发，不直接碰 CAN。

为什么不用 vendor demo（third_party/piper_sdk/piper_sdk/demo/V2/piper_ctrl_go_zero.py）：
  - 它硬编码 "can0"（本机只有 can_left / can_right），且只驱动单臂；
  - 直接开 CAN 会与正在运行的 piper_*_ctrl_node 抢同一条总线（两个进程同时下发命令）。
本工具改为把命令交给节点（节点的 can_port 参数已各自配对），等价效果：
  MotionCtrl_2(0x01,0x01,<speed>,0x00) + JointCtrl(0,0,0,0,0,0) + GripperCtrl(0, ...)

用法（需已 source ROS 环境，或直接用包装命令 return_zero）：
  /usr/bin/python3 examples/piper_dual/go_zero_dual.py                        # 双臂回零
  /usr/bin/python3 examples/piper_dual/go_zero_dual.py --arm left             # 只回左臂
  /usr/bin/python3 examples/piper_dual/go_zero_dual.py --speed-percent 30 --gripper-effort 1.0
  /usr/bin/python3 examples/piper_dual/go_zero_dual.py --seconds 15           # 用 15s 从当前位姿慢慢走到零位
  /usr/bin/python3 examples/piper_dual/go_zero_dual.py --seconds 0            # 直接跳到零位（旧行为）
  /usr/bin/python3 examples/piper_dual/go_zero_dual.py --dry-run              # 只打印计划，不下发
  /usr/bin/python3 examples/piper_dual/go_zero_dual.py --hold --timeout 60    # 持续 60s 下发（做实验当载体流）

护栏：
- 目标臂的 ctrl 节点必须在线（关节命令话题要有订阅者），且 control_mode 必须是 'joint'
  （若是 'mit'，零位会被当 MIT 目标解析，需显式 --allow-non-joint 才放行）；
- 速度按节点规则被夹到 30~100%，夹爪力夹到 0.5~3 N·m；
- 到位后停止发布，节点会用最后的目标保持位姿（不会松臂）；
- 默认走"斜坡"：先读当前位姿，再用 --seconds（默认 6s）余弦起停平滑走到零位，
  每步目标都很小（峰值步长 = 行程 × π/(2×步数)），所以动作慢且连续；--seconds 0 恢复"一次跳到位"。
"""

from __future__ import annotations

import argparse
import math
import subprocess
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
        self._arm_publishers = {
            arm: self.create_publisher(JointState, f"/joint_{arm}_states", 1) for arm in arms
        }
        for arm in arms:
            self.create_subscription(
                JointState,
                f"/puppet/joint_{arm}",
                lambda msg, a=arm: self._arm_state.__setitem__(a, [float(v) for v in msg.position]),
                1,
            )
        self.rate_hz = rate_hz

    def publish_targets(self, targets: dict[str, list[float]], speed_percent: int, gripper_effort: float) -> None:
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
            rclpy.spin_once(self, timeout_sec=period)
            if k % max(1, steps // 6) == 0:
                self.report(f"斜坡 {u * 100:3.0f}%")

    def converged(self) -> bool:
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


def _subscriber_count(arm: str) -> int:
    result = subprocess.run(
        ["ros2", "topic", "info", f"/joint_{arm}_states"], capture_output=True, text=True, timeout=15, check=False
    )
    for line in result.stdout.splitlines():
        if line.startswith("Subscription count:"):
            return int(line.split(":")[1].strip())
    return -1


def _control_mode(arm: str) -> str:
    result = subprocess.run(
        ["ros2", "param", "get", f"/{CTRL_NODE[arm]}", "control_mode"], capture_output=True, text=True, timeout=15, check=False
    )
    for line in result.stdout.splitlines():
        if ":" in line:
            return line.split(":", 1)[1].strip().strip("'\"")
    return "?"


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
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印将下发的计划与等价命令，不发布任何关节命令"
    )
    parser.add_argument("--hold", action="store_true", help="到位后继续按 --timeout 持续发布（做实验时可当载体流）")
    parser.add_argument("--allow-non-joint", action="store_true", help="允许在 control_mode != joint 时执行")
    args = parser.parse_args(argv)
    arms = ARMS if args.arm == "both" else (args.arm,)

    print("回零位计划：")
    for arm in arms:
        subs = _subscriber_count(arm)
        mode = _control_mode(arm)
        print(f"  {arm}: 话题 /joint_{arm}_states 订阅者={subs}  ctrl 节点 control_mode={mode}")
        if subs != 1:
            print(f"[拒绝] {arm}: 关节命令话题没有唯一订阅者（ctrl 节点没在跑？）", file=sys.stderr)
            return 3
        if mode != "joint" and not args.allow_non_joint:
            print(f"[拒绝] {arm}: control_mode={mode}（非 joint），零位会被当 MIT 目标解析；"
                  f"先切回 ros2 param set /{CTRL_NODE[arm]} control_mode joint，或加 --allow-non-joint", file=sys.stderr)
            return 3
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

    rclpy.init()
    node = GoZero(arms, args.rate)
    period = 1.0 / args.rate
    last_report = 0.0
    reached = False
    try:
        # 先拿到当前位姿，作为斜坡起点
        wait_until = time.monotonic() + 3.0
        while not node._arm_state and time.monotonic() < wait_until:
            rclpy.spin_once(node, timeout_sec=0.1)
        starts = {arm: list(values) for arm, values in node._arm_state.items()}

        if args.seconds > 0:
            missing = [arm for arm in arms if arm not in starts]
            if missing:
                print(f"  [warn] 没收到 {missing} 的关节反馈 → 这些臂不走斜坡、直接下发零位", file=sys.stderr)
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
            rclpy.spin_once(node, timeout_sec=period)
        time.sleep(0.3)
        rclpy.spin_once(node, timeout_sec=0.2)
        node.report("结束")
        if args.hold:
            print(f"按 --hold 持续发布了 {args.timeout}s，已停止发布（节点保持最后目标）。")
        elif reached:
            print("已到零位（误差在容差内）。")
        else:
            print(f"未在 {args.timeout}s 内收敛到容差，已停止发布（节点保持最后目标，不会松臂）。")
    except KeyboardInterrupt:
        print("\n被中断：已停止发布，节点保持当前目标。")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
