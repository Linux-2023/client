#!/usr/bin/python3
"""Set the Piper gripper zero point over SocketCAN.

行程 0 的含义是"夹爪闭合参考点"：设完零点后 ``grippers_angle`` 会把当前位置读成 0，
而策略里的夹爪开度（米）按 ``grippers_angle * 0.001 * 0.001`` 映射，所以设零点必须在
你们定义为"闭合"的那个机械位置上做，否则策略的夹爪指令会整体平移。

与 vendor demo 等价（piper_sdk/demo/V2/piper_set_gripper_zero.py）：
    GripperCtrl(0, 1000, 0x00, 0)  ->  1.5s  ->  GripperCtrl(0, 1000, 0x00, 0xAE)

用法（必须用系统解释器，piper_sdk/python-can 装在那里）：
    /usr/bin/python3 examples/piper_dual/set_gripper_zero.py --can can_left --read-only
    /usr/bin/python3 examples/piper_dual/set_gripper_zero.py --can can_left              # 只打印将要发送的报文
    /usr/bin/python3 examples/piper_dual/set_gripper_zero.py --can can_left --confirm    # 真正设置零点

护栏：
- 不加 ``--confirm`` 绝不发送任何 CAN 报文；
- 目标臂的 ``piper_*_ctrl_node`` 仍在运行时拒绝执行（同一总线两个进程发命令会互相干扰），
  确认过风险可用 ``--allow-running-ctrl-node`` 覆盖；
- ``ConnectPort`` 用 ``piper_init=False``，不触发 SDK 的整臂初始化（不碰关节参数）；
- 设置前后各回读一次 ``grippers_angle``，便于判断是否落在预期的机械位置。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from piper_sdk import C_PiperInterface_V2

GRIPPER_UNIT_METERS = 0.001 * 0.001  # grippers_angle 单位 0.001mm -> m
CTRL_NODE_BY_CAN = {"can_left": "piper_left_ctrl_node", "can_right": "piper_right_ctrl_node"}


def _ctrl_node_running(node_name: str) -> bool:
    result = subprocess.run(["pgrep", "-f", f"__node:={node_name}"], capture_output=True, check=False)
    return result.returncode == 0


def _read_grippers_angle(piper: C_PiperInterface_V2) -> int | None:
    try:
        return int(piper.GetArmGripperCtrl().gripper_ctrl.grippers_angle)
    except Exception as exc:  # noqa: BLE001 - feedback may be missing on a silent bus
        print(f"  [warn] 读取夹爪反馈失败: {exc}", file=sys.stderr)
        return None


def _describe(can_port: str, angle: int | None) -> str:
    if angle is None:
        return f"[{can_port}] grippers_angle 不可读"
    return f"[{can_port}] grippers_angle = {angle} ({angle * GRIPPER_UNIT_METERS:.4f} m)"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set the Piper gripper zero point")
    parser.add_argument("--can", required=True, help="SocketCAN interface of the target arm, e.g. can_left")
    parser.add_argument("--ctrl-node", default=None, help="ROS node that owns this CAN bus (default: derived)")
    parser.add_argument("--confirm", action="store_true", help="required before any CAN frame is sent")
    parser.add_argument("--read-only", action="store_true", help="print the current gripper travel and exit")
    parser.add_argument(
        "--close-first",
        action="store_true",
        help="enable the gripper and command travel 0 before setting zero (moves the gripper!)",
    )
    parser.add_argument("--effort", type=int, default=1000, help="gripper effort in 0.001 N/m (default: 1000 = 1 N/m)")
    parser.add_argument("--close-seconds", type=float, default=2.0, help="wait after the closing command")
    parser.add_argument(
        "--allow-running-ctrl-node",
        action="store_true",
        help="override the guard that refuses to run while the arm's ctrl node is up",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ctrl_node = args.ctrl_node or CTRL_NODE_BY_CAN.get(args.can)

    if args.read_only and args.close_first:
        print("--read-only 与 --close-first 互斥", file=sys.stderr)
        return 2

    if not args.confirm and not args.read_only:
        print(f"[dry-run] 将对 {args.can} 发送：", file=sys.stderr)
        if args.close_first:
            print(f"  GripperCtrl(0, {args.effort}, 0x01, 0)   # 使能 + 行程目标 0（会动）", file=sys.stderr)
            print(f"  等待 {args.close_seconds}s", file=sys.stderr)
        print("  GripperCtrl(0, 1000, 0x00, 0)            # 失能", file=sys.stderr)
        print("  等待 1.5s", file=sys.stderr)
        print("  GripperCtrl(0, 1000, 0x00, 0xAE)         # set_zero：把当前位置设为行程 0", file=sys.stderr)
        print(
            f"目标臂 ctrl 节点 {ctrl_node or '(未指定)'} 目前"
            f"{'在运行' if ctrl_node and _ctrl_node_running(ctrl_node) else '未运行'}；加 --confirm 才会真正发送。",
            file=sys.stderr,
        )
        return 0

    if ctrl_node and _ctrl_node_running(ctrl_node):
        if not (args.allow_running_ctrl_node or args.read_only):
            print(
                f"[拒绝] {ctrl_node} 正在运行并占用 {args.can}。先停掉它再设零点"
                f"（确认风险可用 --allow-running-ctrl-node 覆盖）。",
                file=sys.stderr,
            )
            return 3
        print(f"[提示] {ctrl_node} 正在运行（{'只读模式' if args.read_only else '已显式覆盖护栏'}）")


    try:
        piper = C_PiperInterface_V2(args.can)
    except ConnectionError as exc:
        print(f"[错误] 打不开 {args.can}（接口名/总线是否存在？）: {exc}", file=sys.stderr)
        return 2

    try:
        piper.ConnectPort(can_init=True, piper_init=False, start_thread=True)
        time.sleep(0.5)
        print(_describe(args.can, _read_grippers_angle(piper)))
        if args.read_only:
            return 0

        if args.close_first:
            print(f"[{args.can}] 使能并命令行程 0（effort={args.effort}），等待 {args.close_seconds}s ...")
            piper.GripperCtrl(0, args.effort, 0x01, 0)
            time.sleep(args.close_seconds)

        print(f"[{args.can}] 发送 GripperCtrl(0, 1000, 0x00, 0) ...")
        piper.GripperCtrl(0, 1000, 0x00, 0)
        time.sleep(1.5)

        print(f"[{args.can}] 发送 GripperCtrl(0, 1000, 0x00, 0xAE)  # set zero")
        piper.GripperCtrl(0, 1000, 0x00, 0xAE)
        time.sleep(0.5)

        angle = _read_grippers_angle(piper)
        print(_describe(args.can, angle))
        if angle is not None and angle != 0:
            print(
                "  [warn] 零点设置后回读不为 0：确认夹爪处于期望的闭合参考位置，"
                "必要时重新上电后再试一次。",
                file=sys.stderr,
            )
    finally:
        piper.DisconnectPort()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
