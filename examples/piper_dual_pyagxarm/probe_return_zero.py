#!/usr/bin/python3
"""Explicitly authorized 2-second motion probe with passive CAN recording."""
import argparse
import json
import math
from pathlib import Path
import threading
import time

import can
import rclpy
from piper_msgs.msg import PiperStatusMsg

from go_zero_dual import GoZero, wait_for_drivers, check_drivers, spin_period


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-confirmed", action="store_true", required=True)
    parser.add_argument("--driver", choices=("legacy", "pyagxarm"), default="legacy")
    parser.add_argument("--hold-current", action="store_true",
                        help="Keep the sampled joint and gripper targets for the 2-second mode test")
    parser.add_argument("--reset-controller", action="store_true",
                        help="Extra explicit recovery step: one 0x150 reset per arm before the probe")
    args = parser.parse_args()
    if args.driver != "legacy" and args.reset_controller:
        parser.error("Direct CAN recovery is restricted to the legacy probe")
    rclpy.init()
    node = GoZero(("left", "right"), 20)
    buses, threads = {}, []
    stop = threading.Event()
    records, targets, statuses = [], [], {}
    report = {"targets": targets, "can": records, "statuses": statuses}

    def listen(channel, bus):
        while not stop.is_set():
            msg = bus.recv(timeout=0.01)
            if msg is not None and (msg.arbitration_id in (0x150, 0x151, 0x155, 0x156, 0x157, 0x159, 0x2A1)
                                    or 0x2A5 <= msg.arbitration_id <= 0x2A7):
                records.append(dict(channel=channel, timestamp=msg.timestamp,
                                    id=hex(msg.arbitration_id), data=bytes(msg.data).hex()))

    def status(side, message):
        statuses[side] = {key: int(getattr(message, key)) for key in
                          ("ctrl_mode", "arm_status", "mode_feedback", "err_code")}

    try:
        for side in node.arms:
            node.create_subscription(PiperStatusMsg, f"/piper_{side}_ctrl_node/arm_status",
                                     lambda msg, side=side: status(side, msg), 1)
        discovered = wait_for_drivers(node, node.arms, 10)
        if check_drivers(node, node.arms, discovered=discovered) != args.driver:
            raise RuntimeError(f"Expected {args.driver} driver")
        spin_period(node, 0.5)
        if not node.fresh() or len(statuses) != 2:
            raise RuntimeError("Missing fresh feedback/status")
        for channel in ("can_left", "can_right"):
            bus = can.Bus(interface="socketcan", channel=channel, receive_own_messages=False)
            buses[channel] = bus
            worker = threading.Thread(target=listen, args=(channel, bus), daemon=True)
            threads.append(worker)
            worker.start()
        starts = {side: list(node._arm_state[side]) for side in node.arms}
        report["start_positions"] = starts
        report["start_statuses"] = {side: dict(value) for side, value in statuses.items()}
        if args.reset_controller:
            if any(value["arm_status"] or value["err_code"] for value in statuses.values()):
                raise RuntimeError("Refusing mode recovery with an existing arm fault")
            # Piper vendor piper_ctrl_reset.py prescribes MotionCtrl_1(0x02,0,0)
            # when switching from MIT/teaching to position-speed control.
            # This sends no joint target and must be separately operator-authorized.
            report["controller_reset_requested"] = True
            for bus in buses.values():
                bus.send(can.Message(arbitration_id=0x150, data=bytes([2,0,0,0,0,0,0,0]), is_extended_id=False))
            spin_period(node, 0.2)
            if not node.fresh():
                raise RuntimeError("Feedback became stale after mode recovery")
            starts = {side: list(node._arm_state[side]) for side in node.arms}
        begin = time.monotonic()
        while True:
            elapsed = time.monotonic() - begin
            if elapsed >= 2.0:
                break
            if any(value["arm_status"] or value["err_code"] for value in statuses.values()):
                raise RuntimeError(f"Fault during probe: {statuses}")
            alpha = 0.0 if args.hold_current else 0.5 - 0.5 * math.cos(math.pi * elapsed / 6.0)
            target = {side: [v*(1-alpha) for v in values] for side, values in starts.items()}
            node.publish_targets(target, 30, 1.0)
            targets.append(dict(elapsed=elapsed, target=target, measured={side: list(node._arm_state[side]) for side in node.arms},
                                matched={side: node._arm_publishers[side].get_subscription_count() for side in node.arms}))
            spin_period(node, min(0.05, max(0, 2.0-(time.monotonic()-begin))))
        report["publish_duration"] = time.monotonic()-begin
        spin_period(node, 0.5)
        report["end_positions"] = {side: list(node._arm_state[side]) for side in node.arms}
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        stop.set()
        for worker in threads:
            worker.join(timeout=1)
        for bus in buses.values():
            bus.shutdown()
        path = Path(f"/tmp/piper_return_zero_probe_{time.time_ns()}.json")
        path.write_text(json.dumps(report, indent=2))
        counts = {}
        for record in records:
            key = record["channel"]+":"+record["id"]
            counts[key] = counts.get(key, 0)+1
        print(json.dumps({key: value for key, value in report.items() if key not in ("targets", "can")}, indent=2))
        print("Published", len(targets), "steps; CAN counts:", counts, "record:", path)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
