#!/usr/bin/python3
"""Passive SocketCAN snapshot. No SDK, send(), enable, mode, or reset commands."""
import json
import time
import struct

import can


def main():
    buses = {}
    samples = {}
    try:
        for channel in ("can_left", "can_right"):
            buses[channel] = can.Bus(interface="socketcan", channel=channel, receive_own_messages=False)
            samples[channel] = {}
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            for channel, bus in buses.items():
                frame = bus.recv(timeout=0.001)
                if frame is None:
                    continue
                aid, data = frame.arbitration_id, bytes(frame.data)
                if aid in (0x2A1, 0x2A5, 0x2A6, 0x2A7) or 0x261 <= aid <= 0x266:
                    row = samples[channel].setdefault(hex(aid), {"count": 0})
                    row.update(count=row["count"]+1, data=data.hex(), timestamp=frame.timestamp)
                    if len(data) == 8 and 0x2A5 <= aid <= 0x2A7:
                        row["joint_millidegrees"] = struct.unpack(">ii", data)
                    if len(data) == 8 and 0x261 <= aid <= 0x266:
                        row["joint"] = aid - 0x260
                        row["driver_enabled"] = bool(data[5] & 0x40)
                        row["foc_status"] = data[5]
        print(json.dumps(samples, indent=2))
    finally:
        for bus in buses.values():
            bus.shutdown()


if __name__ == "__main__":
    main()
