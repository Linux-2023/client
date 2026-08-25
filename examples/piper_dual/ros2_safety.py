"""Safety latch for official Piper ROS 2 arm status messages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class _StatusSnapshot:
    ctrl_mode: int
    arm_status: int
    mode_feedback: int
    motion_status: int
    err_code: int
    received_at: float


class PiperStatusLatch:
    """Permanently latch unsafe official Piper arm status until process restart."""

    _SIDES = ("left", "right")

    def __init__(self, *, status_watchdog_s: float, expected_mode_feedback: int = 1) -> None:
        self.status_watchdog_s = float(status_watchdog_s)
        if self.status_watchdog_s <= 0:
            raise ValueError("status_watchdog_s must be positive")
        if type(expected_mode_feedback) is not int or expected_mode_feedback not in (0, 1):
            raise ValueError("expected_mode_feedback must be integer 0 or 1")
        self.expected_mode_feedback = expected_mode_feedback
        self._statuses: dict[str, _StatusSnapshot] = {}
        self.fault_reason: str | None = None
        self.fault_side: str | None = None

    def update(self, side: str, message: Any, received_at: float) -> None:
        """Record one side's status sample and permanently latch explicit faults."""
        if side not in self._SIDES:
            raise ValueError(f"unknown Piper side: {side!r}")
        snapshot = _StatusSnapshot(
            ctrl_mode=self._int_field(message, "ctrl_mode"),
            arm_status=self._int_field(message, "arm_status"),
            mode_feedback=self._int_field(message, "mode_feedback"),
            motion_status=self._int_field(message, "motion_status"),
            err_code=self._int_field(message, "err_code"),
            received_at=float(received_at),
        )
        self._statuses[side] = snapshot
        self._latch_bad_field(side, "arm_status", snapshot.arm_status, expected=0)
        self._latch_bad_field(side, "err_code", snapshot.err_code, expected=0)
        self._latch_bad_field(side, "ctrl_mode", snapshot.ctrl_mode, expected=1)
        self._latch_bad_field(side, "mode_feedback", snapshot.mode_feedback, expected=self.expected_mode_feedback)

    def assert_joint_ready(self, *, now: float) -> None:
        """Raise if either side is missing, stale, explicitly faulted, or latched."""
        if self.fault_reason is not None:
            raise RuntimeError(self.fault_reason)
        current_time = float(now)
        for side in self._SIDES:
            snapshot = self._statuses.get(side)
            if snapshot is None:
                self._latch(side, f"{side} status missing")
                raise RuntimeError(self.fault_reason)
            if current_time - snapshot.received_at > self.status_watchdog_s:
                self._latch(side, f"{side} status stale")
                raise RuntimeError(self.fault_reason)

    def _latch_bad_field(self, side: str, field: str, value: int, *, expected: int) -> None:
        if value != expected:
            self._latch(side, f"{side} {field}={value}")

    def _latch(self, side: str, reason: str) -> None:
        if self.fault_reason is None:
            self.fault_side = side
            self.fault_reason = reason

    @staticmethod
    def _int_field(message: Any, field: str) -> int:
        value = getattr(message, field)
        if isinstance(value, bool):
            raise ValueError(f"{field} must be an integer")
        return int(value)
