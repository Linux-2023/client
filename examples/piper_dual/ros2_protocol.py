"""ROS-independent JSON-line protocol for the Piper dual ROS 2 bridge."""

import json
from typing import Any


JsonObject = dict[str, Any]


def _reject_constant(_: str) -> None:
    raise ValueError("Protocol message is not valid JSON")


def encode_message(message: JsonObject) -> str:
    """Encode one protocol message as compact JSON followed by a newline."""
    if not isinstance(message, dict):
        raise ValueError("Protocol messages must be JSON objects")
    try:
        return json.dumps(message, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
    except ValueError as exc:
        raise ValueError("Protocol message is not valid JSON") from exc


def decode_message(line: str) -> JsonObject:
    """Decode one newline-framed JSON object."""
    if not line.strip():
        raise ValueError("Protocol messages must not be blank")

    try:
        message = json.loads(line, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Protocol message is not valid JSON") from exc

    if not isinstance(message, dict):
        raise ValueError("Protocol messages must be JSON objects")
    return message
