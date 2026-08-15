"""ROS-independent JSON-line protocol for the Piper dual ROS 2 bridge."""

import json
from typing import Any


JsonObject = dict[str, Any]


def encode_message(message: JsonObject) -> str:
    """Encode one protocol message as compact JSON followed by a newline."""
    if not isinstance(message, dict):
        raise ValueError("Protocol messages must be JSON objects")
    return json.dumps(message, separators=(",", ":"), ensure_ascii=False) + "\n"


def decode_message(line: str) -> JsonObject:
    """Decode one newline-framed JSON object."""
    if not line.strip():
        raise ValueError("Protocol messages must not be blank")

    try:
        message = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError("Protocol message is not valid JSON") from exc

    if not isinstance(message, dict):
        raise ValueError("Protocol messages must be JSON objects")
    return message
