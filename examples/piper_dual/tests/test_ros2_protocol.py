"""Focused tests for the ROS 2 bridge JSON-line protocol."""

from pathlib import Path
import json
import math
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ros2_protocol import decode_message
from ros2_protocol import encode_message


def test_encode_message_preserves_unicode_status_metadata_with_compact_newline_framing():
    line = encode_message(
        {
            "type": "status",
            "state": "ready",
            "metadata": {"operator": "张三", "note": "café ✓"},
        }
    )

    assert line == '{"type":"status","state":"ready","metadata":{"operator":"张三","note":"café ✓"}}\n'
    assert decode_message(line) == {
        "type": "status",
        "state": "ready",
        "metadata": {"operator": "张三", "note": "café ✓"},
    }


def test_decode_message_rejects_blank_malformed_and_non_object_json():
    bad_lines = ["", "\n", "not-json\n", "[]\n", "null\n", '"text"\n']

    for line in bad_lines:
        with pytest.raises(ValueError):
            decode_message(line)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_decode_message_rejects_non_finite_json_constants(constant):
    with pytest.raises(ValueError, match="not valid JSON"):
        decode_message(f'{{"type":"sensor","value":{constant}}}\n')


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_encode_message_rejects_non_finite_numbers(value):
    with pytest.raises(ValueError, match="not valid JSON"):
        encode_message({"type": "sensor", "value": value})


def test_encode_message_rejects_non_object_messages():
    with pytest.raises(ValueError):
        encode_message(["not", "an", "object"])


def test_sensor_image_event_preserves_base64_payload_without_client_side_decoding():
    event = {
        "type": "sensor",
        "sensor": "cam_high",
        "timestamp": 12.5,
        "jpeg_b64": "/9j/4AAQSkZJRgABAQ==",
    }

    assert decode_message(encode_message(event)) == event


def test_action_request_structure_round_trips_exact_left_and_right_vectors():
    request = {
        "type": "publish_action",
        "left": [0.0, 1.0, 2.5, -3.0, 4.0, 5.0, 6.0],
        "right": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
    }

    assert decode_message(encode_message(request)) == request


def test_stop_request_serializes_as_single_json_object_line():
    assert encode_message({"type": "stop"}) == '{"type":"stop"}\n'
    assert json.loads(encode_message({"type": "stop"})) == {"type": "stop"}
