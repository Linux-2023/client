from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import go_zero_dual as zero


def endpoint(name="pyagxarm_dual_local_driver"):
    return NS(node_name=name, node_namespace="/", topic_type="sensor_msgs/msg/JointState")


@pytest.mark.parametrize("name,kind", [("pyagxarm_dual_local_driver", "pyagxarm"),
                                       ("piper_left_ctrl_node", "legacy")])
def test_driver_discovery(name, kind):
    assert zero.identify_driver("left", [endpoint(name)])[0] == kind


@pytest.mark.parametrize("endpoints", [[], [endpoint(), endpoint()], [endpoint("unknown")]])
def test_missing_duplicate_unknown_driver_rejected(endpoints):
    with pytest.raises(RuntimeError):
        zero.identify_driver("left", endpoints)


def test_pyagxarm_does_not_query_legacy_parameter(monkeypatch):
    node = Mock()
    node.get_subscriptions_info_by_topic.return_value = [endpoint()]
    node.get_name.return_value = "go_zero_dual"
    node.get_namespace.return_value = "/"
    node.get_publishers_info_by_topic.return_value = [endpoint("go_zero_dual")]
    query = Mock(side_effect=AssertionError("must not query old node"))
    monkeypatch.setattr(zero, "legacy_mode", query)
    assert zero.check_drivers(node, ("left", "right")) == "pyagxarm"
    node.get_publishers_info_by_topic.return_value = [endpoint("go_zero_dual"), endpoint("policy_bridge")]
    with pytest.raises(RuntimeError, match="policy_bridge"):
        zero.check_drivers(node, ("left", "right"))


@pytest.mark.parametrize("input_count", [0, 1])
def test_idle_ik_is_allowed_but_active_input_blocks(input_count):
    node = Mock()
    node.get_name.return_value = "go_zero_dual"
    node.get_namespace.return_value = "/"
    node.get_publishers_info_by_topic.return_value = [endpoint("go_zero_dual"), endpoint("piper_pinocchio_ik")]
    node.get_subscriber_names_and_types_by_node.return_value = [
        ("/pos_left_cmd", ["piper_msgs/msg/PosCmd"]),
        ("/pos_right_cmd", ["piper_msgs/msg/PosCmd"]),
    ]
    node.count_publishers.return_value = input_count
    if input_count:
        with pytest.raises(RuntimeError, match="末端输入有发布者"):
            zero.assert_command_sources(node, ("left", "right"))
    else:
        zero.assert_command_sources(node, ("left", "right"))


def test_publish_rechecks_competing_sources(monkeypatch):
    node = Mock()
    node.fresh.return_value = True
    monkeypatch.setattr(zero, "assert_command_sources", Mock(side_effect=RuntimeError("new publisher")))
    with pytest.raises(RuntimeError, match="new publisher"):
        zero.GoZero.publish_targets(node, {"left": [0]*7}, 30, 1)
    node.get_clock.assert_not_called()


def test_feedback_requires_both_arms_and_rejects_nan():
    node = NS(arms=("left", "right"), _arm_state={}, _received_at={})
    zero.GoZero.receive_state(node, "left", NS(position=[0]*7))
    assert not zero.GoZero.fresh(node)
    zero.GoZero.receive_state(node, "right", NS(position=[0]*7))
    assert zero.GoZero.fresh(node)
    node._received_at["left"] = time.monotonic()-1
    assert not zero.GoZero.fresh(node)
    zero.GoZero.receive_state(node, "right", NS(position=[float("nan")]*7))
    assert "right" not in node._arm_state


def test_dry_run_never_publishes(monkeypatch):
    monkeypatch.setattr(zero, "wait_for_drivers", lambda *args: None)
    monkeypatch.setattr(zero, "check_drivers", lambda *args, **kwargs: "pyagxarm")
    node = Mock()
    args = NS(allow_non_joint=False, speed_percent=30, gripper_effort=1,
              rate=20, timeout=20, seconds=6, dry_run=True, discovery_timeout=10)
    assert zero.run(node, args, ("left", "right")) == 0
    node.publish_zero.assert_not_called()
    node.publish_targets.assert_not_called()
    node.ramp_to_zero.assert_not_called()


def test_stale_feedback_blocks_target():
    node = Mock()
    node.fresh.return_value = False
    with pytest.raises(RuntimeError, match="反馈缺失"):
        zero.GoZero.publish_targets(node, {"left": [0]*7}, 30, 1)
    node.get_clock.assert_not_called()


def test_discovery_retries_empty_graph(monkeypatch):
    node = Mock()
    node.get_name.return_value = "go_zero_dual"
    node.get_namespace.return_value = "/"
    node.get_publishers_info_by_topic.return_value = [endpoint("go_zero_dual")]
    node.get_subscriptions_info_by_topic.side_effect = [[], [], [endpoint()], [endpoint()]]
    spin = Mock()
    monkeypatch.setattr(zero.rclpy, "spin_once", spin)
    zero.wait_for_drivers(node, ("left", "right"), 10)
    spin.assert_called_once()


def test_discovery_timeout_reports_visible_graph(monkeypatch):
    node = Mock()
    node.get_subscriptions_info_by_topic.return_value = []
    node.get_publishers_info_by_topic.return_value = []
    node.get_node_names_and_namespaces.return_value = [("go_zero_dual", "/")]
    node.get_topic_names_and_types.return_value = [("/joint_ctrl_cmd_left", [])]
    ticks = iter([0.0, 11.0])
    monkeypatch.setattr(zero.time, "monotonic", lambda: next(ticks))
    with pytest.raises(RuntimeError, match="joint_ctrl_cmd_left"):
        zero.wait_for_drivers(node, ("left", "right"), 10)


@pytest.mark.parametrize("incomplete", ["subscriber", "publisher", "namespace", "type"])
def test_waits_for_endpoint_identity_not_just_count(monkeypatch, incomplete):
    node = Mock()
    node.get_name.return_value = "go_zero_dual"
    node.get_namespace.return_value = "/"
    partial = endpoint("_NODE_NAME_UNKNOWN_")
    if incomplete == "namespace":
        partial = endpoint("piper_left_ctrl_node")
        partial.node_namespace = "_NODE_NAMESPACE_UNKNOWN_"
    elif incomplete == "type":
        partial = endpoint("piper_left_ctrl_node")
        partial.topic_type = ""
    complete = endpoint("piper_left_ctrl_node")
    node.get_subscriptions_info_by_topic.side_effect = [
        [complete if incomplete == "publisher" else partial], [complete],
    ]
    node.get_publishers_info_by_topic.side_effect = [
        [endpoint("go_zero_dual")] + ([partial] if incomplete == "publisher" else []),
        [endpoint("go_zero_dual")],
    ]
    spin = Mock()
    monkeypatch.setattr(zero.rclpy, "spin_once", spin)
    result = zero.wait_for_drivers(node, ("left",), 10)
    assert result["left"] == [complete]
    spin.assert_called_once()


def test_check_uses_confirmed_endpoint_snapshot(monkeypatch):
    node = Mock()
    node.get_subscriptions_info_by_topic.side_effect = AssertionError("do not requery identity")
    monkeypatch.setattr(zero, "assert_command_sources", lambda *args: None)
    monkeypatch.setattr(zero, "legacy_mode", lambda *args: "joint")
    assert zero.check_drivers(node, ("left",), discovered={"left": [endpoint("piper_left_ctrl_node")]}) == "legacy"


def test_permanent_unknown_identity_is_not_accepted(monkeypatch):
    node = Mock()
    node.get_name.return_value = "go_zero_dual"
    node.get_namespace.return_value = "/"
    node.get_subscriptions_info_by_topic.return_value = [endpoint("_NODE_NAME_UNKNOWN_")]
    node.get_publishers_info_by_topic.return_value = [endpoint("go_zero_dual")]
    node.get_node_names_and_namespaces.return_value = []
    node.get_topic_names_and_types.return_value = []
    ticks = iter([0.0, 11.0])
    monkeypatch.setattr(zero.time, "monotonic", lambda: next(ticks))
    with pytest.raises(RuntimeError, match="_NODE_NAME_UNKNOWN_"):
        zero.wait_for_drivers(node, ("left",), 10)


def test_joint_label_does_not_hide_mit_and_zero_speed():
    settings = dict(control_mode="joint", joint_is_mit_mode=173, joint_move_mode=1, joint_speed_override=0)
    with pytest.raises(RuntimeError) as error:
        zero.validate_legacy_settings(endpoint("piper_left_ctrl_node"), settings)
    assert "joint_is_mit_mode=173" in str(error.value)
    assert "joint_speed_override=0" in str(error.value)


def test_normal_legacy_joint_parameters():
    zero.validate_legacy_settings(endpoint("piper_right_ctrl_node"),
        dict(control_mode="joint", joint_is_mit_mode=0, joint_move_mode=1, joint_speed_override=-1))
