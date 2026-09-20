"""Broker trace contracts without a policy server or robot."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from openpi_client import action_chunk_broker


class ChunkPolicy:
    def __init__(self) -> None:
        self.calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = False

    def infer(self, observation: dict) -> dict:
        self.calls += 1
        if self.block and self.calls > 1:
            self.entered.set()
            if not self.release.wait(timeout=5):
                raise RuntimeError("test did not release inference")
        return {"actions": np.arange(50 * 14).reshape(50, 14) + 10000 * (self.calls - 1)}

    def reset(self) -> None:
        self.calls = 0


@pytest.mark.parametrize(
    "broker_class", [action_chunk_broker.ActionChunkBroker, action_chunk_broker.ActionChunkBroker_RTC]
)
def test_brokers_delegate_server_metadata_without_inference(broker_class: type) -> None:
    class ServerPolicy(ChunkPolicy):
        metadata = {
            "rtc_orientation": {"mode": "so3"},
            "policy_config": "pi05_piper_dual_stack_cups_eef_xyz3d",
            "checkpoint_dir": "/checkpoints/xyz3d",
        }

        def get_server_metadata(self) -> dict:
            return self.metadata

    policy = ServerPolicy()
    broker = broker_class(policy, action_horizon=3)
    assert broker.get_server_metadata() == policy.metadata
    policy.metadata = {"rtc_orientation": {"mode": "wrapped-rpy"}}
    assert broker.get_server_metadata() == policy.metadata
    assert policy.calls == 0


@pytest.mark.parametrize(
    "broker_class", [action_chunk_broker.ActionChunkBroker, action_chunk_broker.ActionChunkBroker_RTC]
)
def test_brokers_without_server_metadata_return_empty_dict(broker_class: type) -> None:
    policy = ChunkPolicy()
    broker = broker_class(policy, action_horizon=3)
    assert broker.get_server_metadata() == {}
    assert policy.calls == 0


@pytest.mark.parametrize(
    "broker_class", [action_chunk_broker.ActionChunkBroker, action_chunk_broker.ActionChunkBroker_RTC]
)
def test_chunk_trace_is_off_by_default(broker_class: type) -> None:
    broker = broker_class(ChunkPolicy(), action_horizon=3)
    assert "_chunk_trace" not in broker.infer({"state": np.zeros(14)})


def test_sync_chunk_trace_matches_selected_action_and_reset() -> None:
    broker = action_chunk_broker.ActionChunkBroker(ChunkPolicy(), action_horizon=2, trace_enabled=True)
    records = [broker.infer({"state": np.zeros(14)}) for _ in range(3)]
    assert [record["_chunk_trace"]["chunk_id"] for record in records] == [0, 0, 1]
    assert [record["_chunk_trace"]["chunk_step"] for record in records] == [0, 1, 0]
    assert [record["_chunk_trace"]["chunk_boundary"] for record in records] == [True, False, True]
    assert [record["actions"][0] for record in records] == [0, 14, 10000]
    trace = records[0]["_chunk_trace"]
    assert trace["use_rtc"] is False
    assert trace["action_horizon"] == 2
    assert trace["configured_delay_steps"] == 0
    assert trace["skipped_steps"] == 0
    assert trace["inference_ms"] >= 0
    assert isinstance(trace["selected_timestamp_ns"], int)
    assert isinstance(trace["selected_monotonic_ns"], int)
    assert "_chunk_trace" not in broker._last_results
    broker.reset()
    reset_record = broker.infer({"state": np.zeros(14)})
    assert reset_record["_chunk_trace"]["chunk_id"] == 0
    assert reset_record["_chunk_trace"]["chunk_boundary"] is True


@pytest.mark.parametrize("use_rtc", [True, False])
def test_async_trace_stays_with_selected_action_across_skipped_handoff(use_rtc: bool) -> None:
    policy = ChunkPolicy()
    policy.block = True
    broker = action_chunk_broker.ActionChunkBroker_RTC(
        policy, action_horizon=3, actions_during_latency=2, use_rtc=use_rtc, trace_enabled=True
    )
    try:
        records = [broker.infer({"state": np.zeros(14)}) for _ in range(4)]
        assert policy.entered.wait(timeout=5)
        old_record = broker.infer({"state": np.zeros(14)})
        old_trace = dict(old_record["_chunk_trace"])
        policy.release.set()
        broker._inference_thread.join(timeout=5)
        assert not broker._inference_thread.is_alive()
        new_record = broker.infer({"state": np.zeros(14)})
        trace = new_record["_chunk_trace"]
        assert old_record["_chunk_trace"] == old_trace
        assert old_record["actions"][0] == 4 * 14
        assert old_trace["chunk_id"] == 0
        assert old_trace["chunk_step"] == 4
        assert trace["chunk_id"] == 1
        assert trace["chunk_step"] == 2
        assert trace["skipped_steps"] == 2
        assert trace["chunk_boundary"] is True
        assert trace["configured_delay_steps"] == 2
        assert trace["use_rtc"] is use_rtc
        assert new_record["actions"][0] == 10000 + 2 * 14
        assert records[0]["_chunk_trace"]["chunk_boundary"] is True
        assert records[1]["_chunk_trace"]["chunk_boundary"] is False
        assert "_chunk_trace" not in broker._last_results
    finally:
        policy.release.set()
        if broker._inference_thread is not None:
            broker._inference_thread.join(timeout=5)
    broker.reset()
    reset_record = broker.infer({"state": np.zeros(14)})
    assert reset_record["_chunk_trace"]["chunk_id"] == 0
    assert reset_record["_chunk_trace"]["chunk_step"] == 0
    assert reset_record["_chunk_trace"]["chunk_boundary"] is True


def test_async_broker_holds_final_chunk_target_while_slow_inference_is_pending() -> None:
    policy = ChunkPolicy()
    policy.block = True
    broker = action_chunk_broker.ActionChunkBroker_RTC(
        policy, action_horizon=3, actions_during_latency=2, use_rtc=True, trace_enabled=True
    )
    try:
        records = [broker.infer({"state": np.zeros(14)}) for _ in range(55)]
        assert policy.entered.wait(timeout=5)
        assert records[-1]["actions"][0] == 49 * 14
        assert records[-1]["_chunk_trace"]["chunk_step"] == 49
    finally:
        policy.release.set()
        if broker._inference_thread is not None:
            broker._inference_thread.join(timeout=5)


def test_async_broker_blends_new_chunk_handoff_with_wrapped_rpy() -> None:
    broker = action_chunk_broker.ActionChunkBroker_RTC(
        ChunkPolicy(), action_horizon=3, actions_during_latency=2, handoff_blend_steps=2
    )
    broker._prev_returned_action = {"actions": np.zeros(14, dtype=np.float64)}
    actions = np.ones((50, 14), dtype=np.float64)
    actions[:, 5] = 2 * np.pi - 0.2

    blended = broker._blend_handoff({"actions": actions}, start_step=2)["actions"]

    assert blended[2, 0] == pytest.approx(0.5)
    assert blended[3, 0] == pytest.approx(1.0)
    assert blended[2, 5] == pytest.approx(-0.1)
    assert blended[3, 5] == pytest.approx(-0.2)
    np.testing.assert_array_equal(blended[4], actions[4])
    np.testing.assert_array_equal(actions[:, 0], np.ones(50))
