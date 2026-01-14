"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import paddle
import pytest
import zmq

if not hasattr(paddle, "compat"):

    class _CompatStub:
        def enable_torch_proxy(self, scope=None):
            return None

    paddle.compat = _CompatStub()

from fastdeploy import envs
from fastdeploy.engine.request import Request, RequestMetrics, RequestOutput
from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.splitwise.splitwise_connector import SplitwiseConnector


@dataclass
class DummyParallelConfig:
    local_data_parallel_id: int = 0
    data_parallel_size: int = 1


@dataclass
class DummySchedulerConfig:
    splitwise_role: str = "mixed"


@dataclass
class DummyCacheConfig:
    local_pd_comm_port: int = 12345


@dataclass
class DummyCfg:
    parallel_config: DummyParallelConfig = field(default_factory=DummyParallelConfig)
    scheduler_config: DummySchedulerConfig = field(default_factory=DummySchedulerConfig)
    cache_config: DummyCacheConfig = field(default_factory=DummyCacheConfig)


class DummyWorkerQueue:
    def __init__(self) -> None:
        self.cache_info_calls: List[List[Dict[str, Any]]] = []
        self.disaggregated_calls: List[Any] = []

    def put_cache_info(self, cache_info: List[Dict[str, Any]]) -> None:
        self.cache_info_calls.append(cache_info)

    def put_disaggregated_tasks(self, payload: Any) -> None:
        self.disaggregated_calls.append(payload)


class DummyTask:
    def __init__(self, request_id: str, disaggregate_info: Dict[str, Any], error_msg: str | None = None) -> None:
        self.request_id = request_id
        self.disaggregate_info = disaggregate_info
        self._error_msg = error_msg

    def get(self, key: str, default: Any = None) -> Any:
        if key == "error_msg":
            return self._error_msg
        return default


def _build_connector() -> SplitwiseConnector:
    connector = SplitwiseConnector(cfg=DummyCfg(), worker_queue=DummyWorkerQueue(), resource_manager=None)
    if not hasattr(connector, "push_sockets"):
        connector.push_sockets = {}
    return connector


def test_serialize_deserialize_prefill_roundtrip_uses_paddle_tensor():
    connector = _build_connector()
    token_tensor = paddle.to_tensor([11, 12, 13], dtype="int64")
    request = Request(
        request_id="req-1",
        prompt="hello",
        prompt_token_ids=token_tensor.tolist(),
        prompt_token_ids_len=3,
        messages=None,
        history=None,
        tools=None,
        system=None,
        eos_token_ids=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
        multimodal_inputs=None,
        multimodal_data=None,
        disable_chat_template=False,
        disaggregate_info={"decode_ip": "127.0.0.1", "decode_connector_port": 9000},
        metrics=RequestMetrics(),
    )

    serialized = connector._serialize_message("prefill", [request])
    msg_type, payload = connector._deserialize_message([b"identity"] + serialized)

    assert msg_type == "prefill"
    assert payload[0]["request_id"] == "req-1"
    assert payload[0]["prompt_token_ids"] == token_tensor.tolist()


def test_deserialize_message_rejects_short_frames():
    connector = _build_connector()
    with pytest.raises(ValueError, match="frames too short"):
        connector._deserialize_message([b"identity"])


def test_process_message_cache_sync_updates_state_and_cache_queue():
    connector = _build_connector()
    worker_queue = connector.engine_worker_queue
    payload = [
        {"request_id": "req-ok"},
        {"request_id": "req-error", "error_msg": "bad"},
    ]

    frames = [b"identity"] + connector._serialize_message("cache_sync", payload)
    connector._process_message(frames)

    assert connector.current_request_ids["req-ok"] == "finished"
    assert connector.current_request_ids["req-error"] == "bad"
    assert worker_queue.cache_info_calls == [payload]


def test_check_decode_allocated_handles_finished_and_error_states():
    connector = _build_connector()

    finished_task = Request(
        request_id="req-finished",
        prompt=None,
        prompt_token_ids=None,
        prompt_token_ids_len=None,
        messages=None,
        history=None,
        tools=None,
        system=None,
        eos_token_ids=None,
        sampling_params=None,
        pooling_params=None,
        multimodal_inputs=None,
        multimodal_data=None,
        disable_chat_template=False,
        disaggregate_info={},
    )
    connector.current_request_ids["req-finished"] = "finished"
    ok, msg = connector.check_decode_allocated(finished_task)
    assert (ok, msg) == (True, "")
    assert "req-finished" not in connector.current_request_ids

    error_task = Request(
        request_id="req-error",
        prompt=None,
        prompt_token_ids=None,
        prompt_token_ids_len=None,
        messages=None,
        history=None,
        tools=None,
        system=None,
        eos_token_ids=None,
        sampling_params=None,
        pooling_params=None,
        multimodal_inputs=None,
        multimodal_data=None,
        disable_chat_template=False,
        disaggregate_info={},
    )
    connector.current_request_ids["req-error"] = "allocation_failed"
    ok, msg = connector.check_decode_allocated(error_task)
    assert (ok, msg) == (False, "allocation_failed")
    assert "req-error" not in connector.current_request_ids


def test_send_cache_info_to_prefill_groups_by_addr_and_skips_error():
    connector = _build_connector()
    connector._send_message = Mock()

    tasks = [
        DummyTask(
            request_id="req-1",
            disaggregate_info={
                "prefill_ip": "10.0.0.1",
                "prefill_connector_port": 9001,
                "block_tables": [1, 2, 3],
            },
        ),
        DummyTask(
            request_id="req-err",
            disaggregate_info={
                "prefill_ip": "10.0.0.2",
                "prefill_connector_port": 9002,
                "block_tables": [9],
            },
            error_msg="failed",
        ),
    ]

    connector.send_cache_info_to_prefill(tasks)

    connector._send_message.assert_called_once_with(
        "10.0.0.1:9001",
        "cache_sync",
        [{"request_id": "req-1", "dest_block_ids": [1, 2, 3]}],
    )


def test_init_network_configures_router_and_poller():
    connector = _build_connector()
    mock_socket = Mock()
    mock_poller = Mock()
    connector.zmq_ctx = Mock()
    connector.zmq_ctx.socket.return_value = mock_socket

    with patch("fastdeploy.splitwise.splitwise_connector.zmq.Poller", return_value=mock_poller):
        connector._init_network()

    mock_socket.bind.assert_called_once_with("tcp://*:12345")
    mock_poller.register.assert_called_once_with(mock_socket, zmq.POLLIN)
    assert connector.prefill_cache_info == []


def test_init_non_mixed_creates_network_state():
    cfg = DummyCfg(
        parallel_config=DummyParallelConfig(local_data_parallel_id=1, data_parallel_size=2),
        scheduler_config=DummySchedulerConfig(splitwise_role="prefill"),
    )
    with patch.object(SplitwiseConnector, "_init_network") as mock_init:
        connector = SplitwiseConnector(cfg=cfg, worker_queue=DummyWorkerQueue(), resource_manager=None)

    assert connector.local_data_parallel_id == 1
    assert connector.pull_socket is None
    assert connector.push_sockets == {}
    mock_init.assert_called_once_with()


def test_get_push_socket_reuses_existing_and_handles_zmq_error():
    connector = _build_connector()
    open_socket = Mock()
    open_socket.closed = False
    connector.push_sockets["127.0.0.1:8000"] = open_socket

    same_socket = connector._get_push_socket("127.0.0.1:8000")
    assert same_socket is open_socket

    connector.zmq_ctx = Mock()
    connector.zmq_ctx.socket.side_effect = zmq.ZMQError("boom")
    with pytest.raises(ConnectionError, match="Failed to connect"):
        connector._get_push_socket("127.0.0.1:9000")


def test_get_push_socket_creates_and_configures_socket():
    connector = _build_connector()
    connector.zmq_ctx = Mock()
    new_socket = Mock()
    new_socket.closed = False
    connector.zmq_ctx.socket.return_value = new_socket

    socket = connector._get_push_socket("127.0.0.1:7000")

    assert socket is new_socket
    new_socket.connect.assert_called_once_with("tcp://127.0.0.1:7000")
    assert connector.push_sockets["127.0.0.1:7000"] is new_socket


def test_send_message_serializes_and_sends_payload():
    connector = _build_connector()
    mock_socket = Mock()
    connector._get_push_socket = Mock(return_value=mock_socket)
    request = Request(
        request_id="req-send",
        prompt=None,
        prompt_token_ids=[1, 2],
        prompt_token_ids_len=2,
        messages=None,
        history=None,
        tools=None,
        system=None,
        eos_token_ids=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
        multimodal_inputs=None,
        multimodal_data=None,
        disable_chat_template=False,
        disaggregate_info={},
        metrics=RequestMetrics(),
    )

    connector._send_message("127.0.0.1:9000", "prefill", [request])

    mock_socket.send_multipart.assert_called_once()
    sent_frames = mock_socket.send_multipart.call_args[0][0]
    msg_type, payload = connector._deserialize_message([b"identity"] + sent_frames)
    assert msg_type == "prefill"
    assert payload[0]["request_id"] == "req-send"


def test_send_message_handles_missing_addr_and_errors():
    connector = _build_connector()
    connector._send_message(None, "prefill", [])

    connector._get_push_socket = Mock(side_effect=ConnectionError)
    connector._send_message("127.0.0.1:7000", "prefill", [])

    failing_socket = Mock()
    failing_socket.send_multipart.side_effect = zmq.Again()
    connector._get_push_socket = Mock(return_value=failing_socket)
    connector._send_message("127.0.0.1:7001", "prefill", [])

    crash_socket = Mock()
    crash_socket.send_multipart.side_effect = RuntimeError("boom")
    connector._get_push_socket = Mock(return_value=crash_socket)
    connector.push_sockets["127.0.0.1:7002"] = crash_socket
    connector._send_message("127.0.0.1:7002", "prefill", [])
    assert "127.0.0.1:7002" not in connector.push_sockets


def test_send_splitwise_tasks_updates_roles_and_tracks_ids():
    connector = _build_connector()
    connector._send_message = Mock()
    task = Request(
        request_id="req-role",
        prompt=None,
        prompt_token_ids=[0],
        prompt_token_ids_len=1,
        messages=None,
        history=None,
        tools=None,
        system=None,
        eos_token_ids=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
        multimodal_inputs=None,
        multimodal_data=None,
        disable_chat_template=False,
        disaggregate_info={"decode_ip": "127.0.0.1", "decode_connector_port": 9001},
        metrics=RequestMetrics(),
    )

    connector.send_splitwise_tasks([task], current_id=0)

    assert connector.current_request_ids["req-role"] == "init"
    connector._send_message.assert_called_once()
    assert task.disaggregate_info["role"] == "prefill"


def test_send_splitwise_tasks_skips_missing_disaggregate_info():
    connector = _build_connector()
    connector._send_message = Mock()
    task = Request(
        request_id="req-skip",
        prompt=None,
        prompt_token_ids=[0],
        prompt_token_ids_len=1,
        messages=None,
        history=None,
        tools=None,
        system=None,
        eos_token_ids=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
        multimodal_inputs=None,
        multimodal_data=None,
        disable_chat_template=False,
        disaggregate_info=None,
        metrics=RequestMetrics(),
    )

    connector.send_splitwise_tasks([task], current_id=0)
    connector._send_message.assert_not_called()


def test_send_cache_info_to_messager_handles_v1_and_v0_modes(monkeypatch):
    connector = _build_connector()
    worker_queue = connector.engine_worker_queue

    class _Task:
        def __init__(self, request_id: str, idx: int, disaggregate_info: Dict[str, Any]):
            self.request_id = request_id
            self.idx = idx
            self.block_tables = [1, 2]
            self.need_prefill_tokens = 5
            self.disaggregate_info = disaggregate_info

    task = _Task("req-cache", 7, {"decode_ip": "1.1.1.1"})

    monkeypatch.setattr(envs, "ENABLE_V1_KVCACHE_SCHEDULER", False)
    connector.send_cache_info_to_messager([task], current_id=3)
    assert worker_queue.cache_info_calls[-1][0]["current_id"] == 3

    monkeypatch.setattr(envs, "ENABLE_V1_KVCACHE_SCHEDULER", True)
    connector.send_cache_info_to_messager([task], current_id=9)
    latest_call = worker_queue.cache_info_calls[-1][0]
    assert latest_call["current_id"] == 7
    assert latest_call["need_prefill_tokens"] == 5

    task_without_info = _Task("req-empty", 1, None)
    connector.send_cache_info_to_messager([task_without_info], current_id=1)
    assert worker_queue.cache_info_calls[-1] == []


def test_process_message_prefill_and_decode_dispatches_to_worker_queue():
    connector = _build_connector()
    worker_queue = connector.engine_worker_queue
    request = Request(
        request_id="req-prefill",
        prompt="hi",
        prompt_token_ids=[1],
        prompt_token_ids_len=1,
        messages=None,
        history=None,
        tools=None,
        system=None,
        eos_token_ids=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
        multimodal_inputs=None,
        multimodal_data=None,
        disable_chat_template=False,
        disaggregate_info={},
        metrics=RequestMetrics(),
    )
    decode_output = RequestOutput(request_id="req-decode")

    prefill_frames = [b"id"] + connector._serialize_message("prefill", [request])
    connector._process_message(prefill_frames)
    decode_frames = [b"id"] + connector._serialize_message("decode", [decode_output])
    connector._process_message(decode_frames)

    assert worker_queue.disaggregated_calls[0][0] == "decode"
    assert worker_queue.disaggregated_calls[0][1][0].request_id == "req-prefill"
    assert worker_queue.disaggregated_calls[1][1][0].request_id == "req-decode"


def test_process_message_handles_cache_sync_with_decode_cache_task():
    connector = _build_connector()
    connector.enable_decode_cache_task = True
    payload = [{"request_id": "req-cache"}]
    frames = [b"identity"] + connector._serialize_message("cache_sync", payload)
    connector._process_message(frames)
    assert connector.current_request_ids == {}


def test_process_message_logs_error_on_bad_frames():
    connector = _build_connector()
    connector.logger = Mock()
    connector._process_message([b"only-one-frame"])
    connector.logger.error.assert_called_once()


def test_check_decode_allocated_times_out(monkeypatch):
    connector = _build_connector()
    task = Request(
        request_id="req-timeout",
        prompt=None,
        prompt_token_ids=None,
        prompt_token_ids_len=None,
        messages=None,
        history=None,
        tools=None,
        system=None,
        eos_token_ids=None,
        sampling_params=None,
        pooling_params=None,
        multimodal_inputs=None,
        multimodal_data=None,
        disable_chat_template=False,
        disaggregate_info={},
    )
    connector.current_request_ids["req-timeout"] = "init"

    monkeypatch.setattr(envs, "FD_PREFILL_WAIT_DECODE_RESOURCE_SECONDS", 0)
    monkeypatch.setattr("fastdeploy.splitwise.splitwise_connector.time.sleep", lambda *_: None)

    ok, msg = connector.check_decode_allocated(task)
    assert (ok, msg) == (False, "prefill waits for decode resource timeout")
    assert "req-timeout" not in connector.current_request_ids


def test_check_decode_allocated_returns_immediately_for_empty_or_cached():
    connector = _build_connector()
    no_info_task = Request(
        request_id="req-none",
        prompt=None,
        prompt_token_ids=None,
        prompt_token_ids_len=None,
        messages=None,
        history=None,
        tools=None,
        system=None,
        eos_token_ids=None,
        sampling_params=None,
        pooling_params=None,
        multimodal_inputs=None,
        multimodal_data=None,
        disable_chat_template=False,
        disaggregate_info=None,
    )
    ok, msg = connector.check_decode_allocated(no_info_task)
    assert (ok, msg) == (True, "")

    connector.enable_decode_cache_task = True
    cache_task = Request(
        request_id="req-cache",
        prompt=None,
        prompt_token_ids=None,
        prompt_token_ids_len=None,
        messages=None,
        history=None,
        tools=None,
        system=None,
        eos_token_ids=None,
        sampling_params=None,
        pooling_params=None,
        multimodal_inputs=None,
        multimodal_data=None,
        disable_chat_template=False,
        disaggregate_info={},
    )
    ok, msg = connector.check_decode_allocated(cache_task)
    assert (ok, msg) == (True, "")


def test_send_first_token_wraps_task_list():
    connector = _build_connector()
    connector._send_message = Mock()
    task = Request(
        request_id="req-token",
        prompt=None,
        prompt_token_ids=[1],
        prompt_token_ids_len=1,
        messages=None,
        history=None,
        tools=None,
        system=None,
        eos_token_ids=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
        multimodal_inputs=None,
        multimodal_data=None,
        disable_chat_template=False,
        disaggregate_info={},
        metrics=RequestMetrics(),
    )

    connector.send_first_token({"decode_ip": "1.2.3.4", "decode_connector_port": 7777}, task)
    connector._send_message.assert_called_once_with("1.2.3.4:7777", "decode", [task])
