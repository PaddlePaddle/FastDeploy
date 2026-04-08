# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import paddle

import fastdeploy.envs as envs
import fastdeploy.metrics.trace as tracing
from fastdeploy.entrypoints.openai.serving_completion import OpenAIServingCompletion
from fastdeploy.utils import ErrorCode, ParameterError
from fastdeploy.worker.output import LogprobsLists, LogprobsTensors, SpeculateMetrics


def _make_engine_client():
    return SimpleNamespace(
        is_master=True,
        semaphore=SimpleNamespace(acquire=AsyncMock(), release=Mock()),
        data_processor=SimpleNamespace(
            process_response_dict=Mock(),
            process_logprob_response=Mock(return_value="tok"),
            tokenizer=SimpleNamespace(
                decode=Mock(return_value="decoded"), convert_ids_to_tokens=Mock(return_value="A")
            ),
        ),
        connection_manager=SimpleNamespace(get_connection=AsyncMock(), cleanup_request=AsyncMock()),
        check_model_weight_status=Mock(return_value=False),
        check_health=Mock(return_value=(True, "ok")),
        ori_vocab_size=100,
        abort=AsyncMock(),
        format_and_add_data=AsyncMock(return_value=[1, 2]),
    )


def _make_request(**overrides):
    request = SimpleNamespace(
        prompt="hi",
        prompt_token_ids=None,
        stream=False,
        n=1,
        request_id=None,
        user=None,
        trace_context={},
        model="test-model",
        logprobs=None,
        include_draft_logprobs=False,
        include_logprobs_decode_token=False,
        return_token_ids=False,
        include_stop_str_in_output=False,
        prompt_logprobs=None,
        collect_metrics=False,
        max_streaming_response_tokens=None,
        suffix=None,
        echo=False,
        stream_options=None,
    )

    def to_dict_for_infer(request_id_idx, prompt):
        return {"prompt": prompt, "request_id": request_id_idx, "prompt_tokens": [1], "max_tokens": 2, "metrics": {}}

    request.to_dict_for_infer = to_dict_for_infer
    for key, value in overrides.items():
        setattr(request, key, value)
    return request


async def _assert_error(testcase, serving, request, *, contains=None, code=None, param=None):
    res = await serving.create_completion(request)
    if contains is not None:
        testcase.assertIn(contains, res.error.message)
    if code is not None:
        testcase.assertEqual(res.error.code, code)
    if param is not None:
        testcase.assertEqual(res.error.param, param)
    return res


class _StreamRaiser(SimpleNamespace):
    @property
    def stream(self):
        raise RuntimeError("stream property error")


class TestServingCompletion(unittest.IsolatedAsyncioTestCase):
    async def test_create_completion_branches(self):
        ec = _make_engine_client()
        ec.is_master = False
        with patch("fastdeploy.entrypoints.openai.serving_completion.get_host_ip", return_value="9.9.9.9"):
            serving = OpenAIServingCompletion(ec, None, "pid", ["1.2.3.4", "5.6.7.8"], 10)
            res = await _assert_error(self, serving, _make_request(), contains="Only master node")
        self.assertIn("1.2.3.4", res.error.message)

        models = Mock()
        models.is_supported_model.return_value = (False, "bad-model")
        models.model_paths = [SimpleNamespace(name="good-model")]
        serving = OpenAIServingCompletion(_make_engine_client(), models, "pid", None, 10)
        await _assert_error(self, serving, _make_request(model="bad-model"), code=ErrorCode.MODEL_NOT_SUPPORT)

        serving = OpenAIServingCompletion(_make_engine_client(), None, "pid", None, 10)
        await _assert_error(
            self, serving, _make_request(prompt=["ok", 1], request_id="abc"), contains="If prompt is a list"
        )

        ec = _make_engine_client()
        ec.semaphore.acquire = AsyncMock(side_effect=RuntimeError("boom"))
        serving = OpenAIServingCompletion(ec, None, "pid", None, -1)
        await _assert_error(self, serving, _make_request(prompt=[1, 2], user="user1"), code=ErrorCode.TIMEOUT)

        ec = _make_engine_client()
        ec.format_and_add_data = AsyncMock(side_effect=ParameterError("max_tokens", "bad"))
        serving = OpenAIServingCompletion(ec, None, "pid", None, -1)
        with patch.object(envs, "ENABLE_V1_DATA_PROCESSOR", False):
            res = await _assert_error(self, serving, _make_request(prompt_token_ids=[1, 2]), param="max_tokens")
        ec.semaphore.release.assert_called_once()
        ec = _make_engine_client()
        ec.format_and_add_data = AsyncMock(side_effect=ValueError("bad"))
        serving = OpenAIServingCompletion(ec, None, "pid", None, -1)

        def fake_from_generic_request(_, request_id):
            return {"prompt": "hi", "request_id": request_id, "prompt_tokens": [1], "max_tokens": 2, "metrics": {}}

        with patch.object(envs, "ENABLE_V1_DATA_PROCESSOR", True):
            with patch(
                "fastdeploy.entrypoints.openai.serving_completion.Request.from_generic_request",
                side_effect=fake_from_generic_request,
            ):
                await _assert_error(self, serving, _make_request(prompt="hi"), code=ErrorCode.INVALID_VALUE)
        ec = _make_engine_client()
        ec.format_and_add_data = AsyncMock(return_value=np.array([1, 2]))
        serving = OpenAIServingCompletion(ec, None, "pid", None, -1)
        with patch.object(envs, "ENABLE_V1_DATA_PROCESSOR", False):
            with patch.object(serving, "completion_full_generator", AsyncMock(side_effect=RuntimeError("boom"))):
                await _assert_error(
                    self, serving, _make_request(prompt="hi"), contains="completion_full_generator error"
                )
        serving = OpenAIServingCompletion(_make_engine_client(), None, "pid", None, -1)
        with patch.object(envs, "ENABLE_V1_DATA_PROCESSOR", False):
            with patch.object(serving, "completion_stream_generator", return_value="streamed"):
                res = await serving.create_completion(_make_request(request_id="req123", stream=True))
        self.assertEqual(res, "streamed")
        serving = OpenAIServingCompletion(_make_engine_client(), None, "pid", None, -1)
        with patch.object(envs, "ENABLE_V1_DATA_PROCESSOR", False):
            await _assert_error(
                self, serving, _StreamRaiser(**_make_request().__dict__), contains="create_completion error"
            )

    async def test_completion_full_generator_branches(self):
        ec = _make_engine_client()
        ec.check_model_weight_status = Mock(return_value=True)
        ec.connection_manager.get_connection = AsyncMock(return_value=(Mock(), AsyncMock()))
        serving = OpenAIServingCompletion(ec, None, "pid", None, -1)
        res = await serving.completion_full_generator(_make_request(), 1, "req", 1, "m", [[1, 2]], [["p1", "p2"]], [2])
        self.assertEqual(res.error.code, ErrorCode.INVALID_VALUE)
        ec.connection_manager.cleanup_request.assert_called_once_with("req")
        ec = _make_engine_client()
        timeouts = [asyncio.TimeoutError()] * 30
        rq = AsyncMock()
        spec = SpeculateMetrics(1, 0, 1.0, 1.0, [1], [1.0])
        # fmt: off
        rq.get = AsyncMock(side_effect=timeouts + [[{"request_id": "req_0", "error_code": 200, "metrics": {"arrival_time": 1, "inference_start_time": 1, "first_token_time": 1, "engine_recv_latest_token_time": 2, "speculate_metrics": spec}, "outputs": {"token_ids": [5], "text": "ok", "top_logprobs": [[[5]], [[-0.1]], [[1]]], "draft_top_logprobs": [[[5]], [[-0.2]], [[1]]], "completion_tokens": 1, "num_cache_tokens": 0, "num_image_tokens": 0, "reasoning_token_num": 0}, "finished": True, "trace_carrier": "trace"}]])
        # fmt: on
        ec.connection_manager.get_connection = AsyncMock(return_value=(Mock(), rq))
        serving = OpenAIServingCompletion(ec, None, "pid", None, -1)
        with patch.object(asyncio, "sleep", AsyncMock()):
            with patch.object(tracing, "trace_set_proc_propagate_context", Mock()):
                with patch.object(tracing, "trace_report_span", Mock()):
                    with patch("fastdeploy.entrypoints.openai.serving_completion.trace_print", Mock()):
                        res = await serving.completion_full_generator(
                            _make_request(include_draft_logprobs=True), 1, "req", 1, "m", [[1, 2]], [["p1", "p2"]], [2]
                        )
        self.assertIsNotNone(res)
        ec.connection_manager.cleanup_request.assert_called_once_with("req")
        ec = _make_engine_client()
        rq = AsyncMock()
        rq.get = AsyncMock(return_value=[{"request_id": "req_0", "error_code": 500, "error_msg": "bad"}])
        ec.connection_manager.get_connection = AsyncMock(return_value=(Mock(), rq))
        serving = OpenAIServingCompletion(ec, None, "pid", None, -1)
        res = await serving.completion_full_generator(_make_request(), 1, "req", 1, "m", [[1, 2]], [["p1", "p2"]], [2])
        self.assertIsNone(res)
        ec.connection_manager.cleanup_request.assert_called_once_with("req")

    def test_logprobs_helpers(self):
        serving = OpenAIServingCompletion(_make_engine_client(), None, "pid", None, -1)
        token_ids = paddle.to_tensor([[1, 2]], dtype=paddle.int64)
        logprobs = paddle.to_tensor([[-0.1, -0.2]], dtype=paddle.float32)
        ranks = paddle.to_tensor([1], dtype=paddle.int64)
        res = serving._build_prompt_logprobs(LogprobsTensors(token_ids, logprobs, ranks), 2, False)
        self.assertIsNone(res[1][1].decoded_token)
        ec = _make_engine_client()
        ec.data_processor.process_logprob_response = Mock(return_value="\ufffd")
        serving = OpenAIServingCompletion(ec, None, "pid", None, -1)
        self.assertIsNone(serving._build_logprobs_response(None, request_top_logprobs=0))
        self.assertIsNone(serving._build_logprobs_response(LogprobsLists([[1]], [[-0.1]], [0]), -1))
        res = serving._build_logprobs_response(LogprobsLists([[65]], [[-0.1]], [0]), request_top_logprobs=0)
        self.assertEqual(res.tokens, ["bytes:\\x41"])
        ec = _make_engine_client()
        ec.data_processor.process_logprob_response = Mock(side_effect=RuntimeError("boom"))
        serving = OpenAIServingCompletion(ec, None, "pid", None, -1)
        self.assertIsNone(
            serving._build_logprobs_response(LogprobsLists([[1]], [[-0.1]], [0]), request_top_logprobs=0)
        )
        serving = OpenAIServingCompletion(_make_engine_client(), None, "pid", None, -1)
        res = serving._create_completion_logprobs(
            [[[1], [2]], [[-0.1], [-0.2]], [[1], [1]]], request_logprobs=0, prompt_text_offset=0
        )
        self.assertEqual(len(res.tokens), 2)
        self.assertEqual(len(res.top_logprobs), 2)

    def test_echo_and_response_usage(self):
        serving = OpenAIServingCompletion(_make_engine_client(), None, "pid", None, -1)
        self.assertEqual(serving._echo_back_prompt(_make_request(prompt=["a", "b"]), 1), "b")
        self.assertEqual(serving._echo_back_prompt(_make_request(prompt=[1, 2]), 0), "decoded")
        self.assertEqual(serving._echo_back_prompt(_make_request(prompt=[[1, 2], [3, 4]]), 1), "decoded")
        final_res_batch = [
            {
                "outputs": {
                    "token_ids": [3, 4],
                    "text": "ok",
                    "top_logprobs": None,
                    "draft_top_logprobs": None,
                    "num_cache_tokens": 1,
                    "num_image_tokens": 2,
                    "reasoning_token_num": 0,
                },
                "output_token_ids": 2,
                "metrics": {},
            }
        ]
        res = serving.request_output_to_completion_response(
            final_res_batch,
            _make_request(return_token_ids=False),
            "req",
            1,
            "m",
            [[1, 2]],
            [[3, 4]],
            [["p1", "p2"]],
            [2],
        )
        self.assertEqual(res.usage.completion_tokens, 4)
        self.assertEqual(res.usage.completion_tokens_details.image_tokens, 2)

    async def test_completion_stream_generator_paths(self):
        ec = _make_engine_client()
        ec.semaphore.release = Mock()
        rq = AsyncMock()
        # fmt: off
        # fmt: off
        rq.get = AsyncMock(return_value=[
            {"request_id": "req_0", "error_code": 200, "metrics": {"arrival_time": 1, "inference_start_time": 1, "first_token_time": 1}, "outputs": {"text": "hi", "token_ids": [10], "top_logprobs": [[[10]], [[-0.1]], [[1]]], "draft_top_logprobs": [[[10]], [[-0.2]], [[1]]], "send_idx": 0, "completion_tokens": 1, "num_cache_tokens": 1, "num_image_tokens": 2, "reasoning_token_num": 3, "tool_calls": [], "reasoning_content": "", "skipped": True}, "finished": False},
            {"request_id": "req_0", "error_code": 200, "metrics": {"arrival_time": 1, "inference_start_time": 1, "first_token_time": 1, "engine_recv_latest_token_time": 2}, "outputs": {"text": "ok", "token_ids": [11], "top_logprobs": None, "draft_top_logprobs": None, "send_idx": 1, "completion_tokens": 1, "num_cache_tokens": 0, "num_image_tokens": 0, "reasoning_token_num": 0, "tool_calls": [{"id": "t"}], "reasoning_content": "", "skipped": False}, "finished": True, "trace_carrier": "trace"},
        ])
        # fmt: on
        # fmt: on
        ec.connection_manager.get_connection = AsyncMock(return_value=(Mock(), rq))
        serving = OpenAIServingCompletion(ec, None, "pid", None, -1)
        req = _make_request(
            logprobs=1, include_draft_logprobs=True, stream_options=SimpleNamespace(include_usage=True)
        )
        with patch.object(tracing, "trace_set_proc_propagate_context", Mock()):
            with patch.object(tracing, "trace_report_span", Mock()):
                with patch("fastdeploy.entrypoints.openai.serving_completion.trace_print", Mock()):
                    # fmt: off
                    results = [item async for item in serving.completion_stream_generator(req, 1, "req", 1, "m", [[1, 2]], [["p1", "p2"]], [2])]
                    # fmt: on
        self.assertTrue(any("[DONE]" in r for r in results))
        ec.connection_manager.cleanup_request.assert_called_once_with("req")
        ec.semaphore.release.assert_called_once()

        for kind in ("weight", "timeout", "error_code"):
            ec = _make_engine_client()
            rq = AsyncMock()
            if kind == "weight":
                ec.check_model_weight_status = Mock(return_value=True)
            elif kind == "timeout":
                # fmt: off
                rq.get = AsyncMock(side_effect=[asyncio.TimeoutError()] * 30 + [[{"request_id": "req_0", "error_code": 200, "metrics": {"arrival_time": 1, "inference_start_time": 1, "first_token_time": 1}, "outputs": {"text": "ok", "token_ids": [1], "top_logprobs": None, "draft_top_logprobs": None, "send_idx": 1, "completion_tokens": 1, "num_cache_tokens": 0, "num_image_tokens": 0, "reasoning_token_num": 0, "tool_calls": None, "reasoning_content": "", "skipped": False}, "finished": True}]])
                # fmt: on
            else:
                rq.get = AsyncMock(return_value=[{"request_id": "req_0", "error_code": 500, "error_msg": "bad"}])
            ec.connection_manager.get_connection = AsyncMock(return_value=(Mock(), rq))
            serving = OpenAIServingCompletion(ec, None, "pid", None, -1)
            with patch.object(asyncio, "sleep", AsyncMock()):
                # fmt: off
                results = [item async for item in serving.completion_stream_generator(_make_request(), 1, "req", 1, "m", [[1, 2]], [["p1", "p2"]], [2])]
                # fmt: on
            if kind in ("error_code", "weight"):
                self.assertTrue(any("error" in r for r in results))
            else:
                self.assertTrue(any("[DONE]" in r for r in results))
