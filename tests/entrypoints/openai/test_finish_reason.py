import json
from typing import Any, Dict, List
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np

from fastdeploy.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    CompletionResponse,
    DeltaMessage,
    UsageInfo,
)
from fastdeploy.entrypoints.openai.serving_chat import OpenAIServingChat
from fastdeploy.entrypoints.openai.serving_completion import OpenAIServingCompletion
from fastdeploy.input.ernie4_5_vl_processor import Ernie4_5_VLProcessor
from fastdeploy.utils import data_processor_logger


class TestMultiModalProcessorMaxTokens(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        with patch.object(Ernie4_5_VLProcessor, "__init__", return_value=None):
            self.multi_modal_processor = Ernie4_5_VLProcessor("model_path")
            self.multi_modal_processor.tokenizer = MagicMock()
            self.multi_modal_processor.tokenizer.eos_token_id = 102
            self.multi_modal_processor.tokenizer.pad_token_id = 0
            self.multi_modal_processor.eos_token_ids = [102]
            self.multi_modal_processor.eos_token_id_len = 1
            self.multi_modal_processor.generation_config = MagicMock()
            self.multi_modal_processor.decode_status = {}
            self.multi_modal_processor.tool_parser_dict = {}
            self.multi_modal_processor.ernie4_5_processor = MagicMock()
            self.multi_modal_processor.ernie4_5_processor.request2ids.return_value = {
                "input_ids": np.array([101, 9012, 3456, 102])
            }
            self.multi_modal_processor.ernie4_5_processor.text2ids.return_value = {
                "input_ids": np.array([101, 1234, 5678, 102])
            }
            self.multi_modal_processor._apply_default_parameters = lambda x: x
            self.multi_modal_processor.update_stop_seq = Mock(return_value=([], []))
            self.multi_modal_processor.update_bad_words = Mock(return_value=[])
            self.multi_modal_processor._check_mm_limits = Mock()
            self.multi_modal_processor.append_completion_tokens = Mock()
            self.multi_modal_processor.pack_outputs = lambda x: x

        self.engine_client = Mock()
        self.engine_client.connection_initialized = False
        self.engine_client.connection_manager = AsyncMock()
        self.engine_client.semaphore = Mock()
        self.engine_client.semaphore.acquire = AsyncMock()
        self.engine_client.semaphore.release = Mock()
        self.engine_client.is_master = True
        self.engine_client.check_model_weight_status = Mock(return_value=False)
        self.engine_client.enable_mm = True
        self.engine_client.enable_prefix_caching = False
        self.engine_client.max_model_len = 20
        self.engine_client.data_processor = self.multi_modal_processor

        async def mock_add_data(current_req_dict):
            if current_req_dict.get("max_tokens") is None:
                current_req_dict["max_tokens"] = self.engine_client.max_model_len - 1
            current_req_dict["max_tokens"] = min(
                self.engine_client.max_model_len - 4, max(0, current_req_dict.get("max_tokens"))
            )

        self.engine_client.add_requests = AsyncMock(side_effect=mock_add_data)

        self.chat_serving = OpenAIServingChat(
            engine_client=self.engine_client,
            models=None,
            pid=123,
            ips=None,
            max_waiting_time=30,
            chat_template="default",
            enable_mm_output=True,
            tokenizer_base_url=None,
        )
        self.completion_serving = OpenAIServingCompletion(
            engine_client=self.engine_client, models=None, pid=123, ips=None, max_waiting_time=30
        )

    def _generate_inference_response(
        self, request_id: str, output_token_num: int, tool_call: Any = None
    ) -> List[Dict]:
        outputs = {
            "text": "这是一张风景图"[:output_token_num],
            "token_ids": list(range(output_token_num)),
            "reasoning_content": "推理过程",
            "num_image_tokens": 0,
            "num_cached_tokens": 0,
            "top_logprobs": None,
            "draft_top_logprobs": None,
            "tool_call": None,
        }

        if tool_call:
            outputs["tool_call"] = [
                {"index": 0, "type": "function", "function": {"name": tool_call["name"], "arguments": json.dumps({})}}
            ]

        return [
            {
                "request_id": request_id,
                "outputs": outputs,
                "metrics": {"request_start_time": 0.1},
                "finished": True,
                "error_msg": None,
                "output_token_ids": output_token_num,
            }
        ]

    def _generate_stream_inference_response(
        self, request_id: str, total_token_num: int, tool_call: Any = None
    ) -> List[Dict]:
        stream_responses = []
        for i in range(total_token_num):
            metrics = {}
            if i == 0:
                metrics["first_token_time"] = 0.1
                metrics["inference_start_time"] = 0.1
            else:
                metrics["arrival_time"] = 0.1 * (i + 1)
                metrics["first_token_time"] = None

            if i == total_token_num - 1:
                metrics["request_start_time"] = 0.1

            outputs = {
                "text": chr(ord("a") + i),
                "token_ids": [i + 1],
                "top_logprobs": None,
                "draft_top_logprobs": None,
                "reasoning_token_num": 0,
            }

            if tool_call and isinstance(tool_call, dict) and i == total_token_num - 2:
                delta_msg = DeltaMessage(
                    content="",
                    reasoning_content="",
                    tool_calls=[
                        {
                            "index": 0,
                            "type": "function",
                            "function": {"name": tool_call["name"], "arguments": json.dumps({})},
                        }
                    ],
                    prompt_token_ids=None,
                    completion_token_ids=None,
                )
                outputs["delta_message"] = delta_msg

            frame = [
                {
                    "request_id": f"{request_id}_0",
                    "error_code": 200,
                    "outputs": outputs,
                    "metrics": metrics,
                    "finished": (i == total_token_num - 1),
                    "error_msg": None,
                }
            ]
            stream_responses.append(frame)
        return stream_responses

    @patch.object(data_processor_logger, "info")
    @patch("fastdeploy.entrypoints.openai.serving_chat.ChatResponseProcessor")
    @patch("fastdeploy.entrypoints.openai.serving_chat.api_server_logger")
    async def test_chat_full_max_tokens(self, mock_data_logger, mock_processor_class, mock_api_logger):
        test_cases = [
            {
                "name": "用户传max_tokens=5，生成数=5→length",
                "request": ChatCompletionRequest(
                    model="ernie4.5-vl",
                    messages=[{"role": "user", "content": "描述这张图片"}],
                    stream=False,
                    max_tokens=5,
                    return_token_ids=True,
                ),
                "output_token_num": 5,
                "tool_call": [],
                "expected_finish_reason": "length",
            },
            {
                "name": "用户未传max_tokens，生成数=10→stop",
                "request": ChatCompletionRequest(
                    model="ernie4.5-vl",
                    messages=[{"role": "user", "content": "描述这张图片"}],
                    stream=False,
                    return_token_ids=True,
                ),
                "output_token_num": 10,
                "tool_call": [],
                "expected_finish_reason": "stop",
            },
            {
                "name": "用户未传max_tokens，生成数=16→length",
                "request": ChatCompletionRequest(
                    model="ernie4.5-vl",
                    messages=[{"role": "user", "content": "描述这张图片"}],
                    stream=False,
                    return_token_ids=True,
                ),
                "output_token_num": 16,
                "tool_call": [],
                "expected_finish_reason": "length",
            },
            {
                "name": "用户传max_tokens，生成数=10→stop",
                "request": ChatCompletionRequest(
                    model="ernie4.5-vl",
                    messages=[{"role": "user", "content": "描述这张图片"}],
                    stream=False,
                    max_tokens=50,
                    return_token_ids=True,
                ),
                "output_token_num": 10,
                "tool_call": [],
                "expected_finish_reason": "stop",
            },
            {
                "name": "生成数<max_tokens，触发tool_call→tool_calls",
                "request": ChatCompletionRequest(
                    model="ernie4.5-vl",
                    messages=[{"role": "user", "content": "描述这张图片"}],
                    stream=False,
                    max_tokens=10,
                    return_token_ids=True,
                ),
                "output_token_num": 8,
                "tool_call": {"name": "test_tool"},
                "expected_finish_reason": "tool_calls",
            },
        ]

        mock_response_queue = AsyncMock()
        mock_dealer = Mock()
        self.engine_client.connection_manager.get_connection = AsyncMock(
            return_value=(mock_dealer, mock_response_queue)
        )

        mock_processor_instance = Mock()
        mock_processor_instance.enable_multimodal_content.return_value = True

        async def mock_process_response_chat_async(response, stream, enable_thinking, include_stop_str_in_output):
            yield response

        mock_processor_instance.process_response_chat = mock_process_response_chat_async
        mock_processor_class.return_value = mock_processor_instance

        for case in test_cases:
            with self.subTest(case=case["name"]):
                request_dict = {
                    "messages": case["request"].messages,
                    "chat_template": "default",
                    "request_id": "test_chat_0",
                    "max_tokens": case["request"].max_tokens,
                }
                await self.engine_client.add_requests(request_dict)
                processed_req = self.multi_modal_processor.process_request_dict(
                    request_dict, self.engine_client.max_model_len
                )
                mock_response_queue.get.side_effect = self._generate_inference_response(
                    request_id="test_chat_0", output_token_num=case["output_token_num"], tool_call=case["tool_call"]
                )

                result = await self.chat_serving.chat_completion_full_generator(
                    request=case["request"],
                    request_id="test_chat",
                    model_name="ernie4.5-vl",
                    prompt_token_ids=processed_req["prompt_token_ids"],
                    prompt_tokens="描述这张图片",
                    max_tokens=processed_req["max_tokens"],
                )
                self.assertEqual(
                    result.choices[0].finish_reason, case["expected_finish_reason"], f"场景 {case['name']} 失败"
                )

    @patch.object(data_processor_logger, "info")
    @patch("fastdeploy.entrypoints.openai.serving_completion.api_server_logger")
    async def test_completion_full_max_tokens(self, mock_api_logger, mock_data_logger):
        test_cases = [
            {
                "name": "用户传max_tokens=6，生成数=6→length",
                "request": CompletionRequest(
                    request_id="test_completion",
                    model="ernie4.5-vl",
                    prompt="描述这张图片：<image>xxx</image>",
                    stream=False,
                    max_tokens=6,
                    return_token_ids=True,
                ),
                "output_token_num": 6,
                "expected_finish_reason": "length",
            },
            {
                "name": "用户未传max_tokens，生成数=12→stop",
                "request": CompletionRequest(
                    request_id="test_completion",
                    model="ernie4.5-vl",
                    prompt="描述这张图片：<image>xxx</image>",
                    stream=False,
                    return_token_ids=True,
                ),
                "output_token_num": 12,
                "expected_finish_reason": "stop",
            },
            {
                "name": "用户传max_tokens=20（修正为16），生成数=16→length",
                "request": CompletionRequest(
                    request_id="test_completion",
                    model="ernie4.5-vl",
                    prompt="描述这张图片：<image>xxx</image>",
                    stream=False,
                    max_tokens=20,
                    return_token_ids=True,
                ),
                "output_token_num": 16,
                "expected_finish_reason": "length",
            },
        ]

        mock_dealer = Mock()
        self.engine_client.connection_manager.get_connection = AsyncMock(return_value=(mock_dealer, AsyncMock()))

        for case in test_cases:
            with self.subTest(case=case["name"]):
                request_dict = {
                    "prompt": case["request"].prompt,
                    "request_id": "test_completion",
                    "multimodal_data": {"image": ["xxx"]},
                    "max_tokens": case["request"].max_tokens,
                }
                await self.engine_client.add_requests(request_dict)
                processed_req = self.multi_modal_processor.process_request_dict(
                    request_dict, self.engine_client.max_model_len
                )
                self.engine_client.data_processor.process_response_dict = (
                    lambda data, stream, include_stop_str_in_output: data
                )
                mock_response_queue = AsyncMock()
                mock_response_queue.get.side_effect = lambda: [
                    {
                        "request_id": "test_completion_0",
                        "error_code": 200,
                        "outputs": {
                            "text": "这是一张风景图"[: case["output_token_num"]],
                            "token_ids": list(range(case["output_token_num"])),
                            "top_logprobs": None,
                            "draft_top_logprobs": None,
                        },
                        "metrics": {"request_start_time": 0.1},
                        "finished": True,
                        "error_msg": None,
                        "output_token_ids": case["output_token_num"],
                    }
                ]
                self.engine_client.connection_manager.get_connection.return_value = (mock_dealer, mock_response_queue)

                result = await self.completion_serving.completion_full_generator(
                    request=case["request"],
                    num_choices=1,
                    request_id="test_completion",
                    created_time=1699999999,
                    model_name="ernie4.5-vl",
                    prompt_batched_token_ids=[processed_req["prompt_token_ids"]],
                    prompt_tokens_list=[case["request"].prompt],
                    max_tokens_list=[processed_req["max_tokens"]],
                )

                self.assertIsInstance(result, CompletionResponse)
                self.assertEqual(result.choices[0].finish_reason, case["expected_finish_reason"])

    @patch.object(data_processor_logger, "info")
    @patch("fastdeploy.entrypoints.openai.serving_chat.ChatResponseProcessor")
    @patch("fastdeploy.entrypoints.openai.serving_chat.api_server_logger")
    async def test_chat_stream_max_tokens(self, mock_api_logger, mock_processor_class, mock_data_logger):
        test_cases = [
            {
                "name": "流式-生成数=8（等于max_tokens）→length",
                "request": ChatCompletionRequest(
                    model="ernie4.5-vl",
                    messages=[{"role": "user", "content": "描述这张图片"}],
                    stream=True,
                    max_tokens=8,
                    return_token_ids=True,
                ),
                "total_token_num": 8,
                "tool_call": None,
                "expected_finish_reason": "length",
            },
            {
                "name": "流式-生成数=6（小于max_tokens）+tool_call→tool_calls",
                "request": ChatCompletionRequest(
                    model="ernie4.5-vl",
                    messages=[{"role": "user", "content": "描述这张图片"}],
                    stream=True,
                    max_tokens=10,
                    return_token_ids=True,
                ),
                "total_token_num": 3,
                "tool_call": {"name": "test_tool"},
                "expected_finish_reason": "tool_calls",
            },
            {
                "name": "流式-生成数=7（小于max_tokens）无tool_call→stop",
                "request": ChatCompletionRequest(
                    model="ernie4.5-vl",
                    messages=[{"role": "user", "content": "描述这张图片"}],
                    stream=True,
                    max_tokens=10,
                    return_token_ids=True,
                ),
                "total_token_num": 7,
                "tool_call": None,
                "expected_finish_reason": "stop",
            },
        ]

        mock_dealer = Mock()
        self.engine_client.connection_manager.get_connection = AsyncMock(return_value=(mock_dealer, AsyncMock()))

        mock_processor_instance = Mock()
        mock_processor_instance.enable_multimodal_content.return_value = False

        async def mock_process_response_chat_async(response, stream, enable_thinking, include_stop_str_in_output):
            if isinstance(response, list):
                for res in response:
                    yield res
            else:
                yield response

        mock_processor_instance.process_response_chat = mock_process_response_chat_async
        mock_processor_class.return_value = mock_processor_instance

        for case in test_cases:
            with self.subTest(case=case["name"]):
                request_dict = {
                    "messages": case["request"].messages,
                    "chat_template": "default",
                    "request_id": "test_chat_stream_0",
                    "max_tokens": case["request"].max_tokens,
                }
                await self.engine_client.add_requests(request_dict)
                processed_req = self.multi_modal_processor.process_request_dict(
                    request_dict, self.engine_client.max_model_len
                )

                self.engine_client.data_processor.process_response_dict = (
                    lambda data, stream, include_stop_str_in_output: data
                )

                mock_response_queue = AsyncMock()
                stream_responses = self._generate_stream_inference_response(
                    request_id="test_chat_stream_0_0",
                    total_token_num=case["total_token_num"],
                    tool_call=case["tool_call"],
                )
                mock_response_queue.get.side_effect = stream_responses
                self.engine_client.connection_manager.get_connection.return_value = (mock_dealer, mock_response_queue)

                generator = self.chat_serving.chat_completion_stream_generator(
                    request=case["request"],
                    request_id="test_chat_stream_0",
                    model_name="ernie4.5-vl",
                    prompt_token_ids=processed_req["prompt_token_ids"],
                    prompt_tokens="描述这张图片",
                    max_tokens=processed_req["max_tokens"],
                )

                final_finish_reason = None
                chunks = []
                async for chunk in generator:
                    chunks.append(chunk)
                    if "[DONE]" in chunk:
                        break

                for chunk_str in chunks:
                    if chunk_str.startswith("data: ") and "[DONE]" not in chunk_str:
                        try:
                            json_part = chunk_str.strip().lstrip("data: ").rstrip("\n\n")
                            chunk_dict = json.loads(json_part)
                            if chunk_dict.get("choices") and len(chunk_dict["choices"]) > 0:
                                finish_reason = chunk_dict["choices"][0].get("finish_reason")
                                if finish_reason:
                                    final_finish_reason = finish_reason
                                    break
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

                self.assertEqual(final_finish_reason, case["expected_finish_reason"])

    @patch.object(data_processor_logger, "info")
    @patch("fastdeploy.entrypoints.openai.serving_completion.api_server_logger")
    async def test_completion_stream_max_tokens(self, mock_api_logger, mock_data_logger):
        test_cases = [
            {
                "name": "流式-生成数=7（等于max_tokens）→length",
                "request": CompletionRequest(
                    model="ernie4.5-vl",
                    prompt=["描述这张图片：<image>xxx</image>"],
                    stream=True,
                    max_tokens=7,
                    return_token_ids=True,
                ),
                "total_token_num": 7,
                "expected_finish_reason": "length",
            },
            {
                "name": "流式-生成数=9（小于max_tokens）→stop",
                "request": CompletionRequest(
                    model="ernie4.5-vl",
                    prompt=["描述这张图片：<image>xxx</image>"],
                    stream=True,
                    max_tokens=15,
                    return_token_ids=True,
                ),
                "total_token_num": 9,
                "expected_finish_reason": "stop",
            },
        ]

        mock_dealer = Mock()
        self.engine_client.connection_manager.get_connection = AsyncMock(return_value=(mock_dealer, AsyncMock()))

        for case in test_cases:
            with self.subTest(case=case["name"]):
                request_dict = {
                    "prompt": case["request"].prompt,
                    "multimodal_data": {"image": ["xxx"]},
                    "request_id": "test_completion_stream_0",
                    "max_tokens": case["request"].max_tokens,
                }
                await self.engine_client.add_requests(request_dict)
                processed_req = self.multi_modal_processor.process_request_dict(
                    request_dict, self.engine_client.max_model_len
                )
                self.engine_client.data_processor.process_response_dict = (
                    lambda data, stream, include_stop_str_in_output: data
                )

                mock_response_queue = AsyncMock()
                stream_responses = self._generate_stream_inference_response(
                    request_id="test_completion_stream_0", total_token_num=case["total_token_num"]
                )
                mock_response_queue.get.side_effect = stream_responses
                self.engine_client.connection_manager.get_connection.return_value = (mock_dealer, mock_response_queue)

                generator = self.completion_serving.completion_stream_generator(
                    request=case["request"],
                    num_choices=1,
                    created_time=0,
                    request_id="test_completion_stream",
                    model_name="ernie4.5-vl",
                    prompt_batched_token_ids=[processed_req["prompt_token_ids"]],
                    prompt_tokens_list=case["request"].prompt,
                    max_tokens_list=[processed_req["max_tokens"]],
                )

                final_finish_reason = None
                chunks = []
                async for chunk in generator:
                    chunks.append(chunk)
                    if "[DONE]" in chunk:
                        break

                for chunk_str in chunks:
                    if chunk_str.startswith("data: ") and "[DONE]" not in chunk_str:
                        try:
                            json_part = chunk_str.strip().lstrip("data: ")
                            chunk_dict = json.loads(json_part)
                            if chunk_dict["choices"][0].get("finish_reason"):
                                final_finish_reason = chunk_dict["choices"][0]["finish_reason"]
                                break
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

                self.assertEqual(final_finish_reason, case["expected_finish_reason"], f"场景 {case['name']} 失败")

    @patch.object(data_processor_logger, "info")
    @patch("fastdeploy.entrypoints.openai.serving_completion.api_server_logger")
    async def test_completion_create_max_tokens_list_basic(self, mock_api_logger, mock_data_logger):
        test_cases = [
            {
                "name": "单prompt → max_tokens_list长度1",
                "request": CompletionRequest(
                    request_id="test_single_prompt",
                    model="ernie4.5-vl",
                    prompt="请介绍人工智能的应用",
                    stream=False,
                    max_tokens=10,
                ),
                "mock_max_tokens": 8,
                "expected_max_tokens_list_len": 1,
                "expected_max_tokens_list": [8],
            },
            {
                "name": "多prompt → max_tokens_list长度2",
                "request": CompletionRequest(
                    request_id="test_multi_prompt",
                    model="ernie4.5-vl",
                    prompt=["请介绍Python语言", "请说明机器学习的步骤"],
                    stream=False,
                    max_tokens=15,
                ),
                "mock_max_tokens": [12, 10],
                "expected_max_tokens_list_len": 2,
                "expected_max_tokens_list": [12, 10],
            },
        ]

        async def mock_format_and_add_data(current_req_dict):
            req_idx = int(current_req_dict["request_id"].split("_")[-1])
            if isinstance(case["mock_max_tokens"], list):
                current_req_dict["max_tokens"] = case["mock_max_tokens"][req_idx]
            else:
                current_req_dict["max_tokens"] = case["mock_max_tokens"]
            return [101, 102, 103, 104]

        self.engine_client.format_and_add_data = AsyncMock(side_effect=mock_format_and_add_data)

        async def intercept_generator(**kwargs):
            actual_max_tokens_list = kwargs["max_tokens_list"]
            self.assertEqual(
                len(actual_max_tokens_list),
                case["expected_max_tokens_list_len"],
                f"列表长度不匹配：实际{len(actual_max_tokens_list)}，预期{case['expected_max_tokens_list_len']}",
            )
            self.assertEqual(
                actual_max_tokens_list,
                case["expected_max_tokens_list"],
                f"列表元素不匹配：实际{actual_max_tokens_list}，预期{case['expected_max_tokens_list']}",
            )
            return CompletionResponse(
                id=kwargs["request_id"],
                object="text_completion",
                created=kwargs["created_time"],
                model=kwargs["model_name"],
                choices=[],
                usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            )

        self.completion_serving.completion_full_generator = AsyncMock(side_effect=intercept_generator)

        for case in test_cases:
            with self.subTest(case=case["name"]):
                result = await self.completion_serving.create_completion(request=case["request"])
                self.assertIsInstance(result, CompletionResponse)
