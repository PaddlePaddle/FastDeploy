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

import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.engine.request import Request, RequestOutput
from fastdeploy.input.v1.ernie4_5_processor import Ernie4_5Processor


class MockReasoningParser:
    def get_model_status(self, prompt_token_ids):
        return "think_start"


class TestErnie4_5ProcessorProcessResponseDictStreaming(unittest.TestCase):
    def setUp(self):
        # 创建 Ernie4_5Processor 实例的模拟对象
        with patch.object(Ernie4_5Processor, "__init__", return_value=None) as mock_init:
            self.processor = Ernie4_5Processor("model_path")
            mock_init.side_effect = lambda *args, **kwargs: print(f"__init__ called with {args}, {kwargs}")

        # 设置必要的属性
        self.processor.tokenizer = MagicMock()
        self.processor.tokenizer.eos_token_id = 1
        self.processor.decode_status = {"test": []}
        self.processor.reasoning_end_dict = {}
        self.processor.tool_parser_dict = {}
        self.processor.generation_config = MagicMock()
        self.processor.eos_token_ids = [1]
        self.processor.reasoning_parser = MockReasoningParser()
        self.processor.model_status_dict = {"request-id_0": "think_start", "test": "think_start"}

        # 模拟 ids2tokens 方法
        def mock_ids2tokens(token_ids, task_id):
            self.processor.decode_status[task_id] = "mock_decode_status"
            return "delta_text", [2, 3], "previous_texts"

        self.processor.ids2tokens = mock_ids2tokens

        def mock_messages2ids(request, **kwargs):
            if "chat_template" in kwargs:
                return [1]
            else:
                return [0]

        def mock_apply_default_parameters(request):
            return request

        self.processor.messages2ids = mock_messages2ids
        self.processor._apply_default_parameters = mock_apply_default_parameters

        # 模拟推理解析器
        self.mock_reasoning_parser = MagicMock()
        self.mock_reasoning_parser.__class__.__name__ = "ErnieX1ReasoningParser"
        # self.mock_reasoning_parser.extract_reasoning_content_streaming.return_value = ("reasoning", "text")
        self.processor.reasoning_parser = self.mock_reasoning_parser

        # 模拟工具解析器
        self.mock_tool_parser = MagicMock()
        self.mock_tool_parser.extract_tool_calls_streaming.return_value = None
        self.mock_tool_parser_obj = MagicMock()
        self.mock_tool_parser_obj.return_value = self.mock_tool_parser
        self.processor.tool_parser_obj = self.mock_tool_parser_obj

    def test_process_response_obj_streaming_normal_case(self):
        """测试正常情况下的流式响应处理"""
        # 准备输入
        response_dict = {"finished": False, "request_id": "test", "outputs": {"token_ids": [4, 5]}}
        kwargs = {"enable_thinking": True}
        response = RequestOutput.from_dict(response_dict)

        # 调用方法
        result = self.processor.process_response_obj_streaming(response, **kwargs)

        # 验证结果
        self.assertEqual(result.outputs.completion_tokens, "delta_text")

    def test_process_request_dict(self):
        request_dict = {
            "request_id": "123",
            "messages": [{"role": "user", "content": "Hello!"}],
            "chat_template_kwargs": {"chat_template": "Hello!"},
            "eos_token_ids": [1],
            "temperature": 1,
            "top_p": 1,
        }
        request = Request.from_dict(request_dict)
        request.chat_template_kwargs = {"chat_template": "Hello!"}
        result = self.processor.process_request_dict(request, 100)
        self.assertEqual(result.prompt_token_ids, [1])

    def test_process_response_obj_normal(self):
        mock_tokens = ["reasoning", "token", "list"]
        self.processor.tokenizer.tokenize = MagicMock(return_value=mock_tokens)
        self.processor.reasoning_parser.extract_reasoning_content = MagicMock(
            return_value=("Mock reasoning content", "Mock final text")
        )

        self.processor.tool_parser_obj = None

        response_dict = {
            "request_id": "request-id_0",
            "outputs": {"token_ids": [2, 3, 4, 5, 1], "text": "Initial text", "top_logprobs": []},
            # "finish_reason": "stop",
            "finished": True,
        }
        response = RequestOutput.from_dict(response_dict)
        kwargs = {"enable_thinking": True}

        with patch("fastdeploy.input.ernie4_5_processor.data_processor_logger"):
            result = self.processor.process_response_obj_normal(response, **kwargs)

        self.mock_reasoning_parser.extract_reasoning_content.assert_called_once()
        self.assertEqual(result.outputs.reasoning_content, "Mock reasoning content")
        self.assertEqual(result.outputs.reasoning_token_num, len(mock_tokens))
        self.assertEqual(result.outputs.text, "Mock final text")
        self.assertTrue(hasattr(result.outputs, "completion_tokens"))

    def test_think_status(self):
        """测试 思考机制"""
        request = {
            "prompt": "hello",
            "request_id": "test_1",
            "prompt_token_ids": [1, 2, 3],
            "temperature": 0.7,
            "top_p": 0.9,
        }
        request = Request.from_dict(request)
        self.processor.reasoning_parser = MagicMock()
        self.processor.reasoning_parser.get_model_status.return_value = "think_start"
        self.processor.model_status_dict = {}
        self.processor.process_request_dict(request, max_model_len=512)
        self.assertEqual(request.enable_thinking, True)

        request = {
            "prompt": "hello",
            "request_id": "test",
            "prompt_token_ids": [1, 2, 3],
            "temperature": 0.7,
            "top_p": 0.9,
        }
        request = Request.from_dict(request)
        self.processor.process_request_dict(request, max_model_len=512)
        self.assertEqual(request.enable_thinking, True)


if __name__ == "__main__":
    unittest.main()
