import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from fastdeploy.input.ernie4_5_vl_processor import Ernie4_5_VLProcessor


class MockReasoningParser:
    def get_model_status(self, prompt_token_ids):
        return "think_start"


class TestErnie4_5VLProcessorProcessResponseDictStreaming(unittest.TestCase):
    def setUp(self):
        # 创建 Ernie4_5_VLProcessor 实例的模拟对象
        with patch.object(Ernie4_5_VLProcessor, "__init__", return_value=None) as mock_init:
            self.processor = Ernie4_5_VLProcessor("model_path")
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
        self.processor.model_status_dict = {"test": "think_start"}
        self.processor.ernie4_5_processor = MagicMock()

        # 模拟 ids2tokens 方法
        def mock_ids2tokens(token_ids, task_id):
            return "delta_text", [2, 3], "previous_texts"

        self.processor.ids2tokens = mock_ids2tokens

        def mock_request2ids(request, **kwargs):
            return {"input_ids": np.array([1, 2, 3]), "prompt_token_ids": [0]}

        def mock_check_mm_limits(item):
            pass

        def mock_apply_default_parameters(request):
            return request

        def mock_pack_outputs(outputs):
            return outputs

        self.processor._apply_default_parameters = mock_apply_default_parameters
        self.processor._check_mm_limits = mock_check_mm_limits
        self.processor.ernie4_5_processor.request2ids = mock_request2ids
        self.processor.pack_outputs = mock_pack_outputs

        # 模拟推理解析器
        self.mock_reasoning_parser = MagicMock()
        self.mock_reasoning_parser.extract_reasoning_content_streaming.return_value = None
        self.processor.reasoning_parser = self.mock_reasoning_parser

        # 模拟工具解析器
        self.mock_tool_parser = MagicMock()
        self.mock_tool_parser.extract_tool_calls_streaming.return_value = None
        self.mock_tool_parser_obj = MagicMock()
        self.mock_tool_parser_obj.return_value = self.mock_tool_parser
        self.processor.tool_parser_obj = self.mock_tool_parser_obj

    def test_process_response_dict_streaming_normal_case(self):
        """测试正常情况下的流式响应处理"""
        # 准备输入
        response_dict = {"finished": False, "request_id": "test", "outputs": {"token_ids": [4, 5]}}
        kwargs = {"enable_thinking": True}

        # 调用方法
        result = self.processor.process_response_dict_streaming(response_dict, **kwargs)

        # 验证结果
        self.assertEqual(result["outputs"]["completion_tokens"], "delta_text")

    def test_process_request_dict(self):
        request_dict = {
            "request_id": "123",
            "messages": [{"role": "user", "content": "Hello!"}],
            "chat_template_kwargs": {"chat_template": "Hello!"},
            "eos_token_ids": [1],
            "temperature": 1,
            "top_p": 1,
        }
        result = self.processor.process_request_dict(request_dict, 100)
        self.assertEqual(result["prompt_token_ids"], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
