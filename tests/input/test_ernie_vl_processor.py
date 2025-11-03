import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.input.ernie4_5_vl_processor import Ernie4_5_VLProcessor


class TestErnie4_5_vl_ProcessorProcessResponseDictStreaming(unittest.TestCase):
    def setUp(self):
        # 创建 Ernie4_5Processor 实例的模拟对象
        with patch.object(Ernie4_5_VLProcessor, "__init__", return_value=None) as mock_init:
            self.processor = Ernie4_5_VLProcessor("model_path")
            mock_init.side_effect = lambda *args, **kwargs: print(f"__init__ called with {args}, {kwargs}")

        # 设置必要的属性
        self.processor.tokenizer = MagicMock()
        self.processor.tokenizer.eos_token_id = 1
        self.processor.decode_status = {}
        self.processor.reasoning_end_dict = {}
        self.processor.tool_parser_dict = {}
        self.processor.generation_config = MagicMock()
        self.processor.eos_token_ids = [1]
        self.processor.reasoning_parser = MagicMock()
        self.processor._check_mm_limits = MagicMock()
        self.processor.ernie4_5_processor = MagicMock()
        self.processor.pack_outputs = MagicMock()

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

    def test_process_request_dict_with_options(self):
        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "prompt_token_ids": [1, 1, 1],
        }
        self.processor.process_request_dict(request_dict, 100)
        self.assertEqual(request_dict["enable_thinking"], True)

        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"enable_thinking": True},
            "prompt_token_ids": [1, 1, 1],
        }
        self.processor.process_request_dict(request_dict, 100)
        self.assertEqual(request_dict["enable_thinking"], True)

        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"enable_thinking": False},
            "prompt_token_ids": [1, 1, 1],
        }
        self.processor.process_request_dict(request_dict, 100)
        self.assertEqual(request_dict["enable_thinking"], False)

        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"options": {"thinking_mode": "open"}},
            "prompt_token_ids": [1, 1, 1],
        }
        self.processor.process_request_dict(request_dict, 100)
        self.assertEqual(request_dict["enable_thinking"], True)

        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"options": {"thinking_mode": "close"}},
            "prompt_token_ids": [1, 1, 1],
        }
        self.processor.process_request_dict(request_dict, 100)
        self.assertEqual(request_dict["enable_thinking"], False)

        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"options": {"thinking_mode": "false"}},
            "prompt_token_ids": [1, 1, 1],
        }
        self.processor.process_request_dict(request_dict, 100)
        self.assertEqual(request_dict["enable_thinking"], False)

        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"options": {"thinking_mode": "123"}},
            "prompt_token_ids": [1, 1, 1],
        }
        self.processor.process_request_dict(request_dict, 100)
        self.assertEqual(request_dict["enable_thinking"], True)


if __name__ == "__main__":
    unittest.main()
