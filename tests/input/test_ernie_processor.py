import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.input.ernie_processor import ErnieProcessor


class TestErnieProcessorProcessResponseDictStreaming(unittest.TestCase):
    def setUp(self):
        # 创建 ErnieProcessor 实例的模拟对象
        with patch.object(ErnieProcessor, "__init__", return_value=None) as mock_init:
            self.processor = ErnieProcessor("model_path")
            mock_init.side_effect = lambda *args, **kwargs: print(f"__init__ called with {args}, {kwargs}")

        # 设置必要的属性
        self.processor.tokenizer = MagicMock()
        self.processor.tokenizer.eos_token_id = 1
        self.processor.decode_status = {}
        self.processor.reasoning_end_dict = {}
        self.processor.tool_parser_dict = {}

        # 模拟 ids2tokens 方法
        def mock_ids2tokens(token_ids, task_id):
            return "delta_text", [2, 3], "previous_texts"

        self.processor.ids2tokens = mock_ids2tokens

        # 模拟推理解析器
        self.mock_reasoning_parser = MagicMock()
        self.mock_reasoning_parser.__class__.__name__ = "ErnieX1ReasoningParser"
        self.mock_reasoning_parser.extract_reasoning_content_streaming.return_value = ("reasoning", "text")
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
        response_dict = {"finished": False, "request_id": "req1", "outputs": {"token_ids": [4, 5]}}
        kwargs = {"enable_thinking": True}

        # 调用方法
        result = self.processor.process_response_dict_streaming(response_dict, **kwargs)

        # 验证结果
        self.assertEqual(result["outputs"]["raw_prediction"], "delta_text")


if __name__ == "__main__":
    unittest.main()
