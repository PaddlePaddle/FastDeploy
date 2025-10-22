import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.input.ernie4_5_processor import Ernie4_5Processor


class TestErnie4_5ProcessorProcessResponseDictStreaming(unittest.TestCase):
    def setUp(self):
        # 创建 Ernie4_5Processor 实例的模拟对象
        with patch.object(Ernie4_5Processor, "__init__", return_value=None) as mock_init:
            self.processor = Ernie4_5Processor("model_path")
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

    def test_process_response_dict_streaming_normal_case(self):
        """测试正常情况下的流式响应处理"""
        # 准备输入
        response_dict = {"finished": False, "request_id": "req1", "outputs": {"token_ids": [4, 5]}}
        kwargs = {"enable_thinking": True}

        # 调用方法
        result = self.processor.process_response_dict_streaming(response_dict, **kwargs)

        # 验证结果
        self.assertEqual(result["outputs"]["completion_tokens"], "delta_text")

    def test_process_request_dict(self):
        request_dict = {
            "messages": [{"role": "user", "content": "Hello!"}],
            "chat_template_kwargs": {"chat_template": "Hello!"},
            "eos_token_ids": [1],
            "temperature": 1,
            "top_p": 1,
        }
        result = self.processor.process_request_dict(request_dict, 100)
        self.assertEqual(result["prompt_token_ids"], [1])

    def test_process_response_dict(self):
        """测试 process_response_dict 根据 stream 参数调用正确的子方法"""
        response_dict = {"finished": True, "request_id": "req2", "outputs": {"token_ids": [4, 5]}}

        # 模拟两个子方法
        self.processor.process_response_dict_streaming = MagicMock(return_value={"result": "stream"})
        self.processor.process_response_dict_normal = MagicMock(return_value={"result": "normal"})

        # 情况1：stream=True
        result_stream = self.processor.process_response_dict(response_dict, stream=True)
        self.processor.process_response_dict_streaming.assert_called_once_with(response_dict)
        self.assertEqual(result_stream["result"], "stream")

        # 情况2：stream=False
        result_normal = self.processor.process_response_dict(response_dict, stream=False)
        self.processor.process_response_dict_normal.assert_called_once_with(response_dict)
        self.assertEqual(result_normal["result"], "normal")

    def test_process_response(self):
        """测试 process_response 对完整响应的处理逻辑"""
        # 构造 mock response_dict 对象
        mock_outputs = MagicMock()
        mock_outputs.token_ids = [10, 20, self.processor.tokenizer.eos_token_id]
        mock_outputs.index = 2
        mock_response_dict = MagicMock()
        mock_response_dict.request_id = "req3"
        mock_response_dict.outputs = mock_outputs

        # 模拟 tokenizer.decode
        self.processor.tokenizer.decode = MagicMock(return_value="decoded_text")

        # 模拟 reasoning_parser
        mock_reasoning_parser = MagicMock()
        mock_reasoning_parser.extract_reasoning_content.return_value = ("reasoning_content", "pure_text")
        self.processor.reasoning_parser = mock_reasoning_parser

        # 模拟 tool_parser
        mock_tool_parser = MagicMock()
        mock_tool_parser.extract_tool_calls.return_value = MagicMock(
            tools_called=False, tool_calls=None, content="tool_text"
        )
        self.processor.tool_parser_obj = MagicMock(return_value=mock_tool_parser)

        # 调用方法
        result = self.processor.process_response(mock_response_dict)

        # 验证 tokenizer.decode 被正确调用（去掉 eos_token）
        self.processor.tokenizer.decode.assert_called_once_with([10, 20])

        # 验证 reasoning_parser 被调用并正确赋值
        mock_reasoning_parser.extract_reasoning_content.assert_called_once()
        self.assertEqual(result.outputs.text, "pure_text")
        self.assertEqual(result.outputs.reasoning_content, "reasoning_content")

        # 验证 usage 被正确赋值
        self.assertIn("completion_tokens", result.usage)
        self.assertEqual(result.usage["completion_tokens"], 3)

        # 验证 tool_parser 被正确调用
        mock_tool_parser.extract_tool_calls.assert_called_once()

        # 验证返回结果不为 None
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
