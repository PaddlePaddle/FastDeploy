import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.engine.request import Request
from fastdeploy.input.text_processor import DataProcessor


class TestDataProcessorProcess(unittest.TestCase):
    def setUp(self):
        # 创建 DataProcessor 实例的模拟对象
        with patch.object(DataProcessor, "__init__", return_value=None) as mock_init:
            self.processor = DataProcessor("model_path")
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

        def mock_messages2ids(request, **kwargs):
            if "chat_template" in kwargs:
                return [1]
            else:
                return [0]

        def mock_apply_default_parameters(request):
            return request

        self.processor.messages2ids = mock_messages2ids
        self.processor._apply_default_parameters = mock_apply_default_parameters

    def test_process_request(self):
        request = Request.from_dict(
            {
                "request_id": "123",
                "messages": [{"role": "user", "content": "Hello!"}],
                "eos_token_ids": [1],
                "temperature": 1,
                "top_p": 1,
            }
        )
        chat_template_kwargs = {"chat_template": "Hello!"}
        result = self.processor.process_request(request, 100, chat_template_kwargs=chat_template_kwargs)
        self.assertEqual(result.prompt_token_ids, [1])

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

    def test_process_response_dict_normal(self):
        self.processor.tokenizer.decode_token = MagicMock(return_value=("Mock decoded text", 0, 0))
        self.processor.reasoning_parser.extract_reasoning_content = MagicMock(
            return_value=("Mock reasoning content", "Mock final text")
        )
        mock_tokens = ["mock", "reasoning", "tokens"]
        self.processor.tokenizer.tokenize = MagicMock(return_value=mock_tokens)
        self.processor.tool_parser_obj = None
        response_dict = {
            "request_id": "request-id_0",
            "outputs": {
                "token_ids": [2, 3, 4, 5, 1],
                "text": "Hello",
                "top_logprobs": [{"a": 0.1}, {"b": 0.2}, {"c": 0.3}],
            },
            "finish_reason": "stop",
            "finished": True,
        }
        kwargs = {"enable_thinking": True}
        with patch("fastdeploy.input.text_processor.data_processor_logger"):
            result = self.processor.process_response_dict_normal(response_dict, **kwargs)
        self.assertEqual(result["outputs"]["reasoning_content"], "Mock reasoning content")
        self.assertEqual(result["outputs"]["reasoning_token_num"], len(mock_tokens))
        self.assertEqual(result["outputs"]["text"], "Mock final text")
        self.assertIn("completion_tokens", result["outputs"])


if __name__ == "__main__":
    unittest.main()
