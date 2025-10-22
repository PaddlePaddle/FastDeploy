import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.engine.request import Request

# 导入被测类
from fastdeploy.input.ernie4_5_vl_processor import Ernie4_5_VLProcessor


class TestErnie4_5_VLProcessor(unittest.TestCase):
    """测试 Ernie4_5_VLProcessor 的主要功能"""

    def setUp(self):
        """初始化一个带有 mock 依赖的 Processor"""
        # patch DataProcessor，防止真实加载 tokenizer 或模型
        dp_patcher = patch("fastdeploy.input.ernie4_5_vl_processor.DataProcessor")
        self.addCleanup(dp_patcher.stop)
        MockDP = dp_patcher.start()

        # 模拟 DataProcessor 行为
        self.mock_dp = MockDP.return_value
        self.mock_dp.eval.return_value = None
        self.mock_dp.text2ids.return_value = {
            "input_ids": [1, 2, 3],
            "token_type_ids": [0, 0, 0],
            "position_ids": [[0, 0, 0]],
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "cur_position": 3,
        }
        self.mock_dp.request2ids.return_value = self.mock_dp.text2ids.return_value
        self.mock_dp.image_patch_id = 999
        self.mock_dp.spatial_conv_size = 64
        self.mock_dp.tokenizer = MagicMock()
        self.mock_dp.tokenizer.pad_token_id = 0
        self.mock_dp.tokenizer.eos_token_id = 2

        # patch GenerationConfig
        gen_patcher = patch("fastdeploy.input.ernie4_5_vl_processor.GenerationConfig.from_pretrained")
        self.addCleanup(gen_patcher.stop)
        gen_patcher.start()

        # patch Request.from_dict 避免真实依赖
        req_patcher = patch("fastdeploy.input.ernie4_5_vl_processor.Request.from_dict")
        self.addCleanup(req_patcher.stop)
        self.mock_from_dict = req_patcher.start()
        self.mock_from_dict.side_effect = lambda d: Request(d)

        # 创建 Processor 实例
        self.processor = Ernie4_5_VLProcessor(model_name_or_path="mock_path")

        # mock 父类 tokenizer
        self.processor.tokenizer = MagicMock()
        self.processor.tokenizer.eos_token_id = 2
        self.processor.tokenizer.pad_token_id = 0
        self.processor.tokenizer.decode = MagicMock(return_value="decoded text")

    # ----------------------------- #
    # 测试 process_request_dict
    # ----------------------------- #
    def test_process_request_dict_with_prompt(self):
        """测试含 prompt 的请求"""
        req = {"prompt": "hello world"}
        result = self.processor.process_request_dict(req, max_model_len=10)

        self.assertIsInstance(result, dict)
        self.assertIn("prompt_token_ids", result)
        self.assertIsInstance(result["prompt_token_ids"], list)
        self.assertIn("multimodal_inputs", result)
        self.assertIsInstance(result["multimodal_inputs"], dict)
        self.assertEqual(result["prompt_token_ids_len"], len(result["prompt_token_ids"]))

    def test_process_request_dict_with_messages(self):
        """测试含 messages 的请求"""
        req = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
        result = self.processor.process_request_dict(req)
        self.assertIn("prompt_token_ids", result)
        self.assertIn("multimodal_inputs", result)

    # ----------------------------- #
    # 测试 process_request
    # ----------------------------- #
    def test_process_request(self):
        """测试 process_request 能调用 process_request_dict 并返回正确的 Request"""
        # 模拟 Request 对象
        mock_request = MagicMock()
        mock_request.to_dict.return_value = {"prompt": "test prompt"}
        self.processor.process_request_dict = MagicMock(
            return_value={"prompt": "test prompt", "prompt_token_ids": [1, 2]}
        )
        self.processor._apply_default_parameters = MagicMock(
            return_value=Request({"prompt": "test prompt", "prompt_token_ids": [1, 2]})
        )

        result = self.processor.process_request(mock_request, max_model_len=10)
        self.processor.process_request_dict.assert_called_once()
        self.processor._apply_default_parameters.assert_called_once()
        self.assertIsInstance(result, Request)
        self.assertEqual(result.data["prompt_token_ids"], [1, 2])

    # ----------------------------- #
    # 测试 process_response
    # ----------------------------- #
    def test_process_response(self):
        """测试继承自父类的 process_response"""
        response_dict = MagicMock()
        response_dict.request_id = "123"
        response_dict.outputs = MagicMock()
        response_dict.outputs.token_ids = [1, 2, 3]
        response_dict.outputs.index = 2

        result = self.processor.process_response(response_dict)
        self.assertIsNotNone(result)
        self.assertEqual(result.outputs.text, "decoded text")
        self.processor.tokenizer.decode.assert_called_once()

    # ----------------------------- #
    # 测试 process_response_dict
    # ----------------------------- #
    def test_process_response_dict_non_stream(self):
        """测试非流式返回"""
        mock_normal = MagicMock(return_value={"text": "done"})
        self.processor.process_response_dict_normal = mock_normal

        response = {"outputs": {"token_ids": [1, 2, 3]}, "finished": True, "request_id": "req_1"}
        result = self.processor.process_response_dict(response, stream=False)
        mock_normal.assert_called_once()
        self.assertEqual(result, {"text": "done"})

    def test_process_response_dict_stream(self):
        """测试流式返回"""
        mock_stream = MagicMock(return_value={"delta": "ok"})
        self.processor.process_response_dict_streaming = mock_stream

        response = {"outputs": {"token_ids": [1, 2, 3]}, "finished": True, "request_id": "req_2"}
        result = self.processor.process_response_dict(response, stream=True)
        mock_stream.assert_called_once()
        self.assertEqual(result, {"delta": "ok"})


if __name__ == "__main__":
    unittest.main()
