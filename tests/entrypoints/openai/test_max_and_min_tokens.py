import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.entrypoints.engine_client import EngineClient, EngineError
from fastdeploy.input.ernie4_5_vl_processor.ernie4_5_vl_processor import (
    Ernie4_5_VLProcessor,
)


class TestChatContinuationPreprocess(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        with patch(
            "fastdeploy.input.ernie4_5_vl_processor.ernie4_5_vl_processor.DataProcessor"
        ) as mock_data_processor:
            mock_ernie4_5_processor = MagicMock()
            mock_data_processor.return_value = mock_ernie4_5_processor

            mock_tokenizer = MagicMock()
            mock_tokenizer.eos_token_id = 102
            mock_tokenizer.pad_token_id = 0
            mock_ernie4_5_processor.tokenizer = mock_tokenizer
            mock_ernie4_5_processor.eval = MagicMock()
            mock_ernie4_5_processor.image_patch_id = MagicMock()
            mock_ernie4_5_processor.spatial_conv_size = MagicMock()

            self.ernie_processor = Ernie4_5_VLProcessor(model_name_or_path="mock_model_path")
            self.ernie_processor.ernie4_5_processor = mock_ernie4_5_processor

        def _create_mock_tensor(initial_ids):
            mock_tensor = MagicMock()
            mock_tensor._data = initial_ids
            mock_tensor.extend = lambda x: mock_tensor._data.extend(x)
            mock_tensor.tolist = lambda: mock_tensor._data
            return mock_tensor

        self.ernie_processor.ernie4_5_processor.request2ids.return_value = {
            "input_ids": _create_mock_tensor([101] * 200)
        }
        self.ernie_processor.pack_outputs = lambda x: x

        def mock_append_completion_tokens(multimodal_inputs, completion_token_ids):
            multimodal_inputs["input_ids"].extend(completion_token_ids)

        self.ernie_processor.append_completion_tokens = MagicMock(side_effect=mock_append_completion_tokens)
        self.ernie_processor.eos_token_ids = [102]
        self.ernie_processor._parse_limits = MagicMock(return_value=None)

        with patch.object(EngineClient, "__init__", return_value=None):
            self.engine_client = EngineClient("mock_model_path")
        self.engine_client.data_processor = self.ernie_processor
        self.engine_client.max_model_len = 300
        self.engine_client.enable_mm = False
        self.engine_client.enable_prefix_caching = False
        self.engine_client.zmq_client = MagicMock()
        self.engine_client.valid_parameters = MagicMock()

        self.mock_api_logger = patch("fastdeploy.entrypoints.engine_client.api_server_logger").start()
        self.mock_data_logger = patch(
            "fastdeploy.input.ernie4_5_vl_processor.ernie4_5_vl_processor.data_processor_logger"
        ).start()

    async def asyncTearDown(self):
        patch.stopall()

    def _update_processor_token_ids(self, prompt_token_ids_len: int):
        def _create_mock_tensor(initial_ids):
            mock_tensor = MagicMock()
            mock_tensor._data = initial_ids
            mock_tensor.extend = lambda x: mock_tensor._data.extend(x)
            mock_tensor.tolist = lambda: mock_tensor._data
            return mock_tensor

        self.ernie_processor.ernie4_5_processor.request2ids.return_value = {
            "input_ids": _create_mock_tensor([101] * prompt_token_ids_len)
        }

    @patch("uuid.uuid4", return_value="test-request-id")
    async def test_continuation_first_request(self, mock_uuid):
        request = {"messages": [{"role": "user", "content": "描述这张图片"}], "max_tokens": 50, "min_tokens": 10}

        await self.engine_client.format_and_add_data(request)

        self.assertEqual(request["max_tokens"], 50)
        self.assertEqual(request["min_tokens"], 10)
        self.assertEqual(len(request["prompt_token_ids"]), 200)

    @patch("uuid.uuid4", return_value="test-request-id-2")
    async def test_continuation_second_request(self, mock_uuid):
        self._update_processor_token_ids(prompt_token_ids_len=50)

        request = {
            "messages": [{"role": "user", "content": "描述这张图片"}],
            "completion_token_ids": [103] * 30,
            "max_tokens": 200,
            "min_tokens": 100,
        }

        await self.engine_client.format_and_add_data(request)

        self.assertEqual(request["max_tokens"], 170)
        self.assertEqual(request["min_tokens"], 70)
        self.assertEqual(len(request["prompt_token_ids"]), 80)

    @patch("uuid.uuid4", return_value="test-request-id-3")
    async def test_continuation_boundary_max_tokens_exhausted(self, mock_uuid):
        self._update_processor_token_ids(prompt_token_ids_len=100)

        request = {
            "messages": [{"role": "user", "content": "描述这张图片"}],
            "completion_token_ids": [103] * 190,
            "max_tokens": 200,
            "min_tokens": 5,
        }

        await self.engine_client.format_and_add_data(request)

        self.assertEqual(request["max_tokens"], 10)
        self.assertEqual(request["min_tokens"], 1)

    @patch("uuid.uuid4", return_value="test-request-id-4")
    async def test_continuation_boundary_no_capacity(self, mock_uuid):
        self._update_processor_token_ids(prompt_token_ids_len=260)

        request = {
            "messages": [{"role": "user", "content": "描述这张图片"}],
            "completion_token_ids": [103] * 50,
            "max_tokens": 200,
            "min_tokens": 5,
        }

        with self.assertRaises(EngineError) as ctx:
            await self.engine_client.format_and_add_data(request)

        self.assertIn("Input text is too long", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
