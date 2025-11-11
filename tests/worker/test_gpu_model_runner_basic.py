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

import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Mock paddleformers and related modules BEFORE importing fastdeploy
paddleformers_mock = MagicMock()
sys.modules["paddleformers"] = paddleformers_mock
sys.modules["paddleformers.utils"] = MagicMock()
sys.modules["paddleformers.utils.log"] = MagicMock()
sys.modules["paddleformers.transformers"] = MagicMock()
sys.modules["paddleformers.transformers.configuration_utils"] = MagicMock()

# Mock other potentially missing modules
missing_modules = []
for module_name in ["msgspec", "aistudio_sdk", "modelscope", "fastapi", "huggingface_hub"]:
    try:
        __import__(module_name)
    except ImportError:
        sys.modules[module_name] = MagicMock()
        missing_modules.append(module_name)

import paddle

# Try to import GPUModelRunner
CAN_IMPORT = False
GPUModelRunner = None
IMPORT_ERROR = None

try:
    from fastdeploy.worker.gpu_model_runner import GPUModelRunner

    CAN_IMPORT = True
    if missing_modules:
        print(f"Warning: Mocked modules due to missing dependencies: {', '.join(missing_modules)}")
except Exception as e:
    IMPORT_ERROR = str(e)
    # Set module-level skip marker
    pytestmark = pytest.mark.skip(reason=f"Cannot import GPUModelRunner: {e}")


def create_mock_fd_config():
    """Create a complete mock FDConfig object"""
    mock_config = Mock()

    # model_config
    mock_config.model_config = Mock()
    mock_config.model_config.enable_mm = False
    mock_config.model_config.runner_type = "generation"
    mock_config.model_config.ori_vocab_size = 50000
    mock_config.model_config.max_logprobs = -1
    mock_config.model_config.enable_logprob = False
    mock_config.model_config.dtype = "bfloat16"
    mock_config.model_config.num_hidden_layers = 32
    mock_config.model_config.num_attention_heads = 32
    mock_config.model_config.num_key_value_heads = 32
    mock_config.model_config.head_dim = 128
    mock_config.model_config.max_model_len = 2048
    mock_config.model_config.vocab_size = 50000
    mock_config.model_config.model_type = "llama"

    # cache_config
    mock_config.cache_config = Mock()
    mock_config.cache_config.block_size = 16
    mock_config.cache_config.total_block_num = 1000
    mock_config.cache_config.kv_cache_dtype = "float16"
    mock_config.cache_config.max_encoder_cache = 0

    # scheduler_config
    mock_config.scheduler_config = Mock()
    mock_config.scheduler_config.max_num_seqs = 256
    mock_config.scheduler_config.max_num_batched_tokens = 4096
    mock_config.scheduler_config.splitwise_role = "mixed"

    # parallel_config
    mock_config.parallel_config = Mock()
    mock_config.parallel_config.tensor_parallel_size = 1
    mock_config.parallel_config.engine_worker_queue_port = 6666
    mock_config.parallel_config.use_ep = False

    # speculative_config
    mock_config.speculative_config = Mock()
    mock_config.speculative_config.method = None

    # graph_opt_config
    mock_config.graph_opt_config = Mock()
    mock_config.graph_opt_config.use_cudagraph = False
    mock_config.graph_opt_config.cudagraph_capture_sizes = [1, 2, 4]
    mock_config.graph_opt_config.sot_warmup_sizes = []
    mock_config.graph_opt_config.cudagraph_only_prefill = False
    mock_config.graph_opt_config.graph_opt_level = 0

    # early_stop_config
    mock_config.early_stop_config = Mock()
    mock_config.early_stop_config.enable_early_stop = False

    # structured_outputs_config
    mock_config.structured_outputs_config = Mock()
    mock_config.structured_outputs_config.guided_decoding_backend = "off"

    # load_config
    mock_config.load_config = Mock()
    mock_config.load_config.dynamic_load_weight = False

    # device_config
    mock_config.device_config = Mock()

    # quant_config
    mock_config.quant_config = None

    return mock_config


@pytest.mark.skipif(not CAN_IMPORT, reason="Cannot import GPUModelRunner")
class TestGPUModelRunnerBasic(unittest.TestCase):
    """Test basic functionality of GPUModelRunner"""

    def setUp(self):
        """Set up test environment"""
        if not CAN_IMPORT:
            self.skipTest("Cannot import required modules")

        self.mock_fd_config = create_mock_fd_config()

    def test_initialization_basic(self):
        """Test basic initialization"""
        with patch("fastdeploy.worker.gpu_model_runner.get_attention_backend"):
            with patch("fastdeploy.worker.gpu_model_runner.Sampler"):
                with patch("fastdeploy.worker.gpu_model_runner.GPUMemoryChecker"):
                    runner = GPUModelRunner(
                        fd_config=self.mock_fd_config,
                        device="gpu:0",
                        device_id=0,
                        rank=0,
                        local_rank=0,
                    )

                    # Verify initialization
                    self.assertEqual(runner.rank, 0)
                    self.assertEqual(runner.local_rank, 0)
                    self.assertEqual(runner.device_id, 0)
                    self.assertIsNotNone(runner.share_inputs)
                    self.assertIsInstance(runner.mm_cache, dict)
                    self.assertIsInstance(runner.requests, list)

    def test_exist_prefill(self):
        """Test exist_prefill method - real logic test"""
        with patch("fastdeploy.worker.gpu_model_runner.get_attention_backend"):
            with patch("fastdeploy.worker.gpu_model_runner.Sampler"):
                with patch("fastdeploy.worker.gpu_model_runner.GPUMemoryChecker"):
                    runner = GPUModelRunner(
                        fd_config=self.mock_fd_config,
                        device="gpu:0",
                        device_id=0,
                        rank=0,
                        local_rank=0,
                    )

                    # Test with prefill data
                    runner.share_inputs["seq_lens_encoder"] = paddle.to_tensor([10, 20, 30])
                    self.assertTrue(runner.exist_prefill())

                    # Test without prefill data
                    runner.share_inputs["seq_lens_encoder"] = paddle.zeros([10])
                    self.assertFalse(runner.exist_prefill())

    def test_exist_decode(self):
        """Test exist_decode method - real logic test"""
        with patch("fastdeploy.worker.gpu_model_runner.get_attention_backend"):
            with patch("fastdeploy.worker.gpu_model_runner.Sampler"):
                with patch("fastdeploy.worker.gpu_model_runner.GPUMemoryChecker"):
                    runner = GPUModelRunner(
                        fd_config=self.mock_fd_config,
                        device="gpu:0",
                        device_id=0,
                        rank=0,
                        local_rank=0,
                    )

                    # Test with decode data
                    runner.share_inputs["seq_lens_decoder"] = paddle.to_tensor([5, 10])
                    self.assertTrue(runner.exist_decode())

                    # Test without decode data
                    runner.share_inputs["seq_lens_decoder"] = paddle.zeros([10])
                    self.assertFalse(runner.exist_decode())

    def test_only_prefill(self):
        """Test only_prefill method - real logic test"""
        with patch("fastdeploy.worker.gpu_model_runner.get_attention_backend"):
            with patch("fastdeploy.worker.gpu_model_runner.Sampler"):
                with patch("fastdeploy.worker.gpu_model_runner.GPUMemoryChecker"):
                    runner = GPUModelRunner(
                        fd_config=self.mock_fd_config,
                        device="gpu:0",
                        device_id=0,
                        rank=0,
                        local_rank=0,
                    )

                    # Only prefill, no decode
                    runner.share_inputs["seq_lens_encoder"] = paddle.to_tensor([10, 5])
                    runner.share_inputs["seq_lens_decoder"] = paddle.zeros([10])
                    self.assertTrue(runner.only_prefill())

                    # prefill + decode
                    runner.share_inputs["seq_lens_decoder"] = paddle.to_tensor([5])
                    self.assertFalse(runner.only_prefill())

    def test_only_decode(self):
        """Test only_decode method - real logic test"""
        with patch("fastdeploy.worker.gpu_model_runner.get_attention_backend"):
            with patch("fastdeploy.worker.gpu_model_runner.Sampler"):
                with patch("fastdeploy.worker.gpu_model_runner.GPUMemoryChecker"):
                    runner = GPUModelRunner(
                        fd_config=self.mock_fd_config,
                        device="gpu:0",
                        device_id=0,
                        rank=0,
                        local_rank=0,
                    )

                    # Only decode, no prefill
                    runner.share_inputs["seq_lens_encoder"] = paddle.zeros([10])
                    runner.share_inputs["seq_lens_decoder"] = paddle.to_tensor([5])
                    self.assertTrue(runner.only_decode())

                    # prefill + decode
                    runner.share_inputs["seq_lens_encoder"] = paddle.to_tensor([10])
                    self.assertFalse(runner.only_decode())

    def test_get_attr_from_request(self):
        """Test get_attr_from_request method"""
        with patch("fastdeploy.worker.gpu_model_runner.get_attention_backend"):
            with patch("fastdeploy.worker.gpu_model_runner.Sampler"):
                with patch("fastdeploy.worker.gpu_model_runner.GPUMemoryChecker"):
                    runner = GPUModelRunner(
                        fd_config=self.mock_fd_config,
                        device="gpu:0",
                        device_id=0,
                        rank=0,
                        local_rank=0,
                    )

                    # Create mock requests
                    mock_req1 = Mock()
                    mock_req1.sampling_params = Mock()
                    mock_req1.sampling_params.temperature = 0.7

                    mock_req2 = Mock()
                    mock_req2.sampling_params = Mock()
                    mock_req2.sampling_params.temperature = 0.9

                    requests = [mock_req1, mock_req2]
                    result = runner.get_attr_from_request(requests, "temperature")

                    self.assertEqual(len(result), 2)
                    self.assertIn(0.7, result)
                    self.assertIn(0.9, result)

    def test_get_input_length_list(self):
        """Test get_input_length_list method"""
        with patch("fastdeploy.worker.gpu_model_runner.get_attention_backend"):
            with patch("fastdeploy.worker.gpu_model_runner.Sampler"):
                with patch("fastdeploy.worker.gpu_model_runner.GPUMemoryChecker"):
                    runner = GPUModelRunner(
                        fd_config=self.mock_fd_config,
                        device="gpu:0",
                        device_id=0,
                        rank=0,
                        local_rank=0,
                    )

                    mock_req1 = Mock()
                    mock_req1.prompt_token_len = 10

                    mock_req2 = Mock()
                    mock_req2.prompt_token_len = 20

                    requests = [mock_req1, mock_req2]
                    result = runner.get_input_length_list(requests)

                    self.assertEqual(len(result), 2)
                    self.assertEqual(result[0], 10)
                    self.assertEqual(result[1], 20)

    def test_cal_theortical_kvcache(self):
        """Test cal_theortical_kvcache method"""
        with patch("fastdeploy.worker.gpu_model_runner.get_attention_backend"):
            with patch("fastdeploy.worker.gpu_model_runner.Sampler"):
                with patch("fastdeploy.worker.gpu_model_runner.GPUMemoryChecker"):
                    runner = GPUModelRunner(
                        fd_config=self.mock_fd_config,
                        device="gpu:0",
                        device_id=0,
                        rank=0,
                        local_rank=0,
                    )

                    mock_req = Mock()
                    mock_req.prompt_token_len = 100
                    mock_req.output_len = 50

                    result = runner.cal_theortical_kvcache(mock_req)

                    self.assertIsInstance(result, (int, float))
                    self.assertGreater(result, 0)

    def test_not_need_stop(self):
        """Test not_need_stop method"""
        with patch("fastdeploy.worker.gpu_model_runner.get_attention_backend"):
            with patch("fastdeploy.worker.gpu_model_runner.Sampler"):
                with patch("fastdeploy.worker.gpu_model_runner.GPUMemoryChecker"):
                    runner = GPUModelRunner(
                        fd_config=self.mock_fd_config,
                        device="gpu:0",
                        device_id=0,
                        rank=0,
                        local_rank=0,
                    )

                    mock_req = Mock()
                    mock_req.sampling_params = Mock()
                    mock_req.sampling_params.stop_token_ids = [1, 2, 3]

                    # Token in stop list
                    self.assertFalse(runner.not_need_stop(mock_req, 1))

                    # Token not in stop list
                    self.assertTrue(runner.not_need_stop(mock_req, 99))

    def test_clear_cache(self):
        """Test clear_cache method"""
        with patch("fastdeploy.worker.gpu_model_runner.get_attention_backend"):
            with patch("fastdeploy.worker.gpu_model_runner.Sampler"):
                with patch("fastdeploy.worker.gpu_model_runner.GPUMemoryChecker"):
                    runner = GPUModelRunner(
                        fd_config=self.mock_fd_config,
                        device="gpu:0",
                        device_id=0,
                        rank=0,
                        local_rank=0,
                    )

                    # Add some cache data
                    runner.mm_cache = {"hash1": "data1", "hash2": "data2"}

                    runner.clear_cache()

                    self.assertEqual(len(runner.mm_cache), 0)

    def test_clear_requests(self):
        """Test clear_requests method"""
        with patch("fastdeploy.worker.gpu_model_runner.get_attention_backend"):
            with patch("fastdeploy.worker.gpu_model_runner.Sampler"):
                with patch("fastdeploy.worker.gpu_model_runner.GPUMemoryChecker"):
                    runner = GPUModelRunner(
                        fd_config=self.mock_fd_config,
                        device="gpu:0",
                        device_id=0,
                        rank=0,
                        local_rank=0,
                    )

                    # Add some requests
                    runner.requests = [Mock(), Mock()]

                    runner.clear_requests()

                    self.assertEqual(len(runner.requests), 0)

    def test_scatter_and_cache_features(self):
        """Test scatter_and_cache_features method"""
        with patch("fastdeploy.worker.gpu_model_runner.get_attention_backend"):
            with patch("fastdeploy.worker.gpu_model_runner.Sampler"):
                with patch("fastdeploy.worker.gpu_model_runner.GPUMemoryChecker"):
                    runner = GPUModelRunner(
                        fd_config=self.mock_fd_config,
                        device="gpu:0",
                        device_id=0,
                        rank=0,
                        local_rank=0,
                    )

                    mm_features = paddle.randn([5, 256])
                    mm_num_list = [2, 3]
                    mm_hashes = ["hash1", "hash2"]

                    result = runner.scatter_and_cache_features(mm_features, mm_num_list, mm_hashes)

                    self.assertIsNotNone(result)
                    self.assertEqual(len(runner.mm_cache), 2)
                    self.assertIn("hash1", runner.mm_cache)
                    self.assertIn("hash2", runner.mm_cache)

    def test_prepare_rope3d(self):
        """Test prepare_rope3d method"""
        with patch("fastdeploy.worker.gpu_model_runner.get_attention_backend"):
            with patch("fastdeploy.worker.gpu_model_runner.Sampler"):
                with patch("fastdeploy.worker.gpu_model_runner.GPUMemoryChecker"):
                    with patch("fastdeploy.worker.gpu_model_runner.get_rope_3d") as mock_rope3d:
                        mock_rope3d.return_value = lambda x: x

                        runner = GPUModelRunner(
                            fd_config=self.mock_fd_config,
                            device="gpu:0",
                            device_id=0,
                            rank=0,
                            local_rank=0,
                        )

                        grid_thw = paddle.to_tensor([[1, 2, 2], [1, 3, 3]])
                        result = runner.prepare_rope3d(grid_thw)

                        self.assertIsNotNone(result)
                        self.assertIsInstance(result, paddle.Tensor)

    def test_get_supported_pooling_tasks(self):
        """Test get_supported_pooling_tasks method"""
        with patch("fastdeploy.worker.gpu_model_runner.get_attention_backend"):
            with patch("fastdeploy.worker.gpu_model_runner.Sampler"):
                with patch("fastdeploy.worker.gpu_model_runner.GPUMemoryChecker"):
                    # Test non-pooling model
                    self.mock_fd_config.model_config.runner_type = "generation"
                    runner = GPUModelRunner(
                        fd_config=self.mock_fd_config,
                        device="gpu:0",
                        device_id=0,
                        rank=0,
                        local_rank=0,
                    )

                    result = runner.get_supported_pooling_tasks()
                    self.assertIsInstance(result, list)
                    self.assertEqual(len(result), 0)

    def test_update_share_input_block_num(self):
        """Test update_share_input_block_num method"""
        with patch("fastdeploy.worker.gpu_model_runner.get_attention_backend"):
            with patch("fastdeploy.worker.gpu_model_runner.Sampler"):
                with patch("fastdeploy.worker.gpu_model_runner.GPUMemoryChecker"):
                    runner = GPUModelRunner(
                        fd_config=self.mock_fd_config,
                        device="gpu:0",
                        device_id=0,
                        rank=0,
                        local_rank=0,
                    )

                    # Mock initialize_kv_cache to avoid actual GPU operations
                    runner.initialize_kv_cache = Mock()

                    # Test update block num
                    runner.update_share_input_block_num(num_gpu_blocks=500)

                    self.assertEqual(runner.num_gpu_blocks, 500)
                    runner.initialize_kv_cache.assert_called_once()

    def test_batch_uncached_inputs(self):
        """Test batch_uncached_inputs method"""
        with patch("fastdeploy.worker.gpu_model_runner.get_attention_backend"):
            with patch("fastdeploy.worker.gpu_model_runner.Sampler"):
                with patch("fastdeploy.worker.gpu_model_runner.GPUMemoryChecker"):
                    # Enable multimodal
                    self.mock_fd_config.model_config.enable_mm = True
                    self.mock_fd_config.cache_config.max_encoder_cache = 10

                    runner = GPUModelRunner(
                        fd_config=self.mock_fd_config,
                        device="gpu:0",
                        device_id=0,
                        rank=0,
                        local_rank=0,
                    )

                    # Create mock request with multimodal inputs
                    mock_request = Mock()
                    mock_request.prefill_start_index = 0
                    mock_request.prefill_end_index = 10
                    mock_request.image_type_ids_start = 0
                    mock_request.image_type_ids_end = 5
                    mock_request.image_start = 0
                    mock_request.image_end = 2
                    mock_request.num_image_start = 0
                    mock_request.num_image_end = 1

                    mock_request.multimodal_inputs = {
                        "input_ids": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                        "token_type_ids": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                        "image_type_ids": np.array([1, 1, 1, 1, 1]),
                        "images": np.random.rand(2, 3, 224, 224),
                        "grid_thw": np.array([[1, 14, 14]]),
                        "mm_hashes": ["hash1"],
                    }

                    # Initialize encoder cache
                    runner.encoder_cache = {}

                    result = runner.batch_uncached_inputs(mock_request)

                    self.assertIsNotNone(result)
                    self.assertEqual(len(result), 6)  # Returns 6 elements

    def test_get_chunked_inputs(self):
        """Test get_chunked_inputs method"""
        with patch("fastdeploy.worker.gpu_model_runner.get_attention_backend"):
            with patch("fastdeploy.worker.gpu_model_runner.Sampler"):
                with patch("fastdeploy.worker.gpu_model_runner.GPUMemoryChecker"):
                    self.mock_fd_config.model_config.enable_mm = True

                    runner = GPUModelRunner(
                        fd_config=self.mock_fd_config,
                        device="gpu:0",
                        device_id=0,
                        rank=0,
                        local_rank=0,
                    )

                    # Create mock request
                    mock_request = Mock()
                    mock_request.prefill_start_index = 0
                    mock_request.prefill_end_index = 5
                    mock_request.image_type_ids_start = 0
                    mock_request.image_type_ids_end = 3
                    mock_request.image_start = 0
                    mock_request.image_end = 1
                    mock_request.num_image_start = 0
                    mock_request.num_image_end = 1

                    mock_request.multimodal_inputs = {
                        "input_ids": np.array([1, 2, 3, 4, 5, 6, 7, 8]),
                        "token_type_ids": np.array([0, 0, 0, 0, 0, 0, 0, 0]),
                        "image_type_ids": np.array([1, 1, 1, 2, 2]),
                        "images": np.random.rand(1, 3, 224, 224),
                        "grid_thw": np.array([[1, 14, 14]]),
                        "mm_hashes": ["hash1"],
                    }

                    result = runner.get_chunked_inputs(mock_request)

                    self.assertIsNotNone(result)
                    self.assertEqual(len(result), 6)  # Returns 6 elements

    def test_padding_cudagraph_inputs(self):
        """Test padding_cudagraph_inputs method"""
        with patch("fastdeploy.worker.gpu_model_runner.get_attention_backend"):
            with patch("fastdeploy.worker.gpu_model_runner.Sampler"):
                with patch("fastdeploy.worker.gpu_model_runner.GPUMemoryChecker"):
                    # Enable cudagraph
                    self.mock_fd_config.graph_opt_config.use_cudagraph = True

                    runner = GPUModelRunner(
                        fd_config=self.mock_fd_config,
                        device="gpu:0",
                        device_id=0,
                        rank=0,
                        local_rank=0,
                    )

                    # Initialize forward_meta
                    runner.forward_meta = Mock()
                    runner.forward_meta.ids_remove_padding = paddle.zeros([100])

                    # Test padding
                    runner.padding_cudagraph_inputs()

                    # Verify real_token_num is set
                    self.assertEqual(runner.real_token_num, 100)

    def test_clear_parameters(self):
        """Test clear_parameters method"""
        with patch("fastdeploy.worker.gpu_model_runner.get_attention_backend"):
            with patch("fastdeploy.worker.gpu_model_runner.Sampler"):
                with patch("fastdeploy.worker.gpu_model_runner.GPUMemoryChecker"):
                    # Enable dynamic load
                    self.mock_fd_config.load_config.dynamic_load_weight = True

                    runner = GPUModelRunner(
                        fd_config=self.mock_fd_config,
                        device="gpu:0",
                        device_id=0,
                        rank=0,
                        local_rank=0,
                    )

                    # Mock dynamic weight manager
                    runner.dynamic_weight_manager = Mock()
                    runner.model = Mock()
                    runner.model.clear_grpah_opt_backend = Mock()

                    # Test clear parameters
                    runner.clear_parameters(pid=1)

                    runner.dynamic_weight_manager.clear_parameters.assert_called_once_with(1)

    def test_update_parameters(self):
        """Test update_parameters method"""
        with patch("fastdeploy.worker.gpu_model_runner.get_attention_backend"):
            with patch("fastdeploy.worker.gpu_model_runner.Sampler"):
                with patch("fastdeploy.worker.gpu_model_runner.GPUMemoryChecker"):
                    # Enable dynamic load
                    self.mock_fd_config.load_config.dynamic_load_weight = True

                    runner = GPUModelRunner(
                        fd_config=self.mock_fd_config,
                        device="gpu:0",
                        device_id=0,
                        rank=0,
                        local_rank=0,
                    )

                    # Mock required methods
                    runner.dynamic_weight_manager = Mock()
                    runner.initialize_kv_cache = Mock()
                    runner.capture_model = Mock()

                    # Test update parameters
                    runner.update_parameters(pid=1)

                    runner.dynamic_weight_manager.update_parameters.assert_called_once_with(1)
                    runner.initialize_kv_cache.assert_called_once()

    def test_init_speculative_proposer(self):
        """Test _init_speculative_proposer method"""
        with patch("fastdeploy.worker.gpu_model_runner.get_attention_backend"):
            with patch("fastdeploy.worker.gpu_model_runner.Sampler"):
                with patch("fastdeploy.worker.gpu_model_runner.GPUMemoryChecker"):
                    # Test with no speculative method
                    self.mock_fd_config.speculative_config.method = None

                    runner = GPUModelRunner(
                        fd_config=self.mock_fd_config,
                        device="gpu:0",
                        device_id=0,
                        rank=0,
                        local_rank=0,
                    )

                    # Call init method
                    runner._init_speculative_proposer()

                    # Should set proposer to None
                    self.assertIsNone(runner.proposer)

    def test_init_logits_processor(self):
        """Test _init_logits_processor method"""
        with patch("fastdeploy.worker.gpu_model_runner.get_attention_backend"):
            with patch("fastdeploy.worker.gpu_model_runner.Sampler"):
                with patch("fastdeploy.worker.gpu_model_runner.GPUMemoryChecker"):
                    # Enable guided decoding
                    self.mock_fd_config.structured_outputs_config.guided_decoding_backend = "xgrammar"

                    runner = GPUModelRunner(
                        fd_config=self.mock_fd_config,
                        device="gpu:0",
                        device_id=0,
                        rank=0,
                        local_rank=0,
                    )

                    # Mock guided backend
                    runner.guided_backend = Mock()
                    mock_processor = Mock()
                    runner.guided_backend.get_logits_processor = Mock(return_value=mock_processor)

                    # Create mock request
                    mock_request = Mock()
                    mock_request.guided_json = '{"key": "value"}'
                    mock_request.guided_regex = None
                    mock_request.guided_grammar = None
                    mock_request.structural_tag = None

                    # Test init logits processor
                    processor, key = runner._init_logits_processor(mock_request)

                    self.assertIsNotNone(processor)
                    self.assertEqual(key, ("json", '{"key": "value"}'))


if __name__ == "__main__":
    unittest.main()
