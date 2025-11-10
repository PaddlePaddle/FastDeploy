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

import unittest
from unittest.mock import MagicMock, Mock, patch

import paddle

from fastdeploy.config import FDConfig

# Import the modules to be tested
from fastdeploy.worker.gpu_model_runner import GPUModelRunner


class TestGPUModelRunnerBasic(unittest.TestCase):
    """Test basic functionality of GPUModelRunner"""

    def setUp(self):
        """Set up test environment"""
        # Create basic FDConfig configuration
        self.config = FDConfig(
            model="test_model",
            tensor_parallel_degree=1,
            dtype="float16",
            block_size=16,
            max_model_len=2048,
            gpu_memory_utilization=0.9,
            kv_cache_dtype="float16",
        )

        # Create GPUModelRunner instance (use mock to avoid actual model loading)
        with patch("fastdeploy.worker.gpu_model_runner.get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.config = MagicMock()
            mock_model.config.architectures = ["LlamaForCausalLM"]
            mock_model.config.num_hidden_layers = 12
            mock_model.config.num_attention_heads = 12
            mock_model.config.hidden_size = 768
            mock_get_model.return_value = mock_model

            self.runner = GPUModelRunner(self.config)

    def test_init(self):
        """Test GPUModelRunner initialization"""
        self.assertIsNotNone(self.runner)
        self.assertEqual(self.runner.config, self.config)

    def test_exist_prefill(self):
        """Test exist_prefill method"""
        # Create mock forward_meta
        mock_meta = Mock()
        mock_meta.prefill_len = 10

        result = self.runner.exist_prefill(mock_meta)
        self.assertTrue(result)

        mock_meta.prefill_len = 0
        result = self.runner.exist_prefill(mock_meta)
        self.assertFalse(result)

    def test_exist_decode(self):
        """Test exist_decode method"""
        mock_meta = Mock()
        mock_meta.decode_len = 5

        result = self.runner.exist_decode(mock_meta)
        self.assertTrue(result)

        mock_meta.decode_len = 0
        result = self.runner.exist_decode(mock_meta)
        self.assertFalse(result)

    def test_only_prefill(self):
        """Test only_prefill method"""
        mock_meta = Mock()
        mock_meta.prefill_len = 10
        mock_meta.decode_len = 0

        result = self.runner.only_prefill(mock_meta)
        self.assertTrue(result)

        mock_meta.decode_len = 5
        result = self.runner.only_prefill(mock_meta)
        self.assertFalse(result)

    def test_only_decode(self):
        """Test only_decode method"""
        mock_meta = Mock()
        mock_meta.prefill_len = 0
        mock_meta.decode_len = 5

        result = self.runner.only_decode(mock_meta)
        self.assertTrue(result)

        mock_meta.prefill_len = 10
        result = self.runner.only_decode(mock_meta)
        self.assertFalse(result)

    def test_get_chunked_inputs(self):
        """Test get_chunked_inputs method"""
        mock_meta = Mock()
        mock_meta.decode_len = 0
        mock_meta.prefill_len = 128

        # Mock batch_inputs
        batch_inputs = Mock()
        batch_inputs.input_ids = paddle.randint(0, 1000, [128])
        batch_inputs.seq_lens_encoder = paddle.to_tensor([64, 64])

        result = self.runner.get_chunked_inputs(batch_inputs, mock_meta)
        self.assertIsNotNone(result)

    def test_batch_uncached_inputs(self):
        """Test batch_uncached_inputs method"""
        # Create mock batch_inputs
        batch_inputs = Mock()
        batch_inputs.mm_features = paddle.randn([10, 256])
        batch_inputs.mm_hashes = ["hash1", "hash2"]

        # Create mock forward_meta
        forward_meta = Mock()
        forward_meta.mm_num = 2

        result = self.runner.batch_uncached_inputs(batch_inputs, forward_meta)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_scatter_and_cache_features(self):
        """Test scatter_and_cache_features method"""
        # Create test data
        mm_features = paddle.randn([5, 256])
        mm_num_list = [2, 3]
        mm_hashes = ["hash1", "hash2"]

        # Initialize runner's mm_cache
        self.runner.mm_cache = {}

        result = self.runner.scatter_and_cache_features(mm_features, mm_num_list, mm_hashes)
        self.assertIsNotNone(result)
        self.assertEqual(len(self.runner.mm_cache), 2)

    def test_get_attr_from_request(self):
        """Test get_attr_from_request method"""
        # Create mock request list
        mock_req1 = Mock()
        mock_req1.sampling_params = Mock()
        mock_req1.sampling_params.temperature = 0.7

        mock_req2 = Mock()
        mock_req2.sampling_params = Mock()
        mock_req2.sampling_params.temperature = 0.9

        requests = [mock_req1, mock_req2]

        result = self.runner.get_attr_from_request(requests, "temperature")
        self.assertEqual(len(result), 2)
        self.assertIn(0.7, result)
        self.assertIn(0.9, result)

    def test_get_input_length_list(self):
        """Test get_input_length_list method"""
        mock_req1 = Mock()
        mock_req1.prompt_token_len = 10

        mock_req2 = Mock()
        mock_req2.prompt_token_len = 20

        requests = [mock_req1, mock_req2]

        result = self.runner.get_input_length_list(requests)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], 10)
        self.assertEqual(result[1], 20)

    def test_get_supported_pooling_tasks(self):
        """Test get_supported_pooling_tasks method"""
        # Create mock task list
        mock_task1 = Mock()
        mock_task1.task_type.value = "embedding"

        mock_task2 = Mock()
        mock_task2.task_type.value = "generation"

        mock_task3 = Mock()
        mock_task3.task_type.value = "classification"

        tasks = [mock_task1, mock_task2, mock_task3]

        result = self.runner.get_supported_pooling_tasks(tasks)
        self.assertIsInstance(result, list)

    def test_initialize_forward_meta(self):
        """Test initialize_forward_meta method"""
        # Set necessary runner attributes
        self.runner.config.block_size = 16
        self.runner.config.max_model_len = 2048

        # Create mock requests
        mock_req = Mock()
        mock_req.prompt_token_len = 50
        mock_req.output_len = 100
        requests = [mock_req]

        with patch.object(self.runner, "get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            result = self.runner.initialize_forward_meta(requests)
            self.assertIsNotNone(result)

    def test_update_share_input_block_num(self):
        """Test update_share_input_block_num method"""
        # Create mock forward_meta
        mock_meta = Mock()
        mock_meta.block_num = paddle.to_tensor([10, 20, 30])

        result = self.runner.update_share_input_block_num(mock_meta)
        self.assertIsNone(result)  # This method has no return value

    def test_cal_theortical_kvcache(self):
        """Test cal_theortical_kvcache method"""
        # Set necessary configuration
        self.runner.config.num_hidden_layers = 12
        self.runner.config.num_attention_heads = 12
        self.runner.config.hidden_size = 768
        self.runner.config.kv_cache_dtype = "float16"

        # Create mock request
        mock_req = Mock()
        mock_req.prompt_token_len = 100
        mock_req.output_len = 50

        result = self.runner.cal_theortical_kvcache(mock_req)
        self.assertIsInstance(result, (int, float))
        self.assertGreater(result, 0)

    def test_not_need_stop(self):
        """Test not_need_stop method"""
        # Create mock request
        mock_req = Mock()
        mock_req.sampling_params = Mock()
        mock_req.sampling_params.stop_token_ids = [1, 2, 3]

        # Test when token_id is in stop_token_ids
        result = self.runner.not_need_stop(mock_req, 1)
        self.assertFalse(result)

        # Test when token_id is not in stop_token_ids
        result = self.runner.not_need_stop(mock_req, 99)
        self.assertTrue(result)

    def test_clear_cache(self):
        """Test clear_cache method"""
        # Initialize some cache data
        self.runner.mm_cache = {"hash1": "data1", "hash2": "data2"}

        self.runner.clear_cache()
        self.assertEqual(len(self.runner.mm_cache), 0)

    def test_clear_parameters(self):
        """Test clear_parameters method"""
        # This method mainly clears parameters, test that it doesn't throw exception
        try:
            self.runner.clear_parameters()
            success = True
        except Exception:
            success = False

        self.assertTrue(success)

    def test_clear_requests(self):
        """Test clear_requests method"""
        # Set some request data
        self.runner.requests = [Mock(), Mock()]

        self.runner.clear_requests()
        self.assertEqual(len(self.runner.requests), 0)

    def test_get_model(self):
        """Test get_model method"""
        model = self.runner.get_model()
        self.assertIsNotNone(model)

    def test_execute_empty_input(self):
        """Test _execute_empty_input method"""
        # Create empty batch_inputs and forward_meta
        batch_inputs = Mock()
        batch_inputs.input_ids = paddle.to_tensor([])

        forward_meta = Mock()
        forward_meta.decode_len = 0
        forward_meta.prefill_len = 0

        with patch.object(self.runner, "get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            result = self.runner._execute_empty_input(batch_inputs, forward_meta)
            self.assertIsNone(result)

    def test_dummy_prefill_inputs(self):
        """Test _dummy_prefill_inputs method"""
        # Set configuration
        self.runner.config.block_size = 16
        self.runner.config.max_model_len = 2048

        result = self.runner._dummy_prefill_inputs()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, tuple)

    def test_init_share_inputs(self):
        """Test _init_share_inputs method"""
        # Test initializing shared inputs
        with patch("paddle.empty") as mock_empty:
            mock_tensor = paddle.zeros([100])
            mock_empty.return_value = mock_tensor

            self.runner._init_share_inputs()
            # Verify method executes successfully (doesn't throw exception)
            self.assertTrue(True)

    def test_prepare_inputs(self):
        """Test _prepare_inputs method"""
        # Create mock requests and forward_meta
        mock_req = Mock()
        mock_req.prompt_token_len = 50
        mock_req.output_len = 100
        requests = [mock_req]

        forward_meta = Mock()
        forward_meta.decode_len = 1
        forward_meta.prefill_len = 50

        with patch.object(self.runner, "get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            result = self.runner._prepare_inputs(requests, forward_meta)
            self.assertIsNotNone(result)

    def test_update_chunked_prefill(self):
        """Test _update_chunked_prefill method"""
        # Create mock forward_meta
        forward_meta = Mock()
        forward_meta.prefill_len = 100
        forward_meta.decode_len = 0

        batch_inputs = Mock()
        batch_inputs.input_ids = paddle.randint(0, 1000, [100])

        result = self.runner._update_chunked_prefill(batch_inputs, forward_meta)
        self.assertIsNone(result)  # This method may have no return value or modify the input object

    def test_get_skip_idx(self):
        """Test _get_skip_idx method"""
        # Create mock forward_meta
        forward_meta = Mock()
        forward_meta.prefill_len = 100
        forward_meta.decode_len = 20
        forward_meta.seq_lens_this_time = paddle.to_tensor([50, 30, 20, 20])

        result = self.runner._get_skip_idx(forward_meta)
        self.assertIsNotNone(result)

    def test_pool(self):
        """Test _pool method"""
        # Create mock hidden_states and pooling_metadata
        hidden_states = paddle.randn([100, 768])

        pooling_metadata = Mock()
        pooling_metadata.seq_lens = [50, 50]
        pooling_metadata.pooling_type = "mean"

        with patch.object(self.runner, "pooler") as mock_pooler:
            mock_pooler.return_value = paddle.randn([2, 768])

            result = self.runner._pool(hidden_states, pooling_metadata)
            self.assertIsNotNone(result)

    def test_add_cache(self):
        """Test _add_cache method"""
        # Create test data
        mm_hashes = ["hash1", "hash2"]
        mm_features = paddle.randn([10, 256])
        mm_num_list = [5, 5]

        self.runner.mm_cache = {}

        result = self.runner._add_cache(mm_hashes, mm_features, mm_num_list)
        self.assertIsNone(result)  # This method has no return value
        self.assertEqual(len(self.runner.mm_cache), 2)

    def test_init_speculative_proposer(self):
        """Test _init_speculative_proposer method"""
        # Set speculative related configuration
        self.runner.config.speculative_config = Mock()
        self.runner.config.speculative_config.speculative_type = "mtp"

        with patch("fastdeploy.spec_decode.get_proposer") as mock_get_proposer:
            mock_proposer = MagicMock()
            mock_get_proposer.return_value = mock_proposer

            self.runner._init_speculative_proposer()
            self.assertIsNotNone(self.runner.speculative_proposer)

    def test_init_logits_processor(self):
        """Test _init_logits_processor method"""
        # Test initializing logits processor
        with patch("fastdeploy.model_executor.logits_processor.get_logits_processor") as mock_get_processor:
            mock_processor = MagicMock()
            mock_get_processor.return_value = mock_processor

            self.runner._init_logits_processor()
            self.assertIsNotNone(self.runner.logits_processor)

    def test_insert_tasks_v1(self):
        """Test insert_tasks_v1 method"""
        # Create mock tasks
        mock_task = Mock()
        mock_task.request_id = "req_1"
        mock_task.prompt = "test prompt"
        tasks = [mock_task]

        with patch.object(self.runner, "initialize_forward_meta") as mock_init:
            mock_meta = Mock()
            mock_init.return_value = mock_meta

            result = self.runner.insert_tasks_v1(tasks)
            self.assertIsNotNone(result)

    def test_insert_prefill_inputs(self):
        """Test insert_prefill_inputs method"""
        # Create mock inputs
        inputs = Mock()
        inputs.input_ids = paddle.randint(0, 1000, [50])
        inputs.position_ids = paddle.arange(50)

        forward_meta = Mock()
        forward_meta.prefill_len = 50

        result = self.runner.insert_prefill_inputs(inputs, forward_meta)
        self.assertIsNotNone(result)

    def test_initialize_kv_cache(self):
        """Test initialize_kv_cache method"""
        # Set necessary configuration
        self.runner.config.num_hidden_layers = 12
        self.runner.config.block_size = 16

        with patch.object(self.runner, "get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.config = Mock()
            mock_get_model.return_value = mock_model

            result = self.runner.initialize_kv_cache(num_blocks=100)
            self.assertIsNotNone(result)

    def test_initialize_attn_backend(self):
        """Test _initialize_attn_backend method"""
        # Test initializing attention backend
        with patch("fastdeploy.model_executor.layers.attention.get_attn_backend") as mock_get_backend:
            mock_backend = MagicMock()
            mock_get_backend.return_value = mock_backend

            self.runner._initialize_attn_backend()
            self.assertIsNotNone(self.runner.attn_backend)

    def test_dummy_pooler_run_task(self):
        """Test _dummy_pooler_run_task method"""
        # Create mock task
        mock_task = Mock()
        mock_task.task_type.value = "embedding"
        mock_task.pooling_params = Mock()

        result = self.runner._dummy_pooler_run_task(mock_task)
        self.assertIsNotNone(result)

    def test_dummy_pooler_run(self):
        """Test _dummy_pooler_run method"""
        # Test dummy pooler run
        result = self.runner._dummy_pooler_run()
        self.assertIsNone(result)

    def test_dummy_sampler_run(self):
        """Test _dummy_sampler_run method"""
        # Test dummy sampler run
        with patch.object(self.runner, "sampler") as mock_sampler:
            mock_sampler.return_value = []

            result = self.runner._dummy_sampler_run()
            self.assertIsNone(result)

    def test_dummy_run(self):
        """Test _dummy_run method"""
        # Test dummy run
        with patch.object(self.runner, "get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            try:
                self.runner._dummy_run()
                success = True
            except Exception:
                success = False

            self.assertTrue(success)

    def test_capture_model(self):
        """Test capture_model method"""
        # Test model capture
        with patch.object(self.runner, "get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            try:
                self.runner.capture_model()
                success = True
            except Exception:
                success = False

            self.assertTrue(success)

    def test_sot_warmup(self):
        """Test sot_warmup method"""
        # Test SOT warmup
        with patch.object(self.runner, "get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            try:
                self.runner.sot_warmup()
                success = True
            except Exception:
                success = False

            self.assertTrue(success)

    def test_execute_model(self):
        """Test execute_model method"""
        # Create mock batch_inputs and forward_meta
        batch_inputs = Mock()
        batch_inputs.input_ids = paddle.randint(0, 1000, [10])

        forward_meta = Mock()
        forward_meta.decode_len = 10
        forward_meta.prefill_len = 0

        with patch.object(self.runner, "get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.return_value = paddle.randn([10, 768])
            mock_get_model.return_value = mock_model

            result = self.runner.execute_model(batch_inputs, forward_meta)
            self.assertIsNotNone(result)

    def test_profile_run(self):
        """Test profile_run method"""
        # Create mock tasks
        mock_task = Mock()
        mock_task.request_id = "req_1"
        tasks = [mock_task]

        with patch.object(self.runner, "execute_model") as mock_execute:
            mock_execute.return_value = paddle.randn([10, 768])

            try:
                self.runner.profile_run(tasks, num_blocks=100)
                success = True
            except Exception:
                success = False

            self.assertTrue(success)

    def test_update_parameters(self):
        """Test update_parameters method"""
        # Create mock parameters
        parameters = {"temperature": 0.8, "top_p": 0.9}

        result = self.runner.update_parameters(parameters)
        self.assertIsNone(result)

    def test_padding_cudagraph_inputs(self):
        """Test padding_cudagraph_inputs method"""
        # Create mock batch_inputs
        batch_inputs = Mock()
        batch_inputs.input_ids = paddle.randint(0, 1000, [50])

        forward_meta = Mock()
        forward_meta.decode_len = 50

        result = self.runner.padding_cudagraph_inputs(batch_inputs, forward_meta, max_bs=64)
        self.assertIsNotNone(result)

    def test_init_image_preprocess(self):
        """Test _init_image_preprocess method"""
        # Test initializing image preprocessing
        with patch("fastdeploy.input.get_image_processor") as mock_get_processor:
            mock_processor = MagicMock()
            mock_get_processor.return_value = mock_processor

            self.runner._init_image_preprocess()
            self.assertIsNotNone(self.runner.image_processor)

    def test_preprocess_mm_task(self):
        """Test _preprocess_mm_task method"""
        # Create mock task
        mock_task = Mock()
        mock_task.mm_inputs = {"images": ["image1.jpg"]}
        mock_task.prompt = "Describe this image"

        with patch.object(self.runner, "image_processor") as mock_processor:
            mock_processor.return_value = {"pixel_values": paddle.randn([1, 3, 224, 224])}

            result = self.runner._preprocess_mm_task(mock_task)
            self.assertIsNotNone(result)

    def test_extract_vision_features(self):
        """Test extract_vision_features method"""
        # Create mock tasks
        mock_task = Mock()
        mock_task.mm_inputs = {"images": ["image1.jpg"]}
        tasks = [mock_task]

        with patch.object(self.runner, "get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.config = Mock()
            mock_model.config.architectures = ["LlamaForCausalLM"]
            mock_get_model.return_value = mock_model

            result = self.runner.extract_vision_features(tasks)
            self.assertIsNotNone(result)

    def test_prepare_rope3d(self):
        """Test prepare_rope3d method"""
        # Create test data
        grid_thw = paddle.to_tensor([[1, 2, 2], [1, 3, 3]])

        result = self.runner.prepare_rope3d(grid_thw)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, paddle.Tensor)

    def test_get_prompt_logprobs_list(self):
        """Test _get_prompt_logprobs_list method"""
        # Create mock requests and logits
        mock_req1 = Mock()
        mock_req1.prompt_token_len = 10
        mock_req1.sampling_params = Mock()
        mock_req1.sampling_params.logprobs = 5

        mock_req2 = Mock()
        mock_req2.prompt_token_len = 20
        mock_req2.sampling_params = Mock()
        mock_req2.sampling_params.logprobs = 0

        requests = [mock_req1, mock_req2]
        logits = paddle.randn([30, 32000])

        result = self.runner._get_prompt_logprobs_list(requests, logits)
        self.assertIsInstance(result, list)

    def test_apply_mm_inputs(self):
        """Test _apply_mm_inputs method"""
        # Create mock inputs and mm_features
        inputs = Mock()
        inputs.input_ids = paddle.randint(0, 1000, [100])

        mm_features = paddle.randn([10, 256])

        forward_meta = Mock()
        forward_meta.mm_num = 2

        result = self.runner._apply_mm_inputs(inputs, mm_features, forward_meta)
        self.assertIsNotNone(result)

    def test_extract_vision_features_ernie(self):
        """Test extract_vision_features_ernie method"""
        # Create mock tasks
        mock_task = Mock()
        mock_task.mm_inputs = {"images": ["image1.jpg"]}
        tasks = [mock_task]

        with patch.object(self.runner, "get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.vision_encoder = MagicMock()
            mock_get_model.return_value = mock_model

            result = self.runner.extract_vision_features_ernie(tasks)
            self.assertIsNotNone(result)

    def test_extract_vision_features_qwen(self):
        """Test extract_vision_features_qwen method"""
        # Create mock tasks
        mock_task = Mock()
        mock_task.mm_inputs = {"images": ["image1.jpg"]}
        tasks = [mock_task]

        with patch.object(self.runner, "get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.visual = MagicMock()
            mock_get_model.return_value = mock_model

            result = self.runner.extract_vision_features_qwen(tasks)
            self.assertIsNotNone(result)

    def test_extract_vision_features_paddleocr(self):
        """Test extract_vision_features_paddleocr method"""
        # Create mock tasks
        mock_task = Mock()
        mock_task.mm_inputs = {"images": ["image1.jpg"]}
        tasks = [mock_task]

        with patch.object(self.runner, "get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.encoder = MagicMock()
            mock_get_model.return_value = mock_model

            result = self.runner.extract_vision_features_paddleocr(tasks)
            self.assertIsNotNone(result)

    def test_async_output_busy_loop(self):
        """Test _async_output_busy_loop method"""
        # This is an async method, typically runs in background
        # We test that it can be called without throwing exception
        try:
            # Since it's an async method, we only test its existence
            self.assertTrue(hasattr(self.runner, "_async_output_busy_loop"))
            success = True
        except Exception:
            success = False

        self.assertTrue(success)

    def test_load_model(self):
        """Test load_model method"""
        with patch("fastdeploy.model_executor.model_loader.get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            result = self.runner.load_model()
            self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
