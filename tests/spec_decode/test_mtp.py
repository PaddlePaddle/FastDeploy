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

import paddle

from fastdeploy.spec_decode.mtp import MTPProposer

# Note: This test requires dependencies like paddleformers, paddle, etc.
# In CI environment, these should be available.
# For local testing without dependencies, you may need to install them or use CI.


class TestMTPProposer(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Mock FDConfig
        self.mock_fd_config = MagicMock()
        self.mock_fd_config.model_config = MagicMock()
        self.mock_fd_config.model_config.architectures = ["ErnieMoeForCausalLM"]
        self.mock_fd_config.model_config.num_hidden_layers = 32
        self.mock_fd_config.model_config.max_model_len = 2048
        self.mock_fd_config.model_config.hidden_size = 1024
        self.mock_fd_config.model_config.num_attention_heads = 16
        self.mock_fd_config.model_config.num_key_value_heads = 16
        self.mock_fd_config.model_config.head_dim = 64
        self.mock_fd_config.model_config.rope_theta = 10000.0
        self.mock_fd_config.model_config.enable_logprob = False
        self.mock_fd_config.model_config.dtype = "float16"
        self.mock_fd_config.speculative_config = MagicMock()
        self.mock_fd_config.speculative_config.mtp_strategy = "standard"
        self.mock_fd_config.speculative_config.num_gpu_block_expand_ratio = 1.0
        self.mock_fd_config.speculative_config.model = "test_model"
        self.mock_fd_config.speculative_config.quantization = ""
        self.mock_fd_config.speculative_config.method = "mtp"
        self.mock_fd_config.speculative_config.num_speculative_tokens = 8
        self.mock_fd_config.speculative_config.num_model_steps = 4
        self.mock_fd_config.speculative_config.max_ngram_size = 4
        self.mock_fd_config.speculative_config.min_ngram_size = 2
        self.mock_fd_config.scheduler_config = MagicMock()
        self.mock_fd_config.scheduler_config.splitwise_role = "mixed"
        self.mock_fd_config.scheduler_config.max_num_seqs = 8
        self.mock_fd_config.scheduler_config.max_num_batched_tokens = 4096
        self.mock_fd_config.cache_config = MagicMock()
        self.mock_fd_config.cache_config.block_size = 16
        self.mock_fd_config.cache_config.enc_dec_block_num = 0
        self.mock_fd_config.cache_config.total_block_num = 100
        self.mock_fd_config.cache_config.kv_cache_ratio = 0.9
        self.mock_fd_config.cache_config.enable_prefix_caching = False
        self.mock_fd_config.cache_config.enable_chunked_prefill = False
        self.mock_fd_config.graph_opt_config = MagicMock()
        self.mock_fd_config.graph_opt_config.draft_model_use_cudagraph = False
        self.mock_fd_config.graph_opt_config.cudagraph_capture_sizes = []
        self.mock_fd_config.graph_opt_config.sot_warmup_sizes = []
        self.mock_fd_config.parallel_config = MagicMock()
        self.mock_fd_config.parallel_config.tensor_parallel_size = 1
        self.mock_fd_config.parallel_config.enable_expert_parallel = False
        self.mock_fd_config.quant_config = None
        self.mock_fd_config.load_config = MagicMock()
        self.mock_fd_config.max_num_seqs = 8
        self.mock_fd_config.max_prefill_batch = 4
        self.mock_fd_config.model_config.enable_mm = False

        # Mock main model
        self.mock_main_model = MagicMock()

        # Mock target model inputs
        self.mock_target_model_inputs = {
            "block_tables": paddle.zeros([8, 100], dtype="int32"),
            "input_ids": paddle.zeros([8, 2048], dtype="int64"),
            "seq_lens_this_time": paddle.zeros([8], dtype="int32"),
            "seq_lens_encoder": paddle.zeros([8], dtype="int32"),
            "seq_lens_decoder": paddle.zeros([8], dtype="int32"),
            "step_idx": paddle.zeros([8], dtype="int32"),
            "stop_flags": paddle.zeros([8], dtype="bool"),
            "stop_nums": paddle.zeros([8], dtype="int32"),
            "pre_ids": paddle.zeros([8], dtype="int64"),
            "output_cum_offsets": paddle.zeros([8], dtype="int32"),
            "output_padding_offset": paddle.zeros([8], dtype="int32"),
            "ids_remove_padding": paddle.zeros([8], dtype="int64"),
            "batch_id_per_token": paddle.zeros([8], dtype="int32"),
            "cu_seqlens_q": paddle.zeros([9], dtype="int32"),
            "cu_seqlens_k": paddle.zeros([9], dtype="int32"),
            "decoder_batch_ids": paddle.zeros([8], dtype="int32"),
            "decoder_tile_ids_per_batch": paddle.zeros([8], dtype="int32"),
            "decoder_num_blocks_cpu": paddle.zeros([8], dtype="int32"),
            "decoder_num_blocks_device": paddle.zeros([8], dtype="int32"),
            "decoder_chunk_size_device": paddle.zeros([8], dtype="int32"),
            "max_len_tensor_cpu": paddle.zeros([8], dtype="int32"),
            "encoder_batch_ids": paddle.zeros([8], dtype="int32"),
            "encoder_tile_ids_per_batch": paddle.zeros([8], dtype="int32"),
            "encoder_num_blocks_x_cpu": paddle.zeros([8], dtype="int32"),
            "kv_batch_ids": paddle.zeros([8], dtype="int32"),
            "kv_tile_ids_per_batch": paddle.zeros([8], dtype="int32"),
            "kv_num_blocks_x_cpu": paddle.zeros([8], dtype="int32"),
            "prompt_lens": paddle.zeros([8], dtype="int32"),
            "top_p": paddle.ones([8], dtype="float32") * 0.7,
            "top_k": paddle.ones([8], dtype="int32") * 50,
            "temperature": paddle.ones([8], dtype="float32") * 1.0,
            "eos_token_id": paddle.ones([8, 1], dtype="int64") * 2,
            "penalty_score": paddle.ones([8], dtype="float32"),
            "frequency_score": paddle.zeros([8], dtype="float32"),
            "presence_score": paddle.zeros([8], dtype="float32"),
            "infer_seed": paddle.zeros([8], dtype="int64"),
            "max_dec_len": paddle.ones([8], dtype="int32") * 256,
            "min_dec_len": paddle.zeros([8], dtype="int32"),
            "bad_tokens": paddle.zeros([8, 0], dtype="int64"),
            "draft_tokens": paddle.zeros([8, 10], dtype="int64"),
            "accept_tokens": paddle.zeros([8, 10], dtype="int64"),
            "accept_num": paddle.zeros([8], dtype="int32"),
            "encoder_block_lens": paddle.zeros([8], dtype="int32"),
            "cu_batch_token_offset": paddle.zeros([9], dtype="int32"),
            "temp_scaled_logprobs": None,
            "top_p_normalized_logprobs": None,
            "draft_logits": None,
        }

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    def test_init(self, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader):
        """Test MTPProposer initialization"""
        # Mock model loader
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        # Mock attention backend
        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        # Mock rope
        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])

        # Mock sampler
        mock_sampler_instance = MagicMock()
        mock_sampler.return_value = mock_sampler_instance

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        self.assertIsNotNone(proposer)
        self.assertEqual(proposer.num_main_model_layers, 32)
        self.assertEqual(proposer.local_rank, 0)
        self.assertEqual(proposer.device_id, 0)

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    def test_update_mtp_config(self, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader):
        """Test _update_mtp_config method"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        self.assertEqual(proposer.model_config.num_hidden_layers, 1)
        self.assertEqual(proposer.speculative_config.model_type, "mtp")

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    def test_dummy_prefill_inputs(self, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader):
        """Test dummy_prefill_inputs method"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        proposer.dummy_prefill_inputs(num_tokens=100, batch_size=2, expected_decode_len=10)
        self.assertGreater(proposer.model_inputs["seq_lens_this_time"][0].item(), 0)

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    @patch("fastdeploy.spec_decode.mtp.share_external_data")
    def test_initialize_kv_cache(
        self, mock_share_data, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader
    ):
        """Test initialize_kv_cache method"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        mock_share_data.side_effect = lambda x, y, z: x

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        proposer.initialize_kv_cache(main_model_num_blocks=100)
        self.assertIn("caches", proposer.model_inputs)

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    def test_clear_mtp_cache(self, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader):
        """Test clear_mtp_cache method"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        proposer.model_inputs["caches"] = [paddle.zeros([1])]
        proposer.forward_meta = MagicMock()
        proposer.forward_meta.caches = [paddle.zeros([1])]

        proposer.clear_mtp_cache()
        self.assertNotIn("caches", proposer.model_inputs)

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    def test_update_mtp_block_num(self, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader):
        """Test update_mtp_block_num method"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        with patch.object(proposer, "initialize_kv_cache") as mock_init_cache:

            def side_effect(main_model_num_blocks):
                proposer.num_gpu_blocks = int(
                    main_model_num_blocks * proposer.speculative_config.num_gpu_block_expand_ratio
                )

            mock_init_cache.side_effect = side_effect
            proposer.update_mtp_block_num(num_gpu_blocks=50)
            mock_init_cache.assert_called_once_with(main_model_num_blocks=50)
            self.assertEqual(proposer.main_model_num_gpu_blocks, 50)

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    def test_exist_prefill(self, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader):
        """Test exist_prefill method"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        proposer.share_inputs = {}
        proposer.share_inputs["seq_lens_encoder"] = paddle.zeros([8], dtype="int32")
        result = proposer.exist_prefill()
        self.assertEqual(result, 0)

        proposer.share_inputs["seq_lens_encoder"] = paddle.ones([8], dtype="int32")
        result = proposer.exist_prefill()
        self.assertEqual(result, 1)

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    def test_is_chunk_prefill_enabled(self, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader):
        """Test is_chunk_prefill_enabled method"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        self.assertTrue(proposer.is_chunk_prefill_enabled())

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    def test_initialize_kv_cache_with_prefix_caching(
        self, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader
    ):
        """Test initialize_kv_cache with prefix caching enabled"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        # Enable prefix caching
        self.mock_fd_config.cache_config.enable_prefix_caching = True

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        with patch("fastdeploy.spec_decode.mtp.share_external_data") as mock_share_data:
            mock_share_data.side_effect = lambda x, y, z: x
            proposer.initialize_kv_cache(main_model_num_blocks=100)
            self.assertIn("caches", proposer.model_inputs)

        # Reset
        self.mock_fd_config.cache_config.enable_prefix_caching = False

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    def test_initialize_kv_cache_with_profile(
        self, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader
    ):
        """Test initialize_kv_cache with profile=True"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        proposer.initialize_kv_cache(main_model_num_blocks=100, profile=True)
        self.assertIn("caches", proposer.model_inputs)

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    def test_insert_tasks_v1_prefill(self, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader):
        """Test insert_tasks_v1 with prefill task"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        with patch("fastdeploy.spec_decode.mtp.share_external_data") as mock_share_data:
            mock_share_data.side_effect = lambda x, y, z: x
            proposer.initialize_kv_cache(main_model_num_blocks=100)

        # Create mock request for prefill
        mock_request = MagicMock()
        mock_request.request_id = "test_req_1"
        mock_request.idx = 0
        mock_request.task_type.value = 0  # RequestType.PREFILL
        mock_request.prefill_start_index = 0
        mock_request.prefill_end_index = 10
        mock_request.prompt_token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        mock_request.output_token_ids = []
        mock_request.block_tables = [0, 1]
        mock_request.multimodal_inputs = None

        proposer.insert_tasks_v1([mock_request], num_running_requests=1)
        self.assertFalse(proposer.model_inputs["stop_flags"][0].item())

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    def test_insert_tasks_v1_decode(self, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader):
        """Test insert_tasks_v1 with decode task"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        with patch("fastdeploy.spec_decode.mtp.share_external_data") as mock_share_data:
            mock_share_data.side_effect = lambda x, y, z: x
            proposer.initialize_kv_cache(main_model_num_blocks=100)

        # Create mock request for decode
        mock_request = MagicMock()
        mock_request.request_id = "test_req_2"
        mock_request.idx = 0
        mock_request.task_type.value = 1  # RequestType.DECODE
        mock_request.block_tables = [0, 1, 2]

        proposer.insert_tasks_v1([mock_request], num_running_requests=1)
        # Verify block_tables updated
        self.assertEqual(proposer.model_inputs["encoder_block_lens"][0].item(), 3)

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    def test_insert_tasks_v1_other_type(
        self, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader
    ):
        """Test insert_tasks_v1 with other task type (cleanup)"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        with patch("fastdeploy.spec_decode.mtp.share_external_data") as mock_share_data:
            mock_share_data.side_effect = lambda x, y, z: x
            proposer.initialize_kv_cache(main_model_num_blocks=100)

        # Create mock request for other type
        mock_request = MagicMock()
        mock_request.request_id = "test_req_3"
        mock_request.idx = 0
        mock_request.task_type.value = 2  # Other type

        proposer.insert_tasks_v1([mock_request], num_running_requests=1)
        # Verify cleanup happened
        self.assertTrue(proposer.model_inputs["stop_flags"][0].item())

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    def test_insert_prefill_inputs_mixed_role(
        self, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader
    ):
        """Test insert_prefill_inputs with mixed role"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        # Create mock request without disaggregate_info
        mock_request = MagicMock()
        mock_request.idx = 0
        mock_request.prompt_token_ids = [1, 2, 3, 4, 5]
        mock_request.disaggregate_info = None
        mock_request.prefill_chunk_info = None
        mock_request.get = MagicMock(
            side_effect=lambda key, default=None: {"seq_lens_decoder": 0, "block_tables": [0, 1]}.get(key, default)
        )

        proposer.insert_prefill_inputs([mock_request], num_running_requests=1)
        self.assertEqual(proposer.role, "mixed")

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    def test_insert_prefill_inputs_decode_role(
        self, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader
    ):
        """Test insert_prefill_inputs with decode role from disaggregate_info"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        # Create mock request with decode role
        mock_request = MagicMock()
        mock_request.idx = 0
        mock_request.prompt_token_ids = [1, 2, 3, 4, 5]
        mock_request.draft_token_ids = [0, 6, 7]
        mock_request.disaggregate_info = {"role": "decode"}
        mock_request.block_tables = [0, 1]

        proposer.insert_prefill_inputs([mock_request], num_running_requests=1)
        self.assertEqual(proposer.role, "decode")

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    def test_insert_prefill_inputs_prefill_role(
        self, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader
    ):
        """Test insert_prefill_inputs with prefill role from disaggregate_info"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        # Create mock request with prefill role
        mock_request = MagicMock()
        mock_request.idx = 0
        mock_request.prompt_token_ids = [1, 2, 3, 4, 5]
        mock_request.disaggregate_info = {"role": "prefill"}
        mock_request.get = MagicMock(
            side_effect=lambda key, default=None: {"seq_lens_decoder": 0, "block_tables": [0, 1]}.get(key, default)
        )

        proposer.insert_prefill_inputs([mock_request], num_running_requests=1)
        self.assertEqual(proposer.role, "prefill")

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    def test_insert_prefill_inputs_with_chunked_prefill(
        self, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader
    ):
        """Test insert_prefill_inputs with chunked prefill enabled"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        # Enable chunked prefill
        self.mock_fd_config.cache_config.enable_chunked_prefill = True

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        # Create mock request with chunked prefill
        mock_request = MagicMock()
        mock_request.idx = 0
        mock_request.prompt_token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        mock_request.disaggregate_info = None
        mock_request.prefill_chunk_info = [5, 5]  # Two chunks of 5 tokens each
        mock_request.get = MagicMock(
            side_effect=lambda key, default=None: {"seq_lens_decoder": 0, "block_tables": [0, 1]}.get(key, default)
        )

        proposer.insert_prefill_inputs([mock_request], num_running_requests=1)
        self.assertEqual(proposer.model_inputs["seq_lens_encoder"][0].item(), 5)

        # Reset
        self.mock_fd_config.cache_config.enable_chunked_prefill = False

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    def test_initialize_forward_meta(self, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader):
        """Test _initialize_forward_meta method"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        with patch("fastdeploy.spec_decode.mtp.share_external_data") as mock_share_data:
            mock_share_data.side_effect = lambda x, y, z: x
            proposer.initialize_kv_cache(main_model_num_blocks=100)

        proposer._initialize_forward_meta(step_use_cudagraph=False)
        self.assertIsNotNone(proposer.forward_meta)

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    def test_initialize_forward_meta_with_cudagraph(
        self, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader
    ):
        """Test _initialize_forward_meta with cudagraph enabled"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        # Enable cudagraph
        self.mock_fd_config.graph_opt_config.draft_model_use_cudagraph = True

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        with patch("fastdeploy.spec_decode.mtp.share_external_data") as mock_share_data:
            mock_share_data.side_effect = lambda x, y, z: x
            proposer.initialize_kv_cache(main_model_num_blocks=100)

        proposer._initialize_forward_meta(step_use_cudagraph=True)
        self.assertTrue(proposer.forward_meta.step_use_cudagraph)

        # Reset
        self.mock_fd_config.graph_opt_config.draft_model_use_cudagraph = False

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    @patch("fastdeploy.spec_decode.mtp.draft_model_preprocess")
    @patch("fastdeploy.spec_decode.mtp.eagle_get_hidden_states")
    def test_prepare_inputs(
        self,
        mock_eagle_get,
        mock_preprocess,
        mock_sampler,
        mock_get_rope,
        mock_get_attn_backend,
        mock_get_model_loader,
    ):
        """Test _prepare_inputs method"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        # Mock eagle_get_hidden_states return
        mock_eagle_get.return_value = paddle.zeros([100, 1024])

        full_hidden_states = paddle.zeros([100, 1024])
        proposer._prepare_inputs(full_hidden_states)

        mock_preprocess.assert_called_once()
        mock_eagle_get.assert_called_once()

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    @patch("fastdeploy.spec_decode.mtp.draft_model_update")
    def test_post_process(
        self, mock_update, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader
    ):
        """Test _post_process method"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        sampled_token_ids = paddle.zeros([8], dtype="int64")
        proposer._post_process(sampled_token_ids)

        mock_update.assert_called_once()

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    @patch("fastdeploy.spec_decode.mtp.draft_model_update")
    @patch("fastdeploy.spec_decode.mtp.mtp_save_first_token")
    def test_post_process_prefill_role(
        self, mock_save_first, mock_update, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader
    ):
        """Test _post_process method with prefill role"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        # Set prefill role
        proposer.role = "prefill"
        proposer.parallel_config.tensor_parallel_rank = 0

        sampled_token_ids = paddle.zeros([8], dtype="int64")
        proposer._post_process(sampled_token_ids)

        mock_update.assert_called_once()
        mock_save_first.assert_called_once()

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    @patch("fastdeploy.spec_decode.mtp.eagle_get_self_hidden_states")
    def test_get_self_hidden_states(
        self, mock_eagle_self, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader
    ):
        """Test _get_self_hidden_states method"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        # Set num_model_steps > 1 to initialize last_seq_lens_this_time
        self.mock_fd_config.speculative_config.num_model_steps = 2

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        mock_eagle_self.return_value = paddle.zeros([100, 1024])

        hidden_states = paddle.zeros([100, 1024])
        proposer._get_self_hidden_states(hidden_states)

        mock_eagle_self.assert_called_once()

        # Reset
        self.mock_fd_config.speculative_config.num_model_steps = 4

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    def test_update_task_chunk_prefill_complete(
        self, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader
    ):
        """Test update_task_chunk_prefill when chunk is complete"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        # Create mock task - chunk complete
        mock_task = MagicMock()
        mock_task.idx = 0
        mock_task.prefill_chunk_info = [5, 5]
        mock_task.chunk_idx = 2  # All chunks processed
        mock_task.get = MagicMock(return_value=0)

        proposer.update_task_chunk_prefill(mock_task)
        self.assertEqual(proposer.model_inputs["seq_lens_encoder"][0].item(), 0)
        self.assertEqual(proposer.model_inputs["step_idx"][0].item(), 1)

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    def test_update_task_chunk_prefill_middle_chunk(
        self, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader
    ):
        """Test update_task_chunk_prefill with middle chunk"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        # Create mock task - middle chunk
        mock_task = MagicMock()
        mock_task.idx = 0
        mock_task.prefill_chunk_info = [5, 5, 5]
        mock_task.chunk_idx = 1  # Second chunk
        mock_task.prompt_token_ids = list(range(15))
        mock_task.get = MagicMock(return_value=0)

        proposer.update_task_chunk_prefill(mock_task)
        self.assertEqual(proposer.model_inputs["seq_lens_encoder"][0].item(), 5)
        self.assertEqual(proposer.model_inputs["step_idx"][0].item(), 0)

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    def test_update_task_chunk_prefill_last_chunk(
        self, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader
    ):
        """Test update_task_chunk_prefill with last chunk"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        # Create mock task - last chunk
        mock_task = MagicMock()
        mock_task.idx = 0
        mock_task.prefill_chunk_info = [5, 5]
        mock_task.chunk_idx = 1  # Last chunk
        mock_task.prompt_token_ids = list(range(10))
        mock_task.get = MagicMock(return_value=0)

        proposer.update_task_chunk_prefill(mock_task)
        self.assertEqual(proposer.model_inputs["seq_lens_encoder"][0].item(), 5)

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    @patch("fastdeploy.spec_decode.mtp.draft_model_postprocess")
    @patch("fastdeploy.spec_decode.mtp.mtp_step_paddle")
    def test_update_status(
        self,
        mock_mtp_step,
        mock_postprocess,
        mock_sampler,
        mock_get_rope,
        mock_get_attn_backend,
        mock_get_model_loader,
    ):
        """Test _update_status method"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        with patch("fastdeploy.spec_decode.mtp.envs") as mock_envs:
            mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
            proposer._update_status()

        mock_postprocess.assert_called_once()
        mock_mtp_step.assert_called_once()

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    @patch("fastdeploy.spec_decode.mtp.hybrid_mtp_ngram")
    def test_extend_draft_token_with_ngram_match(
        self, mock_ngram, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader
    ):
        """Test _extend_draft_token_with_ngram_match method"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        # Add required inputs
        self.mock_target_model_inputs["actual_draft_token_num"] = paddle.zeros([8], dtype="int32")

        proposer._extend_draft_token_with_ngram_match()
        mock_ngram.assert_called_once()

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    def test_run_impl(self, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader):
        """Test _run_impl method"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        with (
            patch.object(proposer, "_prepare_inputs") as mock_prepare,
            patch.object(proposer, "_propose") as mock_propose,
            patch.object(proposer, "_update_status") as mock_update,
        ):

            full_hidden_states = paddle.zeros([100, 1024])
            proposer._run_impl(full_hidden_states)

            mock_prepare.assert_called_once_with(full_hidden_states)
            mock_propose.assert_called_once()
            mock_update.assert_called_once()

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    def test_run_impl_with_hybrid_mode(
        self, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader
    ):
        """Test _run_impl method with hybrid mode enabled"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        # Enable hybrid mode
        self.mock_fd_config.speculative_config.mtp_strategy = "with_ngram"
        self.mock_fd_config.speculative_config.num_speculative_tokens = 10
        self.mock_fd_config.speculative_config.num_model_steps = 2

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        with (
            patch.object(proposer, "_prepare_inputs") as mock_prepare,
            patch.object(proposer, "_propose") as mock_propose,
            patch.object(proposer, "_update_status") as mock_update,
            patch.object(proposer, "_extend_draft_token_with_ngram_match") as mock_extend,
        ):

            full_hidden_states = paddle.zeros([100, 1024])
            proposer._run_impl(full_hidden_states)

            mock_prepare.assert_called_once()
            mock_propose.assert_called_once()
            mock_update.assert_called_once()
            mock_extend.assert_called_once()

        # Reset
        self.mock_fd_config.speculative_config.mtp_strategy = "standard"
        self.mock_fd_config.speculative_config.num_speculative_tokens = 8
        self.mock_fd_config.speculative_config.num_model_steps = 4

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    def test_padding_cudagraph_inputs(self, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader):
        """Test padding_cudagraph_inputs method"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        with patch("fastdeploy.spec_decode.mtp.share_external_data") as mock_share_data:
            mock_share_data.side_effect = lambda x, y, z: x
            proposer.initialize_kv_cache(main_model_num_blocks=100)

        proposer._initialize_forward_meta(step_use_cudagraph=False)

        # Test without cudagraph
        proposer.padding_cudagraph_inputs()
        self.assertIsNone(getattr(proposer, "real_token_num", None))

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    def test_padding_cudagraph_inputs_with_cudagraph(
        self, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader
    ):
        """Test padding_cudagraph_inputs with cudagraph enabled"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        # Enable cudagraph
        self.mock_fd_config.graph_opt_config.draft_model_use_cudagraph = True

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        with patch("fastdeploy.spec_decode.mtp.share_external_data") as mock_share_data:
            mock_share_data.side_effect = lambda x, y, z: x
            proposer.initialize_kv_cache(main_model_num_blocks=100)

        proposer._initialize_forward_meta(step_use_cudagraph=True)

        # Set ids_remove_padding for test
        proposer.forward_meta.ids_remove_padding = paddle.zeros([50], dtype="int64")

        proposer.padding_cudagraph_inputs()
        self.assertEqual(proposer.real_token_num, 50)

        # Reset
        self.mock_fd_config.graph_opt_config.draft_model_use_cudagraph = False

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    def test_insert_tasks_v1_with_mm_enabled(
        self, mock_sampler, mock_get_rope, mock_get_attn_backend, mock_get_model_loader
    ):
        """Test insert_tasks_v1 with multimodal enabled"""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mock_get_model_loader.return_value = mock_loader

        mock_attn_backend = MagicMock()
        mock_attn_backend.get_kv_cache_shape.return_value = ([1, 16, 16, 64], [1, 16, 16, 64])
        mock_get_attn_backend.return_value = MagicMock(return_value=mock_attn_backend)

        mock_get_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_sampler.return_value = MagicMock()

        # Enable multimodal
        self.mock_fd_config.model_config.enable_mm = True

        proposer = MTPProposer(
            fd_config=self.mock_fd_config,
            main_model=self.mock_main_model,
            local_rank=0,
            device_id=0,
            target_model_inputs=self.mock_target_model_inputs,
        )

        with patch("fastdeploy.spec_decode.mtp.share_external_data") as mock_share_data:
            mock_share_data.side_effect = lambda x, y, z: x
            proposer.initialize_kv_cache(main_model_num_blocks=100)

        # Create mock request for prefill with mm
        mock_request = MagicMock()
        mock_request.request_id = "test_req_mm"
        mock_request.idx = 0
        mock_request.task_type.value = 0  # RequestType.PREFILL
        mock_request.prefill_start_index = 0
        mock_request.prefill_end_index = 10
        mock_request.prompt_token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        mock_request.output_token_ids = []
        mock_request.block_tables = [0, 1]
        mock_request.multimodal_inputs = {"attention_mask_offset": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}

        proposer.insert_tasks_v1([mock_request], num_running_requests=1)
        self.assertFalse(proposer.model_inputs["stop_flags"][0].item())

        # Reset
        self.mock_fd_config.model_config.enable_mm = False


if __name__ == "__main__":
    unittest.main()
