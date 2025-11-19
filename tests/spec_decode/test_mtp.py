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
                proposer.num_gpu_blocks = int(main_model_num_blocks * proposer.speculative_config.num_gpu_block_expand_ratio)
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


if __name__ == "__main__":
    unittest.main()
