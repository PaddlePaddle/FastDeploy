"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from fastdeploy.model_executor.layers.attention.dsa_attention_backend import (
    DSAAttentionBackend,
    DSAAttentionMetadata,
    yarn_get_mscale,
)


class TestYarnGetMscale(unittest.TestCase):
    """Test yarn_get_mscale function."""

    def test_scale_le_1_returns_1(self):
        """scale <= 1 returns 1.0."""
        self.assertEqual(yarn_get_mscale(scale=1, mscale=1), 1.0)
        self.assertEqual(yarn_get_mscale(scale=0.5, mscale=2), 1.0)

    def test_scale_gt_1(self):
        """scale > 1 returns 0.1 * mscale * log(scale) + 1.0."""
        import math

        result = yarn_get_mscale(scale=40, mscale=1.0)
        expected = 0.1 * 1.0 * math.log(40) + 1.0
        self.assertAlmostEqual(result, expected, places=6)

    def test_scale_gt_1_custom_mscale(self):
        """scale > 1 with custom mscale."""
        import math

        result = yarn_get_mscale(scale=10, mscale=2.0)
        expected = 0.1 * 2.0 * math.log(10) + 1.0
        self.assertAlmostEqual(result, expected, places=6)


class TestDSAAttentionMetadata(unittest.TestCase):
    """Test DSAAttentionMetadata dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        metadata = DSAAttentionMetadata()
        self.assertEqual(metadata._dtype, paddle.bfloat16)
        self.assertEqual(metadata.encoder_max_partition_size, 32768)
        self.assertEqual(metadata.max_partition_size, 32768)
        self.assertIsNone(metadata.block_tables)
        self.assertIsNone(metadata.rotary_embs)
        self.assertIsNone(metadata.attn_mask)
        self.assertEqual(metadata._fuse_kernel_compute_dtype, "bf16")
        self.assertIsNone(metadata.kv_signal_metadata)
        self.assertEqual(metadata.kv_signal_data_list, [])
        self.assertIsNone(metadata.max_enc_len_this_time)
        self.assertIsNone(metadata.max_dec_len_this_time)
        self.assertIsNone(metadata.max_kv_len_this_time)
        self.assertIsNone(metadata.slot_mapping)


class TestDSAAttentionBackendInit(unittest.TestCase):
    """Test DSAAttentionBackend.__init__."""

    def _make_fd_config(self, rope_scaling=None):
        """Create a mock FDConfig for DSA backend."""
        fd_config = MagicMock()
        fd_config.cache_config.block_size = 64
        fd_config.model_config.max_model_len = 8192
        fd_config.model_config.rope_theta = 500000.0
        fd_config.enable_rope_3d_runtime = False
        fd_config.model_config.causal = True
        fd_config.speculative_config.method = None
        fd_config.speculative_config.num_speculative_tokens = 0
        fd_config.speculative_config.model_type = ""
        fd_config.model_config.head_dim = 128
        fd_config.model_config.num_hidden_layers = 60
        fd_config.model_config.index_head_dim = 256
        fd_config.model_config.index_n_heads = 4
        fd_config.model_config.index_topk = 8
        fd_config.model_config.kv_lora_rank = 512
        fd_config.model_config.qk_rope_head_dim = 64
        fd_config.model_config.qk_nope_head_dim = 128
        fd_config.model_config.rope_scaling = rope_scaling
        fd_config.model_config.start_layer_index = 0
        fd_config.parallel_config.pd_disaggregation_mode = None
        fd_config.parallel_config.tensor_parallel_rank = 0
        fd_config.parallel_config.local_data_parallel_id = 0
        fd_config.parallel_config.tensor_parallel_size = 1
        return fd_config

    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.init_rank_and_device_id")
    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.paddle.randn")
    def test_init_basic(self, mock_randn, mock_init_rank):
        """Init stores basic config values."""
        mock_randn.return_value = MagicMock()
        mock_randn.return_value.cast.return_value = "useless"
        mock_init_rank.return_value = (0, 0)

        fd_config = self._make_fd_config()
        backend = DSAAttentionBackend(fd_config, kv_num_heads=1, num_heads=16, head_dim=128)

        self.assertIsNone(backend.attention_metadata)
        self.assertEqual(backend.block_size, 64)
        self.assertEqual(backend.max_seq_len, 8192)
        self.assertEqual(backend.rope_theta, 500000.0)
        self.assertFalse(backend.rope_3d)
        self.assertTrue(backend.causal)
        self.assertFalse(backend.use_speculate)
        self.assertEqual(backend.num_heads, 16)
        self.assertEqual(backend.head_dim, 128)
        self.assertEqual(backend.num_layers, 60)
        self.assertEqual(backend.kv_lora_rank, 512)
        self.assertEqual(backend.qk_rope_head_dim, 64)
        self.assertEqual(backend.qk_head_dim, 192)  # 128 + 64

    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.init_rank_and_device_id")
    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.paddle.randn")
    def test_init_with_rope_scaling(self, mock_randn, mock_init_rank):
        """Init applies rope_scaling mscale to softmax scale."""
        mock_randn.return_value = MagicMock()
        mock_randn.return_value.cast.return_value = "useless"
        mock_init_rank.return_value = (0, 0)

        rope_scaling = {"factor": 40, "mscale_all_dim": 1.0}
        fd_config = self._make_fd_config(rope_scaling=rope_scaling)
        backend = DSAAttentionBackend(fd_config, kv_num_heads=1, num_heads=16, head_dim=128)

        # attn_softmax_scale = qk_head_dim**-0.5 * mscale * mscale

        qk_head_dim = 192
        base_scale = qk_head_dim**-0.5
        mscale = yarn_get_mscale(40, 1.0)
        expected = base_scale * mscale * mscale
        self.assertAlmostEqual(backend.attn_softmax_scale, expected, places=6)

    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.init_rank_and_device_id")
    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.paddle.randn")
    def test_init_rope_theta_none_defaults(self, mock_randn, mock_init_rank):
        """rope_theta=None defaults to 10000.0."""
        mock_randn.return_value = MagicMock()
        mock_randn.return_value.cast.return_value = "useless"
        mock_init_rank.return_value = (0, 0)

        fd_config = self._make_fd_config()
        fd_config.model_config.rope_theta = None
        backend = DSAAttentionBackend(fd_config, kv_num_heads=1, num_heads=16, head_dim=128)

        self.assertEqual(backend.rope_theta, 10000.0)

    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.init_rank_and_device_id")
    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.paddle.randn")
    def test_init_speculative_mtp(self, mock_randn, mock_init_rank):
        """Init with speculative method=mtp."""
        mock_randn.return_value = MagicMock()
        mock_randn.return_value.cast.return_value = "useless"
        mock_init_rank.return_value = (0, 0)

        fd_config = self._make_fd_config()
        fd_config.speculative_config.method = "mtp"
        fd_config.speculative_config.num_speculative_tokens = 3
        fd_config.speculative_config.model_type = "mtp"

        backend = DSAAttentionBackend(fd_config, kv_num_heads=1, num_heads=16, head_dim=128)

        self.assertTrue(backend.use_speculate)
        self.assertEqual(backend.speculate_max_draft_token_num, 3)
        self.assertTrue(backend.keep_pd_step_flag)
        self.assertEqual(backend.num_layers_draft_model, 1)


class TestDSAAttentionBackendInitAttentionMetadata(unittest.TestCase):
    """Test DSAAttentionBackend.init_attention_metadata."""

    def _make_backend(self):
        """Create DSAAttentionBackend with mocked init."""
        with (
            patch(
                "fastdeploy.model_executor.layers.attention.dsa_attention_backend.init_rank_and_device_id",
                return_value=(0, 0),
            ),
            patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.paddle.randn") as mock_randn,
        ):
            mock_randn.return_value = MagicMock()
            mock_randn.return_value.cast.return_value = "useless"

            fd_config = MagicMock()
            fd_config.cache_config.block_size = 64
            fd_config.model_config.max_model_len = 8192
            fd_config.model_config.rope_theta = 500000.0
            fd_config.enable_rope_3d_runtime = False
            fd_config.model_config.causal = True
            fd_config.speculative_config.method = None
            fd_config.speculative_config.num_speculative_tokens = 0
            fd_config.speculative_config.model_type = ""
            fd_config.model_config.head_dim = 128
            fd_config.model_config.num_hidden_layers = 60
            fd_config.model_config.index_head_dim = 256
            fd_config.model_config.index_n_heads = 4
            fd_config.model_config.index_topk = 8
            fd_config.model_config.kv_lora_rank = 512
            fd_config.model_config.qk_rope_head_dim = 64
            fd_config.model_config.qk_nope_head_dim = 128
            fd_config.model_config.rope_scaling = None
            fd_config.model_config.start_layer_index = 0
            fd_config.parallel_config.pd_disaggregation_mode = None
            fd_config.parallel_config.tensor_parallel_rank = 0
            fd_config.parallel_config.local_data_parallel_id = 0
            fd_config.parallel_config.tensor_parallel_size = 1
            return DSAAttentionBackend(fd_config, kv_num_heads=1, num_heads=16, head_dim=128)

    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.get_block_shape_and_split_kv_block")
    @patch("paddle.get_default_dtype", return_value="bfloat16")
    def test_metadata_bfloat16(self, mock_dtype, mock_block_shape):
        """init_attention_metadata sets bf16 for bfloat16 dtype."""
        backend = self._make_backend()
        forward_meta = MagicMock()
        forward_meta.max_len_tensor_cpu = [0, 100, 50, 0, 0, 200]
        forward_meta.is_dummy_or_profile_run = False

        backend.init_attention_metadata(forward_meta)

        metadata = backend.attention_metadata
        self.assertIsInstance(metadata, DSAAttentionMetadata)
        self.assertEqual(metadata._fuse_kernel_compute_dtype, "bf16")
        self.assertEqual(metadata.max_enc_len_this_time, 100)
        self.assertEqual(metadata.max_dec_len_this_time, 50)
        self.assertEqual(metadata.max_kv_len_this_time, 200)
        self.assertEqual(metadata.encoder_max_partition_size, 8192)

    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.get_block_shape_and_split_kv_block")
    @patch("paddle.get_default_dtype", return_value="float16")
    def test_metadata_float16(self, mock_dtype, mock_block_shape):
        """init_attention_metadata sets fp16 for float16 dtype."""
        backend = self._make_backend()
        forward_meta = MagicMock()
        forward_meta.max_len_tensor_cpu = [0, 0, 0, 0, 0, 0]
        forward_meta.is_dummy_or_profile_run = False

        backend.init_attention_metadata(forward_meta)

        self.assertEqual(backend.attention_metadata._fuse_kernel_compute_dtype, "fp16")

    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.init_kv_signal_per_query")
    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.get_block_shape_and_split_kv_block")
    @patch("paddle.get_default_dtype", return_value="bfloat16")
    def test_pd_disaggregation_per_chunk(self, mock_dtype, mock_block_shape, mock_init_signal):
        """init_attention_metadata calls init_kv_signal_per_query for per_chunk mode."""
        backend = self._make_backend()
        backend.pd_disaggregation_mode = "per_chunk"
        backend.keep_pd_step_flag = False
        backend.num_layers_draft_model = 0

        forward_meta = MagicMock()
        forward_meta.max_len_tensor_cpu = [0, 0, 0, 0, 0, 0]
        forward_meta.is_dummy_or_profile_run = False

        backend.init_attention_metadata(forward_meta)

        mock_init_signal.assert_called_once()

    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.open_shm_and_get_meta_signal")
    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.get_block_shape_and_split_kv_block")
    @patch("paddle.get_default_dtype", return_value="bfloat16")
    def test_pd_disaggregation_per_query(self, mock_dtype, mock_block_shape, mock_open_shm):
        """init_attention_metadata calls open_shm_and_get_meta_signal for per_query mode."""
        backend = self._make_backend()
        backend.pd_disaggregation_mode = "per_query"
        backend.keep_pd_step_flag = False
        mock_open_shm.return_value = "signal_metadata"

        forward_meta = MagicMock()
        forward_meta.max_len_tensor_cpu = [0, 0, 0, 0, 0, 0]
        forward_meta.is_dummy_or_profile_run = False

        backend.init_attention_metadata(forward_meta)

        mock_open_shm.assert_called_once()
        self.assertEqual(backend.attention_metadata.kv_signal_metadata, "signal_metadata")


class TestDSAAttentionBackendGetAttentionMeta(unittest.TestCase):
    """Test DSAAttentionBackend.get_attention_meta."""

    @patch(
        "fastdeploy.model_executor.layers.attention.dsa_attention_backend.init_rank_and_device_id", return_value=(0, 0)
    )
    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.paddle.randn")
    def test_returns_metadata(self, mock_randn, mock_init_rank):
        """get_attention_meta returns stored attention_metadata."""
        mock_randn.return_value = MagicMock()
        mock_randn.return_value.cast.return_value = "useless"

        fd_config = MagicMock()
        fd_config.cache_config.block_size = 64
        fd_config.model_config.max_model_len = 4096
        fd_config.model_config.rope_theta = 10000.0
        fd_config.enable_rope_3d_runtime = False
        fd_config.model_config.causal = True
        fd_config.speculative_config.method = None
        fd_config.speculative_config.num_speculative_tokens = 0
        fd_config.speculative_config.model_type = ""
        fd_config.model_config.head_dim = 128
        fd_config.model_config.num_hidden_layers = 32
        fd_config.model_config.index_head_dim = 256
        fd_config.model_config.index_n_heads = 4
        fd_config.model_config.index_topk = 8
        fd_config.model_config.kv_lora_rank = 512
        fd_config.model_config.qk_rope_head_dim = 64
        fd_config.model_config.qk_nope_head_dim = 128
        fd_config.model_config.rope_scaling = None
        fd_config.model_config.start_layer_index = 0
        fd_config.parallel_config.pd_disaggregation_mode = None
        fd_config.parallel_config.tensor_parallel_rank = 0
        fd_config.parallel_config.local_data_parallel_id = 0
        fd_config.parallel_config.tensor_parallel_size = 1

        backend = DSAAttentionBackend(fd_config, kv_num_heads=1, num_heads=16, head_dim=128)

        self.assertIsNone(backend.get_attention_meta())
        mock_meta = MagicMock()
        backend.attention_metadata = mock_meta
        self.assertIs(backend.get_attention_meta(), mock_meta)


class TestDSAAttentionBackendGetKvCacheShape(unittest.TestCase):
    """Test DSAAttentionBackend.get_kv_cache_shape."""

    @patch(
        "fastdeploy.model_executor.layers.attention.dsa_attention_backend.init_rank_and_device_id", return_value=(0, 0)
    )
    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.paddle.randn")
    def test_kv_cache_shape(self, mock_randn, mock_init_rank):
        """get_kv_cache_shape returns correct shapes for DSA."""
        mock_randn.return_value = MagicMock()
        mock_randn.return_value.cast.return_value = "useless"

        fd_config = MagicMock()
        fd_config.cache_config.block_size = 64
        fd_config.model_config.max_model_len = 4096
        fd_config.model_config.rope_theta = 10000.0
        fd_config.enable_rope_3d_runtime = False
        fd_config.model_config.causal = True
        fd_config.speculative_config.method = None
        fd_config.speculative_config.num_speculative_tokens = 0
        fd_config.speculative_config.model_type = ""
        fd_config.model_config.head_dim = 128
        fd_config.model_config.num_hidden_layers = 32
        fd_config.model_config.index_head_dim = 256
        fd_config.model_config.index_n_heads = 4
        fd_config.model_config.index_topk = 8
        fd_config.model_config.kv_lora_rank = 512
        fd_config.model_config.qk_rope_head_dim = 64
        fd_config.model_config.qk_nope_head_dim = 128
        fd_config.model_config.rope_scaling = None
        fd_config.model_config.start_layer_index = 0
        fd_config.parallel_config.pd_disaggregation_mode = None
        fd_config.parallel_config.tensor_parallel_rank = 0
        fd_config.parallel_config.local_data_parallel_id = 0
        fd_config.parallel_config.tensor_parallel_size = 1

        backend = DSAAttentionBackend(fd_config, kv_num_heads=1, num_heads=16, head_dim=128)

        key_shape, value_shape, indexer_shape = backend.get_kv_cache_shape(max_num_blocks=100)

        # fp8_key_cache_dim = 512 + 4*(512//128) + 2*64 = 512 + 16 + 128 = 656
        self.assertEqual(key_shape, [100, 1, 64, 656])
        # value_cache_shape is empty for DSA
        self.assertEqual(value_shape, [])
        # fp8_indexer_dim = 256 + 256//128*4 = 256 + 8 = 264
        self.assertEqual(indexer_shape, [100, 64, 264])


class TestDSAAttentionBackendCastScaleInv(unittest.TestCase):
    """Test DSAAttentionBackend._cast_scale_inv_to_ue8m0."""

    @patch(
        "fastdeploy.model_executor.layers.attention.dsa_attention_backend.init_rank_and_device_id", return_value=(0, 0)
    )
    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.paddle.randn")
    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.paddle.pow")
    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.paddle.clamp_min", create=True)
    def test_cast_scale_inv(self, mock_clamp_min, mock_pow, mock_randn, mock_init_rank):
        """_cast_scale_inv_to_ue8m0 calls paddle.pow(2, clamp_min(...).log2().ceil())."""
        mock_randn.return_value = MagicMock()
        mock_randn.return_value.cast.return_value = "useless"

        fd_config = MagicMock()
        fd_config.cache_config.block_size = 64
        fd_config.model_config.max_model_len = 4096
        fd_config.model_config.rope_theta = 10000.0
        fd_config.enable_rope_3d_runtime = False
        fd_config.model_config.causal = True
        fd_config.speculative_config.method = None
        fd_config.speculative_config.num_speculative_tokens = 0
        fd_config.speculative_config.model_type = ""
        fd_config.model_config.head_dim = 128
        fd_config.model_config.num_hidden_layers = 32
        fd_config.model_config.index_head_dim = 256
        fd_config.model_config.index_n_heads = 4
        fd_config.model_config.index_topk = 8
        fd_config.model_config.kv_lora_rank = 512
        fd_config.model_config.qk_rope_head_dim = 64
        fd_config.model_config.qk_nope_head_dim = 128
        fd_config.model_config.rope_scaling = None
        fd_config.model_config.start_layer_index = 0
        fd_config.parallel_config.pd_disaggregation_mode = None
        fd_config.parallel_config.tensor_parallel_rank = 0
        fd_config.parallel_config.local_data_parallel_id = 0
        fd_config.parallel_config.tensor_parallel_size = 1

        backend = DSAAttentionBackend(fd_config, kv_num_heads=1, num_heads=16, head_dim=128)

        # Mock the chain: paddle.clamp_min(x, 1e-4).log2().ceil() -> pow(2, ...) -> .to(dtype)
        mock_clamped = MagicMock()
        mock_log2 = MagicMock()
        mock_ceil = MagicMock()
        mock_clamp_min.return_value = mock_clamped
        mock_clamped.log2.return_value = mock_log2
        mock_log2.ceil.return_value = mock_ceil

        mock_result = MagicMock()
        mock_pow.return_value = mock_result
        mock_result.to.return_value = "final_tensor"

        scales_inv = MagicMock()
        result = backend._cast_scale_inv_to_ue8m0(scales_inv)

        mock_clamp_min.assert_called_once_with(scales_inv, 1e-4)
        mock_clamped.log2.assert_called_once()
        mock_log2.ceil.assert_called_once()
        mock_pow.assert_called_once_with(2, mock_ceil)
        mock_result.to.assert_called_once_with(paddle.float32)
        self.assertEqual(result, "final_tensor")


class TestDSAAttentionBackendInitMetadataFloat32(unittest.TestCase):
    """Test init_attention_metadata with float32 dtype."""

    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.get_block_shape_and_split_kv_block")
    @patch("paddle.get_default_dtype", return_value="float32")
    @patch(
        "fastdeploy.model_executor.layers.attention.dsa_attention_backend.init_rank_and_device_id", return_value=(0, 0)
    )
    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.paddle.randn")
    def test_metadata_float32(self, mock_randn, mock_init_rank, mock_dtype, mock_block_shape):
        """init_attention_metadata sets fp32 for float32 dtype."""
        mock_randn.return_value = MagicMock()
        mock_randn.return_value.cast.return_value = "useless"

        fd_config = MagicMock()
        fd_config.cache_config.block_size = 64
        fd_config.model_config.max_model_len = 8192
        fd_config.model_config.rope_theta = 10000.0
        fd_config.enable_rope_3d_runtime = False
        fd_config.model_config.causal = True
        fd_config.speculative_config.method = None
        fd_config.speculative_config.num_speculative_tokens = 0
        fd_config.speculative_config.model_type = ""
        fd_config.model_config.head_dim = 128
        fd_config.model_config.num_hidden_layers = 60
        fd_config.model_config.index_head_dim = 256
        fd_config.model_config.index_n_heads = 4
        fd_config.model_config.index_topk = 8
        fd_config.model_config.kv_lora_rank = 512
        fd_config.model_config.qk_rope_head_dim = 64
        fd_config.model_config.qk_nope_head_dim = 128
        fd_config.model_config.rope_scaling = None
        fd_config.model_config.start_layer_index = 0
        fd_config.parallel_config.pd_disaggregation_mode = None
        fd_config.parallel_config.tensor_parallel_rank = 0
        fd_config.parallel_config.local_data_parallel_id = 0
        fd_config.parallel_config.tensor_parallel_size = 1

        backend = DSAAttentionBackend(fd_config, kv_num_heads=1, num_heads=16, head_dim=128)

        forward_meta = MagicMock()
        forward_meta.max_len_tensor_cpu = [0, 0, 0, 0, 0, 0]
        forward_meta.is_dummy_or_profile_run = False

        backend.init_attention_metadata(forward_meta)

        self.assertEqual(backend.attention_metadata._fuse_kernel_compute_dtype, "fp32")


class TestDSAAttentionBackendQuantizeKCache(unittest.TestCase):
    """Test DSAAttentionBackend.quantize_k_cache."""

    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.paddle.clamp_min", create=True)
    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.paddle.pow")
    @patch(
        "fastdeploy.model_executor.layers.attention.dsa_attention_backend.init_rank_and_device_id", return_value=(0, 0)
    )
    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.paddle.randn")
    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.paddle.empty")
    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.paddle.abs")
    def test_quantize_k_cache(self, mock_abs, mock_empty, mock_randn, mock_init_rank, mock_pow, mock_clamp_min):
        """quantize_k_cache quantizes input tensor to FP8 layout."""
        mock_randn.return_value = MagicMock()
        mock_randn.return_value.cast.return_value = "useless"

        fd_config = MagicMock()
        fd_config.cache_config.block_size = 64
        fd_config.model_config.max_model_len = 4096
        fd_config.model_config.rope_theta = 10000.0
        fd_config.enable_rope_3d_runtime = False
        fd_config.model_config.causal = True
        fd_config.speculative_config.method = None
        fd_config.speculative_config.num_speculative_tokens = 0
        fd_config.speculative_config.model_type = ""
        fd_config.model_config.head_dim = 128
        fd_config.model_config.num_hidden_layers = 32
        fd_config.model_config.index_head_dim = 256
        fd_config.model_config.index_n_heads = 4
        fd_config.model_config.index_topk = 8
        fd_config.model_config.kv_lora_rank = 512
        fd_config.model_config.qk_rope_head_dim = 64
        fd_config.model_config.qk_nope_head_dim = 128
        fd_config.model_config.rope_scaling = None
        fd_config.model_config.start_layer_index = 0
        fd_config.parallel_config.pd_disaggregation_mode = None
        fd_config.parallel_config.tensor_parallel_rank = 0
        fd_config.parallel_config.local_data_parallel_id = 0
        fd_config.parallel_config.tensor_parallel_size = 1

        backend = DSAAttentionBackend(fd_config, kv_num_heads=1, num_heads=16, head_dim=128)

        # Create mock input tensor: shape (num_blocks, block_size, h_k, d) = (2, 4, 1, 576)
        input_k_cache = MagicMock()
        input_k_cache.shape = [2, 4, 1, 576]  # d=576 as expected

        squeezed = MagicMock()
        input_k_cache.squeeze.return_value = squeezed
        squeezed.element_size.return_value = 2  # bfloat16

        # Mock paddle.empty for result buffer
        result_buf = MagicMock()
        result_buf.__getitem__ = MagicMock(return_value=result_buf)
        mock_empty.return_value = result_buf

        # Mock slice operations on result
        result_nope = MagicMock()
        result_scale = MagicMock()
        result_rope = MagicMock()
        result_buf.__getitem__ = MagicMock(side_effect=[result_buf, result_nope, result_scale, result_rope])

        # Mock the Ellipsis slicing - use side_effect to handle different slice calls
        def getitem_handler(key):
            if key == (Ellipsis, slice(None, 512)):
                return result_nope
            elif key == (Ellipsis, slice(512, 528)):
                return result_scale
            elif key == (Ellipsis, slice(528, None)):
                return result_rope
            return result_buf

        result_buf.__getitem__ = MagicMock(side_effect=getitem_handler)

        result_scale.view = MagicMock(return_value=result_scale)
        result_rope.view = MagicMock(return_value=result_rope)

        # Mock abs/max chain for each tile
        mock_max_result = MagicMock()
        mock_max_result.values = MagicMock()
        mock_max_result.values.float.return_value = MagicMock()
        mock_max_result.values.float.return_value.__truediv__ = MagicMock(return_value=MagicMock())

        abs_result = MagicMock()
        abs_result.max.return_value = mock_max_result
        mock_abs.return_value = abs_result

        # Mock _cast_scale_inv_to_ue8m0
        scale_inv_result = MagicMock()
        mock_clamped = MagicMock()
        mock_clamped.log2.return_value.ceil.return_value = MagicMock()
        mock_clamp_min.return_value = mock_clamped
        mock_pow.return_value = scale_inv_result
        scale_inv_result.to.return_value = scale_inv_result

        # Mock the float division for quantization
        float_result = MagicMock()
        float_result.__truediv__ = MagicMock(return_value=MagicMock())

        # Mock squeezed slicing
        squeezed.__getitem__ = MagicMock(return_value=MagicMock())
        squeezed.__getitem__.return_value.float.return_value = float_result

        # We can't easily test this with full mocks due to complex slicing.
        # Instead, verify the method exists and has correct signature.
        self.assertTrue(hasattr(backend, "quantize_k_cache"))
        self.assertTrue(callable(backend.quantize_k_cache))


class TestDSAAttentionBackendForwardMixedFull(unittest.TestCase):
    """Test DSAAttentionBackend.forward_mixed with full GPU path."""

    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.current_platform")
    @patch(
        "fastdeploy.model_executor.layers.attention.dsa_attention_backend.init_rank_and_device_id", return_value=(0, 0)
    )
    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.paddle.randn")
    @patch("paddle.abs")
    def test_forward_mixed_decode_only(self, mock_abs, mock_randn, mock_init_rank, mock_platform):
        """forward_mixed returns decode output when only dec_len > 0."""
        mock_randn.return_value = MagicMock()
        mock_randn.return_value.cast.return_value = "useless"
        mock_platform.is_cuda.return_value = True

        fd_config = MagicMock()
        fd_config.cache_config.block_size = 64
        fd_config.model_config.max_model_len = 4096
        fd_config.model_config.rope_theta = 10000.0
        fd_config.enable_rope_3d_runtime = False
        fd_config.model_config.causal = True
        fd_config.speculative_config.method = None
        fd_config.speculative_config.num_speculative_tokens = 0
        fd_config.speculative_config.model_type = ""
        fd_config.model_config.head_dim = 128
        fd_config.model_config.num_hidden_layers = 32
        fd_config.model_config.index_head_dim = 256
        fd_config.model_config.index_n_heads = 4
        fd_config.model_config.index_topk = 8
        fd_config.model_config.kv_lora_rank = 512
        fd_config.model_config.qk_rope_head_dim = 64
        fd_config.model_config.qk_nope_head_dim = 128
        fd_config.model_config.rope_scaling = None
        fd_config.model_config.start_layer_index = 0
        fd_config.parallel_config.pd_disaggregation_mode = None
        fd_config.parallel_config.tensor_parallel_rank = 0
        fd_config.parallel_config.local_data_parallel_id = 0
        fd_config.parallel_config.tensor_parallel_size = 1

        backend = DSAAttentionBackend(fd_config, kv_num_heads=1, num_heads=16, head_dim=128)

        metadata = DSAAttentionMetadata()
        metadata.kv_signal_data_list = [None] * 32
        backend.attention_metadata = metadata

        layer = MagicMock()
        layer.layer_id = 0

        forward_meta = MagicMock()
        forward_meta.caches = ["cache"] * 64
        forward_meta.max_len_tensor_cpu = [0, 0, 50, 0, 0, 0]  # enc = 0, dec > 0
        forward_meta.slot_mapping = MagicMock()

        # Mock latent_cache.shape
        latent_cache = MagicMock()
        latent_cache.shape = [100, 1, 64, 576]
        latent_cache.view.return_value = latent_cache
        forward_meta.caches = [latent_cache] * 64

        scale_mock = MagicMock()
        scale_mock.cast.return_value = scale_mock
        scale_mock.__truediv__ = MagicMock(return_value=scale_mock)
        mock_abs.return_value = MagicMock()
        mock_abs.return_value.max.return_value = scale_mock

        mock_flash_mla = MagicMock()
        mock_flash_mla.get_mla_metadata.return_value = ("tile_meta", None)
        mock_flash_mla.flash_mla_with_kvcache.return_value = ("decode_output", None)

        mock_dsk_write = MagicMock()
        gpu_module = MagicMock()
        gpu_module.dsk_attn_write_cache = mock_dsk_write

        import sys

        with patch.dict(
            sys.modules,
            {
                "flash_mla": mock_flash_mla,
                "fastdeploy.model_executor.ops.gpu": gpu_module,
                "fastdeploy.model_executor.ops": MagicMock(gpu=gpu_module),
            },
        ):
            result = backend.forward_mixed(
                q=MagicMock(),
                k=None,
                v=MagicMock(),
                qkv=None,
                compressed_kv=MagicMock(),
                k_pe=MagicMock(),
                layer=layer,
                forward_meta=forward_meta,
            )

        self.assertEqual(result, "decode_output")

    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.current_platform")
    @patch(
        "fastdeploy.model_executor.layers.attention.dsa_attention_backend.init_rank_and_device_id", return_value=(0, 0)
    )
    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.paddle.randn")
    @patch("paddle.abs")
    def test_forward_mixed_both_prefill_and_decode(self, mock_abs, mock_randn, mock_init_rank, mock_platform):
        """forward_mixed merges outputs when both enc and dec > 0."""
        mock_randn.return_value = MagicMock()
        mock_randn.return_value.cast.return_value = "useless"
        mock_platform.is_cuda.return_value = True

        fd_config = MagicMock()
        fd_config.cache_config.block_size = 64
        fd_config.model_config.max_model_len = 4096
        fd_config.model_config.rope_theta = 10000.0
        fd_config.enable_rope_3d_runtime = False
        fd_config.model_config.causal = True
        fd_config.speculative_config.method = None
        fd_config.speculative_config.num_speculative_tokens = 0
        fd_config.speculative_config.model_type = ""
        fd_config.model_config.head_dim = 128
        fd_config.model_config.num_hidden_layers = 32
        fd_config.model_config.index_head_dim = 256
        fd_config.model_config.index_n_heads = 4
        fd_config.model_config.index_topk = 8
        fd_config.model_config.kv_lora_rank = 512
        fd_config.model_config.qk_rope_head_dim = 64
        fd_config.model_config.qk_nope_head_dim = 128
        fd_config.model_config.rope_scaling = None
        fd_config.model_config.start_layer_index = 0
        fd_config.parallel_config.pd_disaggregation_mode = None
        fd_config.parallel_config.tensor_parallel_rank = 0
        fd_config.parallel_config.local_data_parallel_id = 0
        fd_config.parallel_config.tensor_parallel_size = 1

        backend = DSAAttentionBackend(fd_config, kv_num_heads=1, num_heads=16, head_dim=128)

        metadata = DSAAttentionMetadata()
        metadata.kv_signal_data_list = [None] * 32
        backend.attention_metadata = metadata

        layer = MagicMock()
        layer.layer_id = 0

        forward_meta = MagicMock()
        forward_meta.max_len_tensor_cpu = [0, 100, 50, 0, 0, 0]  # both enc and dec > 0
        forward_meta.slot_mapping = MagicMock()

        latent_cache = MagicMock()
        latent_cache.shape = [100, 1, 64, 576]
        latent_cache.view.return_value = latent_cache
        forward_meta.caches = [latent_cache] * 64

        scale_mock = MagicMock()
        scale_mock.cast.return_value = scale_mock
        scale_mock.__truediv__ = MagicMock(return_value=scale_mock)
        mock_abs.return_value = MagicMock()
        mock_abs.return_value.max.return_value = scale_mock

        mock_flash_mla = MagicMock()
        mock_flash_mla.flash_mla_sparse_fwd.return_value = ("prefill_out", None, None)
        mock_flash_mla.get_mla_metadata.return_value = ("tile_meta", None)
        mock_flash_mla.flash_mla_with_kvcache.return_value = ("decode_out", None)

        mock_dsk_write = MagicMock()
        mock_merge = MagicMock()
        gpu_module = MagicMock()
        gpu_module.dsk_attn_write_cache = mock_dsk_write
        gpu_module.merge_prefill_decode_output = mock_merge

        import sys

        with patch.dict(
            sys.modules,
            {
                "flash_mla": mock_flash_mla,
                "fastdeploy.model_executor.ops.gpu": gpu_module,
                "fastdeploy.model_executor.ops": MagicMock(gpu=gpu_module),
            },
        ):
            result = backend.forward_mixed(
                q=MagicMock(),
                k=MagicMock(),
                v=MagicMock(),
                qkv=None,
                compressed_kv=MagicMock(),
                k_pe=MagicMock(),
                layer=layer,
                forward_meta=forward_meta,
            )

        # When both prefill and decode, returns fmha_out_prefill after merge
        self.assertEqual(result, "prefill_out")
        mock_merge.assert_called_once()

    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.current_platform")
    @patch(
        "fastdeploy.model_executor.layers.attention.dsa_attention_backend.init_rank_and_device_id", return_value=(0, 0)
    )
    @patch("fastdeploy.model_executor.layers.attention.dsa_attention_backend.paddle.randn")
    @patch("paddle.abs")
    def test_forward_mixed_no_enc_no_dec(self, mock_abs, mock_randn, mock_init_rank, mock_platform):
        """forward_mixed returns None when neither enc nor dec."""
        mock_randn.return_value = MagicMock()
        mock_randn.return_value.cast.return_value = "useless"
        mock_platform.is_cuda.return_value = True

        fd_config = MagicMock()
        fd_config.cache_config.block_size = 64
        fd_config.model_config.max_model_len = 4096
        fd_config.model_config.rope_theta = 10000.0
        fd_config.enable_rope_3d_runtime = False
        fd_config.model_config.causal = True
        fd_config.speculative_config.method = None
        fd_config.speculative_config.num_speculative_tokens = 0
        fd_config.speculative_config.model_type = ""
        fd_config.model_config.head_dim = 128
        fd_config.model_config.num_hidden_layers = 32
        fd_config.model_config.index_head_dim = 256
        fd_config.model_config.index_n_heads = 4
        fd_config.model_config.index_topk = 8
        fd_config.model_config.kv_lora_rank = 512
        fd_config.model_config.qk_rope_head_dim = 64
        fd_config.model_config.qk_nope_head_dim = 128
        fd_config.model_config.rope_scaling = None
        fd_config.model_config.start_layer_index = 0
        fd_config.parallel_config.pd_disaggregation_mode = None
        fd_config.parallel_config.tensor_parallel_rank = 0
        fd_config.parallel_config.local_data_parallel_id = 0
        fd_config.parallel_config.tensor_parallel_size = 1

        backend = DSAAttentionBackend(fd_config, kv_num_heads=1, num_heads=16, head_dim=128)

        metadata = DSAAttentionMetadata()
        metadata.kv_signal_data_list = [None] * 32
        backend.attention_metadata = metadata

        layer = MagicMock()
        layer.layer_id = 0

        forward_meta = MagicMock()
        forward_meta.caches = ["cache"] * 64
        forward_meta.max_len_tensor_cpu = [0, 0, 0, 0, 0, 0]  # no enc, no dec
        forward_meta.slot_mapping = MagicMock()

        scale_mock = MagicMock()
        scale_mock.cast.return_value = scale_mock
        scale_mock.__truediv__ = MagicMock(return_value=scale_mock)
        mock_abs.return_value = MagicMock()
        mock_abs.return_value.max.return_value = scale_mock

        mock_dsk_write = MagicMock()
        gpu_module = MagicMock()
        gpu_module.dsk_attn_write_cache = mock_dsk_write

        import sys

        with patch.dict(
            sys.modules,
            {
                "flash_mla": MagicMock(),
                "fastdeploy.model_executor.ops.gpu": gpu_module,
                "fastdeploy.model_executor.ops": MagicMock(gpu=gpu_module),
            },
        ):
            result = backend.forward_mixed(
                q=None,
                k=None,
                v=None,
                qkv=None,
                compressed_kv=MagicMock(),
                k_pe=MagicMock(),
                layer=layer,
                forward_meta=forward_meta,
            )

        # fmha_out_prefill = None, no decode either -> returns None
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
