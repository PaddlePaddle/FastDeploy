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

from fastdeploy.model_executor.layers.attention.block_multihead_attn_backend import (
    BlockAttentionBackend,
    BlockAttentionMetadata,
)


class TestBlockAttentionMetadata(unittest.TestCase):
    """Test BlockAttentionMetadata dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        metadata = BlockAttentionMetadata()
        self.assertIsNone(metadata.encoder_batch_ids)
        self.assertIsNone(metadata.encoder_tile_ids_per_batch)
        self.assertIsNone(metadata.encoder_num_blocks)
        self.assertIsNone(metadata.kv_batch_ids)
        self.assertIsNone(metadata.kv_tile_ids_per_batch)
        self.assertIsNone(metadata.kv_num_blocks)
        self.assertEqual(metadata._dtype, paddle.bfloat16)
        self.assertEqual(metadata.encoder_max_partition_size, 32768)
        self.assertEqual(metadata.max_partition_size, 32768)
        self.assertIsNone(metadata.block_tables)
        self.assertIsNone(metadata.rotary_embs)
        self.assertIsNone(metadata.attn_mask)
        self.assertEqual(metadata._fuse_kernel_compute_dtype, "bf16")
        self.assertIsNone(metadata.kv_signal_metadata)
        self.assertEqual(metadata.kv_signal_data_list, [])


class TestBlockAttentionBackendInit(unittest.TestCase):
    """Test BlockAttentionBackend.__init__."""

    def _make_fd_config(self, block_size=64, max_model_len=4096, rope_theta=None, head_dim=128):
        """Create a mock FDConfig."""
        fd_config = MagicMock()
        fd_config.cache_config.block_size = block_size
        fd_config.model_config.max_model_len = max_model_len
        fd_config.model_config.rope_theta = rope_theta
        fd_config.model_config.head_dim = head_dim
        fd_config.parallel_config.tensor_parallel_rank = 0
        return fd_config

    def test_init_default_rope_theta(self):
        """Init with rope_theta=None defaults to 10000.0."""
        fd_config = self._make_fd_config(rope_theta=None)
        backend = BlockAttentionBackend(fd_config, kv_num_heads=8, num_heads=32, head_dim=128)

        self.assertIsNone(backend.attention_metadata)
        self.assertEqual(backend.block_size, 64)
        self.assertEqual(backend.max_seq_len, 4096)
        self.assertEqual(backend.rope_theta, 10000.0)
        self.assertEqual(backend.rank, 0)
        self.assertEqual(backend.kv_num_heads, 8)
        self.assertEqual(backend.num_heads, 32)
        self.assertEqual(backend.head_dim, 128)

    def test_init_custom_rope_theta(self):
        """Init with custom rope_theta value."""
        fd_config = self._make_fd_config(rope_theta=500000.0)
        backend = BlockAttentionBackend(fd_config, kv_num_heads=4, num_heads=16, head_dim=64)

        self.assertEqual(backend.rope_theta, 500000.0)

    def test_init_stores_config_values(self):
        """Init stores all config values correctly."""
        fd_config = self._make_fd_config(block_size=128, max_model_len=8192, head_dim=256)
        fd_config.parallel_config.tensor_parallel_rank = 3

        backend = BlockAttentionBackend(fd_config, kv_num_heads=2, num_heads=8, head_dim=256)

        self.assertEqual(backend.block_size, 128)
        self.assertEqual(backend.max_seq_len, 8192)
        self.assertEqual(backend.rank, 3)
        self.assertEqual(backend.head_dim, 256)


class TestBlockAttentionBackendInitAttentionMetadata(unittest.TestCase):
    """Test BlockAttentionBackend.init_attention_metadata."""

    def _make_backend(self):
        """Create a BlockAttentionBackend with mock config."""
        fd_config = MagicMock()
        fd_config.cache_config.block_size = 64
        fd_config.model_config.max_model_len = 4096
        fd_config.model_config.rope_theta = 10000.0
        fd_config.model_config.head_dim = 128
        fd_config.parallel_config.tensor_parallel_rank = 0
        return BlockAttentionBackend(fd_config, kv_num_heads=8, num_heads=32, head_dim=128)

    @patch("paddle.get_default_dtype", return_value="bfloat16")
    def test_bfloat16_dtype(self, mock_dtype):
        """Sets bf16 compute dtype for bfloat16."""
        backend = self._make_backend()
        forward_meta = MagicMock()
        forward_meta.block_tables = "mock_block_tables"
        forward_meta.rotary_embs = "mock_rotary_embs"
        forward_meta.attn_mask = "mock_attn_mask"

        backend.init_attention_metadata(forward_meta)

        metadata = backend.attention_metadata
        self.assertIsInstance(metadata, BlockAttentionMetadata)
        self.assertEqual(metadata._dtype, "bfloat16")
        self.assertEqual(metadata._fuse_kernel_compute_dtype, "bf16")
        self.assertEqual(metadata.block_tables, "mock_block_tables")
        self.assertEqual(metadata.rotary_embs, "mock_rotary_embs")
        self.assertEqual(metadata.attn_mask, "mock_attn_mask")

    @patch("paddle.get_default_dtype", return_value="float16")
    def test_float16_dtype(self, mock_dtype):
        """Sets fp16 compute dtype for float16."""
        backend = self._make_backend()
        forward_meta = MagicMock()

        backend.init_attention_metadata(forward_meta)

        self.assertEqual(backend.attention_metadata._fuse_kernel_compute_dtype, "fp16")

    @patch("paddle.get_default_dtype", return_value="float32")
    def test_float32_dtype(self, mock_dtype):
        """Sets fp32 compute dtype for float32."""
        backend = self._make_backend()
        forward_meta = MagicMock()

        backend.init_attention_metadata(forward_meta)

        self.assertEqual(backend.attention_metadata._fuse_kernel_compute_dtype, "fp32")

    @patch("paddle.get_default_dtype", return_value="float64")
    def test_unknown_dtype_keeps_default(self, mock_dtype):
        """Unknown dtype keeps default bf16 compute dtype."""
        backend = self._make_backend()
        forward_meta = MagicMock()

        backend.init_attention_metadata(forward_meta)

        # Default from dataclass is "bf16"
        self.assertEqual(backend.attention_metadata._fuse_kernel_compute_dtype, "bf16")


class TestBlockAttentionBackendGetAttentionMeta(unittest.TestCase):
    """Test BlockAttentionBackend.get_attention_meta."""

    def test_returns_attention_metadata(self):
        """get_attention_meta returns the stored attention_metadata."""
        fd_config = MagicMock()
        fd_config.cache_config.block_size = 64
        fd_config.model_config.max_model_len = 4096
        fd_config.model_config.rope_theta = 10000.0
        fd_config.model_config.head_dim = 128
        fd_config.parallel_config.tensor_parallel_rank = 0

        backend = BlockAttentionBackend(fd_config, kv_num_heads=8, num_heads=32, head_dim=128)

        self.assertIsNone(backend.get_attention_meta())

        mock_metadata = MagicMock()
        backend.attention_metadata = mock_metadata
        self.assertIs(backend.get_attention_meta(), mock_metadata)


class TestBlockAttentionBackendGetKvCacheShape(unittest.TestCase):
    """Test BlockAttentionBackend.get_kv_cache_shape."""

    def _make_backend(self, kv_num_heads=8, block_size=64, head_dim=128):
        """Create a BlockAttentionBackend."""
        fd_config = MagicMock()
        fd_config.cache_config.block_size = block_size
        fd_config.model_config.max_model_len = 4096
        fd_config.model_config.rope_theta = 10000.0
        fd_config.model_config.head_dim = head_dim
        fd_config.parallel_config.tensor_parallel_rank = 0
        return BlockAttentionBackend(fd_config, kv_num_heads=kv_num_heads, num_heads=32, head_dim=head_dim)

    def test_default_no_quant(self):
        """No quantization returns full head_dim shape."""
        backend = self._make_backend(kv_num_heads=8, block_size=64, head_dim=128)

        key_shape, value_shape = backend.get_kv_cache_shape(max_num_blocks=100)

        self.assertEqual(key_shape, [100, 8, 64, 128])
        self.assertEqual(value_shape, [100, 8, 64, 128])

    def test_none_quant_type(self):
        """None quant type returns full head_dim shape."""
        backend = self._make_backend(kv_num_heads=4, block_size=32, head_dim=64)

        key_shape, value_shape = backend.get_kv_cache_shape(max_num_blocks=50, kv_cache_quant_type=None)

        self.assertEqual(key_shape, [50, 4, 32, 64])
        self.assertEqual(value_shape, [50, 4, 32, 64])

    def test_int4_zp_quant(self):
        """int4_zp quantization halves head_dim."""
        backend = self._make_backend(kv_num_heads=8, block_size=64, head_dim=128)

        key_shape, value_shape = backend.get_kv_cache_shape(max_num_blocks=200, kv_cache_quant_type="int4_zp")

        self.assertEqual(key_shape, [200, 8, 64, 64])
        self.assertEqual(value_shape, [200, 8, 64, 64])

    def test_other_quant_type_no_effect(self):
        """Non-int4_zp quant type does not halve head_dim."""
        backend = self._make_backend(kv_num_heads=8, block_size=64, head_dim=128)

        key_shape, value_shape = backend.get_kv_cache_shape(max_num_blocks=100, kv_cache_quant_type="fp8")

        self.assertEqual(key_shape, [100, 8, 64, 128])
        self.assertEqual(value_shape, [100, 8, 64, 128])


class TestBlockAttentionBackendForwardMixed(unittest.TestCase):
    """Test BlockAttentionBackend.forward_mixed."""

    @patch("paddle.incubate.nn.functional.block_multihead_attention")
    def test_forward_mixed_calls_kernel(self, mock_bma):
        """forward_mixed calls block_multihead_attention with correct args."""
        fd_config = MagicMock()
        fd_config.cache_config.block_size = 64
        fd_config.model_config.max_model_len = 4096
        fd_config.model_config.rope_theta = 10000.0
        fd_config.model_config.head_dim = 128
        fd_config.parallel_config.tensor_parallel_rank = 0

        backend = BlockAttentionBackend(fd_config, kv_num_heads=8, num_heads=32, head_dim=128)

        # Set up attention metadata
        metadata = BlockAttentionMetadata()
        metadata.block_tables = "block_tables_tensor"
        metadata.rotary_embs = "rotary_embs_tensor"
        metadata.attn_mask = "attn_mask_tensor"
        metadata._fuse_kernel_compute_dtype = "bf16"
        backend.attention_metadata = metadata

        # Set up forward_meta
        forward_meta = MagicMock()
        forward_meta.caches = ["cache_0", "cache_1", "cache_2", "cache_3"]
        forward_meta.seq_lens_encoder = "seq_lens_encoder"
        forward_meta.seq_lens_decoder = "seq_lens_decoder"
        forward_meta.seq_lens_this_time = "seq_lens_this_time"
        forward_meta.batch_id_per_token = "batch_id_per_token"
        forward_meta.cum_offsets = "cum_offsets"
        forward_meta.cu_seqlens_q = "cu_seqlens_q"
        forward_meta.cu_seqlens_k = "cu_seqlens_k"

        # Set up layer
        layer = MagicMock()
        layer.layer_id = 1
        layer.qkv_scale = 0.125
        layer.qkv_bias = None
        layer.linear_shift = None
        layer.linear_smooth = None
        layer.use_neox_rotary_style = True

        mock_bma.return_value = ("output_tensor",)

        result = backend.forward_mixed(
            q=None,
            k=None,
            v=None,
            qkv="qkv_tensor",
            compressed_kv=None,
            k_pe=None,
            layer=layer,
            forward_meta=forward_meta,
        )

        self.assertEqual(result, "output_tensor")
        mock_bma.assert_called_once()

        # Verify key arguments
        call_args = mock_bma.call_args
        self.assertEqual(call_args[0][0], "qkv_tensor")  # qkv
        self.assertEqual(call_args[0][1], "cache_2")  # caches[2*layer_id]
        self.assertEqual(call_args[0][2], "cache_3")  # caches[2*layer_id+1]
        self.assertEqual(call_args[1]["compute_dtype"], "bf16")
        self.assertEqual(call_args[1]["rope_theta"], 10000.0)


if __name__ == "__main__":
    unittest.main()
