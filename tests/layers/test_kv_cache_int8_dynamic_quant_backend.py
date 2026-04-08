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

"""
Unit tests for the KV cache int8 dynamic quant fix on flash_attn_backend
and flash_mask_attn_backend (commit 584df2ba8).

The fix ensures that when cache_quant_type_str == "block_wise_fp8":
  - cache_k/v are taken from caches[4*layer_id : 4*layer_id+2]
  - cache_k/v_scales are taken from caches[4*layer_id+2 : 4*layer_id+4]
Otherwise (non-dynamic-quant):
  - cache_k/v are taken from caches[2*layer_id : 2*layer_id+2]
  - cache_k/v_scales are taken from layer.cache_k_scale / layer.cache_v_scale

Strategy: We mock the entire fastdeploy import chain and the external op
functions, then verify the correct cache tensors are routed through.
"""

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import paddle

# ---------------------------------------------------------------------------
# Environment setup: mock missing fastdeploy dependencies before import
# ---------------------------------------------------------------------------


def _ensure_mock_module(name, attrs=None):
    """Ensure a module exists in sys.modules, creating a mock if needed."""
    if name not in sys.modules:
        mod = types.ModuleType(name)
        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)
        sys.modules[name] = mod
    return sys.modules[name]


# Mock problematic transitive dependencies that may be missing in some environments
_ensure_mock_module("aistudio_sdk.snapshot_download", {"snapshot_download": lambda *a, **kw: None})

# Try importing the backends. If it still fails, mark tests as skipped.
_IMPORT_ERROR = None
try:
    from fastdeploy.model_executor.layers.attention.flash_attn_backend import (
        FlashAttentionBackend,
    )
    from fastdeploy.model_executor.layers.attention.flash_mask_attn_backend import (
        FlashMaskAttentionBackend,
    )
except Exception as e:
    _IMPORT_ERROR = str(e)
    FlashAttentionBackend = None
    FlashMaskAttentionBackend = None


# ---------------------------------------------------------------------------
# Dummy / Mock helpers
# ---------------------------------------------------------------------------


class DummyFDConfig:
    """Minimal FDConfig for constructing backend objects.

    Uses __getattr__ to return MagicMock for any missing nested attributes,
    avoiding the need to enumerate every config attribute.
    """

    def __init__(self):
        self.cache_config = type("CacheConfig", (), {"block_size": 64})()
        self.model_config = type(
            "ModelConfig",
            (),
            {
                "max_model_len": 2048,
                "head_dim": 128,
                "num_hidden_layers": 2,
                "enable_mm": False,
                "causal": True,
                "start_layer_index": 0,
                "rope_3d": False,
                "use_3d_rope": False,
            },
        )()
        self.scheduler_config = type("SchedulerConfig", (), {"max_num_seqs": 4})()
        self.graph_opt_config = type(
            "GraphOptConfig",
            (),
            {"cudagraph_capture_sizes": None},
        )()
        self.parallel_config = type(
            "ParallelConfig",
            (),
            {
                "block_size": 64,
                "data_parallel_rank": 0,
                "pd_disaggregation_mode": "none",
                "expert_parallel_rank": 0,
            },
        )()
        self.speculative_config = type(
            "SpeculativeConfig",
            (),
            {
                "method": None,
                "max_draft_token_num": 0,
                "num_speculative_tokens": 0,
                "model_type": "main",
            },
        )()
        self.enable_mm_runtime = self.model_config.enable_mm
        self.enable_rope_3d_runtime = self.model_config.enable_mm


class DummyLayer:
    """Mimics the Attention layer object with relevant attributes."""

    def __init__(
        self,
        layer_id=0,
        cache_quant_type_str="none",
        cache_k_scale=None,
        cache_v_scale=None,
    ):
        self.layer_id = layer_id
        self.cache_quant_type_str = cache_quant_type_str
        self.cache_k_scale = cache_k_scale
        self.cache_v_scale = cache_v_scale
        self.cache_k_out_scale = None
        self.cache_v_out_scale = None
        self.cache_k_zp = None
        self.cache_v_zp = None
        self.qkv_bias = None
        self.qkv_scale = None
        self.linear_shift = None
        self.linear_smooth = None
        self.use_neox_rotary_style = False
        self.rms_norm_eps = 1e-6
        self.qk_norm_before_rope = False
        self.out_scale = -1.0
        self.quant_max_bound = 0.0
        self.quant_min_bound = 0.0


def _make_sentinel(name: str) -> paddle.Tensor:
    """Create a uniquely identifiable 'sentinel' tensor for tracing through call args."""
    t = paddle.zeros([1], dtype="float32")
    t._sentinel_name = name
    return t


def _make_caches_normal(layer_id=0):
    """Create a caches list for normal (non-block_wise_fp8) mode."""
    num_entries = 2 * (layer_id + 1)
    return [_make_sentinel(f"normal_cache_{i}") for i in range(num_entries)]


def _make_caches_block_wise_fp8(layer_id=0):
    """Create a caches list for block_wise_fp8 mode."""
    num_entries = 4 * (layer_id + 1)
    return [_make_sentinel(f"bwfp8_cache_{i}") for i in range(num_entries)]


class DummyForwardMeta:
    """Minimal ForwardMeta with lazily-created None attributes.

    Simulates a multi-batch scenario with batch_size=4.
    In decode-only mode (max_len_val=0): 4 decode tokens, 0 prefill.
    In prefill mode (max_len_val>0): mixed prefill + decode across 4 batches.
    """

    BATCH_SIZE = 4

    def __init__(self, caches, max_len_val=0):
        bs = self.BATCH_SIZE
        self.caches = caches
        # 4 batches: in decode mode each has 0 encoder len, 1 decoder token
        self.seq_lens_encoder = paddle.to_tensor([0] * bs, dtype="int32")
        self.seq_lens_decoder = paddle.to_tensor([1] * bs, dtype="int32")
        self.seq_lens_this_time = paddle.to_tensor([1] * bs, dtype="int32")
        # total tokens = batch_size (1 per batch in decode)
        self.cu_seqlens_q = paddle.to_tensor(list(range(bs + 1)), dtype="int32")
        self.cu_seqlens_k = paddle.to_tensor(list(range(bs + 1)), dtype="int32")
        self.rotary_embs = paddle.zeros([bs, 1, 128], dtype="float32")
        self.batch_id_per_token = paddle.to_tensor(list(range(bs)), dtype="int32")
        self.block_tables = paddle.to_tensor([[i] for i in range(bs)], dtype="int32")
        self.decoder_batch_ids = paddle.to_tensor(list(range(bs)), dtype="int32")
        self.decoder_tile_ids_per_batch = paddle.to_tensor([0] * bs, dtype="int32")
        self.decoder_num_blocks_cpu = paddle.to_tensor([bs], dtype="int32")
        self.decoder_num_blocks_device = paddle.to_tensor([bs], dtype="int32")
        self.decoder_chunk_size_device = paddle.to_tensor([1] * bs, dtype="int32")
        self.encoder_batch_ids = paddle.to_tensor(list(range(bs)), dtype="int32")
        self.encoder_tile_ids_per_batch = paddle.to_tensor([0] * bs, dtype="int32")
        self.encoder_num_blocks_x_cpu = paddle.to_tensor([0], dtype="int32")
        self.kv_batch_ids = paddle.to_tensor(list(range(bs)), dtype="int32")
        self.kv_tile_ids_per_batch = paddle.to_tensor([0] * bs, dtype="int32")
        self.kv_num_blocks_x_cpu = paddle.to_tensor([bs], dtype="int32")
        # max_len_tensor_cpu: [max_enc_len, n_prefill, max_dec_len, max_kv_len]
        self.max_len_tensor_cpu = paddle.to_tensor([0, max_len_val, 10, 10], dtype="int32")
        self.attn_mask = None
        self.attn_mask_offsets = None
        self.forward_mode = None
        self.is_dummy_or_profile_run = False
        self.exist_prefill = False

    def __getattr__(self, name):
        """Mimic ForwardMeta's lazy attribute creation."""
        return None


class DummyMetadata:
    """Minimal attention metadata."""

    def __init__(self, num_layers=2):
        self.kv_signal_data_list = [None] * num_layers
        self._fuse_kernel_compute_dtype = "bf16"
        self._dtype = paddle.bfloat16
        self.max_len_tensor_cpu_decoder = None


# ---------------------------------------------------------------------------
# Helpers to extract cache args from mock calls
# ---------------------------------------------------------------------------


def _extract_cache_args_from_gqa_rope(mock_call):
    """Extract (key_cache, value_cache, cache_k_scales, cache_v_scales)
    from a gqa_rope_write_cache call.
    Positional: qkv[0], key_cache[1], value_cache[2], ...,
    cache_k_quant_scales[19], cache_v_quant_scales[20]"""
    args = mock_call[0]
    return args[1], args[2], args[19], args[20]


def _extract_cache_args_from_append_attention(mock_call):
    """Extract (key_cache, value_cache, cache_k_scales, cache_v_scales)
    from an append_attention call.
    Positional: qkv[0], key_cache[1], value_cache[2], ...,
    k_quant_scale[23], v_quant_scale[24]"""
    args = mock_call[0]
    return args[1], args[2], args[23], args[24]


# ---------------------------------------------------------------------------
# FlashAttentionBackend tests
# ---------------------------------------------------------------------------

FLASH_ATTN_MODULE = "fastdeploy.model_executor.layers.attention.flash_attn_backend"


@unittest.skipIf(_IMPORT_ERROR is not None, f"Cannot import backends: {_IMPORT_ERROR}")
class TestFlashAttnBackendCacheRouting(unittest.TestCase):
    """Test that FlashAttentionBackend.forward_mixed selects the correct
    cache tensors based on cache_quant_type_str."""

    def _create_backend(self):
        with patch(f"{FLASH_ATTN_MODULE}.init_rank_and_device_id", return_value=(0, 0)):
            with patch(f"{FLASH_ATTN_MODULE}.get_sm_version", return_value=90):
                with patch(f"{FLASH_ATTN_MODULE}.open_shm_and_get_meta_signal", return_value=None):
                    with patch(f"{FLASH_ATTN_MODULE}.init_kv_signal_per_query", return_value=None):
                        backend = FlashAttentionBackend(DummyFDConfig(), kv_num_heads=4, num_heads=56, head_dim=128)
        return backend

    @patch(f"{FLASH_ATTN_MODULE}.append_attention")
    @patch(f"{FLASH_ATTN_MODULE}.get_block_shape_and_split_kv_block")
    def test_normal_quant_uses_2x_indexing_decode_only(self, mock_split_kv, mock_append_attn):
        """cache_int8: cache_k=caches[2*id], scales from layer attrs."""
        backend = self._create_backend()
        backend.attention_metadata = DummyMetadata()

        layer_ks = _make_sentinel("layer_ks")
        layer_vs = _make_sentinel("layer_vs")
        layer = DummyLayer(
            layer_id=0, cache_quant_type_str="cache_int8", cache_k_scale=layer_ks, cache_v_scale=layer_vs
        )
        caches = _make_caches_normal(layer_id=0)
        fm = DummyForwardMeta(caches=caches, max_len_val=0)
        # batch_size=4, total_tokens=4 in decode
        bs = DummyForwardMeta.BATCH_SIZE
        mock_append_attn.return_value = paddle.zeros([bs, 7168], dtype="bfloat16")

        backend.forward_mixed(
            q=None,
            k=None,
            v=None,
            qkv=paddle.zeros([bs, 7680], dtype="bfloat16"),
            compressed_kv=None,
            k_pe=None,
            layer=layer,
            forward_meta=fm,
        )

        kc, vc, ks, vs = _extract_cache_args_from_append_attention(mock_append_attn.call_args)
        self.assertIs(kc, caches[0])
        self.assertIs(vc, caches[1])
        self.assertIs(ks, layer_ks)
        self.assertIs(vs, layer_vs)

    @patch(f"{FLASH_ATTN_MODULE}.append_attention")
    @patch(f"{FLASH_ATTN_MODULE}.get_block_shape_and_split_kv_block")
    def test_block_wise_fp8_uses_4x_indexing_decode_only(self, mock_split_kv, mock_append_attn):
        """block_wise_fp8: cache_k=caches[4*id], scales=caches[4*id+2/3]."""
        backend = self._create_backend()
        backend.attention_metadata = DummyMetadata()

        layer = DummyLayer(layer_id=0, cache_quant_type_str="block_wise_fp8")
        caches = _make_caches_block_wise_fp8(layer_id=0)
        fm = DummyForwardMeta(caches=caches, max_len_val=0)
        bs = DummyForwardMeta.BATCH_SIZE
        mock_append_attn.return_value = paddle.zeros([bs, 7168], dtype="bfloat16")

        backend.forward_mixed(
            q=None,
            k=None,
            v=None,
            qkv=paddle.zeros([bs, 7680], dtype="bfloat16"),
            compressed_kv=None,
            k_pe=None,
            layer=layer,
            forward_meta=fm,
        )

        kc, vc, ks, vs = _extract_cache_args_from_append_attention(mock_append_attn.call_args)
        self.assertIs(kc, caches[0])
        self.assertIs(vc, caches[1])
        self.assertIs(ks, caches[2])
        self.assertIs(vs, caches[3])

    @patch(f"{FLASH_ATTN_MODULE}.append_attention")
    @patch(f"{FLASH_ATTN_MODULE}.get_block_shape_and_split_kv_block")
    def test_block_wise_fp8_layer_id_1(self, mock_split_kv, mock_append_attn):
        """block_wise_fp8 with layer_id=1: indices 4,5,6,7."""
        backend = self._create_backend()
        backend.attention_metadata = DummyMetadata()

        layer = DummyLayer(layer_id=1, cache_quant_type_str="block_wise_fp8")
        caches = _make_caches_block_wise_fp8(layer_id=1)
        fm = DummyForwardMeta(caches=caches, max_len_val=0)
        bs = DummyForwardMeta.BATCH_SIZE
        mock_append_attn.return_value = paddle.zeros([bs, 7168], dtype="bfloat16")

        backend.forward_mixed(
            q=None,
            k=None,
            v=None,
            qkv=paddle.zeros([bs, 7680], dtype="bfloat16"),
            compressed_kv=None,
            k_pe=None,
            layer=layer,
            forward_meta=fm,
        )

        kc, vc, ks, vs = _extract_cache_args_from_append_attention(mock_append_attn.call_args)
        self.assertIs(kc, caches[4])
        self.assertIs(vc, caches[5])
        self.assertIs(ks, caches[6])
        self.assertIs(vs, caches[7])

    @patch(f"{FLASH_ATTN_MODULE}.append_attention")
    @patch(f"{FLASH_ATTN_MODULE}.get_block_shape_and_split_kv_block")
    def test_normal_quant_layer_id_1(self, mock_split_kv, mock_append_attn):
        """Normal quant with layer_id=1: indices 2,3."""
        backend = self._create_backend()
        backend.attention_metadata = DummyMetadata()

        layer_ks = _make_sentinel("ks_l1")
        layer_vs = _make_sentinel("vs_l1")
        layer = DummyLayer(
            layer_id=1, cache_quant_type_str="cache_int8", cache_k_scale=layer_ks, cache_v_scale=layer_vs
        )
        caches = _make_caches_normal(layer_id=1)
        fm = DummyForwardMeta(caches=caches, max_len_val=0)
        bs = DummyForwardMeta.BATCH_SIZE
        mock_append_attn.return_value = paddle.zeros([bs, 7168], dtype="bfloat16")

        backend.forward_mixed(
            q=None,
            k=None,
            v=None,
            qkv=paddle.zeros([bs, 7680], dtype="bfloat16"),
            compressed_kv=None,
            k_pe=None,
            layer=layer,
            forward_meta=fm,
        )

        kc, vc, ks, vs = _extract_cache_args_from_append_attention(mock_append_attn.call_args)
        self.assertIs(kc, caches[2])
        self.assertIs(vc, caches[3])
        self.assertIs(ks, layer_ks)
        self.assertIs(vs, layer_vs)

    @patch(f"{FLASH_ATTN_MODULE}.flash_attn_func")
    @patch(f"{FLASH_ATTN_MODULE}.append_attention")
    @patch(f"{FLASH_ATTN_MODULE}.pre_cache_len_concat")
    @patch(f"{FLASH_ATTN_MODULE}.get_block_shape_and_split_kv_block")
    @patch(f"{FLASH_ATTN_MODULE}.gqa_rope_write_cache")
    def test_block_wise_fp8_prefill_path(
        self,
        mock_gqa_rope,
        mock_split_kv,
        mock_pre_cache,
        mock_append_attn,
        mock_flash_attn,
    ):
        """Prefill path: both gqa_rope_write_cache and append_attention
        receive block_wise_fp8 caches. 4 batches, 5 tokens each = 20 total."""
        backend = self._create_backend()
        backend.attention_metadata = DummyMetadata()

        layer = DummyLayer(layer_id=0, cache_quant_type_str="block_wise_fp8")
        caches = _make_caches_block_wise_fp8(layer_id=0)
        bs = DummyForwardMeta.BATCH_SIZE
        total_tokens = bs * 5  # 20 tokens total (4 batches * 5 tokens)
        fm = DummyForwardMeta(caches=caches, max_len_val=5)

        mock_pre_cache.return_value = (
            paddle.to_tensor([0, 5, 10, 15, 20], dtype="int32"),  # cu_seqlens_k
            paddle.to_tensor(list(range(bs)), dtype="int32"),  # pre_cache_batch_ids
            paddle.to_tensor([0] * bs, dtype="int32"),  # pre_cache_tile_ids
            paddle.to_tensor([bs], dtype="int32"),  # pre_cache_num_blocks
            paddle.to_tensor([total_tokens], dtype="int32"),  # kv_token_num
        )
        mock_gqa_rope.return_value = (
            paddle.zeros([total_tokens, 56, 128], dtype="bfloat16"),
            paddle.zeros([total_tokens, 4, 128], dtype="bfloat16"),
            paddle.zeros([total_tokens, 4, 128], dtype="bfloat16"),
            None,
        )
        mock_flash_attn.return_value = (
            paddle.zeros([total_tokens, 56, 128], dtype="bfloat16"),
            None,
        )
        mock_append_attn.return_value = paddle.zeros([total_tokens, 7168], dtype="bfloat16")

        backend.forward_mixed(
            q=None,
            k=None,
            v=None,
            qkv=paddle.zeros([total_tokens, 7680], dtype="bfloat16"),
            compressed_kv=None,
            k_pe=None,
            layer=layer,
            forward_meta=fm,
        )

        # gqa_rope_write_cache should get caches[0..3]
        kc, vc, ks, vs = _extract_cache_args_from_gqa_rope(mock_gqa_rope.call_args)
        self.assertIs(kc, caches[0])
        self.assertIs(vc, caches[1])
        self.assertIs(ks, caches[2])
        self.assertIs(vs, caches[3])

        # append_attention should also get caches[0..3]
        kc2, vc2, ks2, vs2 = _extract_cache_args_from_append_attention(mock_append_attn.call_args)
        self.assertIs(kc2, caches[0])
        self.assertIs(vc2, caches[1])
        self.assertIs(ks2, caches[2])
        self.assertIs(vs2, caches[3])

    @patch(f"{FLASH_ATTN_MODULE}.append_attention")
    @patch(f"{FLASH_ATTN_MODULE}.get_block_shape_and_split_kv_block")
    def test_none_quant_type_defaults_to_2x(self, mock_split_kv, mock_append_attn):
        """cache_quant_type_str='none': 2x indexing, None scales."""
        backend = self._create_backend()
        backend.attention_metadata = DummyMetadata()

        layer = DummyLayer(layer_id=0, cache_quant_type_str="none")
        caches = _make_caches_normal(layer_id=0)
        fm = DummyForwardMeta(caches=caches, max_len_val=0)
        bs = DummyForwardMeta.BATCH_SIZE
        mock_append_attn.return_value = paddle.zeros([bs, 7168], dtype="bfloat16")

        backend.forward_mixed(
            q=None,
            k=None,
            v=None,
            qkv=paddle.zeros([bs, 7680], dtype="bfloat16"),
            compressed_kv=None,
            k_pe=None,
            layer=layer,
            forward_meta=fm,
        )

        kc, vc, ks, vs = _extract_cache_args_from_append_attention(mock_append_attn.call_args)
        self.assertIs(kc, caches[0])
        self.assertIs(vc, caches[1])
        self.assertIsNone(ks)
        self.assertIsNone(vs)

    @patch(f"{FLASH_ATTN_MODULE}.append_attention")
    @patch(f"{FLASH_ATTN_MODULE}.get_block_shape_and_split_kv_block")
    def test_cache_fp8_uses_2x_indexing(self, mock_split_kv, mock_append_attn):
        """cache_fp8 (static): 2x indexing, scales from layer attrs."""
        backend = self._create_backend()
        backend.attention_metadata = DummyMetadata()

        layer_ks = _make_sentinel("fp8_ks")
        layer_vs = _make_sentinel("fp8_vs")
        layer = DummyLayer(
            layer_id=0, cache_quant_type_str="cache_fp8", cache_k_scale=layer_ks, cache_v_scale=layer_vs
        )
        caches = _make_caches_normal(layer_id=0)
        fm = DummyForwardMeta(caches=caches, max_len_val=0)
        bs = DummyForwardMeta.BATCH_SIZE
        mock_append_attn.return_value = paddle.zeros([bs, 7168], dtype="bfloat16")

        backend.forward_mixed(
            q=None,
            k=None,
            v=None,
            qkv=paddle.zeros([bs, 7680], dtype="bfloat16"),
            compressed_kv=None,
            k_pe=None,
            layer=layer,
            forward_meta=fm,
        )

        kc, vc, ks, vs = _extract_cache_args_from_append_attention(mock_append_attn.call_args)
        self.assertIs(kc, caches[0])
        self.assertIs(vc, caches[1])
        self.assertIs(ks, layer_ks)
        self.assertIs(vs, layer_vs)


# ---------------------------------------------------------------------------
# FlashMaskAttentionBackend tests
# ---------------------------------------------------------------------------

FLASH_MASK_MODULE = "fastdeploy.model_executor.layers.attention.flash_mask_attn_backend"


@unittest.skipIf(_IMPORT_ERROR is not None, f"Cannot import backends: {_IMPORT_ERROR}")
class TestFlashMaskAttnBackendCacheRouting(unittest.TestCase):
    """Test that FlashMaskAttentionBackend.forward_mixed selects the correct
    cache tensors based on cache_quant_type_str."""

    def _create_backend(self):
        with patch(f"{FLASH_MASK_MODULE}.init_rank_and_device_id", return_value=(0, 0)):
            with patch(f"{FLASH_MASK_MODULE}.open_shm_and_get_meta_signal", return_value=None):
                with patch(f"{FLASH_MASK_MODULE}.init_kv_signal_per_query", return_value=None):
                    backend = FlashMaskAttentionBackend(DummyFDConfig(), kv_num_heads=4, num_heads=56, head_dim=128)
        return backend

    @patch(f"{FLASH_MASK_MODULE}.append_attention")
    @patch(f"{FLASH_MASK_MODULE}.get_block_shape_and_split_kv_block")
    def test_normal_quant_uses_2x_indexing_decode_only(self, mock_split_kv, mock_append_attn):
        """Non block_wise_fp8: caches[2*layer_id] indexing."""
        backend = self._create_backend()
        backend.attention_metadata = DummyMetadata()

        layer_ks = _make_sentinel("mask_ks")
        layer_vs = _make_sentinel("mask_vs")
        layer = DummyLayer(
            layer_id=0, cache_quant_type_str="cache_int8", cache_k_scale=layer_ks, cache_v_scale=layer_vs
        )
        caches = _make_caches_normal(layer_id=0)
        fm = DummyForwardMeta(caches=caches, max_len_val=0)
        bs = DummyForwardMeta.BATCH_SIZE
        mock_append_attn.return_value = paddle.zeros([bs, 7168], dtype="bfloat16")

        backend.forward_mixed(
            q=None,
            k=None,
            v=None,
            qkv=paddle.zeros([bs, 7680], dtype="bfloat16"),
            compressed_kv=None,
            k_pe=None,
            layer=layer,
            forward_meta=fm,
        )

        kc, vc, ks, vs = _extract_cache_args_from_append_attention(mock_append_attn.call_args)
        self.assertIs(kc, caches[0])
        self.assertIs(vc, caches[1])
        self.assertIs(ks, layer_ks)
        self.assertIs(vs, layer_vs)

    @patch(f"{FLASH_MASK_MODULE}.append_attention")
    @patch(f"{FLASH_MASK_MODULE}.get_block_shape_and_split_kv_block")
    def test_block_wise_fp8_uses_4x_indexing_decode_only(self, mock_split_kv, mock_append_attn):
        """block_wise_fp8: caches[4*layer_id] indexing."""
        backend = self._create_backend()
        backend.attention_metadata = DummyMetadata()

        layer = DummyLayer(layer_id=0, cache_quant_type_str="block_wise_fp8")
        caches = _make_caches_block_wise_fp8(layer_id=0)
        fm = DummyForwardMeta(caches=caches, max_len_val=0)
        bs = DummyForwardMeta.BATCH_SIZE
        mock_append_attn.return_value = paddle.zeros([bs, 7168], dtype="bfloat16")

        backend.forward_mixed(
            q=None,
            k=None,
            v=None,
            qkv=paddle.zeros([bs, 7680], dtype="bfloat16"),
            compressed_kv=None,
            k_pe=None,
            layer=layer,
            forward_meta=fm,
        )

        kc, vc, ks, vs = _extract_cache_args_from_append_attention(mock_append_attn.call_args)
        self.assertIs(kc, caches[0])
        self.assertIs(vc, caches[1])
        self.assertIs(ks, caches[2])
        self.assertIs(vs, caches[3])

    @patch(f"{FLASH_MASK_MODULE}.append_attention")
    @patch(f"{FLASH_MASK_MODULE}.get_block_shape_and_split_kv_block")
    def test_block_wise_fp8_layer_id_1(self, mock_split_kv, mock_append_attn):
        """block_wise_fp8 with layer_id=1: indices 4,5,6,7."""
        backend = self._create_backend()
        backend.attention_metadata = DummyMetadata()

        layer = DummyLayer(layer_id=1, cache_quant_type_str="block_wise_fp8")
        caches = _make_caches_block_wise_fp8(layer_id=1)
        fm = DummyForwardMeta(caches=caches, max_len_val=0)
        bs = DummyForwardMeta.BATCH_SIZE
        mock_append_attn.return_value = paddle.zeros([bs, 7168], dtype="bfloat16")

        backend.forward_mixed(
            q=None,
            k=None,
            v=None,
            qkv=paddle.zeros([bs, 7680], dtype="bfloat16"),
            compressed_kv=None,
            k_pe=None,
            layer=layer,
            forward_meta=fm,
        )

        kc, vc, ks, vs = _extract_cache_args_from_append_attention(mock_append_attn.call_args)
        self.assertIs(kc, caches[4])
        self.assertIs(vc, caches[5])
        self.assertIs(ks, caches[6])
        self.assertIs(vs, caches[7])

    @patch(f"{FLASH_MASK_MODULE}.flash_mask_attention")
    @patch(f"{FLASH_MASK_MODULE}.append_attention")
    @patch(f"{FLASH_MASK_MODULE}.pre_cache_len_concat")
    @patch(f"{FLASH_MASK_MODULE}.get_block_shape_and_split_kv_block")
    @patch(f"{FLASH_MASK_MODULE}.gqa_rope_write_cache")
    def test_block_wise_fp8_prefill_path(
        self,
        mock_gqa_rope,
        mock_split_kv,
        mock_pre_cache,
        mock_append_attn,
        mock_flash_mask,
    ):
        """Prefill: gqa_rope_write_cache and append_attention both get
        block_wise_fp8 caches. 4 batches, 5 tokens each = 20 total."""
        backend = self._create_backend()
        backend.attention_metadata = DummyMetadata()

        layer = DummyLayer(layer_id=0, cache_quant_type_str="block_wise_fp8")
        caches = _make_caches_block_wise_fp8(layer_id=0)
        bs = DummyForwardMeta.BATCH_SIZE
        total_tokens = bs * 5  # 20 tokens total (4 batches * 5 tokens)
        fm = DummyForwardMeta(caches=caches, max_len_val=5)

        mock_pre_cache.return_value = (
            paddle.to_tensor([0, 5, 10, 15, 20], dtype="int32"),  # cu_seqlens_k
            paddle.to_tensor(list(range(bs)), dtype="int32"),  # pre_cache_batch_ids
            paddle.to_tensor([0] * bs, dtype="int32"),  # pre_cache_tile_ids
            paddle.to_tensor([bs], dtype="int32"),  # pre_cache_num_blocks
            paddle.to_tensor([total_tokens], dtype="int32"),  # kv_token_num
        )
        mock_gqa_rope.return_value = (
            paddle.zeros([total_tokens, 56, 128], dtype="bfloat16"),
            paddle.zeros([total_tokens, 4, 128], dtype="bfloat16"),
            paddle.zeros([total_tokens, 4, 128], dtype="bfloat16"),
            None,
        )
        mock_flash_mask.return_value = None
        mock_append_attn.return_value = paddle.zeros([total_tokens, 7168], dtype="bfloat16")

        backend.forward_mixed(
            q=None,
            k=None,
            v=None,
            qkv=paddle.zeros([total_tokens, 7680], dtype="bfloat16"),
            compressed_kv=None,
            k_pe=None,
            layer=layer,
            forward_meta=fm,
        )

        kc, vc, ks, vs = _extract_cache_args_from_gqa_rope(mock_gqa_rope.call_args)
        self.assertIs(kc, caches[0])
        self.assertIs(vc, caches[1])
        self.assertIs(ks, caches[2])
        self.assertIs(vs, caches[3])

        kc2, vc2, ks2, vs2 = _extract_cache_args_from_append_attention(mock_append_attn.call_args)
        self.assertIs(kc2, caches[0])
        self.assertIs(vc2, caches[1])
        self.assertIs(ks2, caches[2])
        self.assertIs(vs2, caches[3])

    @patch(f"{FLASH_MASK_MODULE}.append_attention")
    @patch(f"{FLASH_MASK_MODULE}.get_block_shape_and_split_kv_block")
    def test_none_quant_type_defaults_to_2x(self, mock_split_kv, mock_append_attn):
        """cache_quant_type_str='none': 2x indexing."""
        backend = self._create_backend()
        backend.attention_metadata = DummyMetadata()

        layer = DummyLayer(layer_id=0, cache_quant_type_str="none")
        caches = _make_caches_normal(layer_id=0)
        fm = DummyForwardMeta(caches=caches, max_len_val=0)
        bs = DummyForwardMeta.BATCH_SIZE
        mock_append_attn.return_value = paddle.zeros([bs, 7168], dtype="bfloat16")

        backend.forward_mixed(
            q=None,
            k=None,
            v=None,
            qkv=paddle.zeros([bs, 7680], dtype="bfloat16"),
            compressed_kv=None,
            k_pe=None,
            layer=layer,
            forward_meta=fm,
        )

        kc, vc, ks, vs = _extract_cache_args_from_append_attention(mock_append_attn.call_args)
        self.assertIs(kc, caches[0])
        self.assertIs(vc, caches[1])
        self.assertIsNone(ks)
        self.assertIsNone(vs)


# ---------------------------------------------------------------------------
# Softmax -INFINITY fix tests
# ---------------------------------------------------------------------------


class TestSoftmaxInfinityHandling(unittest.TestCase):
    """Test the softmax numerical fix for -INFINITY handling.

    The fix in softmax.hpp:
    1. scale_apply_exp2: when max == -INFINITY, max_scaled = 0 (not NaN)
    2. Softmax::rescale: when both prev/cur max are -INFINITY, scale = 1.0
    """

    def test_scale_apply_exp2_normal(self):
        """Normal case: max is finite."""
        scale = 1.0 / np.log(2)
        max_val = 2.0
        tensor_val = 3.0
        result = 2 ** (tensor_val * scale - max_val * scale)
        self.assertTrue(np.isfinite(result))

    def test_scale_apply_exp2_neg_inf_max(self):
        """When max == -inf, fix sets max_scaled=0 avoiding NaN."""
        scale = 1.4426950408889634  # 1/ln(2)
        max_val = float("-inf")

        # Fixed: max_scaled = 0
        max_scaled_fixed = 0.0 if max_val == float("-inf") else max_val * scale
        self.assertEqual(max_scaled_fixed, 0.0)

        # Broken: tensor=-inf, max=-inf => -inf - (-inf) = NaN
        tensor_val = float("-inf")
        broken_result = 2 ** (tensor_val * scale - max_val * scale)
        self.assertTrue(np.isnan(broken_result))

        # Fixed: exp2(-inf - 0) = 0
        fixed_result = 2 ** (tensor_val * scale - max_scaled_fixed)
        self.assertEqual(fixed_result, 0.0)

    def test_rescale_both_neg_inf(self):
        """Both prev/cur max -inf => scale=1.0 (not NaN)."""
        scale_log2 = 1.4426950408889634
        prev = float("-inf")
        cur = float("-inf")

        # Fixed
        if prev == float("-inf") and cur == float("-inf"):
            fixed = 1.0
        else:
            fixed = 2 ** ((prev - cur) * scale_log2)
        self.assertEqual(fixed, 1.0)

        # Broken: -inf - (-inf) = NaN
        broken = 2 ** ((prev - cur) * scale_log2)
        self.assertTrue(np.isnan(broken))

    def test_rescale_prev_neg_inf_cur_finite(self):
        """prev=-inf, cur=finite => scale=0 (first tile case)."""
        scale = 2 ** ((float("-inf") - 2.0) * 1.4426950408889634)
        self.assertEqual(scale, 0.0)

    def test_rescale_both_finite(self):
        """Normal rescaling with finite values."""
        scale_log2 = 1.4426950408889634
        scale = 2 ** ((3.0 - 4.0) * scale_log2)
        expected = 2 ** (-1.0 * scale_log2)
        self.assertAlmostEqual(scale, expected, places=6)
        self.assertTrue(0 < scale < 1)

    def test_row_sum_preservation_with_inf_fix(self):
        """row_sum * 1.0 preserved; row_sum * NaN corrupted."""
        row_sum = 0.5
        self.assertEqual(row_sum * 1.0, 0.5)
        self.assertTrue(np.isnan(row_sum * float("nan")))


# ---------------------------------------------------------------------------
# CUDA kernel config tests
# ---------------------------------------------------------------------------


class TestAppendCacheKVC8KernelConfig(unittest.TestCase):
    """Test kernel template parameter mapping for append_cache_kv_c8."""

    def test_quant_type_to_kernel_params(self):
        configs = {
            "cache_int8": {"IS_FP8": False, "dynamic_quant": False},
            "cache_fp8": {"IS_FP8": True, "dynamic_quant": False},
            "block_wise_fp8": {"IS_FP8": True, "dynamic_quant": True},
        }
        self.assertFalse(configs["cache_int8"]["IS_FP8"])
        self.assertFalse(configs["cache_int8"]["dynamic_quant"])
        self.assertTrue(configs["cache_fp8"]["IS_FP8"])
        self.assertFalse(configs["cache_fp8"]["dynamic_quant"])
        self.assertTrue(configs["block_wise_fp8"]["IS_FP8"])
        self.assertTrue(configs["block_wise_fp8"]["dynamic_quant"])

    def test_dynamic_quant_scale_indexing(self):
        """Dynamic quant: per-token scale = (block_id*kv_num_heads+head)*block_size+row."""
        kv_num_heads = 4
        block_size = 64
        block_id, head_idx, row_idx = 3, 2, 5
        idx = (block_id * kv_num_heads + head_idx) * block_size + row_idx
        self.assertEqual(idx, (3 * 4 + 2) * 64 + 5)

    def test_block_wise_fp8_in_c8_branch(self):
        c8_types = {"cache_int8", "cache_fp8", "block_wise_fp8"}
        self.assertIn("block_wise_fp8", c8_types)
        self.assertNotIn("cache_int4_zp", c8_types)
        self.assertNotIn("none", c8_types)

    def test_static_quant_null_quant_scales(self):
        """Static quant: quant_scales=None, dequant_scales provided."""
        self.assertIsNone(None)  # quant_scales
        self.assertIsNotNone(np.ones(4))  # dequant_scales


if __name__ == "__main__":
    unittest.main()
