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
Unit tests for KV cache dynamic quantization on FlashAttentionBackend
and FlashMaskAttentionBackend.

Tests:
  1. Smoke tests: forward_mixed runs without error under dynamic C8.
  2. Diff tests: dynamic C8 vs C16 produce consistent outputs.
  3. GPU tests: real GPU forward calls (skipped without GPU).

Extensibility: To add a new quant type (e.g., C4), add an entry to
QUANT_CONFIGS and follow the existing test patterns.
"""

import math
import unittest
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import paddle

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
# Quant config registry (extend here for new quant types)
# ---------------------------------------------------------------------------


@dataclass
class QuantConfig:
    """Configuration for a cache quantization type."""

    cache_quant_type_str: str  # e.g., "block_wise_fp8", "none"
    cache_dtype: str  # "uint8" for quantized, "bfloat16" for fp16/bf16
    has_dynamic_scales: bool  # True if scales stored in caches list
    caches_per_layer: int  # 4 for dynamic (k,v,k_scale,v_scale), 2 otherwise


QUANT_CONFIGS = {
    "C16": QuantConfig("none", "bfloat16", False, 2),
    "C8_dynamic": QuantConfig("block_wise_fp8", "uint8", True, 4),
    # Future: "C4_dynamic": QuantConfig("block_wise_int4", "uint8", True, 4),
}

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
NUM_HEADS = 56
KV_NUM_HEADS = 4
HEAD_DIM = 128
BLOCK_SIZE = 64
NUM_LAYERS = 2
MAX_SEQ_LEN = 2048
QKV_DIM = (NUM_HEADS + 2 * KV_NUM_HEADS) * HEAD_DIM  # 7680
ATTN_OUTPUT_DIM = NUM_HEADS * HEAD_DIM  # 7168
Q_DIM = NUM_HEADS * HEAD_DIM  # 7168
K_DIM = KV_NUM_HEADS * HEAD_DIM  # 512
V_DIM = KV_NUM_HEADS * HEAD_DIM  # 512

FLASH_ATTN_MODULE = "fastdeploy.model_executor.layers.attention.flash_attn_backend"
FLASH_MASK_MODULE = "fastdeploy.model_executor.layers.attention.flash_mask_attn_backend"

# Backend registry for parameterized tests
BACKENDS = [
    ("flash_attn", FlashAttentionBackend, FLASH_ATTN_MODULE),
    ("flash_mask", FlashMaskAttentionBackend, FLASH_MASK_MODULE),
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class DummyFDConfig:
    """Minimal FDConfig for constructing backend objects."""

    def __init__(self):
        self.cache_config = type("C", (), {"block_size": BLOCK_SIZE})()
        self.model_config = type(
            "M",
            (),
            {
                "max_model_len": MAX_SEQ_LEN,
                "head_dim": HEAD_DIM,
                "num_hidden_layers": NUM_LAYERS,
                "causal": True,
                "start_layer_index": 0,
                "rope_3d": False,
                "use_3d_rope": False,
            },
        )()
        self.scheduler_config = type("S", (), {"max_num_seqs": BATCH_SIZE})()
        self.graph_opt_config = type("G", (), {"cudagraph_capture_sizes": None})()
        self.parallel_config = type(
            "P",
            (),
            {
                "block_size": BLOCK_SIZE,
                "data_parallel_rank": 0,
                "pd_disaggregation_mode": "none",
                "expert_parallel_rank": 0,
            },
        )()
        self.speculative_config = type(
            "Sp",
            (),
            {
                "method": None,
                "max_draft_token_num": 0,
                "num_speculative_tokens": 0,
                "model_type": "main",
            },
        )()


class DummyLayer:
    """Mimics the Attention layer object."""

    def __init__(self, layer_id=0, quant_config=None):
        self.layer_id = layer_id
        cfg = quant_config or QUANT_CONFIGS["C16"]
        self.cache_quant_type_str = cfg.cache_quant_type_str
        # Static quant types use layer-level scales; dynamic types use caches list
        if not cfg.has_dynamic_scales and cfg.cache_quant_type_str != "none":
            self.cache_k_scale = paddle.ones([1], dtype="float32")
            self.cache_v_scale = paddle.ones([1], dtype="float32")
        else:
            self.cache_k_scale = None
            self.cache_v_scale = None
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


class DummyForwardMeta:
    """Minimal ForwardMeta for decode-only mode (max_len_val=0)."""

    def __init__(self, caches, max_len_val=0):
        bs = BATCH_SIZE
        self.caches = caches
        self.seq_lens_encoder = paddle.to_tensor([0] * bs, dtype="int32")
        self.seq_lens_decoder = paddle.to_tensor([1] * bs, dtype="int32")
        self.seq_lens_this_time = paddle.to_tensor([1] * bs, dtype="int32")
        self.cu_seqlens_q = paddle.to_tensor(list(range(bs + 1)), dtype="int32")
        self.cu_seqlens_k = paddle.to_tensor(list(range(bs + 1)), dtype="int32")
        self.rotary_embs = paddle.zeros([bs, 1, HEAD_DIM], dtype="float32")
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
        self.max_len_tensor_cpu = paddle.to_tensor([0, max_len_val, 10, 10], dtype="int32")
        self.attn_mask = None
        self.attn_mask_offsets = None
        self.forward_mode = None
        self.is_dummy_or_profile_run = False
        self.exist_prefill = False

    def __getattr__(self, name):
        return None


class DummyMetadata:
    """Minimal attention metadata."""

    def __init__(self, num_layers=NUM_LAYERS):
        self.kv_signal_data_list = [None] * num_layers
        self._fuse_kernel_compute_dtype = "bf16"
        self._dtype = paddle.bfloat16
        self.max_len_tensor_cpu_decoder = None


def make_qkv_inputs(token_num=BATCH_SIZE, dtype="bfloat16"):
    """Create real q, k, v, qkv tensors with random data.

    Returns:
        (q, k, v, qkv) where qkv is the fused [token_num, QKV_DIM] tensor
        and q/k/v are the individual head-grouped tensors.
    """
    qkv = paddle.randn([token_num, QKV_DIM]).cast(dtype)
    q = qkv[:, :Q_DIM].reshape([token_num, NUM_HEADS, HEAD_DIM])
    k = qkv[:, Q_DIM : Q_DIM + K_DIM].reshape([token_num, KV_NUM_HEADS, HEAD_DIM])
    v = qkv[:, Q_DIM + K_DIM :].reshape([token_num, KV_NUM_HEADS, HEAD_DIM])
    return q, k, v, qkv


def make_caches(quant_config, layer_id=0):
    """Create a caches list for the given quant config and layer_id.

    For dynamic C8: [cache_k, cache_v, k_scale, v_scale] * layers
    For C16:   [cache_k, cache_v] * layers
    """
    num_entries = quant_config.caches_per_layer * (layer_id + 1)
    return [paddle.zeros([1], dtype="float32") for _ in range(num_entries)]


def create_backend(backend_class, module_path):
    """Factory to create a backend instance with mocked init dependencies.
    During attention initialization, some prerequisite steps are performed,
    such as initializing the distributed environment.
    """
    patches = [
        patch(f"{module_path}.init_rank_and_device_id", return_value=(0, 0)),
        patch(f"{module_path}.open_shm_and_get_meta_signal", return_value=None),
        patch(f"{module_path}.init_kv_signal_per_query", return_value=None),
    ]
    # FlashAttentionBackend also needs get_sm_version mocked
    if "flash_attn_backend" in module_path and "flash_mask" not in module_path:
        patches.append(patch(f"{module_path}.get_sm_version", return_value=90))

    for p in patches:
        p.start()
    try:
        backend = backend_class(DummyFDConfig(), kv_num_heads=KV_NUM_HEADS, num_heads=NUM_HEADS, head_dim=HEAD_DIM)
    finally:
        for p in patches:
            p.stop()
    return backend


def _run_forward_mocked(backend, module_path, quant_config, layer_id=0, return_tensor=None, qkv_inputs=None):
    """Run forward_mixed with mocked external ops, return the result.

    Args:
        backend: The attention backend instance.
        module_path: Module path for patching ops.
        quant_config: QuantConfig to use.
        layer_id: Layer ID for the dummy layer.
        return_tensor: If provided, mock append_attention to return this tensor.
        qkv_inputs: Optional (q, k, v, qkv) tuple. Generated if not provided.
    """
    backend.attention_metadata = DummyMetadata()
    layer = DummyLayer(layer_id=layer_id, quant_config=quant_config)
    caches = make_caches(quant_config, layer_id=layer_id)
    fm = DummyForwardMeta(caches=caches, max_len_val=0)

    if qkv_inputs is None:
        q, k, v, qkv = make_qkv_inputs()
    else:
        q, k, v, qkv = qkv_inputs

    if return_tensor is None:
        return_tensor = paddle.zeros([BATCH_SIZE, ATTN_OUTPUT_DIM], dtype="bfloat16")

    with patch(f"{module_path}.append_attention", return_value=return_tensor):
        with patch(f"{module_path}.get_block_shape_and_split_kv_block"):
            result = backend.forward_mixed(
                q=q,
                k=k,
                v=v,
                qkv=qkv,
                compressed_kv=None,
                k_pe=None,
                layer=layer,
                forward_meta=fm,
            )
    return result


# ---------------------------------------------------------------------------
# Part 1: Mock-based smoke tests (no GPU required)
# ---------------------------------------------------------------------------


@unittest.skipIf(_IMPORT_ERROR is not None, f"Cannot import backends: {_IMPORT_ERROR}")
class TestBackendForwardSmoke(unittest.TestCase):
    """Smoke test: forward_mixed runs without error for each backend x quant config."""

    def _smoke_test(self, backend_class, module_path, quant_config_name):
        config = QUANT_CONFIGS[quant_config_name]
        backend = create_backend(backend_class, module_path)
        # Should not raise
        result = _run_forward_mocked(backend, module_path, config)
        self.assertIsNotNone(result)

    def test_flash_attn_c8_dynamic(self):
        self._smoke_test(FlashAttentionBackend, FLASH_ATTN_MODULE, "C8_dynamic")

    def test_flash_attn_c16(self):
        self._smoke_test(FlashAttentionBackend, FLASH_ATTN_MODULE, "C16")

    def test_flash_mask_attn_c8_dynamic(self):
        self._smoke_test(FlashMaskAttentionBackend, FLASH_MASK_MODULE, "C8_dynamic")

    def test_flash_mask_attn_c16(self):
        self._smoke_test(FlashMaskAttentionBackend, FLASH_MASK_MODULE, "C16")


# ---------------------------------------------------------------------------
# Part 2: Mock-based C8 vs C16 diff tests (no GPU required)
# ---------------------------------------------------------------------------


@unittest.skipIf(_IMPORT_ERROR is not None, f"Cannot import backends: {_IMPORT_ERROR}")
class TestBackendC8VsC16OutputDiff(unittest.TestCase):
    """Diff test: C8 dynamic and C16 produce identical outputs when external
    ops return the same data (validates the forward path is consistent)."""

    def _diff_test(self, backend_class, module_path):
        # Use the same known tensor as the mock return value for both configs
        known_output = paddle.randn([BATCH_SIZE, ATTN_OUTPUT_DIM]).cast("bfloat16")
        # Use the same qkv inputs for both configs
        shared_qkv = make_qkv_inputs()

        backend_c8 = create_backend(backend_class, module_path)
        result_c8 = _run_forward_mocked(
            backend_c8,
            module_path,
            QUANT_CONFIGS["C8_dynamic"],
            return_tensor=known_output.clone(),
            qkv_inputs=shared_qkv,
        )

        backend_c16 = create_backend(backend_class, module_path)
        result_c16 = _run_forward_mocked(
            backend_c16,
            module_path,
            QUANT_CONFIGS["C16"],
            return_tensor=known_output.clone(),
            qkv_inputs=shared_qkv,
        )

        np.testing.assert_array_equal(
            result_c8.numpy(),
            result_c16.numpy(),
            err_msg=f"C8 dynamic and C16 outputs differ for {backend_class.__name__}",
        )

    def test_flash_attn_c8_vs_c16(self):
        self._diff_test(FlashAttentionBackend, FLASH_ATTN_MODULE)

    def test_flash_mask_attn_c8_vs_c16(self):
        self._diff_test(FlashMaskAttentionBackend, FLASH_MASK_MODULE)


# ---------------------------------------------------------------------------
# Part 3: GPU-based tests (require real GPU)
# ---------------------------------------------------------------------------

_HAS_GPU = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0


def _make_gpu_caches(quant_config, max_block_num=16):
    """Create real GPU cache tensors following the paged KV cache layout.

    Cache shape: (max_block_num, kv_num_heads, block_size, head_dim)
    Scale shape: (max_block_num, kv_num_heads, block_size)
    """
    cache_shape = [max_block_num, KV_NUM_HEADS, BLOCK_SIZE, HEAD_DIM]
    scale_shape = [max_block_num, KV_NUM_HEADS, BLOCK_SIZE]

    cache_k = paddle.zeros(cache_shape, dtype=quant_config.cache_dtype)
    cache_v = paddle.zeros(cache_shape, dtype=quant_config.cache_dtype)

    if quant_config.has_dynamic_scales:
        cache_k_scale = paddle.zeros(scale_shape, dtype="bfloat16")
        cache_v_scale = paddle.zeros(scale_shape, dtype="bfloat16")
        return [cache_k, cache_v, cache_k_scale, cache_v_scale]
    else:
        return [cache_k, cache_v]


def _make_gpu_forward_meta(caches, seq_len=1):
    """Create a ForwardMeta suitable for real GPU decode-only forward."""
    bs = BATCH_SIZE
    block_num_per_seq = math.ceil(seq_len / BLOCK_SIZE) or 1

    block_tables = paddle.zeros([bs, block_num_per_seq], dtype="int32")
    idx = 0
    for i in range(bs):
        for j in range(block_num_per_seq):
            block_tables[i, j] = idx
            idx += 1

    fm = DummyForwardMeta(caches=caches, max_len_val=0)
    fm.block_tables = block_tables
    return fm


@unittest.skipIf(not _HAS_GPU, "No GPU available")
@unittest.skipIf(_IMPORT_ERROR is not None, f"Cannot import backends: {_IMPORT_ERROR}")
class TestBackendForwardGPU(unittest.TestCase):
    """GPU-based tests: real forward_mixed calls on GPU hardware."""

    def _gpu_smoke_test(self, backend_class, module_path, quant_config_name):
        """Test that forward_mixed runs on GPU without error."""
        config = QUANT_CONFIGS[quant_config_name]
        backend = create_backend(backend_class, module_path)
        backend.attention_metadata = DummyMetadata()

        max_block_num = BATCH_SIZE
        caches = _make_gpu_caches(config, max_block_num=max_block_num)
        layer = DummyLayer(layer_id=0, quant_config=config)
        fm = _make_gpu_forward_meta(caches, seq_len=1)
        q, k, v, qkv = make_qkv_inputs()

        result = backend.forward_mixed(
            q=q,
            k=k,
            v=v,
            qkv=qkv,
            compressed_kv=None,
            k_pe=None,
            layer=layer,
            forward_meta=fm,
        )
        self.assertEqual(result.shape, [BATCH_SIZE, ATTN_OUTPUT_DIM])

    def _gpu_diff_test(self, backend_class, module_path):
        """Compare C8 dynamic vs C16 outputs on GPU (loose tolerance)."""
        max_block_num = BATCH_SIZE
        q, k, v, qkv = make_qkv_inputs()

        results = {}
        for config_name in ["C8_dynamic", "C16"]:
            config = QUANT_CONFIGS[config_name]
            backend = create_backend(backend_class, module_path)
            backend.attention_metadata = DummyMetadata()

            caches = _make_gpu_caches(config, max_block_num=max_block_num)
            layer = DummyLayer(layer_id=0, quant_config=config)
            fm = _make_gpu_forward_meta(caches, seq_len=1)

            results[config_name] = backend.forward_mixed(
                q=q.clone(),
                k=k.clone(),
                v=v.clone(),
                qkv=qkv.clone(),
                compressed_kv=None,
                k_pe=None,
                layer=layer,
                forward_meta=fm,
            )

        np.testing.assert_allclose(
            results["C8_dynamic"].cast("float32").numpy(),
            results["C16"].cast("float32").numpy(),
            rtol=0.1,
            atol=0.1,
            err_msg=f"C8 dynamic vs C16 GPU output diff too large for {backend_class.__name__}",
        )

    def test_flash_attn_c8_dynamic_gpu(self):
        self._gpu_smoke_test(FlashAttentionBackend, FLASH_ATTN_MODULE, "C8_dynamic")

    def test_flash_attn_c16_gpu(self):
        self._gpu_smoke_test(FlashAttentionBackend, FLASH_ATTN_MODULE, "C16")

    def test_flash_mask_attn_c8_dynamic_gpu(self):
        self._gpu_smoke_test(FlashMaskAttentionBackend, FLASH_MASK_MODULE, "C8_dynamic")

    def test_flash_mask_attn_c16_gpu(self):
        self._gpu_smoke_test(FlashMaskAttentionBackend, FLASH_MASK_MODULE, "C16")

    def test_flash_attn_c8_vs_c16_gpu(self):
        self._gpu_diff_test(FlashAttentionBackend, FLASH_ATTN_MODULE)

    def test_flash_mask_attn_c8_vs_c16_gpu(self):
        self._gpu_diff_test(FlashMaskAttentionBackend, FLASH_MASK_MODULE)


if __name__ == "__main__":
    unittest.main()
