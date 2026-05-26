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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import paddle

from fastdeploy.model_executor.layers.attention.ops import (
    config_for_attention,
    decode_unified_attention,
    decoder_write_cache_with_rope,
    init_kv_signal_per_query,
    init_signal_layerwise,
    open_shm_and_get_meta_signal,
)
from fastdeploy.spec_decode import SpecMethod

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)
from fastdeploy.model_executor.layers.attention.utils import init_rank_and_device_id


@dataclass
class DecodeUnifiedAttentionMetadata(AttentionMetadata):
    """
    DecodeUnifiedAttentionMetadata
    """

    _dtype: paddle.dtype = paddle.bfloat16
    # pd_disaggregation
    kv_signal_metadata: Optional[paddle.Tensor] = None
    kv_signal_data_list: List[Optional[paddle.Tensor]] = field(default_factory=list)

    _fuse_kernel_compute_dtype: str = "bf16"


def allocate_decode_unified_related_buffer(
    max_batch_size,
    max_model_len,
    encoder_block_shape_q,
    decoder_block_shape_q,
    decoder_step_token_num,
    num_heads,
    kv_num_heads,
    block_size,
    head_dim=128,
    dtype="bfloat16",
):
    # Initialize AttentionBackend buffers
    assert num_heads % kv_num_heads == 0
    assert max_model_len % block_size == 0
    assert max_model_len % encoder_block_shape_q == 0
    group_size = num_heads // kv_num_heads

    res = {}

    # Decode unified attention split ops buffers
    res["max_len_tensor_cpu"] = paddle.full([6], 0, dtype="int32").cpu()
    min_chunk_size = 512
    max_num_chunk = (max_model_len + min_chunk_size - 1) // min_chunk_size
    q_tile_size = 16
    q_tile_num = (decoder_step_token_num * group_size + q_tile_size - 1) // q_tile_size
    res["decode_block_indices"] = paddle.full(
        [max_batch_size * kv_num_heads * max_num_chunk * q_tile_num, 4], 0, dtype="int32"
    )
    res["decode_num_blocks"] = paddle.full([1], 0, dtype="int32")
    res["decode_chunk_size"] = paddle.full([1], 0, dtype="int32")
    res["decode_tmp_workspace"] = paddle.full(
        [max_batch_size * decoder_step_token_num, max_num_chunk, num_heads * head_dim], 0, dtype=dtype
    )
    res["decode_tmp_m"] = paddle.full(
        [max_batch_size * decoder_step_token_num, max_num_chunk, num_heads], 0, dtype="float32"
    )
    res["decode_tmp_d"] = paddle.full(
        [max_batch_size * decoder_step_token_num, max_num_chunk, num_heads], 0, dtype="float32"
    )

    return res


class DecodeUnifiedAttentionBackend(AttentionBackend):
    """
    DecodeUnifiedAttention backend implementation.
    """

    __infer_dynamic_dims_fields__ = ["attention_metadata"]
    attention_metadata: DecodeUnifiedAttentionMetadata

    def __init__(
        self,
        fd_config: FDConfig,
        kv_num_heads: int,
        num_heads: int,
        head_dim: int,
        encoder_block_shape_q: int = -1,
        decoder_block_shape_q: int = -1,
    ) -> None:
        """
        AppendAttentionBackend __init__
        """
        super().__init__()
        self.max_seq_len = fd_config.model_config.max_model_len
        self.causal = getattr(fd_config.model_config, "causal", True)

        self.kv_num_heads = kv_num_heads
        self.num_heads = num_heads
        self.group_size: int = self.num_heads // self.kv_num_heads
        self.head_dim = fd_config.model_config.head_dim
        self.attn_outputsize_tp = self.num_heads * self.head_dim
        self.block_size = fd_config.cache_config.block_size
        self.num_layers: int = fd_config.model_config.num_hidden_layers

        self.speculative_method = fd_config.speculative_config.method
        self.use_speculate = self.speculative_method is not None
        self.speculate_max_draft_token_num = fd_config.speculative_config.num_speculative_tokens
        if not self.use_speculate:
            self.speculate_max_draft_token_num = 0
        self.keep_pd_step_flag: bool = fd_config.speculative_config.model_type == "mtp"
        self.num_layers_draft_model: int = int(fd_config.speculative_config.method == SpecMethod.MTP)

        self.pd_disaggregation_mode: str = fd_config.parallel_config.pd_disaggregation_mode

        self.start_layer_index: int = fd_config.model_config.start_layer_index

        self.rank, self.device_id = init_rank_and_device_id(fd_config)

        self.rope_3d: bool = fd_config.enable_rope_3d_runtime
        if fd_config.speculative_config.model_type != "main":
            self.rope_3d = False
        # Note(ZKK): here must be consistent with append_attn_backend.py
        self.max_tokens_per_batch: int = self.speculate_max_draft_token_num + 1

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize attntion metadata hence all layers in the forward pass can reuse it."""
        metadata = DecodeUnifiedAttentionMetadata()
        metadata._dtype = paddle.get_default_dtype()
        if metadata._dtype == "bfloat16":
            metadata._fuse_kernel_compute_dtype = "bf16"
        elif metadata._dtype == "float16":
            metadata._fuse_kernel_compute_dtype = "fp16"
        elif metadata._dtype == "float32":
            metadata._fuse_kernel_compute_dtype = "fp32"

        # pd_disaggregation
        metadata.kv_signal_data_list = [None] * self.num_layers
        if self.pd_disaggregation_mode == "per_chunk":
            if not self.keep_pd_step_flag and not forward_meta.is_dummy_or_profile_run:
                init_kv_signal_per_query(
                    forward_meta.seq_lens_encoder,
                    forward_meta.seq_lens_this_time,
                    forward_meta.seq_lens_decoder,
                    self.rank,
                    self.num_layers + self.num_layers_draft_model,
                )
        elif self.pd_disaggregation_mode == "per_query":
            metadata.kv_signal_metadata = open_shm_and_get_meta_signal(
                self.rank, int(self.device_id), self.keep_pd_step_flag
            )

        self.attention_metadata = metadata

    def get_attention_meta(self) -> AttentionMetadata:
        """get_attention_meta"""
        return self.attention_metadata

    def _get_identity_rotary_embs(self, original_rotary_embs: paddle.Tensor) -> paddle.Tensor:
        """
        Create identity rotary embeddings (cos=1, sin=0) that make RoPE a no-op.

        This is used when RoPE has already been applied externally (e.g., by PaddleFormers).
        The identity transformation ensures: x * cos(0) + y * sin(0) = x, preserving the input.

        NOTE: Shape can change between prefill/decode, so we check if cached shape matches.
        """
        # Check if we need to recreate (shape mismatch or not cached)
        need_recreate = (
            not hasattr(self, "_identity_rotary_embs")
            or self._identity_rotary_embs is None
            or self._identity_rotary_embs.shape != original_rotary_embs.shape
        )

        if need_recreate:
            # Create identity RoPE: cos=1, sin=0
            identity = paddle.zeros_like(original_rotary_embs)
            identity[0] = 1.0  # cos = 1
            identity[1] = 0.0  # sin = 0
            self._identity_rotary_embs = identity

        return self._identity_rotary_embs

    def get_kv_cache_shape(
        self,
        max_num_blocks: int,
        kv_cache_quant_type: str = None,
    ):
        """
        Calculate kv cache shape
        """
        key_cache_shape = [max_num_blocks, self.kv_num_heads, self.block_size, self.head_dim]
        if kv_cache_quant_type is not None and kv_cache_quant_type == "int4_zp":
            key_cache_shape[-1] = self.head_dim // 2
        value_cache_shape = key_cache_shape
        return key_cache_shape, value_cache_shape

    def forward_mixed(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        qkv: paddle.Tensor,
        compressed_kv: paddle.Tensor,
        k_pe: paddle.Tensor,
        layer: Attention,
        forward_meta: ForwardMeta,
    ) -> paddle.Tensor:
        """
        forward_mixed
        """
        metadata = self.attention_metadata

        rope_already_applied = getattr(forward_meta, "rope_already_applied", False)
        if rope_already_applied and forward_meta.rotary_embs is not None:
            forward_meta.rotary_embs = self._get_identity_rotary_embs(forward_meta.rotary_embs)

        norm_after_rope_in_kernel = not getattr(layer, "qk_norm_before_rope", False)
        q_norm_weight = getattr(layer, "q_norm_weight", None) if norm_after_rope_in_kernel else None
        k_norm_weight = getattr(layer, "k_norm_weight", None) if norm_after_rope_in_kernel else None

        if self.rope_3d:
            assert len(forward_meta.rotary_embs.shape) == 6
        else:
            assert len(forward_meta.rotary_embs.shape) == 5
            if layer.use_neox_rotary_style:
                assert forward_meta.rotary_embs.shape[0:4] == [2, 1, self.max_seq_len, 1]
                # 128 is qwen3
                # 32 is glm
                # 64 is gpt-oss
                assert forward_meta.rotary_embs.shape[4] in [128, 32, 64]

        if self.pd_disaggregation_mode == "per_query":
            metadata.kv_signal_data_list[layer.layer_id] = init_signal_layerwise(
                metadata.kv_signal_metadata,
                layer.layer_id + self.start_layer_index,
            )
        cache_quant_type_str = getattr(layer, "cache_quant_type_str", "none")
        if cache_quant_type_str == "block_wise_fp8":
            cache_k = forward_meta.caches[4 * layer.layer_id]
            cache_v = forward_meta.caches[4 * layer.layer_id + 1]
            cache_k_scales = forward_meta.caches[4 * layer.layer_id + 2]
            cache_v_scales = forward_meta.caches[4 * layer.layer_id + 3]
        else:
            cache_k = forward_meta.caches[2 * layer.layer_id]
            cache_v = forward_meta.caches[2 * layer.layer_id + 1]
            cache_k_scales = getattr(layer, "cache_k_scale", None)
            cache_v_scales = getattr(layer, "cache_v_scale", None)

        if layer.layer_id == 0:
            config_for_attention(
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
                forward_meta.seq_lens_this_time,
                forward_meta.decode_block_indices,
                forward_meta.decode_num_blocks,
                forward_meta.decode_chunk_size,
                forward_meta.max_len_tensor_cpu,
                getattr(layer, "cache_quant_type_str", "none"),
                self.group_size,
                self.kv_num_heads,
                self.max_tokens_per_batch,
            )
        qkv_out = decoder_write_cache_with_rope(
            qkv,
            cache_k,
            cache_v,
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.batch_id_per_token,
            forward_meta.cu_seqlens_q,
            forward_meta.block_tables,
            forward_meta.max_len_tensor_cpu,
            forward_meta.rotary_embs,
            layer.qkv_bias,
            cache_k_scales,
            cache_v_scales,
            getattr(layer, "cache_k_out_scale", None),
            getattr(layer, "cache_v_out_scale", None),
            getattr(layer, "cache_k_zp", None),
            getattr(layer, "cache_v_zp", None),
            metadata.kv_signal_data_list[layer.layer_id],
            q_norm_weight,
            k_norm_weight,
            getattr(layer, "rms_norm_eps", 1e-6),
            getattr(layer, "cache_quant_type_str", "none"),
            layer.use_neox_rotary_style,
            self.rope_3d,
            self.max_seq_len,
            getattr(layer, "quant_max_bound", 0.0),
            getattr(layer, "quant_min_bound", 0.0),
            self.speculative_method is not None,
        )
        res_decoder = paddle.empty(
            [qkv.shape[0], self.num_heads * self.head_dim],
            dtype=qkv.dtype,
        )
        decode_unified_attention(
            qkv_out,
            cache_k,
            cache_v,
            forward_meta.decode_tmp_workspace,
            forward_meta.decode_tmp_m,
            forward_meta.decode_tmp_d,
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.batch_id_per_token,
            forward_meta.cu_seqlens_q,
            forward_meta.block_tables,
            forward_meta.decode_block_indices,
            forward_meta.decode_num_blocks,
            forward_meta.decode_chunk_size,
            forward_meta.max_len_tensor_cpu,
            forward_meta.attn_mask,
            cache_k_scales,
            cache_v_scales,
            getattr(layer, "cache_k_out_scale", None),
            getattr(layer, "cache_v_out_scale", None),
            getattr(layer, "cache_k_zp", None),
            getattr(layer, "cache_v_zp", None),
            forward_meta.attn_mask_offsets,
            getattr(layer, "sinks", None),
            res_decoder,
            getattr(layer, "cache_quant_type_str", "none"),
            self.max_seq_len,
            getattr(layer, "quant_max_bound", 0.0),
            getattr(layer, "quant_min_bound", 0.0),
            self.speculate_max_draft_token_num + 1,
            self.causal,
        )
        return res_decoder
