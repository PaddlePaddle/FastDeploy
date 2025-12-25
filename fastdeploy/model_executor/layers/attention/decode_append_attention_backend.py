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
    decode_append_attention,
    decoder_write_cache_with_rope,
    init_kv_signal_per_query,
    init_signal_layerwise,
    open_shm_and_get_meta_signal,
)

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
class DecodeAppendAttentionMetadata(AttentionMetadata):
    """
    AppendAttentionMetadata
    """

    _dtype: paddle.dtype = paddle.bfloat16
    # pd_disaggregation
    kv_signal_metadata: Optional[paddle.Tensor] = None
    kv_signal_data_list: List[Optional[paddle.Tensor]] = field(default_factory=list)


class DecodeAppendAttentionBackend(AttentionBackend):
    """
    AppendAttentionBackend backend implementation.
    """

    __infer_dynamic_dims_fields__ = ["attention_metadata"]
    attention_metadata: DecodeAppendAttentionMetadata

    def __init__(
        self,
        fd_config: FDConfig,
        max_batch_size: int,
    ) -> None:
        """
        AppendAttentionBackend __init__
        """
        super().__init__()
        self.attention_metadata: DecodeAppendAttentionMetadata = None
        self.block_size: int = fd_config.cache_config.block_size
        self.max_seq_len: int = fd_config.model_config.max_model_len
        self.rope_theta: float = (
            10000.0 if fd_config.model_config.rope_theta is None else fd_config.model_config.rope_theta
        )
        self.rope_3d: bool = getattr(fd_config.model_config, "rope_3d", False) or getattr(
            fd_config.model_config, "use_3d_rope", False
        )
        if fd_config.speculative_config.model_type != "main":
            self.rope_3d = False
        self.causal: bool = getattr(fd_config.model_config, "causal", True)
        self.speculative_method: str = fd_config.speculative_config.method
        self.speculate_max_draft_token_num: int = fd_config.speculative_config.num_speculative_tokens
        self.max_tokens_per_batch = self.speculate_max_draft_token_num + 1
        self.keep_pd_step_flag: bool = fd_config.speculative_config.model_type == "mtp"
        self.num_layers_draft_model: int = int(fd_config.speculative_config.method in ["mtp"])

        self.num_heads: int = (
            fd_config.model_config.num_attention_heads // fd_config.parallel_config.tensor_parallel_size
        )
        self.model_config.kv_num_heads = max(
            1,
            int(fd_config.model_config.num_key_value_heads) // fd_config.parallel_config.tensor_parallel_size,
        )
        self.head_dim: int = fd_config.model_config.head_dim

        self.kv_num_heads: int = self.model_config.kv_num_heads
        self.group_size: int = self.num_heads // self.kv_num_heads
        self.head_dim: int = fd_config.model_config.head_dim
        self.num_layers: int = fd_config.model_config.num_hidden_layers

        self.pd_disaggregation_mode: str = fd_config.parallel_config.pd_disaggregation_mode

        self.start_layer_index: int = fd_config.model_config.start_layer_index

        if fd_config.parallel_config.expert_parallel_rank is None:
            fd_config.parallel_config.expert_parallel_rank = 0

        self.rank, self.device_id = init_rank_and_device_id(fd_config)
        self.use_output = not fd_config.graph_opt_config.full_cuda_graph
        self.fd_config = fd_config
        self.max_batch_size = max_batch_size
        self.buffer: dict = self.allocate_buffer()

    def allocate_buffer(
        self,
    ) -> dict:
        # Initialize AttentionBackend buffers
        assert self.num_heads % self.kv_num_heads == 0
        assert self.max_seq_len % self.block_size == 0

        min_chunk_size = 128
        max_num_chunk = (self.max_seq_len + min_chunk_size - 1) // min_chunk_size

        q_tile_size = 16 if self.max_tokens_per_batch * self.group_size <= 16 else 32
        q_tile_num = (self.max_tokens_per_batch * self.group_size + q_tile_size - 1) // q_tile_size
        buffer = {}
        buffer["max_len_tensor_cpu"] = paddle.full([5], 0, dtype="int32")
        # block_indices: Launched block's indices with 4 dimensions [batch_idx, kv_head_idx, chunk_idx, q_tile_idx] in decode append attention backend
        buffer["block_indices"] = paddle.full(
            [self.max_batch_size * self.kv_num_heads * max_num_chunk * q_tile_num, 4], 0, dtype="int32"
        )
        # num_blocks: Number of Launched blocks in decode append attention backend, researched by config_for_attention op
        buffer["num_blocks"] = paddle.full([1], 0, dtype="int32")
        # chunk_size: Chunk size for split kv cache in decode append attention backend, researched by config_for_attention op
        buffer["chunk_size"] = paddle.full([1], 0, dtype="int32")
        # tmp_workspace: Workspace tensor for temporary store the result before merging in decode append attention backend
        buffer["tmp_workspace"] = paddle.full(
            [self.max_batch_size * self.max_tokens_per_batch, max_num_chunk, self.num_heads * self.head_dim],
            0,
            dtype=self.attention_metadata._dtype,
        )
        # tmp_m: Tmp_m tensor for temporary store the max value before merging in decode append attention backend
        buffer["tmp_m"] = paddle.full(
            [self.max_batch_size * self.max_tokens_per_batch, max_num_chunk, self.num_head], 0, dtype="float32"
        )
        # tmp_d: Tmp_d tensor for temporary store the exponential sum before merging in decode append attention backend
        buffer["tmp_d"] = paddle.full(
            [self.max_batch_size * self.max_tokens_per_batch, max_num_chunk, self.num_head], 0, dtype="float32"
        )
        return buffer

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize attntion metadata hence all layers in the forward pass can reuse it."""
        metadata = DecodeAppendAttentionMetadata()
        metadata._dtype = paddle.get_default_dtype()

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

        self.attention_metadata: AttentionMetadata = metadata

    def get_attntion_meta(self) -> AttentionMetadata:
        """get_attntion_meta"""
        return self.attention_metadata

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
        sliding_window = layer.sliding_window

        if self.rope_3d:
            assert len(forward_meta.rotary_embs.shape) == 6
        else:
            assert len(forward_meta.rotary_embs.shape) == 5
            if layer.use_neox_rotary_style:
                assert forward_meta.rotary_embs.shape[0:4] == [2, 1, self.max_seq_len, 1]
                # 128 is qwen3
                # 32 is glm
                assert forward_meta.rotary_embs.shape[4] in [128, 32]

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
            # print(forward_meta.seq_lens_this_time)
            config_for_attention(
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
                forward_meta.seq_lens_this_time,
                self.buffer["block_indices"],
                self.buffer["num_blocks"],
                self.buffer["chunk_size"],
                self.buffer["max_len_tensor_cpu"],
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
            forward_meta.set_max_lengths,
            forward_meta.rotary_embs,
            getattr(layer, "qkv_bias", None),
            cache_k_scales,
            cache_v_scales,
            getattr(layer, "cache_k_out_scale", None),
            getattr(layer, "cache_v_out_scale", None),
            getattr(layer, "cache_k_zp", None),
            getattr(layer, "cache_v_zp", None),
            metadata.kv_signal_data_list[layer.layer_id],
            getattr(layer, "q_norm_weight", None),
            getattr(layer, "k_norm_weight", None),
            getattr(layer, "rms_norm_eps", 1e-6),
            getattr(layer, "cache_quant_type_str", "none"),
            layer.use_neox_rotary_style,
            self.rope_3d,
            self.max_seq_len,
            getattr(layer, "quant_max_bound", 0.0),
            getattr(layer, "quant_min_bound", 0.0),
            self.speculative_method is not None,
        )
        res = decode_append_attention(
            qkv_out,
            cache_k,
            cache_v,
            self.buffer["tmp_workspace"],
            self.buffer["tmp_m"],
            self.buffer["tmp_d"],
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.batch_id_per_token,
            forward_meta.cu_seqlens_q,
            forward_meta.block_tables,
            self.buffer["block_indices"],
            self.buffer["num_blocks"],
            self.buffer["chunk_size"],
            self.buffer["max_len_tensor_cpu"],
            forward_meta.attn_mask,
            cache_k_scales,
            cache_v_scales,
            getattr(layer, "cache_k_out_scale", None),
            getattr(layer, "cache_v_out_scale", None),
            getattr(layer, "cache_k_zp", None),
            getattr(layer, "cache_v_zp", None),
            forward_meta.attn_mask_offsets,
            getattr(layer, "sinks", None),
            getattr(layer, "cache_quant_type_str", "none"),
            self.max_seq_len,
            getattr(layer, "quant_max_bound", 0.0),
            getattr(layer, "quant_min_bound", 0.0),
            self.max_tokens_per_batch,
            self.causal,
            sliding_window,
        )
        return res
