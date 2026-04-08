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

import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

import paddle

from fastdeploy.platforms import current_platform

if current_platform.is_cuda():
    paddle.enable_compat(scope={"flash_mla"})

from fastdeploy.model_executor.layers.attention.ops import (
    get_block_shape_and_split_kv_block,
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


def yarn_get_mscale(scale=1, mscale=1):
    """ """
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def compute_slot_mapping(
    block_tables: paddle.Tensor,  # [num_reqs, max_blocks_per_req]
    positions: paddle.Tensor,  # [num_tokens] 每个token的位置
    batch_id_per_token: paddle.Tensor,  # [num_tokens] 每个token属于哪个请求
    block_size: int,
) -> paddle.Tensor:
    """
    计算 slot_mapping

    公式: slot = block_id * block_size + offset_in_block
    """
    # 1. 计算每个 token 对应的 block 索引
    block_idx = positions // block_size  # [num_tokens]

    # 2. 从 block_tables 中查表获取 block_id
    # block_tables[batch_id_per_token, block_idx]
    block_ids = block_tables[batch_id_per_token, block_idx]  # [num_tokens]

    # 3. 计算在 block 内的偏移
    block_offset = positions % block_size  # [num_tokens]

    # 4. 计算 slot_mapping
    slot_mapping = block_ids * block_size + block_offset

    return slot_mapping.cast(paddle.int64)


@dataclass
class DSAAttentionMetadata(AttentionMetadata):
    """
    DSAAttentionMetadata for Multi-Layer Attention
    """

    _dtype: paddle.dtype = paddle.bfloat16
    encoder_max_partition_size: int = 32768
    max_partition_size: int = 32768
    block_tables: Optional[paddle.Tensor] = None
    rotary_embs: Optional[paddle.Tensor] = None
    attn_mask: Optional[paddle.Tensor] = None
    _fuse_kernel_compute_dtype: str = "bf16"

    # pd_disaggregation
    kv_signal_metadata: Optional[paddle.Tensor] = None
    kv_signal_data_list: List[Optional[paddle.Tensor]] = field(default_factory=list)

    max_enc_len_this_time: Optional[paddle.Tensor] = None
    max_dec_len_this_time: Optional[paddle.Tensor] = None
    max_kv_len_this_time: Optional[paddle.Tensor] = None

    slot_mapping: Optional[paddle.Tensor] = None


class DSAAttentionBackend(AttentionBackend):
    """
    DSA Attention Backend implementation.
    """

    __infer_dynamic_dims_fields__ = ["attention_metadata"]
    attention_metadata: DSAAttentionMetadata
    flash_attn_func: callable = None

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
        DSAAttentionBackend __init__
        """
        super().__init__()
        self.attention_metadata: DSAAttentionMetadata = None

        # 基础配置
        self.block_size: int = fd_config.cache_config.block_size
        self.max_seq_len: int = fd_config.model_config.max_model_len
        self.rope_theta: float = (
            10000.0 if fd_config.model_config.rope_theta is None else fd_config.model_config.rope_theta
        )
        self.rope_3d: bool = fd_config.enable_rope_3d_runtime
        self.causal: bool = getattr(fd_config.model_config, "causal", True)
        self.speculative_method: str = fd_config.speculative_config.method
        self.use_speculate: bool = self.speculative_method is not None
        self.speculate_max_draft_token_num: int = fd_config.speculative_config.num_speculative_tokens
        self.keep_pd_step_flag: bool = fd_config.speculative_config.model_type == "mtp"
        self.num_layers_draft_model: int = int(fd_config.speculative_config.method in ["mtp"])

        self.num_heads: int = num_heads
        self.head_dim: int = fd_config.model_config.head_dim
        self.num_layers: int = fd_config.model_config.num_hidden_layers

        # Indexer
        self.index_head_dim = fd_config.model_config.index_head_dim
        self.index_n_heads = fd_config.model_config.index_n_heads
        self.index_topk = fd_config.model_config.index_topk
        self.quant_block_size = 128

        # For Multi Head Latent Attention
        self.kv_lora_rank: int = fd_config.model_config.kv_lora_rank
        self.qk_rope_head_dim: int = fd_config.model_config.qk_rope_head_dim
        self.qk_head_dim: int = fd_config.model_config.qk_nope_head_dim + fd_config.model_config.qk_rope_head_dim
        self.attn_softmax_scale: float = self.qk_head_dim**-0.5
        self.rope_scaling = getattr(fd_config.model_config, "rope_scaling", None)
        if self.rope_scaling:
            mscale_all_dim = fd_config.model_config.rope_scaling.get("mscale_all_dim", False)  # 1.0
            scaling_factor = fd_config.model_config.rope_scaling["factor"]  # 40
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.attn_softmax_scale = self.attn_softmax_scale * mscale * mscale

        self.pd_disaggregation_mode: str = fd_config.parallel_config.pd_disaggregation_mode

        self.start_layer_index: int = fd_config.model_config.start_layer_index
        self.device_id: int = os.getenv("CUDA_VISIBLE_DEVICES", None)

        self.rank, self.device_id = init_rank_and_device_id(fd_config)

        self.useless_tensor = paddle.randn([1]).cast("int32")

    def _cast_scale_inv_to_ue8m0(self, scales_inv: paddle.Tensor, out_dtype=paddle.float32) -> paddle.Tensor:
        return paddle.pow(2, paddle.clamp_min(scales_inv, 1e-4).log2().ceil()).to(out_dtype)

    def quantize_k_cache(
        self,
        input_k_cache: paddle.Tensor,  # (num_blocks, block_size, h_k, d)
    ) -> paddle.Tensor:
        """
        Quantize the k-cache
        For more detail about the layout of K/V, please refer to comments in flash_mla_interface.py
        """

        d, d_nope, d_rope, tile_size, num_tiles = 576, 512, 64, 128, 4
        assert input_k_cache.shape[-1] == d
        num_blocks, block_size, h_k, _ = input_k_cache.shape
        assert h_k == 1
        input_k_cache = input_k_cache.squeeze(2)  # [num_blocks, block_size, d]
        input_elem_size = input_k_cache.element_size()

        bytes_per_token = d_nope + num_tiles * 4 + input_elem_size * d_rope
        result = paddle.empty((num_blocks, block_size + 1, bytes_per_token), dtype=paddle.float8_e4m3fn)[
            :, :block_size, :
        ]
        result_k_nope_part = result[..., :d_nope]
        result_k_scale_factor = result[..., d_nope : d_nope + num_tiles * 4].view(paddle.float32)
        result_k_rope_part = result[..., d_nope + num_tiles * 4 :].view(input_k_cache.dtype)
        result_k_rope_part[:] = input_k_cache[..., d_nope:]

        for tile_idx in range(0, num_tiles):
            cur_scale_factors_inv = (
                paddle.abs(input_k_cache[..., tile_idx * tile_size : (tile_idx + 1) * tile_size])
                .max(dim=-1)
                .values.float()
                / 448.0
            )  # [num_blocks, block_size]
            cur_scale_factors_inv = self._cast_scale_inv_to_ue8m0(cur_scale_factors_inv)
            result_k_scale_factor[:, :, tile_idx] = cur_scale_factors_inv

            cur_scale_factors_inv.unsqueeze_(-1)  # [num_blocks, block_size, 1]
            cur_quantized_nope = (
                input_k_cache[..., tile_idx * tile_size : (tile_idx + 1) * tile_size].float()
                / cur_scale_factors_inv.float()
            ).to(paddle.float8_e4m3fn)
            result_k_nope_part[..., tile_idx * tile_size : (tile_idx + 1) * tile_size] = cur_quantized_nope

        result = result.view(num_blocks, block_size, 1, -1)
        return result

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize attention metadata hence all layers in the forward pass can reuse it."""
        metadata = DSAAttentionMetadata()
        metadata.max_partition_size = 32768
        metadata.encoder_max_partition_size = self.max_seq_len
        metadata._dtype = paddle.get_default_dtype()
        if metadata._dtype == "bfloat16":
            metadata._fuse_kernel_compute_dtype = "bf16"
        elif metadata._dtype == "float16":
            metadata._fuse_kernel_compute_dtype = "fp16"
        elif metadata._dtype == "float32":
            metadata._fuse_kernel_compute_dtype = "fp32"

        metadata.block_tables = forward_meta.block_tables
        metadata.rotary_embs = forward_meta.rotary_embs
        metadata.attn_mask = forward_meta.attn_mask
        metadata.pre_caches_length = forward_meta.pre_caches_length

        get_block_shape_and_split_kv_block(
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.decoder_batch_ids,
            forward_meta.decoder_tile_ids_per_batch,
            self.useless_tensor,  # not used in mla
            forward_meta.decoder_num_blocks_device,
            forward_meta.decoder_chunk_size_device,
            forward_meta.max_len_tensor_cpu,
            self.useless_tensor,  # not used in mla
            self.useless_tensor,  # not used in mla
            self.useless_tensor,  # not used in mla
            forward_meta.kv_batch_ids,
            forward_meta.kv_tile_ids_per_batch,
            forward_meta.kv_num_blocks_x_cpu,
            -1,  # not need.
            -1,  # not need.
            -1,  # not need.
            self.block_size,
        )
        # MLA
        metadata.max_enc_len_this_time = forward_meta.max_len_tensor_cpu[1]
        metadata.max_dec_len_this_time = forward_meta.max_len_tensor_cpu[2]
        metadata.max_kv_len_this_time = forward_meta.max_len_tensor_cpu[5]

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

    def get_attention_meta(self) -> AttentionMetadata:
        """get_attention_meta"""
        return self.attention_metadata

    def get_kv_cache_shape(
        self,
        max_num_blocks: int,
        kv_cache_quant_type: str = None,
    ) -> Tuple[int, int, int, int]:
        """
        Calculate kv cache shape for DSA

        see FlashMLA readme.md for details
        In the "FP8 with scale" format, each token's KV cache is 656 Bytes, structured as:
        -   **First 512 bytes:** The "quantized NoPE" part, containing 512 `float8_e4m3` values.
        -   **Next 16 bytes:** Scale factors, containing 4 `float32` values. The first `float32` is the scale for the first 128 `float8_e4m3` values, the second for the next 128, and so on.
        -   **Last 128 bytes:** The "RoPE" part, containing 64 `bfloat16` values. This part is not quantized for accuracy.

        """

        fp8_key_cahe_dim = self.kv_lora_rank + 4 * (self.kv_lora_rank // 128) + 2 * self.qk_rope_head_dim
        fp8_indexer_dim = self.index_head_dim + self.index_head_dim // self.quant_block_size * 4
        key_cache_shape = [max_num_blocks, 1, self.block_size, fp8_key_cahe_dim]
        value_cache_shape = []
        indexer_cache_shape = [max_num_blocks, self.block_size, fp8_indexer_dim]

        return key_cache_shape, value_cache_shape, indexer_cache_shape

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
        Mixed模式的前向传播
        """
        metadata = self.attention_metadata
        # speculate_decoder = self.speculative_method is not None
        # speculate_max_tokens = self.speculate_max_draft_token_num

        if self.pd_disaggregation_mode == "per_query":
            metadata.kv_signal_data_list[layer.layer_id] = init_signal_layerwise(
                metadata.kv_signal_metadata,
                layer.layer_id + self.start_layer_index,
            )

        latent_cache = forward_meta.caches[2 * layer.layer_id] if hasattr(forward_meta, "caches") else None

        if current_platform.is_cuda():
            import flash_mla

            from fastdeploy.model_executor.ops.gpu import dsk_attn_write_cache

        k_range = paddle.tensor(200.0)
        scale = paddle.abs(compressed_kv).max() / k_range

        slot_mapping = compute_slot_mapping(
            forward_meta.block_tables,
            forward_meta.position_ids,
            forward_meta.batch_id_per_token,
            self.block_size,
        )

        dsk_attn_write_cache(
            compressed_kv,
            k_pe,
            latent_cache,
            slot_mapping,
            scale.cast(paddle.float32),
            "fp8_ds_mla",
        )

        fmha_out_prefill = None
        if forward_meta.max_len_tensor_cpu[1]:  # max_enc_len_this_time

            fmha_out_prefill, _, __ = flash_mla.flash_mla_sparse_fwd(
                q,  # q_input.contiguous(),
                k,  # kv.unsqueeze(1),
                v,  # indexer_top_k.unsqueeze(1),
                sm_scale=self.attn_softmax_scale,
            )

        # Decode
        # if k is None:
        if forward_meta.max_len_tensor_cpu[2]:  # max_enc_len_this_time

            tile_scheduler_metadata, _ = flash_mla.get_mla_metadata()

            fmha_out_decode, _ = flash_mla.flash_mla_with_kvcache(
                q.unsqueeze(1).contiguous(),
                latent_cache.transpose([0, 2, 1, 3]).contiguous(),
                None,  # forward_meta.block_tables,
                None,  # cache_seqlens
                512,  # self.qk_nope_head_dim,
                tile_scheduler_metadata,
                None,  # num_splits,
                self.attn_softmax_scale,
                False,  # casual
                True,  # is_fp8_kvcache
                v,  # indices,
                None,  # t.attn_sink,
                None,  # extra_k_cache,
                None,  # extra_indices_in_kvcache: Optional[torch.Tensor] = None,
                None,  # topk_length: Optional[torch.Tensor] = None,
                None,  # extra_topk_length: Optional[torch.Tensor] = None
            )

            if fmha_out_prefill is not None:

                from fastdeploy.model_executor.ops.gpu import (
                    merge_prefill_decode_output,
                )

                merge_prefill_decode_output(
                    fmha_out_prefill,
                    fmha_out_decode,
                    forward_meta.seq_lens_encoder,
                    forward_meta.seq_lens_decoder,
                    forward_meta.seq_lens_this_time,
                    forward_meta.cu_seqlens_q,
                    self.num_heads * 4,
                    128,
                    1,
                )

                return fmha_out_prefill
            else:
                return fmha_out_decode

        return fmha_out_prefill
