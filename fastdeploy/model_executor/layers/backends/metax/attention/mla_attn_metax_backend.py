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

from fastdeploy.model_executor.ops.gpu import (
    decode_mla_write_cache,
    get_block_shape_and_split_kv_block,
    prefill_mla_write_cache,
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
from fastdeploy.model_executor.layers.backends.metax.attention.flash_attention_interface import (
    flash_attn_unpadded_func,
)


def yarn_get_mscale(scale=1, mscale=1):
    """ """
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


@dataclass
class MLAAttentionMetadata(AttentionMetadata):
    """
    MLAAttentionMetadata for Multi-Layer Attention
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


class MetaxMLAAttentionBackend(AttentionBackend):
    """
    MLA Attention Backend implementation.
    """

    __infer_dynamic_dims_fields__ = ["attention_metadata"]
    attention_metadata: MLAAttentionMetadata
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
        MLAAttentionBackend __init__
        """
        super().__init__()
        self.attention_metadata: MLAAttentionMetadata = None

        # 基础配置
        self.block_size: int = fd_config.cache_config.block_size
        self.max_seq_len: int = fd_config.model_config.max_model_len
        self.rope_theta: float = (
            10000.0 if fd_config.model_config.rope_theta is None else fd_config.model_config.rope_theta
        )
        self.rope_3d: bool = getattr(fd_config.model_config, "rope_3d", False)
        self.causal: bool = getattr(fd_config.model_config, "causal", True)
        self.speculative_method: str = fd_config.speculative_config.method
        self.use_speculate: bool = self.speculative_method is not None
        self.speculate_max_draft_token_num: int = fd_config.speculative_config.num_speculative_tokens
        self.keep_pd_step_flag: bool = fd_config.speculative_config.model_type == "mtp"
        self.num_layers_draft_model: int = int(fd_config.speculative_config.method in ["mtp"])

        self.kv_num_heads: int = kv_num_heads
        self.num_heads: int = num_heads
        self.group_size: int = self.num_heads // self.kv_num_heads
        self.head_dim: int = fd_config.model_config.head_dim
        self.num_layers: int = fd_config.model_config.num_hidden_layers
        self.encoder_block_shape_q: int = encoder_block_shape_q
        self.decoder_block_shape_q: int = decoder_block_shape_q

        # For Multi Head Latent Attention
        self.kv_lora_rank: int = fd_config.model_config.kv_lora_rank
        self.qk_rope_head_dim: int = fd_config.model_config.qk_rope_head_dim
        self.qk_head_dim: int = fd_config.model_config.qk_nope_head_dim + fd_config.model_config.qk_rope_head_dim
        self.attn_softmax_scale: float = self.qk_head_dim**-0.5
        if fd_config.model_config.rope_scaling:
            mscale_all_dim = fd_config.model_config.rope_scaling.get("mscale_all_dim", False)  # 1.0
            scaling_factor = fd_config.model_config.rope_scaling["factor"]  # 40
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.attn_softmax_scale = self.attn_softmax_scale * mscale * mscale

        self.pd_disaggregation_mode: str = fd_config.parallel_config.pd_disaggregation_mode

        self.start_layer_index: int = fd_config.model_config.start_layer_index
        self.device_id: int = os.getenv("CUDA_VISIBLE_DEVICES", None)

        self.rank, self.device_id = init_rank_and_device_id(fd_config)

        self.flash_attn_func = flash_attn_unpadded_func
        self.flash_attn_kwargs = {"softmax_scale": self.attn_softmax_scale}

    @paddle.no_grad()
    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize attention metadata hence all layers in the forward pass can reuse it."""
        paddle.device.empty_cache()

        metadata = MLAAttentionMetadata()
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
            forward_meta.decoder_batch_ids,  # decoder_batch_ids_per_ctax
            forward_meta.decoder_tile_ids_per_batch,  # decoder_chunk_ids_per_ctax_each_batch
            forward_meta.decoder_num_blocks_cpu,
            forward_meta.decoder_num_blocks_device,
            forward_meta.decoder_chunk_size_device,
            forward_meta.max_len_tensor_cpu,
            forward_meta.encoder_batch_ids,
            forward_meta.encoder_tile_ids_per_batch,
            forward_meta.encoder_num_blocks_x_cpu,
            forward_meta.kv_batch_ids,
            forward_meta.kv_tile_ids_per_batch,
            forward_meta.kv_num_blocks_x_cpu,
            self.encoder_block_shape_q,
            self.decoder_block_shape_q,
            self.group_size,
            self.block_size,
        )

        # MLA
        metadata.max_enc_len_this_time = forward_meta.max_len_tensor_cpu[1].item()
        metadata.max_dec_len_this_time = forward_meta.max_len_tensor_cpu[2]
        metadata.max_kv_len_this_time = forward_meta.max_len_tensor_cpu[5]

        # pd_disaggregation
        metadata.kv_signal_data_list = [None] * self.num_layers

        self.attention_metadata: AttentionMetadata = metadata

        seq_lens_decoder = forward_meta.seq_lens_decoder.squeeze(-1)
        seq_lens_this_time = forward_meta.seq_lens_this_time.squeeze(-1)
        non_zero_index = seq_lens_this_time.nonzero().flatten()
        seq_lens_decoder = seq_lens_decoder[non_zero_index]
        seq_lens_this_time = seq_lens_this_time[non_zero_index]

        self.seq_lens_this_time = list(seq_lens_this_time.cpu())
        self.seq_lens_this_time_max = max(self.seq_lens_this_time)
        self.seq_lens_this_time_min = min(self.seq_lens_this_time)
        self.seq_lens = seq_lens_decoder + seq_lens_this_time
        self.block_tables = forward_meta.block_tables[non_zero_index]

    def get_attntion_meta(self) -> AttentionMetadata:
        """get_attntion_meta"""
        return self.attention_metadata

    def get_kv_cache_shape(
        self,
        max_num_blocks: int,
        kv_cache_quant_type: str = None,
    ) -> Tuple[int, int, int, int]:
        """
        Calculate kv cache shape for MLA
        """
        return (
            max_num_blocks,
            1,
            self.block_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
        )

    def compute_flash_mla(
        self,
        query: paddle.Tensor,
        latent_cache: paddle.Tensor,
        forward_meta: ForwardMeta,
    ) -> paddle.Tensor:
        from flash_mla_paddle import flash_mla_with_kvcache, get_mla_metadata

        assert latent_cache is not None

        latent_cache = latent_cache.transpose([0, 2, 1, 3])
        seq_len_q = self.seq_lens_this_time_max
        num_heads_q = self.num_heads
        num_heads_kv = latent_cache.shape[2]
        head_dim_v = self.kv_lora_rank
        head_dim_qk = self.kv_lora_rank + self.qk_rope_head_dim

        if seq_len_q != self.seq_lens_this_time_min:
            query = paddle.stack(
                [
                    paddle.concat([x, paddle.zeros((seq_len_q - x.shape[0], x.shape[1]), dtype=x.dtype)])
                    for x in paddle.split(query, self.seq_lens_this_time)
                ]
            )

        query = query.reshape([-1, seq_len_q, num_heads_q, head_dim_qk])

        tile_scheduler_metadata, num_splits = get_mla_metadata(
            self.seq_lens, seq_len_q * num_heads_q // num_heads_kv, num_heads_kv
        )

        assert tile_scheduler_metadata.shape[0] != 0

        out = flash_mla_with_kvcache(
            query,
            latent_cache,
            self.block_tables,
            self.seq_lens,
            head_dim_v,
            tile_scheduler_metadata,
            num_splits,
            softmax_scale=self.attn_softmax_scale,
            causal=self.causal,
        )[0]

        if seq_len_q != self.seq_lens_this_time_min:
            out = paddle.concat([paddle.split(x, [n, seq_len_q - n])[0] for x, n in zip(out, self.seq_lens_this_time)])
        else:
            out = out.reshape([-1, num_heads_q, head_dim_v])

        return out

    def forward_extend(
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
        Prefill阶段的前向传播
        """
        paddle.device.empty_cache()

        metadata = self.attention_metadata

        latent_cache = forward_meta.caches[layer.layer_id] if hasattr(forward_meta, "caches") else None

        # 写入缓存
        prefill_mla_write_cache(
            compressed_kv,
            k_pe,
            latent_cache,
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.batch_id_per_token,
            forward_meta.cu_seqlens_q,
            metadata.block_tables,
            "none",
            getattr(forward_meta, "max_input_length", -1),
        )

        # Flash注意力计算
        fmha_out = self.flash_attn_func(
            q,
            k,
            v,
            forward_meta.cu_seqlens_q,
            forward_meta.cu_seqlens_k,
            metadata.max_enc_len_this_time,
            metadata.max_enc_len_this_time,
            causal=self.causal,
            **self.flash_attn_kwargs,
        )[0]

        return fmha_out

    def forward_decode(
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
        Decode阶段的前向传播
        """
        metadata = self.attention_metadata

        latent_cache = forward_meta.caches[layer.layer_id] if hasattr(forward_meta, "caches") else None

        # 获取推测解码参数
        speculate_decoder = self.speculative_method is not None

        # 写入缓存
        decode_mla_write_cache(
            compressed_kv,
            k_pe,
            latent_cache,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_encoder,
            forward_meta.batch_id_per_token,
            forward_meta.cu_seqlens_q,
            metadata.block_tables,
            "none",
            self.max_seq_len,
            speculate_decoder,
        )

        # 多头潜在注意力计算
        fmha_out = self.compute_flash_mla(q, latent_cache, forward_meta)

        return fmha_out

    @paddle.no_grad()
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
        if k is not None:
            return self.forward_extend(q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta)
        else:
            return self.forward_decode(q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta)
