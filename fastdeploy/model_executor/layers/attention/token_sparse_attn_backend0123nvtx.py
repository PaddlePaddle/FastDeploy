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
try:
    from paddle.nn.functional.flash_attention import flash_attention_v3_varlen
except:
    flash_attention_v3_varlen = None

import math

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)
from fastdeploy.model_executor.layers.attention.ops import (
    append_attention,
    flash_mask_attention,
    get_block_shape_and_split_kv_block,
    gqa_rope_write_cache,
    init_kv_signal_per_query,
    init_signal_layerwise,
    open_shm_and_get_meta_signal,
    pre_cache_len_concat,
    flash_attn_rewrite_cachekv_cuda,
    gqa_decoder_rope_norm_with_write_cache
)
from fastdeploy.model_executor.layers.attention.utils import init_rank_and_device_id

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta

from fastdeploy.platforms import current_platform

if current_platform.is_cuda():
    from fastdeploy.model_executor.ops.gpu import merge_prefill_decode_output
else:
    merge_prefill_decode_output = None

import os
import nvtx

use_fa3 = False

@dataclass
class TokenSparseAttnMetadata(AttentionMetadata):
    """
    TokenSparseAttnMetadata
    """

    rotary_embs: Optional[paddle.Tensor] = None
    block_tables: Optional[paddle.Tensor] = None

    cu_seqlens_q: paddle.Tensor = None
    cu_seqlens_k: paddle.Tensor = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0

    pre_cache_batch_ids = None
    pre_cache_tile_ids_per_batch = None
    pre_cache_num_blocks_cpu = None
    kv_token_num_cpu = None

    # pd_disaggregation
    kv_signal_metadata: Optional[paddle.Tensor] = None
    kv_signal_data_list: List[Optional[paddle.Tensor]] = field(default_factory=list)

    _fuse_kernel_compute_dtype: str = "bf16"
    _dtype: paddle.dtype = paddle.bfloat16

    max_len_tensor_cpu: paddle.Tensor = None
    max_len_tensor_cpu_decoder: paddle.Tensor = None


class TokenSparseAttnBackend(AttentionBackend):
    """
    FlashAttentionBackend backend implementation
    """

    __infer_dynamic_dims_fields__ = ["attention_metadata"]
    attention_metadata: TokenSparseAttnMetadata
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
        TokenSparseAttentionBackend __init__
        """
        super().__init__()
        self.attention_metadata: TokenSparseAttnMetadata = None
        self.block_size: int = fd_config.cache_config.block_size
        self.max_seq_len = fd_config.model_config.max_model_len
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
        self.keep_pd_step_flag: bool = fd_config.speculative_config.model_type == "mtp"
        self.num_layers_draft_model: int = int(fd_config.speculative_config.method in ["mtp"])

        self.kv_num_heads: int = kv_num_heads
        self.num_heads: int = num_heads
        self.group_size: int = self.num_heads // self.kv_num_heads
        self.head_dim: int = fd_config.model_config.head_dim
        self.num_layers: int = fd_config.model_config.num_hidden_layers
        self.max_partition_size: int = int(os.getenv("FLAGS_max_partition_size", 80000))
        self.encoder_block_shape_q: int = encoder_block_shape_q
        self.decoder_block_shape_q: int = decoder_block_shape_q

        self.pd_disaggregation_mode: str = fd_config.parallel_config.pd_disaggregation_mode

        self.start_layer_index: int = fd_config.model_config.start_layer_index


        if fd_config.parallel_config.expert_parallel_rank is None:
            fd_config.parallel_config.expert_parallel_rank = 0
        
        self.rank, self.device_id = init_rank_and_device_id(fd_config)
        self.use_output = not fd_config.graph_opt_config.full_cuda_graph
        if self.use_output:
            flag = "FLAGS_cuda_graph_blacklist"
            paddle.set_flags(
                {
                    flag: ",".join(
                        list(
                            set(
                                paddle.get_flags(flag)[flag].split(",")
                                + ["custom_op.static_op_append_attention_with_output_"]
                            )
                        )
                    )
                }
            )
        self.fd_config = fd_config


        self.tsa_index_head_dim = fd_config.model_config.tsa_index_head_dim
        self.tsa_index_key_heads = fd_config.model_config.tsa_index_key_heads
        self.tsa_index_topk = fd_config.model_config.tsa_index_topk

        self.zero_seq_enc_lens_for_decode = paddle.zeros(
            shape=[fd_config.scheduler_config.max_num_seqs, 1], dtype=paddle.int32
        )

        if self.flash_attn_func is None:
            prop = paddle.device.cuda.get_device_properties()
            cc = prop.major * 10 + prop.minor
            is_current_sm_supported = cc >= 90
            is_paddle_supported = any(num >= 90 for num in paddle.version.cuda_archs())
            if is_current_sm_supported and is_paddle_supported:
                self.flash_attn_func = flash_attention_v3_varlen
                print("The current platform supports Flash Attention V3.")
                self.flash_attn_kwargs = {}
            else:
                self.flash_attn_func = flash_attn_unpadded
                self.flash_attn_kwargs = {"scale": self.head_dim**-0.5, "training": False}
                print(
                    "The current platform does not support Flash Attention V3, so Flash Attention V2 will be used instead."
                )
        self.attn_outputsize_tp = self.num_heads * self.head_dim

        #page
        self.key_new = paddle.zeros(shape=[(fd_config.scheduler_config.max_num_seqs)*((self.tsa_index_topk)//self.block_size), self.kv_num_heads, self.block_size, self.head_dim], dtype="bfloat16")
        self.value_new = paddle.zeros(shape=[(fd_config.scheduler_config.max_num_seqs)*((self.tsa_index_topk)//self.block_size), self.kv_num_heads, self.block_size, self.head_dim], dtype="bfloat16")
        
        if self.num_heads // self.kv_num_heads == 14:
            self.q_share_tokens = 9
        elif self.num_heads // self.kv_num_heads == 8:
            self.q_share_tokens = 8
        self.k_share_tokens = 8

        # self.offsets = paddle.zeros([self.max_seq_len],dtype="int32")
        # self.buffer = paddle.zeros([2048 * 2048], dtype=paddle.uint8)
        self.tsa_index_topk = 4096
        # self.tmp = paddle.arange(0,self.max_seq_len,dtype="int32").reshape(-1,8)
        # self.lengths = paddle.zeros([self.max_seq_len],dtype="int32")
        # self.tsa_indexer = TSAIndexer(fd_config)

    def get_attntion_meta(self):
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
        value_cache_shape = [max_num_blocks, self.kv_num_heads, self.block_size, self.head_dim]
        indexer_k_cache_shape = (max_num_blocks,self.tsa_index_key_heads,self.block_size,self.tsa_index_head_dim)
        if kv_cache_quant_type is not None and kv_cache_quant_type == "int4_zp":
            key_cache_shape = [
                max_num_blocks,
                self.kv_num_heads,
                self.block_size,
                self.head_dim // 2,
            ]
            value_cache_shape = [
                max_num_blocks,
                self.kv_num_heads,
                self.block_size,
                self.head_dim // 2,
            ]
        return key_cache_shape, value_cache_shape, indexer_k_cache_shape

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        metadata = TokenSparseAttnMetadata()

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

        metadata.max_partition_size = self.max_partition_size
        metadata.encoder_max_partition_size = self.max_seq_len
        metadata._dtype = paddle.get_default_dtype()
        if metadata._dtype == "bfloat16":
            metadata._fuse_kernel_compute_dtype = "bf16"
        elif metadata._dtype == "float16":
            metadata._fuse_kernel_compute_dtype = "fp16"
        elif metadata._dtype == "float32":
            metadata._fuse_kernel_compute_dtype = "fp32"

        get_block_shape_and_split_kv_block(
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.decoder_batch_ids,
            forward_meta.decoder_tile_ids_per_batch,
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

        (
            forward_meta.attn_cu_seqlens_k,
            forward_meta.pre_cache_batch_ids,
            forward_meta.pre_cache_tile_ids_per_batch,
            forward_meta.pre_cache_num_blocks_cpu,
            forward_meta.kv_token_num_cpu,
        ) = pre_cache_len_concat(
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.max_len_tensor_cpu[2],
            self.block_size,
        )

        metadata.max_len_tensor_cpu_decoder = paddle.clone(forward_meta.max_len_tensor_cpu)
        metadata.max_len_tensor_cpu_decoder[1] = 0

        self.attention_metadata = metadata

        #====================sparse indexer=============================
        self.sparse_max_len_tensor_cpu_decoder = paddle.clone(metadata.max_len_tensor_cpu_decoder)
        self.sparse_max_len_tensor_cpu_decoder[2] = self.tsa_index_topk-1 if self.sparse_max_len_tensor_cpu_decoder[2]> (self.tsa_index_topk-1) else self.sparse_max_len_tensor_cpu_decoder[2]
        self.sparse_max_len_tensor_cpu_decoder[3] = self.tsa_index_topk if self.sparse_max_len_tensor_cpu_decoder[3]>self.tsa_index_topk else self.sparse_max_len_tensor_cpu_decoder[3]
        self.sparse_max_len_tensor_cpu_decoder[4] = self.sparse_max_len_tensor_cpu_decoder[2]
        self.sparse_max_len_tensor_cpu_decoder[5] = self.sparse_max_len_tensor_cpu_decoder[3]
        
        tmp = paddle.clone(forward_meta.seq_lens_decoder)
        tmp = paddle.clamp(tmp,max=self.tsa_index_topk-1)
        forward_meta.sparse_seq_lens_decoder.copy_(tmp, False)

        tmp2 = paddle.clone(forward_meta.attn_mask_offsets)
        tmp2 = paddle.clamp(tmp2,max=self.tsa_index_topk)
        forward_meta.sparse_attn_mask_offsets.copy_(tmp2, False)
        #====================sparse indexer=============================



    def indexer_extend_forward(self, forward_meta: ForwardMeta, query_states, key_states):
        with nvtx.annotate("pack_q_head", color="blue"):
            q = query_states.reshape(-1,self.kv_num_heads,self.num_heads//4,self.head_dim)
            q = paddle.mean(q, axis=2).reshape(-1,self.kv_num_heads,self.head_dim)

        with nvtx.annotate("pack_k_token", color="yellow"):
            k = paddle.nn.functional.avg_pool1d(
                key_states.transpose([1, 2, 0]).contiguous(), 
                kernel_size=self.k_share_tokens, 
                stride=self.k_share_tokens, 
                ceil_mode=True
            )#.transpose([2, 0, 1])
        
        #score = paddle.matmul(q.transpose([1,0,2]),k.transpose([1,2,0]))
        with nvtx.annotate("gemm_q*k", color="green"):
            score = paddle.einsum("shd,hdm->hsm", q, k)
            score[..., -1] = 1e10
            score[..., :64//self.k_share_tokens] = 1e10
        
        with nvtx.annotate("pack_q_token", color="red"):
            k_pack_seq = math.ceil(key_states.shape[0]/self.k_share_tokens)
            score_qzip = paddle.nn.functional.max_pool1d(
                score.transpose([0, 2, 1]).contiguous(), 
                kernel_size=self.q_share_tokens, 
                stride=self.q_share_tokens, 
                ceil_mode=True
            ).transpose([2, 0, 1]).reshape([-1,k_pack_seq])

        with nvtx.annotate("topk", color="yellow"):
            token_sparse_index_packk = paddle.full([score_qzip.shape[0], self.tsa_index_topk//self.k_share_tokens], -1,dtype='int32')

            from fastdeploy.model_executor.ops.gpu import radix_topk_ragged_transform
            ks,ke = forward_meta.attn_mask_offsets[::2].contiguous(),forward_meta.attn_mask_offsets[1::2].contiguous()
            
            q_div = query_states.shape[0] // self.q_share_tokens
            mask = paddle.zeros([score_qzip.shape[0]//self.kv_num_heads],dtype="int32")
            mask[:q_div] = (ke[:q_div*self.q_share_tokens].reshape([-1,self.q_share_tokens])[:,-1] +self.k_share_tokens-1)//self.k_share_tokens
            mask[-1] = (ke[-1]+self.k_share_tokens-1)//self.k_share_tokens


            radix_topk_ragged_transform(
                score_qzip, 
                token_sparse_index_packk, 
                ks,#self.offsets,
                mask.contiguous(),#self.lengths,
                None,#forward_meta.seq_lens_decoder,
                None,#forward_meta.batch_id_per_token,
                None,#self.buffer
                self.tsa_index_topk//self.k_share_tokens
            )
        with nvtx.annotate("trick_-1", color="blue"):
            #NOTE: attention need  index >=0 
            token_sparse_index_packk = paddle.clip(token_sparse_index_packk,min=0)
        
        with nvtx.annotate("depack_k", color="red"):
            token_sparse_index = (
                token_sparse_index_packk * self.k_share_tokens
            ).unsqueeze(-1) + paddle.arange(self.k_share_tokens,dtype="int32")
        
        # breakpoint()
        # tmp = paddle.topk(score_qzip, self.tsa_index_topk//self.k_share_tokens,axis =-1)[1]
        # paddle.set_printoptions(precision=4, threshold=160, edgeitems=40, sci_mode=None, linewidth=80)
        # token_sparse_index = paddle.repeat_interleave(token_sparse_index,[1,4])
        
        token_sparse_index = token_sparse_index.reshape([-1,self.kv_num_heads,self.tsa_index_topk])
        
        
        return token_sparse_index



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
        token_sparse_index: paddle.Tensor,
    ):
        # token_sparse_index = forward_meta.token_sparse_index
        metadata = self.attention_metadata

        if self.pd_disaggregation_mode == "per_query":
            metadata.kv_signal_data_list[layer.layer_id] = init_signal_layerwise(
                metadata.kv_signal_metadata,
                layer.layer_id + self.start_layer_index,
            )

        if forward_meta.max_len_tensor_cpu[1] > 0:
            q, k, v, _ = gqa_rope_write_cache(
                qkv,
                forward_meta.caches[3 * layer.layer_id],
                forward_meta.caches[3 * layer.layer_id + 1],
                forward_meta.cu_seqlens_q,
                forward_meta.attn_cu_seqlens_k,
                forward_meta.rotary_embs,
                forward_meta.seq_lens_this_time,
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
                forward_meta.batch_id_per_token,
                forward_meta.block_tables,
                forward_meta.kv_batch_ids,
                forward_meta.kv_tile_ids_per_batch,
                forward_meta.kv_num_blocks_x_cpu,
                forward_meta.pre_cache_batch_ids,
                forward_meta.pre_cache_tile_ids_per_batch,
                forward_meta.pre_cache_num_blocks_cpu,
                getattr(layer, "q_norm_weight", None),
                getattr(layer, "k_norm_weight", None),
                getattr(layer, "cache_k_scale", None),
                getattr(layer, "cache_v_scale", None),
                getattr(layer, "cache_k_out_scale", None),
                getattr(layer, "cache_v_out_scale", None),
                getattr(layer, "cache_k_zp", None),
                getattr(layer, "cache_v_zp", None),
                metadata.kv_signal_data_list[layer.layer_id],
                forward_meta.kv_token_num_cpu[0].item(),
                self.max_seq_len,
                getattr(layer, "rms_norm_eps", 1e-6),
                layer.use_neox_rotary_style,
                getattr(layer, "cache_quant_type_str", "none"),
                self.rope_3d,
            )

            if (os.getenv("FD_ATTENTION_BACKEND", False)) == 'TSA_ATTN' and layer.layer_id > 10:
                with nvtx.annotate("indexer", color="blue"):
                    token_sparse_index = self.indexer_extend_forward(forward_meta,q,k)
                with nvtx.annotate("sparse attention", color="blue"):
                    import sparse_attn
                    res_encoder = paddle.zeros([q.shape[0], self.num_heads * self.head_dim], dtype=q.dtype)
                    sparse_attn.flash_attn_fwd(
                        q,
                        k,
                        v,
                        forward_meta.cu_seqlens_q,
                        forward_meta.cu_seqlens_k,
                        token_sparse_index,
                        res_encoder,
                        int(self.q_share_tokens)
                    )
                    res_encoder = res_encoder.reshape([-1, self.attn_outputsize_tp])
            else:
                with nvtx.annotate("full_attention", color="blue"):
                    res_encoder = self.flash_attn_func(
                        q,
                        k,
                        v,
                        forward_meta.cu_seqlens_q,
                        forward_meta.cu_seqlens_q,
                        max_seqlen_q=forward_meta.max_len_tensor_cpu[0],
                        max_seqlen_k=forward_meta.max_len_tensor_cpu[3],
                        causal=self.causal,
                        **self.flash_attn_kwargs,
                    )[0].reshape([-1, self.attn_outputsize_tp])
            
            
            # breakpoint()
            # sparse_attn.flash_attn_fwd(q,k,v,forward_meta.cu_seqlens_q,forward_meta.cu_seqlens_k,token_sparse_index+8,out1,int(share_tokens))
            # if paddle.isnan(res_encoder).sum().item() > 0:
            #     print("error_text_sample_states is Nan")

            #NOTE (changwenbin) We want to use FlashMask, but there's a bug in the kernel right now.
            # res_encoder = paddle.zeros([qkv.shape[0], self.num_heads * self.head_dim], dtype=qkv.dtype)
            # flash_mask_attention(
            #     q,
            #     k,
            #     v,
            #     forward_meta.cu_seqlens_q,
            #     forward_meta.attn_cu_seqlens_k,
            #     forward_meta.seq_lens_encoder,
            #     res_encoder,
            #     forward_meta.attn_mask_offsets,
            #     self.num_heads,
            #     self.kv_num_heads,
            #     self.head_dim,
            #     self.max_seq_len,
            #     q.shape[0],
            #     k.shape[0],
            # )

            #NOTE (changwenbin) This is the current full attention.
            # res_encoder1 = self.flash_attn_func(
            #     q,
            #     k,
            #     v,
            #     forward_meta.cu_seqlens_q,
            #     forward_meta.cu_seqlens_q,#forward_meta.attn_cu_seqlens_k,
            #     max_seqlen_q=forward_meta.max_len_tensor_cpu[0],
            #     max_seqlen_k=forward_meta.max_len_tensor_cpu[3],
            #     causal=self.causal,
            #     **self.flash_attn_kwargs,
            # )[0].reshape([-1, self.attn_outputsize_tp])
            # breakpoint()

            # res_encoder = self.flash_attn_func(q,k,v,forward_meta.cu_seqlens_q,forward_meta.cu_seqlens_q,max_seqlen_q=forward_meta.max_len_tensor_cpu[0],max_seqlen_k=forward_meta.max_len_tensor_cpu[3],causal=self.causal,**self.flash_attn_kwargs,)[0].reshape([-1, self.attn_outputsize_tp])
        
        if  forward_meta.max_len_tensor_cpu[2]>0: 
            if token_sparse_index is not None:
                breakpoint()
                gqa_decoder_rope_norm_with_write_cache(
                    qkv,
                    forward_meta.caches[3 * layer.layer_id],
                    forward_meta.caches[3 * layer.layer_id + 1],
                    self.zero_seq_enc_lens_for_decode, # batch=1的情况，如果没有batch在decode，这个会返回空
                    forward_meta.seq_lens_decoder,
                    forward_meta.seq_lens_this_time,
                    forward_meta.batch_id_per_token,
                    forward_meta.cu_seqlens_q,
                    forward_meta.block_tables,
                    forward_meta.rotary_embs,
                    layer.qkv_bias,
                    layer.qkv_scale,
                    getattr(layer, "cache_k_scale", None),
                    getattr(layer, "cache_v_scale", None),
                    getattr(layer, "cache_k_out_scale", None),
                    getattr(layer, "cache_v_out_scale", None),
                    getattr(layer, "cache_k_zp", None),
                    getattr(layer, "cache_v_zp", None),
                    metadata.kv_signal_data_list[layer.layer_id],
                    getattr(layer, "q_norm_weight", None),
                    getattr(layer, "k_norm_weight", None),
                    getattr(layer, "rms_norm_eps", 1e-6),
                    metadata._fuse_kernel_compute_dtype,
                    getattr(layer, "cache_quant_type_str", "none"),
                    layer.use_neox_rotary_style,
                    self.rope_3d,
                    self.max_seq_len,
                    self.speculative_method is not None,
                )

                flash_attn_rewrite_cachekv_cuda(
                    forward_meta.caches[3 * layer.layer_id],
                    forward_meta.caches[3 * layer.layer_id + 1],
                    self.key_new,
                    self.value_new,
                    token_sparse_index,
                    forward_meta.block_tables,
                    forward_meta.seq_lens_decoder,
                    forward_meta.cu_seqlens_q
                )

                res_decoder = append_attention(
                    qkv,
                    self.key_new,#forward_meta.caches[3 * layer.layer_id],
                    self.value_new,#forward_meta.caches[3 * layer.layer_id + 1],
                    self.zero_seq_enc_lens_for_decode, # batch=1的情况，如果没有batch在decode，这个会返回空
                    forward_meta.sparse_seq_lens_decoder,
                    forward_meta.seq_lens_this_time,
                    forward_meta.batch_id_per_token,
                    forward_meta.cu_seqlens_q,
                    forward_meta.sparse_block_tables,#forward_meta.block_tables,
                    forward_meta.encoder_batch_ids,
                    forward_meta.encoder_tile_ids_per_batch,
                    forward_meta.encoder_num_blocks_x_cpu,
                    forward_meta.kv_batch_ids,
                    forward_meta.kv_tile_ids_per_batch,
                    forward_meta.kv_num_blocks_x_cpu,
                    forward_meta.decoder_batch_ids,  # from buffer
                    forward_meta.decoder_tile_ids_per_batch,  # from buffer
                    forward_meta.decoder_num_blocks_cpu,
                    self.sparse_max_len_tensor_cpu_decoder,
                    None,#forward_meta.rotary_embs,
                    forward_meta.attn_mask,
                    layer.qkv_bias,
                    layer.qkv_scale,
                    getattr(layer, "cache_k_scale", None),
                    getattr(layer, "cache_v_scale", None),
                    getattr(layer, "cache_k_out_scale", None),
                    getattr(layer, "cache_v_out_scale", None),
                    getattr(layer, "cache_k_zp", None),
                    getattr(layer, "cache_v_zp", None),
                    layer.linear_shift,
                    layer.linear_smooth,
                    forward_meta.sparse_attn_mask_offsets,
                    metadata.kv_signal_data_list[layer.layer_id],
                    None,#getattr(layer, "q_norm_weight", None),
                    None,#getattr(layer, "k_norm_weight", None),
                    getattr(layer, "sinks", None),
                    getattr(layer, "rms_norm_eps", 1e-6),
                    metadata._fuse_kernel_compute_dtype,
                    getattr(layer, "cache_quant_type_str", "none"),
                    layer.use_neox_rotary_style,
                    self.rope_3d,
                    self.max_seq_len,
                    getattr(layer, "quant_max_bound", 0.0),
                    getattr(layer, "quant_min_bound", 0.0),
                    getattr(layer, "out_scale", -1.0),
                    self.encoder_block_shape_q,
                    self.decoder_block_shape_q,
                    self.max_partition_size,
                    self.max_seq_len,
                    self.speculate_max_draft_token_num + 1,
                    self.causal,
                    self.speculative_method is not None,
                )
                # print("res_decoder.shape",res_decoder.shape)
                # res_decoder = qkv.reshape(-1,40,128)[:,:32,:].reshape(-1,32*128)
                
            else:
                res_decoder = append_attention(
                    qkv,
                    forward_meta.caches[3 * layer.layer_id],
                    forward_meta.caches[3 * layer.layer_id + 1],
                    self.zero_seq_enc_lens_for_decode, # batch=1的情况，如果没有batch在decode，这个会返回空
                    forward_meta.seq_lens_decoder,
                    forward_meta.seq_lens_this_time,
                    forward_meta.batch_id_per_token,
                    forward_meta.cu_seqlens_q,
                    forward_meta.block_tables,
                    forward_meta.encoder_batch_ids,
                    forward_meta.encoder_tile_ids_per_batch,
                    forward_meta.encoder_num_blocks_x_cpu,
                    forward_meta.kv_batch_ids,
                    forward_meta.kv_tile_ids_per_batch,
                    forward_meta.kv_num_blocks_x_cpu,
                    forward_meta.decoder_batch_ids,  # from buffer
                    forward_meta.decoder_tile_ids_per_batch,  # from buffer
                    forward_meta.decoder_num_blocks_cpu,
                    metadata.max_len_tensor_cpu_decoder,
                    forward_meta.rotary_embs,
                    forward_meta.attn_mask,
                    layer.qkv_bias,
                    layer.qkv_scale,
                    getattr(layer, "cache_k_scale", None),
                    getattr(layer, "cache_v_scale", None),
                    getattr(layer, "cache_k_out_scale", None),
                    getattr(layer, "cache_v_out_scale", None),
                    getattr(layer, "cache_k_zp", None),
                    getattr(layer, "cache_v_zp", None),
                    layer.linear_shift,
                    layer.linear_smooth,
                    forward_meta.attn_mask_offsets,
                    metadata.kv_signal_data_list[layer.layer_id],
                    getattr(layer, "q_norm_weight", None),
                    getattr(layer, "k_norm_weight", None),
                    getattr(layer, "sinks", None),
                    getattr(layer, "rms_norm_eps", 1e-6),
                    metadata._fuse_kernel_compute_dtype,
                    getattr(layer, "cache_quant_type_str", "none"),
                    layer.use_neox_rotary_style,
                    self.rope_3d,
                    self.max_seq_len,
                    getattr(layer, "quant_max_bound", 0.0),
                    getattr(layer, "quant_min_bound", 0.0),
                    getattr(layer, "out_scale", -1.0),
                    self.encoder_block_shape_q,
                    self.decoder_block_shape_q,
                    self.max_partition_size,
                    self.max_seq_len,
                    self.speculate_max_draft_token_num + 1,
                    self.causal,
                    self.speculative_method is not None,
                )

            if forward_meta.max_len_tensor_cpu[1] > 0:
                merge_prefill_decode_output(
                    res_encoder,
                    res_decoder,
                    forward_meta.seq_lens_encoder,
                    forward_meta.seq_lens_decoder,
                    forward_meta.seq_lens_this_time,
                    forward_meta.cu_seqlens_q,
                    self.num_heads,
                    self.head_dim,
                    self.speculate_max_draft_token_num + 1,
                )
                return res_encoder
            else:
                return res_decoder
        
        return res_encoder
