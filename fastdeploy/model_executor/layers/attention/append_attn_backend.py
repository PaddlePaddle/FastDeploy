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

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import paddle

from fastdeploy.model_executor.layers.attention.ops import (
    append_attention,
    append_attention_with_output,
    get_block_shape_and_split_kv_block,
    init_kv_signal_per_query,
    init_signal_layerwise,
    open_shm_and_get_meta_signal,
)

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta

import numpy as np

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)
from fastdeploy.model_executor.layers.attention.utils import init_rank_and_device_id

from fastdeploy.model_executor.debug_utils import full_debug_print

@dataclass
class AppendAttentionMetadata(AttentionMetadata):
    """
    AppendAttentionMetadata
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


def allocate_launch_related_buffer(
    max_batch_size,
    max_model_len,
    encoder_block_shape_q,
    decoder_block_shape_q,
    decoder_step_token_num,
    num_heads,
    kv_num_heads,
    block_size,
):
    # Initialize AttentionBackend buffers
    group_size = np.ceil(num_heads / kv_num_heads)

    # NOTE: (changwenbin) When using auto_chunk,
    # decode_max_tile_size must take into account the maximum case, where *1024 can cover 128K.
    decode_max_tile_size = (
        1024 * max_batch_size * np.ceil((decoder_step_token_num * group_size) / decoder_block_shape_q)
    )
    encode_max_tile_size = max_batch_size * np.ceil((max_model_len * group_size) / encoder_block_shape_q)
    kv_max_tile_size = max_batch_size * np.ceil(max_model_len / block_size)
    res = {}
    res["decoder_batch_ids"] = paddle.full([int(decode_max_tile_size)], 0, dtype="int32")
    res["decoder_tile_ids_per_batch"] = paddle.full([int(decode_max_tile_size)], 0, dtype="int32")
    res["decoder_num_blocks_cpu"] = paddle.full([1], 0, dtype="int32").pin_memory()
    # NOTE: (changwenbin) MLA kernel only needs decoder_num_blocks_device in place of GPU tensor,
    # adapted to cudagraph.
    res["decoder_num_blocks_device"] = paddle.full([1], 0, dtype="int32")
    res["decoder_chunk_size_device"] = paddle.full([1], 64, dtype="int32")
    res["max_len_tensor_cpu"] = paddle.full([9], 0, dtype="int32").cpu()

    res["encoder_batch_ids"] = paddle.full([int(encode_max_tile_size)], 0, dtype="int32")
    res["encoder_tile_ids_per_batch"] = paddle.full([int(encode_max_tile_size)], 0, dtype="int32")
    res["encoder_num_blocks_x_cpu"] = paddle.full([1], 0, dtype="int32").cpu()

    res["kv_batch_ids"] = paddle.full([int(kv_max_tile_size)], 0, dtype="int32")
    res["kv_tile_ids_per_batch"] = paddle.full([int(kv_max_tile_size)], 0, dtype="int32")
    res["kv_num_blocks_x_cpu"] = paddle.full([1], 0, dtype="int32").cpu()

    return res


class AppendAttentionBackend(AttentionBackend):
    """
    AppendAttentionBackend backend implementation.
    """

    __infer_dynamic_dims_fields__ = ["attention_metadata"]
    attention_metadata: AppendAttentionMetadata

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
        self.attention_metadata: AppendAttentionMetadata = None
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
        self.keep_pd_step_flag: bool = fd_config.speculative_config.model_type == "mtp"
        self.num_layers_draft_model: int = int(fd_config.speculative_config.method in ["mtp"])

        self.kv_num_heads: int = kv_num_heads
        self.num_heads: int = num_heads
        self.group_size: int = self.num_heads // self.kv_num_heads
        self.head_dim: int = fd_config.model_config.head_dim
        self.num_layers: int = fd_config.model_config.num_hidden_layers
        self.max_partition_size: int = int(os.getenv("FLAGS_max_partition_size", 1024))
        self.encoder_block_shape_q: int = encoder_block_shape_q
        self.decoder_block_shape_q: int = decoder_block_shape_q

        self.pd_disaggregation_mode: str = fd_config.parallel_config.pd_disaggregation_mode

        self.start_layer_index: int = fd_config.model_config.start_layer_index

        if fd_config.parallel_config.expert_parallel_rank is None:
            fd_config.parallel_config.expert_parallel_rank = 0

        self.rank, self.device_id = init_rank_and_device_id(fd_config)
        self.use_output = not fd_config.graph_opt_config.full_cuda_graph
        self.fd_config = fd_config

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize attntion metadata hence all layers in the forward pass can reuse it."""
        metadata = AppendAttentionMetadata()
        metadata.max_partition_size = self.max_partition_size
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
        
        # full_debug_print(None, "C++ OP get_block_shape_and_split_kv_block", header="OBSERVATION POINT 3: C++ OP CALL")
        
        # # 打印C++ OP的所有输入
        # full_debug_print(forward_meta.seq_lens_encoder, "INPUT: seq_lens_encoder")
        # full_debug_print(forward_meta.seq_lens_decoder, "INPUT: seq_lens_decoder")
        # full_debug_print(forward_meta.seq_lens_this_time, "INPUT: seq_lens_this_time")
        # full_debug_print(self.encoder_block_shape_q, "ATTR: encoder_block_shape_q")
        # full_debug_print(self.decoder_block_shape_q, "ATTR: decoder_block_shape_q")
        # full_debug_print(self.group_size, "ATTR: group_size")
        # full_debug_print(self.block_size, "ATTR: block_size")

        # get_block_shape_and_split_kv_block(
        #     forward_meta.seq_lens_encoder,
        #     forward_meta.seq_lens_decoder,
        #     forward_meta.seq_lens_this_time,
        #     forward_meta.decoder_batch_ids,
        #     forward_meta.decoder_tile_ids_per_batch,
        #     forward_meta.decoder_num_blocks_cpu,
        #     forward_meta.decoder_num_blocks_device,
        #     forward_meta.decoder_chunk_size_device,
        #     forward_meta.max_len_tensor_cpu,
        #     forward_meta.encoder_batch_ids,
        #     forward_meta.encoder_tile_ids_per_batch,
        #     forward_meta.encoder_num_blocks_x_cpu,
        #     forward_meta.kv_batch_ids,
        #     forward_meta.kv_tile_ids_per_batch,
        #     forward_meta.kv_num_blocks_x_cpu,
        #     self.encoder_block_shape_q,
        #     self.decoder_block_shape_q,
        #     self.group_size,
        #     self.block_size,
        #     self.speculate_max_draft_token_num + 1,
        # )
        # # 打印C++ OP的所有输出 (修改后的值)
        # full_debug_print(forward_meta.encoder_num_blocks_x_cpu, "OUTPUT: encoder_num_blocks_x_cpu")
        # full_debug_print(forward_meta.max_len_tensor_cpu, "OUTPUT: max_len_tensor_cpu")

        # pd_disaggregation
        metadata.kv_signal_data_list = [None] * self.num_layers
        if self.pd_disaggregation_mode == "per_chunk":
            if not self.keep_pd_step_flag:
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
        
        
        try:
            from paddleformers.utils.log import logger
            is_prefill = int(paddle.max(forward_meta.seq_lens_encoder).item()) > 0
            if is_prefill: # 只在 prefill 阶段打印
                logger.info("\n\n" + "="*30)
                logger.info("GOLDEN METADATA DUMP (FROM OLD VERSION)")
                logger.info("="*30)
                
                def log_tensor_as_list(tensor, name):
                    if tensor is not None:
                        # 使用 .numpy() 获取 CPU 上的值
                        logger.info(f"'{name}': {tensor.numpy().flatten().tolist()},")
                    else:
                        logger.info(f"'{name}': None,")

                log_tensor_as_list(forward_meta.encoder_num_blocks_x_cpu, "encoder_num_blocks_x_cpu")
                log_tensor_as_list(forward_meta.kv_num_blocks_x_cpu, "kv_num_blocks_x_cpu")
                log_tensor_as_list(forward_meta.decoder_num_blocks_cpu, "decoder_num_blocks_cpu")
                log_tensor_as_list(forward_meta.max_len_tensor_cpu, "max_len_tensor_cpu")
                log_tensor_as_list(forward_meta.max_len_kv_cpu, "max_len_kv_cpu")
                
                # (可选，但最好有) 如果 encoder_batch_ids 等张量尺寸不大，也打印出来
                log_tensor_as_list(forward_meta.encoder_batch_ids, "encoder_batch_ids")
                log_tensor_as_list(forward_meta.encoder_tile_ids_per_batch, "encoder_tile_ids_per_batch")
                
                logger.info("="*30 + "\n\n")
        except Exception as e:
            # 忽略可能的打印错误
            pass

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
        value_cache_shape = [max_num_blocks, self.kv_num_heads, self.block_size, self.head_dim]
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

        # ==================== [START] 全面 Debug 打印代码 ====================
        is_capturing_cudagraph = forward_meta.step_use_cudagraph

        if not is_capturing_cudagraph and layer.layer_id == 7: # 只打印我们关心的GQA层
            try:
                from fastdeploy.model_executor.models.minimax_m1 import print_tensor_stats
                from paddleformers.utils.log import logger
            except ImportError:
                import logging
                logger = logging.getLogger(__name__)
                def print_tensor_stats(tensor, name):
                    if tensor is not None:
                        logger.info(f"--- [FD DEBUG] {name} Shape: {tensor.shape}, DType: {tensor.dtype} ---")
                    else:
                        logger.info(f"--- [FD DEBUG] {name} is None ---")
            
            logger.info(f"\n{'='*25}\n[FD DEBUG] DETAILED DUMP for append_attention @ Layer {layer.layer_id}\n{'='*25}")

            # 1. 打印 atención 模块的配置参数
            logger.info(f"--- [FD DEBUG] Attention Config ---")
            logger.info(f"  - use_neox_rotary_style: {layer.use_neox_rotary_style}")
            logger.info(f"  - sliding_window: {layer.sliding_window}")
            logger.info(f"  - causal: {self.causal}")
            logger.info(f"  - speculative_method is not None: {self.speculative_method is not None}")
            logger.info(f"  - head_dim: {self.head_dim}, num_heads: {self.num_heads}, num_kv_heads: {self.kv_num_heads}")
            logger.info(f"--------------------------\n")
            
            # 2. 打印核心输入张量 (qkv)
            print_tensor_stats(qkv, f"FD_L{layer.layer_id}_INPUT:qkv_combined")

            # 3. 打印 KV Cache 相关张量
            cache_k = forward_meta.caches[2 * layer.layer_id]
            cache_v = forward_meta.caches[2 * layer.layer_id + 1]
            print_tensor_stats(cache_k, f"FD_L{layer.layer_id}_INPUT:cache_k")
            print_tensor_stats(cache_v, f"FD_L{layer.layer_id}_INPUT:cache_v")
            print_tensor_stats(forward_meta.block_tables, f"FD_L{layer.layer_id}_META:block_tables")

            # 4. 打印序列长度和位置信息
            print_tensor_stats(forward_meta.seq_lens_encoder, f"FD_L{layer.layer_id}_META:seq_lens_encoder")
            print_tensor_stats(forward_meta.seq_lens_decoder, f"FD_L{layer.layer_id}_META:seq_lens_decoder")
            print_tensor_stats(forward_meta.seq_lens_this_time, f"FD_L{layer.layer_id}_META:seq_lens_this_time")
            print_tensor_stats(forward_meta.batch_id_per_token, f"FD_L{layer.layer_id}_META:batch_id_per_token")
            print_tensor_stats(forward_meta.cu_seqlens_q, f"FD_L{layer.layer_id}_META:cu_seqlens_q")

            # 5. 打印 RoPE 查找表
            metadata = self.attention_metadata
            print_tensor_stats(metadata.rotary_embs, f"FD_L{layer.layer_id}_INPUT:rotary_embs_table")

            # 6. 打印用于 Kernel 内部计算的 Tile/Block 划分信息 (非常重要！)
            print_tensor_stats(forward_meta.encoder_batch_ids, f"FD_L{layer.layer_id}_META:encoder_batch_ids")
            print_tensor_stats(forward_meta.encoder_tile_ids_per_batch, f"FD_L{layer.layer_id}_META:encoder_tile_ids_per_batch")
            print_tensor_stats(forward_meta.encoder_num_blocks_x_cpu, f"FD_L{layer.layer_id}_META:encoder_num_blocks_x_cpu")
            
            print_tensor_stats(forward_meta.kv_batch_ids, f"FD_L{layer.layer_id}_META:kv_batch_ids")
            print_tensor_stats(forward_meta.kv_tile_ids_per_batch, f"FD_L{layer.layer_id}_META:kv_tile_ids_per_batch")
            print_tensor_stats(forward_meta.kv_num_blocks_x_cpu, f"FD_L{layer.layer_id}_META:kv_num_blocks_x_cpu")

            print_tensor_stats(forward_meta.decoder_batch_ids, f"FD_L{layer.layer_id}_META:decoder_batch_ids")
            print_tensor_stats(forward_meta.decoder_tile_ids_per_batch, f"FD_L{layer.layer_id}_META:decoder_tile_ids_per_batch")
            print_tensor_stats(forward_meta.decoder_num_blocks_cpu, f"FD_L{layer.layer_id}_META:decoder_num_blocks_cpu")

            # 7. 打印 max_len_tensor_cpu (非常重要！)
            print_tensor_stats(forward_meta.max_len_tensor_cpu, f"FD_L{layer.layer_id}_META:max_len_tensor_cpu")
            
            logger.info(f"\n{'='*25}\n[FD DEBUG] END OF DUMP for append_attention @ Layer {layer.layer_id}\n{'='*25}\n")
        
        # ==================== [END] 全面 Debug 打印代码 ====================
  
        
        metadata = self.attention_metadata

        sliding_window = layer.sliding_window

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
            full_debug_print(None, "C++ OP get_block_shape_and_split_kv_block", header="OBSERVATION POINT 3: C++ OP CALL")
            
            # 打印C++ OP的所有输入
            full_debug_print(forward_meta.seq_lens_encoder, "INPUT: seq_lens_encoder")
            full_debug_print(forward_meta.seq_lens_decoder, "INPUT: seq_lens_decoder")
            full_debug_print(forward_meta.seq_lens_this_time, "INPUT: seq_lens_this_time")
            full_debug_print(self.encoder_block_shape_q, "ATTR: encoder_block_shape_q")
            full_debug_print(self.decoder_block_shape_q, "ATTR: decoder_block_shape_q")
            full_debug_print(self.group_size, "ATTR: group_size")
            full_debug_print(self.block_size, "ATTR: block_size")

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
                self.speculate_max_draft_token_num + 1,
            )
            # 打印C++ OP的所有输出 (修改后的值)
            full_debug_print(forward_meta.encoder_num_blocks_x_cpu, "OUTPUT: encoder_num_blocks_x_cpu")
            full_debug_print(forward_meta.max_len_tensor_cpu, "OUTPUT: max_len_tensor_cpu")

        if self.use_output:
            quant_max_bound = getattr(layer, "quant_max_bound", 0.0)
            cache_quant_type = getattr(layer, "cache_quant_type_str", "none")
            compute_type = metadata._fuse_kernel_compute_dtype
            out_scale = getattr(layer, "out_scale", -1.0)
            # 1. get output datatype
            qkv_dtype = qkv.dtype
            if qkv_dtype == paddle.float16:
                D_type = paddle.float16
            elif qkv_dtype == paddle.bfloat16:
                D_type = paddle.bfloat16
            elif qkv_dtype == paddle.int32:
                if compute_type == "bf16":
                    D_type = paddle.bfloat16
                elif compute_type == "fp16":
                    D_type = paddle.float16
                else:
                    raise NotImplementedError("Only supported attr of qkv_type in ['float16', 'bfloat16'].")
            else:
                raise NotImplementedError("Only supported attr of qkv_type in ['float16', 'bfloat16', 'int32'].")
            # 2.Extract related parameters
            token_nums = qkv.shape[0]
            head_dims = self.head_dim if cache_quant_type != "cache_int4_zp" else self.head_dim * 2
            q_num_heads = self.num_heads
            # 3. generate output tensor of different dtypes
            if out_scale > 0.0:
                if abs(quant_max_bound - 127) < 0.000001:
                    res = paddle.empty([token_nums, q_num_heads * head_dims], dtype="int8")
                elif abs(quant_max_bound - 448) < 0.000001:
                    res = paddle.empty([token_nums, q_num_heads * head_dims], dtype="float8_e4m3fn")
                else:
                    raise NotImplementedError("Only supported attr of quant_max_bound in ['127', '448'].")
            else:
                res = paddle.empty([token_nums, q_num_heads * head_dims], dtype=D_type)

            res = append_attention_with_output(
                qkv,
                cache_k,
                cache_v,
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
                forward_meta.seq_lens_this_time,
                forward_meta.batch_id_per_token,
                forward_meta.cu_seqlens_q,
                metadata.block_tables,
                forward_meta.encoder_batch_ids,
                forward_meta.encoder_tile_ids_per_batch,
                forward_meta.encoder_num_blocks_x_cpu,
                forward_meta.kv_batch_ids,
                forward_meta.kv_tile_ids_per_batch,
                forward_meta.kv_num_blocks_x_cpu,
                forward_meta.decoder_batch_ids,
                forward_meta.decoder_tile_ids_per_batch,
                forward_meta.decoder_num_blocks_cpu,
                forward_meta.max_len_tensor_cpu,
                res,
                metadata.rotary_embs,
                metadata.attn_mask,
                layer.qkv_bias,
                layer.qkv_scale,
                cache_k_scales,
                cache_v_scales,
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
                metadata.max_partition_size,
                metadata.encoder_max_partition_size,
                self.speculate_max_draft_token_num + 1,
                self.causal,
                self.speculative_method is not None,
                sliding_window,
            )
        else:
            res = append_attention(
                qkv,
                cache_k,
                cache_v,
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
                forward_meta.seq_lens_this_time,
                forward_meta.batch_id_per_token,
                forward_meta.cu_seqlens_q,
                metadata.block_tables,
                forward_meta.encoder_batch_ids,
                forward_meta.encoder_tile_ids_per_batch,
                forward_meta.encoder_num_blocks_x_cpu,
                forward_meta.kv_batch_ids,
                forward_meta.kv_tile_ids_per_batch,
                forward_meta.kv_num_blocks_x_cpu,
                forward_meta.decoder_batch_ids,
                forward_meta.decoder_tile_ids_per_batch,
                forward_meta.decoder_num_blocks_cpu,
                forward_meta.max_len_tensor_cpu,
                metadata.rotary_embs,
                metadata.attn_mask,
                layer.qkv_bias,
                layer.qkv_scale,
                cache_k_scales,
                cache_v_scales,
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
                metadata.max_partition_size,
                metadata.encoder_max_partition_size,
                self.speculate_max_draft_token_num + 1,
                self.causal,
                self.speculative_method is not None,
                sliding_window,
            )
        return res
