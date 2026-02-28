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

from fastdeploy import envs
from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)
from fastdeploy.model_executor.layers.attention.utils import init_rank_and_device_id
from fastdeploy.platforms import current_platform

# Import unified extend attention for deterministic mode with prefix caching
try:
    from fastdeploy.model_executor.layers.attention.triton_ops import (
        extend_attention_fwd_unified,
    )

    TRITON_UNIFIED_ATTENTION_AVAILABLE = True
except ImportError:
    TRITON_UNIFIED_ATTENTION_AVAILABLE = False


@dataclass
class AppendAttentionMetadata(AttentionMetadata):
    """
    AppendAttentionMetadata
    """

    _dtype: paddle.dtype = paddle.bfloat16
    encoder_max_partition_size: int = 32768
    max_partition_size: int = 32768
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
    assert num_heads % kv_num_heads == 0
    assert max_model_len % block_size == 0
    assert max_model_len % encoder_block_shape_q == 0
    group_size = num_heads // kv_num_heads

    # NOTE: (changwenbin) When using auto_chunk,
    # decode_max_tile_size must take into account the maximum case, where *1024 can cover 128K.
    decode_max_tile_size = (
        1024 * max_batch_size * (int)(np.ceil(decoder_step_token_num * group_size / decoder_block_shape_q))
    )
    encode_max_tile_size = max_batch_size * (max_model_len * group_size // encoder_block_shape_q)
    kv_max_tile_size = max_batch_size * (max_model_len // block_size)
    res = {}
    res["decoder_batch_ids"] = paddle.full([decode_max_tile_size], 0, dtype="int32")
    res["decoder_tile_ids_per_batch"] = paddle.full([decode_max_tile_size], 0, dtype="int32")
    if current_platform.is_maca():
        res["decoder_num_blocks_cpu"] = paddle.full([1], 0, dtype="int32").cpu()
    else:
        res["decoder_num_blocks_cpu"] = paddle.full([1], 0, dtype="int32").pin_memory()
    # NOTE: (changwenbin) MLA kernel only needs decoder_num_blocks_device in place of GPU tensor,
    # adapted to cudagraph.
    res["decoder_num_blocks_device"] = paddle.full([1], 0, dtype="int32")
    res["decoder_chunk_size_device"] = paddle.full([1], 64, dtype="int32")
    res["max_len_tensor_cpu"] = paddle.full([9], 0, dtype="int32").cpu()

    res["encoder_batch_ids"] = paddle.full([encode_max_tile_size], 0, dtype="int32")
    res["encoder_tile_ids_per_batch"] = paddle.full([encode_max_tile_size], 0, dtype="int32")
    res["encoder_num_blocks_x_cpu"] = paddle.full([1], 0, dtype="int32").cpu()

    res["kv_batch_ids"] = paddle.full([kv_max_tile_size], 0, dtype="int32")
    res["kv_tile_ids_per_batch"] = paddle.full([kv_max_tile_size], 0, dtype="int32")
    res["kv_num_blocks_x_cpu"] = paddle.full([1], 0, dtype="int32").cpu()
    return res


class AppendAttentionBackend(AttentionBackend):
    """
    AppendAttentionBackend backend implementation.
    """

    __infer_dynamic_dims_fields__ = ["attention_metadata"]
    attention_metadata: AppendAttentionMetadata
    enable_ids_reorder: bool = envs.FD_PD_REORDER

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
        if self.use_output:
            flag = "FLAGS_cuda_graph_blacklist"
            paddle.set_flags(
                {
                    flag: ",".join(
                        list(
                            set(
                                paddle.get_flags(flag)[flag].split(",")
                                + [
                                    "custom_op.static_op_append_attention_with_output_",
                                    "custom_op.static_op_get_block_shape_and_split_kv_block",
                                ]
                            )
                        )
                    )
                }
            )
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

        # Check if we should use unified Triton kernel for deterministic mode
        # with prefix caching
        has_prefix_cache_hit = self._has_prefix_cache_hit(forward_meta)

        if envs.FD_DETERMINISTIC_MODE and has_prefix_cache_hit:
            # Defensive check: Triton kernel must be available for deterministic mode
            if not TRITON_UNIFIED_ATTENTION_AVAILABLE:
                raise RuntimeError(
                    "FD_DETERMINISTIC_MODE is enabled with prefix cache hit, "
                    "but the unified Triton kernel is not available. "
                    "This may be due to import failure or missing dependencies. "
                    "Please ensure Triton is properly installed and the kernel can be imported."
                )
            return self._forward_extend_unified_triton(q, k, v, qkv, layer, forward_meta, metadata)

        # - PaddleFormers fallback: rope_already_applied=True -> use identity RoPE (cos=1, sin=0)
        rope_already_applied = getattr(forward_meta, "rope_already_applied", False)
        if rope_already_applied and forward_meta.rotary_embs is not None:
            forward_meta.rotary_embs = self._get_identity_rotary_embs(forward_meta.rotary_embs)

        sliding_window = layer.sliding_window

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
            # print(forward_meta.seq_lens_this_time)
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
                res = paddle.zeros([token_nums, q_num_heads * head_dims], dtype=D_type)

            res = append_attention_with_output(
                qkv,
                cache_k,
                cache_v,
                forward_meta.seq_lens_encoder,
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
                forward_meta.decoder_batch_ids,
                forward_meta.decoder_tile_ids_per_batch,
                forward_meta.decoder_num_blocks_cpu,
                forward_meta.max_len_tensor_cpu,
                res,
                forward_meta.rotary_embs,
                forward_meta.attn_mask,
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
                q_norm_weight,
                k_norm_weight,
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
                forward_meta.block_tables,
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
                forward_meta.rotary_embs,
                forward_meta.attn_mask,
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
                q_norm_weight,
                k_norm_weight,
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

    def _has_prefix_cache_hit(self, forward_meta: ForwardMeta) -> bool:
        """
        Check if any request has prefix cache hit.

        Args:
            forward_meta: Forward metadata containing prefix_lens

        Returns:
            True if any request has prefix cache hit (prefix_lens > 0)
        """
        # During CUDA graph capture, we cannot use .item() as it causes
        # CUDA synchronization which is not allowed during capture.
        # Return False early to avoid the synchronization.
        if getattr(forward_meta, "step_use_cudagraph", False):
            return False

        prefix_lens = getattr(forward_meta, "prefix_lens", None)
        if prefix_lens is None:
            return False
        # Check if any prefix_lens > 0
        return (prefix_lens > 0).any().item()

    def _forward_extend_unified_triton(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        qkv: paddle.Tensor,
        layer: Attention,
        forward_meta: ForwardMeta,
        metadata: AppendAttentionMetadata,
    ) -> paddle.Tensor:
        """
        Forward using unified Triton kernel for deterministic mode with prefix caching.

        This method is called when:
        1. FD_DETERMINISTIC_MODE is enabled
        2. There is prefix cache hit (prefix_lens > 0)

        The unified kernel processes both prefix KV and extend KV in a single pass,
        ensuring deterministic behavior regardless of cache hit/miss status.

        Implementation steps:
        1. Check RoPE status (defensive programming)
        2. Extract Q from qkv tensor (after RoPE application)
        3. Build unified KV indices from block_tables and prefix_lens
        4. Call extend_attention_fwd_unified Triton kernel
        """
        # Step 1: Defensive check for RoPE status
        # The unified Triton kernel assumes RoPE is already applied to Q and cache K.
        # - If rope_already_applied=True: PaddleFormers mode, RoPE applied externally -> OK
        # - If rope_already_applied=False: RoPE needs to be applied inside kernel -> NOT SUPPORTED
        rope_already_applied = getattr(forward_meta, "rope_already_applied", False)
        if not rope_already_applied:
            raise NotImplementedError(
                "Unified Triton kernel for deterministic mode currently only supports "
                "PaddleFormers mode (rope_already_applied=True). "
                "For non-PaddleFormers mode, the kernel needs RoPE support which is not implemented yet. "
                "Please use PaddleFormers model or disable prefix caching in deterministic mode."
            )

        # Step 2: Get cache tensors
        cache_quant_type_str = getattr(layer, "cache_quant_type_str", "none")
        if cache_quant_type_str == "block_wise_fp8":
            cache_k = forward_meta.caches[4 * layer.layer_id]
            cache_v = forward_meta.caches[4 * layer.layer_id + 1]
            # FP8 quantization not supported in unified kernel yet
            raise NotImplementedError(
                "Unified Triton kernel does not support FP8 quantization yet. "
                "Please disable prefix caching or deterministic mode."
            )
        else:
            cache_k = forward_meta.caches[2 * layer.layer_id]
            cache_v = forward_meta.caches[2 * layer.layer_id + 1]

        # Step 3: Get metadata
        prefix_lens = forward_meta.prefix_lens  # [batch_size], number of cached tokens per sequence
        seq_lens_this_time = forward_meta.seq_lens_this_time  # [batch_size, 1]
        cu_seqlens_q = forward_meta.cu_seqlens_q  # [batch_size + 1, 1]
        block_tables = forward_meta.block_tables  # [max_num_seqs, max_blocks_per_seq]

        # Determine batch size (number of active sequences)
        # cu_seqlens_q shape is [batch_size + 1, 1], get batch_size from it
        batch_size = cu_seqlens_q.shape[0] - 1

        # Step 4: Extract Q from qkv tensor
        # qkv shape: [token_num, num_heads + 2 * kv_num_heads, head_dim]
        # We need Q which is the first num_heads columns
        token_num = qkv.shape[0]
        q_num_heads = self.num_heads
        head_dim = self.head_dim

        # Extract Q: [token_num, num_heads, head_dim]
        # qkv is either [token_num, num_heads + 2*kv_num_heads, head_dim] or
        # [token_num, (num_heads + 2*kv_num_heads) * head_dim]
        if len(qkv.shape) == 3:
            # Shape: [token_num, num_heads + 2*kv_num_heads, head_dim]
            q_tensor = qkv[:, :q_num_heads, :]
        else:
            # Shape: [token_num, (num_heads + 2*kv_num_heads) * head_dim]
            q_tensor = qkv[:, : q_num_heads * head_dim].reshape([token_num, q_num_heads, head_dim])

        # Step 5: Prepare output tensor
        # Output shape: [token_num, num_heads, head_dim]
        o_tensor = paddle.zeros([token_num, q_num_heads, head_dim], dtype=q_tensor.dtype, place=q_tensor.place)

        # Step 6: Build unified KV indices
        # We need to construct:
        # - prefix_kv_indptr: [batch_size + 1] - prefix KV indptr
        # - prefix_kv_indices: prefix KV block indices
        # - extend_kv_indices: extend KV block indices
        # - extend_start_loc: [batch_size] - extend start location
        # - extend_seq_lens: [batch_size] - extend sequence lengths

        unified_kv_indptr, unified_kv_indices, computed_prefix_lens = self._build_unified_kv_indices_impl(
            block_tables=block_tables,
            prefix_lens=prefix_lens,
            seq_lens_this_time=seq_lens_this_time,
            cu_seqlens_q=cu_seqlens_q,
            batch_size=batch_size,
            block_size=self.block_size,
        )

        # Step 7: Compute max extend length for grid configuration
        # extend_seq_lens = seq_lens_this_time for each sequence
        extend_seq_lens = seq_lens_this_time[:batch_size].flatten()
        max_len_extend = int(extend_seq_lens.max().item()) if batch_size > 0 else 0

        if max_len_extend == 0:
            # No extend tokens, return zeros
            return o_tensor.reshape([token_num, q_num_heads * head_dim])

        # Step 8: Prepare qo_indptr (query offsets)
        # cu_seqlens_q is [batch_size + 1, 1], we need [batch_size + 1]
        qo_indptr = cu_seqlens_q.flatten().astype("int32")

        # Step 9: Compute sm_scale
        sm_scale = 1.0 / (head_dim**0.5)

        # Step 10: Get logit cap if applicable
        logit_cap = getattr(layer, "logit_cap", 0.0) if hasattr(layer, "logit_cap") else 0.0

        # Step 11: Call the unified Triton kernel
        extend_attention_fwd_unified(
            q=q_tensor,
            o=o_tensor,
            k_buffer=cache_k,
            v_buffer=cache_v,
            qo_indptr=qo_indptr,
            kv_indptr=unified_kv_indptr,
            kv_indices=unified_kv_indices,
            prefix_lens=computed_prefix_lens.astype("int32"),
            max_len_extend=max_len_extend,
            sm_scale=sm_scale,
            logit_cap=logit_cap,
            is_causal=self.causal,
        )

        # Return output in the expected format: [token_num, num_heads * head_dim]
        return o_tensor.reshape([token_num, q_num_heads * head_dim])

    def _build_unified_kv_indices_impl(
        self,
        block_tables: paddle.Tensor,
        prefix_lens: paddle.Tensor,
        seq_lens_this_time: paddle.Tensor,
        cu_seqlens_q: paddle.Tensor,
        batch_size: int,
        block_size: int,
    ) -> tuple:
        """
        Build unified KV indices from block_tables and prefix_lens.

        This method constructs the data structures needed for the unified Triton kernel:
        - unified_kv_indptr: [batch_size + 1] - cumulative count of KV blocks per sequence
        - unified_kv_indices: flattened block indices for all sequences
        - prefix_lens: [batch_size] - number of prefix blocks per sequence

        Args:
            block_tables: [max_num_seqs, max_blocks_per_seq] - block IDs for each sequence
            prefix_lens: [batch_size] - number of cached tokens per sequence
            seq_lens_this_time: [batch_size, 1] - current sequence lengths
            cu_seqlens_q: [batch_size + 1, 1] - cumulative query sequence lengths
            batch_size: number of active sequences
            block_size: number of tokens per block

        Returns:
            (unified_kv_indptr, unified_kv_indices, prefix_block_lens)
        """
        place = block_tables.place

        # Convert prefix_lens from tokens to blocks
        # prefix_block_lens[i] = number of blocks that contain prefix tokens
        prefix_block_lens = (prefix_lens[:batch_size] + block_size - 1) // block_size

        # Convert seq_lens_this_time from tokens to blocks
        seq_lens_flat = seq_lens_this_time[:batch_size].flatten()
        total_block_lens = (seq_lens_flat + block_size - 1) // block_size

        # Build unified_kv_indptr
        unified_lens = total_block_lens.astype("int32")
        # Use existing tensor to create zeros on the same device
        zeros_tensor = unified_lens[:1] * 0
        unified_kv_indptr = paddle.concat(
            [
                zeros_tensor,
                paddle.cumsum(unified_lens, axis=0).astype("int32"),
            ]
        )

        # Build unified_kv_indices by extracting blocks from block_tables
        # For each sequence, we need to copy its blocks (prefix + extend)
        total_blocks = int(unified_kv_indptr[-1].item())
        unified_kv_indices = paddle.empty([total_blocks], dtype="int64")
        # Ensure it's on the same device
        unified_kv_indices = unified_kv_indices._copy_to(place, False)

        # Use a simple loop to copy block indices
        # TODO: Optimize with a Triton kernel for large batch sizes
        offset = 0
        for i in range(batch_size):
            # Get the number of blocks for this sequence
            num_blocks = int(total_block_lens[i].item())
            if num_blocks > 0:
                # Extract blocks from block_tables[i, :num_blocks]
                blocks = block_tables[i, :num_blocks]
                unified_kv_indices[offset : offset + num_blocks] = blocks.astype("int64")
                offset += num_blocks

        return unified_kv_indptr, unified_kv_indices, prefix_block_lens.astype("int32")
