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
from paddle.nn.functional.flash_attention import flash_attn_unpadded
from paddleformers.utils.log import logger

try:
    from paddle.nn.functional.flash_attention import flash_attention_v3_varlen
except Exception as e:
    logger.debug(f"flash_attention_v3_varlen not available: {e}")
    flash_attention_v3_varlen = None

try:
    from paddle.nn.functional.flash_attention import flashmask_attention
except Exception as e:
    logger.debug(f"flashmask_attention not available: {e}")
    flashmask_attention = None

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)
from fastdeploy.model_executor.layers.attention.ops import (
    append_attention,
    config_for_attention,
    decode_unified_attention,
    decoder_write_cache_with_rope,
    get_attn_mask_q,
    get_block_shape_and_split_kv_block,
    gqa_rope_write_cache,
    init_kv_signal_per_query,
    init_signal_layerwise,
    open_shm_and_get_meta_signal,
    pre_cache_len_concat,
)
from fastdeploy.model_executor.layers.attention.utils import init_rank_and_device_id
from fastdeploy.model_executor.utils import get_sm_version
from fastdeploy.utils import register_custom_python_op

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta

import os

from fastdeploy import envs
from fastdeploy.platforms import current_platform

flashmask_attention_v4 = None

if current_platform.is_cuda():
    from fastdeploy.model_executor.ops.gpu import merge_prefill_decode_output
else:
    merge_prefill_decode_output = None


from fastdeploy.spec_decode import SpecMethod

FLASH_ATTN_VERSION = None

from fastdeploy.model_executor.utils import try_import


def init_flash_attn_version():
    """
    init_flash_attn_version
    """
    if current_platform.is_cuda():
        global FLASH_ATTN_VERSION
        sm_version = get_sm_version()
        if sm_version >= 100:
            try:
                paddle.enable_compat(scope={"cutlass"})
                try:
                    old_api = try_import(["paddlefleet.ops"])
                    if old_api is not None:
                        from paddlefleet.ops import is_flash_mask_available

                        if is_flash_mask_available():
                            from paddlefleet.ops.flash_mask.cute.interface import (
                                flashmask_attention as fa4,
                            )
                        else:
                            raise ModuleNotFoundError("flash_mask not available.")
                    else:
                        from paddlefleet_ops import is_flash_mask_available

                        if is_flash_mask_available():
                            from paddlefleet_ops.flash_mask.cute.interface import (
                                flashmask_attention as fa4,
                            )
                        else:
                            raise ModuleNotFoundError("flash_mask not available.")

                except (ImportError, ModuleNotFoundError):
                    logger.info(f"The current platform[sm{get_sm_version()}] can't import Flash Attention V4.")

                global flashmask_attention_v4
                flashmask_attention_v4 = fa4
                FLASH_ATTN_VERSION = 4
                logger.info("The current platform supports Flash Attention V4.")
            except ImportError:
                logger.info(f"The current platform[sm{get_sm_version()}] can't import Flash Attention V4.")

        if FLASH_ATTN_VERSION is None:
            if sm_version == 90 and 90 in paddle.version.cuda_archs():
                FLASH_ATTN_VERSION = 3
                logger.info("The current platform supports Flash Attention V3.")
            else:
                FLASH_ATTN_VERSION = 2
                logger.info("The current platform only support Flash Attention V2.")
    else:
        logger.info("Only support CUDA version flash attention.")


def _is_deterministic_mode():
    """Check if FD_DETERMINISTIC_MODE is enabled."""
    return envs.FD_DETERMINISTIC_MODE


init_flash_attn_version()


def flash_attn_func(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    cu_seqlens_q: Optional[paddle.Tensor] = None,
    cu_seqlens_k: Optional[paddle.Tensor] = None,
    max_seqlen_q: Optional[paddle.Tensor] = None,
    max_seqlen_k: Optional[paddle.Tensor] = None,
    attn_mask_q: Optional[paddle.Tensor] = None,
    causal: bool = True,
    num_heads: int = None,
    kv_num_heads: int = None,
    head_dim: int = 128,
    version: Optional[int] = None,
):
    if FLASH_ATTN_VERSION is None:
        init_flash_attn_version()
    if version is None:
        version = FLASH_ATTN_VERSION

    if version == 4:
        assert (
            flashmask_attention_v4 is not None
        ), "Cannot import flashmask_attention from flash_mask.cute.interface, please install it first"
        assert attn_mask_q is not None, "FA4 requires attn_mask_q"
        assert num_heads is not None
        assert kv_num_heads is not None
        original_flash_attn_version = paddle.base.framework.get_flags(["FLAGS_flash_attn_version"])[
            "FLAGS_flash_attn_version"
        ]
        with paddle.no_grad():
            try:
                paddle.set_flags({"FLAGS_flash_attn_version": 4})
                out = flashmask_attention_v4(
                    q.reshape([1, -1, num_heads, head_dim]),
                    k.reshape([1, -1, kv_num_heads, head_dim]),
                    v.reshape([1, -1, kv_num_heads, head_dim]),
                    startend_row_indices=attn_mask_q,
                    causal=False,
                    return_softmax_lse=True,
                    training=True,
                )
            finally:
                paddle.set_flags({"FLAGS_flash_attn_version": original_flash_attn_version})
        return out

    if attn_mask_q is not None:
        assert flashmask_attention is not None
        out = flashmask_attention(
            q.reshape([1, -1, num_heads, head_dim]),
            k.reshape([1, -1, kv_num_heads, head_dim]),
            v.reshape([1, -1, kv_num_heads, head_dim]),
            startend_row_indices=attn_mask_q,
            causal=False,
        )
    else:
        if version == 3:
            out = flash_attention_v3_varlen(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=causal,
            )
        else:
            out = flash_attn_unpadded(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=causal,
                scale=head_dim**-0.5,
                training=False,
            )
    return out


@dataclass
class FlashAttentionMetadata(AttentionMetadata):
    """
    FlashAttentionMetadata
    """

    cu_seqlens_k: Optional[paddle.Tensor] = None

    pre_cache_batch_ids = None
    pre_cache_tile_ids_per_batch = None
    pre_cache_num_blocks_cpu = None
    kv_token_num_cpu = None

    # pd_disaggregation
    kv_signal_metadata: Optional[paddle.Tensor] = None
    kv_signal_data_list: List[Optional[paddle.Tensor]] = field(default_factory=list)

    _fuse_kernel_compute_dtype: str = "bf16"
    _dtype: paddle.dtype = paddle.bfloat16

    max_len_tensor_cpu_decoder: Optional[paddle.Tensor] = None

    attn_mask_q: Optional[paddle.Tensor] = None


class FlashAttnLayerCtx:
    """
    Container for optional/variable Tensor args and scalar constants for
    python_op_flash_attn_forward.  Passed as a single const (non-Tensor)
    parameter so that register_custom_python_op never sees it as a mutable
    pir.Value regardless of whether the fields are None or real Tensors.
    This pattern follows python_op_fused_moe_kernel_paddle's quant_config arg.
    The object is created once in FlashAttentionBackend.__init__ and its
    attributes are updated in-place each call, so SOT trace never inlines
    __init__ and treats it as a stable ObjectVariable constant.
    """


def _python_op_flash_attn_forward_infer_meta(
    qkv,
    cache_k,
    cache_v,
    seq_lens_encoder,
    seq_lens_decoder,
    seq_lens_this_time,
    batch_id_per_token,
    cu_seqlens_q,
    cu_seqlens_k,
    block_tables,
    encoder_batch_ids,
    encoder_tile_ids_per_batch,
    encoder_num_blocks_x_cpu,
    kv_batch_ids,
    kv_tile_ids_per_batch,
    kv_num_blocks_x_cpu,
    decoder_batch_ids,
    decoder_tile_ids_per_batch,
    decoder_num_blocks_cpu,
    decoder_num_blocks_device,
    decoder_chunk_size_device,
    max_len_tensor_cpu,
    rotary_embs,
    layer_ctx,
):
    token_num = qkv.shape[0]
    return paddle.static.MetaTensor(shape=[token_num, layer_ctx.num_heads * layer_ctx.head_dim], dtype=qkv.dtype)


@register_custom_python_op(
    name="python_op_flash_attn_forward",
    infer_meta=_python_op_flash_attn_forward_infer_meta,
    input_names=[
        "qkv",
        "cache_k",
        "cache_v",
        "seq_lens_encoder",
        "seq_lens_decoder",
        "seq_lens_this_time",
        "batch_id_per_token",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "block_tables",
        "encoder_batch_ids",
        "encoder_tile_ids_per_batch",
        "encoder_num_blocks_x_cpu",
        "kv_batch_ids",
        "kv_tile_ids_per_batch",
        "kv_num_blocks_x_cpu",
        "decoder_batch_ids",
        "decoder_tile_ids_per_batch",
        "decoder_num_blocks_cpu",
        "decoder_num_blocks_device",
        "decoder_chunk_size_device",
        "max_len_tensor_cpu",
        "rotary_embs",
    ],
    output_names=["out"],
    inplace_map={},
)
def python_op_flash_attn_forward(
    qkv,
    cache_k,
    cache_v,
    seq_lens_encoder,
    seq_lens_decoder,
    seq_lens_this_time,
    batch_id_per_token,
    cu_seqlens_q,
    cu_seqlens_k,
    block_tables,
    encoder_batch_ids,
    encoder_tile_ids_per_batch,
    encoder_num_blocks_x_cpu,
    kv_batch_ids,
    kv_tile_ids_per_batch,
    kv_num_blocks_x_cpu,
    decoder_batch_ids,
    decoder_tile_ids_per_batch,
    decoder_num_blocks_cpu,
    decoder_num_blocks_device,
    decoder_chunk_size_device,
    max_len_tensor_cpu,
    rotary_embs,
    layer_ctx,
):
    """
    Wraps FlashAttentionBackend forward_mixed as a single py_op so SOT treats
    it as a black box (no BreakGraphError from int(TensorVariable)). The op is
    placed in FLAGS_cuda_graph_blacklist and runs eagerly outside the CUDA
    graph every step.

    get_block_shape_and_split_kv_block is called INSIDE this py-op (guarded by
    layer_id == 0 so it runs once per step). It must run here rather than as a
    separately-blacklisted custom op because it conditionally skips the DtoH
    copy that refreshes max_len_tensor_cpu when IsCUDAGraphCapturing() is true
    (see get_block_shape_and_split_kv_block.cu). Only by running fully eagerly
    inside this py-op is IsCUDAGraphCapturing() guaranteed false at replay, so
    max_len_tensor_cpu / decoder_num_blocks_cpu are always fresh; otherwise
    downstream pre_cache_len_concat would size buffers from a stale (warmup)
    max_dec_len and write out of bounds (CUDA illegal memory access).
    """
    cache_k_scales = layer_ctx.cache_k_scales
    cache_v_scales = layer_ctx.cache_v_scales
    cache_k_out_scale = layer_ctx.cache_k_out_scale
    cache_v_out_scale = layer_ctx.cache_v_out_scale
    cache_k_zp = layer_ctx.cache_k_zp
    cache_v_zp = layer_ctx.cache_v_zp
    attn_mask = layer_ctx.attn_mask
    attn_mask_offsets = layer_ctx.attn_mask_offsets
    qkv_bias = layer_ctx.qkv_bias
    qkv_scale = layer_ctx.qkv_scale
    linear_shift = layer_ctx.linear_shift
    linear_smooth = layer_ctx.linear_smooth
    q_norm_weight = layer_ctx.q_norm_weight
    k_norm_weight = layer_ctx.k_norm_weight
    kv_signal_data = layer_ctx.kv_signal_data
    sinks = layer_ctx.sinks
    decode_block_indices = layer_ctx.decode_block_indices
    decode_num_blocks = layer_ctx.decode_num_blocks
    decode_chunk_size = layer_ctx.decode_chunk_size
    decode_tmp_workspace = layer_ctx.decode_tmp_workspace
    decode_tmp_m = layer_ctx.decode_tmp_m
    decode_tmp_d = layer_ctx.decode_tmp_d
    num_heads = layer_ctx.num_heads
    kv_num_heads = layer_ctx.kv_num_heads
    head_dim = layer_ctx.head_dim
    attn_outputsize_tp = layer_ctx.attn_outputsize_tp
    max_seq_len = layer_ctx.max_seq_len
    encoder_block_shape_q = layer_ctx.encoder_block_shape_q
    decoder_block_shape_q = layer_ctx.decoder_block_shape_q
    group_size = layer_ctx.group_size
    block_size = layer_ctx.block_size
    max_partition_size = layer_ctx.max_partition_size
    max_tokens_per_batch = layer_ctx.max_tokens_per_batch
    speculate_max_draft_token_num = layer_ctx.speculate_max_draft_token_num
    causal = layer_ctx.causal
    use_speculate = layer_ctx.use_speculate
    rope_3d = layer_ctx.rope_3d
    fuse_kernel_compute_dtype = layer_ctx.fuse_kernel_compute_dtype
    use_decode_unified_attention = layer_ctx.use_decode_unified_attention
    rms_norm_eps = layer_ctx.rms_norm_eps
    cache_quant_type_str = layer_ctx.cache_quant_type_str
    use_neox_rotary_style = layer_ctx.use_neox_rotary_style
    quant_max_bound = layer_ctx.quant_max_bound
    quant_min_bound = layer_ctx.quant_min_bound
    out_scale = layer_ctx.out_scale

    # get_block_shape_and_split_kv_block writes the per-step attention metadata
    # (decoder/encoder/kv split buffers + max_len_tensor_cpu) in-place. Run it
    # once per step (layer 0) fully eagerly here so the guarded DtoH copies that
    # refresh max_len_tensor_cpu / decoder_num_blocks_cpu always execute
    # (IsCUDAGraphCapturing() is false inside this eager py-op). Layers 1..N read
    # the buffers written by layer 0.
    if layer_ctx.layer_id == 0:
        get_block_shape_and_split_kv_block(
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            decoder_batch_ids,
            decoder_tile_ids_per_batch,
            decoder_num_blocks_cpu,
            decoder_num_blocks_device,
            decoder_chunk_size_device,
            max_len_tensor_cpu,
            encoder_batch_ids,
            encoder_tile_ids_per_batch,
            encoder_num_blocks_x_cpu,
            kv_batch_ids,
            kv_tile_ids_per_batch,
            kv_num_blocks_x_cpu,
            encoder_block_shape_q,
            decoder_block_shape_q,
            group_size,
            block_size,
        )

    # NOTE: This metadata (pre_cache_len_concat / get_attn_mask_q / config_for_attention)
    # depends only on tensor inputs that are identical across all layers within a step.
    # It is computed unconditionally here (rather than cached at layer_id==0 on layer_ctx)
    # because under piecewise cudagraph the py-op body runs at graph-execution time while
    # layer_ctx (a shared const object) only ever holds its last-traced values, which would
    # make any cross-layer caching stale. Per-layer recompute is cheap and correct.
    # get_block_shape_and_split_kv_block runs eagerly inside this py-op at layer 0
    # (IsCUDAGraphCapturing() is false here), so its blocking DtoH copy DOES refresh
    # max_len_tensor_cpu with the real per-step values (see get_block_shape_...cu:296).
    # Read the host scalars straight from it, exactly like the eager (non-cudagraph) path.
    #
    # Do NOT recompute these from the live seq_lens tensors: index [3] must be
    #   max(seq_len_decoder + seq_len_this_time) over slots with seq_len_this_time > 0
    # (GetMaxLenKernel skips seq_len_this_time <= 0). A naive
    #   (seq_lens_decoder + seq_lens_this_time).max() over ALL slots picks up stale
    # seq_lens_decoder in inactive/padded slots -> too-large max_seqlen_k -> the FA3
    # varlen kernel schedules K tiles past the end of k/v -> illegal memory access
    # (flash_fwd_launch_template.h:160).
    max_len_cpu = max_len_tensor_cpu
    use_fa_do_prefill = int(max_len_cpu[1]) > 0
    if use_fa_do_prefill:
        max_len_tensor_cpu_decoder = paddle.clone(max_len_tensor_cpu)
        max_len_tensor_cpu_decoder[1] = 0
        (
            cu_seqlens_k,
            pre_cache_batch_ids,
            pre_cache_tile_ids_per_batch,
            pre_cache_num_blocks_cpu,
            kv_token_num_cpu,
        ) = pre_cache_len_concat(
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            max_len_cpu[2],
            block_size,
        )
        if FLASH_ATTN_VERSION == 4 or attn_mask_offsets is not None:
            attn_mask_q = get_attn_mask_q(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                attn_mask_kv=attn_mask_offsets,
                kv_token_num=int(kv_token_num_cpu[0]),
            )
        else:
            attn_mask_q = None
    else:
        max_len_tensor_cpu_decoder = max_len_tensor_cpu
        pre_cache_batch_ids = None
        pre_cache_tile_ids_per_batch = None
        pre_cache_num_blocks_cpu = None
        kv_token_num_cpu = None
        attn_mask_q = None
    if use_decode_unified_attention:
        config_for_attention(
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            decode_block_indices,
            decode_num_blocks,
            decode_chunk_size,
            max_len_tensor_cpu,
            cache_quant_type_str,
            group_size,
            kv_num_heads,
            max_tokens_per_batch,
        )

    token_num = qkv.shape[0]
    batch_id_per_token = batch_id_per_token.flatten()
    real_len = batch_id_per_token.shape[0]
    if real_len < token_num:
        batch_id_per_token = paddle.nn.functional.pad(batch_id_per_token, [0, token_num - real_len], value=-1)
    elif real_len > token_num:
        batch_id_per_token = batch_id_per_token[:token_num]

    if use_fa_do_prefill:
        q, k, v, _ = gqa_rope_write_cache(
            qkv,
            cache_k,
            cache_v,
            cu_seqlens_q,
            cu_seqlens_k,
            rotary_embs,
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
            batch_id_per_token,
            block_tables,
            kv_batch_ids,
            kv_tile_ids_per_batch,
            kv_num_blocks_x_cpu,
            pre_cache_batch_ids,
            pre_cache_tile_ids_per_batch,
            pre_cache_num_blocks_cpu,
            q_norm_weight,
            k_norm_weight,
            cache_k_scales,
            cache_v_scales,
            cache_k_out_scale,
            cache_v_out_scale,
            cache_k_zp,
            cache_v_zp,
            kv_signal_data,
            int(kv_token_num_cpu[0]),
            max_seq_len,
            rms_norm_eps,
            use_neox_rotary_style,
            cache_quant_type_str,
            rope_3d,
        )
        res_encoder = flash_attn_func(
            q,
            k,
            v,
            cu_seqlens_q[: cu_seqlens_k.shape[0]],
            cu_seqlens_k,
            max_seqlen_q=max_len_cpu[0],
            max_seqlen_k=max_len_cpu[3],
            attn_mask_q=attn_mask_q,
            causal=causal,
            num_heads=num_heads,
            kv_num_heads=kv_num_heads,
            head_dim=head_dim,
        )[0].reshape([-1, attn_outputsize_tp])

    if use_decode_unified_attention:
        qkv_out = decoder_write_cache_with_rope(
            qkv,
            cache_k,
            cache_v,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            batch_id_per_token,
            cu_seqlens_q,
            block_tables,
            max_len_tensor_cpu,
            rotary_embs,
            qkv_bias,
            cache_k_scales,
            cache_v_scales,
            cache_k_out_scale,
            cache_v_out_scale,
            cache_k_zp,
            cache_v_zp,
            kv_signal_data,
            q_norm_weight,
            k_norm_weight,
            rms_norm_eps,
            cache_quant_type_str,
            use_neox_rotary_style,
            rope_3d,
            max_seq_len,
            quant_max_bound,
            quant_min_bound,
            use_speculate,
        )
        res_decoder = paddle.empty([qkv.shape[0], num_heads * head_dim], dtype=qkv.dtype)
        if use_fa_do_prefill:
            res_decoder[: res_encoder.shape[0]] = res_encoder
        decode_unified_attention(
            qkv_out,
            cache_k,
            cache_v,
            decode_tmp_workspace,
            decode_tmp_m,
            decode_tmp_d,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            batch_id_per_token,
            cu_seqlens_q,
            block_tables,
            decode_block_indices,
            decode_num_blocks,
            decode_chunk_size,
            max_len_tensor_cpu,
            attn_mask,
            cache_k_scales,
            cache_v_scales,
            cache_k_out_scale,
            cache_v_out_scale,
            cache_k_zp,
            cache_v_zp,
            attn_mask_offsets,
            sinks,
            res_decoder,
            cache_quant_type_str,
            max_seq_len,
            quant_max_bound,
            quant_min_bound,
            speculate_max_draft_token_num + 1,
            causal,
        )
        return res_decoder
    else:
        res_decoder = append_attention(
            qkv,
            cache_k,
            cache_v,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            batch_id_per_token,
            cu_seqlens_q,
            block_tables,
            encoder_batch_ids,
            encoder_tile_ids_per_batch,
            encoder_num_blocks_x_cpu,
            kv_batch_ids,
            kv_tile_ids_per_batch,
            kv_num_blocks_x_cpu,
            decoder_batch_ids,
            decoder_tile_ids_per_batch,
            decoder_num_blocks_cpu,
            max_len_tensor_cpu_decoder,
            rotary_embs,
            attn_mask,
            qkv_bias,
            qkv_scale,
            cache_k_scales,
            cache_v_scales,
            cache_k_out_scale,
            cache_v_out_scale,
            cache_k_zp,
            cache_v_zp,
            linear_shift,
            linear_smooth,
            attn_mask_offsets,
            kv_signal_data,
            q_norm_weight,
            k_norm_weight,
            sinks,
            rms_norm_eps,
            fuse_kernel_compute_dtype,
            cache_quant_type_str,
            use_neox_rotary_style,
            rope_3d,
            max_seq_len,
            quant_max_bound,
            quant_min_bound,
            out_scale,
            encoder_block_shape_q,
            decoder_block_shape_q,
            max_partition_size,
            max_seq_len,
            speculate_max_draft_token_num + 1,
            causal,
            use_speculate,
        )
        if use_fa_do_prefill:
            merge_prefill_decode_output(
                res_encoder,
                res_decoder,
                seq_lens_encoder,
                seq_lens_decoder,
                seq_lens_this_time,
                cu_seqlens_q,
                num_heads,
                head_dim,
                speculate_max_draft_token_num + 1,
            )
            attn_out = paddle.empty([qkv.shape[0], num_heads * head_dim], dtype=qkv.dtype)
            attn_out[: res_encoder.shape[0]] = res_encoder
            return attn_out
        else:
            return res_decoder


class FlashAttentionBackend(AttentionBackend):
    """
    FlashAttentionBackend backend implementation
    """

    __infer_dynamic_dims_fields__ = ["attention_metadata"]
    attention_metadata: FlashAttentionMetadata

    def __init__(
        self,
        fd_config: FDConfig,
        kv_num_heads: int,
        num_heads: int,
        head_dim: int,
        encoder_block_shape_q: int = -1,
        decoder_block_shape_q: int = -1,
    ):
        """
        FlashAttentionBackend __init__
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
        self.encoder_block_shape_q: int = encoder_block_shape_q
        self.decoder_block_shape_q: int = decoder_block_shape_q

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
        self.max_partition_size: int = int(os.getenv("FLAGS_max_partition_size", 1024))
        self.max_tokens_per_batch: int = self.speculate_max_draft_token_num + 1
        if FLASH_ATTN_VERSION is None:
            init_flash_attn_version()

        # In static split-graph (piecewise cudagraph) mode, the dynamic attention
        # region cannot be captured. python_op_flash_attn_forward wraps the whole
        # dynamic logic (including get_block_shape_and_split_kv_block) into a single
        # py-op; blacklist it so it runs eagerly outside the CUDA graph. Running
        # get_block_shape inside this eager py-op (rather than as a separate
        # blacklisted custom op) guarantees IsCUDAGraphCapturing() is false when its
        # guarded DtoH copies refresh max_len_tensor_cpu / decoder_num_blocks_cpu.
        if not fd_config.graph_opt_config.full_cuda_graph:
            flag = "FLAGS_cuda_graph_blacklist"
            paddle.set_flags(
                {
                    flag: ",".join(
                        list(
                            set(
                                paddle.get_flags(flag)[flag].split(",")
                                + [
                                    "py_op.python_op_flash_attn_forward_",
                                ]
                            )
                        )
                    )
                }
            )

        # Per-layer, id-stable context objects carrying that layer's non-Tensor
        # scalars and (layer-static) weight tensors into python_op_flash_attn_forward
        # as one const param. IMPORTANT: there must be a DISTINCT object per layer.
        #
        # Under piecewise cudagraph the traced forward runs ONCE; python_op_flash_attn_forward's
        # body then runs eagerly each step reading the const object by reference. A single
        # shared object would freeze to the LAST layer's values at trace time, so every
        # layer's node would read layer N-1's weights. Keying by layer_id gives each node its
        # own frozen-but-correct object (layer weights are static across steps). The map is
        # populated lazily at trace time and reused across re-traces (stable ids -> stable
        # register_op specializations).
        self.layer_ctxs: dict = {}

    def get_attention_meta(self):
        """get_attention_meta"""
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

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        metadata = FlashAttentionMetadata()

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

        if metadata._dtype == "bfloat16":
            metadata._fuse_kernel_compute_dtype = "bf16"
        elif metadata._dtype == "float16":
            metadata._fuse_kernel_compute_dtype = "fp16"
        elif metadata._dtype == "float32":
            metadata._fuse_kernel_compute_dtype = "fp32"

        self.attention_metadata = metadata

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
    ):
        metadata = self.attention_metadata

        if self.pd_disaggregation_mode == "per_query":
            metadata.kv_signal_data_list[layer.layer_id] = init_signal_layerwise(
                metadata.kv_signal_metadata,
                layer.layer_id + self.start_layer_index,
            )

        if int(os.getenv("USE_TBO", "0")) == 1:
            if hasattr(forward_meta, "tbo_microbatch_id"):
                # here we only let the last microbatch invoke cache kv transfer！
                if forward_meta.tbo_microbatch_id == 0:
                    os.environ["FLAGS_fmt_write_cache_completed_signal"] = "0"
                elif forward_meta.tbo_microbatch_id == 1:
                    os.environ["FLAGS_fmt_write_cache_completed_signal"] = "1"

        norm_after_rope_in_kernel = not getattr(layer, "qk_norm_before_rope", False)
        q_norm_weight = getattr(layer, "q_norm_weight", None) if norm_after_rope_in_kernel else None
        k_norm_weight = getattr(layer, "k_norm_weight", None) if norm_after_rope_in_kernel else None

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

        ctx = self.layer_ctxs.get(layer.layer_id)
        if ctx is None:
            # First trace of this layer: create its own context and bake the
            # layer-invariant scalars once.
            ctx = FlashAttnLayerCtx()
            ctx.num_heads = self.num_heads
            ctx.kv_num_heads = self.kv_num_heads
            ctx.head_dim = self.head_dim
            ctx.attn_outputsize_tp = self.attn_outputsize_tp
            ctx.max_seq_len = self.max_seq_len
            ctx.encoder_block_shape_q = self.encoder_block_shape_q
            ctx.decoder_block_shape_q = self.decoder_block_shape_q
            ctx.group_size = self.group_size
            ctx.block_size = self.block_size
            ctx.max_partition_size = self.max_partition_size
            ctx.max_tokens_per_batch = self.max_tokens_per_batch
            ctx.speculate_max_draft_token_num = self.speculate_max_draft_token_num
            ctx.causal = self.causal
            ctx.use_speculate = self.use_speculate
            ctx.rope_3d = self.rope_3d
            self.layer_ctxs[layer.layer_id] = ctx
        ctx.cache_k_scales = cache_k_scales
        ctx.cache_v_scales = cache_v_scales
        ctx.cache_k_out_scale = getattr(layer, "cache_k_out_scale", None)
        ctx.cache_v_out_scale = getattr(layer, "cache_v_out_scale", None)
        ctx.cache_k_zp = getattr(layer, "cache_k_zp", None)
        ctx.cache_v_zp = getattr(layer, "cache_v_zp", None)
        ctx.attn_mask = forward_meta.attn_mask
        ctx.attn_mask_offsets = getattr(forward_meta, "attn_mask_offsets", None)
        ctx.qkv_bias = layer.qkv_bias
        ctx.qkv_scale = layer.qkv_scale
        ctx.linear_shift = layer.linear_shift
        ctx.linear_smooth = layer.linear_smooth
        ctx.q_norm_weight = q_norm_weight
        ctx.k_norm_weight = k_norm_weight
        ctx.kv_signal_data = metadata.kv_signal_data_list[layer.layer_id]
        ctx.sinks = getattr(layer, "sinks", None)
        ctx.decode_block_indices = getattr(forward_meta, "decode_block_indices", None)
        ctx.decode_num_blocks = getattr(forward_meta, "decode_num_blocks", None)
        ctx.decode_chunk_size = getattr(forward_meta, "decode_chunk_size", None)
        ctx.decode_tmp_workspace = getattr(forward_meta, "decode_tmp_workspace", None)
        ctx.decode_tmp_m = getattr(forward_meta, "decode_tmp_m", None)
        ctx.decode_tmp_d = getattr(forward_meta, "decode_tmp_d", None)
        ctx.fuse_kernel_compute_dtype = metadata._fuse_kernel_compute_dtype
        ctx.use_decode_unified_attention = envs.USE_DECODE_UNIFIED_ATTENTION
        ctx.rms_norm_eps = getattr(layer, "rms_norm_eps", 1e-6)
        ctx.cache_quant_type_str = cache_quant_type_str
        ctx.use_neox_rotary_style = layer.use_neox_rotary_style
        ctx.quant_max_bound = getattr(layer, "quant_max_bound", 0.0)
        ctx.quant_min_bound = getattr(layer, "quant_min_bound", 0.0)
        ctx.out_scale = getattr(layer, "out_scale", -1.0)
        ctx.layer_id = layer.layer_id

        return python_op_flash_attn_forward(
            qkv,
            cache_k,
            cache_v,
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.batch_id_per_token,
            forward_meta.cu_seqlens_q,
            forward_meta.cu_seqlens_k,
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
            forward_meta.decoder_num_blocks_device,
            forward_meta.decoder_chunk_size_device,
            forward_meta.max_len_tensor_cpu,
            forward_meta.rotary_embs,
            layer_ctx=ctx,
        )
