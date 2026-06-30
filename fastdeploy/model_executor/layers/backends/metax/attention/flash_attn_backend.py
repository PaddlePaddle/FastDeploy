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

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import paddle
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta, ForwardMode
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)
from fastdeploy.model_executor.layers.attention.utils import init_rank_and_device_id
from fastdeploy.model_executor.layers.backends.metax.attention.flash_attention_interface import (
    flash_attn_kvcache_func,
    flash_attn_unpadded_func,
)
from fastdeploy.model_executor.ops.gpu import cache_kv_with_rope
from fastdeploy.model_executor.ops.gpu import merge_qkv as merge_qkv_cu
from fastdeploy.model_executor.ops.gpu import split_qkv as split_qkv_cu
from fastdeploy.spec_decode import SpecMethod


_METAX_ATTENTION_DEBUG_COUNT = 0


def _debug_array_health(name: str, value) -> None:
    global _METAX_ATTENTION_DEBUG_COUNT
    if os.getenv("FD_METAX_ATTENTION_DEBUG", "0") != "1":
        return

    limit = int(os.getenv("FD_METAX_ATTENTION_DEBUG_LIMIT", "200") or "200")
    if _METAX_ATTENTION_DEBUG_COUNT >= limit:
        return

    try:
        if isinstance(value, paddle.Tensor):
            data = value.astype("float32").cpu().numpy()
            shape = list(value.shape)
        else:
            data = np.asarray(value, dtype=np.float32)
            shape = list(data.shape)

        nan_count = int(np.isnan(data).sum())
        inf_count = int(np.isinf(data).sum())
        finite = data[np.isfinite(data)]
        min_value = float(finite.min()) if finite.size else None
        max_value = float(finite.max()) if finite.size else None
        logger.warning(
            f"FD_METAX_ATTENTION_DEBUG {name}: shape={shape} "
            f"nan={nan_count} inf={inf_count} min={min_value} max={max_value}"
        )
        _METAX_ATTENTION_DEBUG_COUNT += 1
    except Exception as exc:
        logger.warning(f"FD_METAX_ATTENTION_DEBUG {name}: health check failed: {exc}")
        _METAX_ATTENTION_DEBUG_COUNT += 1


@dataclass
class FlashAttentionMetadata(AttentionMetadata):
    """
    FlashAttentionMetadata
    """

    max_len_kv: paddle.Tensor = None
    set_max_lengths: int = -1
    encoder_batch_ids: paddle.Tensor = None
    encoder_tile_ids_per_batch: paddle.Tensor = None
    encoder_num_blocks: paddle.Tensor = None
    kv_batch_ids: paddle.Tensor = None
    kv_tile_ids_per_batch: paddle.Tensor = None
    kv_num_blocks: paddle.Tensor = None
    decoder_batch_ids: paddle.Tensor = None
    decoder_tile_ids_per_batch: paddle.Tensor = None
    decoder_num_blocks: paddle.Tensor = None
    cu_seqlens_q_decode: paddle.Tensor = None
    batch_ids_per_token_decode: paddle.Tensor = None
    seq_lens_decode: paddle.Tensor = None
    block_table_decode: paddle.Tensor = None

    _dtype: paddle.dtype = paddle.bfloat16
    encoder_max_partition_size: int = 32768
    max_partition_size: int = 32768
    block_tables: Optional[paddle.Tensor] = None
    attn_mask: Optional[paddle.Tensor] = None
    encoder_block_shape_q: int = -1
    decoder_block_shape_q: int = -1
    _fuse_kernel_compute_dtype: str = "bf16"

    # pd_disaggregation
    kv_signal_metadata: Optional[paddle.Tensor] = None
    kv_signal_data_list: List[Optional[paddle.Tensor]] = field(default_factory=list)


class FlashAttentionBackend(AttentionBackend):
    """
    FlashAttentionBackend backend implementation.
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
    ) -> None:
        """
        FlashAttentionBackend __init__
        """
        super().__init__()
        self.attention_metadata: FlashAttentionMetadata = FlashAttentionMetadata()
        self.record_block_table_metadata = {}
        self.block_size: int = fd_config.cache_config.block_size
        self.max_seq_len: int = fd_config.model_config.max_model_len
        self.rope_theta: float = (
            10000.0 if fd_config.model_config.rope_theta is None else fd_config.model_config.rope_theta
        )
        self.rope_3d: bool = fd_config.enable_rope_3d_runtime
        self.causal: bool = getattr(fd_config.model_config, "causal", True)
        self.speculative_method = fd_config.speculative_config.method
        self.use_speculate: bool = self.speculative_method is not None
        self.speculate_max_draft_token_num: int = fd_config.speculative_config.num_speculative_tokens
        self.keep_pd_step_flag: bool = fd_config.speculative_config.model_type == "mtp"
        self.num_layers_draft_model: int = int(fd_config.speculative_config.method == SpecMethod.MTP)
        self.encoder_block_shape_q: int = encoder_block_shape_q
        self.decoder_block_shape_q: int = decoder_block_shape_q

        self.kv_num_heads: int = kv_num_heads
        self.num_heads: int = num_heads
        self.head_dim: int = fd_config.model_config.head_dim
        self.total_num_heads = self.num_heads + 2 * self.kv_num_heads
        self.total_hidden_dim = self.total_num_heads * self.head_dim
        self.dtype = paddle.get_default_dtype()
        self.num_layers: int = fd_config.model_config.num_hidden_layers
        self.max_partition_size: int = int(os.getenv("FLAGS_max_partition_size", 32768))
        self.separate_decode_kv = os.getenv("FD_METAX_SEPARATE_DECODE_KV", "0") == "1"
        self.safe_prefill_attn = os.getenv("FD_METAX_SAFE_PREFILL_ATTN", "0") == "1"
        self.safe_decode_attn = os.getenv("FD_METAX_SAFE_DECODE_ATTN", "0") == "1" or self.safe_prefill_attn
        self.safe_rope_source = os.getenv("FD_METAX_SAFE_ROPE_SOURCE", "auto").lower()

        self.pd_disaggregation_mode: str = fd_config.parallel_config.pd_disaggregation_mode

        self.start_layer_index: int = fd_config.model_config.start_layer_index

        if fd_config.parallel_config.expert_parallel_rank is None:
            fd_config.parallel_config.expert_parallel_rank = 0

        self.rank, self.device_id = init_rank_and_device_id(fd_config)
        self.enable_mm = fd_config.enable_mm_runtime
        self.model_type = fd_config.model_config.model_type
        self.is_neox_style = False
        if "paddleocr" in fd_config.model_config.model_type:
            self.is_neox_style = True

        max_num_seqs = fd_config.scheduler_config.max_num_seqs
        self.attention_metadata.decoder_batch_ids = paddle.empty(shape=[max_num_seqs], dtype="int32")
        self.attention_metadata.cu_seqlens_q_decode = paddle.empty(shape=[max_num_seqs + 1], dtype="int32")
        self.attention_metadata.batch_ids_per_token_decode = paddle.empty(shape=[max_num_seqs], dtype="int32")
        self.attention_metadata.seq_lens_decode = paddle.empty(shape=[max_num_seqs, 1], dtype="int32")
        self.attention_metadata.block_table_decode = paddle.empty(
            shape=[
                max_num_seqs,
                self.max_seq_len // self.block_size + fd_config.cache_config.enc_dec_block_num,
            ],
            dtype="int32",
        )

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize attntion metadata hence all layers in the forward pass can reuse it."""
        forward_meta.forward_mode = ForwardMode.NATIVE
        self.prefill_info_dict = {}
        self.decode_info_dict = {}
        self.hybrid_stage_meta = None
        self.prefill_qkv = None
        self.decode_qkv = None
        self.merged_output = None

        prefill_non_zeros_ids = forward_meta.seq_lens_this_time > 1
        decode_non_zeros_ids = forward_meta.seq_lens_this_time == 1
        self.prefill_info_dict["batch_ids"] = paddle.where(prefill_non_zeros_ids)[0].astype("int32")
        self.decode_info_dict["batch_ids"] = paddle.where(decode_non_zeros_ids)[0].astype("int32")

        self.prefill_len = len(self.prefill_info_dict["batch_ids"])
        self.decode_len = len(self.decode_info_dict["batch_ids"])
        self.has_prefill = self.prefill_len > 0
        self.has_decode = self.decode_len > 0

        if self.has_prefill:
            batch_ids_prefill = self.prefill_info_dict["batch_ids"]

            seq_lens_this_time_prefill = forward_meta.seq_lens_this_time[batch_ids_prefill]
            self.prefill_info_dict["cu_seqlens_q"] = paddle.concat(
                [paddle.zeros([1], dtype="int32"), paddle.cumsum(seq_lens_this_time_prefill, axis=0).astype("int32")],
                axis=0,
            )
            self.prefill_info_dict["seq_lens_prefill"] = paddle.zeros(self.prefill_len, dtype="int32")

            local_ids = paddle.arange(self.prefill_len, dtype="int64").astype("int32")
            self.prefill_info_dict["batch_ids_per_token"] = paddle.repeat_interleave(
                local_ids, repeats=seq_lens_this_time_prefill, axis=0
            )

        if self.has_decode:
            batch_ids_decode = self.decode_info_dict["batch_ids"]

            seq_lens_this_time_decode = forward_meta.seq_lens_this_time[batch_ids_decode]
            cu_seqlens_q_decode = paddle.concat(
                [paddle.zeros([1], dtype="int32"), paddle.cumsum(seq_lens_this_time_decode, axis=0).astype("int32")],
                axis=0,
            )

            local_ids = paddle.arange(self.decode_len, dtype="int64").astype("int32")
            batch_ids_per_token_decode = paddle.repeat_interleave(local_ids, repeats=seq_lens_this_time_decode, axis=0)

            self.attention_metadata.decoder_batch_ids[: self.decode_len].copy_(batch_ids_decode)  # global batch id
            self.attention_metadata.cu_seqlens_q_decode[: self.decode_len + 1].copy_(cu_seqlens_q_decode)
            self.attention_metadata.batch_ids_per_token_decode[: self.decode_len].copy_(batch_ids_per_token_decode)
            self.attention_metadata.seq_lens_decode[: self.decode_len].copy_(
                forward_meta.seq_lens_decoder[batch_ids_decode]
            )
            self.attention_metadata.block_table_decode[: self.decode_len].copy_(
                forward_meta.block_tables[batch_ids_decode, :]
            )

        if self.has_prefill and self.has_decode:
            non_zeros_mask = forward_meta.seq_lens_this_time != 0
            seq_lens_non_zeros = forward_meta.seq_lens_this_time[non_zeros_mask].astype("int32")

            global_sequence_offsets = paddle.zeros(seq_lens_non_zeros.shape[0] + 1, dtype="int32")
            global_sequence_offsets[1:] = paddle.cumsum(seq_lens_non_zeros)

            is_prefill_array = seq_lens_non_zeros > 1

            group_boundary = paddle.where(is_prefill_array[1:] != is_prefill_array[:-1])[0].astype("int32") + 1
            group_starts = paddle.concat((paddle.zeros([1], dtype="int32"), group_boundary))
            group_ends = paddle.concat(
                (group_boundary, paddle.full([1], fill_value=seq_lens_non_zeros.shape[0], dtype="int32"))
            )

            compact_meta = []
            prefill_ptr = 0
            decode_ptr = 0

            for start, end in zip(group_starts, group_ends):
                is_prefill = is_prefill_array[start]
                g_start = global_sequence_offsets[start]
                g_end = global_sequence_offsets[end]
                num_tokens = g_end - g_start

                if is_prefill:
                    # [0, prefill_start, prefill_end, global_start, global_end]
                    compact_meta.append([0, prefill_ptr, prefill_ptr + num_tokens, g_start, g_end])
                    prefill_ptr += num_tokens
                else:
                    # [1, decode_start, decode_end, global_start, global_end]
                    compact_meta.append([1, decode_ptr, decode_ptr + num_tokens, g_start, g_end])
                    decode_ptr += num_tokens

            self.hybrid_stage_meta = paddle.to_tensor(compact_meta, dtype="int32")
            self.prefill_qkv = paddle.zeros([prefill_ptr, self.total_hidden_dim], dtype=self.dtype)
            self.decode_qkv = paddle.zeros([decode_ptr, self.total_hidden_dim], dtype=self.dtype)
            self.merged_output = paddle.zeros(
                [prefill_ptr + decode_ptr, self.num_heads, self.head_dim], dtype=self.dtype
            )

    def get_attention_meta(self) -> AttentionMetadata:
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
        key_cache_shape = value_cache_shape = [max_num_blocks, self.block_size, self.kv_num_heads, self.head_dim]

        if kv_cache_quant_type is not None and kv_cache_quant_type == "int4_zp":
            key_cache_shape = value_cache_shape = [
                max_num_blocks,
                self.kv_num_heads,
                self.block_size,
                self.head_dim // 2,
            ]

        return key_cache_shape, value_cache_shape

    def split_pd_qkv(self, qkv):
        split_qkv_cu(qkv, self.hybrid_stage_meta, self.prefill_qkv, self.decode_qkv)

    def merge_pd_output(self, prefill_out, decode_out):
        merge_qkv_cu(prefill_out, decode_out, self.hybrid_stage_meta, self.merged_output)

    def apply_rope_prefill(self, qkv, rotary_embs, caches_k, caches_v, block_tables):
        return cache_kv_with_rope(
            qkv,
            rotary_embs,
            self.prefill_info_dict["batch_ids_per_token"],
            self.prefill_info_dict["batch_ids"],
            self.prefill_info_dict["cu_seqlens_q"],
            self.prefill_info_dict["seq_lens_prefill"],
            caches_k,
            caches_v,
            block_tables,
            self.num_heads,
            self.kv_num_heads,
            self.head_dim,
            self.block_size,
            out_dims=3,
            neox_style=self.is_neox_style,  # is neox style
        )

    def apply_rope_decode(self, qkv, rotary_embs):
        return cache_kv_with_rope(
            qkv,
            rotary_embs,
            self.attention_metadata.batch_ids_per_token_decode,
            self.attention_metadata.decoder_batch_ids,
            self.attention_metadata.cu_seqlens_q_decode,
            self.attention_metadata.seq_lens_decode,
            None,
            None,
            None,
            self.num_heads,
            self.kv_num_heads,
            self.head_dim,
            -1,
            out_dims=4,
            neox_style=self.is_neox_style,  # is neox style
        )

    def apply_rope_decode_and_cache(self, qkv, rotary_embs, caches_k, caches_v, block_tables):
        return cache_kv_with_rope(
            qkv,
            rotary_embs,
            self.attention_metadata.batch_ids_per_token_decode,
            self.attention_metadata.decoder_batch_ids,
            self.attention_metadata.cu_seqlens_q_decode,
            self.attention_metadata.seq_lens_decode,
            caches_k,
            caches_v,
            block_tables,
            self.num_heads,
            self.kv_num_heads,
            self.head_dim,
            self.block_size,
            out_dims=4,
            neox_style=self.is_neox_style,  # is neox style
        )

    def _safe_default_rotary_for_position(self, position_idx):
        indices = np.arange(0, self.head_dim, 2, dtype=np.float32)
        inv_freq = 1.0 / (float(self.rope_theta) ** (indices / float(self.head_dim)))
        freqs = np.float32(position_idx) * inv_freq
        cos_half = np.cos(freqs).astype(np.float32)
        sin_half = np.sin(freqs).astype(np.float32)
        if self.is_neox_style:
            return (
                np.concatenate([cos_half, cos_half], axis=-1),
                np.concatenate([sin_half, sin_half], axis=-1),
            )
        return cos_half, sin_half

    def _rotary_values_are_valid(self, cos, sin):
        if cos is None or sin is None:
            return False
        cos = np.asarray(cos, dtype=np.float32)
        sin = np.asarray(sin, dtype=np.float32)
        if not np.all(np.isfinite(cos)) or not np.all(np.isfinite(sin)):
            return False
        max_abs = max(float(np.max(np.abs(cos))), float(np.max(np.abs(sin))))
        return max_abs <= 2.0

    def _safe_rotary_for_token(self, rotary_embs_np, global_batch_idx, position_idx):
        if self.safe_rope_source == "cpu_default":
            return self._safe_default_rotary_for_position(position_idx)

        if rotary_embs_np.ndim == 6:
            cos = rotary_embs_np[global_batch_idx, 0, 0, position_idx, 0]
            sin = rotary_embs_np[global_batch_idx, 1, 0, position_idx, 0]
        elif rotary_embs_np.ndim == 5:
            cos = rotary_embs_np[0, 0, position_idx, 0]
            sin = rotary_embs_np[1, 0, position_idx, 0]
        else:
            raise ValueError(f"Unsupported rotary_embs shape: {rotary_embs_np.shape}")

        if self.safe_rope_source == "auto" and not self._rotary_values_are_valid(cos, sin):
            return self._safe_default_rotary_for_position(position_idx)

        return cos, sin

    def _safe_apply_rotary_np(self, x, cos, sin):
        half = self.head_dim // 2
        if self.is_neox_style:
            cos = cos[..., :half]
            sin = sin[..., :half]
            left = x[..., :half]
            right = x[..., half:]
            return np.concatenate([left * cos - right * sin, right * cos + left * sin], axis=-1)

        cos = cos[..., :half]
        sin = sin[..., :half]
        even = x[..., 0::2]
        odd = x[..., 1::2]
        out = np.empty_like(x)
        out[..., 0::2] = even * cos - odd * sin
        out[..., 1::2] = odd * cos + even * sin
        return out

    def _safe_cache_prefill_kv(
        self,
        k_np,
        v_np,
        caches_k,
        caches_v,
        block_tables,
        batch_ids,
        cu_seqlens,
        seq_lens_prefill,
        place,
    ):
        block_tables_np = block_tables.cpu().numpy()
        flat_indices = []
        token_indices = []

        for local_batch_idx, global_batch_idx in enumerate(batch_ids):
            start = int(cu_seqlens[local_batch_idx])
            end = int(cu_seqlens[local_batch_idx + 1])
            base_position = int(seq_lens_prefill[local_batch_idx])
            for token_offset in range(end - start):
                position = base_position + token_offset
                block_table_idx = position // self.block_size
                block_id = int(block_tables_np[int(global_batch_idx), block_table_idx])
                if block_id < 0:
                    continue
                flat_indices.append(block_id * self.block_size + position % self.block_size)
                token_indices.append(start + token_offset)

        if not flat_indices:
            logger.warning("FD_METAX_ATTENTION_DEBUG safe prefill did not find valid KV cache slots")
            return

        flat_indices_np = np.asarray(flat_indices, dtype=np.int64)
        token_indices_np = np.asarray(token_indices, dtype=np.int64)
        k_updates_np = k_np[token_indices_np]
        v_updates_np = v_np[token_indices_np]
        _debug_array_health("prefill.safe_cache_flat_indices", flat_indices_np)
        _debug_array_health("prefill.safe_cache_k_updates", k_updates_np)
        _debug_array_health("prefill.safe_cache_v_updates", v_updates_np)

        try:
            flat_indices_tensor = paddle.to_tensor(flat_indices_np, dtype="int64", place=place)
            flat_k = caches_k.reshape([-1, self.kv_num_heads, self.head_dim])
            flat_v = caches_v.reshape([-1, self.kv_num_heads, self.head_dim])
            k_updates = paddle.to_tensor(k_updates_np, dtype=caches_k.dtype, place=place)
            v_updates = paddle.to_tensor(v_updates_np, dtype=caches_v.dtype, place=place)
            updated_k = paddle.scatter(flat_k, flat_indices_tensor, k_updates, overwrite=True)
            updated_v = paddle.scatter(flat_v, flat_indices_tensor, v_updates, overwrite=True)
            caches_k.copy_(updated_k.reshape(caches_k.shape), False)
            caches_v.copy_(updated_v.reshape(caches_v.shape), False)
            return
        except Exception as exc:
            logger.warning(f"FD_METAX_ATTENTION_DEBUG device KV cache update failed, using CPU fallback: {exc}")

        caches_k_np = caches_k.astype("float32").cpu().numpy()
        caches_v_np = caches_v.astype("float32").cpu().numpy()
        for flat_index, token_index in zip(flat_indices, token_indices):
            block_id = flat_index // self.block_size
            block_offset = flat_index % self.block_size
            caches_k_np[block_id, block_offset] = k_np[token_index]
            caches_v_np[block_id, block_offset] = v_np[token_index]
        caches_k.copy_(paddle.to_tensor(caches_k_np, dtype=caches_k.dtype, place=place), False)
        caches_v.copy_(paddle.to_tensor(caches_v_np, dtype=caches_v.dtype, place=place), False)

    def _safe_apply_rope_prefill(self, qkv, rotary_embs, caches_k, caches_v, block_tables):
        output_dtype = qkv.dtype
        output_place = qkv.place
        qkv_np = qkv.astype("float32").cpu().numpy()
        rotary_embs_np = rotary_embs.astype("float32").cpu().numpy()
        _debug_array_health("prefill.safe_qkv_input", qkv_np)
        _debug_array_health("prefill.safe_rotary_embs", rotary_embs_np)

        q_size = self.num_heads * self.head_dim
        kv_size = self.kv_num_heads * self.head_dim
        q_raw = qkv_np[:, :q_size].reshape([qkv_np.shape[0], self.num_heads, self.head_dim])
        k_raw = qkv_np[:, q_size : q_size + kv_size].reshape([qkv_np.shape[0], self.kv_num_heads, self.head_dim])
        v_raw = qkv_np[:, q_size + kv_size : q_size + 2 * kv_size].reshape(
            [qkv_np.shape[0], self.kv_num_heads, self.head_dim]
        )

        q_out = q_raw.copy()
        k_out = k_raw.copy()
        v_out = v_raw.copy()

        batch_ids = self.prefill_info_dict["batch_ids"].cpu().numpy().reshape(-1).astype(np.int64)
        cu_seqlens = self.prefill_info_dict["cu_seqlens_q"].cpu().numpy().reshape(-1).astype(np.int64)
        seq_lens_prefill = self.prefill_info_dict["seq_lens_prefill"].cpu().numpy().reshape(-1).astype(np.int64)
        _debug_array_health("prefill.safe_batch_ids", batch_ids)
        _debug_array_health("prefill.safe_cu_seqlens", cu_seqlens)
        _debug_array_health("prefill.safe_seq_lens_prefill", seq_lens_prefill)

        for local_batch_idx, global_batch_idx in enumerate(batch_ids):
            start = int(cu_seqlens[local_batch_idx])
            end = int(cu_seqlens[local_batch_idx + 1])
            base_position = int(seq_lens_prefill[local_batch_idx])
            if end <= start:
                continue
            cos_values = []
            sin_values = []
            for token_offset in range(end - start):
                cos, sin = self._safe_rotary_for_token(rotary_embs_np, int(global_batch_idx), base_position + token_offset)
                cos_values.append(cos)
                sin_values.append(sin)
            cos_np = np.asarray(cos_values, dtype=np.float32)[:, None, :]
            sin_np = np.asarray(sin_values, dtype=np.float32)[:, None, :]
            if local_batch_idx == 0:
                _debug_array_health("prefill.safe_cos_first_batch", cos_np[: min(4, cos_np.shape[0])])
                _debug_array_health("prefill.safe_sin_first_batch", sin_np[: min(4, sin_np.shape[0])])
            q_out[start:end] = self._safe_apply_rotary_np(q_raw[start:end], cos_np, sin_np)
            k_out[start:end] = self._safe_apply_rotary_np(k_raw[start:end], cos_np, sin_np)

        _debug_array_health("prefill.safe_q_after_rope", q_out)
        _debug_array_health("prefill.safe_k_after_rope", k_out)
        _debug_array_health("prefill.safe_v_after_rope", v_out)
        self._safe_cache_prefill_kv(
            k_out,
            v_out,
            caches_k,
            caches_v,
            block_tables,
            batch_ids,
            cu_seqlens,
            seq_lens_prefill,
            output_place,
        )
        return (
            paddle.to_tensor(q_out, dtype=output_dtype, place=output_place),
            paddle.to_tensor(k_out, dtype=output_dtype, place=output_place),
            paddle.to_tensor(v_out, dtype=output_dtype, place=output_place),
        )

    def _safe_prefill_attention(self, q, k, v, cu_seqlens_q):
        output_dtype = q.dtype
        output_place = q.place
        q_np = q.astype("float32").cpu().numpy()
        k_np = k.astype("float32").cpu().numpy()
        v_np = v.astype("float32").cpu().numpy()
        _debug_array_health("prefill.q_after_rope", q_np)
        _debug_array_health("prefill.k_after_rope", k_np)
        _debug_array_health("prefill.v_after_rope", v_np)
        if self.num_heads != self.kv_num_heads:
            if self.num_heads % self.kv_num_heads != 0:
                raise ValueError(f"num_heads {self.num_heads} must be divisible by kv_num_heads {self.kv_num_heads}")
            repeat = self.num_heads // self.kv_num_heads
            k_np = np.repeat(k_np, repeats=repeat, axis=1)
            v_np = np.repeat(v_np, repeats=repeat, axis=1)
        seqlens = [int(x) for x in cu_seqlens_q.cpu().numpy().reshape(-1).tolist()]

        outputs = []
        for start, end in zip(seqlens[:-1], seqlens[1:]):
            q_seg = np.transpose(q_np[start:end], [1, 0, 2])
            k_seg = np.transpose(k_np[start:end], [1, 2, 0])
            v_seg = np.transpose(v_np[start:end], [1, 0, 2])
            scores = np.matmul(q_seg, k_seg) * (self.head_dim**-0.5)
            if self.causal:
                seq_len = end - start
                causal_mask = np.triu(np.full([seq_len, seq_len], -np.inf, dtype=np.float32), k=1)
                scores = scores + causal_mask[None, :, :]
            scores = scores - np.max(scores, axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs = probs / np.sum(probs, axis=-1, keepdims=True)
            outputs.append(np.transpose(np.matmul(probs, v_seg), [1, 0, 2]))

        prefill_out = np.concatenate(outputs, axis=0) if len(outputs) > 1 else outputs[0]
        return paddle.to_tensor(prefill_out, dtype=output_dtype, place=output_place)

    def _safe_decode_attention(self, decode_qkv, rotary_embs, caches_k, caches_v):
        output_dtype = decode_qkv.dtype
        output_place = decode_qkv.place

        qkv_np = decode_qkv.astype("float32").cpu().numpy()
        rotary_embs_np = rotary_embs.astype("float32").cpu().numpy()
        caches_k_np = caches_k.astype("float32").cpu().numpy()
        caches_v_np = caches_v.astype("float32").cpu().numpy()

        q_size = self.num_heads * self.head_dim
        kv_size = self.kv_num_heads * self.head_dim
        q_raw = qkv_np[:, :q_size].reshape([qkv_np.shape[0], self.num_heads, self.head_dim])
        k_raw = qkv_np[:, q_size : q_size + kv_size].reshape([qkv_np.shape[0], self.kv_num_heads, self.head_dim])
        v_raw = qkv_np[:, q_size + kv_size : q_size + 2 * kv_size].reshape(
            [qkv_np.shape[0], self.kv_num_heads, self.head_dim]
        )

        batch_ids = self.decode_info_dict["batch_ids"].cpu().numpy().reshape(-1).astype(np.int64)
        cu_decode = (
            self.attention_metadata.cu_seqlens_q_decode[: self.decode_len + 1]
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.int64)
        )
        seq_lens_decode = (
            self.attention_metadata.seq_lens_decode[: self.decode_len]
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.int64)
        )
        block_tables = self.attention_metadata.block_table_decode[: self.decode_len].cpu().numpy()

        outputs = np.empty([qkv_np.shape[0], self.num_heads, self.head_dim], dtype=np.float32)
        repeat = self.num_heads // self.kv_num_heads
        if self.num_heads % self.kv_num_heads != 0:
            raise ValueError(f"num_heads {self.num_heads} must be divisible by kv_num_heads {self.kv_num_heads}")

        _debug_array_health("decode.safe_qkv_input", qkv_np)
        _debug_array_health("decode.safe_batch_ids", batch_ids)
        _debug_array_health("decode.safe_cu_seqlens", cu_decode)
        _debug_array_health("decode.safe_seq_lens", seq_lens_decode)

        for local_batch_idx, global_batch_idx in enumerate(batch_ids):
            start = int(cu_decode[local_batch_idx])
            end = int(cu_decode[local_batch_idx + 1])
            base_position = int(seq_lens_decode[local_batch_idx])
            for token_offset, row_idx in enumerate(range(start, end)):
                position = base_position + token_offset
                cos, sin = self._safe_rotary_for_token(rotary_embs_np, int(global_batch_idx), position)
                cos_np = np.asarray(cos, dtype=np.float32)[None, None, :]
                sin_np = np.asarray(sin, dtype=np.float32)[None, None, :]
                q_cur = self._safe_apply_rotary_np(q_raw[row_idx : row_idx + 1], cos_np, sin_np)[0]
                k_cur = self._safe_apply_rotary_np(k_raw[row_idx : row_idx + 1], cos_np, sin_np)[0]
                v_cur = v_raw[row_idx]

                block_id = int(block_tables[local_batch_idx, position // self.block_size])
                if block_id < 0:
                    raise ValueError(f"Invalid decode block id {block_id} for position {position}")
                block_offset = position % self.block_size
                caches_k_np[block_id, block_offset] = k_cur
                caches_v_np[block_id, block_offset] = v_cur

                k_ctx = []
                v_ctx = []
                for ctx_position in range(position + 1):
                    ctx_block_id = int(block_tables[local_batch_idx, ctx_position // self.block_size])
                    if ctx_block_id < 0:
                        continue
                    ctx_block_offset = ctx_position % self.block_size
                    k_ctx.append(caches_k_np[ctx_block_id, ctx_block_offset])
                    v_ctx.append(caches_v_np[ctx_block_id, ctx_block_offset])

                k_ctx = np.asarray(k_ctx, dtype=np.float32)
                v_ctx = np.asarray(v_ctx, dtype=np.float32)
                if repeat != 1:
                    k_ctx = np.repeat(k_ctx, repeats=repeat, axis=1)
                    v_ctx = np.repeat(v_ctx, repeats=repeat, axis=1)

                scores = np.einsum("hd,thd->ht", q_cur, k_ctx) * (self.head_dim**-0.5)
                scores = scores - np.max(scores, axis=-1, keepdims=True)
                probs = np.exp(scores)
                probs = probs / np.sum(probs, axis=-1, keepdims=True)
                outputs[row_idx] = np.einsum("ht,thd->hd", probs, v_ctx)

        _debug_array_health("decode.safe_output", outputs)
        caches_k.copy_(paddle.to_tensor(caches_k_np, dtype=caches_k.dtype, place=output_place), False)
        caches_v.copy_(paddle.to_tensor(caches_v_np, dtype=caches_v.dtype, place=output_place), False)
        return paddle.to_tensor(outputs, dtype=output_dtype, place=output_place)

    def forward_prefill(self, prefill_qkv, layer_id, k_cache_id, v_cache_id, forward_meta: ForwardMeta):
        _debug_array_health("prefill.qkv_input", prefill_qkv)
        if self.safe_prefill_attn:
            q, k, v = self._safe_apply_rope_prefill(
                prefill_qkv,
                forward_meta.rotary_embs,
                forward_meta.caches[k_cache_id],
                forward_meta.caches[v_cache_id],
                forward_meta.block_tables,
            )
        else:
            q, k, v = self.apply_rope_prefill(
                prefill_qkv,
                forward_meta.rotary_embs,
                forward_meta.caches[k_cache_id],
                forward_meta.caches[v_cache_id],
                forward_meta.block_tables,
            )

        if self.safe_prefill_attn:
            return self._safe_prefill_attention(q, k, v, self.prefill_info_dict["cu_seqlens_q"])

        prefill_out = flash_attn_unpadded_func(
            q,
            k,
            v,
            self.prefill_info_dict["cu_seqlens_q"],
            self.prefill_info_dict["cu_seqlens_q"],
            max_seqlen_q=self.max_seq_len,
            max_seqlen_k=self.max_seq_len,
            attn_mask=forward_meta.attn_mask,
            causal=self.causal,
        )[0]

        return prefill_out

    def forward_decode(self, decode_qkv, k_cache_id, v_cache_id, forward_meta: ForwardMeta):
        k_cache = forward_meta.caches[k_cache_id]
        v_cache = forward_meta.caches[v_cache_id]

        if self.safe_decode_attn:
            return self._safe_decode_attention(decode_qkv, forward_meta.rotary_embs, k_cache, v_cache)

        if self.separate_decode_kv:
            q, _, _ = self.apply_rope_decode_and_cache(
                decode_qkv,
                forward_meta.rotary_embs,
                k_cache,
                v_cache,
                forward_meta.block_tables,
            )
            decode_out = flash_attn_kvcache_func(
                q,
                k_cache,
                v_cache,
                self.attention_metadata.seq_lens_decode + 1,
                self.attention_metadata.block_table_decode,
                None,
                None,
                rotary_cos=None,
                rotary_sin=None,
                causal=self.causal,
                is_rotary_interleaved=True,
            )[0].squeeze(1)
            return decode_out

        q, k, v = self.apply_rope_decode(decode_qkv, forward_meta.rotary_embs)

        decode_out = flash_attn_kvcache_func(
            q,
            k_cache,
            v_cache,
            self.attention_metadata.seq_lens_decode,
            self.attention_metadata.block_table_decode,
            k,
            v,
            rotary_cos=None,
            rotary_sin=None,
            causal=self.causal,
            is_rotary_interleaved=True,
        )[0].squeeze(1)

        return decode_out

    @paddle.no_grad()
    def forward_native_backend(self, q, k, v, qkv, layer, forward_meta: ForwardMeta):

        layer_id = layer.layer_id
        k_cache_id = layer_id * 2
        v_cache_id = k_cache_id + 1

        if self.has_prefill and not self.has_decode:
            out = self.forward_prefill(qkv, layer_id, k_cache_id, v_cache_id, forward_meta)

        elif self.has_decode and not self.has_prefill:
            out = self.forward_decode(qkv, k_cache_id, v_cache_id, forward_meta)

        elif self.has_prefill and self.has_decode:
            self.split_pd_qkv(qkv)
            prefill_output = self.forward_prefill(self.prefill_qkv, layer_id, k_cache_id, v_cache_id, forward_meta)
            decode_output = self.forward_decode(self.decode_qkv, k_cache_id, v_cache_id, forward_meta)
            self.merge_pd_output(prefill_output, decode_output)
            out = self.merged_output
        else:
            out_shape = [qkv.shape[0], self.num_heads * self.head_dim]
            if qkv.dim() != 2:
                out_shape = [qkv.shape[0], self.num_heads, self.head_dim]
            return paddle.empty(out_shape, dtype=qkv.dtype)

        if qkv.dim() == 2:
            out = out.view([-1, self.num_heads * self.head_dim])

        return out
