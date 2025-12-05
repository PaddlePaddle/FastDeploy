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

import paddle

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
from fastdeploy.model_executor.ops.gpu import apply_rope


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
    rotary_cos_prefill: paddle.Tensor = None
    rotary_sin_prefill: paddle.Tensor = None
    rotary_cos_decode: paddle.Tensor = None
    rotary_sin_decode: paddle.Tensor = None

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
        self.rope_3d: bool = getattr(fd_config.model_config, "rope_3d", False)
        self.causal: bool = getattr(fd_config.model_config, "causal", True)
        self.speculative_method: str = fd_config.speculative_config.method
        self.use_speculate: bool = self.speculative_method is not None
        self.speculate_max_draft_token_num: int = fd_config.speculative_config.num_speculative_tokens
        self.keep_pd_step_flag: bool = fd_config.speculative_config.model_type == "mtp"
        self.num_layers_draft_model: int = int(fd_config.speculative_config.method in ["mtp"])
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

        self.pd_disaggregation_mode: str = fd_config.parallel_config.pd_disaggregation_mode

        self.start_layer_index: int = fd_config.model_config.start_layer_index

        if fd_config.parallel_config.expert_parallel_rank is None:
            fd_config.parallel_config.expert_parallel_rank = 0

        self.rank, self.device_id = init_rank_and_device_id(fd_config)
        self.enable_mm = fd_config.model_config.enable_mm
        max_num_seqs = fd_config.scheduler_config.max_num_seqs
        if self.enable_mm:
            self.attention_metadata.rotary_cos_decode = paddle.empty(
                shape=[max_num_seqs, 1, 1, self.head_dim],
                dtype="float32",
            )
            self.attention_metadata.rotary_sin_decode = paddle.empty(
                shape=[max_num_seqs, 1, 1, self.head_dim],
                dtype="float32",
            )

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize attntion metadata hence all layers in the forward pass can reuse it."""
        forward_meta.forward_mode = ForwardMode.NATIVE
        self.prefill_info_dict = {}
        self.decode_info_dict = {}

        prefill_non_zeros_ids = forward_meta.seq_lens_this_time > 1
        decode_non_zeros_ids = forward_meta.seq_lens_this_time == 1
        self.prefill_info_dict["batch_ids"] = paddle.where(prefill_non_zeros_ids)[0]
        self.decode_info_dict["batch_ids"] = paddle.where(decode_non_zeros_ids)[0]

        self.prefill_len = len(self.prefill_info_dict["batch_ids"])
        self.decode_len = len(self.decode_info_dict["batch_ids"])

        # only prefill
        if self.decode_len == 0:
            cu_seq_ids = list(range(self.prefill_len + 1))
            self.prefill_info_dict["cu_seqlens_q"] = forward_meta.cu_seqlens_q[cu_seq_ids].astype("int32")
        # only decode
        elif self.prefill_len == 0:
            pass
        # both prefill and decode
        else:
            prefill_num_tokens = paddle.sum(forward_meta.seq_lens_this_time[prefill_non_zeros_ids])
            decode_num_tokens = paddle.sum(forward_meta.seq_lens_this_time[decode_non_zeros_ids])

            self.prefill_info_dict["cu_seqlens_q"] = paddle.zeros(
                [self.prefill_len + 1], dtype=forward_meta.cu_seqlens_q.dtype
            )
            self.prefill_info_dict["cu_seqlens_q"][1:] = forward_meta.seq_lens_encoder[
                self.prefill_info_dict["batch_ids"], 0
            ]
            self.prefill_info_dict["cu_seqlens_q"] = paddle.cumsum(self.prefill_info_dict["cu_seqlens_q"]).astype(
                "int32"
            )

            self.prefill_qkv = paddle.zeros([prefill_num_tokens, self.total_hidden_dim], dtype=self.dtype)
            self.decode_qkv = paddle.zeros([decode_num_tokens, self.total_hidden_dim], dtype=self.dtype)
            self.merged_output = paddle.zeros(
                [prefill_num_tokens + decode_num_tokens, self.num_heads, self.head_dim], dtype=self.dtype
            )

            prefill_start, decode_start, start = 0, 0, 0
            non_zeros_ids = forward_meta.seq_lens_this_time != 0
            non_zeros_seq_lens = forward_meta.seq_lens_this_time[non_zeros_ids]
            end = non_zeros_seq_lens[0]
            if end > 1:
                last_stage = "prefill"
                prefill_end = end
                decode_end = 0
            else:
                last_stage = "decode"
                prefill_end = 0
                decode_end = end

            self.prefill_info_dict["id_group"] = []
            self.prefill_info_dict["reverse_id_group"] = []
            self.decode_info_dict["id_group"] = []
            self.decode_info_dict["reverse_id_group"] = []
            self.record_stages = []
            for seq_len in non_zeros_seq_lens[1:]:
                if seq_len > 1:
                    if last_stage == "decode":
                        self.record_stages.append((last_stage, len(self.decode_info_dict["id_group"])))
                        self.decode_info_dict["id_group"].append((decode_start, decode_end))
                        self.decode_info_dict["reverse_id_group"].append((start, end))
                        decode_start = decode_end
                        start = end
                        last_stage = "prefill"
                    prefill_end += seq_len
                    end += seq_len
                else:
                    if last_stage == "prefill":
                        self.record_stages.append((last_stage, len(self.prefill_info_dict["id_group"])))
                        self.prefill_info_dict["id_group"].append((prefill_start, prefill_end))
                        self.prefill_info_dict["reverse_id_group"].append((start, end))
                        prefill_start = prefill_end
                        start = end
                        last_stage = "decode"
                    decode_end += seq_len
                    end += seq_len

            if prefill_start < prefill_end:
                self.record_stages.append(("prefill", len(self.prefill_info_dict["id_group"])))
                self.prefill_info_dict["id_group"].append((prefill_start, prefill_end))
                self.prefill_info_dict["reverse_id_group"].append((start, end))
            if decode_start < decode_end:
                self.record_stages.append(("decode", len(self.decode_info_dict["id_group"])))
                self.decode_info_dict["id_group"].append((decode_start, decode_end))
                self.decode_info_dict["reverse_id_group"].append((start, end))

        self.batch_ids_prefill = paddle.to_tensor(self.prefill_info_dict["batch_ids"])
        self.batch_ids_decode = paddle.to_tensor(self.decode_info_dict["batch_ids"])
        self.seq_lens_dec = forward_meta.seq_lens_decoder[self.batch_ids_decode, 0]
        self.block_table_dec = forward_meta.block_tables[self.batch_ids_decode, :]
        # update prefilling rope
        self.update_rotary_embs_prefill(forward_meta)
        # update decoding rope
        self.update_rotary_embs_decoder(forward_meta)

    def update_rotary_embs_prefill(self, forward_meta: ForwardMeta):
        if self.batch_ids_prefill.shape[0] == 0 or forward_meta.rotary_embs is None:
            return

        batch_ids = self.batch_ids_prefill
        seq_lens_this_time = forward_meta.seq_lens_this_time[batch_ids]
        cached_kv_lens = forward_meta.seq_lens_decoder[batch_ids, 0]

        all_indices = []
        for i in range(len(batch_ids)):
            start_pos = cached_kv_lens[i]
            seq_len_i = seq_lens_this_time[i]
            if seq_len_i > 0:
                indices_i = paddle.arange(start_pos, start_pos + seq_len_i, dtype="int64")
                all_indices.append(indices_i)
        if not all_indices:
            return

        all_indices = paddle.concat(all_indices)  # [token_num]
        if self.enable_mm:
            gather_nd_indices = paddle.stack(
                [  # [token_num, 2]
                    paddle.repeat_interleave(batch_ids, repeats=seq_lens_this_time, axis=0),
                    all_indices,
                ],
                axis=1,
            )
            gathered_embs = paddle.gather_nd(
                forward_meta.rotary_embs.squeeze([2]).transpose(
                    [0, 2, 1, 3, 4]
                ),  # [B, 2, 1, S, 1, D // 2] -> [B, S, 2, 1, D // 2]
                gather_nd_indices,
            )  # [token_num, 2, 1, D // 2]
            rot_cos = gathered_embs[:, 0, :, :]  # [token_num, 1, D // 2]
            rot_sin = gathered_embs[:, 1, :, :]
        else:
            gathered_embs = paddle.gather(
                forward_meta.rotary_embs.squeeze([1]), all_indices, axis=1  # [2, 1, S, 1, D // 2] -> [2, S, 1, D // 2]
            )  # [2, token_num, 1, D // 2]
            rot_cos = gathered_embs[0, :, :, :]  # [token_num, 1, D // 2]
            rot_sin = gathered_embs[1, :, :, :]

        self.attention_metadata.rotary_cos_prefill = paddle.repeat_interleave(
            rot_cos, repeats=2, axis=-1
        )  # [token_num, 1, D]
        self.attention_metadata.rotary_sin_prefill = paddle.repeat_interleave(rot_sin, repeats=2, axis=-1)

    def update_rotary_embs_decoder(self, forward_meta: ForwardMeta):
        if not self.enable_mm:  # only initialize once for text-only model
            if self.attention_metadata.rotary_cos_decode is None or self.attention_metadata.rotary_sin_decode is None:
                self.attention_metadata.rotary_cos_decode = forward_meta.rotary_embs[0, 0, :, 0, :].astype(self.dtype)
                self.attention_metadata.rotary_sin_decode = forward_meta.rotary_embs[1, 0, :, 0, :].astype(self.dtype)
        elif self.batch_ids_decode.shape[0] > 0:
            bs = self.batch_ids_decode.shape[0]
            index = paddle.concat(
                [self.batch_ids_decode.view([-1, 1]), self.seq_lens_dec.to("int64").view([-1, 1])], axis=1
            )
            rot_cos = paddle.gather_nd(forward_meta.rotary_embs[:, 0, 0, :, 0, :], index).view([bs, 1, 1, -1])
            rot_sin = paddle.gather_nd(forward_meta.rotary_embs[:, 1, 0, :, 0, :], index).view([bs, 1, 1, -1])
            self.attention_metadata.rotary_cos_decode[:bs].copy_(paddle.repeat_interleave(rot_cos, repeats=2, axis=-1))
            self.attention_metadata.rotary_sin_decode[:bs].copy_(paddle.repeat_interleave(rot_sin, repeats=2, axis=-1))

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
        key_cache_shape = value_cache_shape = [max_num_blocks, self.block_size, self.kv_num_heads, self.head_dim]

        if kv_cache_quant_type is not None and kv_cache_quant_type == "int4_zp":
            key_cache_shape = value_cache_shape = [
                max_num_blocks,
                self.kv_num_heads,
                self.block_size,
                self.head_dim // 2,
            ]

        return key_cache_shape, value_cache_shape

    def apply_rope_native(self, qk, cos, sin):
        rotate_half = paddle.reshape(
            paddle.stack([-qk[..., 1::2], qk[..., 0::2]], axis=-1),
            paddle.shape(qk),
        )
        out = paddle.add(paddle.multiply(qk, cos), paddle.multiply(rotate_half, sin))
        return paddle.cast(out, qk.dtype)

    def split_pd_qkv(self, qkv):

        for ids, reverse_ids in zip(self.prefill_info_dict["id_group"], self.prefill_info_dict["reverse_id_group"]):
            self.prefill_qkv[ids[0] : ids[1], :] = qkv[reverse_ids[0] : reverse_ids[1], :]

        for ids, reverse_ids in zip(self.decode_info_dict["id_group"], self.decode_info_dict["reverse_id_group"]):
            self.decode_qkv[ids[0] : ids[1], :] = qkv[reverse_ids[0] : reverse_ids[1], :]

        return self.prefill_qkv, self.decode_qkv

    def merge_pd_output(self, prefill_out, decode_out):
        for stage, idx in self.record_stages:
            if stage == "prefill":
                ids = self.prefill_info_dict["id_group"][idx]
                reverse_ids = self.prefill_info_dict["reverse_id_group"][idx]
                self.merged_output[reverse_ids[0] : reverse_ids[1], :, :] = prefill_out[ids[0] : ids[1], :, :]
            else:
                ids = self.decode_info_dict["id_group"][idx]
                reverse_ids = self.decode_info_dict["reverse_id_group"][idx]
                self.merged_output[reverse_ids[0] : reverse_ids[1], :, :] = decode_out[ids[0] : ids[1], :, :]
        return self.merged_output

    def update_kv_cache(
        self, k, v, k_cache_id, v_cache_id, layer_id, forward_meta: ForwardMeta, specific_batch_ids=None
    ):
        tensor_start = 0
        for batch_idx in range(forward_meta.block_tables.shape[0]):
            if specific_batch_ids is not None and batch_idx not in specific_batch_ids:
                continue
            seq_len = forward_meta.seq_lens_this_time[batch_idx]
            if seq_len == 0:
                continue
            tensor_end = tensor_start + seq_len
            slice_trans_k = k[tensor_start:tensor_end, :, :]
            slice_trans_v = v[tensor_start:tensor_end, :, :]

            cur_block_tables = forward_meta.block_tables[batch_idx]
            cur_used_block_tables = cur_block_tables[cur_block_tables != -1]

            # encoder prefil
            if seq_len > 1:
                cache_start = 0
                cur_used_num_blocks = cur_used_block_tables.shape[0]

                for i, block_id in enumerate(cur_used_block_tables):

                    # last block: seq_len - cache_start <= block_size
                    if i == cur_used_num_blocks - 1:
                        cache_end = seq_len - cache_start
                        assert cache_end <= self.block_size

                        forward_meta.caches[k_cache_id][block_id, 0:cache_end, :, :] = slice_trans_k[
                            cache_start:seq_len, :, :
                        ]
                        forward_meta.caches[v_cache_id][block_id, 0:cache_end, :, :] = slice_trans_v[
                            cache_start:seq_len, :, :
                        ]
                        if layer_id == self.num_layers - 1:
                            self.record_block_table_metadata[batch_idx] = {
                                "block_id": block_id.item(),
                                "cache_end": cache_end,
                            }
                    # non last block: seq_lens_this_time > block_size
                    else:
                        assert seq_len > self.block_size
                        cache_end = cache_start + self.block_size
                        forward_meta.caches[k_cache_id][block_id] = slice_trans_k[cache_start:cache_end, :, :]
                        forward_meta.caches[v_cache_id][block_id] = slice_trans_v[cache_start:cache_end, :, :]
                        cache_start += self.block_size
            tensor_start = tensor_end

    def forward_prefill(self, prefill_qkv, layer_id, k_cache_id, v_cache_id, forward_meta: ForwardMeta):
        qkv = prefill_qkv.view([-1, self.num_heads + self.kv_num_heads * 2, self.head_dim])
        q, k, v = qkv.split(num_or_sections=[self.num_heads, self.kv_num_heads, self.kv_num_heads], axis=-2)
        q, k = apply_rope(q, k, self.attention_metadata.rotary_cos_prefill, self.attention_metadata.rotary_sin_prefill)

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

        self.update_kv_cache(k, v, k_cache_id, v_cache_id, layer_id, forward_meta, self.batch_ids_prefill)

        return prefill_out

    def forward_decode(self, decode_qkv, k_cache_id, v_cache_id, forward_meta: ForwardMeta):
        qkv = decode_qkv.view([-1, 1, self.num_heads + self.kv_num_heads * 2, self.head_dim])
        q, k, v = qkv.split(num_or_sections=[self.num_heads, self.kv_num_heads, self.kv_num_heads], axis=-2)

        if self.enable_mm:  # vl
            q, k = apply_rope(
                q, k, self.attention_metadata.rotary_cos_decode, self.attention_metadata.rotary_sin_decode
            )
            rotary_cos = None
            rotary_sin = None
        else:
            rotary_cos = self.attention_metadata.rotary_cos_decode
            rotary_sin = self.attention_metadata.rotary_sin_decode

        decode_out = flash_attn_kvcache_func(
            q,
            forward_meta.caches[k_cache_id],
            forward_meta.caches[v_cache_id],
            self.seq_lens_dec,
            self.block_table_dec,
            k,
            v,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            causal=self.causal,
            is_rotary_interleaved=True,
        )[0].squeeze(1)

        return decode_out

    @paddle.no_grad()
    def forward_native_backend(self, q, k, v, qkv, layer, forward_meta: ForwardMeta):

        layer_id = layer.layer_id
        k_cache_id = layer_id * 2
        v_cache_id = k_cache_id + 1

        if self.decode_len == 0:
            out = self.forward_prefill(qkv, layer_id, k_cache_id, v_cache_id, forward_meta)

        elif self.prefill_len == 0:
            out = self.forward_decode(qkv, k_cache_id, v_cache_id, forward_meta)

        else:
            prefill_qkv, decode_qkv = self.split_pd_qkv(qkv)
            prefill_output = self.forward_prefill(prefill_qkv, layer_id, k_cache_id, v_cache_id, forward_meta)
            decode_output = self.forward_decode(decode_qkv, k_cache_id, v_cache_id, forward_meta)
            out = self.merge_pd_output(prefill_output, decode_output)

        if qkv.dim() == 2:
            out = out.view([-1, self.num_heads * self.head_dim])

        return out
