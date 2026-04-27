# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from dataclasses import dataclass

import paddle

from fastdeploy import envs
from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import MetaxForwardMeta
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)
from fastdeploy.model_executor.ops.gpu import (
    flash_attn_varlen_forward,
    flash_attn_kvcache_forward,
    write_cache_kv_with_rope,
    rotary_position_embedding,
    write_cache_kv,
)

if envs.FD_METAX_ENABLE_FA_SPLIT:
    @dataclass
    class FlashAttentionMetadata(AttentionMetadata):
        """
        FlashAttentionMetadata
        """

        num_requests: int = 0
        num_actual_tokens: int = 0

        num_prefills: int = 0
        num_prefill_tokens: int = 0
        prefill_query_start_loc: paddle.Tensor = None
        prefill_prefix_kv_lens: paddle.Tensor = None
        prefill_block_tables: paddle.Tensor = None

        num_decodes: int = 0
        num_decode_tokens: int = 0
        decode_cache_seq_lens: paddle.Tensor = None
        decode_block_tables: paddle.Tensor = None

        output: paddle.Tensor = None


    class MetaxFlashAttentionBackend(AttentionBackend):
        """
        FlashAttentionBackend backend implementation.
        """

        __infer_dynamic_dims_fields__ = ["attention_metadata"]
        attention_metadata: FlashAttentionMetadata
        enable_ids_reorder: bool = True

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
            self.attention_metadata: FlashAttentionMetadata = None
            self.max_seq_len: int = fd_config.model_config.max_model_len
            self.max_num_seqs: int = fd_config.scheduler_config.max_num_seqs
            self.causal: bool = getattr(fd_config.model_config, "causal", True)
            self.rope_3d: bool = getattr(fd_config.model_config, "rope_3d", False) or getattr(
                fd_config.model_config, "use_3d_rope", False
            )

            self.kv_num_heads: int = kv_num_heads
            self.num_heads: int = num_heads
            self.head_dim: int = fd_config.model_config.head_dim
            self.block_size: int = fd_config.cache_config.block_size
            self.num_layers: int = fd_config.model_config.num_hidden_layers
            self.softmax_scale: float = self.head_dim**-0.5

            self.speculative_method: str = fd_config.speculative_config.method
            self.use_speculate: bool = self.speculative_method is not None
            self.speculate_max_draft_token_num: int = fd_config.speculative_config.num_speculative_tokens
            self.keep_pd_step_flag: bool = fd_config.speculative_config.model_type == "mtp"
            self.num_layers_draft_model: int = int(fd_config.speculative_config.method in ["mtp"])
            self.start_layer_index: int = fd_config.model_config.start_layer_index

            self.block_tables_buffer = paddle.zeros(
                [
                    self.max_num_seqs,
                    self.max_seq_len // self.block_size + fd_config.cache_config.enc_dec_block_num,
                ],
                dtype=paddle.int32,
            )
            self.prefill_query_start_loc_buffer = paddle.zeros([self.max_num_seqs + 1], dtype=paddle.int32)
            self.prefill_prefix_kv_lens_buffer = paddle.zeros([self.max_num_seqs + 1], dtype=paddle.int32)
            self.decode_cache_seq_lens_buffer = paddle.zeros([self.max_num_seqs], dtype=paddle.int32)
            self.output_buffer = paddle.empty([self.max_seq_len, self.num_heads * self.head_dim], dtype=paddle.bfloat16)

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
            key_cache_shape = [max_num_blocks, self.block_size, self.kv_num_heads, self.head_dim]
            value_cache_shape = key_cache_shape
            return key_cache_shape, value_cache_shape

        def init_attention_metadata(self, forward_meta: MetaxForwardMeta):
            """Initialize attention metadata hence all layers in the forward pass can reuse it."""
            metadata = FlashAttentionMetadata()

            ids_remove_padding = forward_meta.ids_remove_padding
            rotary_embs = forward_meta.rotary_embs
            seq_lens_this_time = forward_meta.seq_lens_this_time
            seq_lens_encoder = forward_meta.seq_lens_encoder
            seq_lens_decoder = forward_meta.seq_lens_decoder
            batch_id_per_token = forward_meta.batch_id_per_token
            cu_seqlens_q = forward_meta.cu_seqlens_q
            cu_seqlens_k = forward_meta.cu_seqlens_k
            block_tables = forward_meta.block_tables

            if envs.FD_DEBUG:
                print(f"start_layer_index: {self.start_layer_index}")
                print(f"ids_remove_padding: {ids_remove_padding}")
                print(f"rotary_embs: {rotary_embs.shape}")
                print(f"seq_lens_encoder: {seq_lens_encoder}")
                print(f"seq_lens_decoder: {seq_lens_decoder}")
                print(f"seq_lens_this_time: {seq_lens_this_time}")
                print(f"batch_id_per_token: {batch_id_per_token}")
                print(f"cu_seqlens_q: {cu_seqlens_q}")
                print(f"cu_seqlens_k: {cu_seqlens_k}")
                print(f"block_tables: {block_tables}")

            request_batch_ids = paddle.where(seq_lens_this_time > 0)[0]
            prefill_batch_ids = paddle.where(seq_lens_encoder > 0)[0]

            num_requests = request_batch_ids.shape[0]
            num_prefills = prefill_batch_ids.shape[0]
            num_decodes = num_requests - num_prefills

            num_actual_tokens = ids_remove_padding.shape[0]
            if self.use_speculate:
                num_prefill_tokens = paddle.sum(seq_lens_encoder).item()
                num_decode_tokens = num_actual_tokens - num_prefill_tokens
            else:
                num_decode_tokens = num_decodes
                num_prefill_tokens = num_actual_tokens - num_decode_tokens

            request_block_tables = self.block_tables_buffer[:num_requests]
            prefill_query_start_loc = self.prefill_query_start_loc_buffer[: num_prefills + 1]
            prefill_prefix_kv_lens = self.prefill_prefix_kv_lens_buffer[: num_prefills + 1]
            decode_cache_seq_lens = self.decode_cache_seq_lens_buffer[:num_decodes]

            request_seq_lens_this_time = seq_lens_this_time[request_batch_ids].squeeze(-1)
            request_seq_lens_decoder = seq_lens_decoder[request_batch_ids].squeeze(-1)
            request_seq_lens = request_seq_lens_decoder + request_seq_lens_this_time
            
            request_block_tables.copy_(block_tables[request_batch_ids])

            if num_requests < self.max_num_seqs:
                self.block_tables_buffer[num_requests:] = self.block_tables_buffer[num_requests - 1]

            if num_prefills > 0:
                prefill_query_start_loc[1:].copy_(
                    request_seq_lens_this_time[num_decodes:].cumsum(axis=0, dtype=paddle.int32)
                )
                prefill_prefix_kv_lens[1:].copy_(request_seq_lens[num_decodes:].cumsum(axis=0, dtype=paddle.int32))

            if num_prefills < self.max_num_seqs:
                self.prefill_query_start_loc_buffer[num_prefills + 1 :] = prefill_query_start_loc[num_prefills]
                self.prefill_prefix_kv_lens_buffer[num_prefills + 1 :] = prefill_prefix_kv_lens[num_prefills]

            if num_decodes > 0:
                if not self.rope_3d:
                    decode_cache_seq_lens.copy_(request_seq_lens_decoder[:num_decodes])
                else:
                    decode_cache_seq_lens.copy_(request_seq_lens[:num_decodes])

            if num_decodes < self.max_num_seqs:
                self.decode_cache_seq_lens_buffer[num_decodes:] = self.decode_cache_seq_lens_buffer[num_decodes - 1]

            if envs.FD_DEBUG:
                print(f"num_requests: {num_requests} = {num_prefills} + {num_decodes}")
                print(f"num_actual_tokens: {num_actual_tokens} = {num_prefill_tokens} + {num_decode_tokens}")

            metadata.num_requests = num_requests
            metadata.num_actual_tokens = num_actual_tokens

            metadata.num_prefills = num_prefills
            metadata.num_prefill_tokens = num_prefill_tokens
            metadata.prefill_query_start_loc = prefill_query_start_loc
            metadata.prefill_prefix_kv_lens = prefill_prefix_kv_lens
            metadata.prefill_block_tables = request_block_tables[num_decodes:]

            metadata.num_decodes = num_decodes
            metadata.num_decode_tokens = num_decode_tokens
            metadata.decode_cache_seq_lens = decode_cache_seq_lens
            metadata.decode_block_tables = request_block_tables[:num_decodes]

            metadata.output = self.output_buffer[:num_actual_tokens]

            self.attention_metadata = metadata

        def _forward_extend(
            self,
            qkv: paddle.Tensor,
            layer: Attention,
            forward_meta: MetaxForwardMeta,
        ) -> paddle.Tensor:
            metadata = self.attention_metadata

            key_cache = forward_meta.caches[2 * layer.layer_id]
            value_cache = forward_meta.caches[2 * layer.layer_id + 1]

            rotary_position_embedding(
                qkv,
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
                forward_meta.batch_id_per_token,
                forward_meta.cu_seqlens_q,
                forward_meta.rotary_embs,
                self.num_heads,
                self.kv_num_heads,
                self.head_dim,
                self.max_seq_len,
                layer.use_neox_rotary_style,
                self.rope_3d,
            )

            write_cache_kv(
                qkv,
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
                forward_meta.batch_id_per_token,
                forward_meta.cu_seqlens_q,
                forward_meta.block_tables,
                key_cache,
                value_cache,
                self.num_heads,
                self.kv_num_heads,
                self.head_dim,
                self.max_seq_len,
            )

            prefill_qkv = qkv[metadata.num_decode_tokens :].view(
                [-1, self.num_heads + 2 * self.kv_num_heads, self.head_dim]
            )
            query, key, value = prefill_qkv.split([self.num_heads, self.kv_num_heads, self.kv_num_heads], axis=-2)

            flash_attn_varlen_forward(
                q=query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=metadata.prefill_query_start_loc,
                cu_seqlens_k=metadata.prefill_prefix_kv_lens,
                block_tables=metadata.prefill_block_tables,
                alibi_slopes=None,
                out=metadata.output[metadata.num_decode_tokens :],
                max_seqlen_q=self.max_seq_len,
                max_seqlen_k=self.max_seq_len,
                dropout_p=0.0,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
                window_size_left=-1,
                window_size_right=-1,
                softcap=0.0,
            )

            return metadata.output

        def _forward_decode(
            self,
            qkv: paddle.Tensor,
            layer: Attention,
            forward_meta: MetaxForwardMeta,
        ) -> paddle.Tensor:
            metadata = self.attention_metadata

            key_cache = forward_meta.caches[2 * layer.layer_id]
            value_cache = forward_meta.caches[2 * layer.layer_id + 1]

            if self.rope_3d:
                write_cache_kv_with_rope(
                    qkv,
                    forward_meta.seq_lens_encoder,
                    forward_meta.seq_lens_decoder,
                    forward_meta.batch_id_per_token,
                    forward_meta.cu_seqlens_q,
                    forward_meta.block_tables,
                    forward_meta.rotary_embs,
                    key_cache,
                    value_cache,
                    self.num_heads,
                    self.kv_num_heads,
                    self.head_dim,
                    self.max_seq_len,
                    layer.use_neox_rotary_style,
                    self.rope_3d,
                    self.use_speculate,
                )

                decode_qkv = qkv[: metadata.num_decode_tokens].view(
                    [metadata.num_decodes, -1, self.num_heads + 2 * self.kv_num_heads, self.head_dim]
                )
                query, key, value = decode_qkv[:, :, : self.num_heads, :], None, None
                rotary_embs = (None, None)

            else:
                decode_qkv = qkv[: metadata.num_decode_tokens].view(
                    [metadata.num_decodes, -1, self.num_heads + 2 * self.kv_num_heads, self.head_dim]
                )
                query, key, value = decode_qkv.split([self.num_heads, self.kv_num_heads, self.kv_num_heads], axis=-2)
                rotary_embs = (
                    forward_meta.rotary_embs_bf16[0, 0, :, 0, :],
                    forward_meta.rotary_embs_bf16[1, 0, :, 0, :],
                )

            flash_attn_kvcache_forward(
                q=query,
                k_cache=key_cache,
                v_cache=value_cache,
                k=key,
                v=value,
                rotary_cos=rotary_embs[0],
                rotary_sin=rotary_embs[1],
                cache_seqlens=metadata.decode_cache_seq_lens,
                cache_batch_idx=None,
                cache_leftpad=None,
                block_tables=metadata.decode_block_tables,
                alibi_slopes=None,
                out=metadata.output[: metadata.num_decode_tokens],
                softmax_scale=self.softmax_scale,
                causal=self.causal,
                window_size_left=-1,
                window_size_right=-1,
                softcap=0.0,
                rotary_interleaved=(not layer.use_neox_rotary_style),
                num_splits=1,
            )

            return metadata.output

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
            forward_meta: MetaxForwardMeta,
        ) -> paddle.Tensor:
            metadata = self.attention_metadata

            if metadata.num_prefills > 0:
                self._forward_extend(qkv, layer, forward_meta)

            if metadata.num_decodes > 0:
                self._forward_decode(qkv, layer, forward_meta)

            return metadata.output
else:
    @dataclass
    class FlashAttentionMetadata(AttentionMetadata):
        """
        FlashAttentionMetadata
        """

        num_requests: int = 0
        num_actual_tokens: int = 0
        query_start_loc: paddle.Tensor = None
        kv_start_loc: paddle.Tensor = None
        block_tables: paddle.Tensor = None
        output: paddle.Tensor = None


    class MetaxFlashAttentionBackend(AttentionBackend):
        """
        FlashAttentionBackend backend implementation.
        """

        __infer_dynamic_dims_fields__ = ["attention_metadata"]
        attention_metadata: FlashAttentionMetadata
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
            FlashAttentionBackend __init__
            """
            super().__init__()
            self.max_seq_len: int = fd_config.model_config.max_model_len
            self.max_num_seqs: int = fd_config.scheduler_config.max_num_seqs
            self.causal: bool = getattr(fd_config.model_config, "causal", True)
            self.rope_3d: bool = getattr(fd_config.model_config, "rope_3d", False) or getattr(
                fd_config.model_config, "use_3d_rope", False
            )

            self.kv_num_heads: int = kv_num_heads
            self.num_heads: int = num_heads
            self.head_dim: int = fd_config.model_config.head_dim
            self.block_size: int = fd_config.cache_config.block_size
            self.num_layers: int = fd_config.model_config.num_hidden_layers
            self.softmax_scale: float = self.head_dim**-0.5

            self.speculative_method: str = fd_config.speculative_config.method
            self.use_speculate: bool = self.speculative_method is not None
            self.speculate_max_draft_token_num: int = fd_config.speculative_config.num_speculative_tokens
            self.keep_pd_step_flag: bool = fd_config.speculative_config.model_type == "mtp"
            self.num_layers_draft_model: int = int(fd_config.speculative_config.method in ["mtp"])
            self.start_layer_index: int = fd_config.model_config.start_layer_index

            self.block_tables_buffer = paddle.zeros(
                [
                    self.max_num_seqs,
                    self.max_seq_len // self.block_size + fd_config.cache_config.enc_dec_block_num,
                ],
                dtype=paddle.int32,
            )
            self.query_start_loc_buffer = paddle.zeros([self.max_num_seqs + 1], dtype=paddle.int32)
            self.kv_start_loc_buffer = paddle.zeros([self.max_num_seqs + 1], dtype=paddle.int32)
            self.output_buffer = paddle.empty([self.max_seq_len, self.num_heads, self.head_dim], dtype=paddle.bfloat16)


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
            key_cache_shape = [max_num_blocks, self.block_size, self.kv_num_heads, self.head_dim]
            value_cache_shape = key_cache_shape
            return key_cache_shape, value_cache_shape

        def init_attention_metadata(self, forward_meta: MetaxForwardMeta):
            """Initialize attention metadata hence all layers in the forward pass can reuse it."""
            metadata = FlashAttentionMetadata()

            ids_remove_padding = forward_meta.ids_remove_padding
            rotary_embs = forward_meta.rotary_embs
            seq_lens_this_time = forward_meta.seq_lens_this_time
            seq_lens_encoder = forward_meta.seq_lens_encoder
            seq_lens_decoder = forward_meta.seq_lens_decoder
            batch_id_per_token = forward_meta.batch_id_per_token
            cu_seqlens_q = forward_meta.cu_seqlens_q
            block_tables = forward_meta.block_tables

            if envs.FD_DEBUG:
                print(f"start_layer_index: {self.start_layer_index}")
                print(f"ids_remove_padding: {ids_remove_padding}")
                print(f"rotary_embs: {rotary_embs.shape}")
                print(f"seq_lens_encoder: {seq_lens_encoder}")
                print(f"seq_lens_decoder: {seq_lens_decoder}")
                print(f"seq_lens_this_time: {seq_lens_this_time}")
                print(f"batch_id_per_token: {batch_id_per_token}")
                print(f"cu_seqlens_q: {cu_seqlens_q}")
                print(f"block_tables: {block_tables}")

            request_batch_ids = paddle.where(seq_lens_this_time > 0)[0]
            num_requests = request_batch_ids.shape[0]
            num_actual_tokens = ids_remove_padding.shape[0]

            request_seq_lens_this_time = seq_lens_this_time[request_batch_ids, 0]
            request_seq_lens_decoder = seq_lens_decoder[request_batch_ids, 0]
            request_seq_lens = request_seq_lens_decoder + request_seq_lens_this_time

            request_block_tables = self.block_tables_buffer[:num_requests]
            request_query_start_loc = self.query_start_loc_buffer[: num_requests + 1]
            request_kv_start_loc = self.kv_start_loc_buffer[: num_requests + 1]

            request_block_tables.copy_(block_tables[request_batch_ids])

            request_query_start_loc[1:].copy_(request_seq_lens_this_time.cumsum(axis=0, dtype=paddle.int32))
            request_kv_start_loc[1:].copy_(request_seq_lens.cumsum(axis=0, dtype=paddle.int32))
            
            if num_requests < self.max_num_seqs:
                self.block_tables_buffer[num_requests:] = self.block_tables_buffer[num_requests - 1]
                self.query_start_loc_buffer[num_requests + 1 :] = self.query_start_loc_buffer[num_requests]
                self.kv_start_loc_buffer[num_requests + 1 :] = self.kv_start_loc_buffer[num_requests]

            metadata.num_requests = num_requests
            metadata.num_actual_tokens = num_actual_tokens
            metadata.query_start_loc = request_query_start_loc
            metadata.kv_start_loc = request_kv_start_loc
            metadata.block_tables = request_block_tables
            metadata.output = self.output_buffer[:num_actual_tokens]

            self.attention_metadata: AttentionMetadata = metadata

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
            forward_meta: MetaxForwardMeta,
        ) -> paddle.Tensor:
            metadata = self.attention_metadata

            key_cache = forward_meta.caches[2 * layer.layer_id]
            value_cache = forward_meta.caches[2 * layer.layer_id + 1]

            rotary_position_embedding(
                qkv,
                None,
                forward_meta.seq_lens_decoder,
                forward_meta.batch_id_per_token,
                forward_meta.cu_seqlens_q,
                forward_meta.rotary_embs,
                self.num_heads,
                self.kv_num_heads,
                self.head_dim,
                self.max_seq_len,
                layer.use_neox_rotary_style,
                self.rope_3d,
            )

            write_cache_kv(
                qkv,
                None,
                forward_meta.seq_lens_decoder,
                forward_meta.batch_id_per_token,
                forward_meta.cu_seqlens_q,
                forward_meta.block_tables,
                key_cache,
                value_cache,
                self.num_heads,
                self.kv_num_heads,
                self.head_dim,
                self.max_seq_len,
            )

            query = qkv.view([-1, self.num_heads + 2 * self.kv_num_heads, self.head_dim])[:, : self.num_heads, :]

            output = metadata.output

            flash_attn_varlen_forward(
                q=query,
                k=key_cache,
                v=value_cache,
                out=output,
                cu_seqlens_q=metadata.query_start_loc,
                cu_seqlens_k=metadata.kv_start_loc,
                block_tables=metadata.block_tables,
                alibi_slopes=None,
                max_seqlen_q=self.max_seq_len,
                max_seqlen_k=self.max_seq_len,
                dropout_p=0.0,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
                window_size_left=-1,
                window_size_right=-1,
                softcap=0.0,
            )

            output = output.view([-1, self.num_heads * self.head_dim])

            return output
