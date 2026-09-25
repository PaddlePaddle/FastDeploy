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

import json
import math
import os
import weakref
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import paddle
from paddleformers.utils.log import logger

from fastdeploy import envs
from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention.flash_attn_backend import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
    flash_attn_func,
)
from fastdeploy.model_executor.layers.attention.ops import (
    decoder_write_cache_with_rope,
    get_block_shape_and_split_kv_block,
    gqa_rope_write_cache,
    init_signal_layerwise,
    pre_cache_len_concat,
)
from fastdeploy.platforms import current_platform

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta


_STAGE1_COARSE_WINDOWS_PER_SPLIT = 16
_STAGE1_MAX_CANDIDATE_BLOCKS = 2048
_STAGE2_BLOCKS_PER_SPLIT = 2
_PREFILL_QUERY_CHUNK_SIZE = 4096
_PREFILL_QUERY_TILE_SIZE = 128
_SELECTOR_REFERENCE_DEFINITION = (
    "top blocks by exact dense softmax attention mass, summed over the query heads sharing one KV head"
)


@dataclass
class InfLLMV2AttentionMetadata(FlashAttentionMetadata):
    """Transient InfLLM-V2 metadata for one model forward."""

    topk_indices: Optional[paddle.Tensor] = None
    block_scores: Optional[paddle.Tensor] = None
    selected_counts: Optional[paddle.Tensor] = None
    compressed_k: Optional[paddle.Tensor] = None
    compressed_k2: Optional[paddle.Tensor] = None
    kernel_size: int = 32
    kernel_stride: int = 16
    topk: int = 64
    dense_len: int = 8192
    block_size: int = 64
    init_blocks: int = 1
    local_blocks: int = 32
    selected_capacity: int = 128


def _config_value(fd_config: FDConfig, name: str, default):
    model_config = fd_config.model_config
    sparse_config = getattr(model_config, "sparse_config", None)
    if isinstance(sparse_config, dict) and name in sparse_config:
        return sparse_config[name]
    if hasattr(model_config, name):
        return getattr(model_config, name)
    if hasattr(fd_config, name):
        return getattr(fd_config, name)
    return default


class InfLLMV2AttentionBackend(FlashAttentionBackend):
    """MiniCPM4.1 CUDA backend implementing InfLLM-V2."""

    __infer_dynamic_dims_fields__ = ["attention_metadata"]
    attention_metadata: InfLLMV2AttentionMetadata

    def __init__(
        self,
        fd_config: FDConfig,
        kv_num_heads: int,
        num_heads: int,
        head_dim: int,
        encoder_block_shape_q: int = -1,
        decoder_block_shape_q: int = -1,
    ) -> None:
        super().__init__(
            fd_config=fd_config,
            kv_num_heads=kv_num_heads,
            num_heads=num_heads,
            head_dim=head_dim,
            encoder_block_shape_q=encoder_block_shape_q,
            decoder_block_shape_q=decoder_block_shape_q,
        )
        if current_platform.is_cuda() and paddle.device.cuda.get_device_capability()[0] < 8:
            raise ValueError("InfLLM-V2 requires NVIDIA compute capability 8.0 or newer.")

        self.kernel_size = int(_config_value(fd_config, "kernel_size", 32))
        self.kernel_stride = int(_config_value(fd_config, "kernel_stride", 16))
        self.topk = int(_config_value(fd_config, "topk", 64))
        self.dense_len = int(_config_value(fd_config, "dense_len", 8192))
        self.init_blocks = int(_config_value(fd_config, "init_blocks", 1))
        self.window_size = int(_config_value(fd_config, "window_size", 2048))
        self.sparse_prefill = bool(_config_value(fd_config, "sparse_prefill", True))
        self.prefill_query_chunk_size = int(
            _config_value(fd_config, "prefill_query_chunk_size", _PREFILL_QUERY_CHUNK_SIZE)
        )

        if self.kv_num_heads <= 0 or self.num_heads % self.kv_num_heads != 0:
            raise ValueError("InfLLM-V2 num_heads must be divisible by kv_num_heads.")
        if self.kernel_size <= 0 or self.kernel_stride <= 0:
            raise ValueError("InfLLM-V2 kernel_size and kernel_stride must be positive.")
        if self.block_size % self.kernel_stride != 0:
            raise ValueError("InfLLM-V2 block_size must be divisible by kernel_stride.")
        if self.block_size % (4 * self.kernel_stride) != 0:
            raise ValueError("InfLLM-V2 block_size must be divisible by 4 * kernel_stride.")
        if self.topk <= 0:
            raise ValueError("InfLLM-V2 topk must be positive.")
        if self.dense_len < 4 * self.kernel_size:
            raise ValueError("InfLLM-V2 dense_len must cover at least one coarse semantic window.")
        if self.init_blocks < 0 or self.init_blocks >= self.topk:
            raise ValueError("InfLLM-V2 init_blocks must be non-negative and smaller than topk.")
        if self.window_size < 0 or self.window_size % self.block_size != 0:
            raise ValueError("InfLLM-V2 window_size must be a non-negative multiple of block_size.")
        if self.prefill_query_chunk_size <= 0 or self.prefill_query_chunk_size % _PREFILL_QUERY_TILE_SIZE != 0:
            raise ValueError("InfLLM-V2 prefill_query_chunk_size must be a positive multiple of 128.")

        max_candidate_blocks = (self.max_seq_len + self.block_size - 1) // self.block_size
        if max_candidate_blocks > _STAGE1_MAX_CANDIDATE_BLOCKS:
            raise ValueError("InfLLM-V2 supports at most 2048 candidate blocks per request.")
        self.local_blocks = self.window_size // self.block_size
        self.selected_capacity = max(
            self.topk + self.local_blocks,
            (self.dense_len + self.block_size - 1) // self.block_size,
        )
        if self.selected_capacity * self.block_size > 8192:
            raise ValueError("InfLLM-V2 selected capacity may cover at most 8192 tokens.")

        tensor_parallel_size = int(fd_config.parallel_config.tensor_parallel_size)
        global_kv_heads = getattr(fd_config.model_config, "num_key_value_heads_list", None)
        if global_kv_heads is None:
            global_kv_heads = [
                int(
                    getattr(
                        fd_config.model_config,
                        "num_key_value_heads",
                        self.kv_num_heads * tensor_parallel_size,
                    )
                )
            ]
        if tensor_parallel_size > min(int(value) for value in global_kv_heads):
            raise ValueError("InfLLM-V2 does not support tensor parallel KV-head replication.")
        if self.speculative_method is not None:
            raise ValueError("InfLLM-V2 does not support speculative decoding.")

        graph_opt_config = getattr(fd_config, "graph_opt_config", None)
        if graph_opt_config is not None and getattr(graph_opt_config, "use_cudagraph", False):
            raise ValueError("InfLLM-V2 requires CUDA Graph to be disabled.")
        cache_config = fd_config.cache_config
        if getattr(cache_config, "enable_prefix_caching", False):
            raise ValueError("InfLLM-V2 does not support prefix caching.")
        if getattr(cache_config, "num_cpu_blocks", 0) or getattr(cache_config, "kvcache_storage_backend", None):
            raise ValueError("InfLLM-V2 does not support KV-cache offload.")
        scheduler_config = getattr(fd_config, "scheduler_config", None)
        if self.pd_disaggregation_mode not in (None, "None") or (
            scheduler_config is not None and getattr(scheduler_config, "splitwise_role", "mixed") != "mixed"
        ):
            raise ValueError("InfLLM-V2 does not support P/D disaggregation.")
        if envs.ENABLE_V1_KVCACHE_MANAGER:
            raise ValueError("InfLLM-V2 does not support the V1 KV-cache manager.")

        self._compressed_k: Optional[paddle.Tensor] = None
        self._compressed_k2: Optional[paddle.Tensor] = None
        self._compressed_cache_owner: Optional[weakref.ReferenceType[paddle.Tensor]] = None
        self._workspace_key = None
        self._topk_indices_ws: Optional[paddle.Tensor] = None
        self._block_scores_ws: Optional[paddle.Tensor] = None
        self._selected_counts_ws: Optional[paddle.Tensor] = None
        self._coarse_lse_ws: Optional[paddle.Tensor] = None
        self._coarse_partial_max_ws: Optional[paddle.Tensor] = None
        self._coarse_partial_sum_ws: Optional[paddle.Tensor] = None
        self._attention_out_ws: Optional[paddle.Tensor] = None
        self._partial_acc_ws: Optional[paddle.Tensor] = None
        self._partial_max_ws: Optional[paddle.Tensor] = None
        self._partial_sum_ws: Optional[paddle.Tensor] = None
        self._logged_sparse_activation = False
        self._logged_sparse_prefill_activation = False

        trace_path = os.getenv("FD_INFLLMV2_SELECTOR_TRACE_PATH")
        self._selector_trace_path = Path(trace_path).resolve() if trace_path else None
        self._selector_trace_rank = int(os.getenv("FD_INFLLMV2_SELECTOR_TRACE_RANK", "0"))
        self._selector_trace_layer = int(os.getenv("FD_INFLLMV2_SELECTOR_TRACE_LAYER", "0"))
        self._selector_trace_max_samples = int(os.getenv("FD_INFLLMV2_SELECTOR_TRACE_MAX_SAMPLES", "16"))
        self._selector_trace_samples = []
        self._selector_trace_last_position = {}
        self._selector_trace_request_index = {}
        self._selector_trace_next_request_index = 0
        if self._selector_trace_path is not None:
            if self._selector_trace_rank < 0 or self._selector_trace_layer < 0:
                raise ValueError("InfLLM-V2 selector trace rank and layer must be non-negative.")
            if self._selector_trace_max_samples <= 0:
                raise ValueError("InfLLM-V2 selector trace max samples must be positive.")
            if self._selector_trace_path.exists():
                raise FileExistsError(f"InfLLM-V2 selector trace already exists: {self._selector_trace_path}")
            if not self._selector_trace_path.parent.is_dir():
                raise FileNotFoundError(
                    f"InfLLM-V2 selector trace parent directory does not exist: {self._selector_trace_path.parent}"
                )

    def _split_qkv(self, qkv: paddle.Tensor):
        if qkv is None:
            raise ValueError("InfLLM-V2 sparse decode requires fused qkv input.")
        if len(qkv.shape) != 2:
            raise ValueError(f"fused qkv must have shape [tokens, width], got {list(qkv.shape)}.")
        q_width = self.num_heads * self.head_dim
        kv_width = self.kv_num_heads * self.head_dim
        expected_width = q_width + 2 * kv_width
        if qkv.shape[-1] != expected_width:
            raise ValueError(f"fused qkv last dimension must be {expected_width}, got {qkv.shape[-1]}.")
        q, k, v = paddle.split(qkv, [q_width, kv_width, kv_width], axis=-1)
        return (
            paddle.reshape(q, [-1, self.num_heads, self.head_dim]),
            paddle.reshape(k, [-1, self.kv_num_heads, self.head_dim]),
            paddle.reshape(v, [-1, self.kv_num_heads, self.head_dim]),
        )

    def get_kv_cache_shape(self, max_num_blocks: int, kv_cache_quant_type: Optional[str] = None):
        if kv_cache_quant_type not in (None, "none"):
            raise ValueError("InfLLM-V2 requires an unquantized paged KV cache.")
        shape = [max_num_blocks, self.kv_num_heads, self.block_size, self.head_dim]
        return shape, shape

    def get_additional_cache_block_bytes(self, cache_dtype_bytes: int) -> int:
        if cache_dtype_bytes <= 0:
            raise ValueError("cache_dtype_bytes must be positive.")
        semantic_slots = self.block_size // self.kernel_stride
        semantic_slots += self.block_size // (4 * self.kernel_stride)
        return cache_dtype_bytes * self.kv_num_heads * self.head_dim * semantic_slots

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        super().init_attention_metadata(forward_meta)
        base_metadata = self.attention_metadata
        metadata = InfLLMV2AttentionMetadata(
            **{field.name: getattr(base_metadata, field.name) for field in fields(FlashAttentionMetadata)}
        )
        metadata.compressed_k = self._compressed_k
        metadata.compressed_k2 = self._compressed_k2
        metadata.kernel_size = self.kernel_size
        metadata.kernel_stride = self.kernel_stride
        metadata.topk = self.topk
        metadata.dense_len = self.dense_len
        metadata.block_size = self.block_size
        metadata.init_blocks = self.init_blocks
        metadata.local_blocks = self.local_blocks
        metadata.selected_capacity = self.selected_capacity
        self.attention_metadata = metadata
        forward_meta.attn_metadata = metadata

    def reset_runtime_cache(self) -> None:
        self._compressed_k = None
        self._compressed_k2 = None
        self._compressed_cache_owner = None
        if hasattr(self, "attention_metadata"):
            self.attention_metadata.compressed_k = None
            self.attention_metadata.compressed_k2 = None

    def _get_layer_cache(self, layer: paddle.nn.Layer, forward_meta: ForwardMeta):
        if layer is None or not hasattr(layer, "layer_id"):
            raise ValueError("InfLLM-V2 requires an attention layer with layer_id.")
        if getattr(layer, "cache_quant_type_str", "none") != "none":
            raise ValueError("InfLLM-V2 requires an unquantized paged KV cache.")
        cache_index = 2 * layer.layer_id
        caches = forward_meta.caches
        if caches is None or len(caches) <= cache_index + 1:
            raise ValueError("InfLLM-V2 forward metadata does not contain this layer's K/V cache.")
        cache_k, cache_v = caches[cache_index], caches[cache_index + 1]
        if cache_k is None or cache_v is None:
            raise ValueError("InfLLM-V2 layer K/V cache must not be None.")
        return cache_k, cache_v

    def _ensure_compressed_cache(self, cache_k: paddle.Tensor):
        owner = self._compressed_cache_owner() if self._compressed_cache_owner is not None else None
        if owner is not None and owner._is_shared_buffer_with(cache_k):
            return self._compressed_k, self._compressed_k2
        fine_shape = [
            cache_k.shape[0],
            self.kv_num_heads,
            self.block_size // self.kernel_stride,
            self.head_dim,
        ]
        coarse_shape = [
            cache_k.shape[0],
            self.kv_num_heads,
            self.block_size // (4 * self.kernel_stride),
            self.head_dim,
        ]
        self._compressed_k = paddle.zeros(fine_shape, dtype=cache_k.dtype)
        self._compressed_k2 = paddle.zeros(coarse_shape, dtype=cache_k.dtype)
        self._compressed_cache_owner = weakref.ref(cache_k)
        self.attention_metadata.compressed_k = self._compressed_k
        self.attention_metadata.compressed_k2 = self._compressed_k2
        return self._compressed_k, self._compressed_k2

    def _ensure_workspace(
        self,
        query_tokens: int,
        max_blocks_per_seq: int,
        dtype,
        stage2_blocks_per_split: int = _STAGE2_BLOCKS_PER_SPLIT,
        allocate_attention: bool = True,
    ) -> None:
        if max_blocks_per_seq > _STAGE1_MAX_CANDIDATE_BLOCKS:
            raise ValueError("InfLLM-V2 Stage 1 supports at most 2048 candidate blocks.")
        if stage2_blocks_per_split <= 0:
            raise ValueError("InfLLM-V2 Stage 2 blocks per split must be positive.")
        key = (query_tokens, max_blocks_per_seq, dtype, stage2_blocks_per_split, allocate_attention)
        if key == self._workspace_key:
            return
        self._topk_indices_ws = paddle.empty([query_tokens, self.kv_num_heads, self.selected_capacity], dtype="int32")
        self._block_scores_ws = paddle.empty([query_tokens, self.kv_num_heads, max_blocks_per_seq], dtype="float32")
        self._selected_counts_ws = paddle.empty([query_tokens, self.kv_num_heads], dtype="int32")
        self._coarse_lse_ws = paddle.empty([query_tokens, self.num_heads], dtype="float32")
        max_visible_length = max_blocks_per_seq * self.block_size
        coarse_kernel = 4 * self.kernel_size
        coarse_stride = 4 * self.kernel_stride
        max_coarse_windows = (
            0 if max_visible_length < coarse_kernel else (max_visible_length - coarse_kernel) // coarse_stride + 1
        )
        coarse_splits = max(
            1,
            (max_coarse_windows + _STAGE1_COARSE_WINDOWS_PER_SPLIT - 1) // _STAGE1_COARSE_WINDOWS_PER_SPLIT,
        )
        partial_shape = [query_tokens, self.num_heads, coarse_splits]
        self._coarse_partial_max_ws = paddle.empty(partial_shape, dtype="float32")
        self._coarse_partial_sum_ws = paddle.empty(partial_shape, dtype="float32")
        if allocate_attention:
            self._attention_out_ws = paddle.empty([query_tokens, self.num_heads, self.head_dim], dtype=dtype)
            kv_splits = (self.selected_capacity + stage2_blocks_per_split - 1) // stage2_blocks_per_split
            self._partial_acc_ws = paddle.empty(
                [query_tokens, self.num_heads, kv_splits, self.head_dim], dtype="float32"
            )
            self._partial_max_ws = paddle.empty([query_tokens, self.num_heads, kv_splits], dtype="float32")
            self._partial_sum_ws = paddle.empty([query_tokens, self.num_heads, kv_splits], dtype="float32")
        else:
            self._attention_out_ws = None
            self._partial_acc_ws = None
            self._partial_max_ws = None
            self._partial_sum_ws = None
        self._workspace_key = key

    def _release_workspace(self) -> None:
        self._workspace_key = None
        self._topk_indices_ws = None
        self._block_scores_ws = None
        self._selected_counts_ws = None
        self._coarse_lse_ws = None
        self._coarse_partial_max_ws = None
        self._coarse_partial_sum_ws = None
        self._attention_out_ws = None
        self._partial_acc_ws = None
        self._partial_max_ws = None
        self._partial_sum_ws = None

    @staticmethod
    def _load_sparse_ops():
        try:
            from fastdeploy.model_executor.ops.gpu import (
                infllmv2_attention_forward,
                infllmv2_select_blocks,
                infllmv2_update_compressed_k,
            )
        except ImportError as exc:
            raise RuntimeError(
                "INFLLMV2_ATTN requires rebuilt infllmv2_update_compressed_k, "
                "infllmv2_select_blocks, and infllmv2_attention_forward custom ops."
            ) from exc
        return infllmv2_update_compressed_k, infllmv2_select_blocks, infllmv2_attention_forward

    def _update_compressed_cache(
        self, current_tokens: paddle.Tensor, cache_k: paddle.Tensor, forward_meta: ForwardMeta
    ) -> None:
        compressed_k, compressed_k2 = self._ensure_compressed_cache(cache_k)
        update_compressed_k, _, _ = self._load_sparse_ops()
        outputs = update_compressed_k(
            current_tokens,
            cache_k,
            compressed_k,
            compressed_k2,
            forward_meta.block_tables,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.batch_id_per_token,
            forward_meta.cu_seqlens_q,
            self.kernel_size,
            self.kernel_stride,
        )
        if isinstance(outputs, (tuple, list)):
            self._compressed_k, self._compressed_k2 = outputs
        self.attention_metadata.compressed_k = self._compressed_k
        self.attention_metadata.compressed_k2 = self._compressed_k2

    def _prepare_sparse_runtime(self, layer: paddle.nn.Layer, forward_meta: ForwardMeta) -> None:
        metadata = self.attention_metadata
        if self.pd_disaggregation_mode == "per_query":
            metadata.kv_signal_data_list[layer.layer_id] = init_signal_layerwise(
                metadata.kv_signal_metadata, layer.layer_id + self.start_layer_index
            )
        if int(os.getenv("USE_TBO", "0")) == 1 and hasattr(forward_meta, "tbo_microbatch_id"):
            os.environ["FLAGS_fmt_write_cache_completed_signal"] = str(forward_meta.tbo_microbatch_id)
        if layer.layer_id == 0:
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

    def _prepare_sparse_prefill_runtime(self, layer: paddle.nn.Layer, forward_meta: ForwardMeta) -> None:
        """Prepare the model-specific cache metadata required by sparse prefill."""
        self._prepare_sparse_runtime(layer, forward_meta)
        if layer.layer_id != 0 or forward_meta.max_len_tensor_cpu[1].item() <= 0:
            return
        (
            forward_meta.cu_seqlens_k,
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

    def _write_prefill_cache(
        self,
        qkv: paddle.Tensor,
        cache_k: paddle.Tensor,
        cache_v: paddle.Tensor,
        layer: paddle.nn.Layer,
        forward_meta: ForwardMeta,
    ):
        norm_after_rope = not getattr(layer, "qk_norm_before_rope", False)
        q_norm_weight = getattr(layer, "q_norm_weight", None) if norm_after_rope else None
        k_norm_weight = getattr(layer, "k_norm_weight", None) if norm_after_rope else None
        return gqa_rope_write_cache(
            qkv,
            cache_k,
            cache_v,
            forward_meta.cu_seqlens_q,
            forward_meta.cu_seqlens_k,
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
            q_norm_weight,
            k_norm_weight,
            None,
            None,
            getattr(layer, "cache_k_out_scale", None),
            getattr(layer, "cache_v_out_scale", None),
            getattr(layer, "cache_k_zp", None),
            getattr(layer, "cache_v_zp", None),
            self.attention_metadata.kv_signal_data_list[layer.layer_id],
            forward_meta.kv_token_num_cpu[0].item(),
            self.max_seq_len,
            getattr(layer, "rms_norm_eps", 1e-6),
            layer.use_neox_rotary_style,
            "none",
            self.rope_3d,
        )[:3]

    def _write_decode_cache(
        self,
        qkv: paddle.Tensor,
        cache_k: paddle.Tensor,
        cache_v: paddle.Tensor,
        layer: paddle.nn.Layer,
        forward_meta: ForwardMeta,
    ) -> paddle.Tensor:
        norm_after_rope = not getattr(layer, "qk_norm_before_rope", False)
        q_norm_weight = getattr(layer, "q_norm_weight", None) if norm_after_rope else None
        k_norm_weight = getattr(layer, "k_norm_weight", None) if norm_after_rope else None
        return decoder_write_cache_with_rope(
            qkv,
            cache_k,
            cache_v,
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.batch_id_per_token,
            forward_meta.cu_seqlens_q,
            forward_meta.block_tables,
            forward_meta.max_len_tensor_cpu,
            forward_meta.rotary_embs,
            layer.qkv_bias,
            getattr(layer, "cache_k_scale", None),
            getattr(layer, "cache_v_scale", None),
            getattr(layer, "cache_k_out_scale", None),
            getattr(layer, "cache_v_out_scale", None),
            getattr(layer, "cache_k_zp", None),
            getattr(layer, "cache_v_zp", None),
            self.attention_metadata.kv_signal_data_list[layer.layer_id],
            q_norm_weight,
            k_norm_weight,
            getattr(layer, "rms_norm_eps", 1e-6),
            "none",
            layer.use_neox_rotary_style,
            self.rope_3d,
            self.max_seq_len,
            getattr(layer, "quant_max_bound", 0.0),
            getattr(layer, "quant_min_bound", 0.0),
            False,
        )

    def _trace_rank_matches(self) -> bool:
        rank = int(os.getenv("PADDLE_TRAINER_ID", os.getenv("RANK", "0")))
        return rank == self._selector_trace_rank

    def _trace_request_slot(self, batch_id: int, query_position: int) -> int:
        last_position = self._selector_trace_last_position.get(batch_id)
        if last_position is None or query_position <= last_position:
            request_index = self._selector_trace_next_request_index
            self._selector_trace_next_request_index += 1
            self._selector_trace_request_index[batch_id] = request_index
        self._selector_trace_last_position[batch_id] = query_position
        return self._selector_trace_request_index[batch_id]

    def _write_selector_trace(self) -> None:
        payload = {
            "schema_version": 1,
            "kind": "infllmv2_selector_samples",
            "reference_definition": _SELECTOR_REFERENCE_DEFINITION,
            "rank": self._selector_trace_rank,
            "layer": self._selector_trace_layer,
            "block_size": self.block_size,
            "kernel_size": self.kernel_size,
            "kernel_stride": self.kernel_stride,
            "topk": self.topk,
            "dense_len": self.dense_len,
            "init_blocks": self.init_blocks,
            "local_blocks": self.local_blocks,
            "selected_capacity": self.selected_capacity,
            "samples": self._selector_trace_samples,
        }
        temporary_path = self._selector_trace_path.with_name(f".{self._selector_trace_path.name}.tmp")
        with temporary_path.open("w", encoding="utf-8") as trace_file:
            json.dump(payload, trace_file, indent=2, sort_keys=True)
            trace_file.write("\n")
        os.replace(temporary_path, self._selector_trace_path)

    @paddle.no_grad()
    def _record_selector_trace(
        self,
        post_rope_q: paddle.Tensor,
        cache_k: paddle.Tensor,
        forward_meta: ForwardMeta,
        layer: paddle.nn.Layer,
        topk_indices: paddle.Tensor,
        selected_counts: paddle.Tensor,
    ) -> None:
        if (
            self._selector_trace_path is None
            or layer.layer_id != self._selector_trace_layer
            or not self._trace_rank_matches()
            or forward_meta.is_dummy_or_profile_run
            or len(self._selector_trace_samples) >= self._selector_trace_max_samples
        ):
            return

        batch_ids = forward_meta.batch_id_per_token.numpy().tolist()
        seq_lens_decoder = forward_meta.seq_lens_decoder.numpy().tolist()
        cu_seqlens_q = forward_meta.cu_seqlens_q.numpy().tolist()
        counts = selected_counts.numpy()
        selected = topk_indices.numpy()
        group_size = self.num_heads // self.kv_num_heads
        attention_scale = 1.0 / math.sqrt(self.head_dim)

        for token_id, batch_id in enumerate(batch_ids):
            if len(self._selector_trace_samples) >= self._selector_trace_max_samples:
                break
            if batch_id < 0 or batch_id >= len(seq_lens_decoder):
                raise ValueError(f"InfLLM-V2 selector trace received invalid batch id {batch_id}.")
            query_offset = token_id - cu_seqlens_q[batch_id]
            query_position = seq_lens_decoder[batch_id] + query_offset
            visible_length = query_position + 1
            if visible_length < self.dense_len:
                continue
            valid_blocks = (visible_length + self.block_size - 1) // self.block_size
            physical_blocks = forward_meta.block_tables[batch_id, :valid_blocks].astype("int64")
            logical_k = paddle.index_select(cache_k, physical_blocks, axis=0)
            logical_k = paddle.transpose(logical_k, [1, 0, 2, 3])
            logical_k = paddle.reshape(logical_k, [self.kv_num_heads, valid_blocks * self.block_size, self.head_dim])
            logical_k = paddle.cast(logical_k[:, :visible_length, :], "float32")
            grouped_q = paddle.reshape(
                paddle.cast(post_rope_q[token_id], "float32"),
                [self.kv_num_heads, group_size, self.head_dim],
            )
            logits = paddle.matmul(grouped_q, paddle.transpose(logical_k, [0, 2, 1])) * attention_scale
            probabilities = paddle.nn.functional.softmax(logits, axis=-1)
            padded_tokens = valid_blocks * self.block_size - visible_length
            if padded_tokens:
                probabilities = paddle.nn.functional.pad(probabilities, [0, padded_tokens])
            block_mass = paddle.reshape(
                probabilities,
                [self.kv_num_heads, group_size, valid_blocks, self.block_size],
            ).sum(axis=[1, 3])
            request_index = self._trace_request_slot(batch_id, query_position)

            for kv_head in range(self.kv_num_heads):
                if len(self._selector_trace_samples) >= self._selector_trace_max_samples:
                    break
                selected_count = int(counts[token_id, kv_head])
                if selected_count <= 0 or selected_count > valid_blocks:
                    raise ValueError(
                        f"InfLLM-V2 selector trace selected count {selected_count} is invalid for {valid_blocks} blocks."
                    )
                selected_blocks = sorted(int(value) for value in selected[token_id, kv_head, :selected_count])
                reference = paddle.topk(block_mass[kv_head], k=selected_count, largest=True, sorted=False).indices
                reference_blocks = sorted(int(value) for value in reference.numpy().tolist())
                query_head_start = kv_head * group_size
                self._selector_trace_samples.append(
                    {
                        "rank": self._selector_trace_rank,
                        "layer": self._selector_trace_layer,
                        "request_index": request_index,
                        "query_index": token_id,
                        "query_offset": query_offset,
                        "query_position": query_position,
                        "kv_head": kv_head,
                        "query_head_start": query_head_start,
                        "query_head_end": query_head_start + group_size,
                        "block_size": self.block_size,
                        "topk": self.topk,
                        "selected_count": selected_count,
                        "selected_blocks": selected_blocks,
                        "reference_blocks": reference_blocks,
                        "reference_metric": _SELECTOR_REFERENCE_DEFINITION,
                    }
                )
        if self._selector_trace_samples:
            self._write_selector_trace()

    def _can_use_sparse_prefill(self, qkv: paddle.Tensor, forward_meta: ForwardMeta) -> bool:
        """Return whether this forward has the single-request causal layout supported by sparse prefill."""
        if not self.sparse_prefill or not self.causal or qkv is None:
            return False
        if forward_meta.block_tables.shape[0] != 1:
            return False
        if getattr(forward_meta, "attn_mask_offsets", None) is not None:
            return False
        if forward_meta.max_len_tensor_cpu[2].item() != 0:
            # A non-zero decoder prefix is the shared chunked-prefill path.  It
            # keeps the dense fallback until that path can supply block-aligned
            # per-chunk metadata without a device-to-host synchronization.
            return False
        prompt_tokens = int(qkv.shape[0])
        sparse_start = ((self.dense_len + self.block_size - 1) // self.block_size) * self.block_size
        selected_count = self.topk + self.local_blocks
        current_blocks = (_PREFILL_QUERY_TILE_SIZE + self.block_size - 1) // self.block_size
        has_full_selection = selected_count > current_blocks and sparse_start // self.block_size + 1 >= selected_count
        production_shape = (
            self.block_size == 64
            and self.head_dim == 128
            and self.num_heads // self.kv_num_heads == 16
            and qkv.dtype in (paddle.float16, paddle.bfloat16)
        )
        return production_shape and has_full_selection and prompt_tokens > sparse_start

    @staticmethod
    def _single_sequence_cu_seqlens(length: int, place) -> paddle.Tensor:
        return paddle.to_tensor([0, length], dtype="int32", place=place)

    def _dense_prefill_segment(
        self,
        query: paddle.Tensor,
        key: paddle.Tensor,
        value: paddle.Tensor,
    ) -> paddle.Tensor:
        query_length = int(query.shape[0])
        key_length = int(key.shape[0])
        cu_query = self._single_sequence_cu_seqlens(query_length, query.place)
        cu_key = self._single_sequence_cu_seqlens(key_length, query.place)
        return flash_attn_func(
            query,
            key,
            value,
            cu_query,
            cu_key,
            max_seqlen_q=query_length,
            max_seqlen_k=key_length,
            causal=True,
            num_heads=self.num_heads,
            kv_num_heads=self.kv_num_heads,
            head_dim=self.head_dim,
            version=2,
        )[0]

    def _select_prefill_query_blocks(
        self,
        query: paddle.Tensor,
        block_tables: paddle.Tensor,
        sparse_start: int,
        sparse_end: int,
    ) -> paddle.Tensor:
        sparse_tokens = sparse_end - sparse_start
        full_tiles = sparse_tokens // _PREFILL_QUERY_TILE_SIZE
        remainder = sparse_tokens % _PREFILL_QUERY_TILE_SIZE
        representative_parts = []
        position_parts = []
        if full_tiles:
            full_end = sparse_start + full_tiles * _PREFILL_QUERY_TILE_SIZE
            representative_parts.append(
                query[sparse_start + _PREFILL_QUERY_TILE_SIZE - 1 : full_end : _PREFILL_QUERY_TILE_SIZE]
            )
            position_parts.append(
                paddle.arange(
                    sparse_start + _PREFILL_QUERY_TILE_SIZE - 1,
                    full_end,
                    _PREFILL_QUERY_TILE_SIZE,
                    dtype="int32",
                )
            )
        if remainder:
            representative_parts.append(query[sparse_end - 1 : sparse_end])
            position_parts.append(paddle.to_tensor([sparse_end - 1], dtype="int32", place=query.place))
        representative_query = (
            representative_parts[0] if len(representative_parts) == 1 else paddle.concat(representative_parts, axis=0)
        )
        query_tiles = int(representative_query.shape[0])
        pseudo_block_tables = paddle.tile(block_tables[:1], [query_tiles, 1])
        pseudo_seq_lens_decoder = (
            position_parts[0] if len(position_parts) == 1 else paddle.concat(position_parts, axis=0)
        )
        pseudo_seq_lens_this_time = paddle.ones([query_tiles], dtype="int32")
        pseudo_batch_ids = paddle.arange(query_tiles, dtype="int32")
        pseudo_cu_seqlens_q = paddle.arange(query_tiles + 1, dtype="int32")

        self._ensure_workspace(
            query_tiles,
            int(block_tables.shape[1]),
            query.dtype,
            allocate_attention=False,
        )
        _, select_blocks, _ = self._load_sparse_ops()
        selection = select_blocks(
            representative_query,
            self._compressed_k,
            self._compressed_k2,
            pseudo_block_tables,
            pseudo_seq_lens_decoder,
            pseudo_seq_lens_this_time,
            pseudo_batch_ids,
            pseudo_cu_seqlens_q,
            self._topk_indices_ws,
            self._block_scores_ws,
            self._selected_counts_ws,
            self._coarse_lse_ws,
            self._coarse_partial_max_ws,
            self._coarse_partial_sum_ws,
            self.block_size,
            self.kernel_size,
            self.kernel_stride,
            self.topk,
            self.dense_len,
            self.init_blocks,
            self.local_blocks,
        )
        return selection[0]

    def _gather_prefill_cache_blocks(
        self,
        cache: paddle.Tensor,
        block_tables: paddle.Tensor,
        logical_blocks: paddle.Tensor,
    ) -> paddle.Tensor:
        query_tiles = int(logical_blocks.shape[0])
        selected_blocks = int(logical_blocks.shape[2])
        expanded_tables = paddle.tile(block_tables[:1].unsqueeze(1), [query_tiles, self.kv_num_heads, 1])
        physical_blocks = paddle.take_along_axis(expanded_tables, logical_blocks, axis=2)
        kv_heads = paddle.arange(self.kv_num_heads, dtype="int32").reshape([1, -1, 1])
        kv_heads = paddle.tile(kv_heads, [query_tiles, 1, selected_blocks])
        gather_indices = paddle.stack([physical_blocks, kv_heads], axis=-1)
        gathered = paddle.gather_nd(cache, gather_indices)
        return paddle.transpose(gathered, [0, 2, 3, 1, 4]).reshape(
            [query_tiles, selected_blocks * self.block_size, self.kv_num_heads, self.head_dim]
        )

    def _sparse_prefill_tile_batch(
        self,
        query: paddle.Tensor,
        key_cache: paddle.Tensor,
        value_cache: paddle.Tensor,
        block_tables: paddle.Tensor,
        selected_blocks: paddle.Tensor,
        first_position: int,
        query_tile_size: int,
    ) -> paddle.Tensor:
        query_tiles = int(selected_blocks.shape[0])
        current_blocks = (query_tile_size + self.block_size - 1) // self.block_size
        selected_count = self.topk + self.local_blocks
        history_count = selected_count - current_blocks
        if history_count <= 0:
            raise ValueError("InfLLM-V2 sparse prefill requires at least one selected history block.")

        # Stage 1 returns logical block IDs in ascending order and forces the
        # current logical blocks into the selection, so those blocks occupy the
        # final slots and the preceding slots are fully visible history.
        history = selected_blocks[:, :, :history_count]
        history_k = self._gather_prefill_cache_blocks(key_cache, block_tables, history)
        history_v = self._gather_prefill_cache_blocks(value_cache, block_tables, history)

        first_logical_block = first_position // self.block_size
        current = paddle.arange(
            first_logical_block,
            first_logical_block + query_tiles * current_blocks,
            dtype="int32",
        ).reshape([query_tiles, 1, current_blocks])
        current = paddle.tile(current, [1, self.kv_num_heads, 1])
        current_k = self._gather_prefill_cache_blocks(key_cache, block_tables, current)
        current_v = self._gather_prefill_cache_blocks(value_cache, block_tables, current)
        current_k = current_k[:, :query_tile_size]
        current_v = current_v[:, :query_tile_size]

        tiled_query = query.reshape([query_tiles, query_tile_size, self.num_heads, self.head_dim])
        history_output, _, history_lse, _ = paddle._C_ops.flash_attn(
            tiled_query,
            history_k,
            history_v,
            None,
            None,
            0.0,
            False,
            False,
            True,
            "",
        )
        current_output, _, current_lse, _ = paddle._C_ops.flash_attn(
            tiled_query,
            current_k,
            current_v,
            None,
            None,
            0.0,
            True,
            False,
            True,
            "",
        )
        history_lse = paddle.transpose(history_lse, [0, 2, 1]).unsqueeze(-1)
        current_lse = paddle.transpose(current_lse, [0, 2, 1]).unsqueeze(-1)
        maximum_lse = paddle.maximum(history_lse, current_lse)
        history_weight = paddle.exp(history_lse - maximum_lse).astype(query.dtype)
        current_weight = paddle.exp(current_lse - maximum_lse).astype(query.dtype)
        output = (history_output * history_weight + current_output * current_weight) / (
            history_weight + current_weight
        )
        return output.reshape([-1, self.num_heads, self.head_dim])

    def _sparse_prefill_attention(
        self,
        query: paddle.Tensor,
        key: paddle.Tensor,
        value: paddle.Tensor,
        key_cache: paddle.Tensor,
        value_cache: paddle.Tensor,
        block_tables: paddle.Tensor,
    ) -> paddle.Tensor:
        total_tokens = int(query.shape[0])
        sparse_start = ((self.dense_len + self.block_size - 1) // self.block_size) * self.block_size
        sparse_end = total_tokens
        outputs = [self._dense_prefill_segment(query[:sparse_start], key[:sparse_start], value[:sparse_start])]
        selected_blocks = self._select_prefill_query_blocks(query, block_tables, sparse_start, sparse_end)
        sparse_tokens = sparse_end - sparse_start
        full_tiles = sparse_tokens // _PREFILL_QUERY_TILE_SIZE
        tiles_per_chunk = self.prefill_query_chunk_size // _PREFILL_QUERY_TILE_SIZE
        for first_tile in range(0, full_tiles, tiles_per_chunk):
            last_tile = min(full_tiles, first_tile + tiles_per_chunk)
            token_begin = sparse_start + first_tile * _PREFILL_QUERY_TILE_SIZE
            token_end = sparse_start + last_tile * _PREFILL_QUERY_TILE_SIZE
            outputs.append(
                self._sparse_prefill_tile_batch(
                    query[token_begin:token_end],
                    key_cache,
                    value_cache,
                    block_tables,
                    selected_blocks[first_tile:last_tile],
                    token_begin,
                    _PREFILL_QUERY_TILE_SIZE,
                )
            )
        remainder = sparse_tokens % _PREFILL_QUERY_TILE_SIZE
        if remainder:
            token_begin = sparse_start + full_tiles * _PREFILL_QUERY_TILE_SIZE
            outputs.append(
                self._sparse_prefill_tile_batch(
                    query[token_begin:sparse_end],
                    key_cache,
                    value_cache,
                    block_tables,
                    selected_blocks[full_tiles : full_tiles + 1],
                    token_begin,
                    remainder,
                )
            )
        return paddle.concat(outputs, axis=0)

    def _forward_sparse_prefill(
        self, qkv: paddle.Tensor, layer: paddle.nn.Layer, forward_meta: ForwardMeta
    ) -> paddle.Tensor:
        cache_k, cache_v = self._get_layer_cache(layer, forward_meta)
        self._prepare_sparse_prefill_runtime(layer, forward_meta)
        post_rope_q, post_rope_k, post_rope_v = self._write_prefill_cache(qkv, cache_k, cache_v, layer, forward_meta)
        self._update_compressed_cache(qkv, cache_k, forward_meta)
        try:
            output = self._sparse_prefill_attention(
                post_rope_q,
                post_rope_k,
                post_rope_v,
                cache_k,
                cache_v,
                forward_meta.block_tables,
            )
        finally:
            self._release_workspace()
        if layer.layer_id == 0 and not self._logged_sparse_prefill_activation:
            logger.info(
                "InfLLM-V2 sparse prefill activated: "
                f"query_tile={_PREFILL_QUERY_TILE_SIZE}, query_chunk={self.prefill_query_chunk_size}, "
                f"selected_blocks={self.topk + self.local_blocks}, dense_len={self.dense_len}"
            )
            self._logged_sparse_prefill_activation = True
        return paddle.reshape(output, [-1, self.num_heads * self.head_dim])

    def _forward_sparse_decode(
        self, qkv: paddle.Tensor, layer: paddle.nn.Layer, forward_meta: ForwardMeta
    ) -> paddle.Tensor:
        cache_k, cache_v = self._get_layer_cache(layer, forward_meta)
        owner = self._compressed_cache_owner() if self._compressed_cache_owner is not None else None
        if owner is None or not owner._is_shared_buffer_with(cache_k):
            raise RuntimeError(
                "InfLLM-V2 sparse decode requires semantic K summaries initialized "
                "by prefill on the same paged KV cache."
            )
        self._prepare_sparse_runtime(layer, forward_meta)
        post_rope_qkv = self._write_decode_cache(qkv, cache_k, cache_v, layer, forward_meta)
        post_rope_q, _, _ = self._split_qkv(post_rope_qkv)
        self._update_compressed_cache(post_rope_qkv, cache_k, forward_meta)

        _, select_blocks, sparse_attention = self._load_sparse_ops()
        self._ensure_workspace(int(post_rope_q.shape[0]), int(forward_meta.block_tables.shape[1]), post_rope_q.dtype)
        selection = select_blocks(
            post_rope_q,
            self._compressed_k,
            self._compressed_k2,
            forward_meta.block_tables,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.batch_id_per_token,
            forward_meta.cu_seqlens_q,
            self._topk_indices_ws,
            self._block_scores_ws,
            self._selected_counts_ws,
            self._coarse_lse_ws,
            self._coarse_partial_max_ws,
            self._coarse_partial_sum_ws,
            self.block_size,
            self.kernel_size,
            self.kernel_stride,
            self.topk,
            self.dense_len,
            self.init_blocks,
            self.local_blocks,
        )
        metadata = self.attention_metadata
        metadata.topk_indices, metadata.block_scores, metadata.selected_counts = selection[:3]
        self._record_selector_trace(
            post_rope_q,
            cache_k,
            forward_meta,
            layer,
            metadata.topk_indices,
            metadata.selected_counts,
        )
        output = sparse_attention(
            post_rope_q,
            cache_k,
            cache_v,
            forward_meta.block_tables,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.batch_id_per_token,
            forward_meta.cu_seqlens_q,
            metadata.topk_indices,
            self._attention_out_ws,
            self._partial_acc_ws,
            self._partial_max_ws,
            self._partial_sum_ws,
        )
        if isinstance(output, (tuple, list)):
            output = output[0]
        if layer.layer_id == 0 and not self._logged_sparse_activation:
            logger.info(
                "InfLLM-V2 sparse decode activated: "
                f"block_size={self.block_size}, kernel={self.kernel_size}/{self.kernel_stride}, "
                f"topk={self.topk}, local_blocks={self.local_blocks}, dense_len={self.dense_len}"
            )
            self._logged_sparse_activation = True
        return paddle.reshape(output, [-1, self.num_heads * self.head_dim])

    def _forward_dense_and_update(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        qkv: paddle.Tensor,
        compressed_kv: paddle.Tensor,
        k_pe: paddle.Tensor,
        layer: paddle.nn.Layer,
        forward_meta: ForwardMeta,
    ) -> paddle.Tensor:
        output = super().forward_mixed(q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta)
        cache_k, _ = self._get_layer_cache(layer, forward_meta)
        current_tokens = qkv if qkv is not None else q
        if current_tokens is None:
            raise ValueError("InfLLM-V2 prefill requires qkv or q tokens for semantic-cache update.")
        self._update_compressed_cache(current_tokens, cache_k, forward_meta)
        return output

    def forward_decode(self, q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta):
        return self._forward_sparse_decode(qkv, layer, forward_meta)

    def forward_extend(self, q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta):
        if self._can_use_sparse_prefill(qkv, forward_meta):
            return self._forward_sparse_prefill(qkv, layer, forward_meta)
        return self._forward_dense_and_update(q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta)

    def forward_mixed(self, q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta):
        if getattr(forward_meta, "exist_prefill", False):
            if self._can_use_sparse_prefill(qkv, forward_meta):
                return self._forward_sparse_prefill(qkv, layer, forward_meta)
            return self._forward_dense_and_update(q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta)
        return self._forward_sparse_decode(qkv, layer, forward_meta)

    def forward_native_backend(self, q, k, v, qkv, layer, forward_meta):
        return self._forward_dense_and_update(q, k, v, qkv, None, None, layer, forward_meta)
