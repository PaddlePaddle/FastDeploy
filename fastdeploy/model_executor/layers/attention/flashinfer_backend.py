"""
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
"""

import numpy as np
import paddle

paddle.compat.enable_torch_proxy(scope={"flashinfer"})
import triton
import triton.language as tl
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)
from flashinfer.decode import fast_decode_plan
from triton import cdiv

from fastdeploy import envs
from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
)
from fastdeploy.model_executor.layers.attention.utils import init_rank_and_device_id
from fastdeploy.model_executor.ops.gpu import (
    fused_rotary_position_encoding,
    reshape_and_cache_flash,
)
from fastdeploy.worker.gpu.buffer_utils import CpuGpuBuffer


def _per_head_rms_norm(
    x: paddle.Tensor,
    weight: paddle.Tensor,
    eps: float = 1e-6,
) -> paddle.Tensor:
    """Apply per-head RMSNorm.

    Args:
        x: [num_tokens, num_heads, head_dim]
        weight: [head_dim] - learnable scale parameter
        eps: epsilon for numerical stability

    Returns:
        Normalized tensor with same shape as x.
    """
    orig_dtype = x.dtype
    x_fp32 = x.astype("float32")
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    x_normed = x_fp32 * paddle.rsqrt(variance + eps)
    return (x_normed * weight.astype("float32")).astype(orig_dtype)


class FlashInferAttentionBackend(AttentionBackend):
    def __init__(self, fd_config: FDConfig, kv_num_heads: int, num_heads: int, head_dim: int):
        super().__init__()
        self.fd_config = fd_config
        self.max_seq_len: int = fd_config.model_config.max_model_len
        self.num_kv_heads: int = kv_num_heads
        self.num_qo_heads: int = num_heads
        self.head_dim: int = fd_config.model_config.head_dim
        self.block_size: int = fd_config.cache_config.block_size
        self.num_layers: int = fd_config.model_config.num_hidden_layers
        self.rank, self.device_id = init_rank_and_device_id(fd_config)
        self.device = f"cuda:{self.device_id}"
        self.pin_memory = True

        self.sm_scale: float = float(1.0 / (self.head_dim**0.5))
        self.window_left: int = getattr(fd_config.model_config, "window_size", 0)
        self.logits_soft_cap: float = 0.0

        self.reorder_batch_threshold = 1
        if envs.FD_ENABLE_RL:
            self.decode_fixed_split_size = 2048
            self.prefill_fixed_split_size = 4096
            self.disable_split_kv = True
        else:
            self.decode_fixed_split_size = -1
            self.prefill_fixed_split_size = -1
            self.disable_split_kv = False

        self.q_data_dtype = "bfloat16"
        self.kv_cache_dtype = "bfloat16"
        self.o_data_dtype = "bfloat16"

        # Shared workspace buffer for FlashInfer kernels (~394 MB)
        self._workspace_buffer = paddle.empty([394 * 1024 * 1024], dtype=paddle.uint8)

        # Speculative config
        self.speculative_config = fd_config.speculative_config
        self.num_spec_tokens = getattr(self.speculative_config, "num_speculative_tokens", 0)

        self.max_num_seqs = fd_config.scheduler_config.max_num_seqs
        max_num_blocks_per_req = cdiv(fd_config.model_config.max_model_len, fd_config.cache_config.block_size)
        self.max_num_blocks = self.max_num_seqs * max_num_blocks_per_req
        self.paged_kv_indptr = self._make_buffer(self.max_num_seqs + 1)
        self.paged_kv_indptr_cpu_buffer = paddle.zeros_like(self.paged_kv_indptr.cpu)
        self.paged_kv_indices = self._make_buffer(self.max_num_blocks)
        self.paged_kv_last_page_len = self._make_buffer(self.max_num_seqs)

        # DCP world size (default to 1 if not available)
        self.dcp_world_size = getattr(fd_config.parallel_config, "data_parallel_size", 1)

        # Dummy scale tensors for non-fp8 mode (C++ kernel requires non-None tensors)
        self._dummy_scale = paddle.empty([1], dtype=paddle.float32)

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        # For full cudagraph capture, one `decode_wrapper` for each batch
        # size is needed for FlashInfer.
        self._decode_wrappers_cudagraph: dict[int, BatchDecodeWithPagedKVCacheWrapper] = {}
        self._decode_wrapper = None
        self._decode_cudagraph_max_bs = min((1 + self.num_spec_tokens) * self.max_num_seqs, 512)

        num_seqs = forward_meta.input_batch.num_seqs
        num_decodes = forward_meta.input_batch.num_decodes
        num_prefills = forward_meta.input_batch.num_prefills
        self.num_prefills = num_prefills
        self.num_decode_tokens = forward_meta.input_batch.num_decode_tokens  # 投机解码下取上限
        self.num_prefill_tokens = forward_meta.input_batch.num_prefill_tokens

        seq_lens_np = forward_meta.input_batch.seq_lens_np
        qo_indptr_cpu = paddle.to_tensor(forward_meta.input_batch.query_start_loc_np, place=paddle.CPUPlace())

        block_table_tensor = forward_meta.block_table_tensor

        num_blocks_np = (seq_lens_np + (self.block_size - 1)) // self.block_size
        paged_kv_indices = self._compute_flashinfer_kv_metadata(
            num_blocks_np,
            seq_lens_np,
            block_table_tensor,
            num_seqs,
            self.block_size,
        )

        if num_prefills > 0:
            # Slices for shared prefill metadata
            prefill_start = num_decodes
            qo_indptr_prefill_cpu = qo_indptr_cpu[prefill_start:] - qo_indptr_cpu[prefill_start]
            assert qo_indptr_prefill_cpu.shape[0] == num_prefills + 1

            self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(self._workspace_buffer, "NHD", backend="fa2")

            # Slicing CPU buffers that are only needed for FI native prefills
            paged_kv_last_page_len_prefill_cpu = self.paged_kv_last_page_len.cpu[prefill_start:num_seqs]
            assert paged_kv_last_page_len_prefill_cpu.shape[0] == num_prefills
            paged_kv_indptr_prefill_cpu = self.paged_kv_indptr.cpu[prefill_start : num_seqs + 1]
            assert paged_kv_indptr_prefill_cpu.shape[0] == num_prefills + 1

            self.prefill_wrapper.plan(
                qo_indptr=qo_indptr_prefill_cpu,
                paged_kv_indptr=paged_kv_indptr_prefill_cpu,
                paged_kv_indices=paged_kv_indices,
                paged_kv_last_page_len=paged_kv_last_page_len_prefill_cpu,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_dim,
                page_size=self.block_size,
                causal=True,
                sm_scale=self.sm_scale,
                window_left=self.window_left,
                logits_soft_cap=self.logits_soft_cap,
                q_data_type=self.q_data_dtype,
                kv_data_type=self.kv_cache_dtype,
                fixed_split_size=self.prefill_fixed_split_size,
                disable_split_kv=self.disable_split_kv,
            )

        if num_decodes > 0:
            assert seq_lens_np is not None
            pure_decode = num_prefills == 0
            use_cudagraph = (
                forward_meta.step_use_cudagraph
                and pure_decode
                and self.num_decode_tokens <= self._decode_cudagraph_max_bs
            )
            num_input_tokens = self.num_decode_tokens

            self.decode_wrapper = self._get_decode_wrapper(num_input_tokens, use_cudagraph)
            # Use the persistent buffer with padding length,
            # instead of the same address but chunked version
            # in atten_metadata when using cudagraph.
            fast_plan_decode(
                self.decode_wrapper,
                indptr_cpu=self.paged_kv_indptr.cpu[: num_input_tokens + 1],
                indices=paged_kv_indices,
                last_page_len_cpu=self.paged_kv_last_page_len.cpu[:num_input_tokens],
                num_qo_heads=self.num_qo_heads * self.dcp_world_size,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=self.block_size,
                pos_encoding_mode="NONE",
                sm_scale=self.sm_scale,
                window_left=self.window_left,
                logits_soft_cap=self.logits_soft_cap,
                q_data_type=self.q_data_dtype,
                kv_data_type=self.kv_cache_dtype,
                fixed_split_size=self.decode_fixed_split_size,
                disable_split_kv=self.disable_split_kv,
            )

    def _get_decode_wrapper(self, batch_size: int, use_cudagraph: bool = False):
        if use_cudagraph:
            decode_wrapper = self._decode_wrappers_cudagraph.get(batch_size, None)
        else:
            decode_wrapper = self._decode_wrapper

        if decode_wrapper is None:
            if use_cudagraph:
                paged_kv_indptr = self.paged_kv_indptr.gpu[: batch_size + 1]
                paged_kv_indices = self.paged_kv_indices.gpu
                paged_kv_last_page_len = self.paged_kv_last_page_len.gpu[:batch_size]
            else:
                paged_kv_indptr = None
                paged_kv_indices = None
                paged_kv_last_page_len = None
            decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self._workspace_buffer,
                "NHD",
                use_cuda_graph=use_cudagraph,
                paged_kv_indptr_buffer=paged_kv_indptr,
                paged_kv_indices_buffer=paged_kv_indices,
                paged_kv_last_page_len_buffer=paged_kv_last_page_len,
                # Tensor cores are enabled by default because the perf would be
                # at least as good as cuda cores for all attention ops in latest
                # gpus.
                use_tensor_cores=True,
            )

            # save the decode wrapper
            if use_cudagraph:
                self._decode_wrappers_cudagraph[batch_size] = decode_wrapper
            else:
                self._decode_wrapper = decode_wrapper

        return decode_wrapper

    def _make_buffer(self, *size: int, dtype: paddle.dtype = paddle.int32) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *size,
            dtype=dtype,
            pin_memory=self.pin_memory,
            with_numpy=True,
        )

    def _compute_flashinfer_kv_metadata(
        self,
        num_blocks_np: np.ndarray,
        seq_lens_np: np.ndarray,
        block_table_tensor: paddle.Tensor,
        num_seqs: int,
        page_size: int,
    ) -> paddle.Tensor:
        """
        Compute paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len for FlashInfer
        attention.

        Results are stored in self.paged_kv_indptr,
        self.paged_kv_indices, self.paged_kv_last_page_len buffers.

        Returns paged_kv_indices, a GPU tensor with shape [num_actual_pages].
        """
        # write self.paged_kv_indptr_cpu inplace (0-index is always 0)
        np.cumsum(
            num_blocks_np,
            dtype=np.int32,
            out=self.paged_kv_indptr.np[1 : num_seqs + 1],
        )
        # after this line (e.g., for cuda graphs), we need to copy the data to
        # self.paged_kv_indptr_buffer to avoid race condition.
        self.paged_kv_indptr_cpu_buffer[: num_seqs + 1] = self.paged_kv_indptr.cpu[: num_seqs + 1]
        paged_kv_indptr = self.paged_kv_indptr.gpu[: num_seqs + 1]
        paged_kv_indptr.copy_(self.paged_kv_indptr_cpu_buffer[: num_seqs + 1], non_blocking=True)

        # write self.paged_kv_indices inplace
        num_actual_pages = self.paged_kv_indptr.np[num_seqs]
        paged_kv_indices = self.paged_kv_indices.gpu[:num_actual_pages]
        _copy_page_indices_kernel[(num_seqs,)](
            paged_kv_indices,
            block_table_tensor,
            block_table_tensor.stride(0),
            paged_kv_indptr,
            BLOCK_SIZE=1024,
        )

        # write self.paged_kv_last_page_len_cpu inplace
        paged_kv_last_page_len_np = seq_lens_np % page_size
        self.paged_kv_last_page_len.np[:num_seqs] = np.where(
            (paged_kv_last_page_len_np == 0) & (seq_lens_np != 0),
            page_size,
            paged_kv_last_page_len_np,
        )
        self.paged_kv_last_page_len.gpu[:num_seqs].copy_(self.paged_kv_last_page_len.cpu[:num_seqs], non_blocking=True)
        return paged_kv_indices

    def get_kv_cache_shape(
        self,
        max_num_blocks: int,
        kv_cache_quant_type: str = None,
    ):
        """
        KV cache layout:  [num_blocks, block_size, num_kv_heads, head_dim]  (NHD)

        This matches the memory layout expected by reshape_and_cache_flash and by
        the BatchPrefill/DecodeWrapper when initialized with kv_layout="NHD".
        """
        shape = [max_num_blocks, self.block_size, self.num_kv_heads, self.head_dim]
        return shape, shape  # key_cache_shape, value_cache_shape

    def forward_mixed(
        self,
        query,
        key,
        value,
        qkv,
        compressed_kv,
        k_pe,
        layer: Attention,
        forward_meta: ForwardMeta,
    ) -> paddle.Tensor:
        # Split qkv if q/k/v not provided separately
        if query is None and qkv is not None:
            num_tokens = qkv.shape[0]
            q_size = layer.num_heads * layer.head_dim
            kv_size = layer.kv_num_heads * layer.head_dim
            query = qkv[:, :q_size].reshape([num_tokens, layer.num_heads, layer.head_dim])
            key = qkv[:, q_size : q_size + kv_size].reshape([num_tokens, layer.kv_num_heads, layer.head_dim])
            value = qkv[:, q_size + kv_size : q_size + 2 * kv_size].reshape(
                [num_tokens, layer.kv_num_heads, layer.head_dim]
            )

        # Apply RoPE (rotary positional embedding) before writing KV cache.
        # FlashInfer uses pos_encoding_mode="NONE", so RoPE must be applied externally.
        if forward_meta.positions is not None and forward_meta.cos_sin_cache is not None:
            num_tokens = query.shape[0]
            # fused_rotary_position_encoding is in-place; ensure q/k are contiguous.
            q_flat = query.reshape([num_tokens, -1]).contiguous()
            k_flat = key.reshape([num_tokens, -1]).contiguous()
            fused_rotary_position_encoding(
                q_flat,
                k_flat,
                forward_meta.positions.cast(paddle.int32),
                forward_meta.cos_sin_cache,
                layer.head_dim,
                layer.use_neox_rotary_style,
            )
            query = q_flat.reshape([num_tokens, layer.num_heads, layer.head_dim])
            key = k_flat.reshape([num_tokens, layer.kv_num_heads, layer.head_dim])

        # Apply QK Norm after RoPE (when qk_norm_before_rope=False and use_qk_norm=True).
        # In V0, this is done inside the fused append_attention kernel.
        if getattr(layer, "use_qk_norm", False) and not getattr(layer, "qk_norm_before_rope", False):
            q_norm_weight = getattr(layer, "q_norm_weight", None)
            k_norm_weight = getattr(layer, "k_norm_weight", None)
            if q_norm_weight is not None:
                eps = getattr(layer, "rms_norm_eps", 1e-6)
                query = _per_head_rms_norm(query, q_norm_weight, eps)
                key = _per_head_rms_norm(key, k_norm_weight, eps)

        # Determine cache layout: 4 entries per layer (fp8) vs 2 entries per layer
        num_caches = len(forward_meta.caches)
        is_block_wise_fp8 = num_caches == 4 * self.num_layers

        if is_block_wise_fp8:
            cache_k = forward_meta.caches[4 * layer.layer_id]
            cache_v = forward_meta.caches[4 * layer.layer_id + 1]
            cache_k_scales = forward_meta.caches[4 * layer.layer_id + 2]
            cache_v_scales = forward_meta.caches[4 * layer.layer_id + 3]
            kv_cache_dtype = "fp8_e4m3"
        else:
            cache_k = forward_meta.caches[2 * layer.layer_id]
            cache_v = forward_meta.caches[2 * layer.layer_id + 1]
            if not hasattr(self, "_dummy_scale") or self._dummy_scale is None:
                self._dummy_scale = paddle.empty([1], dtype=paddle.float32)
            cache_k_scales = self._dummy_scale
            cache_v_scales = self._dummy_scale
            kv_cache_dtype = "auto"

        reshape_and_cache_flash(
            key,
            value,
            cache_k,
            cache_v,
            forward_meta.slot_mapping,
            cache_k_scales,
            cache_v_scales,
            kv_cache_dtype,
        )

        # For FlashInfer wrapper, pass None scales when not using fp8
        fi_k_scale = cache_k_scales if is_block_wise_fp8 else None
        fi_v_scale = cache_v_scales if is_block_wise_fp8 else None

        output = paddle.empty([query.shape[0], layer.num_heads, layer.head_dim], dtype=query.dtype)

        if self.num_prefill_tokens > 0:
            prefill_query = query[: self.num_prefill_tokens]
            self.prefill_wrapper.run(
                prefill_query,
                (cache_k, cache_v),
                k_scale=fi_k_scale,
                v_scale=fi_v_scale,
                out=output[: self.num_prefill_tokens],
            )

        if self.num_decode_tokens > 0:
            decode_query = query[self.num_prefill_tokens :]
            self.decode_wrapper.run(
                decode_query,
                (cache_k, cache_v),
                k_scale=fi_k_scale,
                v_scale=fi_v_scale,
                out=output[self.num_prefill_tokens :],
            )

        return output.reshape([query.shape[0], -1])

    def forward_decode(
        self,
        query,
        key,
        value,
        qkv,
        compressed_kv,
        k_pe,
        layer: Attention,
        forward_meta: ForwardMeta,
    ) -> paddle.Tensor:
        return self.forward_mixed(query, key, value, qkv, compressed_kv, k_pe, layer, forward_meta)


def fast_plan_decode(
    decode_wrapper,
    indptr_cpu: paddle.Tensor,
    indices: paddle.Tensor,
    last_page_len_cpu: paddle.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    pos_encoding_mode: str = "NONE",
    window_left: int = -1,
    logits_soft_cap: float | None = None,
    q_data_type: str | paddle.dtype | None = "float16",
    kv_data_type: str | paddle.dtype | None = None,
    data_type: str | paddle.dtype | None = None,
    sm_scale: float | None = None,
    rope_scale: float | None = None,
    rope_theta: float | None = None,
    non_blocking: bool = True,
    fixed_split_size: int = -1,
    disable_split_kv: bool = False,
) -> None:
    """
    A faster version of BatchDecodeWithPagedKVCacheWrapper::plan used for
    cudagraph capture/replay, while the no cudagraph version turns back
    to the original plan.
    using original plan after passing host-side buffers:
    - only host-to-device copy of indptr and last_page_len buffers
    Modifications for cudagraph:
    - only host-to-device copy of indptr and last_page_len buffers.
    - avoid device-to-device copy of indices buffer.

    Part of the code get inspiration from the original plan from FlashInfer repo
    and the implementation of fast_decode_plan for FlashInfer in SGlang repo.
    """
    # Warm up with the original plan if it is first call, and always run the
    # original plan if we run for dynamic shape. For fixed shape (cudagraph),
    # this warm up is to generate the _cached_module for the decode wrapper.
    if not decode_wrapper.is_cuda_graph_enabled or getattr(decode_wrapper, "fd_first_call", True):
        decode_wrapper.plan(
            indptr=indptr_cpu,
            indices=indices,
            last_page_len=last_page_len_cpu,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            pos_encoding_mode=pos_encoding_mode,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            data_type=data_type,
            sm_scale=sm_scale,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            non_blocking=non_blocking,
            block_tables=None,
            seq_lens=None,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
        )
        decode_wrapper.fd_first_call = False
        return

    assert decode_wrapper.is_cuda_graph_enabled, "Should be cudagraph only here"

    fast_decode_plan(
        decode_wrapper,
        indptr=indptr_cpu,
        indices=indices,
        last_page_len=last_page_len_cpu,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        pos_encoding_mode=pos_encoding_mode,
        window_left=window_left,
        logits_soft_cap=logits_soft_cap,
        q_data_type=q_data_type,
        kv_data_type=kv_data_type,
        data_type=data_type,
        sm_scale=sm_scale,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
        non_blocking=non_blocking,
        fixed_split_size=fixed_split_size,
        disable_split_kv=disable_split_kv,
    )


@triton.jit
def _copy_page_indices_kernel(
    page_indices,
    block_table,
    block_table_stride,
    cu_num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    row_ptr = block_table + req_idx * block_table_stride
    start_idx = tl.load(cu_num_blocks + req_idx)
    end_idx = tl.load(cu_num_blocks + req_idx + 1)
    num_blocks = end_idx - start_idx

    offset = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        block_ids = tl.load(row_ptr + i + offset, mask=i + offset < num_blocks)
        tl.store(
            page_indices + start_idx + i + offset,
            block_ids,
            mask=i + offset < num_blocks,
        )
