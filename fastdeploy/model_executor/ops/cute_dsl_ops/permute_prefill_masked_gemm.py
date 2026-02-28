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

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import paddle
from cutlass.cute.runtime import from_dlpack

from .utils import paddle2cute_dtype_map


@cute.kernel
def prefill_permute_to_masked_gemm_kernel(
    x: cute.Tensor,
    x_coord: cute.Tensor,
    scale: cute.Tensor,
    scale_coord: cute.Tensor,
    topk_ids: cute.Tensor,
    permute_x: cute.Tensor,
    permute_scale: cute.Tensor,
    permuted_indice_map: cute.Tensor,
    token_nums_per_expert: cute.Tensor,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
    num_tokens,
    max_num_tokens_per_expert: int,
    TOP_K: cutlass.Constexpr,
):
    """
    CUDA kernel for permuting input tokens and scales to expert-grouped layout for MoE masked GEMM.

    This kernel redistributes input data from token-major layout [num_tokens, hidden] to
    expert-major layout [num_experts, max_tokens_per_expert, hidden] based on top-k routing decisions.
    Each token is copied to its assigned expert's buffer using atomic operations to track offsets.

    Args:
        x: Input tensor of shape [num_tokens, hidden], containing token hidden states.
        x_coord: Identity coordinate tensor for x, used for bounds checking.
        scale: Input scale tensor of shape [num_tokens, hidden_scale], for quantized inputs.
        scale_coord: Identity coordinate tensor for scale, used for bounds checking.
        topk_ids: Expert indices tensor of shape [num_tokens, TOP_K], containing assigned expert IDs.
        permute_x: Output tensor of shape [num_experts, max_token_num, hidden] for permuted tokens.
        permute_scale: Output tensor of shape [num_experts, max_token_num, hidden_scale] for permuted scales.
        token_nums_per_expert: Counter tensor of shape [num_experts, 1], tracks tokens per expert atomically.
        thr_layout: Thread layout for tiled copy operations.
        val_layout: Value layout defining vector size for copy operations.
        num_tokens: Total number of input tokens to process.
        TOP_K: Compile-time constant for number of experts per token.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    grid_dimx, _, _ = cute.arch.grid_dim()
    block_dimx, _, _ = cute.arch.block_dim()

    copy_atom_x = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), x.element_type)
    copy_atom_scale = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), scale.element_type)
    copy_atom_top_k = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), topk_ids.element_type)

    tiled_copy_x = cute.make_tiled_copy_tv(copy_atom_x, thr_layout, val_layout)
    tiled_copy_scale = cute.make_tiled_copy_tv(copy_atom_scale, thr_layout, val_layout)

    thread_copy_x = tiled_copy_x.get_slice(tidx)
    thread_copy_scale = tiled_copy_scale.get_slice(tidx)

    smem = cutlass.utils.SmemAllocator()
    offset_tensor = smem.allocate_tensor(
        token_nums_per_expert.element_type, cute.make_ordered_layout(1, order=(0)), 16
    )
    top_k_idx_tensor = smem.allocate_tensor(topk_ids.element_type, cute.make_ordered_layout(TOP_K, order=(0)), 16)

    for token_idx in range(bidx, num_tokens, grid_dimx):
        local_x = x[token_idx, None]
        local_x_coord = x_coord[token_idx, None]
        local_scale = scale[token_idx, None]
        local_scale_coord = scale_coord[token_idx, None]

        if tidx == 0:
            cute.copy(copy_atom_top_k, topk_ids[token_idx, None], top_k_idx_tensor)

        cute.arch.sync_threads()

        for expert_slot in range(TOP_K):
            expert_idx = cutlass.Int64(top_k_idx_tensor[expert_slot])
            if expert_idx != -1:
                if tidx == 0:
                    offset_tensor[0] = cute.arch.atomic_add(token_nums_per_expert[expert_idx, None].iterator, 1)
                    permuted_indice_map[token_idx, expert_slot] = cutlass.Int32(
                        expert_idx * max_num_tokens_per_expert + offset_tensor[0]
                    )

                cute.arch.sync_threads()

                local_permute_x = permute_x[expert_idx, offset_tensor[0], None]
                local_permute_scale = permute_scale[expert_idx, offset_tensor[0], None]

                thread_x = thread_copy_x.partition_S(local_x)
                thread_permute_x = thread_copy_x.partition_D(local_permute_x)

                thread_x_coord = thread_copy_x.partition_S(local_x_coord)
                frag_x_coord = cute.make_rmem_tensor(thread_x_coord.shape, cutlass.Boolean)
                for i in range(0, cute.size(frag_x_coord), 1):
                    is_valid = cute.elem_less(thread_x_coord[i], x.shape)
                    frag_x_coord[i] = is_valid

                thread_scale_coord = thread_copy_scale.partition_S(local_scale_coord)
                frag_scale_coord = cute.make_rmem_tensor(thread_scale_coord.shape, cutlass.Boolean)
                for i in range(0, cute.size(frag_scale_coord), 1):
                    is_valid = cute.elem_less(thread_scale_coord[i], scale.shape)
                    frag_scale_coord[i] = is_valid

                thread_scale = thread_copy_scale.partition_S(local_scale)
                thread_permute_scale = thread_copy_scale.partition_D(local_permute_scale)

                cute.copy(copy_atom_x, thread_x, thread_permute_x, pred=frag_x_coord)
                cute.copy(copy_atom_scale, thread_scale, thread_permute_scale, pred=frag_scale_coord)
                cute.arch.sync_threads()


@cute.jit
def prefill_permute_to_masked_gemm(
    x: cute.Tensor,
    scale: cute.Tensor,
    topk_ids: cute.Tensor,
    permute_x: cute.Tensor,
    permute_scale: cute.Tensor,
    permuted_indice_map: cute.Tensor,
    token_nums_per_expert: cute.Tensor,
    stream: cuda.CUstream,
    topk: cutlass.Constexpr,
    copy_bits: cutlass.Constexpr = 128,
):
    """
    JIT-compiled host function that launches the prefill permute kernel.

    This function prepares the kernel launch configuration and invokes the CUDA kernel
    to permute input tokens from token-major to expert-major layout. It automatically
    determines the optimal grid size based on the GPU's SM count.

    Args:
        x: Input tensor of shape [num_tokens, hidden], containing token hidden states.
        scale: Input scale tensor of shape [num_tokens, hidden_scale], for quantized inputs.
        topk_ids: Expert indices tensor of shape [num_tokens, topk], containing assigned expert IDs.
        permute_x: Output tensor for permuted tokens, shape [num_experts, max_token_num, hidden].
        permute_scale: Output tensor for permuted scales, shape [num_experts, max_token_num, hidden_scale].
        token_nums_per_expert: Counter tensor to track number of tokens assigned to each expert.
        topk: Compile-time constant for number of experts selected per token.
        copy_bits: Bit width for vectorized copy operations, default 128 bits.
    """
    num_tokens = cute.size(x, [0])
    max_num_tokens_per_expert = cute.size(permute_x, [1])

    x_type = x.element_type
    scale_type = scale.element_type
    vector_size = copy_bits // x_type.width if x_type.width > scale_type.width else copy_bits // scale_type.width

    x_coord = cute.make_identity_tensor(x.shape)
    scale_coord = cute.make_identity_tensor(scale.shape)

    thr_layout = cute.make_ordered_layout(512, order=(0))
    val_layout = cute.make_ordered_layout(vector_size, order=(0))

    device_props = paddle.device.cuda.get_device_properties()
    num_block_x = device_props.multi_processor_count * 2
    prefill_permute_to_masked_gemm_kernel(
        x,
        x_coord,
        scale,
        scale_coord,
        topk_ids,
        permute_x,
        permute_scale,
        permuted_indice_map,
        token_nums_per_expert,
        thr_layout,
        val_layout,
        num_tokens,
        max_num_tokens_per_expert,
        topk,
    ).launch(
        grid=[num_block_x, 1, 1],
        block=[cute.cosize(thr_layout), 1, 1],
        stream=stream,
    )


def call_prefill_permute_to_masked_gemm(
    x: paddle.Tensor,
    scale: paddle.Tensor,
    topk_ids: paddle.Tensor,
    num_local_experts: int,
    max_token_num: int,
):
    """
    High-level Python interface for the prefill permute operation in MoE layers.

    This function provides a convenient interface to permute input tokens and their scales
    from token-major layout to expert-major layout for subsequent masked GEMM operations.
    It handles memory allocation, dtype conversion, kernel compilation caching, and execution.

    The permutation reorders data so that all tokens assigned to the same expert are
    contiguous in memory, enabling efficient batched matrix multiplications per expert.

    Args:
        x: Input hidden states tensor of shape [num_tokens, hidden], PaddlePaddle tensor.
        scale: Input scale tensor of shape [num_tokens, hidden_scale], for quantized computations.
        topk_ids: Expert routing indices of shape [num_tokens, topk], where each row contains
                  the indices of top-k experts selected for that token. Use -1 for invalid experts.
        num_local_experts: Number of local experts on this device.
        max_token_num: Maximum number of tokens that can be assigned to any single expert.

    Returns:
        tuple: A tuple containing:
            - permute_x: Permuted hidden states of shape [num_local_experts, max_token_num, hidden].
            - permute_scale: Permuted scales of shape [num_local_experts, max_token_num, hidden_scale].
            - token_nums_per_expert: Tensor of shape [num_local_experts, 1] containing the actual
                                     number of tokens assigned to each expert.
    """
    num_worst_tokens = x.shape[0]
    hidden = x.shape[1]
    hidden_scale = scale.shape[1]
    topk = topk_ids.shape[1]

    permute_x = paddle.empty([num_local_experts, max_token_num, hidden], dtype=x.dtype)
    permute_scale = paddle.empty([num_local_experts, hidden_scale, max_token_num], dtype=scale.dtype)
    permute_scale = permute_scale.transpose((0, 2, 1))

    permuted_indice_map = paddle.full([num_worst_tokens, topk], fill_value=-1, dtype="int32")
    token_nums_per_expert = paddle.zeros([num_local_experts, 1], dtype="int32")

    x_dtype = paddle2cute_dtype_map[x.dtype]
    scale_dtype = paddle2cute_dtype_map[scale.dtype]
    topk_ids_dtype = paddle2cute_dtype_map[topk_ids.dtype]

    compile_key = (x_dtype, scale_dtype, topk_ids_dtype, num_local_experts, max_token_num, hidden, hidden_scale, topk)

    x_tensor = from_dlpack(x).mark_compact_shape_dynamic(mode=0)
    scale_tensor = from_dlpack(scale).mark_compact_shape_dynamic(mode=0)
    topk_ids_tensor = from_dlpack(topk_ids).mark_compact_shape_dynamic(mode=0)
    permute_x_tensor = from_dlpack(permute_x)
    permute_scale_tensor = from_dlpack(permute_scale)
    permuted_indice_map_tensor = from_dlpack(permuted_indice_map).mark_compact_shape_dynamic(mode=0)
    token_nums_per_expert_tensor = from_dlpack(token_nums_per_expert)

    if compile_key not in call_prefill_permute_to_masked_gemm.compile_cache:
        stream = cute.runtime.make_fake_stream()
        if topk == 4:
            compiled_func = cute.compile(
                prefill_permute_to_masked_gemm,
                x_tensor,
                scale_tensor,
                topk_ids_tensor,
                permute_x_tensor,
                permute_scale_tensor,
                permuted_indice_map_tensor,
                token_nums_per_expert_tensor,
                stream,
                4,
                options="--generate-line-info",
            )
        elif topk == 8:
            compiled_func = cute.compile(
                prefill_permute_to_masked_gemm,
                x_tensor,
                scale_tensor,
                topk_ids_tensor,
                permute_x_tensor,
                permute_scale_tensor,
                permuted_indice_map_tensor,
                token_nums_per_expert_tensor,
                stream,
                8,
                options="--generate-line-info",
            )
        call_prefill_permute_to_masked_gemm.compile_cache[compile_key] = compiled_func

    stream = cuda.CUstream(paddle.device.current_stream().stream_base.cuda_stream)

    call_prefill_permute_to_masked_gemm.compile_cache[compile_key](
        x_tensor,
        scale_tensor,
        topk_ids_tensor,
        permute_x_tensor,
        permute_scale_tensor,
        permuted_indice_map_tensor,
        token_nums_per_expert_tensor,
        stream,
    )

    return permute_x, permute_scale, permuted_indice_map, token_nums_per_expert


call_prefill_permute_to_masked_gemm.compile_cache = {}
