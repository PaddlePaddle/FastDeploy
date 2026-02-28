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
def depermute_prefill_combine_kernel(
    x: cute.Tensor,
    x_coord: cute.Tensor,
    indice_map: cute.Tensor,
    topk_weights: cute.Tensor,
    depermuted_x: cute.Tensor,
    depermuted_x_coord: cute.Tensor,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
    num_worst_tokens,
    max_num_tokens_per_expert,
    VEC_SIZE: cutlass.Constexpr,
    TOP_K: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    grid_dimx, _, _ = cute.arch.grid_dim()
    block_dimx, _, _ = cute.arch.block_dim()

    copy_atom_x = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), x.element_type)
    copy_atom_indices = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), indice_map.element_type)
    copy_atom_top_k_weights = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), topk_weights.element_type)
    tiled_copy_x = cute.make_tiled_copy_tv(copy_atom_x, thr_layout, val_layout)
    thread_copy_x = tiled_copy_x.get_slice(tidx)

    smem = cutlass.utils.SmemAllocator()
    ori_2_permute_indice_tensor = smem.allocate_tensor(
        indice_map.element_type, cute.make_ordered_layout((TOP_K), order=(0)), 16
    )
    topk_weights_tensor = smem.allocate_tensor(
        topk_weights.element_type, cute.make_ordered_layout((TOP_K), order=(0)), 16
    )

    for token_idx in range(bidx, num_worst_tokens, grid_dimx):

        local_depermuted_x = depermuted_x[token_idx, None]
        thread_depermuted_x = thread_copy_x.partition_S(local_depermuted_x)
        frag_depermuted_x = cute.make_fragment_like(thread_depermuted_x)
        frag_depermuted_x.fill(0.0)
        frag_depermuted_x_reg = frag_depermuted_x.load().to(cutlass.Float32)

        local_depermuted_x_coord = depermuted_x_coord[token_idx, None]
        thread_depermuted_x_coord = thread_copy_x.partition_S(local_depermuted_x_coord)
        frag_depermuted_x_coord = cute.make_rmem_tensor(thread_depermuted_x_coord.shape, cutlass.Boolean)
        for i in range(0, cute.size(frag_depermuted_x_coord), 1):
            is_valid = cute.elem_less(thread_depermuted_x_coord[i], depermuted_x.shape)
            frag_depermuted_x_coord[i] = is_valid

        if tidx == 0:
            cute.copy(copy_atom_indices, indice_map[token_idx, None], ori_2_permute_indice_tensor)

        if tidx == 32:
            cute.copy(copy_atom_top_k_weights, topk_weights[token_idx, None], topk_weights_tensor)

        cute.arch.sync_threads()

        need_store = False

        for expert_slot in range(TOP_K):
            indice = ori_2_permute_indice_tensor[expert_slot]
            if indice >= 0:
                need_store = True

                topk_weight = topk_weights_tensor[expert_slot]
                permuted_local_expert = cutlass.Int64(indice // max_num_tokens_per_expert)
                permuted_offset_in_expert = cutlass.Int64(indice % max_num_tokens_per_expert)

                local_x = x[permuted_local_expert, permuted_offset_in_expert, None]
                local_x_coord = x_coord[permuted_local_expert, permuted_offset_in_expert, None]
                thread_x = thread_copy_x.partition_S(local_x)
                thread_x_coord = thread_copy_x.partition_S(local_x_coord)

                frag_x_coord = cute.make_rmem_tensor(thread_x_coord.shape, cutlass.Boolean)
                for i in range(0, cute.size(frag_x_coord), 1):
                    is_valid = cute.elem_less(thread_x_coord[i], x.shape)
                    frag_x_coord[i] = is_valid

                frag_x = cute.make_fragment_like(thread_x)
                cute.copy(copy_atom_x, thread_x, frag_x, pred=frag_x_coord)
                frag_x_reg = frag_x.load().to(cutlass.Float32)

                frag_depermuted_x_reg += frag_x_reg * topk_weight

        # Write result back after all experts are processed
        if need_store:
            frag_depermuted_x.store(frag_depermuted_x_reg.to(thread_depermuted_x.element_type))
            cute.arch.sync_threads()
            cute.copy(copy_atom_x, frag_depermuted_x, thread_depermuted_x, pred=frag_depermuted_x_coord)


@cute.jit
def depermute_prefill_combine(
    x: cute.Tensor,
    indice_map: cute.Tensor,
    topk_weights: cute.Tensor,
    depermuted_x: cute.Tensor,
    stream: cuda.CUstream,
    topk: cutlass.Constexpr,
    copy_bits: cutlass.Constexpr = 128,
):

    x_type = x.element_type
    num_worst_tokens = depermuted_x.shape[0]
    max_num_tokens_per_expert = cute.size(x, [1])

    vector_size = copy_bits // x_type.width
    thr_layout = cute.make_ordered_layout(512, order=(0))
    val_layout = cute.make_ordered_layout(vector_size, order=(0))

    x_coord = cute.make_identity_tensor(x.shape)
    depermuted_x_coord = cute.make_identity_tensor(depermuted_x.shape)

    device_props = paddle.device.cuda.get_device_properties()
    num_block_x = min(device_props.multi_processor_count * 2, num_worst_tokens)
    depermute_prefill_combine_kernel(
        x,
        x_coord,
        indice_map,
        topk_weights,
        depermuted_x,
        depermuted_x_coord,
        thr_layout,
        val_layout,
        num_worst_tokens,
        max_num_tokens_per_expert,
        vector_size,
        topk,
    ).launch(
        grid=[num_block_x, 1, 1],
        block=[cute.cosize(thr_layout), 1, 1],
        stream=stream,
    )


def call_depermute_prefill_combine(
    x: paddle.Tensor,
    indice_map: paddle.Tensor,
    topk_weights: paddle.Tensor,
    num_worst_tokens: int,
):
    topk = topk_weights.shape[1]
    num_local_experts = x.shape[0]
    max_num_tokens_per_expert = x.shape[1]
    hidden = x.shape[2]

    depermuted_x = paddle.empty([num_worst_tokens, hidden], dtype=x.dtype)
    x_dtype = paddle2cute_dtype_map[x.dtype]
    topk_weights_dtype = paddle2cute_dtype_map[topk_weights.dtype]

    compile_key = (x_dtype, topk_weights_dtype, num_local_experts, max_num_tokens_per_expert, hidden, topk)

    x_tensor = from_dlpack(x)
    indice_map_tensor = from_dlpack(indice_map).mark_compact_shape_dynamic(mode=0)
    topk_weights_tensor = from_dlpack(topk_weights).mark_compact_shape_dynamic(mode=0)
    depermuted_x_tensor = from_dlpack(depermuted_x).mark_compact_shape_dynamic(mode=0)

    if compile_key not in call_depermute_prefill_combine.compile_cache:
        stream = cute.runtime.make_fake_stream()
        if topk == 4:
            compiled_func = cute.compile(
                depermute_prefill_combine,
                x_tensor,
                indice_map_tensor,
                topk_weights_tensor,
                depermuted_x_tensor,
                stream,
                4,
                options="--generate-line-info",
            )
        elif topk == 8:
            compiled_func = cute.compile(
                depermute_prefill_combine,
                x_tensor,
                indice_map_tensor,
                topk_weights_tensor,
                depermuted_x_tensor,
                stream,
                8,
                options="--generate-line-info",
            )
        call_depermute_prefill_combine.compile_cache[compile_key] = compiled_func

    stream = cuda.CUstream(paddle.device.current_stream().stream_base.cuda_stream)
    call_depermute_prefill_combine.compile_cache[compile_key](
        x_tensor,
        indice_map_tensor,
        topk_weights_tensor,
        depermuted_x_tensor,
        stream,
    )
    return depermuted_x


call_depermute_prefill_combine.compile_cache = {}
