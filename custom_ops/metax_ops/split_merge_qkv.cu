// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_runtime.h>
#include <paddle/extension.h>
#include "helper.h"

template <typename T, int VecSize, bool IsSplit = false>
__global__ void RunDispatchQKV(const T* __restrict__ src_ptr_0,
                               const T* __restrict__ src_ptr_1,
                               const int* __restrict__ meta_ptr,
                               const int group_num,
                               const int hidden_dims,
                               const int64_t max_elements,
                               T* __restrict__ dst_ptr_0,
                               T* __restrict__ dst_ptr_1) {
  extern __shared__ int s_meta[];
  for (int i = threadIdx.x; i < group_num * 5; i += blockDim.x) {
    s_meta[i] = meta_ptr[i];
  }
  __syncthreads();

  using VecT = AlignedVector<T, VecSize>;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  int64_t step = (int64_t)gridDim.x * blockDim.x * VecSize;

  const T* src_ptrs[2] = {src_ptr_0, src_ptr_1};
  T* dst_ptrs[2] = {dst_ptr_0, dst_ptr_1};

  for (int64_t linear_index = global_thread_idx * VecSize;
       linear_index < max_elements;
       linear_index += step) {
    int token_idx = linear_index / hidden_dims;
    int hidden_idx = linear_index % hidden_dims;

    int stage = 0, start = 0, qkv_start = 0;
    for (int gidx = 0; gidx < group_num; ++gidx) {
      int base = gidx * 5;
      int g_start = s_meta[base + 3];
      int g_end = s_meta[base + 4];

      if (token_idx >= g_start && token_idx < g_end) {
        stage = s_meta[base + 0];
        start = s_meta[base + 1];
        qkv_start = g_start;
        break;
      }
    }

    int local_token_idx = token_idx - qkv_start + start;
    int64_t local_offset = (int64_t)local_token_idx * hidden_dims + hidden_idx;

    if constexpr (IsSplit) {
      T* target_dst = dst_ptrs[stage];
      Load(src_ptr_0 + linear_index,
           reinterpret_cast<VecT*>(target_dst + local_offset));
    } else {
      const T* target_src = src_ptrs[stage];
      Load(target_src + local_offset,
           reinterpret_cast<VecT*>(dst_ptr_0 + linear_index));
    }
  }
}

void SplitQKV(const paddle::Tensor& qkv,
              const paddle::Tensor& hybrid_meta,
              paddle::Tensor& prefill_qkv,
              paddle::Tensor& decode_qkv) {
  auto qkv_shape = qkv.shape();
  int token_num = qkv_shape[0];

  if (token_num == 0) {
    return;
  }

  int64_t linear_elem_num = qkv.numel();
  int hidden_dims = static_cast<int>(linear_elem_num / token_num);
  auto dtype = qkv.dtype();
  auto group_num = hybrid_meta.shape()[0];
  auto stream = qkv.stream();

  constexpr int pack_size = 4;
  constexpr int block_dims = 128;
  const int pack_num = linear_elem_num / pack_size;
  int grid_dims = 1;
  GetNumBlocks<block_dims>(pack_num, &grid_dims);
  size_t shared_mem_size = group_num * 5 * sizeof(int);

  switch (dtype) {
    case paddle::DataType::BFLOAT16:
      RunDispatchQKV<__maca_bfloat16, pack_size, true>
          <<<grid_dims, block_dims, shared_mem_size, stream>>>(
              reinterpret_cast<const __maca_bfloat16*>(
                  qkv.data<paddle::bfloat16>()),
              static_cast<const __maca_bfloat16*>(nullptr),
              reinterpret_cast<const int*>(hybrid_meta.data<int>()),
              group_num,
              hidden_dims,
              linear_elem_num,
              reinterpret_cast<__maca_bfloat16*>(
                  prefill_qkv.data<paddle::bfloat16>()),
              reinterpret_cast<__maca_bfloat16*>(
                  decode_qkv.data<paddle::bfloat16>()));
      break;
    case paddle::DataType::FLOAT16:
      RunDispatchQKV<__half, pack_size, true>
          <<<grid_dims, block_dims, shared_mem_size, stream>>>(
              reinterpret_cast<const __half*>(qkv.data<paddle::float16>()),
              static_cast<const __half*>(nullptr),
              reinterpret_cast<const int*>(hybrid_meta.data<int>()),
              group_num,
              hidden_dims,
              linear_elem_num,
              reinterpret_cast<__half*>(prefill_qkv.data<paddle::float16>()),
              reinterpret_cast<__half*>(decode_qkv.data<paddle::float16>()));
      break;
    default:
      PD_THROW("Only support qkv dtype of BF16 and F16");
  }
}

void MergeQKV(const paddle::Tensor& prefill_out,
              const paddle::Tensor& decdoe_out,
              const paddle::Tensor& hybrid_meta,
              paddle::Tensor& merged_out) {
  auto merged_out_shape = merged_out.shape();
  int token_num = merged_out_shape[0];

  if (token_num == 0) {
    return;
  }

  int64_t linear_elem_num = merged_out.numel();
  int hidden_dims = static_cast<int>(linear_elem_num / token_num);
  auto dtype = merged_out.dtype();
  auto group_num = hybrid_meta.shape()[0];
  auto stream = merged_out.stream();

  constexpr int pack_size = 4;
  constexpr int block_dims = 128;
  const int pack_num = linear_elem_num / pack_size;
  int grid_dims = 1;
  GetNumBlocks<block_dims>(pack_num, &grid_dims);
  size_t shared_mem_size = group_num * 5 * sizeof(int);

  switch (dtype) {
    case paddle::DataType::BFLOAT16:
      RunDispatchQKV<__maca_bfloat16, pack_size>
          <<<grid_dims, block_dims, shared_mem_size, stream>>>(
              reinterpret_cast<const __maca_bfloat16*>(
                  prefill_out.data<paddle::bfloat16>()),
              reinterpret_cast<const __maca_bfloat16*>(
                  decdoe_out.data<paddle::bfloat16>()),
              reinterpret_cast<const int*>(hybrid_meta.data<int>()),
              group_num,
              hidden_dims,
              linear_elem_num,
              reinterpret_cast<__maca_bfloat16*>(
                  merged_out.data<paddle::bfloat16>()),
              static_cast<__maca_bfloat16*>(nullptr));
      break;
    case paddle::DataType::FLOAT16:
      RunDispatchQKV<__half, pack_size>
          <<<grid_dims, block_dims, shared_mem_size, stream>>>(
              reinterpret_cast<const __half*>(
                  prefill_out.data<paddle::float16>()),
              reinterpret_cast<const __half*>(
                  decdoe_out.data<paddle::float16>()),
              reinterpret_cast<const int*>(hybrid_meta.data<int>()),
              group_num,
              hidden_dims,
              linear_elem_num,
              reinterpret_cast<__half*>(merged_out.data<paddle::float16>()),
              static_cast<__half*>(nullptr));
      break;
    default:
      PD_THROW("Only support qkv dtype of BF16 and F16");
  }
}

std::vector<std::vector<int64_t>> SplitQKVInferShape(
    const std::vector<int64_t>& qkv_shape,
    const std::vector<int64_t>& hybrid_meta_shape,
    const std::vector<int64_t>& prefill_qkv_shape,
    const std::vector<int64_t>& decode_qkv_shape) {
  return {qkv_shape, hybrid_meta_shape, prefill_qkv_shape, decode_qkv_shape};
}

std::vector<paddle::DataType> SplitQKVInferDtype(
    const paddle::DataType& qkv_dtype,
    const paddle::DataType& hybrid_meta_dtype,
    const paddle::DataType& prefill_qkv_dtype,
    const paddle::DataType& decode_qkv_dtype) {
  return {qkv_dtype, hybrid_meta_dtype, prefill_qkv_dtype, decode_qkv_dtype};
}

std::vector<std::vector<int64_t>> MergeQKVInferShape(
    const std::vector<int64_t>& prefill_out_shape,
    const std::vector<int64_t>& decode_out_shape,
    const std::vector<int64_t>& hybrid_meta_shape,
    const std::vector<int64_t>& merged_out_shape) {
  return {
      prefill_out_shape, decode_out_shape, hybrid_meta_shape, merged_out_shape};
}

std::vector<paddle::DataType> MergeQKVInferDtype(
    const paddle::DataType& prefill_out_dtype,
    const paddle::DataType& decode_out_dtype,
    const paddle::DataType& hybrid_meta_dtype,
    const paddle::DataType& merged_out_dtype) {
  return {
      prefill_out_dtype, decode_out_dtype, hybrid_meta_dtype, merged_out_dtype};
}

PD_BUILD_OP(split_qkv)
    .Inputs({
        "qkv",
        "hybrid_meta",
        "prefill_qkv",
        "decode_qkv",
    })
    .SetKernelFn(PD_KERNEL(SplitQKV))
    .SetInferShapeFn(PD_INFER_SHAPE(SplitQKVInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SplitQKVInferDtype));

PD_BUILD_OP(merge_qkv)
    .Inputs({
        "prefill_out",
        "decode_out",
        "hybrid_meta",
        "merged_out",
    })
    .SetKernelFn(PD_KERNEL(MergeQKV))
    .SetInferShapeFn(PD_INFER_SHAPE(MergeQKVInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MergeQKVInferDtype));
