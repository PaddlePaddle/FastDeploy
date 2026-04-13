// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "helper.h"
#include "paddle/extension.h"

__global__ void get_attn_mask_q_kernel(
    int* __restrict__ startend_row_indices_ptr,
    const int* attn_mask_kv_ptr,
    const int* cu_seqlens_q,
    const int* cu_seqlens_k,
    const int kv_token_num,
    const int max_batch_size) {
  constexpr int VecSize = 4;
  const uint32_t tid = threadIdx.x, bid = blockIdx.x;
  int startend_row_vec[2];
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  for (uint32_t cu_seqlens_k_idx = bid * blockDim.x + tid;
       cu_seqlens_k_idx < kv_token_num;
       cu_seqlens_k_idx += blockDim.x * gridDim.x) {
    uint32_t batch_id = 0;

    for (int i = 0; i < max_batch_size; ++i) {
      if (cu_seqlens_k_idx >= cu_seqlens_k[i] &&
          cu_seqlens_k_idx < cu_seqlens_k[i + 1]) {
        batch_id = i;
        break;
      }
    }
    const uint32_t this_batch_q_start = cu_seqlens_q[batch_id];
    const uint32_t this_batch_q_end = cu_seqlens_q[batch_id + 1];
    const uint32_t this_batch_q_len = this_batch_q_end - this_batch_q_start;
    const uint32_t kv_start = cu_seqlens_k[batch_id];
    const uint32_t kv_end = cu_seqlens_k[batch_id + 1];
    const uint32_t kv_len = kv_end - kv_start;
    const uint32_t cache_k_idx = cu_seqlens_k_idx - kv_start;

    startend_row_vec[0] = this_batch_q_end;
    // startend_row_vec[1] = cu_seqlens_q[max_batch_size];
    // startend_row_vec[2] = 0;
    startend_row_vec[1] = this_batch_q_end;
    for (int this_batch_q_idx = this_batch_q_start;
         this_batch_q_idx < this_batch_q_end;
         ++this_batch_q_idx) {
      // const int append_mask_k_start = attn_mask_kv_ptr ?
      // attn_mask_kv_ptr[this_batch_q_idx * 2 + 0] : 0;
      const int append_mask_k_end =
          attn_mask_kv_ptr ? attn_mask_kv_ptr[this_batch_q_idx * 2 + 1] - 1
                           : this_batch_q_idx - this_batch_q_start + kv_len -
                                 (this_batch_q_len);
      if (cache_k_idx <= append_mask_k_end) {
        startend_row_vec[1] = min(startend_row_vec[1], this_batch_q_idx);
        // 可提前跳出循环
        break;
      }
    }
    reinterpret_cast<int2*>(startend_row_indices_ptr +
                            cu_seqlens_k_idx * 2)[0] =
        reinterpret_cast<int2*>(startend_row_vec)[0];
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

std::vector<paddle::Tensor> get_attn_mask_q(
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& cu_seqlens_k,
    const paddle::optional<paddle::Tensor>& attn_mask_kv,
    const int kv_token_num) {
  paddle::Tensor attn_mask_startend_row_indices = GetEmptyTensor(
      {1, 1, kv_token_num, 2}, paddle::DataType::INT32, cu_seqlens_k.place());
  const int max_batch_size = cu_seqlens_k.dims()[0] - 1;
  constexpr int block_size = 512;
  int grid_size = div_up(kv_token_num, block_size);
#ifdef PADDLE_WITH_CUSTOM_DEVICE_METAX_GPU
  get_attn_mask_q_kernel<<<grid_size, block_size, 0, cu_seqlens_k.stream()>>>(
      attn_mask_startend_row_indices.data<int>(),
      attn_mask_kv ? attn_mask_kv.get().data<int>() : nullptr,
      cu_seqlens_q.data<int>(),
      cu_seqlens_k.data<int>(),
      kv_token_num,
      max_batch_size);
#else
  launchWithPdlWhenEnabled(
      get_attn_mask_q_kernel,
      grid_size,
      block_size,
      0,
      cu_seqlens_k.stream(),
      attn_mask_startend_row_indices.data<int>(),
      attn_mask_kv ? attn_mask_kv.get().data<int>() : nullptr,
      cu_seqlens_q.data<int>(),
      cu_seqlens_k.data<int>(),
      kv_token_num,
      max_batch_size);
#endif
  return {attn_mask_startend_row_indices};
}

std::vector<paddle::DataType> GetAttnMaskQInferDtype(
    const paddle::DataType& cu_seqlens_q_dtype,
    const paddle::DataType& cu_seqlens_k_dtype,
    const paddle::optional<paddle::DataType>& attn_mask_kv_dtype) {
  return {paddle::DataType::INT32};
}

std::vector<std::vector<int64_t>> GetAttnMaskQInferShape(
    const std::vector<int64_t>& cu_seqlens_q_shape,
    const std::vector<int64_t>& cu_seqlens_k_shape,
    const paddle::optional<std::vector<int64_t>>& attn_mask_kv_shape,
    const int kv_token_num) {
  return {{1, 1, kv_token_num, 2}};
}

PD_BUILD_STATIC_OP(get_attn_mask_q)
    .Inputs({"cu_seqlens_q",
             "cu_seqlens_k",
             paddle::Optional("attn_mask_offsets")})
    .Outputs({"attn_mask_q"})
    .Attrs({"kv_token_num: int"})
    .SetKernelFn(PD_KERNEL(get_attn_mask_q))
    .SetInferShapeFn(PD_INFER_SHAPE(GetAttnMaskQInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetAttnMaskQInferDtype));
