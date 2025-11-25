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

#include <cassert>
#include "helper.h"
#include "paddle/extension.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

// Helper function to find decode token position
__device__ int find_decode_position(int output_idx,
                                    const int* cu_seqlens,
                                    const int64_t* prompt_lens,
                                    int batch_size) {
  int remaining = output_idx;
  for (int i = 0; i < batch_size; ++i) {
    int seq_start = cu_seqlens[i];
    int seq_end = cu_seqlens[i + 1];
    int prompt_len = prompt_lens[i];
    int decode_in_seq = seq_end - seq_start - prompt_len;

    if (remaining < decode_in_seq) {
      return seq_start + prompt_len + remaining;
    }
    remaining -= decode_in_seq;
  }
  return -1;  // Should not reach here
}

// Helper function to find prefill token position
__device__ int find_prefill_position(int output_idx,
                                     const int* cu_seqlens,
                                     const int64_t* prompt_lens,
                                     int batch_size) {
  int remaining = output_idx;
  for (int i = 0; i < batch_size; ++i) {
    int seq_start = cu_seqlens[i];
    int prompt_len = prompt_lens[i];

    if (remaining < prompt_len) {
      return seq_start + remaining;
    }
    remaining -= prompt_len;
  }
  return -1;  // Should not reach here
}

// CUDA kernel for reordering decode tokens
__global__ void reorder_decode_kernel(const int64_t* x,
                                      int64_t* x_out,
                                      const int* batch_ids,
                                      int* batch_ids_out,
                                      const int* cu_seqlens,
                                      const int64_t* prompt_lens,
                                      int batch_size,
                                      int output_offset,
                                      int max_output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= max_output) return;

  // Find corresponding input position for this decode token
  int input_pos =
      find_decode_position(idx, cu_seqlens, prompt_lens, batch_size);

  x_out[output_offset + idx] = x[input_pos];
  batch_ids_out[output_offset + idx] = batch_ids[input_pos];
}

// CUDA kernel for reordering prefill tokens

__global__ void count_decode_tokens_kernel(const int* cu_seqlens,
                                           const int64_t* prompt_lens,
                                           int batch_size,
                                           int64_t* total_decode) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size) return;

  int64_t seq_len = cu_seqlens[idx + 1] - cu_seqlens[idx];
  int64_t diff = seq_len - prompt_lens[idx];
  assert(diff >= 0);
  atomicAdd(reinterpret_cast<unsigned long long*>(total_decode),
            static_cast<unsigned long long>(diff));
}

__global__ void reorder_prefill_kernel(const int64_t* x,
                                       int64_t* x_out,
                                       const int* batch_ids,
                                       int* batch_ids_out,
                                       const int* cu_seqlens,
                                       const int64_t* prompt_lens,
                                       int batch_size,
                                       int output_offset,
                                       int total_prefill) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_prefill) return;
  int input_pos =
      find_prefill_position(idx, cu_seqlens, prompt_lens, batch_size);
  x_out[output_offset + idx] = x[input_pos];
  batch_ids_out[output_offset + idx] = batch_ids[input_pos];
}

std::vector<std::vector<int64_t>> ReorderSplitPrefillAndDecodeInferShape(
    const std::vector<int64_t>& x_remove_padding_shape,
    const std::vector<int64_t>& batch_id_per_token_shape,
    const std::vector<int64_t>& cu_seqlens_q_shape,
    const std::vector<int64_t>& prompt_lens_shape) {
  int64_t total_tokens = x_remove_padding_shape[0];
  return {{total_tokens}, {total_tokens}, {1}};
}

std::vector<paddle::DataType> ReorderSplitPrefillAndDecodeInferDtype(
    const paddle::DataType& x_remove_padding_dtype,
    const paddle::DataType& batch_id_per_token_dtype,
    const paddle::DataType& cu_seqlens_q_dtype,
    const paddle::DataType& prompt_lens_dtype) {
  return {x_remove_padding_dtype,
          batch_id_per_token_dtype,
          paddle::DataType::INT64};
}

std::vector<paddle::Tensor> ReorderSplitPrefillAndDecode(
    const paddle::Tensor& x_remove_padding,
    const paddle::Tensor& batch_id_per_token,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& prompt_lens) {
// Get device info
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          x_remove_padding.place()));
  auto stream = dev_ctx->stream();
#else
  auto stream = x_remove_padding.stream();
#endif

  // Get input data
  const int64_t* prompt_lens_ptr = prompt_lens.data<int64_t>();
  const int* batch_id_ptr = batch_id_per_token.data<int>();
  const int* cu_seqlens_ptr = cu_seqlens_q.data<int>();
  int batch_size = cu_seqlens_q.shape()[0] - 1;
  int total_tokens = x_remove_padding.shape()[0];

  if (total_tokens < 1) {
    PD_THROW(
        "reorder_split_prefill_and_decode op can't support input that is "
        "empty");
  }

  // Prepare output tensors
  auto x_reorder = paddle::experimental::empty_like(x_remove_padding);
  auto batch_id_reorder = paddle::experimental::empty_like(batch_id_per_token);
  // Count decode tokens on device
  auto num_decode_tokens =
      paddle::full({1}, 0, paddle::DataType::INT64, x_remove_padding.place());

#ifdef PADDLE_WITH_COREX
  int block_size =
      std::min((total_tokens + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE, 128);
#else
  int block_size =
      min((total_tokens + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE, 128);
#endif

  {
    int grid_size = (batch_size + block_size - 1) / block_size;
    count_decode_tokens_kernel<<<grid_size, block_size, 0, stream>>>(
        cu_seqlens_ptr,
        prompt_lens_ptr,
        batch_size,
        num_decode_tokens.data<int64_t>());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      PD_THROW("count_decode_tokens_kernel launch failed: ",
               cudaGetErrorString(err));
    }
  }

  int64_t total_decode =
      num_decode_tokens.copy_to(paddle::CPUPlace(), true).data<int64_t>()[0];

  // Get input/output data pointers
  const int64_t* x_ptr = x_remove_padding.data<int64_t>();
  int64_t* x_reorder_ptr = x_reorder.data<int64_t>();
  int* batch_id_reorder_ptr = batch_id_reorder.data<int>();

  // Launch CUDA kernel to reorder data
  // First pass: collect decode tokens
  {
    if (total_decode > 0) {
      int grid_size = (total_decode + block_size - 1) / block_size;
      reorder_decode_kernel<<<grid_size, block_size, 0, stream>>>(
          x_ptr,
          x_reorder_ptr,
          batch_id_ptr,
          batch_id_reorder_ptr,
          cu_seqlens_ptr,
          prompt_lens_ptr,
          batch_size,
          0,
          total_decode);
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        PD_THROW("reorder_decode_kernel launch failed: ",
                 cudaGetErrorString(err));
      }
    }
  }

  // Second pass: collect prefill tokens
  {
    int total_prefill = total_tokens - total_decode;
    if (total_prefill > 0) {
      int grid_size = (total_prefill + block_size - 1) / block_size;
      reorder_prefill_kernel<<<grid_size, block_size, 0, stream>>>(
          x_ptr,
          x_reorder_ptr,
          batch_id_ptr,
          batch_id_reorder_ptr,
          cu_seqlens_ptr,
          prompt_lens_ptr,
          batch_size,
          total_decode,
          total_prefill);
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        PD_THROW("reorder_prefill_kernel launch failed: ",
                 cudaGetErrorString(err));
      }
    }
  }
  return {x_reorder, batch_id_reorder, num_decode_tokens};
}

PD_BUILD_STATIC_OP(reorder_split_prefill_and_decode)
    .Inputs({"x_remove_padding",
             "batch_id_per_token",
             "cu_seqlens_q",
             "prompt_lens"})
    .Outputs({"x_reorder", "batch_id_reorder", "num_decode_tokens"})
    .SetKernelFn(PD_KERNEL(ReorderSplitPrefillAndDecode))
    .SetInferShapeFn(PD_INFER_SHAPE(ReorderSplitPrefillAndDecodeInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(ReorderSplitPrefillAndDecodeInferDtype));
