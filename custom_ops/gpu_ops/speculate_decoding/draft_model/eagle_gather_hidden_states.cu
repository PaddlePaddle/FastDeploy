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

#include <cooperative_groups.h>

#include "paddle/extension.h"

#include "helper.h"

template <typename T, int VecSize>
__global__ void EagleGatherHiddenStatesKernel(
    T* output_data,
    int* position_map,
    int* output_token_num,
    const T* input,
    const int* cu_seqlens_q,
    const int* seq_lens_this_time,
    const int* seq_lens_decoder,
    const int* seq_lens_encoder,
    const int* batch_id_per_token_output,
    const int* cu_seqlens_q_output,
    const int dim_embed,
    const int64_t input_token_num,
    const int real_bsz) {
  if (blockIdx.x >= 1) {
    return;
  }

  extern __shared__ int smem[];
  int* in_count = smem;
  int* out_count = smem + real_bsz;
  int* in_offsets = smem + 2 * real_bsz;
  int* out_offsets = smem + 3 * real_bsz;

  // Phase 1: compute position_map (parallelized across threads in block 0)
  // Phase 1a: each thread computes counts for its batch elements
  for (int t = threadIdx.x; t < real_bsz; t += blockDim.x) {
    int cur_seq_len = seq_lens_this_time[t];
    // has seq in curent batch
    if (cur_seq_len > 0) {
      in_count[t] = cur_seq_len;
      out_count[t] = 1;
    } else {
      in_count[t] = 0;
      out_count[t] = 0;
    }
  }
  __syncthreads();

  // Phase 1b: prefix sum (thread 0 computes exclusive prefix sums)
  if (threadIdx.x == 0) {
    int in_acc = 0, out_acc = 0;
    for (int i = 0; i < real_bsz; i++) {
      in_acc += in_count[i];
      in_offsets[i] = in_acc - 1;
      out_offsets[i] = out_acc;
      out_acc += out_count[i];
    }
    output_token_num[0] = out_acc;
  }
  __syncthreads();

  // Phase 1c: each thread fills position_map for its batch elements
  for (int t = threadIdx.x; t < real_bsz; t += blockDim.x) {
    int in_off = in_offsets[t];
    int out_off = out_offsets[t];
    if (seq_lens_this_time[t] > 0) {
      position_map[in_off] = out_off;
    }
  }
  __syncthreads();

  using LoadT = AlignedVector<T, VecSize>;
  LoadT src_vec;

  int elem_cnt = input_token_num * dim_embed;
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int elem_idx = global_idx * VecSize; elem_idx < elem_cnt;
       elem_idx += blockDim.x * gridDim.x * VecSize) {
    int ori_token_idx = elem_idx / dim_embed;
    int token_idx = position_map[ori_token_idx];
    if (token_idx >= 0) {
      int offset = elem_idx % dim_embed;
      Load<T, VecSize>(input + ori_token_idx * dim_embed + offset, &src_vec);
      Store<T, VecSize>(src_vec, output_data + token_idx * dim_embed + offset);
    }
  }
}

template <paddle::DataType D>
std::vector<paddle::Tensor> DispatchDtype(
    const paddle::Tensor& input,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& batch_id_per_token_output,
    const paddle::Tensor& cu_seqlens_q_output,
    const paddle::Tensor& real_output_token_num) {
  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  auto input_token_num = input.shape()[0];
  auto dim_embed = input.shape()[1];
  const int real_bsz = seq_lens_this_time.shape()[0];

  auto position_map = paddle::empty(
      {input_token_num}, seq_lens_this_time.dtype(), input.place());
  cudaMemsetAsync(position_map.data<int>(),
                  0xFF,
                  input_token_num * sizeof(int),
                  input.stream());

  // TODO(yaohuicong): not need this params in future
  auto output_token_num =
      paddle::empty({1}, seq_lens_this_time.dtype(), input.place());

  // Pre-allocate output with max possible size (input_token_num)
  auto out = paddle::zeros({real_bsz, dim_embed}, input.dtype(), input.place());

  // only launch one block for position_map computation
  constexpr int block_size = 512;
  constexpr int grid_size = 1;

  // Calculate shared memory size: 4 int arrays of size real_bsz
  size_t smem_size = 4 * real_bsz * sizeof(int);

  // Determine vectorization size based on data type and dim_embed
  constexpr int VecSize = 4;

  // Launch kernel
  EagleGatherHiddenStatesKernel<data_t, VecSize>
      <<<grid_size, block_size, smem_size, input.stream()>>>(
          out.data<data_t>(),
          position_map.data<int>(),
          output_token_num.data<int>(),
          input.data<data_t>(),
          cu_seqlens_q.data<int>(),
          seq_lens_this_time.data<int>(),
          seq_lens_decoder.data<int>(),
          seq_lens_encoder.data<int>(),
          batch_id_per_token_output.data<int>(),
          cu_seqlens_q_output.data<int>(),
          static_cast<int>(dim_embed),
          input_token_num,
          real_bsz);

  // Return output and output_token_num
  return {out, output_token_num};
}

// Wrapper function for PD_BUILD_STATIC_OP
std::vector<paddle::Tensor> EagleGatherHiddenStates(
    const paddle::Tensor& input,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& batch_id_per_token_output,
    const paddle::Tensor& cu_seqlens_q_output,
    const paddle::Tensor& real_output_token_num) {
  switch (input.dtype()) {
    case paddle::DataType::BFLOAT16:
      return DispatchDtype<paddle::DataType::BFLOAT16>(
          input,
          cu_seqlens_q,
          seq_lens_this_time,
          seq_lens_decoder,
          seq_lens_encoder,
          batch_id_per_token_output,
          cu_seqlens_q_output,
          real_output_token_num);
    case paddle::DataType::FLOAT16:
      return DispatchDtype<paddle::DataType::FLOAT16>(input,
                                                      cu_seqlens_q,
                                                      seq_lens_this_time,
                                                      seq_lens_decoder,
                                                      seq_lens_encoder,
                                                      batch_id_per_token_output,
                                                      cu_seqlens_q_output,
                                                      real_output_token_num);
    case paddle::DataType::FLOAT32:
      return DispatchDtype<paddle::DataType::FLOAT32>(input,
                                                      cu_seqlens_q,
                                                      seq_lens_this_time,
                                                      seq_lens_decoder,
                                                      seq_lens_encoder,
                                                      batch_id_per_token_output,
                                                      cu_seqlens_q_output,
                                                      real_output_token_num);
    default:
      PD_THROW("eagle_gather_hidden_states: NOT supported data type.");
  }
}

PD_BUILD_STATIC_OP(eagle_gather_hidden_states)
    .Inputs({"input",
             "cu_seqlens_q",
             "seq_lens_this_time",
             "seq_lens_decoder",
             "seq_lens_encoder",
             "batch_id_per_token_output",
             "cu_seqlens_q_output",
             "real_output_token_num"})
    .Outputs({"out", "output_token_num"})
    .SetKernelFn(PD_KERNEL(EagleGatherHiddenStates));
