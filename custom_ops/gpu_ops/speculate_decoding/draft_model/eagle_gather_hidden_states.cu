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

namespace cg = cooperative_groups;

// Fused kernel: block 0 computes position_map and output_token_num in parallel
// (one thread per batch element), then all blocks synchronize via
// cooperative_groups grid sync, and finally all threads perform the hidden
// states gather in parallel.
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
  cg::grid_group grid = cg::this_grid();

  // Dynamic shared memory layout: [in_count|out_count|in_offsets|out_offsets]
  extern __shared__ int smem[];
  int* in_count = smem;
  int* out_count = smem + real_bsz;
  int* in_offsets = smem + 2 * real_bsz;
  int* out_offsets = smem + 3 * real_bsz;

  // Phase 1: compute position_map (parallelized across threads in block 0)
  if (blockIdx.x == 0) {
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
        in_offsets[i] = in_acc;
        out_offsets[i] = out_acc;
        in_acc += in_count[i];
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
        // For gather: map input token to output position
        // Use last token of each sequence
        int last_token_idx = in_off + in_count[t] - 1;
        position_map[last_token_idx] = out_off;
      }
    }
  }

  // Phase 2: grid-wide sync to ensure position_map is ready
  grid.sync();

  // Phase 3: gather hidden states in parallel
  using LoadT = AlignedVector<T, VecSize>;
  LoadT src_vec;

  int elem_cnt = input_token_num * dim_embed;
  int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int elem_idx = global_thread_idx * VecSize; elem_idx < elem_cnt;
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
  int real_bsz = seq_lens_this_time.shape()[0];

  auto position_map = paddle::empty(
      {input_token_num}, seq_lens_this_time.dtype(), input.place());
  cudaMemsetAsync(position_map.data<int>(),
                  0xFF,
                  input_token_num * sizeof(int),
                  input.stream());

  // TODO(yaohuicong): not need this params in future
  auto output_token_num =
      paddle::empty({1}, seq_lens_this_time.dtype(), input.place());

  // Pre-allocate output with max possible size (real_bsz)
  auto out = paddle::zeros({real_bsz, dim_embed}, input.dtype(), input.place());

  constexpr int VecSize = 4;
  int elem_cnt = input_token_num * dim_embed;
  assert(elem_cnt % VecSize == 0);

  // Grid size linearly related to real_bsz for cooperative launch efficiency
  // and CUDA graph capture friendliness
  constexpr int thread_per_block = 128;
  constexpr int DESIRED_BLOCKS_PER_BATCH = 4;
  int dynamic_smem_size = 4 * real_bsz * static_cast<int>(sizeof(int));

  // Cooperative launch limit: use conservative smem upper bound for caching
  static const int max_grid_size = [&]() {
    int blocks_per_sm = 0;
    constexpr int smem_upper_bound = 4 * 512 * sizeof(int);  // 8KB
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        EagleGatherHiddenStatesKernel<DataType_, VecSize>,
        thread_per_block,
        smem_upper_bound);
    int dev = 0;
    cudaGetDevice(&dev);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    return blocks_per_sm * sms;
  }();

  int blocks_per_batch =
      std::min(DESIRED_BLOCKS_PER_BATCH, max_grid_size / std::max(real_bsz, 1));
  blocks_per_batch = std::max(blocks_per_batch, 1);
  int grid_size = std::min(real_bsz * blocks_per_batch, max_grid_size);
  grid_size = std::max(grid_size, 1);

  const DataType_* input_ptr =
      reinterpret_cast<const DataType_*>(input.data<data_t>());
  const int* cu_seqlens_q_ptr = cu_seqlens_q.data<int>();
  const int* seq_lens_this_time_ptr = seq_lens_this_time.data<int>();
  const int* seq_lens_decoder_ptr = seq_lens_decoder.data<int>();
  const int* seq_lens_encoder_ptr = seq_lens_encoder.data<int>();
  const int* batch_id_per_token_output_ptr =
      batch_id_per_token_output.data<int>();
  const int* cu_seqlens_q_output_ptr = cu_seqlens_q_output.data<int>();
  DataType_* output_data_ptr = reinterpret_cast<DataType_*>(out.data<data_t>());
  int* position_map_ptr = position_map.data<int>();
  int* output_token_num_ptr = output_token_num.data<int>();
  int dim_embed_int = static_cast<int>(dim_embed);
  int64_t input_token_num_int64 = input_token_num;

  void* kernel_args[] = {&output_data_ptr,
                         &position_map_ptr,
                         &output_token_num_ptr,
                         &input_ptr,
                         &cu_seqlens_q_ptr,
                         &seq_lens_this_time_ptr,
                         &seq_lens_decoder_ptr,
                         &seq_lens_encoder_ptr,
                         &batch_id_per_token_output_ptr,
                         &cu_seqlens_q_output_ptr,
                         &dim_embed_int,
                         &input_token_num_int64,
                         &real_bsz};

  cudaLaunchCooperativeKernel(
      (void*)EagleGatherHiddenStatesKernel<DataType_, VecSize>,
      dim3(grid_size),
      dim3(thread_per_block),
      kernel_args,
      dynamic_smem_size,
      input.stream());

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
