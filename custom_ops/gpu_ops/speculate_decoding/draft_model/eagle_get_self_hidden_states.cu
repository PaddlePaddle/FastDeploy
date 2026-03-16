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

#include <cooperative_groups.h>

#include "paddle/extension.h"

#include "helper.h"

namespace cg = cooperative_groups;

// Fused kernel: thread 0 of block 0 computes position_map and output_token_num,
// then all blocks synchronize via cooperative_groups grid sync, and finally
// all threads perform the hidden states rebuild in parallel.
template <typename T, int VecSize>
__global__ void rebuildSelfHiddenStatesKernel(
    const T* input,
    const int* last_seq_lens_this_time,
    const int* seq_lens_this_time,
    const int64_t* step_idx,
    int* position_map,
    int* output_token_num,
    T* out,
    const int bsz,
    const int dim_embed,
    const int input_token_num) {
  cg::grid_group grid = cg::this_grid();

  // Phase 1: compute position_map (single thread)
  // TODO(yaohuicong): paralize phase 1
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int in_offset = 0;
    int out_offset = 0;
    for (int i = 0; i < bsz; ++i) {
      int cur_seq_lens_this_time = seq_lens_this_time[i];
      int cur_last_seq_lens_this_time = last_seq_lens_this_time[i];
      // 1. encoder
      if (step_idx[i] == 1 && cur_seq_lens_this_time > 0) {
        position_map[in_offset] = out_offset++;
        in_offset += 1;
        // 2. decoder
      } else if (cur_seq_lens_this_time > 0) /* =1 */ {
        position_map[in_offset + cur_last_seq_lens_this_time - 1] =
            out_offset++;
        in_offset += cur_last_seq_lens_this_time;
        // 3. stop
      } else {
        // first token end
        if (step_idx[i] == 1) {
          in_offset += cur_last_seq_lens_this_time > 0 ? 1 : 0;
          // normal end
        } else {
          in_offset += cur_last_seq_lens_this_time;
        }
      }
    }
    output_token_num[0] = out_offset;
  }

  // Phase 2: grid-wide sync to ensure position_map is ready
  grid.sync();

  // Phase 3: rebuild hidden states in parallel
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
      Store<T, VecSize>(src_vec, out + token_idx * dim_embed + offset);
    }
  }
}

template <paddle::DataType D>
std::vector<paddle::Tensor> DispatchDtype(
    const paddle::Tensor& input,
    const paddle::Tensor& last_seq_lens_this_time,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& step_idx) {
  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  auto input_token_num = input.shape()[0];
  auto dim_embed = input.shape()[1];
  int bsz = seq_lens_this_time.shape()[0];

  auto position_map = paddle::empty(
      {input_token_num}, seq_lens_this_time.dtype(), input.place());
  cudaMemsetAsync(position_map.data<int>(),
                  0xFF,
                  input_token_num * sizeof(int),
                  input.stream());

  auto output_token_num =
      paddle::empty({1}, seq_lens_this_time.dtype(), input.place());

  // Pre-allocate output with max possible size (input_token_num)
  auto out =
      paddle::empty({input_token_num, dim_embed}, input.dtype(), input.place());

  constexpr int packSize = VEC_16B / (sizeof(DataType_));
  int elem_cnt = input_token_num * dim_embed;
  assert(elem_cnt % packSize == 0);

  int pack_num = elem_cnt / packSize;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  grid_size = std::max(grid_size, 1);

  // Clamp grid_size to max cooperative launch limit
  int max_blocks_per_sm = 0;
  constexpr int thread_per_block = 128;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_blocks_per_sm,
      rebuildSelfHiddenStatesKernel<DataType_, packSize>,
      thread_per_block,
      0);
  int dev = 0;
  cudaGetDevice(&dev);
  int sm_count = 0;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
  int max_grid_size = max_blocks_per_sm * sm_count;
  grid_size = std::min(grid_size, max_grid_size);

  const DataType_* input_ptr =
      reinterpret_cast<const DataType_*>(input.data<data_t>());
  const int* last_seq_lens_this_time_ptr = last_seq_lens_this_time.data<int>();
  const int* seq_lens_this_time_ptr = seq_lens_this_time.data<int>();
  const int64_t* step_idx_ptr = step_idx.data<int64_t>();
  int* position_map_ptr = position_map.data<int>();
  int* output_token_num_ptr = output_token_num.data<int>();
  DataType_* out_ptr = reinterpret_cast<DataType_*>(out.data<data_t>());
  int dim_embed_int = static_cast<int>(dim_embed);
  int input_token_num_int = static_cast<int>(input_token_num);

  void* kernel_args[] = {&input_ptr,
                         &last_seq_lens_this_time_ptr,
                         &seq_lens_this_time_ptr,
                         &step_idx_ptr,
                         &position_map_ptr,
                         &output_token_num_ptr,
                         &out_ptr,
                         &bsz,
                         &dim_embed_int,
                         &input_token_num_int};

  cudaLaunchCooperativeKernel(
      (void*)rebuildSelfHiddenStatesKernel<DataType_, packSize>,
      dim3(grid_size),
      dim3(thread_per_block),
      kernel_args,
      0,
      input.stream());

  return {out, output_token_num};
}

std::vector<paddle::Tensor> EagleGetSelfHiddenStates(
    const paddle::Tensor& input,
    const paddle::Tensor& last_seq_lens_this_time,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& step_idx) {
  switch (input.dtype()) {
    case paddle::DataType::BFLOAT16:
      return DispatchDtype<paddle::DataType::BFLOAT16>(
          input, last_seq_lens_this_time, seq_lens_this_time, step_idx);
    case paddle::DataType::FLOAT16:
      return DispatchDtype<paddle::DataType::FLOAT16>(
          input, last_seq_lens_this_time, seq_lens_this_time, step_idx);
    default:
      PD_THROW("Not support this data type");
  }
}

PD_BUILD_STATIC_OP(eagle_get_self_hidden_states)
    .Inputs(
        {"input", "last_seq_lens_this_time", "seq_lens_this_time", "step_idx"})
    .Outputs({"out", "output_token_num"})
    .SetKernelFn(PD_KERNEL(EagleGetSelfHiddenStates));
