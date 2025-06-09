// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/core/memory/memcpy.h"

__global__ void pre_cache_len_concat(const int* __restrict__ seq_lens_decoder,
                               const int* __restrict__ seq_lens_this_time,
                               int* __restrict__ cu_seqlens_k,
                               int* __restrict__ batch_ids,
                               int* __restrict__ tile_ids_per_batch,
                               int* __restrict__ num_blocks_x,
                               int* __restrict__ kv_token_num,
                               const int bsz,
                               const int num_row_per_block) {
  if (threadIdx.x == 0) {
    int gridx = 0;
    int index = 0;
    int total_tokens = 0;
    cu_seqlens_k[0] = 0;
    for (uint32_t bid = 0; bid < bsz; bid++) {
      int cache_len = seq_lens_decoder[bid];
      const int q_len = seq_lens_this_time[bid];
      if (q_len <= 0) {
        cache_len = 0;
      }
      const int loop_times = div_up(cache_len, num_row_per_block);
      for (uint32_t tile_id = 0; tile_id < loop_times; tile_id++) {
        batch_ids[index] = bid;
        tile_ids_per_batch[index++] = tile_id;
      }
      gridx += loop_times;
      total_tokens += (cache_len + q_len);
      cu_seqlens_k[bid + 1] = total_tokens;
    }
    *num_blocks_x = gridx;
    *kv_token_num = total_tokens;
  }
}

std::vector<paddle::Tensor> PreCacheLenConcat(
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const int max_dec_len,
    const int block_size) {
  auto stream = seq_lens_decoder.stream();
  auto place = seq_lens_decoder.place();
  int bsz = seq_lens_this_time.shape()[0];
  const uint32_t max_tile_size_per_bs_pre_cache = div_up(max_dec_len, block_size);

  paddle::Tensor cu_seqlens_k = GetEmptyTensor(
    {bsz + 1},
    paddle::DataType::INT32,
    place);
  paddle::Tensor pre_cache_batch_ids = GetEmptyTensor(
    {bsz * max_tile_size_per_bs_pre_cache},
    paddle::DataType::INT32,
    place);
  paddle::Tensor pre_cache_tile_ids_per_batch = GetEmptyTensor(
    {bsz * max_tile_size_per_bs_pre_cache},
    paddle::DataType::INT32,
    place);
  paddle::Tensor pre_cache_num_blocks =
    GetEmptyTensor({1}, paddle::DataType::INT32, place);
  paddle::Tensor kv_token_num =
    GetEmptyTensor({1}, paddle::DataType::INT32, place);

  pre_cache_len_concat<<<1, 32, 0, stream>>>(
    seq_lens_decoder.data<int>(),
    seq_lens_this_time.data<int>(),
    cu_seqlens_k.data<int>(),
    pre_cache_batch_ids.data<int>(),
    pre_cache_tile_ids_per_batch.data<int>(),
    pre_cache_num_blocks.data<int>(),
    kv_token_num.data<int>(),
    bsz,
    block_size
  );
  paddle::Tensor pre_cache_num_blocks_cpu = pre_cache_num_blocks.copy_to(paddle::CPUPlace(), false);
  paddle::Tensor kv_token_num_cpu = kv_token_num.copy_to(paddle::CPUPlace(), false);

  return {cu_seqlens_k,
          pre_cache_batch_ids,
          pre_cache_tile_ids_per_batch,
          pre_cache_num_blocks_cpu, /*cpu*/
          kv_token_num_cpu /*cpu*/
          };
}

std::vector<paddle::DataType> PreCacheLenConcatInferDtype(
    const paddle::DataType& seq_lens_decoder_dtype,
    const paddle::DataType& seq_lens_this_time_dtype) {
  return {paddle::DataType::INT32,
          paddle::DataType::INT32,
          paddle::DataType::INT32,
          paddle::DataType::INT32,
          paddle::DataType::INT32};
}

std::vector<std::vector<int64_t>> PreCacheLenConcatInferShape(
    const std::vector<int64_t>& seq_lens_decoder_shape,
    const std::vector<int64_t>& seq_lens_this_time_shape) {
  std::vector<int64_t> dynamic_shape = {-1};
  return {{seq_lens_this_time_shape[0] + 1},
          dynamic_shape,
          dynamic_shape,
          {1},
          {1}};
}

PD_BUILD_STATIC_OP(pre_cache_len_concat)
    .Inputs({"seq_lens_decoder",
             "seq_lens_this_time"})
    .Outputs({"cu_seqlens_k",
              "pre_cache_batch_ids",
              "pre_cache_tile_ids_per_batch",
              "pre_cache_num_blocks_cpu", /*cpu*/
              "kv_token_num_cpu"}) /*cpu*/
    .Attrs({"max_dec_len: int",
            "block_size: int"})
    .SetKernelFn(PD_KERNEL(PreCacheLenConcat))
    .SetInferShapeFn(PD_INFER_SHAPE(PreCacheLenConcatInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(PreCacheLenConcatInferDtype));
