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

#include <cooperative_groups.h>

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

namespace cg = cooperative_groups;

__global__ void SpeculatePreProcessKernel(int64_t *ids_remove_padding,
                                          int *batch_id_per_token,
                                          int *cu_seqlens_q,
                                          int *cu_seqlens_k,
                                          int *seq_lens_output,
                                          int *cu_seq_lens_q_output,
                                          int *batch_id_per_token_output,
                                          int *real_output_token_num,
                                          const int64_t *input_data,
                                          const int *seq_lens,
                                          const int max_seq_len,
                                          const int64_t *draft_tokens,
                                          const int *seq_lens_encoder,
                                          const int max_draft_tokens_per_batch,
                                          const int real_bsz) {
  auto grid = cg::this_grid();
  const int bi = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;

  int cum_seq_len = 0;

  // compute sum of seq_lens[0, 1, 2, ...,bi] per warp
  for (int i = lane_id; i < bi + 1; i += WARP_SIZE) {
    cum_seq_len += seq_lens[i];
  }

#pragma unroll
  for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
    cum_seq_len += __shfl_xor_sync(0xffffffff, cum_seq_len, mask);
  }

  if (tid == 0) {
    cu_seqlens_q[bi + 1] = cum_seq_len;
    cu_seqlens_k[bi + 1] = cum_seq_len;
  }

  if (bi == 0 && tid == 0) {
    cu_seqlens_q[0] = 0;
    cu_seqlens_k[0] = 0;
  }

  for (int i = tid; i < seq_lens[bi]; i += blockDim.x) {
    const int tgt_seq_id = cum_seq_len - seq_lens[bi] + i;
    if (max_draft_tokens_per_batch > 0 && seq_lens_encoder[bi] <= 0) {
      // speculative decoding
      const int src_seq_id = bi * max_draft_tokens_per_batch + i;
      ids_remove_padding[tgt_seq_id] = draft_tokens[src_seq_id];
    } else {
      // Non-speculative decoding
      const int src_seq_id = bi * max_seq_len + i;
      ids_remove_padding[tgt_seq_id] = input_data[src_seq_id];
    }

    batch_id_per_token[tgt_seq_id] = bi;
  }

  for (int bid = blockIdx.x * blockDim.x + threadIdx.x; bid < real_bsz;
       bid += gridDim.x * blockDim.x) {
    if (seq_lens[bid] == 0) {
      seq_lens_output[bid] = 0;
    } else if (seq_lens[bid] == 1) {
      seq_lens_output[bid] = 1;
    } else if (seq_lens_encoder[bid] != 0) {
      seq_lens_output[bid] = 1;
    } else {
      seq_lens_output[bid] = seq_lens[bid];
    }
  }

  grid.sync();

  int cum_seq_len_output = 0;

  // compute sum of seq_lens_output[0,1,2,...,bi] per warp
  for (int i = lane_id; i < bi + 1; i += WARP_SIZE) {
    cum_seq_len_output += seq_lens_output[i];
  }

#pragma unroll
  for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
    cum_seq_len_output += __shfl_xor_sync(0xffffffff, cum_seq_len_output, mask);
  }

  if (tid == 0) {
    cu_seq_lens_q_output[bi + 1] = cum_seq_len_output;
  }

  if (bi == 0 && tid == 0) {
    cu_seq_lens_q_output[0] = 0;
  }

  // get real output token num
  if (bi == real_bsz - 1 && tid == 0) {
    real_output_token_num[0] = cum_seq_len_output;
  }

  for (int i = tid; i < seq_lens_output[bi]; i += blockDim.x) {
    const int tgt_seq_id_output = cum_seq_len_output - seq_lens_output[bi] + i;
    batch_id_per_token_output[tgt_seq_id_output] = bi;
  }
}

std::vector<paddle::Tensor> SpeculatePreProcess(
    const int64_t cpu_token_num,
    const paddle::Tensor &input_ids,
    const paddle::Tensor &seq_len,
    const paddle::Tensor &draft_tokens,
    const paddle::Tensor &seq_lens_encoder,
    const paddle::Tensor &seq_lens_decoder) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          input_ids.place()));
  auto cu_stream = dev_ctx->stream();
#else
  auto cu_stream = input_ids.stream();
#endif
  std::vector<int64_t> input_ids_shape = input_ids.shape();
  const int bsz = seq_len.shape()[0];
  const int max_seq_len = input_ids_shape[1];
  const int token_num_data = cpu_token_num;
  auto ids_remove_padding = paddle::empty(
      {token_num_data}, paddle::DataType::INT64, input_ids.place());
  auto batch_id_per_token = paddle::empty(
      {token_num_data}, paddle::DataType::INT32, input_ids.place());
  auto cu_seqlens_q =
      paddle::empty({bsz + 1}, paddle::DataType::INT32, input_ids.place());
  auto cu_seqlens_k =
      paddle::empty({bsz + 1}, paddle::DataType::INT32, input_ids.place());

#ifdef PADDLE_WITH_COREX
  int blockSize =
      std::min((token_num_data + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE, 128);
#else
  int blockSize =
      min((token_num_data + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE, 128);
#endif

  const int max_draft_tokens_per_batch = draft_tokens.shape()[1];

  auto seq_lens_output =
      paddle::empty({bsz}, paddle::DataType::INT32, input_ids.place());
  auto cu_seq_lens_q_output =
      paddle::empty({bsz + 1}, paddle::DataType::INT32, input_ids.place());
  auto batch_id_per_token_output =
      paddle::empty({bsz * max_draft_tokens_per_batch},
                    paddle::DataType::INT32,
                    input_ids.place());
  auto real_output_token_num =
      paddle::empty({1}, paddle::DataType::INT32, input_ids.place());

  if (token_num_data == 0) {
    return {ids_remove_padding,
            batch_id_per_token,
            cu_seqlens_q,
            cu_seqlens_k,
            cu_seq_lens_q_output,
            batch_id_per_token_output,
            real_output_token_num};
  }

  int64_t *ids_remove_padding_ptr = ids_remove_padding.data<int64_t>();
  int *batch_id_per_token_ptr = batch_id_per_token.data<int>();
  int *cu_seqlens_q_ptr = cu_seqlens_q.data<int>();
  int *cu_seqlens_k_ptr = cu_seqlens_k.data<int>();
  int *seq_lens_output_ptr = seq_lens_output.data<int>();
  int *cu_seq_lens_q_output_ptr = cu_seq_lens_q_output.data<int>();
  int *batch_id_per_token_output_ptr = batch_id_per_token_output.data<int>();
  int *real_output_token_num_ptr = real_output_token_num.data<int>();
  const int64_t *input_data_ptr = input_ids.data<int64_t>();
  const int *seq_len_ptr = seq_len.data<int>();
  const int64_t *draft_tokens_ptr = draft_tokens.data<int64_t>();
  const int *seq_lens_encoder_ptr = seq_lens_encoder.data<int>();

  void *kernel_args[] = {(void *)&ids_remove_padding_ptr,
                         (void *)&batch_id_per_token_ptr,
                         (void *)&cu_seqlens_q_ptr,
                         (void *)&cu_seqlens_k_ptr,
                         (void *)&seq_lens_output_ptr,
                         (void *)&cu_seq_lens_q_output_ptr,
                         (void *)&batch_id_per_token_output_ptr,
                         (void *)&real_output_token_num_ptr,
                         (void *)&input_data_ptr,
                         (void *)&seq_len_ptr,
                         (void *)&max_seq_len,
                         (void *)&draft_tokens_ptr,
                         (void *)&seq_lens_encoder_ptr,
                         (void *)&max_draft_tokens_per_batch,
                         (void *)&bsz};

  cudaLaunchCooperativeKernel((void *)SpeculatePreProcessKernel,
                              dim3(bsz),
                              dim3(blockSize),
                              kernel_args,
                              0,
                              cu_stream);

  return {ids_remove_padding,
          batch_id_per_token,
          cu_seqlens_q,
          cu_seqlens_k,
          cu_seq_lens_q_output,
          batch_id_per_token_output,
          real_output_token_num};
}

PD_BUILD_STATIC_OP(speculate_pre_process)
    .Inputs({"input_ids",
             "seq_len",
             "draft_tokens",
             "seq_lens_encoder",
             "seq_lens_decoder"})
    .Outputs({"ids_remove_padding",
              "batch_id_per_token",
              "cu_seqlens_q",
              "cu_seqlens_k",
              "cu_seq_lens_q_output",
              "batch_id_per_token_output",
              "real_output_token_num"})
    .Attrs({"cpu_token_num: int64_t"})
    .SetKernelFn(PD_KERNEL(SpeculatePreProcess));
