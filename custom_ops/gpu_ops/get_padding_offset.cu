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

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

__global__ void PrefixSumKernel(int64_t *ids_remove_padding,
                                int *batch_id_per_token,
                                int *cu_seqlens_q,
                                int *cu_seqlens_k,
                                const int64_t *input_data,
                                const int *seq_lens,
                                const int max_seq_len,
                                const int64_t *draft_tokens,
                                const int *seq_lens_encoder,
                                const int max_draft_tokens_per_batch) {
  const int bi = blockIdx.x;
  const int tid = threadIdx.x;
#ifdef PADDLE_WITH_COREX
  const int warp_id = threadIdx.x / 64;
  const int lane_id = threadIdx.x % 64;
#else
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
#endif

  int cum_seq_len = 0;

  // compute sum of seq_lens[0,1,2,...,bi]
  for (int i = lane_id; i < bi + 1; i += WARP_SIZE) {
    cum_seq_len += seq_lens[i];
  }

  for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
    const int tmp = __shfl_up_sync(0xffffffff, cum_seq_len, offset);
    if (lane_id >= offset) cum_seq_len += tmp;
  }

  cum_seq_len = __shfl_sync(0xffffffff, cum_seq_len, WARP_SIZE - 1);

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
}

std::vector<paddle::Tensor> GetPaddingOffset(
    const paddle::Tensor &input_ids,
    const paddle::Tensor &seq_len,
    const paddle::optional<paddle::Tensor> &draft_tokens,
    const paddle::optional<paddle::Tensor> &seq_lens_encoder,
    const int64_t cpu_token_num) {
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
  auto x_remove_padding = paddle::full(
      {token_num_data}, 2, paddle::DataType::INT64, input_ids.place());
  auto batch_id_per_token = paddle::full(
      {token_num_data}, -1, paddle::DataType::INT32, input_ids.place());
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

  int max_draft_tokens_per_batch = -1;
  if (draft_tokens) {
    max_draft_tokens_per_batch = draft_tokens.get().shape()[1];
  }

  PrefixSumKernel<<<bsz, blockSize, 0, cu_stream>>>(
      x_remove_padding.data<int64_t>(),
      batch_id_per_token.data<int>(),
      cu_seqlens_q.data<int>(),
      cu_seqlens_k.data<int>(),
      input_ids.data<int64_t>(),
      seq_len.data<int>(),
      max_seq_len,
      draft_tokens ? draft_tokens.get().data<int64_t>() : nullptr,
      seq_lens_encoder ? seq_lens_encoder.get().data<int32_t>() : nullptr,
      max_draft_tokens_per_batch);

  return {x_remove_padding, batch_id_per_token, cu_seqlens_q, cu_seqlens_k};
}

std::vector<std::vector<int64_t>> GetPaddingOffsetInferShape(
    const std::vector<int64_t> &input_ids_shape,
    const std::vector<int64_t> &token_num_shape,
    const std::vector<int64_t> &seq_len_shape) {
  int64_t bsz = seq_len_shape[0];
  int64_t seq_len = input_ids_shape[1];
  return {{-1}, {-1}, {bsz + 1}, {bsz + 1}};
}

std::vector<paddle::DataType> GetPaddingOffsetInferDtype(
    const paddle::DataType &input_ids_dtype,
    const paddle::DataType &token_num_dtype,
    const paddle::DataType &seq_len_dtype) {
  return {input_ids_dtype, seq_len_dtype, seq_len_dtype, seq_len_dtype};
}

PD_BUILD_STATIC_OP(get_padding_offset)
    .Inputs({"input_ids",
             "seq_len",
             paddle::Optional("draft_tokens"),
             paddle::Optional("seq_lens_encoder")})
    .Outputs({"x_remove_padding",
              "batch_id_per_token",
              "cu_seqlens_q",
              "cu_seqlens_k"})
    .Attrs({"cpu_token_num: int64_t"})
    .SetKernelFn(PD_KERNEL(GetPaddingOffset))
    .SetInferShapeFn(PD_INFER_SHAPE(GetPaddingOffsetInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetPaddingOffsetInferDtype));
