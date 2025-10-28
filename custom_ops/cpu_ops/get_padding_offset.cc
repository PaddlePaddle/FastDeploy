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

#include "paddle/extension.h"

void remove_padding(int64_t *output_data,
                    const int64_t *input_data,
                    const int *seq_lens,
                    const int *cum_offsets,
                    const int sequence_length,
                    const int bsz) {
  for (int bi = 0; bi < bsz; ++bi) {
    for (int i = 0; i < seq_lens[bi]; ++i) {
      const int tgt_seq_id = bi * sequence_length - cum_offsets[bi] + i;
      const int src_seq_id = bi * sequence_length + i;
      output_data[tgt_seq_id] = input_data[src_seq_id];
    }
  }
}

void get_padding_offset_kernel(int *padding_offset,
                               int *cum_offsets_out,
                               int *cu_seqlens_q,
                               int *cu_seqlens_k,
                               const int *cum_offsets,
                               const int *seq_lens,
                               const int max_seq_len,
                               const int bsz) {
  for (int bi = 0; bi < bsz; ++bi) {
    int cum_offset = bi == 0 ? 0 : cum_offsets[bi - 1];
    auto seq_len_now = seq_lens[bi];
    for (int i = 0; i < seq_len_now; ++i) {
      padding_offset[bi * max_seq_len - cum_offset + i] = cum_offset;
    }
    cum_offsets_out[bi] = cum_offset;
    int cum_seq_len = (bi + 1) * max_seq_len - cum_offsets[bi];
    cu_seqlens_q[bi + 1] = cum_seq_len;
    cu_seqlens_k[bi + 1] = cum_seq_len;
  }
}

std::vector<paddle::Tensor> GetPaddingOffset(const paddle::Tensor &input_ids,
                                             const paddle::Tensor &cum_offsets,
                                             const paddle::Tensor &token_num,
                                             const paddle::Tensor &seq_len) {
  std::vector<int64_t> input_ids_shape = input_ids.shape();
  const int bsz = seq_len.shape()[0];
  const int seq_length = input_ids_shape[1];
  auto cum_offsets_out = cum_offsets.copy_to(paddle::CPUPlace(), false);
  auto cpu_token_num = token_num.copy_to(paddle::CPUPlace(), false);

  const int token_num_data = cpu_token_num.data<int64_t>()[0];
  auto x_remove_padding = paddle::empty(
      {token_num_data}, paddle::DataType::INT64, input_ids.place());
  auto padding_offset = paddle::empty(
      {token_num_data}, paddle::DataType::INT32, input_ids.place());
  auto cu_seqlens_q =
      paddle::full({bsz + 1}, 0, paddle::DataType::INT32, input_ids.place());
  auto cu_seqlens_k =
      paddle::full({bsz + 1}, 0, paddle::DataType::INT32, input_ids.place());
  get_padding_offset_kernel(padding_offset.data<int>(),
                            cum_offsets_out.data<int>(),
                            cu_seqlens_q.data<int>(),
                            cu_seqlens_k.data<int>(),
                            cum_offsets.data<int>(),
                            seq_len.data<int>(),
                            seq_length,
                            bsz);
  remove_padding(x_remove_padding.data<int64_t>(),
                 input_ids.data<int64_t>(),
                 seq_len.data<int>(),
                 cum_offsets_out.data<int>(),
                 seq_length,
                 bsz);
  return {x_remove_padding, padding_offset, cu_seqlens_q, cu_seqlens_k};
}

std::vector<std::vector<int64_t>> GetPaddingOffsetInferShape(
    const std::vector<int64_t> &input_ids_shape,
    const std::vector<int64_t> &cum_offsets_shape,
    const std::vector<int64_t> &token_num_shape,
    const std::vector<int64_t> &seq_len_shape) {
  int64_t bsz = seq_len_shape[0];
  int64_t seq_len = input_ids_shape[1];
  return {{-1}, {-1}, {bsz + 1}, {bsz + 1}};
}

std::vector<paddle::DataType> GetPaddingOffsetInferDtype(
    const paddle::DataType &input_ids_dtype,
    const paddle::DataType &cum_offsets_dtype,
    const paddle::DataType &token_num_dtype,
    const paddle::DataType &seq_len_dtype) {
  return {input_ids_dtype, seq_len_dtype, seq_len_dtype, seq_len_dtype};
}

PD_BUILD_STATIC_OP(get_padding_offset_cpu)
    .Inputs({"input_ids", "cum_offsets", "token_num", "seq_len"})
    .Outputs(
        {"x_remove_padding", "padding_offset", "cu_seqlens_q", "cu_seqlens_k"})
    .SetKernelFn(PD_KERNEL(GetPaddingOffset))
    .SetInferShapeFn(PD_INFER_SHAPE(GetPaddingOffsetInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetPaddingOffsetInferDtype));
