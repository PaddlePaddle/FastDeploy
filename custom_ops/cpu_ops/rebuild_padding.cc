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

#include <vector>
#include "paddle/extension.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

template <typename T>
void RebuildPaddingCPUImpl(T *output_data,
                           const T *input_data,
                           const int *cu_seqlens_q_data,
                           const int *seq_len_this_time_data,
                           const int *seq_lens_decoder_data,
                           const int *seq_lens_encoder_data,
                           int max_input_length,
                           int dim_embed,
                           const int elem_nums) {
  for (int i = 0; i < elem_nums; ++i) {
    const int bi = i / dim_embed;
    const int bias_idx = i % dim_embed;
    int seq_id = 0;

    if (seq_len_this_time_data[bi] == 0) {
      continue;
    }
    if (seq_lens_decoder_data[bi] == 0 && seq_lens_encoder_data[bi] == 0) {
      continue;
    }

    if (seq_lens_encoder_data[bi] > 0) {
      seq_id = seq_lens_encoder_data[bi] - 1;
    }

    const int ori_token_idx = cu_seqlens_q_data[bi] + seq_id;
    const int src_offset = ori_token_idx * dim_embed + bias_idx;

    output_data[i] = input_data[src_offset];
  }
}

template <typename T>
void RebuildAppendPaddingCPUImpl(T *output_data,
                                 const T *input_data,
                                 const int *cu_seqlens_q_data,
                                 const int *seq_len_this_time_data,
                                 const int *seq_lens_decoder_data,
                                 const int *seq_lens_encoder_data,
                                 const int *output_padding_offset_data,
                                 const int max_input_length,
                                 const int dim_embed,
                                 const int64_t output_elem_nums) {
  for (int i = 0; i < output_elem_nums; ++i) {
    int out_token_id = i / dim_embed;
    int ori_token_id = out_token_id + output_padding_offset_data[out_token_id];
    int bi = ori_token_id / max_input_length;
    if (seq_len_this_time_data[bi] == 0 ||
        (seq_lens_decoder_data[bi] == 0 && seq_lens_encoder_data[bi] == 0)) {
      continue;
    }
    int seq_id = 0;

    if (seq_lens_encoder_data[bi] > 0) {
      seq_id = seq_lens_encoder_data[bi] - 1;
    }
    int input_token_id = cu_seqlens_q_data[bi] + seq_id;
    int bias_idx = i % dim_embed;
    int src_offset = input_token_id * dim_embed + bias_idx;

    output_data[i] = input_data[src_offset];
  }
}

std::vector<paddle::Tensor> RebuildPaddingCPU(
    const paddle::Tensor &tmp_out,
    const paddle::Tensor &cu_seqlens_q,
    const paddle::Tensor &seq_len_this_time,
    const paddle::Tensor &seq_lens_decoder,
    const paddle::Tensor &seq_lens_encoder,
    const paddle::optional<paddle::Tensor> &output_padding_offset,
    int max_input_length) {
  auto tmp_out_cpu = tmp_out.copy_to(paddle::CPUPlace(), true);
  auto cu_seqlens_q_cpu = cu_seqlens_q.copy_to(paddle::CPUPlace(), true);
  auto seq_len_this_time_cpu =
      seq_len_this_time.copy_to(paddle::CPUPlace(), true);
  auto seq_lens_decoder_cpu =
      seq_lens_decoder.copy_to(paddle::CPUPlace(), true);
  auto seq_lens_encoder_cpu =
      seq_lens_encoder.copy_to(paddle::CPUPlace(), true);
  paddle::optional<paddle::Tensor> output_padding_offset_cpu;
  if (output_padding_offset) {
    output_padding_offset_cpu =
        output_padding_offset->copy_to(paddle::CPUPlace(), true);
  }

  int token_num = tmp_out_cpu.shape()[0];
  int dim_embed = tmp_out_cpu.shape()[1];
  int bsz = cu_seqlens_q_cpu.shape()[0] - 1;

  paddle::Tensor out;
  if (output_padding_offset_cpu) {
    int need_delete_token_num = 0;
    for (int i = 0; i < bsz; ++i) {
      if (seq_lens_encoder_cpu.data<int>()[i] > 0) {
        need_delete_token_num += seq_lens_encoder_cpu.data<int>()[i] - 1;
      }
    }
    int output_token_num = token_num - need_delete_token_num;
    out = paddle::full({output_token_num, dim_embed},
                       0,
                       tmp_out_cpu.dtype(),
                       paddle::CPUPlace());
  } else {
    out = paddle::full(
        {bsz, dim_embed}, 0, tmp_out_cpu.dtype(), paddle::CPUPlace());
  }

  const int *cu_seqlens_q_data = cu_seqlens_q_cpu.data<int>();
  const int *seq_len_this_time_data = seq_len_this_time_cpu.data<int>();
  const int *seq_lens_decoder_data = seq_lens_decoder_cpu.data<int>();
  const int *seq_lens_encoder_data = seq_lens_encoder_cpu.data<int>();
  int elem_nums = out.numel();

  if (output_padding_offset_cpu) {
    const int *output_padding_offset_data =
        output_padding_offset_cpu->data<int>();
    switch (tmp_out_cpu.dtype()) {
      case paddle::DataType::FLOAT32:
        RebuildAppendPaddingCPUImpl<float>(out.data<float>(),
                                           tmp_out_cpu.data<float>(),
                                           cu_seqlens_q_data,
                                           seq_len_this_time_data,
                                           seq_lens_decoder_data,
                                           seq_lens_encoder_data,
                                           output_padding_offset_data,
                                           max_input_length,
                                           dim_embed,
                                           elem_nums);
        break;
      case paddle::DataType::FLOAT16:
        RebuildAppendPaddingCPUImpl<paddle::float16>(
            out.data<paddle::float16>(),
            tmp_out_cpu.data<paddle::float16>(),
            cu_seqlens_q_data,
            seq_len_this_time_data,
            seq_lens_decoder_data,
            seq_lens_encoder_data,
            output_padding_offset_data,
            max_input_length,
            dim_embed,
            elem_nums);
        break;
      case paddle::DataType::BFLOAT16:
        RebuildAppendPaddingCPUImpl<paddle::bfloat16>(
            out.data<paddle::bfloat16>(),
            tmp_out_cpu.data<paddle::bfloat16>(),
            cu_seqlens_q_data,
            seq_len_this_time_data,
            seq_lens_decoder_data,
            seq_lens_encoder_data,
            output_padding_offset_data,
            max_input_length,
            dim_embed,
            elem_nums);
        break;
      default:
        PD_THROW(
            "Unsupported data type for rebuild_padding_cpu. "
            "Only float32, float16, and bfloat16 are supported.");
    }
  } else {
    switch (tmp_out_cpu.dtype()) {
      case paddle::DataType::FLOAT32:
        RebuildPaddingCPUImpl<float>(out.data<float>(),
                                     tmp_out_cpu.data<float>(),
                                     cu_seqlens_q_data,
                                     seq_len_this_time_data,
                                     seq_lens_decoder_data,
                                     seq_lens_encoder_data,
                                     max_input_length,
                                     dim_embed,
                                     elem_nums);
        break;
      case paddle::DataType::FLOAT16:
        RebuildPaddingCPUImpl<paddle::float16>(
            out.data<paddle::float16>(),
            tmp_out_cpu.data<paddle::float16>(),
            cu_seqlens_q_data,
            seq_len_this_time_data,
            seq_lens_decoder_data,
            seq_lens_encoder_data,
            max_input_length,
            dim_embed,
            elem_nums);
        break;
      case paddle::DataType::BFLOAT16:
        RebuildPaddingCPUImpl<paddle::bfloat16>(
            out.data<paddle::bfloat16>(),
            tmp_out_cpu.data<paddle::bfloat16>(),
            cu_seqlens_q_data,
            seq_len_this_time_data,
            seq_lens_decoder_data,
            seq_lens_encoder_data,
            max_input_length,
            dim_embed,
            elem_nums);
        break;
      default:
        PD_THROW(
            "Unsupported data type for rebuild_padding_cpu. "
            "Only float32, float16, and bfloat16 are supported.");
    }
  }
  return {out};
}

std::vector<std::vector<int64_t>> RebuildPaddingInferShape(
    const std::vector<int64_t> &tmp_out_shape,
    const std::vector<int64_t> &cu_seqlens_q_shape,
    const std::vector<int64_t> &seq_len_this_time_shape,
    const std::vector<int64_t> &seq_lens_decoder_shape,
    const std::vector<int64_t> &seq_lens_encoder_shape,
    const paddle::optional<std::vector<int64_t>> &output_padding_offset_shape) {
  int64_t dim_embed = tmp_out_shape[1];
  if (output_padding_offset_shape) {
    return {{-1, dim_embed}};
  } else {
    int64_t bsz = cu_seqlens_q_shape[0] - 1;
    return {{bsz, dim_embed}};
  }
}

std::vector<paddle::DataType> RebuildPaddingInferDtype(
    const paddle::DataType &tmp_out_dtype,
    const paddle::DataType &cu_seqlens_q_dtype,
    const paddle::DataType &seq_len_this_time_dtype,
    const paddle::DataType &seq_lens_decoder_dtype,
    const paddle::DataType &seq_lens_encoder_dtype,
    const paddle::optional<paddle::DataType> &output_padding_offset_dtype) {
  return {tmp_out_dtype};
}

PD_BUILD_STATIC_OP(rebuild_padding_cpu)
    .Inputs({"tmp_out",
             "cu_seqlens_q",
             "seq_len_this_time",
             "seq_lens_decoder",
             "seq_lens_encoder",
             paddle::Optional("output_padding_offset")})
    .Outputs({"out"})
    .Attrs({"max_input_length: int"})
    .SetKernelFn(PD_KERNEL(RebuildPaddingCPU))
    .SetInferShapeFn(PD_INFER_SHAPE(RebuildPaddingInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RebuildPaddingInferDtype));
