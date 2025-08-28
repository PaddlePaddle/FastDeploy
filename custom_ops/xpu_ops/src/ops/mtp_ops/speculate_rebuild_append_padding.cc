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

#include <paddle/phi/backends/xpu/xpu_context.h>
#include "paddle/extension.h"
#include "xpu/plugin.h"

namespace api = baidu::xpu::api;
std::vector<paddle::Tensor> RebuildAppendPadding(
    const paddle::Tensor& full_hidden_states,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& seq_len_encoder,
    const paddle::Tensor& seq_len_decoder,
    const paddle::Tensor& output_padding_offset,
    int max_seq_len) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  api::Context* ctx = xpu_ctx->x_context();
  if (full_hidden_states.is_cpu()) {
    ctx = new api::Context(api::kCPU);
  }

  int dim_embed = full_hidden_states.shape()[1];
  int output_token_num = output_padding_offset.shape()[0];
  int elem_nums = output_token_num * dim_embed;

  auto out = paddle::full({output_token_num, dim_embed},
                          0,
                          full_hidden_states.dtype(),
                          full_hidden_states.place());

  int r;
  switch (full_hidden_states.dtype()) {
    case paddle::DataType::BFLOAT16:
      using XPUTypeBF16 = typename XPUTypeTrait<bfloat16>::Type;
      typedef paddle::bfloat16 bf16_data_t;
      r = api::plugin::speculate_rebuild_append_padding<XPUTypeBF16>(
          ctx,
          const_cast<XPUTypeBF16*>(reinterpret_cast<const XPUTypeBF16*>(
              full_hidden_states.data<bf16_data_t>())),
          const_cast<int*>(cum_offsets.data<int>()),
          const_cast<int*>(seq_len_encoder.data<int>()),
          const_cast<int*>(seq_len_decoder.data<int>()),
          const_cast<int*>(output_padding_offset.data<int>()),
          max_seq_len,
          dim_embed,
          elem_nums,
          reinterpret_cast<XPUTypeBF16*>(out.data<bf16_data_t>()));
      PD_CHECK(r == 0, "xpu::plugin::speculate_rebuild_append_padding failed.");
      return {out};
    case paddle::DataType::FLOAT16:
      using XPUTypeFP16 = typename XPUTypeTrait<float16>::Type;
      typedef paddle::float16 fp16_data_t;
      r = api::plugin::speculate_rebuild_append_padding<XPUTypeFP16>(
          ctx,
          const_cast<XPUTypeFP16*>(reinterpret_cast<const XPUTypeFP16*>(
              full_hidden_states.data<fp16_data_t>())),
          const_cast<int*>(cum_offsets.data<int>()),
          const_cast<int*>(seq_len_encoder.data<int>()),
          const_cast<int*>(seq_len_decoder.data<int>()),
          const_cast<int*>(output_padding_offset.data<int>()),
          max_seq_len,
          dim_embed,
          elem_nums,
          reinterpret_cast<XPUTypeFP16*>(out.data<fp16_data_t>()));
      PD_CHECK(r == 0, "xpu::plugin::speculate_rebuild_append_padding failed.");
      return {out};
    case paddle::DataType::FLOAT32:
      r = api::plugin::speculate_rebuild_append_padding<float>(
          ctx,
          const_cast<float*>(full_hidden_states.data<float>()),
          const_cast<int*>(cum_offsets.data<int>()),
          const_cast<int*>(seq_len_encoder.data<int>()),
          const_cast<int*>(seq_len_decoder.data<int>()),
          const_cast<int*>(output_padding_offset.data<int>()),
          max_seq_len,
          dim_embed,
          elem_nums,
          out.data<float>());
      PD_CHECK(r == 0, "xpu::plugin::speculate_rebuild_append_padding failed.");
      return {out};
    default:
      PD_THROW("Unsupported data type.");
  }
}

std::vector<std::vector<int64_t>> RebuildAppendPaddingInferShape(
    const std::vector<int64_t>& full_hidden_states_shape,
    const std::vector<int64_t>& cum_offsets_shape,
    const std::vector<int64_t>& seq_len_encoder_shape,
    const std::vector<int64_t>& seq_len_decoder_shape,
    const std::vector<int64_t>& output_padding_offset_shape) {
  const int64_t output_token_num = output_padding_offset_shape[0];
  const int64_t dim_embed = full_hidden_states_shape[1];
  std::vector<int64_t> out_shape = {output_token_num, dim_embed};
  return {out_shape};
}

std::vector<paddle::DataType> RebuildAppendPaddingInferDtype(
    const paddle::DataType& full_hidden_states_dtype,
    const paddle::DataType& cum_offsets_dtype,
    const paddle::DataType& seq_len_encoder_dtype,
    const paddle::DataType& seq_len_decoder_dtype,
    const paddle::DataType& output_padding_offset_dtype) {
  return {full_hidden_states_dtype};
}

PD_BUILD_OP(speculate_rebuild_append_padding)
    .Inputs({"full_hidden_states",
             "cum_offsets",
             "seq_len_encoder",
             "seq_len_decoder",
             "output_padding_offset"})
    .Attrs({"max_seq_len: int"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(RebuildAppendPadding))
    .SetInferShapeFn(PD_INFER_SHAPE(RebuildAppendPaddingInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RebuildAppendPaddingInferDtype));
