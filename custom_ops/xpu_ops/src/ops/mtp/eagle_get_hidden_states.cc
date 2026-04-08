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

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

namespace api = baidu::xpu::api;
std::vector<paddle::Tensor> EagleGetHiddenStates(
    const paddle::Tensor& input,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& stop_flags,
    const paddle::Tensor& accept_nums,
    const paddle::Tensor& base_model_seq_lens_this_time,
    const paddle::Tensor& base_model_seq_lens_encoder,
    const int actual_draft_token_num) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  api::Context* ctx = xpu_ctx->x_context();
  if (input.is_cpu()) {
    ctx = new api::Context(api::kCPU);
  }

  auto input_token_num = input.shape()[0];
  auto dim_embed = input.shape()[1];
  int bsz = seq_lens_this_time.shape()[0];
  auto position_map = paddle::full(
      {input_token_num}, -1, seq_lens_this_time.dtype(), input.place());
  auto output_token_num = paddle::empty(
      {1}, seq_lens_this_time.dtype(), seq_lens_this_time.place());

  int r = api::plugin::compute_order(ctx,
                                     seq_lens_this_time.data<int>(),
                                     seq_lens_encoder.data<int>(),
                                     base_model_seq_lens_this_time.data<int>(),
                                     base_model_seq_lens_encoder.data<int>(),
                                     accept_nums.data<int>(),
                                     position_map.data<int>(),
                                     output_token_num.data<int>(),
                                     bsz,
                                     actual_draft_token_num,
                                     input_token_num);
  PD_CHECK(r == 0, "xpu::plugin::compute_order failed.");

  int output_token_num_cpu =
      output_token_num.copy_to(paddle::CPUPlace(), false).data<int>()[0];
  auto out = paddle::empty(
      {output_token_num_cpu, dim_embed}, input.dtype(), input.place());
  int elem_cnt = input_token_num * dim_embed;

  switch (input.dtype()) {
    case paddle::DataType::BFLOAT16:
      using XPUTypeBF16 = typename XPUTypeTrait<bfloat16>::Type;
      typedef paddle::bfloat16 bf16_data_t;
      r = api::plugin::rebuild_hidden_states(
          ctx,
          reinterpret_cast<const XPUTypeBF16*>(input.data<bf16_data_t>()),
          position_map.data<int>(),
          reinterpret_cast<XPUTypeBF16*>(out.data<bf16_data_t>()),
          dim_embed,
          elem_cnt);
      PD_CHECK(r == 0, "xpu::plugin::rebuild_hidden_states failed.");
      return {out};
    case paddle::DataType::FLOAT16:
      using XPUTypeFP16 = typename XPUTypeTrait<float16>::Type;
      typedef paddle::float16 fp16_data_t;
      r = api::plugin::rebuild_hidden_states(
          ctx,
          reinterpret_cast<const XPUTypeFP16*>(input.data<fp16_data_t>()),
          position_map.data<int>(),
          reinterpret_cast<XPUTypeFP16*>(out.data<fp16_data_t>()),
          dim_embed,
          elem_cnt);
      PD_CHECK(r == 0, "xpu::plugin::rebuild_hidden_states failed.");
      return {out};
    case paddle::DataType::FLOAT32:
      r = api::plugin::rebuild_hidden_states(
          ctx,
          reinterpret_cast<const float*>(input.data<float>()),
          position_map.data<int>(),
          reinterpret_cast<float*>(out.data<float>()),
          dim_embed,
          elem_cnt);
      PD_CHECK(r == 0, "xpu::plugin::rebuild_hidden_states failed.");
      return {out};
    default:
      PD_THROW("Unsupported data type.");
  }
}

PD_BUILD_STATIC_OP(eagle_get_hidden_states)
    .Inputs({"input",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "stop_flags",
             "accept_nums",
             "base_model_seq_lens_this_time",
             "base_model_seq_lens_encoder"})
    .Attrs({"actual_draft_token_num: int"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(EagleGetHiddenStates));
