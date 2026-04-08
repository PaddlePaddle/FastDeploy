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
#include <xft/xdnn_plugin.h>
#include "paddle/extension.h"
#include "xpu/plugin.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

std::vector<paddle::Tensor> TextImageGatherScatter(
    paddle::Tensor& input,
    paddle::Tensor& text_input,
    paddle::Tensor& image_input,
    paddle::Tensor& token_type_ids,
    paddle::Tensor& text_index,
    paddle::Tensor& image_index,
    const bool is_scatter) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);

  const int64_t token_num = input.dims()[0];
  const int64_t hidden_size = input.dims()[1];
  const int64_t text_token_num = text_input.dims()[0];
  const int64_t image_token_num = image_input.dims()[0];

  switch (input.type()) {
    case paddle::DataType::BFLOAT16: {
      using XPUType = typename XPUTypeTrait<bfloat16>::Type;
      typedef paddle::bfloat16 data_t;
      int r = baidu::xpu::api::plugin::text_image_gather_scatter<XPUType>(
          xpu_ctx->x_context(),
          reinterpret_cast<XPUType*>(input.data<data_t>()),
          reinterpret_cast<XPUType*>(text_input.data<data_t>()),
          reinterpret_cast<XPUType*>(image_input.data<data_t>()),
          reinterpret_cast<int*>(token_type_ids.data<int>()),
          reinterpret_cast<int*>(text_index.data<int>()),
          reinterpret_cast<int*>(image_index.data<int>()),
          token_num,
          text_token_num,
          image_token_num,
          hidden_size,
          is_scatter);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "text_image_gather_scatter");
      break;
    }
    default: {
      PD_THROW("NOT supported data type. Only support BFLOAT16. ");
      break;
    }
  }
  return {input, text_input, image_input};
}

PD_BUILD_STATIC_OP(text_image_gather_scatter)
    .Inputs({"input",
             "text_input",
             "image_input",
             "token_type_ids",
             "text_index",
             "image_index"})
    .Outputs({"output", "text_input_out", "image_input_out"})
    .Attrs({"is_scatter:bool"})
    .SetInplaceMap({{"input", "output"},
                    {"text_input", "text_input_out"},
                    {"image_input", "image_input_out"}})
    .SetKernelFn(PD_KERNEL(TextImageGatherScatter));
