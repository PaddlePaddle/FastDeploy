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

void TextImageIndexOut(const paddle::Tensor& token_type_ids,
                       const paddle::Tensor& text_index,
                       const paddle::Tensor& image_index) {
  if (token_type_ids.type() != paddle::DataType::INT32 ||
      text_index.type() != paddle::DataType::INT32 ||
      image_index.type() != paddle::DataType::INT32) {
    PD_THROW("NOT supported data type. Only support BFLOAT16. ");
  }
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  const int64_t token_num = token_type_ids.shape()[0];
  int r = baidu::xpu::api::plugin::text_image_index_out(
      xpu_ctx->x_context(),
      token_type_ids.data<int32_t>(),
      const_cast<int32_t*>(text_index.data<int32_t>()),
      const_cast<int32_t*>(image_index.data<int32_t>()),
      token_num);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "text_image_index_out");
}

PD_BUILD_OP(text_image_index_out)
    .Inputs({"token_type_ids", "text_index", "image_index"})
    .Outputs({"text_index_out", "image_index_out"})
    .SetInplaceMap({{"text_index", "text_index_out"},
                    {"image_index", "image_index_out"}})
    .SetKernelFn(PD_KERNEL(TextImageIndexOut));
