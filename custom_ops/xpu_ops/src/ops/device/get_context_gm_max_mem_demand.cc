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

#include <fcntl.h>
#include <paddle/phi/backends/xpu/xpu_context.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdlib>
#include <random>
#include "paddle/extension.h"
#include "xpu/plugin.h"
#include "xpu/xpuml.h"

std::vector<paddle::Tensor> GetMaxMemDemand(int64_t device_id) {
  if (device_id == -1) {
    device_id = phi::backends::xpu::GetXPUCurrentDeviceId();
  }
  phi::XPUPlace place(device_id);
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext *>(dev_ctx);

  paddle::Tensor max_mem_demand = paddle::zeros({1}, paddle::DataType::INT64);

  max_mem_demand.data<int64_t>()[0] =
      xpu_ctx->x_context()->_gm_mgr.get_max_mem_demand();
  return {max_mem_demand};
}

std::vector<std::vector<int64_t>> GetMaxMemDemandInferShape() { return {{1}}; }

std::vector<paddle::DataType> GetMaxMemDemandInferDtype() {
  return {paddle::DataType::INT64};
}

PD_BUILD_OP(xpu_get_context_gm_max_mem_demand)
    .Inputs({})
    .Attrs({"devicd_id: int64_t"})
    .Outputs({"max_mem_demand"})
    .SetKernelFn(PD_KERNEL(GetMaxMemDemand))
    .SetInferShapeFn(PD_INFER_SHAPE(GetMaxMemDemandInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetMaxMemDemandInferDtype));
