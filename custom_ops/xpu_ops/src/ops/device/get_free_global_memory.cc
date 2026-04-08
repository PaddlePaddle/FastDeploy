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

std::vector<paddle::Tensor> GetFreeGlobalMemory(int64_t device_id) {
  if (device_id == -1) {
    device_id = phi::backends::xpu::GetXPUCurrentDeviceId();
  }

  paddle::Tensor free_global_memory =
      paddle::zeros({1}, paddle::DataType::INT64);

  xpumlDevice_t device_handle;
  xpumlInit();
  xpumlDeviceGetHandleByIndex(device_id, &device_handle);
  xpumlMemory_t device_memory;
  xpumlDeviceGetMemoryInfo(device_handle, &device_memory);
  free_global_memory.data<int64_t>()[0] = device_memory.freeGlobalMemory;
  return {free_global_memory};
}

std::vector<std::vector<int64_t>> GetFreeGlobalMemoryInferShape() {
  return {{1}};
}

std::vector<paddle::DataType> GetFreeGlobalMemoryInferDtype() {
  return {paddle::DataType::INT64};
}

PD_BUILD_OP(xpu_get_free_global_memory)
    .Inputs({})
    .Attrs({"devicd_id: int64_t"})
    .Outputs({"free_global_memory"})
    .SetKernelFn(PD_KERNEL(GetFreeGlobalMemory))
    .SetInferShapeFn(PD_INFER_SHAPE(GetFreeGlobalMemoryInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetFreeGlobalMemoryInferDtype));
