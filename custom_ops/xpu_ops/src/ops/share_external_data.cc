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
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_meta.h"
#include "xpu/plugin.h"
#include "xpu_multiprocess.h"  // NOLINT(build/include_subdir)

std::vector<paddle::Tensor> ShareExternalData(const paddle::Tensor& input,
                                              const std::string shm_name,
                                              const std::vector<int>& shape,
                                              bool use_ipc) {
  sharedMemoryInfo info;
  int ret = sharedMemoryOpen(shm_name.c_str(), sizeof(shmStruct), &info);
  PD_CHECK(ret == 0, "sharedMemoryOpen failed");
  volatile shmStruct* shm = static_cast<volatile shmStruct*>(info.addr);
  void* data_ptr_addr = nullptr;
  if (use_ipc) {
#if XPURT_VERSION_MAJOR == 5
    int ret = xpu_ipc_open_memhandle(&data_ptr_addr,
                                     *(XPUIpcMemHandle*)&shm->memHandle,
                                     0x01);  // NOLINT
    PD_CHECK(ret == XPU_SUCCESS, shm_name, " xpu_ipc_open_memhandle failed");
#elif XPURT_VERSION_MAJOR == 4
    PD_THROW("kl2 not support prefix cache");
#endif
  } else {
    data_ptr_addr = reinterpret_cast<void*>(shm->data_ptr_addr);
  }

  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  paddle::Tensor output = paddle::from_blob(
      data_ptr_addr, shape, input.dtype(), input.layout(), place);

  sharedMemoryClose(&info);
  return {output};
}

PD_BUILD_OP(share_external_data)
    .Inputs({"input"})
    .Outputs({"output"})
    .Attrs({"shm_name: std::string",
            "shape: std::vector<int>",
            "use_ipc: bool"})
    .SetKernelFn(PD_KERNEL(ShareExternalData));
