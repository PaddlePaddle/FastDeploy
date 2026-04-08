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
#include <paddle/phi/backends/xpu/xpu_context.h>
#include "paddle/extension.h"
#include "xpu/plugin.h"
#include "xpu_multiprocess.h"  // NOLINT

void ReadDataIpc(const paddle::Tensor &tmp_input, const std::string &shm_name) {
  volatile shmStruct *shm = NULL;
  sharedMemoryInfo info;
  int ret = sharedMemoryOpen(shm_name.c_str(), sizeof(shmStruct), &info);
  PD_CHECK(ret == 0, "sharedMemoryOpen failed");

  shm = static_cast<volatile shmStruct *>(info.addr);
  void *ptr = nullptr;
#if XPURT_VERSION_MAJOR == 5
  ret = xpu_ipc_open_memhandle(
      &ptr, *(XPUIpcMemHandle *)&shm->memHandle, 0x01);  // NOLINT
#elif XPURT_VERSION_MAJOR == 4
  PD_THROW("kl2 not support prefix cache");
#endif
  PD_CHECK(ret == XPU_SUCCESS, "xpu_ipc_open_memhandle failed");
  PD_CHECK(tmp_input.place().GetType() == phi::AllocationType::CPU);
  // switch (tmp_input.dtype()) {
  //   case paddle::DataType::FLOAT32:
  //     ret = xpu_memcpy(const_cast<float *>(tmp_input.data<float>()),
  //                      ptr,
  //                      tmp_input.numel() * sizeof(float),
  //                      XPUMemcpyKind::XPU_DEVICE_TO_HOST);
  //     break;
  //   case paddle::DataType::FLOAT16:
  //     ret = xpu_memcpy(const_cast<phi::dtype::float16 *>(
  //                          tmp_input.data<phi::dtype::float16>()),
  //                      ptr,
  //                      tmp_input.numel() * sizeof(phi::dtype::float16),
  //                      XPUMemcpyKind::XPU_DEVICE_TO_HOST);
  //     break;
  //   case paddle::DataType::UINT8:
  //     ret = xpu_memcpy(const_cast<uint8_t *>(tmp_input.data<uint8_t>()),
  //                      ptr,
  //                      tmp_input.numel() * sizeof(uint8_t),
  //                      XPUMemcpyKind::XPU_DEVICE_TO_HOST);
  //     break;
  //   default:
  //     PD_THROW("not support dtype: ",
  //     phi::DataTypeToString(tmp_input.dtype()));
  // }
  // PD_CHECK(ret == XPU_SUCCESS, "not support dtype");
  // ret = xpu_ipc_close_memhandle(ptr);
  // PD_CHECK(ret == XPU_SUCCESS, "not support dtype");

  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext *>(dev_ctx);
  void *data_ptr = reinterpret_cast<void *>(shm->data_ptr_addr);
  auto x = paddle::from_blob(data_ptr,
                             tmp_input.shape(),
                             tmp_input.dtype(),
                             tmp_input.layout(),
                             place);
  paddle::Tensor y = tmp_input.copy_to(place, false);
  ret = baidu::xpu::api::scale<float, float>(xpu_ctx->x_context(),
                                             x.data<float>(),
                                             y.data<float>(),
                                             tmp_input.numel(),
                                             true,
                                             1.f,
                                             2.f);
  PD_CHECK(ret == XPU_SUCCESS, "add2 fail");
  ret = xpu_memcpy(const_cast<float *>(tmp_input.data<float>()),
                   y.data<float>(),
                   tmp_input.numel() * sizeof(float),
                   XPUMemcpyKind::XPU_DEVICE_TO_HOST);
  PD_CHECK(ret == XPU_SUCCESS, "xpu_memcpy fail");

  sharedMemoryClose(&info);
}

PD_BUILD_OP(read_data_ipc)
    .Inputs({"tmp_input"})
    .Attrs({"shm_name: std::string"})
    .Outputs({"tmp_input_out"})
    .SetInplaceMap({{"tmp_input", "tmp_input_out"}})
    .SetKernelFn(PD_KERNEL(ReadDataIpc));
