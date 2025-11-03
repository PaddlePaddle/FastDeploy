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
#include "xpu_multiprocess.h"  // NOLINT

template <typename T>
void set_data_ipc(const paddle::Tensor &tmp_input,
                  const std::string &shm_name) {
  sharedMemoryInfo info;
  volatile shmStruct *shm = NULL;
  int ret = sharedMemoryCreate(shm_name.c_str(), sizeof(*shm), &info);
  PD_CHECK(ret == 0, "sharedMemoryCreate failed");
  shm = (volatile shmStruct *)info.addr;
  memset((void *)shm, 0, sizeof(*shm));  // NOLINT

  void *data_ptr_now =
      reinterpret_cast<void *>(const_cast<T *>(tmp_input.data<T>()));
#if XPURT_VERSION_MAJOR == 5
  ret = xpu_ipc_get_memhandle((XPUIpcMemHandle *)&shm->memHandle,  // NOLINT
                              data_ptr_now);
#elif XPURT_VERSION_MAJOR == 4
  PD_THROW("kl2 not support prefix cache");
#endif
  PD_CHECK(ret == XPU_SUCCESS, "xpu_ipc_get_memhandle failed");
  shm->data_ptr_addr = reinterpret_cast<uint64_t>((data_ptr_now));
}

void SetDataIpc(const paddle::Tensor &tmp_input, const std::string &shm_name) {
  switch (tmp_input.type()) {
    case paddle::DataType::FLOAT16: {
      return set_data_ipc<paddle::float16>(tmp_input, shm_name);
    }
    case paddle::DataType::FLOAT32: {
      return set_data_ipc<float>(tmp_input, shm_name);
    }
    case paddle::DataType::INT8: {
      return set_data_ipc<int8_t>(tmp_input, shm_name);
    }
    case paddle::DataType::UINT8: {
      return set_data_ipc<uint8_t>(tmp_input, shm_name);
    }
    case paddle::DataType::BFLOAT16: {
      return set_data_ipc<paddle::bfloat16>(tmp_input, shm_name);
    }
    default: {
      PD_THROW("NOT supported data type.");
      break;
    }
  }
}

PD_BUILD_OP(set_data_ipc)
    .Inputs({"tmp_input"})
    .Attrs({"shm_name: std::string"})
    .Outputs({"tmp_input_out"})
    .SetInplaceMap({{"tmp_input", "tmp_input_out"}})
    .SetKernelFn(PD_KERNEL(SetDataIpc));
