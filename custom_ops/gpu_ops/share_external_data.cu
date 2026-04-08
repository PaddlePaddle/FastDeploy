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

#include "helper.h"
#include<stdlib.h>
#include<string.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<unistd.h>
#include<fcntl.h>
#include<sys/mman.h>
#include<stdio.h>
#include "cuda_multiprocess.h"
#include "paddle/phi/core/tensor_meta.h"


std::vector<paddle::Tensor> ShareExternalData(paddle::Tensor& input,
                                              const std::string shm_name,
                                              const std::vector<int>& shape) {
  volatile shmStruct *shm = NULL;
  sharedMemoryInfo info;
  if (sharedMemoryOpen(shm_name.c_str(), sizeof(shmStruct), &info) != 0) {
    printf("Failed to create shared memory slab\n");
    printf("Func ShareExternalData. Shm_name: %s\n", shm_name.c_str());
    exit(EXIT_FAILURE);
  }
  shm = (volatile shmStruct *)info.addr;
  void *ptr = nullptr;
#ifdef PADDLE_WITH_HIP
  checkCudaErrors(
      hipIpcOpenMemHandle(&ptr,
                           *(hipIpcMemHandle_t *)&shm->memHandle,  // NOLINT
                           hipIpcMemLazyEnablePeerAccess));
#else
  checkCudaErrors(
      cudaIpcOpenMemHandle(&ptr,
                           *(cudaIpcMemHandle_t *)&shm->memHandle,  // NOLINT
                           cudaIpcMemLazyEnablePeerAccess));
#endif

  paddle::Tensor tmp_tensor = paddle::from_blob(
    ptr,
    shape,
    input.type()
  );
  sharedMemoryClose(&info);
  return {tmp_tensor};
}

PD_BUILD_STATIC_OP(share_external_data)
    .Inputs({"input"})
    .Outputs({"output"})
    .Attrs({"shm_name: std::string", "shape: std::vector<int>"})
    .SetKernelFn(PD_KERNEL(ShareExternalData));
