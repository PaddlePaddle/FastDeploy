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

#include "cuda_multiprocess.h"
#include "helper.h"

namespace {
int sharedMemoryOpen2(const char *name, size_t sz, sharedMemoryInfo *info) {
    info->size = sz;
    info->shmFd = shm_open(name, O_RDWR, 0777);
    if (info->shmFd < 0) {
        return errno;
    }

    info->addr =
        mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
    if (info->addr == NULL) {
        return errno;
    }

    return 0;
}
}  // namespace

std::vector<paddle::Tensor> GetDataPtrIpc(const paddle::Tensor &tmp_input,
                                          const std::string &shm_name) {
    auto out_data_ptr_tensor =
        paddle::full({1}, 0, paddle::DataType::INT64, paddle::CPUPlace());
    auto out_data_ptr_tensor_ptr = out_data_ptr_tensor.data<int64_t>();
    volatile shmStruct *shm = NULL;
    sharedMemoryInfo info;
    if (sharedMemoryOpen2(shm_name.c_str(), sizeof(shmStruct), &info) != 0) {
        printf("Failed to create shared memory slab\n");
        printf("Func GetDataPtrIpc. Shm_name: %s\n", shm_name.c_str());
        exit(EXIT_FAILURE);
    }
    shm = (volatile shmStruct *)info.addr;
    void *ptr = nullptr;
    checkCudaErrors(cudaIpcOpenMemHandle(&ptr,
                                         *(cudaIpcMemHandle_t *)&shm->memHandle,
                                         cudaIpcMemLazyEnablePeerAccess));

    out_data_ptr_tensor_ptr[0] = reinterpret_cast<int64_t>(ptr);
    return {out_data_ptr_tensor};
}

PD_BUILD_STATIC_OP(get_data_ptr_ipc)
    .Inputs({"tmp_input"})
    .Attrs({"shm_name: std::string"})
    .Outputs({"data_ptr"})
    .SetKernelFn(PD_KERNEL(GetDataPtrIpc));
