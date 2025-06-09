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

#include "cuda_multiprocess.h"
#include "paddle/extension.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#define GPU(str) hip##str
#else
#define GPU(str) cuda##str
#endif

template <typename T>
__global__ void set_data(T *input, int n) {
    if (threadIdx.x == 0) {
        for (int i = 0; i < n; ++i) {
            *(input + i) = static_cast<T>(i);
            printf("set[%d]: %f\n", i, *(input + i));
        }
    }
}

template <typename T>
__global__ void print_data(T *input, int n) {
    if (threadIdx.x == 0) {
        for (int i = 0; i < n; ++i) {
            printf("input[%d]: %f\n", i, input[i]);
        }
    }
}

void ReadDataIpc(const paddle::Tensor &tmp_input,
                 int64_t data_ptr,
                 const std::string &shm_name) {
    volatile shmStruct *shm = NULL;
    sharedMemoryInfo info;
    if (sharedMemoryOpen(shm_name.c_str(), sizeof(shmStruct), &info) != 0) {
        printf("Failed to create shared memory slab\n");
        printf("Func ReadDataIpc. Shm_name: %s\n", shm_name.c_str());
        exit(EXIT_FAILURE);
    }
    shm = (volatile shmStruct *)info.addr;
    void *ptr = nullptr;
    checkCudaErrors(
        GPU(IpcOpenMemHandle)(&ptr,
                              *(GPU(IpcMemHandle_t) *)&shm->memHandle,
                              GPU(IpcMemLazyEnablePeerAccess)));
    printf("ptr: %p\n", ptr);
    print_data<float><<<1, 1>>>(reinterpret_cast<float *>(ptr), 10);
    GPU(DeviceSynchronize)();
    checkCudaErrors(GPU(IpcCloseMemHandle)(ptr));
    sharedMemoryClose(&info);
}

PD_BUILD_STATIC_OP(read_data_ipc)
    .Inputs({"tmp_input"})
    .Attrs({"data_ptr: int64_t", "shm_name: std::string"})
    .Outputs({"tmp_input_out"})
    .SetInplaceMap({{"tmp_input", "tmp_input_out"}})
    .SetKernelFn(PD_KERNEL(ReadDataIpc));
