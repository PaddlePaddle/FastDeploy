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
#include "cuda_multiprocess.h"

int sharedMemoryCreate(const char *name, size_t sz, sharedMemoryInfo *info) {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  info->size = sz;
  info->shmHandle = CreateFileMapping(INVALID_HANDLE_VALUE, NULL,
                                      PAGE_READWRITE, 0, (DWORD)sz, name);
  if (info->shmHandle == 0) {
    return GetLastError();
  }

  info->addr = MapViewOfFile(info->shmHandle, FILE_MAP_ALL_ACCESS, 0, 0, sz);
  if (info->addr == NULL) {
    return GetLastError();
  }

  return 0;
#else
  int status = 0;

  info->size = sz;

  info->shmFd = shm_open(name, O_RDWR | O_CREAT, 0777);
  if (info->shmFd < 0) {
    return errno;
  }

  status = ftruncate(info->shmFd, sz);
  if (status != 0) {
    return status;
  }

  info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
  if (info->addr == NULL) {
    return errno;
  }

  return 0;
#endif
}

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
__global__ void print_data(const T *input, int n) {
  if (threadIdx.x == 0) {
    for (int i = 0; i < n; ++i) {
      printf("input[%d]: %f\n", i, input[i]);
    }
  }
}

template <paddle::DataType D>
void set_data_ipc(const paddle::Tensor& tmp_input,
                  const std::string& shm_name) {
  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  sharedMemoryInfo info;
  volatile shmStruct *shm = NULL;
  if (sharedMemoryCreate(shm_name.c_str(), sizeof(*shm), &info) != 0) {
      printf("Failed to create shared memory slab\n");
      printf("Func sharedMemoryCreate. Shm_name: %s\n", shm_name.c_str());
      exit(EXIT_FAILURE);
  }
  shm = (volatile shmStruct *)info.addr;
  memset((void *)shm, 0, sizeof(*shm));

  void *data_ptr_now = reinterpret_cast<void*>(const_cast<data_t*>(tmp_input.data<data_t>()));
#ifdef PADDLE_WITH_HIP
  checkCudaErrors(hipIpcGetMemHandle((hipIpcMemHandle_t *)&shm->memHandle, data_ptr_now));
#else
  checkCudaErrors(cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm->memHandle, data_ptr_now));
#endif


}

void SetDataIpc(const paddle::Tensor& tmp_input,
                const std::string& shm_name) {
    std::vector<int64_t> shape = tmp_input.shape();

    switch (tmp_input.type()) {
      case paddle::DataType::BFLOAT16: {
        return set_data_ipc<paddle::DataType::BFLOAT16>(
          tmp_input,
          shm_name
        );
      }
      case paddle::DataType::FLOAT16: {
        return set_data_ipc<paddle::DataType::FLOAT16>(
          tmp_input,
          shm_name
        );
      }
      case paddle::DataType::FLOAT32: {
        return set_data_ipc<paddle::DataType::FLOAT32>(
          tmp_input,
          shm_name
        );
      }
      case paddle::DataType::INT8: {
        return set_data_ipc<paddle::DataType::INT8>(
          tmp_input,
          shm_name
        );
      }
      case paddle::DataType::UINT8: {
        return set_data_ipc<paddle::DataType::UINT8>(
          tmp_input,
          shm_name
        );
      }
      default: {
          PD_THROW(
              "NOT supported data type. "
              "Only float16, bfloat16 and float32 are supported. ");
          break;
      }
    }
}

PD_BUILD_STATIC_OP(set_data_ipc)
    .Inputs({"tmp_input"})
    .Attrs({ "shm_name: std::string"})
    .Outputs({"tmp_input_out"})
    .SetInplaceMap({{"tmp_input", "tmp_input_out"}})
    .SetKernelFn(PD_KERNEL(SetDataIpc));
