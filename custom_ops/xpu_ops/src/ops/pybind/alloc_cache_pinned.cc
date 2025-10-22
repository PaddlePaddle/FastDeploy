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

#include <sys/mman.h>          // NOLINT
#include "cuda_runtime_api.h"  // NOLINT
#include "ops/pybind/pybind.h"
#include "paddle/extension.h"
#include "xpu/runtime.h"

void check_xpu_error(int error) {
  if (error != XPU_SUCCESS) {
    throw XPUError(error);
  }
}

// 封装xpu_host_alloc的Python函数
uintptr_t custom_xpu_host_alloc(size_t size, unsigned int flags) {
  void* ptr = nullptr;
  // check_xpu_error(xpu_host_alloc(&ptr, size, flags));
  ptr = malloc(size);
  PD_CHECK(ptr != nullptr);
  PD_CHECK(mlock(ptr, size) == 0);
  return reinterpret_cast<uintptr_t>(ptr);
}

// 封装xpu_host_free的Python函数
void custom_xpu_host_free(uintptr_t ptr) {
  check_xpu_error(xpu_host_free(reinterpret_cast<void*>(ptr)));
}

// 封装cudaHostRegister的Python函数，将可分页内存注册为锁页的
void xpu_cuda_host_register(uintptr_t ptr, size_t size, unsigned int flags) {
  cudaError_t e = cudaHostRegister(reinterpret_cast<void*>(ptr), size, flags);
  PD_CHECK(e == cudaSuccess, cudaGetErrorString(e));
}
