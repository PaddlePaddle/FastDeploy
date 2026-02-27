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

#include "paddle/extension.h"
#include "pybind11/pybind11.h"
namespace py = pybind11;

// 自定义异常类，用于处理CUDA错误
class CudaError : public std::exception {
 public:
  explicit CudaError(cudaError_t error) : error_(error) {}

  const char* what() const noexcept override {
    return cudaGetErrorString(error_);
  }

 private:
  cudaError_t error_;
};

// 检查CUDA错误并抛出异常
void check_cuda_error(cudaError_t error) {
  if (error != cudaSuccess) {
    throw CudaError(error);
  }
}

// 封装cudaHostAlloc的Python函数
uintptr_t cuda_host_alloc(size_t size,
                          unsigned int flags = cudaHostAllocDefault) {
  void* ptr = nullptr;
  check_cuda_error(cudaHostAlloc(&ptr, size, flags));
  return reinterpret_cast<uintptr_t>(ptr);
}

// 封装cudaFreeHost的Python函数
void cuda_host_free(uintptr_t ptr) {
  check_cuda_error(cudaFreeHost(reinterpret_cast<void*>(ptr)));
}

paddle::Tensor GetStop(paddle::Tensor& not_need_stop);

void SetStop(paddle::Tensor& not_need_stop, bool flag);

PYBIND11_MODULE(fastdeploy_ops, m) {
  /**
   * alloc_cache_pinned.cc
   * cuda_host_alloc
   * cuda_host_free
   */
  m.def("cuda_host_alloc",
        &cuda_host_alloc,
        "Allocate pinned memory",
        py::arg("size"),
        py::arg("flags") = cudaHostAllocDefault);
  m.def(
      "cuda_host_free", &cuda_host_free, "Free pinned memory", py::arg("ptr"));
  py::register_exception<CudaError>(m, "CudaError");

  m.def("get_stop", &GetStop, "get_stop function");

  m.def("set_stop", &SetStop, "set_stop function");
}
